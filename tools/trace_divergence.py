#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
逐 step 二分非确定性来源 —— 定位训练轨迹中「第一个产生差异的位置」。

背景（2026-10-08）：
  ETTh1-96 Stage-1 在 det_mode=0 下，同 seed 重复 4 次 MSE 极差 0.89%，
  且 1 个 epoch（66 步）后权重就已经不同；而 det_mode=1 下 9/9 次完全一致。
  算子级隔离测试（Linear/池化/Conv/GELU/LayerNorm/Adam 200 步/完整模型前后向）
  全部显示确定性 → 说明触发点藏在真实数据路径或训练循环的某处。

本脚本通过「给真实 Exp 打 hook」的方式（不复制训练循环，保证与实验完全一致），
逐步记录每个 step 的：
    in_x / in_mark  → 输入批次哈希     （差异 ⇒ 数据加载/RNG 问题）
    out             → forward 输出哈希 （差异 ⇒ 模型算子问题）
    grad            → 反向梯度哈希     （差异 ⇒ 反向 kernel 问题）
    param_post      → optimizer.step 后参数哈希（差异 ⇒ 优化器问题）

用两个独立进程各跑一次（模拟真实的两次重复实验），再 --compare 比对。

用法：
  1) python tools/trace_divergence.py --log <真实训练日志> --out D:/fsdownload/trace_run1.json --seed 2021
  2) python tools/trace_divergence.py --log <同上>          --out D:/fsdownload/trace_run2.json --seed 2021
  3) python tools/trace_divergence.py --compare D:/fsdownload/trace_run1.json D:/fsdownload/trace_run2.json

可选：--det 1 打开确定性开关（应与 --det 0 形成对照）
      --epochs N 多跑几个 epoch（默认 1，找到第一个分叉就够）
"""
import os
import sys
import json
import re
import ast
import types
import hashlib
import argparse
import random

import numpy as np
import torch


# --------------------------------------------------------------------------- #
# 哈希工具：所有张量统一转 float32/CPU/contiguous 再取 md5，保证跨进程可比
# --------------------------------------------------------------------------- #
class _StopTraining(Exception):
    """记录满指定步数后用于中断训练（保留真实 LR 调度的模式下使用）"""


def thash(t):
    if t is None:
        return 'None'
    if isinstance(t, (list, tuple)):
        return 'L[' + ','.join(thash(x) for x in t) + ']'
    if not torch.is_tensor(t):
        return repr(t)
    a = t.detach().to('cpu', torch.float32).contiguous().numpy()
    return hashlib.md5(a.tobytes()).hexdigest()[:16]


def dict_hash(items):
    s = '|'.join('%s=%s' % (n, h) for n, h in items)
    return hashlib.md5(s.encode()).hexdigest()[:16]


def param_hash(m):
    return dict_hash((n, thash(p)) for n, p in sorted(m.named_parameters()))


def grad_hash(m):
    return dict_hash((n, thash(p.grad) if p.grad is not None else 'N')
                     for n, p in sorted(m.named_parameters()))


def env_report():
    return {
        'torch': torch.__version__,
        'cuda': torch.version.cuda,
        'cudnn_deterministic': bool(torch.backends.cudnn.deterministic),
        'cudnn_benchmark': bool(torch.backends.cudnn.benchmark),
        'cudnn_allow_tf32': bool(torch.backends.cudnn.allow_tf32),
        'matmul_allow_tf32': bool(torch.backends.cuda.matmul.allow_tf32),
        'cublas_workspace': os.environ.get('CUBLAS_WORKSPACE_CONFIG', '<unset>'),
        'use_deterministic_algorithms': bool(torch.are_deterministic_algorithms_enabled()),
        'float32_matmul_precision': torch.get_float32_matmul_precision(),
    }


# --------------------------------------------------------------------------- #
# 从真实训练日志还原 args（保证配置与实验逐字段一致）
# --------------------------------------------------------------------------- #
def parse_real_args(log_path):
    txt = open(log_path, encoding='utf-8', errors='ignore').read()
    ns_str = re.search(r'Namespace\(([^\n]*)\)', txt).group(1)
    # ast.literal_eval 不接受函数调用，故先 parse 成 AST 再逐个取关键字字面量
    tree = ast.parse('dict(' + ns_str + ')', mode='eval')
    kw = {n.arg: ast.literal_eval(n.value) for n in tree.body.keywords}
    return types.SimpleNamespace(**kw)


# --------------------------------------------------------------------------- #
# 主流程：跑一次并记录逐 step 哈希
# --------------------------------------------------------------------------- #
def run_trace(a):
    # ---------------- 环境（复现目标配置） ----------------
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(a.seed)
        torch.cuda.manual_seed_all(a.seed)

    if a.det == 1:
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
    if a.no_tf32:
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False

    # ---------------- 还原真实配置 ----------------
    if not os.path.isabs(a.log) or not os.path.exists(a.log):
        print('[trace] 找不到日志: %s' % a.log)
        return 1
    args = parse_real_args(a.log)
    if a.keep_epochs:
        # ⚠️ 关键：train_epochs 会决定 OneCycleLR 的 total_steps（= epochs × steps_per_epoch），
        #    进而决定每个 step 的 LR。若强行改成 1，整条数值轨迹与真实实验完全不同
        #    （实测 train loss 0.3209 vs 真实 0.3688）→ 结论不可用。
        #    所以保持原值，改成「记录满 max_steps 步后主动中断」。
        print('[trace] 保持日志里的 train_epochs=%s（保证 LR 调度与真实实验一致），'
              '记录满 %d 步后中断' % (args.train_epochs, a.max_steps))
    else:
        args.train_epochs = a.epochs
    args.itr = 1
    # 关掉多 GPU / 保持单卡，避免引入额外变量
    args.use_multi_gpu = False
    args.use_gpu = bool(torch.cuda.is_available())

    sys.path.insert(0, os.getcwd())
    from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast

    print('[trace] 配置来源: %s' % a.log)
    print('[trace] model=%s data=%s enc_in=%s pred_len=%s batch=%s epochs=%s'
          % (args.model, args.data, args.enc_in, args.pred_len,
             args.batch_size, args.train_epochs))
    print('[trace] 环境: %s' % json.dumps(env_report(), ensure_ascii=False))

    exp = Exp_Long_Term_Forecast(args)

    # ---------------- 打 hook ----------------
    records = []
    grad_detail = []            # [{ 'i': step, 'g': {参数名: 梯度哈希} }] —— 用于把分叉定位到具体算子
    state = {'step': 0, 'recording': True}

    layer_records = []          # [{ 'i': step, 'layers': {子模块名: 输出哈希} }]

    def pre_hook(mod, inputs):
        if not mod.training:
            return
        # 步数预算：逐子模块哈希需要 GPU→CPU 同步，286 步 × 100+ 层会拖到 10 分钟以上。
        # 非确定性在第 1 个 epoch 内就会显现，前 max_steps 步足以命中（默认 64）。
        if len(records) >= a.max_steps:
            state['recording'] = False
            return
        state['recording'] = True
        # 全参数哈希每步要几百次 GPU→CPU 同步，是耗时主因；
        # 长程追踪（如跑满 286 步）时用 --param-every 10 稀疏化，只保证逐 step 的
        # forward 输出哈希（out）密集，足以定位第一个分叉 step。
        state['hash_now'] = (len(records) % max(1, a.param_every) == 0)
        records.append({
            'i': state['step'],
            'in_x': thash(inputs[0]) if len(inputs) > 0 else 'NA',
            'in_mark': thash(inputs[1]) if len(inputs) > 1 else 'NA',
        })
        layer_records.append({'i': state['step'], 'layers': {}})

    def post_hook(mod, inputs, output):
        if not mod.training or not state.get('recording'):
            return
        records[-1]['out'] = thash(output)
        if state.get('hash_now'):
            records[-1]['param_pre'] = param_hash(mod)
        state['step'] += 1

    exp.model.register_forward_pre_hook(pre_hook)
    exp.model.register_forward_hook(post_hook)

    # 逐子模块 hook：把分叉精确定位到具体层，进而指认是哪个算子的 CUDA kernel
    def make_layer_hook(name):
        def _h(mod, inp, out):
            if not mod.training or not state.get('recording') or not layer_records:
                return
            try:
                layer_records[-1]['layers'][name] = thash(out)
            except Exception:
                pass
        return _h

    for _name, _mod in exp.model.named_modules():
        if _name and a.layers:
            _mod.register_forward_hook(make_layer_hook(_name))

    # 记录梯度与 step 后参数：必须用 PyTorch 官方的 step hook。
    # ⚠️ 不要用 `opt.step = func` 覆盖实例方法 —— PyTorch 2.6 的 OneCycleLR 在 __init__
    #    里会调用 patch_track_step_called()，它要求 opt.step 是 bound method
    #    （会访问 step_fn.__func__）；覆盖成普通函数会直接 AttributeError（踩过这个坑）。
    _orig_select = exp._select_optimizer

    def _select():
        opt = _orig_select()

        def _pre(optimizer, args_, kwargs_):
            # step 之前梯度尚未清零，可拿到本步真实梯度
            if records and state.get('recording') and state.get('hash_now') \
                    and getattr(exp.model, 'training', False):
                records[-1]['grad'] = grad_hash(exp.model)
                if a.grad_detail:
                    grad_detail.append({
                        'i': records[-1]['i'],
                        'g': {n: thash(p.grad) if p.grad is not None else 'N'
                              for n, p in sorted(exp.model.named_parameters())},
                    })

        def _post(optimizer, args_, kwargs_):
            if records and state.get('recording') and state.get('hash_now') \
                    and getattr(exp.model, 'training', False):
                records[-1]['param_post'] = param_hash(exp.model)
                # 保留真实 LR 调度时，记录满 max_steps 步就在 optimizer.step() 之后安全中断
                # （不要在前向 hook 里抛异常 —— autograd 图在飞行中，容易二次报错）
                if a.keep_epochs and len(records) >= a.max_steps:
                    raise _StopTraining()

        opt.register_step_pre_hook(_pre)
        opt.register_step_post_hook(_post)
        return opt

    exp._select_optimizer = _select

    # ---------------- 跑一个（或多个）epoch ----------------
    setting = 'trace_%s_s%d_d%d' % (
        os.path.basename(a.log).replace('.log', ''), a.seed, a.det)
    stopped_early = False
    try:
        exp.train(setting)
    except _StopTraining:
        stopped_early = True
        print('[trace] 已在第 %d 步后中断（保留真实 LR 调度）' % len(records))

    payload = {
        'log': a.log,
        'seed': a.seed,
        'det': a.det,
        'no_tf32': bool(a.no_tf32),
        'keep_epochs': bool(a.keep_epochs),
        'stopped_early': stopped_early,
        'env': env_report(),
        'n_records': len(records),
        'n_steps_layers': len(layer_records),
        'records': records,
        'layer_hashes': layer_records,
        'grad_detail': grad_detail,
    }
    with open(a.out, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False)
    print('[trace] 写出 %s  共 %d 个 step 记录' % (a.out, len(records)))
    return 0


# --------------------------------------------------------------------------- #
# 比对模式
# --------------------------------------------------------------------------- #
def run_compare(files):
    data = [json.load(open(f, encoding='utf-8')) for f in files]
    recs = [d['records'] for d in data]
    print('=' * 78)
    print('比对 %d 次运行' % len(data))
    for d, f in zip(data, files):
        print('  %-46s seed=%s det=%s steps=%d' % (os.path.basename(f), d['seed'], d['det'], d['n_records']))
    n = min(len(r) for r in recs)
    print('  共同可比的 step 数: %d' % n)
    print('=' * 78)

    keys = ['in_x', 'in_mark', 'out', 'param_pre', 'grad', 'param_post']
    labels = {
        'in_x': '输入批次 x_enc',
        'in_mark': '输入时间标记 x_mark',
        'out': 'forward 输出',
        'param_pre': 'forward 后参数（未更新）',
        'grad': '反向梯度',
        'param_post': 'optimizer.step 后参数',
    }
    first = None
    for i in range(n):
        for k in keys:
            vals = {r[i].get(k) for r in recs}
            if len(vals) > 1:
                first = (i, k)
                break
        if first:
            break

    if first is None:
        print('✅ 前 %d 个 step 全部逐位一致（in_x/in_mark/out/grad/param 全部相同）' % n)
        return 0

    i, k = first
    print('🎯 第一个分叉位置: **第 %d 步**，分歧量 = **%s**（%s）' % (i, k, labels.get(k, k)))
    print()
    print('   逐步哈希（step %d 附近）:' % i)
    for j in range(max(0, i - 2), min(n, i + 3)):
        mark = ' <<< 分叉' if j == i else ''
        print('   step %3d:%s' % (j, mark))
        for kk in keys:
            vals = [r[j].get(kk) for r in recs]
            same = '==' if len(set(vals)) == 1 else '!='
            print('     %-12s %s   %s' % (kk, same, '  '.join(str(v) for v in vals)))

    # 归因结论
    print()
    print('   ---------- 归因 ----------')
    if k == 'in_x' or k == 'in_mark':
        print('   ⇒ 输入批次就不同：问题在**数据加载 / 随机数消费顺序**，不在算子。')
    elif k == 'out':
        print('   ⇒ 输入相同但 forward 输出不同：问题在**模型前向算子**（CUDA kernel 非确定）。')
    elif k == 'param_pre':
        print('   ⇒ 输出相同但参数快照不同：理论上不应发生，检查 hook 时序。')
    elif k == 'grad':
        print('   ⇒ 输出相同但梯度不同：问题在**反向传播 kernel**（atomicAdd / 归约顺序）。')
    elif k == 'param_post':
        print('   ⇒ 梯度相同但更新后参数不同：问题在**优化器 kernel**。')

    # ---------- 层级定位：把分叉点精确到子模块 ----------
    lh = [d.get('layer_hashes') or [] for d in data]
    if all(lh) and any(len(x.get('layers') or {}) for x in lh[0]):
        m = min(len(x) for x in lh)
        print()
        print('   ---------- 层级定位（逐 step × 逐子模块输出哈希）----------')
        if m == 0:
            print('   无层级记录（模型未进入 training 模式？）')
        else:
            found = False
            for step_i in range(m):
                for nm in sorted(lh[0][step_i]['layers'].keys()):
                    vals = {x[step_i]['layers'].get(nm) for x in lh}
                    if len(vals) > 1:
                        print('   🎯 最早分叉的子模块: step %d, layer = `%s`' % (step_i, nm))
                        found = True
                        break
                if found:
                    break
            if not found:
                print('   前 %d 步所有子模块输出完全一致 ⇒ 非确定性不在前向算子层' % m)
    # ---------- 梯度级定位：把分叉精确到「哪个参数的梯度先不同」----------
    gd = [d.get('grad_detail') or [] for d in data]
    if all(gd) and len(gd[0]) > 0:
        print()
        print('   ---------- 梯度级定位（逐 step × 逐参数梯度哈希）----------')
        m = min(len(x) for x in gd)
        found = False
        for si in range(m):
            names = sorted(gd[0][si]['g'].keys())
            diff = [n for n in names
                    if len({x[si]['g'].get(n) for x in gd}) > 1]
            if diff:
                print('   🎯 step %d（记录序号 %d）先分叉的参数 **%d 个**，前几个：'
                      % (gd[0][si]['i'], si, len(diff)))
                for n in diff[:8]:
                    vals = [x[si]['g'].get(n) for x in gd]
                    print('      %-52s %s' % (n, '  '.join(str(v) for v in vals)))
                # 归因：看参数名落在哪个算子族
                fam = {}
                for n in diff:
                    key = '其他'
                    for pat in ('down_sampling_layers', 'dec_embedding', 'enc_embedding',
                                'pdm_blocks', 'layer_norm', 'norm', 'projection',
                                'predict', 'season', 'trend', 'dyn', 'cal'):
                        if pat in n:
                            key = pat
                            break
                    fam[key] = fam.get(key, 0) + 1
                print('      参数族分布: %s' % fam)
                found = True
                break
        if not found:
            print('   前 %d 步所有参数梯度完全一致 ⇒ 该窗口内没有分叉' % m)
    return 0


# --------------------------------------------------------------------------- #
# 命令行
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--log', default='D:/fsdownload/determinism_test_det0/same1.log',
                    help='真实训练日志，用于还原逐字段一致的 args')
    ap.add_argument('--out', default=None, help='本次记录的 JSON 输出路径')
    ap.add_argument('--seed', type=int, default=2021)
    ap.add_argument('--det', type=int, default=0, help='1 = 打开确定性开关（对照用）')
    ap.add_argument('--no-tf32', dest='no_tf32', action='store_true',
                    help='强制关闭 TF32（cudnn + matmul），用于验证 TF32 假设')
    ap.add_argument('--epochs', type=int, default=1)
    ap.add_argument('--max-steps', dest='max_steps', type=int, default=64,
                    help='最多记录多少个训练 step（层哈希需 GPU→CPU 同步，过多会极慢）')
    ap.add_argument('--layers', type=int, default=1,
                    help='1 = 记录逐子模块输出哈希（慢，用于层定位）；0 = 只记 in_x/out/grad/param')
    ap.add_argument('--param-every', dest='param_every', type=int, default=1,
                    help='每 N 步才哈希一次全参数（grad/param_pre/param_post），稀疏化提速')
    ap.add_argument('--grad-detail', dest='grad_detail', action='store_true',
                    help='逐参数记录梯度哈希，用于把分叉定位到具体算子（体积略大）')
    ap.add_argument('--keep-epochs', dest='keep_epochs', action='store_true',
                    help='保留日志里的 train_epochs（保证 OneCycleLR 总步数与真实实验一致），'
                         '记录满 max_steps 步后中断。**追真实分叉点必须开这个**')
    ap.add_argument('--compare', nargs='*', default=None, help='比对模式：传入多个 JSON')
    a = ap.parse_args()

    if a.compare:
        return run_compare(a.compare)
    if not a.out:
        print('需要 --out（或在 --compare 模式）')
        return 1
    return run_trace(a)


if __name__ == '__main__':
    sys.exit(main())
