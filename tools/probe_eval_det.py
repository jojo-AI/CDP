#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
eval 阶段确定性探针（2026-10-08 新增）

背景：
  weather-96 det_mode=1 同 seed 三次重复，第 1 个 epoch 的 Vali/Test Loss 就已经不同
  （相对差 ~1e-5），但逐 step 训练轨迹前 64 步参数/梯度/输出逐位一致。
  ⇒ 需要把「训练阶段」与「评估阶段」拆开判定：到底是训练轨迹分叉，还是
     `exp.vali()` 的 eval 前向（no_grad 下走不同 kernel）自己就是不确定的。

本脚本只做一件事：**加载同一份 checkpoint**，跑一遍 eval 前向，
把每个 batch 的输出张量哈希、整段 loss 打印到全精度。
两次独立进程各跑一次 → 哈希不同 ⇒ eval 前向本身不确定；哈希相同 ⇒ eval 是干净的。

用法：
  python tools/probe_eval_det.py --log <真实日志> --ckpt <checkpoint 目录> \
        --det 1 --seed 2021 --flag val  --out D:/fsdownload/evaldet_v1.json
  python tools/probe_eval_det.py ... 同参数 ...                      --out D:/fsdownload/evaldet_v2.json
  python tools/probe_eval_det.py --compare D:/fsdownload/evaldet_v1.json D:/fsdownload/evaldet_v2.json
"""
import os
import sys
import json
import hashlib
import argparse
import random

import numpy as np
import torch


def thash(t):
    a = t.detach().to('cpu', torch.float32).contiguous().numpy()
    return hashlib.md5(a.tobytes()).hexdigest()[:16]


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


def run_probe(a):
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

    sys.path.insert(0, os.getcwd())
    sys.path.insert(0, os.path.join(os.getcwd(), 'tools'))
    from trace_divergence import parse_real_args
    from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast

    if not os.path.exists(a.log):
        print('[probe] 找不到日志: %s' % a.log)
        return 1

    args = parse_real_args(a.log)
    args.use_multi_gpu = False
    args.use_gpu = bool(torch.cuda.is_available())
    args.itr = 1

    print('[probe] 配置: %s  flag=%s  ckpt=%s' % (a.log, a.flag, a.ckpt))
    print('[probe] 环境: %s' % json.dumps(env_report(), ensure_ascii=False))

    exp = Exp_Long_Term_Forecast(args)

    ckpt_path = a.ckpt
    if not ckpt_path.endswith('.pth'):
        ckpt_path = os.path.join(ckpt_path, 'checkpoint.pth')
    sd = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    if isinstance(sd, dict) and 'state_dict' in sd:
        sd = sd['state_dict']
    missing, unexpected = exp.model.load_state_dict(sd, strict=False)
    print('[probe] 载入权重: %s  missing=%d unexpected=%d  (ckpt md5=%s)'
          % (os.path.basename(os.path.dirname(ckpt_path)), len(missing),
             len(unexpected), hashlib.md5(open(ckpt_path, 'rb').read()).hexdigest()[:16]))
    exp.model.to(exp.device)

    criterion = exp._select_criterion()
    data_set, loader = exp._get_data(flag=a.flag)

    # ---- 完全复刻 exp.vali() 的前向，只是额外记录哈希 ----
    exp.model.eval()
    hashes = []
    total_loss = []
    chunk = []
    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(loader):
            batch_x = batch_x.float().to(exp.device)
            batch_y = batch_y.float().to(exp.device)
            batch_x_mark = batch_x_mark.float().to(exp.device)
            batch_y_mark = batch_y_mark.float().to(exp.device)

            if 'PEMS' == args.data or 'Solar' == args.data:
                batch_x_mark = None
                batch_y_mark = None

            if args.down_sampling_layers == 0:
                dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(exp.device)
            else:
                dec_inp = None

            if args.output_attention:
                outputs = exp.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
            else:
                outputs = exp.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

            f_dim = -1 if args.features == 'MS' else 0
            outputs = outputs[:, -args.pred_len:, f_dim:]
            batch_y_ = batch_y[:, -args.pred_len:, f_dim:].to(exp.device)

            loss = criterion(outputs.detach(), batch_y_)
            total_loss.append(loss.item())
            hashes.append(thash(outputs))
            if i < 5:
                chunk.append({'i': i, 'hash': hashes[-1], 'loss': repr(loss.item())})

    mean_loss = float(np.average(total_loss))
    concat = hashlib.md5('|'.join(hashes).encode()).hexdigest()[:16]
    payload = {
        'log': a.log,
        'ckpt': ckpt_path,
        'ckpt_md5_full': hashlib.md5(open(ckpt_path, 'rb').read()).hexdigest(),
        'flag': a.flag,
        'seed': a.seed,
        'det': a.det,
        'no_tf32': bool(a.no_tf32),
        'env': env_report(),
        'n_batches': len(hashes),
        'concat_hash': concat,
        'mean_loss': mean_loss,
        'mean_loss_repr': repr(mean_loss),
        'first5': chunk,
        'hashes': hashes,
    }
    with open(a.out, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False)

    print('[probe] n_batches=%d  concat_hash=%s  mean_loss=%s'
          % (len(hashes), concat, repr(mean_loss)))
    for c in chunk:
        print('        batch %d  %s  loss=%s' % (c['i'], c['hash'], c['loss']))
    print('[probe] 写出 %s' % a.out)
    return 0


def run_compare(files):
    data = [json.load(open(f, encoding='utf-8')) for f in files]
    print('=' * 78)
    print('eval 确定性比对（%d 次）' % len(data))
    for d, f in zip(data, files):
        print('  %-34s flag=%s det=%s n_batches=%d concat=%s mean_loss=%s'
              % (os.path.basename(f), d['flag'], d['det'], d['n_batches'],
                 d['concat_hash'], d['mean_loss_repr']))
    print('-' * 78)
    if len({d['concat_hash'] for d in data}) == 1:
        print('✅ eval 前向逐 batch 输出完全一致 ⇒ eval 阶段确定性，分叉不在这里')
    else:
        print('❌ eval 前向输出不同 ⇒ **eval 前向本身不确定**，这就是同种子分叉的来源之一')
        n = min(len(d['hashes']) for d in data)
        for i in range(n):
            vals = {d['hashes'][i] for d in data}
            if len(vals) > 1:
                print('   第一个分叉 batch: %d（共 %d）' % (i, n))
                for d, f in zip(data, files):
                    print('     %-30s %s' % (os.path.basename(f), d['hashes'][i]))
                break
    lv = {d['mean_loss_repr'] for d in data}
    if len(lv) > 1:
        ms = [d['mean_loss'] for d in data]
        print('   mean_loss 极差 = %.3e（相对 %.3e%%）'
              % (max(ms) - min(ms), (max(ms) - min(ms)) / min(ms) * 100))
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--log', default='D:/fsdownload/det_weather_det1/same1.log')
    ap.add_argument('--ckpt', default=None, help='checkpoint 目录或 .pth 路径')
    ap.add_argument('--flag', default='val', choices=['val', 'test', 'train'])
    ap.add_argument('--out', default=None)
    ap.add_argument('--seed', type=int, default=2021)
    ap.add_argument('--det', type=int, default=1)
    ap.add_argument('--no-tf32', dest='no_tf32', action='store_true')
    ap.add_argument('--compare', nargs='*', default=None)
    a = ap.parse_args()
    if a.compare:
        return run_compare(a.compare)
    if not a.out or not a.ckpt:
        print('需要 --out 与 --ckpt')
        return 1
    return run_probe(a)


if __name__ == '__main__':
    sys.exit(main())
