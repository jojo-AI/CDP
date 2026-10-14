# -*- coding: utf-8 -*-
"""验证集平台期 checkpoint 集成评估（SWA 实验核心工具）。

动机（见 D:/fsdownload/multiseed_findings_20261008.md §7/§9）：
    现有协议用 argmin(val) 选「最优 epoch」。在长 horizon / 短 epoch 设定上 val 曲线是平的，
    argmin 等价于在平台里抽签，导致 test 指标跨种子抖动可达 5%（ETTh1-720）。
    本工具用「验证集平台期内的 checkpoint 集成」替换该抽签：
        plateau = { e : val_mae[e] <= min(val_mae) * (1 + eps) }
        pred_ens = mean_{e in plateau} pred_e
    平台期由验证集决定（不使用测试集），因此无泄漏。

用法：
    python tools/eval_ckpt_ensemble.py \
        --log /d/fsdownload/xxx.log \
        --ckpt-dir ./checkpoints/<setting> \
        --eps 0.01 \
        --out /d/fsdownload/ens_xxx.json

输出（JSON + 屏幕）：
    protocol_argmin_val  现有协议：argmin(val) 的单个 checkpoint
    protocol_ens_pred    平台期「预测平均」
    protocol_ens_weight  平台期「权重平均」（SWA）
    ref_oracle_argmin_test  参考上界：argmin(test)（仅用于诊断，不可写进论文）
"""
import argparse
import ast
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast


def parse_real_args(log_path):
    """从训练日志的 Namespace(...) 反解出 argparse 命名空间（第一份即训练那次）。"""
    txt = open(log_path, encoding='utf-8', errors='ignore').read()
    m = re.search(r'Namespace\(([^\n]*)\)', txt)
    if not m:
        raise RuntimeError('日志里找不到 Namespace(...)：%s' % log_path)
    tree = ast.parse('dict(' + m.group(1) + ')', mode='eval')
    return {n.arg: ast.literal_eval(n.value) for n in tree.body.keywords}


def forward_loss(model, loader, device, down_sampling_layers, pred_len, label_len,
                 features, data, want_pred=False):
    """流式统计一个 loader 的 (mse, mae, n)，可选返回预测/真值拼接（仅小数据集用）。"""
    model.eval()
    n_elem = 0
    sum_abs = 0.0
    sum_sq = 0.0
    preds, trues = [], []
    with torch.no_grad():
        for batch_x, batch_y, batch_x_mark, batch_y_mark in loader:
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)
            if data in ('PEMS', 'Solar'):
                batch_x_mark = None
                batch_y_mark = None
            if down_sampling_layers == 0:
                dec_inp = torch.zeros_like(batch_y[:, -pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :label_len, :], dec_inp], dim=1).float().to(device)
            else:
                dec_inp = None
            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            diff = outputs - batch_y
            n_elem += int(diff.numel())
            sum_abs += float(diff.abs().sum())
            sum_sq += float((diff ** 2).sum())
            if want_pred:
                preds.append(outputs.detach().cpu())
                trues.append(batch_y.detach().cpu())
    mse = sum_sq / n_elem
    mae = sum_abs / n_elem
    if want_pred:
        return mse, mae, torch.cat(preds), torch.cat(trues)
    return mse, mae, None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--log', required=True, help='训练日志（用于反解 Namespace）')
    ap.add_argument('--ckpt-dir', dest='ckpt_dir', required=True, help='含 epoch_XXX.pth 的目录')
    ap.add_argument('--eps', type=float, default=0.01, help='平台期阈值：val <= min(val)*(1+eps)')
    ap.add_argument('--out', default='', help='结果 JSON 输出路径')
    a = ap.parse_args()

    kw = parse_real_args(a.log)
    kw['is_training'] = 0
    kw['use_multi_gpu'] = False
    kw['use_amp'] = False
    kw['output_attention'] = False
    kw['use_gpu'] = bool(torch.cuda.is_available())
    kw['gpu'] = 0
    args = argparse.Namespace(**kw)

    ckpts = sorted(f for f in os.listdir(a.ckpt_dir)
                   if re.match(r'epoch_\d+\.pth$', f))
    if not ckpts:
        print('!! %s 下没有 epoch_*.pth' % a.ckpt_dir)
        return 1

    exp = Exp_Long_Term_Forecast(args)
    device = exp.device
    val_data, val_loader = exp._get_data(flag='val')
    test_data, test_loader = exp._get_data(flag='test')

    common = dict(model=exp.model, device=device,
                  down_sampling_layers=args.down_sampling_layers, pred_len=args.pred_len,
                  label_len=args.label_len, features=args.features, data=args.data)

    per_epoch = []
    print('%-10s %-12s %-12s' % ('epoch', 'val MAE', 'test MSE'))
    for f in ckpts:
        ep = int(re.match(r'epoch_(\d+)\.pth', f).group(1))
        sd = torch.load(os.path.join(a.ckpt_dir, f), map_location='cpu')
        exp.model.load_state_dict(sd)
        vm, _, _, _ = forward_loss(**common, loader=val_loader)
        tm, ta, _, _ = forward_loss(**common, loader=test_loader)
        per_epoch.append(dict(epoch=ep, val_mae=vm, test_mse=tm, test_mae=ta))
        print('%-10d %-12.7f %-12.7f' % (ep, vm, tm))

    vals = np.array([r['val_mae'] for r in per_epoch])
    tests = np.array([r['test_mse'] for r in per_epoch])
    i_val = int(vals.argmin())
    i_test = int(tests.argmin())
    thr = vals.min() * (1.0 + a.eps)
    plateau = [i for i in range(len(per_epoch)) if vals[i] <= thr]

    print('\n--- 选点诊断 ---')
    print('  平台期(eps=%.3f)：epoch %s（共 %d 个）'
          % (a.eps, [per_epoch[i]['epoch'] for i in plateau], len(plateau)))
    print('  argmin(val)  → epoch %d  val=%.7f  test MSE=%.7f'
          % (per_epoch[i_val]['epoch'], vals[i_val], tests[i_val]))
    print('  argmin(test) → epoch %d  test MSE=%.7f（仅诊断用）'
          % (per_epoch[i_test]['epoch'], tests[i_test]))
    print('  平台期内 test MSE 散布：%.5f ~ %.5f（%.3f%%）'
          % (tests[plateau].min(), tests[plateau].max(),
             (tests[plateau].max() - tests[plateau].min()) / tests[plateau].min() * 100))

    # ---- 平台期：预测平均 ----
    acc_p = None
    trues = None
    acc_mae_val = 0.0
    acc_mae_test = 0.0
    n_val_elem = 0
    n_test_elem = 0
    for i in plateau:
        f = 'epoch_%03d.pth' % per_epoch[i]['epoch']
        sd = torch.load(os.path.join(a.ckpt_dir, f), map_location='cpu')
        exp.model.load_state_dict(sd)
        _, vmae, vp, vt = forward_loss(**common, loader=val_loader, want_pred=True)
        _, tmae, tp, tt = forward_loss(**common, loader=test_loader, want_pred=True)
        acc_mae_val += vmae
        acc_mae_test += tmae
        if acc_p is None:
            acc_p, trues = tp.clone(), tt.clone()
        else:
            acc_p += tp
        n_val_elem = vp.numel()

    ens_pred = acc_p / len(plateau)
    diff = ens_pred - trues
    ens_test_mse = float((diff ** 2).mean())
    ens_test_mae = float(diff.abs().mean())
    # 平台期 val 平均误差（用各 epoch 的 val 预测平均）
    acc_v = None
    for i in plateau:
        f = 'epoch_%03d.pth' % per_epoch[i]['epoch']
        sd = torch.load(os.path.join(a.ckpt_dir, f), map_location='cpu')
        exp.model.load_state_dict(sd)
        _, vmae, vp, vt = forward_loss(**common, loader=val_loader, want_pred=True)
        acc_v = vp.clone() if acc_v is None else acc_v + vp
    ens_val = acc_v / len(plateau)
    ens_val_mae = float((ens_val - vt).abs().mean())

    # ---- 平台期：权重平均（SWA）----
    avg = None
    for i in plateau:
        f = 'epoch_%03d.pth' % per_epoch[i]['epoch']
        sd = torch.load(os.path.join(a.ckpt_dir, f), map_location='cpu')
        if avg is None:
            avg = {k: v.float().clone() for k, v in sd.items()}
        else:
            for k in avg:
                if avg[k].is_floating_point():
                    avg[k] += sd[k].float()
    for k in avg:
        if avg[k].is_floating_point():
            avg[k] /= len(plateau)
        else:
            avg[k] = avg[k].to(sd[k].dtype)
    exp.model.load_state_dict(avg)
    wa_val, _, _, _ = forward_loss(**common, loader=val_loader)
    wa_mse, wa_mae, _, _ = forward_loss(**common, loader=test_loader)

    print('\n--- 结果 ---')
    print('  ① 现有协议 argmin(val)      : test MSE %.7f / MAE %.7f' % (per_epoch[i_val]['test_mse'], per_epoch[i_val]['test_mae']))
    print('  ② 平台期 预测平均（推荐）    : val MAE %.7f | test MSE %.7f / MAE %.7f' % (ens_val_mae, ens_test_mse, ens_test_mae))
    print('  ③ 平台期 权重平均 SWA        : val MAE %.7f | test MSE %.7f / MAE %.7f' % (wa_val, wa_mse, wa_mae))
    print('  ④ 参考上界 argmin(test)     : test MSE %.7f / MAE %.7f' % (per_epoch[i_test]['test_mse'], per_epoch[i_test]['test_mae']))

    payload = dict(log=a.log, ckpt_dir=a.ckpt_dir, eps=a.eps,
                   per_epoch=per_epoch,
                   plateau_epochs=[per_epoch[i]['epoch'] for i in plateau],
                   protocol_argmin_val=dict(epoch=per_epoch[i_val]['epoch'],
                                            test_mse=per_epoch[i_val]['test_mse'],
                                            test_mae=per_epoch[i_val]['test_mae']),
                   protocol_ens_pred=dict(n=len(plateau), val_mae=ens_val_mae,
                                          test_mse=ens_test_mse, test_mae=ens_test_mae),
                   protocol_ens_weight=dict(n=len(plateau), val_mae=wa_val,
                                            test_mse=wa_mse, test_mae=wa_mae),
                   ref_oracle_argmin_test=dict(epoch=per_epoch[i_test]['epoch'],
                                               test_mse=per_epoch[i_test]['test_mse'],
                                               test_mae=per_epoch[i_test]['test_mae']))
    if a.out:
        with open(a.out, 'w', encoding='utf-8') as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)
        print('\n已写出 %s' % a.out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
