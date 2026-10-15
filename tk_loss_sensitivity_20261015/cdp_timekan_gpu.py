# -*- coding: utf-8 -*-
"""
CDP-on-TimeKAN (GPU build; auto-falls back to CPU if no CUDA).

Protocol (mirrors TimeKAN official exp_long_term_forecasting.py, ETTh scripts):
  - baseline: Adam(lr=0.01) + OneCycleLR, MSE train loss, batch 128,
    train_epochs=10, patience=10 (never early-stops), metrics in scaled space.
  - mode cdp      : two-stage, backbone FROZEN; PSC linear calibration
                    y_cal = y_base + A*trend + B*noise + C*freq + d
                    trained with L1 (CAO), model selection on val MAE.
  - mode head     : PSC + unfreeze predict_layer/projection_layer (CSC-lite:
                    MAE-side recalibration of the prediction head), joint L1.

Usage:
  python cdp_timekan_cpu.py --data ETTh1 --pred_len 96 --mode baseline
  python cdp_timekan_cpu.py --data ETTh1 --pred_len 96 --mode cdp
  python cdp_timekan_cpu.py --data ETTh1 --pred_len 96 --mode head
"""
import argparse
import copy
import json
import math
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from data_provider.data_factory import data_provider  # noqa: E402
from models.TimeKAN import Model  # noqa: E402


def metrics(pred, true):
    return float(np.mean((pred - true) ** 2)), float(np.mean(np.abs(pred - true)))


def fft_top1_extrapolate(xw, seq_len, pred_len):
    """xw: [B, C, L] -> [B, C, T] dominant-frequency continuation (training-free)."""
    X = torch.fft.rfft(xw, dim=-1)
    amp = torch.abs(X)[:, :, 1:]
    idx = torch.argmax(amp, dim=-1, keepdim=True) + 1
    a = torch.gather(torch.abs(X), -1, idx).squeeze(-1)
    ph = torch.gather(torch.angle(X), -1, idx).squeeze(-1)
    grid = torch.arange(seq_len, seq_len + pred_len, dtype=xw.dtype, device=xw.device)
    f = (2.0 / seq_len) * a.unsqueeze(-1) * torch.cos(
        2 * math.pi * idx * grid / seq_len + ph.unsqueeze(-1))
    return f


def input_side_features(batch_x, seq_len, pred_len):
    """trend = last-value + mean-slope extrapolation [B,C,T];
    noise = RMS of first differences [B,C,1];
    freq  = FFT top-1 extrapolation [B,C,T]."""
    xw = batch_x.permute(0, 2, 1)
    last = xw[:, :, -1:]
    k = min(16, seq_len - 1)
    slope = torch.mean(torch.diff(xw[:, :, -k - 1:], dim=-1), dim=-1, keepdim=True)
    grid = torch.arange(1, pred_len + 1, dtype=xw.dtype, device=xw.device)
    trend = last + slope * grid.unsqueeze(0).unsqueeze(0)
    noise = torch.sqrt(torch.mean(torch.diff(xw, dim=-1) ** 2, dim=-1, keepdim=True))
    freq = fft_top1_extrapolate(xw, seq_len, pred_len)
    return trend, noise, freq


class CalibratedModel(nn.Module):
    """PSC: y_cal = y_base + A*t_hat + B*n_hat + C*f_hat + d  ([T] vectors)."""
    def __init__(self, pred_len):
        super().__init__()
        self.A = nn.Parameter(torch.zeros(pred_len))
        self.B = nn.Parameter(torch.zeros(pred_len))
        self.C = nn.Parameter(torch.zeros(pred_len))
        self.d = nn.Parameter(torch.zeros(pred_len))

    def forward(self, y_base, t_hat, n_hat, f_hat):
        return y_base + self.A * t_hat + self.B * n_hat + self.C * f_hat + self.d


def run():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='ETTh1')
    ap.add_argument('--pred_len', type=int, default=96)
    ap.add_argument('--mode', choices=['baseline', 'cdp', 'head', 'cao'], default='baseline')
    ap.add_argument('--epochs', type=int, default=10)
    # batch_size: 0/负数 = 按通道数自动选择（见下方 auto_batch_size）。
    # 原因：本包 cfg 用 channel_independence=1，显存占用 ≈ batch × enc_in，
    #   ECL(321ch) 实测 10.3 GiB/12 GiB；traffic(862ch) 若沿用 128 必然 OOM。
    ap.add_argument('--batch_size', type=int, default=0)
    ap.add_argument('--lr', type=float, default=0.01)
    ap.add_argument('--cdp_epochs', type=int, default=60)
    ap.add_argument('--cdp_patience', type=int, default=8)
    ap.add_argument('--cdp_lr', type=float, default=5e-3)
    # [2026-10-15 新增，向后兼容] Stage-2 训练损失。默认 l1 与原行为逐位一致，
    # 故既有 232 组结果不受影响、仍可复现。仅用于超参敏感性对照实验。
    ap.add_argument('--cdp_loss', choices=['l1', 'mse'], default='l1',
                    help='Stage-2 training loss (selection rule stays val-MAE)')
    ap.add_argument('--suffix', default='',
                    help='suffix appended to result JSON filename')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--enc_in', type=int, default=7)
    args_cli = ap.parse_args()

    # ---- auto batch size（保持 weather/ECL 与既有结果一致，只给 traffic 降 batch）----
    if args_cli.batch_size <= 0:
        if args_cli.enc_in <= 400:          # weather(21) / electricity(321)
            args_cli.batch_size = 128
        elif args_cli.enc_in <= 600:
            args_cli.batch_size = 64
        else:                                # traffic(862) 等宽通道数据集
            args_cli.batch_size = 32
    print(f'[auto] enc_in={args_cli.enc_in} -> batch_size={args_cli.batch_size}',
          flush=True)

    torch.manual_seed(args_cli.seed)
    np.random.seed(args_cli.seed)
    torch.set_num_threads(int(os.environ.get('CDP_THREADS',
        max(1, os.cpu_count() - 1))))

    if args_cli.data.startswith('ETT'):
        root_path = os.path.join(ROOT, 'dataset', 'ETT-small') + os.sep
    else:
        root_path = os.path.join(ROOT, 'dataset', 'custom') + os.sep

    cfg = SimpleNamespace(
        task_name='long_term_forecast', seq_len=96, label_len=0,
        pred_len=args_cli.pred_len, e_layers=2, enc_in=args_cli.enc_in, c_out=args_cli.enc_in,
        d_model=16, d_ff=32, dropout=0.1, embed='timeF', freq='h',
        down_sampling_layers=1, down_sampling_window=2, begin_order=0,
        moving_avg=25, channel_independence=1, use_norm=1,
        use_future_temporal_feature=0, decomp_method='moving_avg',
        data=args_cli.data, root_path=root_path,
        data_path=args_cli.data + '.csv', features='M', target='OT',
        train_only=False, batch_size=args_cli.batch_size,
        num_workers=0, seasonal_patterns='Monthly',
    )

    loaders = {}
    for flag in ['train', 'val', 'test']:
        ds, _ = data_provider(cfg, flag)
        loaders[flag] = DataLoader(ds, batch_size=cfg.batch_size,
                                   shuffle=(flag == 'train'),
                                   drop_last=(flag == 'train'),
                                   num_workers=0)
        print(f'[{args_cli.data}] {flag}: {len(ds)} windows')

    ckpt_dir = os.path.join(ROOT, 'cdp_ckpt')
    os.makedirs(ckpt_dir, exist_ok=True)
    tag = f'{args_cli.data}_{args_cli.pred_len}_seed{args_cli.seed}'
    ckpt_path = os.path.join(ckpt_dir, f'timekan_{tag}.pth')
    res_path = os.path.join(ROOT, 'cdp_results')
    os.makedirs(res_path, exist_ok=True)

    DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device:', DEV)
    model = Model(cfg).to(DEV)
    criterion_mse = nn.MSELoss()
    criterion_l1 = nn.L1Loss()
    # Stage-2 校准器训练损失：默认 l1（与原行为一致）
    cdp_criterion = criterion_mse if args_cli.cdp_loss == 'mse' else criterion_l1

    def backbone(batch_x):
        return model(batch_x, None, None, None)  # [B, pred_len, C] scaled space

    def predict(loader, cal=None):
        model.eval()
        if cal is not None:
            cal.eval()
        # 在线累加指标（O(1) 内存）。原实现用 np.concatenate 物化 (N,pred_len,C) 巨数组：
        # traffic-720 test = 2789*720*862 ≈ 1.73e9 元素 = 6.45 GiB × (preds+trues) ≈ 13 GiB → CPU RAM OOM。
        # 数学上与 metrics(np.concatenate(...)) 等价（都是全元素均值）。
        n_el = 0
        sum_se = 0.0
        sum_ae = 0.0
        with torch.no_grad():
            for batch_x, batch_y, _bm, _bdy in loader:
                batch_x, batch_y = batch_x.float().to(DEV), batch_y.float().to(DEV)
                y_base = backbone(batch_x)
                if cal is not None:
                    t_hat, n_hat, f_hat = input_side_features(
                        batch_x, cfg.seq_len, args_cli.pred_len)
                    yb = y_base.permute(0, 2, 1)                        # [B,C,T]
                    yc = cal(yb, t_hat, n_hat, f_hat).permute(0, 2, 1)
                    out = yc
                else:
                    out = y_base
                diff = out - batch_y[:, -args_cli.pred_len:, :]
                sum_se += float((diff * diff).sum().item())
                sum_ae += float(diff.abs().sum().item())
                n_el += int(diff.numel())
        return sum_se / max(n_el, 1), sum_ae / max(n_el, 1)

    if args_cli.mode == 'baseline':
        optim = torch.optim.Adam(model.parameters(), lr=args_cli.lr)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optim, max_lr=args_cli.lr,
            steps_per_epoch=len(loaders['train']), epochs=args_cli.epochs)
        best_val, best_state = float('inf'), None
        for ep in range(1, args_cli.epochs + 1):
            model.train()
            tl, nb = 0.0, 0
            for batch_x, batch_y, _bm, _bdy in loaders['train']:
                batch_x, batch_y = batch_x.float().to(DEV), batch_y.float().to(DEV)
                optim.zero_grad()
                out = backbone(batch_x)
                loss = criterion_mse(out, batch_y[:, -args_cli.pred_len:, :])
                loss.backward()
                optim.step()
                scheduler.step()
                tl += float(loss); nb += 1
            vm, vmae = predict(loaders['val'])
            tm, tmae = predict(loaders['test'])
            star = ''
            if vm < best_val:
                best_val = vm
                best_state = copy.deepcopy(model.state_dict())
                star = ' *'
            print(f'ep{ep:02d} train_mse={tl/max(nb,1):.4f} val_mse={vm:.4f} '
                  f'test_mse={tm:.4f} test_mae={tmae:.4f}{star}')
        model.load_state_dict(best_state)
        tm, tmae = predict(loaders['test'])
        torch.save(best_state, ckpt_path)
        res = {'data': args_cli.data, 'pred_len': args_cli.pred_len,
               'mode': 'baseline', 'test_mse': tm, 'test_mae': tmae,
               'best_val_mse': best_val, 'seed': args_cli.seed}
        with open(os.path.join(res_path, f'baseline_{tag}.json'), 'w') as f:
            json.dump(res, f, indent=2)
        print(f'BASELINE {args_cli.data}-{args_cli.pred_len}: MSE={tm:.4f} MAE={tmae:.4f}')
        return

    # ---------------- two-stage: load frozen backbone ----------------
    model.load_state_dict(torch.load(ckpt_path, weights_only=True, map_location=DEV))
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    base_mse, base_mae = predict(loaders['test'])
    print(f'frozen backbone test: MSE={base_mse:.4f} MAE={base_mae:.4f}')

    if args_cli.mode in ('head', 'cao'):
        for name, p in model.named_parameters():
            if name.startswith('predict_layer') or name.startswith('projection_layer'):
                p.requires_grad_(True)
        train_params = [p for p in model.parameters() if p.requires_grad]
        print('unfroze params:', sum(p.numel() for p in train_params))
    else:
        train_params = None

    cal = CalibratedModel(args_cli.pred_len).to(DEV)
    if args_cli.mode == 'cao':
        optim = torch.optim.Adam(train_params, lr=1e-3)
    elif train_params is not None:
        optim = torch.optim.Adam([
            {'params': train_params, 'lr': 1e-3},
            {'params': cal.parameters(), 'lr': args_cli.cdp_lr},
        ])
    else:
        optim = torch.optim.Adam(cal.parameters(), lr=args_cli.cdp_lr)

    best_val, best_state, bad = float('inf'), None, 0
    for ep in range(1, args_cli.cdp_epochs + 1):
        if args_cli.mode in ('head', 'cao'):
            model.train()
        cal.train()
        tl, nb = 0.0, 0
        for batch_x, batch_y, _bm, _bdy in loaders['train']:
            batch_x, batch_y = batch_x.float().to(DEV), batch_y.float().to(DEV)
            optim.zero_grad()
            if args_cli.mode == 'cao':
                out = backbone(batch_x)             # grads flow to unfrozen head
                loss = criterion_l1(out, batch_y[:, -args_cli.pred_len:, :])
            else:
                if args_cli.mode == 'head':
                    y_base = backbone(batch_x)      # grads flow to unfrozen head
                else:
                    with torch.no_grad():
                        y_base = backbone(batch_x)
                t_hat, n_hat, f_hat = input_side_features(
                    batch_x, cfg.seq_len, args_cli.pred_len)
                yb = y_base.permute(0, 2, 1)
                by = batch_y[:, -args_cli.pred_len:, :].permute(0, 2, 1)
                yc = cal(yb, t_hat, n_hat, f_hat)
                loss = cdp_criterion(yc, by)
            loss.backward()
            optim.step()
            tl += float(loss); nb += 1
        use_cal = None if args_cli.mode == 'cao' else cal
        vm, vmae = predict(loaders['val'], cal=use_cal)
        tm, tmae = predict(loaders['test'], cal=use_cal)
        v = vmae
        star = ''
        if v < best_val:
            best_val, bad = v, 0
            best_state = (copy.deepcopy(cal.state_dict()),
                          copy.deepcopy(model.state_dict())
                          if args_cli.mode in ('head', 'cao') else None)
            star = ' *'
        else:
            bad += 1
        print(f'{args_cli.mode} ep{ep:02d} train_l1={tl/max(nb,1):.4f} val_mae={vmae:.4f} '
              f'test_mse={tm:.4f} test_mae={tmae:.4f}{star}')
        if bad >= args_cli.cdp_patience:
            print(f'{args_cli.mode} early stop at ep{ep}')
            break

    cal_state, model_state = best_state
    cal.load_state_dict(cal_state)
    if model_state is not None:
        model.load_state_dict(model_state)
    tm, tmae = predict(loaders['test'], cal=None if args_cli.mode == 'cao' else cal)
    dm = (tm - base_mse) / base_mse * 100
    da = (tmae - base_mae) / base_mae * 100
    with torch.no_grad():
        pnorms = {k: float(v.norm()) for k, v in cal.state_dict().items()}
    res = {'data': args_cli.data, 'pred_len': args_cli.pred_len, 'mode': args_cli.mode,
           'base_mse': base_mse, 'base_mae': base_mae,
           'cdp_mse': tm, 'cdp_mae': tmae,
           'delta_mse_pct': dm, 'delta_mae_pct': da,
           'param_norms': pnorms, 'seed': args_cli.seed,
           'cdp_loss': args_cli.cdp_loss, 'cdp_lr': args_cli.cdp_lr}
    with open(os.path.join(res_path, f'cdp_{args_cli.mode}_{tag}{args_cli.suffix}.json'), 'w') as f:
        json.dump(res, f, indent=2)
    print('=' * 60)
    print(f'RESULT {args_cli.data}-{args_cli.pred_len} ({args_cli.mode})')
    print(f'  TimeKAN       MSE={base_mse:.4f}  MAE={base_mae:.4f}')
    print(f'  TimeKAN+CDP   MSE={tm:.4f}  MAE={tmae:.4f}')
    print(f'  delta         MSE={dm:+.2f}%  MAE={da:+.2f}%')
    print(f'  |A|={pnorms["A"]:.4f} |B|={pnorms["B"]:.4f} |C|={pnorms["C"]:.4f} |d|={pnorms["d"]:.4f}')


if __name__ == '__main__':
    run()
