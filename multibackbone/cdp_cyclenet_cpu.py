# -*- coding: utf-8 -*-
"""
CDP-on-CycleNet (CPU/GPU auto).

Protocol mirrors CycleNet official scripts (NeurIPS 2024, Linear-Input-96):
  - baseline: Adam(lr=0.01) + OneCycleLR(pct_start=0.3), MSE train loss,
    train_epochs=30, patience=5 early-stop on val MSE, model selection on val MSE,
    metrics in scaled space, test loader drop_last=False (their factory default).
  - per-dataset official hyperparams (Linear-Input-96):
      ETTh1/ETTh2: cycle=24, batch=256; ETTm1/ETTm2: cycle=96, batch=256;
      weather: cycle=144, batch=256; electricity: cycle=168, batch=64;
      traffic: cycle=168, batch=64.
  - mode cdp  : two-stage, backbone FROZEN; PSC linear calibration
                y_cal = y_base + A*trend + B*noise + C*freq + d, L1 (CAO), val-MAE selection.
  - mode head : PSC + unfreeze prediction head (self.model Linear/MLP), joint L1.
  - mode cao  : only unfreeze head + L1, no PSC.

Usage:
  python cdp_cyclenet_cpu.py --data ETTh1 --pred_len 96 --mode baseline --seed 2021
  python cdp_cyclenet_cpu.py --data ETTh1 --pred_len 96 --mode head --seed 2021
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
from models.CycleNet import Model  # noqa: E402

# official per-dataset hyperparams: (cycle, batch_size, enc_in)
OFFICIAL = {
    'ETTh1': (24, 256, 7), 'ETTh2': (24, 256, 7),
    'ETTm1': (96, 256, 7), 'ETTm2': (96, 256, 7),
    'weather': (144, 256, 21), 'electricity': (168, 64, 321),
    'traffic': (168, 64, 862),
}


def metrics(pred, true):
    return float(np.mean((pred - true) ** 2)), float(np.mean(np.abs(pred - true)))


def fft_top1_extrapolate(xw, seq_len, pred_len):
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
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--patience', type=int, default=5)
    ap.add_argument('--lr', type=float, default=0.01)
    ap.add_argument('--cdp_epochs', type=int, default=60)
    ap.add_argument('--cdp_patience', type=int, default=8)
    ap.add_argument('--cdp_lr', type=float, default=5e-3)
    ap.add_argument('--seed', type=int, default=2021)
    ap.add_argument('--enc_in', type=int, default=None)
    ap.add_argument('--batch_size', type=int, default=None)
    ap.add_argument('--cycle', type=int, default=None)
    args_cli = ap.parse_args()

    cyc, bs, enc = OFFICIAL.get(args_cli.data, (24, 256, 7))
    enc = args_cli.enc_in or enc
    bs = args_cli.batch_size or bs
    cyc = args_cli.cycle or cyc

    torch.manual_seed(args_cli.seed)
    np.random.seed(args_cli.seed)
    torch.set_num_threads(int(os.environ.get('CDP_THREADS', max(1, os.cpu_count() - 1))))

    if args_cli.data.startswith('ETT'):
        root_path = os.path.join(ROOT, 'dataset', 'ETT-small') + os.sep
        data_key = args_cli.data
    else:
        root_path = os.path.join(ROOT, 'dataset', 'custom') + os.sep
        data_key = 'custom'  # factory dict key; file is dataset/custom/<data>.csv

    cfg = SimpleNamespace(
        task_name='long_term_forecast', seq_len=96, label_len=0,
        pred_len=args_cli.pred_len, e_layers=2, enc_in=enc, c_out=enc,
        d_model=512, d_ff=2048, dropout=0.0, embed='timeF', freq='h',
        use_revin=1, cycle=cyc, model_type='linear',
        data=data_key, root_path=root_path,
        data_path=args_cli.data + '.csv', features='M', target='OT',
        batch_size=bs, num_workers=0,
    )

    loaders = {}
    for flag in ['train', 'val', 'test']:
        ds, loader = data_provider(cfg, flag)  # their factory: test drop_last=False already
        loaders[flag] = loader
        print(f'[{args_cli.data}] {flag}: {len(ds)} windows')

    ckpt_dir = os.path.join(ROOT, 'cdp_ckpt')
    os.makedirs(ckpt_dir, exist_ok=True)
    tag = f'{args_cli.data}_{args_cli.pred_len}_seed{args_cli.seed}'
    ckpt_path = os.path.join(ckpt_dir, f'cyclenet_{tag}.pth')
    res_path = os.path.join(ROOT, 'cdp_results')
    os.makedirs(res_path, exist_ok=True)

    DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device:', DEV)
    model = Model(cfg).to(DEV)
    criterion_mse = nn.MSELoss()
    criterion_l1 = nn.L1Loss()

    def backbone(batch_x, batch_cycle):
        return model(batch_x, batch_cycle)  # [B, pred_len, C] scaled space

    def predict(loader, cal=None):
        model.eval()
        if cal is not None:
            cal.eval()
        preds, trues = [], []
        with torch.no_grad():
            for batch_x, batch_y, _xm, _ym, batch_cycle in loader:
                batch_x = batch_x.float().to(DEV)
                batch_cycle = batch_cycle.int().to(DEV)
                y_base = backbone(batch_x, batch_cycle)
                if cal is not None:
                    t_hat, n_hat, f_hat = input_side_features(
                        batch_x, cfg.seq_len, args_cli.pred_len)
                    yb = y_base.permute(0, 2, 1)
                    out = cal(yb, t_hat, n_hat, f_hat).permute(0, 2, 1)
                else:
                    out = y_base
                preds.append(out.detach().cpu().numpy())
                trues.append(batch_y[:, -args_cli.pred_len:, :].detach().cpu().numpy())
        p, t = np.concatenate(preds), np.concatenate(trues)
        return metrics(p, t)

    if args_cli.mode == 'baseline':
        optim = torch.optim.Adam(model.parameters(), lr=args_cli.lr)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optim, max_lr=args_cli.lr, pct_start=0.3,
            steps_per_epoch=len(loaders['train']), epochs=args_cli.epochs)
        best_val, best_state, bad = float('inf'), None, 0
        for ep in range(1, args_cli.epochs + 1):
            model.train()
            tl, nb = 0.0, 0
            for batch_x, batch_y, _xm, _ym, batch_cycle in loaders['train']:
                batch_x, batch_y = batch_x.float().to(DEV), batch_y.float().to(DEV)
                batch_cycle = batch_cycle.int().to(DEV)
                optim.zero_grad()
                out = backbone(batch_x, batch_cycle)
                loss = criterion_mse(out, batch_y[:, -args_cli.pred_len:, :])
                loss.backward()
                optim.step()
                scheduler.step()
                tl += float(loss); nb += 1
            vm, _ = predict(loaders['val'])
            tm, tmae = predict(loaders['test'])
            star = ''
            if vm < best_val:
                best_val, best_state, bad = vm, copy.deepcopy(model.state_dict()), 0
                star = ' *'
            else:
                bad += 1
            print(f'ep{ep:02d} train_mse={tl/max(nb,1):.4f} val_mse={vm:.4f} '
                  f'test_mse={tm:.4f} test_mae={tmae:.4f}{star}')
            if bad >= args_cli.patience:
                print(f'baseline early stop at ep{ep}')
                break
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
            if name.startswith('model.'):  # prediction head = self.model (Linear/MLP)
                p.requires_grad_(True)
        train_params = [p for p in model.parameters() if p.requires_grad]
        print('unfroze head params:', sum(p.numel() for p in train_params))
    else:
        train_params = None

    cal = CalibratedModel(args_cli.pred_len)
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
        for batch_x, batch_y, _xm, _ym, batch_cycle in loaders['train']:
            batch_x, batch_y = batch_x.float().to(DEV), batch_y.float().to(DEV)
            batch_cycle = batch_cycle.int().to(DEV)
            optim.zero_grad()
            if args_cli.mode == 'cao':
                out = backbone(batch_x, batch_cycle)
                loss = criterion_l1(out, batch_y[:, -args_cli.pred_len:, :])
            else:
                if args_cli.mode == 'head':
                    y_base = backbone(batch_x, batch_cycle)
                else:
                    with torch.no_grad():
                        y_base = backbone(batch_x, batch_cycle)
                t_hat, n_hat, f_hat = input_side_features(
                    batch_x, cfg.seq_len, args_cli.pred_len)
                yb = y_base.permute(0, 2, 1)
                by = batch_y[:, -args_cli.pred_len:, :].permute(0, 2, 1)
                yc = cal(yb, t_hat, n_hat, f_hat)
                loss = criterion_l1(yc, by)
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
           'param_norms': pnorms, 'seed': args_cli.seed}
    with open(os.path.join(res_path, f'cdp_{args_cli.mode}_{tag}.json'), 'w') as f:
        json.dump(res, f, indent=2)
    print('=' * 60)
    print(f'RESULT {args_cli.data}-{args_cli.pred_len} ({args_cli.mode})')
    print(f'  CycleNet       MSE={base_mse:.4f}  MAE={base_mae:.4f}')
    print(f'  CycleNet+CDP   MSE={tm:.4f}  MAE={tmae:.4f}')
    print(f'  delta          MSE={dm:+.2f}%  MAE={da:+.2f}%')
    print(f'  |A|={pnorms["A"]:.4f} |B|={pnorms["B"]:.4f} |C|={pnorms["C"]:.4f} |d|={pnorms["d"]:.4f}')


if __name__ == '__main__':
    run()
