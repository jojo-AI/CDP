# -*- coding: utf-8 -*-
"""
CDP 效率实测：参数量 / FLOPs / 推理延迟 / 吞吐 / 峰值显存。

为什么单独写这个：
  训练日志只给了 params（+0.43%），审稿人 P3-1 要的是 **计算开销**（FLOPs 与延迟）。
  训练代价太高无法做效率扫描，但**推理测量只需要 forward**，单配置秒级完成。

用法：
  python tools/measure_efficiency.py --pilot          # 2 个配置试跑，用于估算总耗时
  python tools/measure_efficiency.py                  # 全量 16 形状 x 3 变体 = 48 次
  python tools/measure_efficiency.py --variants cdp   # 只测某一类变体

变体：
  baseline : 官方 TimeMixer（clean，论文 baseline 骨干）
  stage1   : dynamic 骨干（逐样本自适应权重，Stage-1 产物）
  cdp      : dynamic 骨干 + CDP 校准层（最终方法）

产物：
  D:/fsdownload/efficiency_results.csv  （机器可读）
  stdout 汇总表
"""
import argparse
import csv
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# 各数据集的模型结构（取自真实训练日志的 Namespace，勿凭记忆改）
# ---------------------------------------------------------------------------
DATASETS = {
    'ETTh1':   dict(enc_in=7,   d_model=16, e_layers=2, d_ff=32, batch=128, freq='h'),
    'ETTh2':   dict(enc_in=7,   d_model=16, e_layers=2, d_ff=32, batch=128, freq='h'),
    'ETTm1':   dict(enc_in=7,   d_model=16, e_layers=2, d_ff=32, batch=128, freq='h'),
    'ETTm2':   dict(enc_in=7,   d_model=16, e_layers=2, d_ff=32, batch=128, freq='h'),
    'weather': dict(enc_in=21,  d_model=16, e_layers=3, d_ff=32, batch=128, freq='h'),
    'ECL':     dict(enc_in=321, d_model=16, e_layers=3, d_ff=32, batch=32,  freq='h'),
    'Traffic': dict(enc_in=862, d_model=32, e_layers=3, d_ff=64, batch=16,  freq='h'),
}

HORIZONS = [96, 192, 336, 720]
SEQ_LEN = 96
N_MARK = 4          # timeF 时间特征维度
WARMUP = 10
ITERS = 50


class Cfg(object):
    """复刻 run.py 的 Namespace，只保留建模必需的字段。"""

    def __init__(self, ds, pred_len, dynamic=False):
        d = DATASETS[ds]
        self.task_name = 'long_term_forecast'
        self.seq_len = SEQ_LEN
        self.label_len = 0
        self.pred_len = pred_len
        self.enc_in = d['enc_in']
        self.c_out = d['enc_in']
        self.d_model = d['d_model']
        self.e_layers = d['e_layers']
        self.d_ff = d['d_ff']
        self.moving_avg = 25
        self.dropout = 0.1
        self.embed = 'timeF'
        self.freq = d['freq']
        self.channel_independence = 1
        self.decomp_method = 'moving_avg'
        self.use_norm = 1
        self.down_sampling_layers = 3
        self.down_sampling_window = 2
        self.down_sampling_method = 'avg'
        self.num_class = 1
        self.use_future_temporal_feature = 0
        self.use_amp = False
        # 骨干模式
        self.backbone_mode = 'dynamic' if dynamic else 'clean'
        # CDP 专有
        self.cdp_M = 4
        self.cdp_S = 16
        self.cdp_gain = 1
        self.cdp_out_affine = 1
        self.cdp_stage_w = 1
        self.cdp_err_decay = 0.9
        self.cdp_lambda_t = 0.01
        # ↓ 以下三项必须与最终配方（v5final / at_cal1）一致，否则参数量与 FLOPs 会偏小
        self.cdp_noise_all = 1
        self.cdp_chan_rank = 8
        self.cdp_chan_rank_mul = 8
        self.cdp_unfreeze_head = 1
        self.cdp_unfreeze_norm = 1
        self.cdp_unfreeze_blocks = 0
        self.cdp_dyncal = 1
        self.cdp_unfreeze_pdm = 1


def build(ds, pred_len, variant):
    """返回 (model, batch_size)。"""
    import importlib
    if variant == 'cdp':
        import models.TimeMixer_CDP as M
        cfg = Cfg(ds, pred_len, dynamic=True)
    elif variant == 'stage1':
        import models.TimeMixer as M
        cfg = Cfg(ds, pred_len, dynamic=True)
    else:
        import models.TimeMixer as M
        cfg = Cfg(ds, pred_len, dynamic=False)
    importlib.reload(M)   # 避免 backbone_mode 全局状态串味
    model = M.Model(cfg)
    return model, DATASETS[ds]['batch'], cfg


def flops_of(model, args):
    """用 torch 内置 FlopCounterMode 统计一次前向的 FLOPs。"""
    from torch.utils.flop_counter import FlopCounterMode
    model.eval()
    counter = FlopCounterMode(display=False)
    with torch.no_grad(), counter:
        model(*args)
    return counter.get_total_flops()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pilot', action='store_true')
    ap.add_argument('--variants', default='baseline,stage1,cdp')
    ap.add_argument('--out', default='D:/fsdownload/efficiency_results.csv')
    args = ap.parse_args()

    variants = [v.strip() for v in args.variants.split(',') if v.strip()]
    if args.pilot:
        jobs = [('ETTh1', 96), ('Traffic', 96)]
    else:
        jobs = [(ds, h) for ds in DATASETS for h in HORIZONS]

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    rows = []
    t_start = time.time()
    n_done = 0

    for ds, h in jobs:
        for variant in variants:
            t0 = time.time()
            model, bs, cfg = build(ds, h, variant)
            model = model.to(dev)
            enc_in = cfg.enc_in
            x_enc = torch.randn(bs, SEQ_LEN, enc_in, device=dev)
            x_mark_enc = torch.randn(bs, SEQ_LEN, N_MARK, device=dev)
            x_dec = torch.randn(bs, cfg.pred_len, enc_in, device=dev)
            x_mark_dec = torch.randn(bs, cfg.pred_len, N_MARK, device=dev)
            fargs = (x_enc, x_mark_enc, x_dec, x_mark_dec)

            model.eval()
            with torch.no_grad():
                for _ in range(WARMUP):
                    model(*fargs)
                if dev == 'cuda':
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                t_first = time.perf_counter()
                for _ in range(ITERS):
                    model(*fargs)
                if dev == 'cuda':
                    torch.cuda.synchronize()
                dt = (time.perf_counter() - t_first) / ITERS
                peak = (torch.cuda.max_memory_allocated() / 1024 ** 2) if dev == 'cuda' else float('nan')

            fl = flops_of(model, fargs)
            n_all = sum(p.numel() for p in model.parameters())
            n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)

            rows.append(dict(
                dataset=ds, pred_len=h, variant=variant, enc_in=enc_in, batch=bs,
                params_all=n_all, params_trainable=n_tr,
                flops_batch=fl, flops_per_sample=fl / bs,
                latency_ms=dt * 1000, throughput_sps=bs / dt,
                peak_mem_mib=peak, wall_s=time.time() - t0,
            ))
            n_done += 1
            del model
            if dev == 'cuda':
                torch.cuda.empty_cache()
            print('[%2d] %-8s h%-4d %-8s lat=%7.2f ms  mem=%6.0f MiB  params=%9d  (%4.1fs)'
                  % (n_done, ds, h, variant, dt * 1000, peak, n_all, time.time() - t0))
            sys.stdout.flush()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    total = time.time() - t_start
    print('\n%d 次测量完成，共 %.1f 秒（平均 %.1f 秒/次）' % (n_done, total, total / n_done))
    if args.pilot:
        full = len(DATASETS) * len(HORIZONS) * len(variants)
        print('全量 %d 次预计耗时：%.1f 分钟' % (full, total / n_done * full / 60.0))
        print('（估算式：%d 形状 x %d 变体 = %d 次）' % (len(DATASETS) * len(HORIZONS), len(variants), full))
    print('CSV ->', args.out)


if __name__ == '__main__':
    main()
