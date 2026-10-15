# M/K Scan Results (TimeMixer++ + CDP, ETTh1-96, seed 2021)

Two-stage protocol, self-sufficient mode: the K=4 default backbone is retrained locally;
the M-sweep reuses this frozen backbone for Stage-2 only (all five M points share one backbone,
internally self-consistent); each K point retrains its own backbone (down_sampling_layers = K-1).
Stage-2 = archived winner at_cal1 recipe (L1 loss, val-MAE selection). Run date: 2026-10-15,
single consumer GPU (NVIDIA CMP 170HX 64GB), 9/9 points status=ok, wall clock ~18 min.
Numbers are internally self-consistent but NOT directly comparable to the archived main-experiment
winner (different retrained backbone).

## M sweep (segmentation ratio, K=4 frozen backbone)

| seg_ratio r | cdp_M (=4r) | MSE | MAE | dMAE% vs r=1.0 |
|---|---|---|---|---|
| 0.6 | 2.4 | 0.366966 | 0.385733 | -0.049% |
| 0.8 | 3.2 | 0.367393 | 0.385910 | -0.003% |
| 1.0 | 4.0 | 0.367236 | 0.385920 | +0.000% |
| 1.2 | 4.8 | 0.367909 | 0.386033 | +0.029% |
| 1.5 | 6.0 | 0.367961 | 0.386083 | +0.042% |

Total MAE span 0.091%: flat, no exploitable valley beyond the noise floor.

## K sweep (resolution level, M=4, per-K retrained backbone)

| K | down_sampling_layers | Stage-1 MSE | Stage-1 MAE | Stage-2 MSE | Stage-2 MAE | dMAE% vs K=4 |
|---|---|---|---|---|---|---|
| 2 | 1 | 0.370667 | 0.389643 | 0.364498 | 0.385637 | -0.073% |
| 3 | 2 | 0.378612 | 0.389406 | 0.372163 | 0.386538 | +0.160% |
| 4 | 3 | 0.381617 | 0.394284 | 0.367236 | 0.385920 | +0.000% |
| 5 | 4 | 0.389634 | 0.397292 | 0.374906 | 0.389472 | +0.920% |
| 6 | 5 | 0.376889 | 0.389191 | 0.373908 | 0.387602 | +0.436% |

Default K=4 sits in the valley (K=3 and K=5 both worse); K=2's edge is within the noise floor.

## Anchor consistency check

mk_K4 (Stage-2 on the K=4 backbone) vs mk_M_r1.0 (same point via the M-sweep path):
MSE diff 4.33e-09, MAE diff 3.57e-08 (identical up to float noise).

## Files

- `mk_*.json` / `st1_*.json`: per-point results (axis, seg_ratio/K, cdp_M, down_sampling_layers, mse, mae, stage1_mse/mae, backbone_source, stage1_ckpt, status, log).
- `mk_scan_console.log`: runner console output.
- Runner code: `mk_scan/run_batch_mk.py`, `mk_scan/summarize_mk.py`; entry: `run_all.sh`.