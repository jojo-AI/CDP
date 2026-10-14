# CDP: Calibration-aligned Dynamic Patching for Time Series Forecasting

Official implementation of **CDP** — a training-free *dynamic patching* (DPS) module
plus a component-side / prediction-side *calibration stack* (MA / CSC / CAO / PSC),
aligned to the delivered metric (MAE) rather than the training loss.

CDP is a plug-in framework evaluated on TimeMixer++ as the primary backbone and
transferred unchanged to **TimeKAN** (ICLR 2025), **CycleNet** (NeurIPS 2024) and
**iTransformer** (ICLR 2024) under their official protocols.

## Repository layout

```
run.py                     single entry point (Stage-1 backbone / Stage-2 CDP, --model)
run_mk.py                  generated variant used by the M/K sensitivity sweep
models/
  TimeMixer.py             TimeMixer++ backbone (clean/dynamic season-trend mixing)
  TimeMixer_CDP.py         CDP plug-in: DPS (dynamic patch segmentation),
                           MA (cross-resolution alignment), CE, CSC, PSC, DynFinalCal
exp/                       experiment drivers (long/short-term forecast, CDP variant)
data_provider/             dataset loaders (ETT, electricity, traffic, weather, ...)
utils/                     metrics, losses, tools
tools/                     analysis: select_best, count_wins, decompose_gain,
                           multiseed stats, efficiency measurement, determinism probes
scripts/
  long_term_forecast/      per-dataset baseline scripts (ETT/ECL/Traffic/Weather/Solar)
  *.sh                     two-stage runners: run_stage1_st.sh, run_cdp_v4.sh,
                           run_autotune.sh, run_multiseed.sh, ...
```

## Requirements

```bash
pip install -r requirements.txt   # Python 3.10+ (developed on 3.13), PyTorch + CUDA
```

Datasets: standard TimeMixer/Long-Term-Series-Benchmark layout
(`dataset/ETT-small/ETTh1.csv`, `dataset/electricity/electricity.csv`,
`dataset/traffic/traffic.csv`, `dataset/weather/weather.csv`).

## Quick start (ETTh1-96, two-stage protocol)

```bash
# ---- Stage 1: backbone (dynamic season-trend mixing, pure-L1 objective) ----
python -u run.py --task_name long_term_forecast --model TimeMixer --features M \
  --seq_len 96 --label_len 0 --d_layers 1 --des ST1 --itr 1 --learning_rate 0.01 \
  --num_workers 0 --train_loss l1 --train_l1_mu 0.0 \
  --down_sampling_layers 3 --down_sampling_method avg --down_sampling_window 2 \
  --train_epochs 30 --patience 8 \
  --root_path ./dataset/ETT-small/ --data_path ETTh1.csv --data ETTh1 \
  --enc_in 7 --dec_in 7 --c_out 7 --e_layers 2 --factor 1 --d_model 16 --d_ff 32 \
  --batch_size 128 --pred_len 96 \
  --model_id ETTh1_st1_96_dynamic_l10.0 --backbone_mode dynamic

# ---- Stage 2: CDP calibration (frozen backbone + calibration stack) ----
python -u run.py --task_name long_term_forecast --model TimeMixer_CDP --features M \
  ... --model_id ETTh1_v4_96_at_cal1 \
  --cdp_backbone_checkpoint ./checkpoints/<stage-1 dir> \
  --cdp_learning_rate 5e-4 --cdp_loss l1 --cdp_monitor mae \
  --cdp_unfreeze_head 1 --cdp_unfreeze_norm 1 --cdp_chan_rank 8 --cdp_chan_rank_mul 8 \
  --backbone_mode dynamic --cdp_unfreeze_pdm 1 --cdp_st_lr_ratio 0.2 --cdp_dyncal 1 \
  --cdp_cal_lr_ratio 1
```

Full per-dataset commands (all 7 datasets × 4 horizons, stage-by-stage) are
documented in the paper's reproducibility appendix; the driver scripts
(`run_stage1_st.sh`, `run_cdp_v4.sh`, `run_autotune.sh`) encode the same protocol.

## Protocol notes

- All deltas are computed against faithfully reproduced baselines under identical
  fixed seeds (2021–2026), never against numbers quoted from other papers.
- Checkpoints under `checkpoints/` and datasets under `dataset/` are not committed.
- Key switches: `--cdp_M` (DPS resolution factor), `--cdp_S` (descriptor dim),
  `--cdp_loss` / `--cdp_monitor` (Stage-2 objective / selection metric),
  `--backbone_mode clean|dynamic`, `--train_loss l1|mse` (Stage-1 objective).

## License

Released for academic review and research use.
