#!/bin/bash
# =============================================================================
# 根因追查 第二轮（2026-10-08）
#   目标：判定 weather-96 det_mode=1 同 seed 分叉（ep1 Vali Loss 差 ~1e-5）
#         到底来自「训练轨迹」还是「eval 前向」。
#
#   线索：weather det=1 三次重复
#     ep1 train 0.3688166/0.3688172/0.3688168   ← 只差 7 位小数
#     ep8 起 val 已差 0.5%；best-val epoch 分别落在 ep6/ep9/ep9
#     best-val 三值 0.2597705/0.2592743/0.2593278 → 近乎平局
#     ⇒ 猜测：训练只有 ~1e-6 的 FP 噪声，被「平坦 val plateau + argmin 选择」放大成 1.13%
#
#   A 段：同一 checkpoint 两次独立进程跑 eval 前向 —— 判 eval 是否干净
#   B 段：真实数据跑满 1 个 epoch（286 步）逐步哈希 —— 判训练轨迹哪一步先分叉
#
#   注意：本脚本新增，不改动任何在跑的脚本。
# =============================================================================
PY="/c/Users/Administrator/.workbuddy/binaries/python/envs/default/Scripts/python.exe"
cd "/d/fsdownload/TimeMixer" || exit 1
LOG="/d/fsdownload/rootcause2.log"
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

SMOKE="D:/fsdownload/det_weather_det1/same1.log"
CK="./checkpoints/long_term_forecast_weather_detsmoke_d1_same1_none_TimeMixer_custom_sl96_pl96_dm16_nh8_el3_dl1_df32_fc3_ebtimeF_dtTrue_DETSMOKE_0"

say "########## ROOTCAUSE ROUND-2 START (pid=$$) ##########"

say "=== A. eval 前向确定性（同一 checkpoint，两次独立进程，flag=val）==="
for r in 1 2; do
  say "--- A run$r ---"
  "$PY" -u tools/probe_eval_det.py --log "$SMOKE" --ckpt "$CK" --flag val --det 1 \
        --seed 2021 --out "D:/fsdownload/evaldet_val_${r}.json" 2>&1 \
    | grep -avE "^\[PDM\]|^\[CDP\]" | tee -a "$LOG"
done
"$PY" -u tools/probe_eval_det.py --compare D:/fsdownload/evaldet_val_1.json D:/fsdownload/evaldet_val_2.json 2>&1 | tee -a "$LOG"

say "=== B. 训练轨迹完整性（跑满 286 步 = 1 个 epoch，稀疏参数哈希）==="
for r in 1 2; do
  say "--- B run$r ---"
  "$PY" -u tools/trace_divergence.py --log "$SMOKE" --out "D:/fsdownload/tw_smoke_${r}.json" \
        --seed 2021 --det 1 --epochs 1 --max-steps 286 --layers 0 --param-every 10 2>&1 \
    | grep -avE "^\[PDM\]|^\[CDP\]" | tee -a "$LOG"
done
"$PY" -u tools/trace_divergence.py --compare D:/fsdownload/tw_smoke_1.json D:/fsdownload/tw_smoke_2.json 2>&1 | tee -a "$LOG"

say "########## ROOTCAUSE ROUND-2 DONE ##########"
