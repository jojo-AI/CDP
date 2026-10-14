#!/bin/bash
# =============================================================================
# SWA 实验收尾：补 Stage-2 的逐 epoch 存点 + 集成评估
#
# 背景：run_swa_etth1_720.sh 跑完后发现 Stage-2（model=TimeMixer_CDP）**没有**存逐 epoch
#   checkpoint —— 因为 `--save_all_epochs` 当时只加在 exp_long_term_forecasting.py（骨干路径），
#   CDP 走的是 exp/exp_long_term_forecasting_cdp.py。已在后者补上同样的开关。
#   本脚本用已有的 Stage-1 目录重跑 3 个种子的 Stage-2（每个约 2 min）+ 集成评估，然后交棒 Traffic。
#
# 用法：bash scripts/run_swa_st2_ens.sh     （约 10 min）
# =============================================================================
PY="/c/Users/Administrator/.workbuddy/binaries/python/envs/default/Scripts/python.exe"
cd "/d/fsdownload/TimeMixer" || exit 1

LOG="/d/fsdownload/swa_etth1_720.log"
LOCK="/d/fsdownload/.multiseed.lock"
SEEDS="${SEEDS_OVR:-2021 2022 2023}"
EPS="${EPS_OVR:-0.01}"

say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

# ---- 等锁 ----
acquired=0
for i in $(seq 1 60); do
  if mkdir "$LOCK" 2>/dev/null; then echo $$ > "$LOCK/pid"; acquired=1; break; fi
  say "等待 GPU 队列释放（$LOCK，pid=$(cat "$LOCK/pid" 2>/dev/null))…"
  sleep 30
done
if [ "$acquired" -ne 1 ]; then say "!! 等锁超时，退出"; exit 1; fi
cleanup() { rm -rf "$LOCK"; }
trap cleanup EXIT

ST2_COMMON="--task_name long_term_forecast --model TimeMixer_CDP --features M --seq_len 96 --label_len 0 \
  --d_layers 1 --des CDPv4 --itr 1 --num_workers 0 \
  --down_sampling_layers 3 --down_sampling_method avg --down_sampling_window 2 \
  --cdp_stage_w 1 --cdp_M 4 --cdp_S 16 --cdp_err_decay 0.9 --cdp_lambda_t 0.01 \
  --train_epochs 20 --patience 12"
DSARGS="--root_path ./dataset/ETT-small/ --data_path ETTh1.csv --data ETTh1 \
  --enc_in 7 --dec_in 7 --c_out 7 --e_layers 2 --factor 1 --d_model 16 --d_ff 32 --batch_size 128 --pred_len 720"
ST2_EXTRA="--backbone_mode dynamic --cdp_learning_rate 5e-4 --cdp_loss l1 --cdp_monitor mae \
  --cdp_dyncal 1 --cdp_unfreeze_pdm 1 --cdp_st_lr_ratio 0.2 --cdp_cal_lr_ratio 5"

say "########## SWA Stage-2 集成补跑 START (pid=$$) seeds='$SEEDS' ##########"

for SEED in $SEEDS; do
  M1="ETTh1_st1_720_swa_s${SEED}"
  M2="ETTh1_swa_720_lt_cal5_s${SEED}"
  L2="/d/fsdownload/swa_st2_s${SEED}.log"

  D1=$(ls -dt ./checkpoints/*"$M1"* 2>/dev/null | head -1)
  if [ -z "$D1" ]; then say "!!! seed=$SEED 找不到 Stage-1 目录 → 跳过"; continue; fi

  say "--- [seed=$SEED] Stage-2 训练（--save_all_epochs 1）→ $M2"
  "$PY" -u run.py --is_training 1 $ST2_COMMON --learning_rate 0.001 $DSARGS $ST2_EXTRA \
      --model_id "$M2" --cdp_backbone_checkpoint "$D1" --fix_seed "$SEED" --save_all_epochs 1 > "$L2" 2>&1
  D2=$(ls -dt ./checkpoints/*"$M2"* 2>/dev/null | head -1)
  if [ -z "$D2" ]; then say "!!! seed=$SEED Stage-2 找不到目录 → 跳过"; continue; fi
  n_ep=$(ls "$D2"/epoch_*.pth 2>/dev/null | wc -l)
  say "    Stage-2 目录 $D2（epoch_*.pth = $n_ep）"

  say "--- [seed=$SEED] Stage-2 集成评估"
  "$PY" -u tools/eval_ckpt_ensemble.py --log "$L2" --ckpt-dir "$D2" --eps "$EPS" \
      --out "/d/fsdownload/swa_ens_st2_s${SEED}.json" >> "$LOG" 2>&1 || say "!!! Stage-2 评估失败"
done

say "--- 重新汇总 ---"
"$PY" -u tools/swa_summary.py >> "$LOG" 2>&1 || say "!!! 汇总失败"
say "########## SWA Stage-2 集成补跑 DONE ##########"

# ---- 交棒给 Traffic-96（先释放锁）----
trap - EXIT
rm -rf "$LOCK"
say "已释放 GPU 锁，交棒给 Traffic-96 队列（seed 2022，约 5.15 h）"
bash scripts/run_multiseed_traffic96.sh >> /d/fsdownload/multiseed_traffic96.log 2>&1
say "Traffic-96 队列结束"
