#!/bin/bash
# =============================================================================
# 多种子补跑 · Headline 设定（2026-10-08 用户批准）
#
# 决策背景（见 D:/fsdownload/determinism_rootcause_report_20261008.md）：
#   同种子不可复现的根因 = 反向 kernel 稀有 CUDA 归约竞争（1 ULP），PyTorch 关不掉；
#   放大机制 = 「按 val 选 best epoch」的平台期平局 ⇒ 最终指标同种子散 MSE ~1.2% / MAE ~0.75%。
#   同种子散度与换种子效应同量级 ⇒ 多种子误差棒是唯一诚实做法。
#
# 协议（用户选择「同现有协议」）：
#   - det_mode 保持默认 0（与已报告的 28 设定完全同一协议）→ 现有 seed-2021 数字无需重跑
#   - 只追加 seed 2022 / 2023，共 2 个新种子
#   - 每个设定完整重跑两阶段：Stage-1（骨干重训）→ Stage-2（CDP 校准），两阶段都带 --fix_seed
#
# 覆盖的 3 个 headline 设定（配方从原始训练日志的 Namespace 逐字段反解，保证一致）：
#   1) ETTh1-720  winner = lt_cal5           （Stage-1: dynamic+l1, lr 0.01；Stage-2: lr 0.001）
#   2) ETTm1-96   winner = cdp_cal5_ETTm1_96 （Stage-1: dynamic+l1；Stage-2: chan_rank 8）
#   3) weather-96 winner = at_sw1            （Stage-1: dynamic+l1 + handtune_init 0；Stage-2: step_weight 1.0）
#
# ⚠️ 不包含 Traffic-96：它的 Stage-2 实测 926 s/epoch × 20 epoch = 5.1 h/种子，
#    2 个种子 = 10.2 h，超出 2–3 h 预算 → 单独立项等用户决策。
#
# 实测成本（3 设定 × 2 种子）：
#   ETTh1-720  Stage-1 12ep×3.1s  + Stage-2 20ep×14.7s ≈ 6.5 min/种子
#   ETTm1-96   Stage-1 13ep×11.3s + Stage-2 13ep×25.4s ≈ 8   min/种子
#   weather-96 Stage-1 20ep×30.2s + Stage-2 13ep×67.7s ≈ 25  min/种子
#   ⇒ 合计 ≈ 40 min/种子 ⇒ 2 个种子 ≈ 1.5–2 h（含评估与开销）
#
# 用法：bash scripts/run_multiseed.sh            （后台跑）
#      SEEDS_OVR="2022" bash scripts/run_multiseed.sh   （只跑某几个种子）
# 结果：D:/fsdownload/multiseed_results.txt   日志：D:/fsdownload/multiseed.log
# =============================================================================
PY="/c/Users/Administrator/.workbuddy/binaries/python/envs/default/Scripts/python.exe"
cd "/d/fsdownload/TimeMixer" || exit 1

RES="/d/fsdownload/multiseed_results.txt"
LOG="/d/fsdownload/multiseed.log"
LOCK="/d/fsdownload/.multiseed.lock"
SEEDS="${SEEDS_OVR:-2022 2023}"

say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

# ---- 并发守卫：与训练队列用不同的锁，避免互相抢 GPU ----
if ! mkdir "$LOCK" 2>/dev/null; then
  OLD=$(cat "$LOCK/pid" 2>/dev/null || echo '?')
  echo "!!! 已有多种子队列在跑（$LOCK, pid=$OLD）→ 退出"
  exit 1
fi
echo $$ > "$LOCK/pid"
cleanup() { rm -rf "$LOCK"; }
trap cleanup EXIT

# ---- 幂等：已完成的 (设定, 种子) 直接跳过 ----
# ⚠️ 必须用前缀匹配：结果行末尾带后缀（如 _s2022 ）。
done_tag() { grep -aqE "(^| )$1[^ ]* " "$RES" 2>/dev/null; }

# ---- 公共参数（与历史实验逐字段一致）----
ST1_COMMON="--task_name long_term_forecast --model TimeMixer --features M --seq_len 96 --label_len 0 \
  --d_layers 1 --des ST1 --itr 1 --learning_rate 0.01 --num_workers 0 \
  --train_loss l1 --train_l1_mu 0.0 \
  --down_sampling_layers 3 --down_sampling_method avg --down_sampling_window 2 \
  --train_epochs 30 --patience 8"

ST2_COMMON="--task_name long_term_forecast --model TimeMixer_CDP --features M --seq_len 96 --label_len 0 \
  --d_layers 1 --des CDPv4 --itr 1 --num_workers 0 \
  --down_sampling_layers 3 --down_sampling_method avg --down_sampling_window 2 \
  --cdp_stage_w 1 --cdp_M 4 --cdp_S 16 --cdp_err_decay 0.9 --cdp_lambda_t 0.01 \
  --train_epochs 20 --patience 12"

# ---- 逐设定：架构参数 / Stage-1 附加 / Stage-2 附加 ----
setting_args() {
  case "$1" in
    ETTh1-720)
      DSARGS="--root_path ./dataset/ETT-small/ --data_path ETTh1.csv --data ETTh1 \
        --enc_in 7 --dec_in 7 --c_out 7 --e_layers 2 --factor 1 --d_model 16 --d_ff 32 --batch_size 128 --pred_len 720"
      ST1_EXTRA="--backbone_mode dynamic"
      ST1_MID_BASE="ETTh1_st1_720_dynamic_l10.0"
      ST2_LR="0.001"
      ST2_EXTRA="--backbone_mode dynamic --cdp_learning_rate 5e-4 --cdp_loss l1 --cdp_monitor mae \
        --cdp_dyncal 1 --cdp_unfreeze_pdm 1 --cdp_st_lr_ratio 0.2 --cdp_cal_lr_ratio 5"
      ST2_TAG="lt_cal5"
      ;;
    ETTm1-96)
      DSARGS="--root_path ./dataset/ETT-small/ --data_path ETTm1.csv --data ETTm1 \
        --enc_in 7 --dec_in 7 --c_out 7 --e_layers 2 --factor 1 --d_model 16 --d_ff 32 --batch_size 128 --pred_len 96"
      ST1_EXTRA="--backbone_mode dynamic"
      ST1_MID_BASE="ETTm1_st1_96_dynamic_l10.0"
      ST2_LR="0.01"
      ST2_EXTRA="--backbone_mode dynamic --cdp_learning_rate 5e-4 --cdp_loss l1 --cdp_monitor mae \
        --cdp_unfreeze_head 1 --cdp_unfreeze_norm 1 --cdp_chan_rank 8 --cdp_chan_rank_mul 8 \
        --cdp_unfreeze_pdm 1 --cdp_st_lr_ratio 0.2 --cdp_dyncal 1 --cdp_cal_lr_ratio 5"
      ST2_TAG="cdp_cal5"
      ;;
    weather-96)
      DSARGS="--root_path ./dataset/weather/ --data_path weather.csv --data custom \
        --enc_in 21 --dec_in 21 --c_out 21 --e_layers 3 --factor 3 --d_model 16 --d_ff 32 --batch_size 128 --pred_len 96"
      ST1_EXTRA="--backbone_mode dynamic --handtune_init 0"
      ST1_MID_BASE="weather_st1_96_dyn0_l10.0"
      ST2_LR="0.01"
      ST2_EXTRA="--backbone_mode dynamic --cdp_learning_rate 5e-4 --cdp_loss l1 --cdp_monitor mae \
        --cdp_unfreeze_head 1 --cdp_unfreeze_norm 1 --cdp_chan_rank 8 --cdp_chan_rank_mul 8 \
        --cdp_unfreeze_pdm 1 --cdp_st_lr_ratio 0.2 --cdp_dyncal 1 --cdp_cal_lr_ratio 5 --cdp_step_weight 1.0"
      ST2_TAG="at_sw1"
      ;;
    *) echo "unknown setting $1"; exit 1 ;;
  esac
  DS_NAME="${1%-*}"; H="${1##*-}"
}

run_one() {   # $1 = setting, $2 = seed
  local SET="$1" SEED="$2"
  setting_args "$SET"
  local ST1_MID="${ST1_MID_BASE}_s${SEED}"
  local ST2_MID="${DS_NAME}_ms_${H}_${ST2_TAG}_s${SEED}"
  local T1="ms_${DS_NAME}_${H}_st1_s${SEED}" T2="ms_${ST2_TAG}_s${SEED}"

  if done_tag "$T2"; then
    say "跳过 $SET seed=$SEED（结果已存在）"
    return 0
  fi

  # ---------------- Stage-1：骨干重训 ----------------
  if ! done_tag "$T1" || [ ! -d "$(ls -d ./checkpoints/long_term_forecast_${ST1_MID}_* 2>/dev/null | head -1)" ]; then
    say "--- [$SET seed=$SEED] Stage-1 → $ST1_MID"
    "$PY" -u run.py --is_training 1 $ST1_COMMON $DSARGS $ST1_EXTRA \
        --model_id "$ST1_MID" --fix_seed "$SEED" > "/d/fsdownload/ms_${DS_NAME}_${H}_st1_s${SEED}.log" 2>&1
    # Stage-1 自身的评测（用于增益拆解，便宜）
    "$PY" -u run.py --is_training 0 $ST1_COMMON $DSARGS $ST1_EXTRA \
        --model_id "$ST1_MID" --fix_seed "$SEED" >> "/d/fsdownload/ms_${DS_NAME}_${H}_st1_s${SEED}.log" 2>&1
    local r1=$(grep -a -oE "mse:[0-9.]+, mae:[0-9.]+" "/d/fsdownload/ms_${DS_NAME}_${H}_st1_s${SEED}.log" | tail -1)
    echo "${DS_NAME} h${H} ${T1}  ${r1}" >> "$RES"
    say "    Stage-1 完成: ${r1}"
  fi

  local BASE=$(ls -d ./checkpoints/long_term_forecast_${ST1_MID}_* 2>/dev/null | grep -v CDP | head -1)
  if [ -z "$BASE" ]; then say "!!! [$SET seed=$SEED] 找不到 Stage-1 骨干 $ST1_MID → 跳过 Stage-2"; return 1; fi

  # ---------------- Stage-2：CDP 校准 ----------------
  say "--- [$SET seed=$SEED] Stage-2 → $ST2_MID  (backbone=$(basename "$BASE"))"
  local ML="/d/fsdownload/ms_${DS_NAME}_${H}_${ST2_TAG}_s${SEED}.log"
  "$PY" -u run.py --is_training 1 $ST2_COMMON --learning_rate "$ST2_LR" $DSARGS $ST2_EXTRA \
      --model_id "$ST2_MID" --cdp_backbone_checkpoint "$BASE" --fix_seed "$SEED" > "$ML" 2>&1
  "$PY" -u run.py --is_training 0 $ST2_COMMON --learning_rate "$ST2_LR" $DSARGS $ST2_EXTRA \
      --model_id "$ST2_MID" --cdp_backbone_checkpoint "$BASE" --fix_seed "$SEED" >> "$ML" 2>&1
  local r2=$(grep -a -oE "mse:[0-9.]+, mae:[0-9.]+" "$ML" | tail -1)
  echo "${DS_NAME} h${H} ${T2}  ${r2}" >> "$RES"
  say "    Stage-2 完成: ${r2}"
}

say "########## MULTISEED HEADLINE START (pid=$$) seeds='$SEEDS' ##########"
touch "$RES"
for SEED in $SEEDS; do
  for SET in ETTh1-720 ETTm1-96 weather-96; do
    run_one "$SET" "$SEED" || say "!!! $SET seed=$SEED 失败，继续下一个"
  done
done
say "########## MULTISEED HEADLINE DONE ##########"
say "结果文件 $(wc -l < "$RES") 行:"
tail -20 "$RES" | while read -r l; do say "  $l"; done
