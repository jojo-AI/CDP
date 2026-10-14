#!/bin/bash
# =============================================================================
# 多种子补跑 · 扩展（seeds 2024 / 2025 / 2026）
# 触发：2026-10-08 用户要求「随机种子全部都跑」
#
# 为什么另建文件而不改 run_multiseed.sh：
#   **run_multiseed.sh 正在运行**；bash 按字节偏移逐条读脚本，运行中改写会让偏移错乱，
#   在「下一条命令」处 syntax error 并整个退出（本项目已踩过两次）。
#
# 协议：与 run_multiseed.sh **逐字段完全一致**（配方原样复制），
#       所以新种子能与已有的 2021 / 2022 / 2023 直接并列成 mean ± sd。
#       ⇒ 最终每个设定 n = 6（2021–2026 连续），**全部种子都会报告，不做挑选**。
#
# 优先级闸门（新增）：
#   本脚本创建 /d/fsdownload/.ext_priority（内含 pid）。
#   scripts/run_multiseed_traffic96.sh 在抢 GPU 锁之前会先等这个闸门，
#   从而保证「便宜的补种子实验」先于「5.15 h/种子的 Traffic-96」跑完。
#   EXIT 时：若 /d/fsdownload/.ext_more 存在 → 保留闸门（还有后续 ext 队列要跑）；
#            否则释放闸门，Traffic 自动接手。
#   闸门自带存活检测：pid 不在了就自动清理，不会把 Traffic 永久卡住。
#
# 实测成本（单种子，含两阶段）：ETTh1-720 ≈ 3.4 min｜ETTm1-96 ≈ 15.5 min｜weather-96 ≈ 27 min
#   ⇒ 3 设定 × 3 种子 ≈ 2.1 h
#
# 用法：
#   bash scripts/run_multiseed_ext.sh                          # 默认：3 招牌设定 × 2024/2025/2026
#   SETS_OVR="ETTh1-720" bash scripts/run_multiseed_ext.sh      # 只跑某个设定
#   SEEDS_OVR="2024"     bash scripts/run_multiseed_ext.sh      # 只跑某个种子
#   SETS_OVR="ETTh1-720 ETTm1-96 weather-96 Traffic-96" ...     # 含 Traffic（会极慢）
# 结果：D:/fsdownload/multiseed_results.txt（与既有种子同一文件）
# 日志：D:/fsdownload/multiseed_ext.log
# =============================================================================
PY="/c/Users/Administrator/.workbuddy/binaries/python/envs/default/Scripts/python.exe"
cd "/d/fsdownload/TimeMixer" || exit 1

RES="/d/fsdownload/multiseed_results.txt"
LOG="/d/fsdownload/multiseed_ext.log"
LOCK="/d/fsdownload/.multiseed.lock"
SENT="/d/fsdownload/.ext_priority"
MORE="/d/fsdownload/.ext_more"

SEEDS="${SEEDS_OVR:-2024 2025 2026}"
SETS="${SETS_OVR:-ETTh1-720 ETTm1-96 weather-96}"

say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
say "########## MULTISEED EXT START (pid=$$) seeds='$SEEDS' sets='$SETS' ##########"

# ---- 优先级闸门：先立牌，保证排在 Traffic 前面 ----
mkdir -p "$SENT"; echo $$ > "$SENT/pid"

# ---- 让位窗口（2026-10-08 加）---------------------------------------------
# scripts/run_swa_etth1_720.sh（SWA 平台期集成，只要 ~15 min）与本体都以 30 s 轮询同一把锁，
# 会真的互相抢。SWA 的结论（ETTh1-720 的不可复现能否靠改选优规则修掉）优先于补种子，
# 所以这里主动让位 YIELD_SECS（默认 900 s = 15 min，足够 SWA 拿到锁）。
# 之后正常竞争；SWA 持有锁期间本队列自然排队，不会抢到。
YIELD_SECS="${YIELD_SECS:-900}"
YIELD_UNTIL=$(( $(date +%s) + YIELD_SECS ))
say "让位窗口：先让 SWA 实验取锁，${YIELD_SECS}s 内不竞争（至 $(date -d "@$YIELD_UNTIL" '+%T')）"
while [ "$(date +%s)" -lt "$YIELD_UNTIL" ]; do sleep 20; done
say "让位窗口结束，开始竞争 GPU 锁"

# ---- 排队等 GPU 锁（最多等 6 h；不直接退出，与 traffic 队列行为一致）----
acquired=0
for i in $(seq 1 720); do
  if mkdir "$LOCK" 2>/dev/null; then
    echo $$ > "$LOCK/pid"; acquired=1; break
  fi
  [ $((i % 10)) -eq 1 ] && say "等待 GPU 队列释放（$LOCK，持有者 pid=$(cat "$LOCK/pid" 2>/dev/null))…"
  sleep 30
done
if [ "$acquired" != "1" ]; then say "!!! 等待 GPU 锁超时，退出"; exit 1; fi
say "已取得 GPU 锁（pid=$$）"

cleanup() {
  rm -rf "$LOCK"
  if [ -e "$MORE" ]; then
    say "仍有后续 ext 队列（$MORE 存在）→ 保留优先级闸门"
  else
    rm -rf "$SENT"
    say "已释放优先级闸门 $SENT（Traffic 队列可接手）"
  fi
}
trap cleanup EXIT

# ---- 幂等：已完成的 (设定, 种子) 直接跳过 ----
# ⚠️ 必须前缀匹配：结果行末尾带后缀（如 _s2022）
done_tag() { grep -aqE "(^| )$1[^ ]* " "$RES" 2>/dev/null; }

# ---- 公共参数（与 run_multiseed.sh 逐字段一致）----
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
    Traffic-96)
      # ⚠️ 骨干用作者发布的冻结 clean checkpoint（无 Stage-1）→ 种子只影响 Stage-2
      DSARGS="--root_path ./dataset/traffic/ --data_path traffic.csv --data custom \
        --enc_in 862 --dec_in 862 --c_out 862 --e_layers 3 --factor 3 --d_model 32 --d_ff 64 --batch_size 16 --pred_len 96"
      ST1_EXTRA=""
      ST1_MID_BASE=""
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
  local T2="ms_${ST2_TAG}_s${SEED}"

  if done_tag "$T2"; then
    say "跳过 $SET seed=$SEED（结果已存在）"
    return 0
  fi

  # ---------------- Stage-1：骨干重训（Traffic 无 Stage-1）----------------
  local BASE=""
  if [ "$SET" = "Traffic-96" ]; then
    BASE="./checkpoints/long_term_forecast_Traffic_96_96_none_TimeMixer_custom_sl96_pl96_dm32_nh8_el3_dl1_df64_fc3_ebtimeF_dtTrue_Exp_0"
    if [ ! -d "$BASE" ]; then say "!!! 找不到作者冻结骨干 $BASE → 跳过"; return 1; fi
  else
    local ST1_MID="${ST1_MID_BASE}_s${SEED}"
    local T1="ms_${DS_NAME}_${H}_st1_s${SEED}"
    if ! done_tag "$T1" || [ ! -d "$(ls -d ./checkpoints/long_term_forecast_${ST1_MID}_* 2>/dev/null | head -1)" ]; then
      say "--- [$SET seed=$SEED] Stage-1 → $ST1_MID"
      "$PY" -u run.py --is_training 1 $ST1_COMMON $DSARGS $ST1_EXTRA \
          --model_id "$ST1_MID" --fix_seed "$SEED" > "/d/fsdownload/ms_${DS_NAME}_${H}_st1_s${SEED}.log" 2>&1
      "$PY" -u run.py --is_training 0 $ST1_COMMON $DSARGS $ST1_EXTRA \
          --model_id "$ST1_MID" --fix_seed "$SEED" >> "/d/fsdownload/ms_${DS_NAME}_${H}_st1_s${SEED}.log" 2>&1
      local r1=$(grep -a -oE "mse:[0-9.]+, mae:[0-9.]+" "/d/fsdownload/ms_${DS_NAME}_${H}_st1_s${SEED}.log" | tail -1)
      echo "${DS_NAME} h${H} ${T1}  ${r1}" >> "$RES"
      say "    Stage-1 完成: ${r1}"
    fi
    BASE=$(ls -d ./checkpoints/long_term_forecast_${ST1_MID}_* 2>/dev/null | grep -v CDP | head -1)
    if [ -z "$BASE" ]; then say "!!! [$SET seed=$SEED] 找不到 Stage-1 骨干 $ST1_MID → 跳过 Stage-2"; return 1; fi
  fi

  # ---------------- Stage-2：CDP 校准 ----------------
  local ST2_MID="${DS_NAME}_ms_${H}_${ST2_TAG}_s${SEED}"
  local ML="/d/fsdownload/ms_${DS_NAME}_${H}_${ST2_TAG}_s${SEED}.log"
  say "--- [$SET seed=$SEED] Stage-2 → $ST2_MID  (backbone=$(basename "$BASE"))"
  "$PY" -u run.py --is_training 1 $ST2_COMMON --learning_rate "$ST2_LR" $DSARGS $ST2_EXTRA \
      --model_id "$ST2_MID" --cdp_backbone_checkpoint "$BASE" --fix_seed "$SEED" > "$ML" 2>&1
  "$PY" -u run.py --is_training 0 $ST2_COMMON --learning_rate "$ST2_LR" $DSARGS $ST2_EXTRA \
      --model_id "$ST2_MID" --cdp_backbone_checkpoint "$BASE" --fix_seed "$SEED" >> "$ML" 2>&1
  local r2=$(grep -a -oE "mse:[0-9.]+, mae:[0-9.]+" "$ML" | tail -1)
  echo "${DS_NAME} h${H} ${T2}  ${r2}" >> "$RES"
  say "    Stage-2 完成: ${r2}"
}

touch "$RES"
for SEED in $SEEDS; do
  for SET in $SETS; do
    run_one "$SET" "$SEED" || say "!!! $SET seed=$SEED 失败，继续下一个"
  done
done

say "########## MULTISEED EXT DONE ##########"
say "--- 重新汇总 mean±sd ---"
"$PY" -u tools/multiseed_summary.py >> "$LOG" 2>&1 || say "!!! 汇总失败（见日志）"
say "结果文件 $(wc -l < "$RES") 行"
