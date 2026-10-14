#!/bin/bash
# =============================================================================
# 多种子补跑 · Traffic-96 扩展（2026-10-08 用户追加批准：补 1 个种子 = seed 2022）
#
# 为什么单独一个脚本：`scripts/run_multiseed.sh` 正在跑，**绝不能在运行期间改它**
# （bash 按字节偏移读脚本，改写会在"下一条命令"处 syntax error 并整个退出 —— 已踩过）。
#
# Traffic-96 的特殊性：
#   - 骨干用**作者发布的冻结 checkpoint**（clean），没有 Stage-1 → 种子只影响 Stage-2
#   - Stage-2 实测 926 s/epoch × 20 epoch ≈ 5.1 h
#   - batch_size = 16（只有 h720 那一档用 8）
#
# 并发策略：复用同一个锁 `/d/fsdownload/.multiseed.lock`，
#   先等前一个队列自然结束再独占 GPU（历史实测：4 队列并发会让单 epoch 从 38s 涨到 220s，约 5.7×）。
#
# 用法：bash scripts/run_multiseed_traffic96.sh    （后台跑，约 5.5 h）
# =============================================================================
PY="/c/Users/Administrator/.workbuddy/binaries/python/envs/default/Scripts/python.exe"
cd "/d/fsdownload/TimeMixer" || exit 1

RES="/d/fsdownload/multiseed_results.txt"
LOG="/d/fsdownload/multiseed_traffic96.log"
LOCK="/d/fsdownload/.multiseed.lock"
SEED="${SEED_OVR:-2022}"

say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
say "########## MULTISEED TRAFFIC-96 START (pid=$$) seed=$SEED ##########"

# ---- 优先级闸门（2026-10-08 新增）-----------------------------------------
# 让「便宜的补种子实验」(scripts/run_multiseed_ext.sh) 先跑完，再占用 GPU 5.15 h/种子。
# 闸门 = /d/fsdownload/.ext_priority（目录，内含 pid）。自带存活检测：pid 已消失 → 自动清闸。
SENT="/d/fsdownload/.ext_priority"
for i in $(seq 1 1800); do            # 最多等 5 小时，避免把 Traffic 永久卡住
  [ ! -e "$SENT" ] && break
  EPID=$(cat "$SENT/pid" 2>/dev/null)
  if [ -n "$EPID" ] && ! kill -0 "$EPID" 2>/dev/null; then
    say "ext 队列（pid=$EPID）已结束 → 清理优先级闸门"
    rm -rf "$SENT"; break
  fi
  [ $((i % 6)) -eq 1 ] && say "等待高优先级 ext 补种子队列（pid=${EPID:-?}）…"
  sleep 10
done

# ---- 排队等 GPU：直到拿到锁 ----
acquired=0
for i in $(seq 1 720); do          # 最多等 6 小时（30s × 720）
  if mkdir "$LOCK" 2>/dev/null; then
    echo $$ > "$LOCK/pid"; acquired=1; break
  fi
  [ $((i % 10)) -eq 1 ] && say "等待前一个队列释放 GPU（$LOCK 存在，pid=$(cat "$LOCK/pid" 2>/dev/null))…"
  sleep 30
done
if [ "$acquired" != "1" ]; then say "!!! 等待超时，退出"; exit 1; fi
cleanup() { rm -rf "$LOCK"; }
trap cleanup EXIT

done_tag() { grep -aqE "(^| )$1[^ ]* " "$RES" 2>/dev/null; }
touch "$RES"

BASE="./checkpoints/long_term_forecast_Traffic_96_96_none_TimeMixer_custom_sl96_pl96_dm32_nh8_el3_dl1_df64_fc3_ebtimeF_dtTrue_Exp_0"
if [ ! -d "$BASE" ]; then say "!!! 找不到作者冻结骨干 $BASE"; exit 1; fi

DSARGS="--root_path ./dataset/traffic/ --data_path traffic.csv --data custom \
  --enc_in 862 --dec_in 862 --c_out 862 --e_layers 3 --factor 3 --d_model 32 --d_ff 64 \
  --batch_size 16 --pred_len 96"

ST2_COMMON="--task_name long_term_forecast --model TimeMixer_CDP --features M --seq_len 96 --label_len 0 \
  --d_layers 1 --des CDPv4 --itr 1 --learning_rate 0.01 --num_workers 0 \
  --down_sampling_layers 3 --down_sampling_method avg --down_sampling_window 2 \
  --cdp_stage_w 1 --cdp_M 4 --cdp_S 16 --cdp_err_decay 0.9 --cdp_lambda_t 0.01 \
  --train_epochs 20 --patience 12"

# 与 winner at_sw1_Traffic_96_ats1_ 逐字段一致（从原始日志 Namespace 反解）
ST2_EXTRA="--backbone_mode dynamic --cdp_learning_rate 5e-4 --cdp_loss l1 --cdp_monitor mae \
  --cdp_unfreeze_head 1 --cdp_unfreeze_norm 1 --cdp_chan_rank 8 --cdp_chan_rank_mul 8 \
  --cdp_unfreeze_pdm 1 --cdp_st_lr_ratio 0.2 --cdp_dyncal 1 --cdp_cal_lr_ratio 5 --cdp_step_weight 1.0"

MID="Traffic_ms_96_at_sw1_s${SEED}"
# 🔴 2026-10-08 修复：TAG 原为 `ms_at_sw1_s${SEED}`，与 weather-96 在 run_multiseed.sh 里
#   写下的同名 tag 完全撞车 → done_tag 的前缀匹配命中了 weather 的结果行，
#   导致 Traffic-96 seed2022 **被误判为"已有结果"而跳过**（28/28 里这一档从未跑过）。
#   改为带数据集前缀的全局唯一 tag，杜绝跨数据集撞名。
TAG="Traffic_ms_96_at_sw1_s${SEED}"
ML="/d/fsdownload/ms_Traffic_96_at_sw1_s${SEED}.log"

if done_tag "$TAG"; then say "跳过：$TAG 已有结果"; exit 0; fi

say "--- [Traffic-96 seed=$SEED] Stage-2 → $MID (backbone=作者冻结 clean)"
"$PY" -u run.py --is_training 1 $ST2_COMMON $DSARGS $ST2_EXTRA \
    --model_id "$MID" --cdp_backbone_checkpoint "$BASE" --fix_seed "$SEED" > "$ML" 2>&1
"$PY" -u run.py --is_training 0 $ST2_COMMON $DSARGS $ST2_EXTRA \
    --model_id "$MID" --cdp_backbone_checkpoint "$BASE" --fix_seed "$SEED" >> "$ML" 2>&1
r=$(grep -a -oE "mse:[0-9.]+, mae:[0-9.]+" "$ML" | tail -1)
echo "Traffic h96 ${TAG}  ${r}" >> "$RES"
say "    Stage-2 完成: ${r}"
say "########## MULTISEED TRAFFIC-96 DONE ##########"
