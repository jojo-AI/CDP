#!/bin/bash
# =============================================================================
# Traffic-720 单独接管队列（batch_size 16 -> 8，防 OOM）
# -----------------------------------------------------------------------------
# 背景（2026-10-07 19:40，用户决策）：
#   Traffic 每 epoch 耗时随 pred_len 暴涨：h96/h192 ≈ 950s，h336 = 1750s；
#   外推 h720 单 epoch 约 3800s+（20 epoch ≈ 21h）。且 h336 训练时显存已占
#   12064/12288 MiB（98%），h720 输出头更大 -> OOM 风险实打实。
#   用户拍板：h720 改 batch_size=8、保持 20 epoch。
#
# 为什么不改 run_traffic_trim.sh / run_cdp_v4.sh：
#   2026-10-06 教训 —— 绝不在运行时 Write/Edit 任何 scripts/*.sh，
#   bash 按字节偏移读脚本，改写会导致 syntax error 并整个退出。
#   因此这里用「新文件名」独立接管，配方与 v5final 完全一致，只改 batch。
#
# 流程：
#   1) 等 h336 v5final 结果落盘（由 trim 队列写入）
#   2) 收掉 trim 队列 + 它可能刚拉起的 h720(batch16) 训练进程
#   3) 清掉 trim 的 stale 锁（kill -9 不触发它的 trap），重建锁
#   4) 直接调 run.py 跑 h720（batch_size 8，其余配方不变）
#   5) 追加结果到 v4_results.txt，刷新最优榜
#
# 用法: bash scripts/run_traffic_h720_b8.sh   （Bash 工具 run_in_background=true）
# =============================================================================
cd /d/fsdownload/TimeMixer || exit 1

LOG=/d/fsdownload/traffic_h720_b8.log
RES=/d/fsdownload/v4_results.txt
LOCK=/d/fsdownload/.remaining.lock
PY="/c/Users/Administrator/.workbuddy/binaries/python/envs/default/Scripts/python.exe"
PL=720
TAG=v5final_Traffic_${PL}
DS=Traffic

say () { echo "===== [$(date '+%F %T')] $* =====" | tee -a "$LOG"; }

say "TRAFFIC H720 (batch=8) TAKEOVER START (pid=$$)"

# ---------------------------------------------------------------------------
# 1) 等 h336 结果落盘（最多 6 小时）
# ---------------------------------------------------------------------------
for i in $(seq 1 1080); do
  if grep -aE "(^| )v5final_Traffic_336[^ ]* " "$RES" 2>/dev/null | grep -aq "mse:"; then
    say "h336 结果已落盘"
    break
  fi
  sleep 20
done

# ---------------------------------------------------------------------------
# 2) 收掉 trim 队列 + 残留训练进程
# ---------------------------------------------------------------------------
say "收掉 trim 队列与残留训练进程"
for p in $(ps -ef | grep -E "run_traffic_trim\.sh" | grep -v grep | awk '{print $2}'); do
  kill -9 "$p" 2>/dev/null && say "  killed trim msys=$p"
done
sleep 2
for p in $(ps -ef | grep -E "run_cdp_v4\.sh|run\.py" | grep -v grep | awk '{print $2}'); do
  w=$(cat /proc/$p/winpid 2>/dev/null)
  kill -9 "$p" 2>/dev/null
  if [ -n "$w" ]; then taskkill //F //PID "$w" >/dev/null 2>&1 && say "  killed msys=$p win=$w"; fi
done
sleep 3

# ---------------------------------------------------------------------------
# 3) 重建锁
# ---------------------------------------------------------------------------
OLD=$(cat "$LOCK/pid" 2>/dev/null)
if [ -n "$OLD" ] && kill -0 "$OLD" 2>/dev/null; then
  say "!! 锁仍被活进程 $OLD 持有，退出以免并发"
  exit 1
fi
rm -rf "$LOCK"; mkdir -p "$LOCK" || exit 1
echo $$ > "$LOCK/pid"
trap 'rm -rf "$LOCK"' EXIT
say "已接管锁（stale holder=${OLD:-none}）"

# ---------------------------------------------------------------------------
# 4) 幂等检查 + 骨干确认
# ---------------------------------------------------------------------------
if grep -aE "(^| )${TAG}[^ ]* " "$RES" 2>/dev/null | grep -aq "mse:"; then
  say "[skip] ${TAG} 已有结果，直接刷新榜单"
  "$PY" tools/select_best.py >> "$LOG" 2>&1
  exit 0
fi

CK=$(ls -d ./checkpoints/long_term_forecast_Traffic_96_${PL}_none_TimeMixer_* 2>/dev/null \
     | grep -viE "cdp|st1|v4|v31|_at|v5|cal" | head -1)
if [ -z "$CK" ]; then say "!! 无骨干 ckpt: Traffic h${PL}"; exit 1; fi

# 配方 = v5final（与 run_traffic_trim.sh 完全一致），唯一差异：--batch_size 8
ARGS="--task_name long_term_forecast --model TimeMixer_CDP --features M --seq_len 96 --label_len 0 \
--d_layers 1 --des CDPv4 --itr 1 --learning_rate 0.01 --num_workers 0 \
--down_sampling_layers 3 --down_sampling_method avg --down_sampling_window 2 \
--cdp_stage_w 1 --cdp_M 4 --cdp_S 16 --cdp_err_decay 0.9 --cdp_lambda_t 0.01 \
--train_epochs 20 --patience 12 \
--root_path ./dataset/traffic/ --data_path traffic.csv --data custom \
--enc_in 862 --dec_in 862 --c_out 862 --e_layers 3 --factor 3 --d_model 32 --d_ff 64 \
--batch_size 8 --pred_len ${PL} --model_id Traffic_v4_fin__${PL}_${TAG} \
--cdp_backbone_checkpoint ${CK} \
--cdp_learning_rate 5e-4 --cdp_loss mse_l1 --cdp_l1_mu 0.5 --cdp_monitor mae \
--cdp_unfreeze_head 1 --cdp_unfreeze_norm 1 --cdp_chan_rank 8 --cdp_chan_rank_mul 8 \
--backbone_mode dynamic --cdp_unfreeze_pdm 1 --cdp_st_lr_ratio 0.2 --cdp_dyncal 1 \
--cdp_loss l1 --cdp_monitor mae --cdp_cal_lr_ratio 5"

LOGF=/d/fsdownload/v4_Traffic_${PL}_fin__${TAG}.log
say "RUN ${TAG} (batch=8, ep=20 pat=12, backbone=$(basename "$CK"))"
"$PY" -u run.py --is_training 1 $ARGS > "$LOGF" 2>&1
"$PY" -u run.py --is_training 0 $ARGS >> "$LOGF" 2>&1

R=$(grep -a -oE "mse:[0-9.]+, mae:[0-9.]+" "$LOGF" | tail -1)
if [ -n "$R" ]; then
  echo "${DS} h${PL} ${TAG}_fin_  ${R}" >> "$RES"
  say "RESULT ${TAG}: ${R}"
else
  say "!! 无结果，请检查 ${LOGF}"
fi

# ---------------------------------------------------------------------------
# 5) 收尾
# ---------------------------------------------------------------------------
say "刷新最优榜"
"$PY" tools/select_best.py >> "$LOG" 2>&1
say "TRAFFIC H720 QUEUE DONE"
