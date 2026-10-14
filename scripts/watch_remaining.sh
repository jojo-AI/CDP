#!/bin/bash
# =============================================================================
# run_remaining.sh 看门狗（自动重启队列）
# -----------------------------------------------------------------------------
# 背景（2026-10-05）：单次 automation 会话结束后，其后台进程会被连带终止，
#   导致 run_remaining.sh 反复被杀。而每个候选（如 ECL h96 at_sw1 ≈ 2h）
#   比会话间隔还长 → 每次都从 epoch 1 重训，零净进展（今日已重训 4 次）。
#   本看门狗每 60s 检查一次：若 run_remaining.sh 已不在运行，立刻把它拉回来。
#
# 安全性：run_remaining.sh 自带 .remaining.lock（且能自动清理死锁），
#         因此重复拉起绝不会造成两个队列并发抢 GPU。
#
# 启动：Bash 工具 run_in_background=true  ->  bash scripts/watch_remaining.sh
# 退出：remaining.log 出现 "REMAINING QUEUE DONE" 时自动退出；或最多循环 2000 次。
# =============================================================================
cd /d/fsdownload/TimeMixer || exit 1
LOG=/d/fsdownload/watch_remaining.log

echo "===== [$(date '+%F %T')] WATCHDOG START (pid=$$) =====" >> "$LOG"

queue_running () {
  ps -ef | grep -E "bash scripts/run_remaining\.sh" | grep -qv grep
}

for i in $(seq 1 2000); do
  if grep -aq "REMAINING QUEUE DONE" /d/fsdownload/remaining.log; then
    echo "[$(date '+%T')] remaining.log 出现 DONE -> watchdog 退出" >> "$LOG"
    break
  fi
  if queue_running; then
    sleep 60
    continue
  fi
  echo "[$(date '+%F %T')] 队列不在运行 -> 重新拉起 run_remaining.sh" >> "$LOG"
  bash scripts/run_remaining.sh >> /d/fsdownload/remaining.log 2>&1 &
  sleep 90
done
