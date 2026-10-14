#!/bin/bash
# =============================================================================
# 续跑「剩余缺口」队列（严格串行、幂等、带锁）
# -----------------------------------------------------------------------------
# 背景：run_serial_all.sh 的 Phase 1-6（ETT 四数据集 + weather）已全部完成，
#       队列在 2026-09-29 08:08 跑 Phase-7 ECL h96 at_sw1 时被杀，此后 GPU 空转。
#       本脚本只补跑 Phase 7 / Phase 8，不做任何 ETT 段工作。
#
#   Phase A) ECL     96 192 336 720   （含 Stage-1 骨干重训）
#   Phase B) Traffic 96 192 336 720   （SKIP_ST1=1，仅 CDP 校准，用作者 clean 骨干）
#   Phase C) 自动刷新最优榜 tools/select_best.py -> D:/fsdownload/leaderboard.md
#
# 幂等：run_autotune.sh 内部按结果文件跳过已完成项，可随时中断重跑。
# 锁  ：/d/fsdownload/.remaining.lock（trap EXIT 释放），防止重复拉起。
#
# 用法: bash scripts/run_remaining.sh   （务必用 run_in_background=true 启动）
# =============================================================================
LOCK=/d/fsdownload/.remaining.lock

# 持锁者是否真的还活着？（SIGKILL / 会话被终止时 EXIT trap 不会触发，会留下死锁）
holder_alive () {            # $1 = 记录的 pid
  [ -n "$1" ] || return 1
  kill -0 "$1" 2>/dev/null || return 1
  ps -ef | awk -v P="$1" '$2==P' | grep -q "run_remaining"
}

if ! mkdir "$LOCK" 2>/dev/null; then
  OLD=$(cat "$LOCK/pid" 2>/dev/null)
  if holder_alive "$OLD"; then
    echo "another run_remaining.sh (pid=$OLD) already holds $LOCK -> exit"
    exit 0
  fi
  echo "stale lock ($LOCK, dead pid=${OLD:-unknown}) -> removing and taking over"
  rm -rf "$LOCK"
  mkdir "$LOCK" 2>/dev/null || { echo "lock race -> exit"; exit 0; }
fi
trap 'rm -rf "$LOCK"' EXIT
echo $$ > "$LOCK/pid"

cd /d/fsdownload/TimeMixer || exit 1
LOG=/d/fsdownload/remaining.log
PY="/c/Users/Administrator/.workbuddy/binaries/python/envs/default/Scripts/python.exe"

say () { echo "===== [$(date '+%F %T')] $* =====" | tee -a "$LOG"; }
say "REMAINING QUEUE START (pid=$$)"

# ---- 0) 等残留训练进程退出（最多 10 分钟），避免抢 GPU ----
for i in $(seq 1 60); do
  if ! ps -ef | grep -E "run\.py" | grep -v grep > /dev/null 2>&1; then
    say "GPU 空闲，开始续跑队列"
    break
  fi
  sleep 10
done

# ---- A) ECL 4 档（Phase 7）----
say "PHASE A :: ECL 96 192 336 720"
PHASE=1 BUDGET=lean bash scripts/run_autotune.sh ECL 96 192 336 720 >> "$LOG" 2>&1 \
  || say "FAILED :: ECL"

# ---- B) Traffic 4 档（Phase 8，跳过 Stage-1）----
say "PHASE B :: Traffic 96 192 336 720"
SKIP_ST1=1 BUDGET=lean PHASE=1 bash scripts/run_autotune.sh Traffic 96 192 336 720 >> "$LOG" 2>&1 \
  || say "FAILED :: Traffic"

# ---- C) 收尾：刷新最优榜 ----
say "刷新最优榜"
"$PY" tools/select_best.py >> "$LOG" 2>&1
say "REMAINING QUEUE DONE"
