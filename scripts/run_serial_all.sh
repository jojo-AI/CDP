#!/bin/bash
# =============================================================================
# 严格串行总队列（一次只跑一个训练进程）
# -----------------------------------------------------------------------------
# 背景：并发 4 条队列时 GPU 98% 打满、单 epoch 从 35s 涨到 220s（6×），
#       总吞吐反而下降。本脚本把全部剩余工作串成一条线，逐个跑完再跑下一个。
#
# 特性：
#   * 幂等：所有 stage 都按结果文件跳过已完成项，可随时中断 / 重跑。
#   * 单点失败不影响后续：每个 phase 用 `|| say FAILED` 包住。
#   * 顺序 = 先"补齐覆盖（每个数据集至少有一份新标准结果）"，再"加深度调优"。
#
# 顺序：
#   1) weather      720               （接续被中断的新标准 Stage-2）
#   2) ETTh2        96 192 336 720
#   3) ETTm1        96 192 336 720
#   4) ETTm2        96 192 336 720
#   5) weather      96 192 336        （新标准 Phase-1）
#   6) ETTh1        Phase-2 补 (96/192/336)
#   7) ECL          96 192 336 720    （含 Stage-1 骨干重训，最贵之一）
#   8) Traffic      96 192 336 720    （SKIP_ST1，只做 CDP 校准）
#
# 用法: bash scripts/run_serial_all.sh    （务必用 run_in_background 启动）
# =============================================================================
cd /d/fsdownload/TimeMixer || exit 1

# ---- 防撞车（2026-10-05 补）------------------------------------------------
# 本脚本无锁，历史上曾与 run_remaining.sh 并发启动 → 两个队列抢同一张 GPU
# （单 epoch 从 38s 涨到 220s）。现改为：只要 run_remaining 的锁还在，一律退出。
# 剩余工作请统一用 scripts/run_remaining.sh 续跑。
if [ -d /d/fsdownload/.remaining.lock ]; then
  echo "!! /d/fsdownload/.remaining.lock 存在：run_remaining.sh 队列持有中（或残留死锁）。"
  echo "!! 为避免两个队列并发抢 GPU，run_serial_all.sh 退出。"
  echo "!! 请改用 bash scripts/run_remaining.sh ；仅在确认队列已死且锁残留时才手动 rm -rf 该锁。"
  exit 0
fi
LOG=/d/fsdownload/serial_all.log
PY="/c/Users/Administrator/.workbuddy/binaries/python/envs/default/Scripts/python.exe"

say () { echo "===== [$(date '+%F %T')] $* =====" | tee -a "$LOG"; }

# 让 tee 不缓冲，日志能实时看到进度
say "SERIAL QUEUE START (pid=$$)"

# ---- 0) 等残留训练进程退出（最多 5 分钟），避免抢 GPU ----
for i in $(seq 1 30); do
  if ! ps -ef | grep -E "run\.py|run_cdp_v4\.sh|run_autotune\.sh|run_ett_extra\.sh|run_master_queue\.sh" \
       | grep -v grep > /dev/null 2>&1; then
    say "GPU 空闲，开始串行队列"
    break
  fi
  sleep 10
done

# ---- 1) weather 720：补 Stage-1 hi0 + Phase-1 候选（v5final / at_cal1 / at_sw1）----
say "PHASE 1 :: weather 720"
PHASE=1 bash scripts/run_autotune.sh weather 720 >> "$LOG" 2>&1 \
  || say "FAILED :: weather 720"

# ---- 2) ETTh2 全 4 档（96/192 只补缺的 cal1，336/720 走完整流水线）----
say "PHASE 2 :: ETTh2 96 192 336 720"
bash scripts/run_ett_extra.sh ETTh2 96 192 336 720 >> "$LOG" 2>&1 \
  || say "FAILED :: ETTh2"

# ---- 3) ETTm1 全 4 档（当前 0/4，无任何结果）----
say "PHASE 3 :: ETTm1 96 192 336 720"
bash scripts/run_ett_extra.sh ETTm1 96 192 336 720 >> "$LOG" 2>&1 \
  || say "FAILED :: ETTm1"

# ---- 4) ETTm2 全 4 档（当前 0/4）----
say "PHASE 4 :: ETTm2 96 192 336 720"
bash scripts/run_ett_extra.sh ETTm2 96 192 336 720 >> "$LOG" 2>&1 \
  || say "FAILED :: ETTm2"

# ---- 5) weather 96/192/336：补新标准 Stage-2（当前只有旧协议 cdp_on_st1）----
say "PHASE 5 :: weather 96 192 336"
PHASE=1 bash scripts/run_autotune.sh weather 96 192 336 >> "$LOG" 2>&1 \
  || say "FAILED :: weather 96/192/336"

# ---- 6) ETTh1 Phase-2 补跑（96/192/336 各补 at_sw2/at_st05/at_ep30/at_ep30c1）----
#        720 已双破 5%，PHASE=2 会自动跳过（FORCE=0）
say "PHASE 6 :: ETTh1 Phase-2 (96 192 336 720)"
PHASE=2 bash scripts/run_autotune.sh ETTh1 96 192 336 720 >> "$LOG" 2>&1 \
  || say "FAILED :: ETTh1 Phase-2"

# ---- 7) ECL：321 通道，含 Stage-1 骨干重训 ----
say "PHASE 7 :: ECL 96 192 336 720"
PHASE=1 BUDGET=lean bash scripts/run_autotune.sh ECL 96 192 336 720 >> "$LOG" 2>&1 \
  || say "FAILED :: ECL"

# ---- 8) Traffic：862 通道，跳过 Stage-1，只做 CDP 校准 ----
say "PHASE 8 :: Traffic 96 192 336 720"
SKIP_ST1=1 BUDGET=lean PHASE=1 bash scripts/run_autotune.sh Traffic 96 192 336 720 >> "$LOG" 2>&1 \
  || say "FAILED :: Traffic"

# ---- 收尾：刷新最优榜 ----
say "刷新最优榜"
"$PY" tools/select_best.py >> "$LOG" 2>&1
say "SERIAL QUEUE DONE"
