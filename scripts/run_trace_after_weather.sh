#!/bin/bash
# =============================================================================
# 等待 weather 确定性补验结束后，自动执行「逐 step 分叉定位」并输出比对结论。
#
# 为什么要链式：确定性测试必须独占 GPU（并发会污染结果），
#   所以 trace 必须等 weather 冒烟的两个 det 批次全部跑完才能开始。
#
# 产出：/d/fsdownload/trace_result.log（含归因结论）
#       /d/fsdownload/trace_run{1,2}.json（逐步哈希原始记录）
# =============================================================================
cd /d/fsdownload/TimeMixer || exit 1
PY="/c/Users/Administrator/.workbuddy/binaries/python/envs/default/Scripts/python.exe"
OUT=/d/fsdownload/trace_result.log

echo "waiter started $(date '+%F %T')" > "$OUT"

# --- 1) 等 weather det0 批次结束（det1 在其之前）---
until grep -q "SMOKE weather DONE" /d/fsdownload/weather_det0.log 2>/dev/null; do
  sleep 60
done
echo "weather smoke finished $(date '+%F %T')" >> "$OUT"

# --- 2) 确认没有残留训练进程（避免抢 GPU 影响确定性）---
for i in $(seq 1 30); do
  n=$(ps -ef | grep -c "[r]un\.py")
  [ "$n" = "0" ] && break
  echo "  等待残留进程退出 (n=$n) ..." >> "$OUT"
  sleep 20
done
sleep 10

LOG="D:/fsdownload/determinism_test_det0/same1.log"

# --- 3) 两个独立进程，同 seed，det_mode=0（= 现有全部实验的配置）---
"$PY" -u tools/trace_divergence.py --log "$LOG" --out D:/fsdownload/trace_run1.json \
      --seed 2021 --det 0 --epochs 1 >> "$OUT" 2>&1
echo "--- run1 done $(date '+%T') ---" >> "$OUT"

"$PY" -u tools/trace_divergence.py --log "$LOG" --out D:/fsdownload/trace_run2.json \
      --seed 2021 --det 0 --epochs 1 >> "$OUT" 2>&1
echo "--- run2 done $(date '+%T') ---" >> "$OUT"

# --- 4) 比对定位 ---
"$PY" -u tools/trace_divergence.py --compare D:/fsdownload/trace_run1.json D:/fsdownload/trace_run2.json >> "$OUT" 2>&1

echo "TRACE ALL DONE $(date '+%F %T')" >> "$OUT"
