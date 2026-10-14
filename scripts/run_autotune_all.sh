#!/bin/bash
# 串行跑完剩余全部设定（等当前 run_autotune.sh 结束后再启动，避免争抢 GPU）。
# 顺序：weather(骨干已就绪) -> ECL(需从头训骨干) -> Traffic(SKIP_ST1，只做 CDP 校准)
#
# Traffic 说明：862 通道，Stage-1 重训骨干代价极高，故 SKIP_ST1=1 + BUDGET=lean；
#   此时脚本会退回作者原始骨干（run_cdp_v4.sh 的 BASE 兜底逻辑）。
cd /d/fsdownload/TimeMixer || exit 1
LOG=/d/fsdownload/autotune_all.log
echo "===== [$(date '+%F %T')] autotune_all 启动，等待当前 autotune 结束 =====" >> $LOG

# 1) 等现有 ETTh1 队列退出
# 注意：Git Bash 下 pgrep -f 对「带参数的长命令行」匹配不可靠，改用 ps + grep 直接判定。
# 注意2：必须排除 run_autotune_all.sh 自身的启动行（它第 9 行的 echo 里含 "ETTh1 队列已结束"，
#        但脚本名不含该串；真正的坑是 grep 会把本脚本 pid 也算进来 → 死循环）。用 -w + 精确列。
wait_queue () {
  while ps -ef | grep -v grep | grep -E "run_autotune\.sh ${1}( |$)|run_cdp_v4\.sh ${1}( |$)" \
        | grep -v "run_autotune_all" > /dev/null 2>&1; do
    sleep 60
  done
}
wait_queue ETTh1
echo "===== [$(date '+%F %T')] ETTh1 队列已结束，开始 weather =====" >> $LOG

# 防重复启动：若已有另一个 autotune_all 在做真正的推进（非本进程），直接退出。
# 历史故障：15:43 与 16:46 各拉起一次，两份队列互相抢 GPU，且都卡在 wait_queue 死循环。
# 注意：不能再用 ps|grep|wc 自数——管道自身的子 shell 会被计进来（曾误报 3 个）。
#       改为用 lock 目录做互斥，最可靠。
LOCK=/d/fsdownload/.autotune_all.lock
if [ -d "$LOCK" ]; then
  echo "!!!!! [$(date '+%F %T')] 已有 autotune_all 在运行（lock=$LOCK），本进程退出 !!!!!" >> $LOG
  exit 1
fi
mkdir -p "$LOCK" && echo $$ > "$LOCK/pid"
trap 'rm -rf "$LOCK"' EXIT INT TERM

# 2) weather：Stage-1 骨干已就绪（会被跳过），直接跑 4 个 horizon 的 Phase-1 候选
for h in 96 192 336 720; do
  echo "--- [$(date '+%T')] weather h$h ---" >> $LOG
  PHASE=1 bash scripts/run_autotune.sh weather $h >> $LOG 2>&1
  echo "--- [$(date '+%T')] weather h$h 结束 (exit=$?) ---" >> $LOG
done

# 3) ECL：需要 Stage-1 重训骨干（321 通道，可用）
for h in 96 192 336 720; do
  echo "--- [$(date '+%T')] ECL h$h ---" >> $LOG
  PHASE=1 BUDGET=lean bash scripts/run_autotune.sh ECL $h >> $LOG 2>&1
  echo "--- [$(date '+%T')] ECL h$h 结束 (exit=$?) ---" >> $LOG
done

# 4) Traffic：跳过 Stage-1（超大通道），只做 CDP 校准候选
for h in 96 192 336 720; do
  echo "--- [$(date '+%T')] Traffic h$h ---" >> $LOG
  SKIP_ST1=1 BUDGET=lean PHASE=1 bash scripts/run_autotune.sh Traffic $h >> $LOG 2>&1
  echo "--- [$(date '+%T')] Traffic h$h 结束 (exit=$?) ---" >> $LOG
done

echo "===== [$(date '+%F %T')] autotune_all 全部完成 =====" >> $LOG
cd /d/fsdownload && /c/Users/Administrator/.workbuddy/binaries/python/envs/default/Scripts/python.exe TimeMixer/tools/select_best.py >> $LOG 2>&1
echo "AUTOTUNE_ALL DONE"
