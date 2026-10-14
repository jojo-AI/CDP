#!/bin/bash
# 补齐「扩展数据集」队列：等 autotune_all 的主锁释放（weather→ECL→Traffic 跑完）后，
# 依次跑 ETTh2(336,720 补缺) -> ETTm1(全4) -> ETTm2(全4)。
#
# 为什么需要它：run_autotune_all.sh 只覆盖有作者预训练骨干的 weather/ECL/Traffic，
# 而 ETTh2/ETTm1/ETTm2 必须自训基线，需走 run_ett_extra.sh。28 设定目标缺这 12 个。
#
# 用法: bash run_queue_extras.sh          # 后台串行等待并执行
cd /d/fsdownload/TimeMixer || exit 1
LOG=/d/fsdownload/extras_queue.log
LOCK=/d/fsdownload/.autotune_all.lock

echo "===== [$(date '+%F %T')] extras 队列启动，等待主队列释放锁 =====" >> $LOG

# 等主队列（autotune_all）释放锁；最长等 24h，避免永久挂住
i=0
while [ -d "$LOCK" ]; do
  sleep 120
  i=$((i+1))
  if [ $i -gt 720 ]; then
    echo "!!!!! [$(date '+%F %T')] 等待超时(24h)，主队列锁仍存在，extras 退出 !!!!!" >> $LOG
    exit 1
  fi
done
echo "===== [$(date '+%F %T')] 主队列已结束，extras 开始 =====" >> $LOG

# ETTh2：96/192 已完成，补 336/720
echo "--- [$(date '+%T')] ETTh2 h336 h720 ---" >> $LOG
bash scripts/run_ett_extra.sh ETTh2 336 720 >> $LOG 2>&1
echo "--- [$(date '+%T')] ETTh2 结束 (exit=$?) ---" >> $LOG

# ETTm1：全 4 个
echo "--- [$(date '+%T')] ETTm1 96 192 336 720 ---" >> $LOG
bash scripts/run_ett_extra.sh ETTm1 96 192 336 720 >> $LOG 2>&1
echo "--- [$(date '+%T')] ETTm1 结束 (exit=$?) ---" >> $LOG

# ETTm2：全 4 个（注意 d_model=32）
echo "--- [$(date '+%T')] ETTm2 96 192 336 720 ---" >> $LOG
bash scripts/run_ett_extra.sh ETTm2 96 192 336 720 >> $LOG 2>&1
echo "--- [$(date '+%T')] ETTm2 结束 (exit=$?) ---" >> $LOG

echo "===== [$(date '+%F %T')] extras 队列全部完成 =====" >> $LOG
cd /d/fsdownload && /c/Users/Administrator/.workbuddy/binaries/python/envs/default/Scripts/python.exe TimeMixer/tools/select_best.py >> $LOG 2>&1
echo "EXTRAS_QUEUE DONE"
