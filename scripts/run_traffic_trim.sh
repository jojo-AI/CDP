#!/bin/bash
# =============================================================================
# Traffic 快速收尾队列（只跑每档的最优候选 v5final，跳过 at_cal1 / at_sw1）
# -----------------------------------------------------------------------------
# 背景（2026-10-07）：用户要求「尽快出最终结果、只保留最优候选」。
#   原 run_remaining.sh 的 PHASE B 会对 Traffic 每档跑 3 个候选（v5final/at_cal1/at_sw1），
#   实测单候选 ~5.7h，4 档 12 个候选 ≈ 68h。
#   而 Traffic-96 的三候选实测打成平手（差异 <0.001），探索价值已耗尽
#   → 剩余 3 档只跑 v5final，省下 2/3 时间。
#
# 流程：
#   0) 等 h192 v5final 当前这次训练把结果写进 v4_results.txt（旧链条自己会写）
#   1) 立刻收掉旧队列（此时它在跑 at_cal1，只浪费 <1 分钟）
#   2) 解除看门狗（写 REMAINING QUEUE DONE 标记，watchdog 会自行退出）
#   3) 接管锁，逐档跑 v5final（幂等：已有结果则跳过）
#   4) 刷新最优榜
#
# 用法: bash scripts/run_traffic_trim.sh   （用 Bash 工具 run_in_background=true 启动）
# =============================================================================
cd /d/fsdownload/TimeMixer || exit 1
LOG=/d/fsdownload/traffic_trim.log
RES=/d/fsdownload/v4_results.txt
REMLOG=/d/fsdownload/remaining.log
LOCK=/d/fsdownload/.remaining.lock
PY="/c/Users/Administrator/.workbuddy/binaries/python/envs/default/Scripts/python.exe"
CORE="--backbone_mode dynamic --cdp_unfreeze_pdm 1 --cdp_st_lr_ratio 0.2 --cdp_dyncal 1 --cdp_loss l1 --cdp_monitor mae"

say () { echo "===== [$(date '+%F %T')] $* =====" | tee -a "$LOG"; }
has_result () { grep -aE "(^| )$1[^ ]* " "$RES" 2>/dev/null | grep -aq "mse:"; }

say "TRAFFIC TRIM QUEUE START (pid=$$)"

# ---------------------------------------------------------------------------
# 0) 等 h192 v5final 落盘（最多 6 小时）。两种结束条件：
#    结果已写 / 训练进程消失。若进程消失但结果没写 → 后面按正常流程重跑该档。
# ---------------------------------------------------------------------------
say "等待 h192 v5final 结果落盘（旧队列写入）..."
for i in $(seq 1 1080); do
  has_result "Traffic h192 v5final" && { say "h192 v5final 结果已落盘"; break; }
  ps -ef | grep -E "run\.py" | grep -qv grep || { say "训练进程已消失（结果未落盘）"; break; }
  sleep 20
done

# ---------------------------------------------------------------------------
# 1) 收掉旧队列：bash 脚本用 MSYS kill，原生 python 用 winpid + taskkill
# ---------------------------------------------------------------------------
say "收掉旧队列进程"
for p in $(ps -ef | grep -E "run_remaining\.sh|run_autotune\.sh|run_cdp_v4\.sh" | grep -v grep | awk '{print $2}'); do
  kill -9 "$p" 2>/dev/null && say "  killed bash msys=$p"
done
sleep 2
for p in $(ps -ef | grep -E "run\.py" | grep -v grep | awk '{print $2}'); do
  w=$(cat /proc/$p/winpid 2>/dev/null)
  if [ -n "$w" ]; then taskkill //F //PID "$w" >/dev/null 2>&1 && say "  killed python msys=$p win=$w"; fi
done
sleep 3

# ---------------------------------------------------------------------------
# 2) 解除看门狗：它只在 remaining.log 未出现 DONE 时才会重启 run_remaining.sh
# ---------------------------------------------------------------------------
if ! grep -aq "REMAINING QUEUE DONE" "$REMLOG"; then
  echo "===== [$(date '+%F %T')] REMAINING QUEUE DONE (superseded by run_traffic_trim.sh) =====" >> "$REMLOG"
fi
for p in $(ps -ef | grep -E "watch_remaining\.sh" | grep -v grep | awk '{print $2}'); do
  kill -9 "$p" 2>/dev/null && say "  killed watchdog msys=$p"
done

# ---------------------------------------------------------------------------
# 3) 接管锁（先确认原持锁者已死）
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
# 4) 逐档跑 v5final（只此一个候选）
# ---------------------------------------------------------------------------
for PL in 192 336 720; do
  TAG="v5final_Traffic_${PL}"
  if has_result "$TAG"; then
    say "[skip] ${TAG} 已有结果"
    continue
  fi
  CK=$(ls -d ./checkpoints/long_term_forecast_Traffic_96_${PL}_none_TimeMixer_* 2>/dev/null \
        | grep -viE "cdp|st1|v4|v31|_at|v5|cal" | head -1)
  if [ -z "$CK" ]; then say "!! 无骨干 ckpt: Traffic h${PL}"; continue; fi
  say "RUN ${TAG} (backbone=$(basename "$CK"))"
  BASE_OVR="$CK" MIDP="_fin_" EP_OVR=20 PAT_OVR=12 TAG_OVR="$TAG" \
    ST2_EXTRA="$CORE --cdp_cal_lr_ratio 5" \
    bash scripts/run_cdp_v4.sh Traffic ${PL} custom >> "$LOG" 2>&1 || say "FAILED ${TAG}"
done

# ---------------------------------------------------------------------------
# 5) 收尾
# ---------------------------------------------------------------------------
say "刷新最优榜"
"$PY" tools/select_best.py >> "$LOG" 2>&1
say "TRAFFIC TRIM QUEUE DONE"
