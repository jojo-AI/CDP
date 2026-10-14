#!/bin/bash
# =============================================================================
# 第四轮：再做 2 个独立进程（共 4 次），确认「反向梯度分叉」是
#   （a）固定复现的（每次都卡在同一步同一参数）还是
#   （b）稀有、随机的（每次落在不同 step）—— 这决定了它是不是「每次运行都发生」的
#
# 第三轮结论：真实 LR 调度下，run1 vs run2 的
#   第一分叉 = 第 258 步，且 **只差 1 个参数的梯度：predict_layers.1.weight**
#   （in_x / in_mark / out / param_pre 全一致；且该步 Adam 更新后 param_post 又一致）
# =============================================================================
PY="/c/Users/Administrator/.workbuddy/binaries/python/envs/default/Scripts/python.exe"
cd "/d/fsdownload/TimeMixer" || exit 1
LOG="/d/fsdownload/rootcause4.log"
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

SMOKE="D:/fsdownload/det_weather_det1/same1.log"
say "########## ROOTCAUSE ROUND-4 START (pid=$$) ##########"
for r in 3 4; do
  say "--- grad-detail run$r ---"
  "$PY" -u tools/trace_divergence.py --log "$SMOKE" --out "D:/fsdownload/tgd_${r}.json" \
        --seed 2021 --det 1 --keep-epochs --max-steps 286 \
        --layers 0 --param-every 1 --grad-detail 2>&1 \
    | grep -avE "^\[PDM\]|^\[CDP\]" | grep -aE "中断|写出|Epoch" | tee -a "$LOG"
done
say "--- compare 4 次 ---"
"$PY" -u tools/trace_divergence.py --compare D:/fsdownload/tgd_1.json D:/fsdownload/tgd_2.json \
      D:/fsdownload/tgd_3.json D:/fsdownload/tgd_4.json 2>&1 | tail -30 | tee -a "$LOG"
say "########## ROOTCAUSE ROUND-4 DONE ##########"
