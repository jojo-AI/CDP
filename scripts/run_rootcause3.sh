#!/bin/bash
# =============================================================================
# 根因追查 第三轮：把「step 150 反向梯度分叉」定位到具体算子（2026-10-08）
#
# 第二轮结论：
#   A 段：同一 checkpoint 两次独立进程的 eval 前向逐 batch 输出哈希完全一致
#         ⇒ 分叉不在评估阶段
#   B 段：跑满 286 步（1 个 epoch）的逐 step 哈希显示
#         - in_x / in_mark / out / param_pre（含前 149 步全部）逐位一致
#         - 🎯 第一个分叉发生在 **第 150 步**，分歧量 = **grad（反向梯度）**
#         ⇒ 非确定性在**反向传播 kernel**，与作者原始实现无关，也不是数据加载问题
#
# 本轮要做的：保留真实 LR 调度（--keep-epochs，否则 OneCycleLR 的 total_steps 被改，
#   整条轨迹与真实实验不一致，实测 train loss 0.3209 vs 真实 0.3688），
#   逐参数记录梯度哈希 → 看是哪个参数（哪个算子）的梯度先不同。
#
# 用法：bash scripts/run_rootcause3.sh   （后台跑，约 8-10 分钟）
# =============================================================================
PY="/c/Users/Administrator/.workbuddy/binaries/python/envs/default/Scripts/python.exe"
cd "/d/fsdownload/TimeMixer" || exit 1
LOG="/d/fsdownload/rootcause3.log"
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

SMOKE="D:/fsdownload/det_weather_det1/same1.log"
say "########## ROOTCAUSE ROUND-3 START (pid=$$) ##########"

for r in 1 2; do
  say "--- grad-detail run$r（真实 LR 调度，记录满 286 步即中断）---"
  "$PY" -u tools/trace_divergence.py --log "$SMOKE" --out "D:/fsdownload/tgd_${r}.json" \
        --seed 2021 --det 1 --keep-epochs --max-steps 286 \
        --layers 0 --param-every 1 --grad-detail 2>&1 \
    | grep -avE "^\[PDM\]|^\[CDP\]" | tee -a "$LOG"
done

say "--- compare ---"
"$PY" -u tools/trace_divergence.py --compare D:/fsdownload/tgd_1.json D:/fsdownload/tgd_2.json 2>&1 | tee -a "$LOG"

say "########## ROOTCAUSE ROUND-3 DONE ##########"
