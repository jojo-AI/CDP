#!/bin/bash
# =============================================================================
# 主队列：把「Stage-1 骨干重训（可学习季节/趋势权重）+ Stage-2 CDP 校准」
# 推到全部数据集 × horizon。
#
# 设计要点：
#   * 可断点续跑：结果文件里已有该 tag 就跳过（重复执行安全）。
#   * 顺序按「快 → 慢」：ETTh1/weather → ECL → Traffic，
#     保证在 GPU 时间有限时先把能验证的部分做完。
#   * Traffic 只做 Stage-2（862 通道全量重训在本机不可行，约 24h/次）。
#
# 用法: bash run_master_queue.sh [dataset...]     默认: ETTh1 weather ECL Traffic
# 环境变量:
#   HZ_LIST    要跑的 horizon 列表（默认 "96 192 336 720"）
#   EP_ST1     Stage-1 训练 epoch（默认 30）
#   EP_ST2     Stage-2 训练 epoch（默认 30）
#   ST2_EXTRA  Stage-2 附加参数（默认 v4 赢家配方）
# =============================================================================
PY="/c/Users/Administrator/.workbuddy/binaries/python/envs/default/Scripts/python.exe"
cd "/d/fsdownload/TimeMixer" || exit 1
RES="/d/fsdownload/v4_results.txt"
touch "$RES"

DSLIST=${*:-"ETTh1 weather ECL Traffic"}
HZ_LIST=${HZ_LIST:-"96 192 336 720"}
EP_ST1=${EP_ST1:-30}
ST1_LOSS=${ST1_LOSS:-l1}
ST1_MU=${ST1_MU:-0.5}
EP_ST2=${EP_ST2:-30}
ST2_EXTRA=${ST2_EXTRA:-"--backbone_mode dynamic --cdp_unfreeze_pdm 1 --cdp_st_lr_ratio 0.2 --cdp_dyncal 1 --cdp_loss l1 --cdp_monitor mae --cdp_cal_lr_ratio 5"}

has_result () { grep -qE "(^| )$1[^ ]* " "$RES" 2>/dev/null; }

for DS in $DSLIST; do
  for PL in $HZ_LIST; do
    ### Stage 1：骨干重训（可学习的季节/趋势混合 + MSE/L1 混合损失）；Traffic 跳过
    if [ "$DS" != "Traffic" ]; then
      CK=$(ls -d ./checkpoints/long_term_forecast_${DS}_st1_${PL}_dynamic_${ST1_LOSS}* 2>/dev/null | head -1)
      if [ -z "$CK" ]; then
        echo "===== [$(date '+%T')] STAGE1 ${DS} h${PL} backbone(dynamic + mse_l1 mu=${ST1_MU}) ====="
        TLOSS=${ST1_LOSS} TL1MU=${ST1_MU} EP_OVR=${EP_ST1} bash scripts/run_stage1_st.sh ${DS} ${PL} dynamic
        CK=$(ls -d ./checkpoints/long_term_forecast_${DS}_st1_${PL}_dynamic_${ST1_LOSS}* 2>/dev/null | head -1)
      else
        echo "===== [skip] stage1 ${DS} h${PL} (ckpt exists: $CK) ====="
      fi
      ### Stage 2：在本方法骨架上做 CDP 校准
      if [ -n "$CK" ] && ! has_result "cdp_on_st1_${DS}_${PL}" && ! has_result "cdp_cal5_${DS}_${PL}"; then
        echo "===== [$(date '+%T')] STAGE2 ${DS} h${PL} on method backbone (cal5) ====="
        BASE_OVR="$CK" MIDP="_ds" EP_OVR=${EP_ST2} PAT_OVR=8 TAG_OVR="cdp_cal5_${DS}_${PL}" \
          ST2_EXTRA="$ST2_EXTRA" bash scripts/run_cdp_v4.sh ${DS} ${PL} custom
      fi
    fi

    ### Stage 2b：在作者原始（clean）骨架上做 CDP 校准——用于对标论文中的 TimeMixer 基线
    ### 非 Traffic 数据集默认跳过（本方法的主实验是「dynamic+L1 重训骨干 → CDP」），
    ### 需要「CDP 单独贡献」的消融时把 SKIP_CLEAN 设为 0。
    if [ "${SKIP_CLEAN:-1}" = "0" ] && ! has_result "cdp_on_clean_${DS}_${PL}"; then
      echo "===== [$(date '+%T')] STAGE2 ${DS} h${PL} on clean backbone ====="
      MIDP="_dc" EP_OVR=${EP_ST2} PAT_OVR=8 TAG_OVR="cdp_on_clean_${DS}_${PL}" \
        ST2_EXTRA="$ST2_EXTRA" bash scripts/run_cdp_v4.sh ${DS} ${PL} custom
    fi
  done
done
echo "MASTER QUEUE DONE"
