#!/bin/bash
# ETTh1-96：在「dynamic+L1 重训骨干」之上，叠加 CDP + 部分解冻骨干（blk1/blk2/blk3）
PY="/c/Users/Administrator/.workbuddy/binaries/python/envs/default/Scripts/python.exe"
cd "/d/fsdownload/TimeMixer" || exit 1
CK="long_term_forecast_ETTh1_st1_96_dynamic_l10.5_none_TimeMixer_ETTh1_sl96_pl96_dm16_nh8_el2_dl1_df32_fc1_ebtimeF_dtTrue_ST1_0"
BASE="--backbone_mode dynamic --cdp_unfreeze_pdm 1 --cdp_st_lr_ratio 0.2 --cdp_dyncal 1 --cdp_loss mse_l1 --cdp_l1_mu 1.0 --cdp_monitor mae"
for BLK in 1 2 3; do
  TAG="stack_blk${BLK}_ETTh1_96"
  if grep -q " $TAG " /d/fsdownload/v4_results.txt 2>/dev/null; then echo "skip $TAG"; continue; fi
  echo "===== [$(date '+%T')] $TAG ====="
  BASE_OVR="$CK" MIDP="_s1d" EP_OVR=30 PAT_OVR=8 TAG_OVR="$TAG" \
    ST2_EXTRA="$BASE --cdp_unfreeze_blocks ${BLK} --cdp_head_lr_ratio 0.3" \
    bash scripts/run_cdp_v4.sh ETTh1 96 custom
done
echo "STACK BLK DONE"
