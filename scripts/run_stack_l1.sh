#!/bin/bash
# ETTh1-96：在「dynamic + 纯 L1」重训骨干之上做 CDP stage-2（cdp_loss=l1）
PY="/c/Users/Administrator/.workbuddy/binaries/python/envs/default/Scripts/python.exe"
cd "/d/fsdownload/TimeMixer" || exit 1
CK="long_term_forecast_ETTh1_st1_96_dynamic_l10.0_none_TimeMixer_ETTh1_sl96_pl96_dm16_nh8_el2_dl1_df32_fc1_ebtimeF_dtTrue_ST1_0"
TAG="stack_l1_ETTh1_96"
if grep -q " $TAG " /d/fsdownload/v4_results.txt 2>/dev/null; then echo "skip $TAG"; exit 0; fi
BASE_OVR="$CK" MIDP="_s1l1" EP_OVR=30 PAT_OVR=8 TAG_OVR="$TAG" \
  ST2_EXTRA="--backbone_mode dynamic --cdp_unfreeze_pdm 1 --cdp_st_lr_ratio 0.2 --cdp_dyncal 1 --cdp_loss l1 --cdp_monitor mae" \
  bash scripts/run_cdp_v4.sh ETTh1 96 custom
echo "STACK L1 DONE"
