#!/bin/bash
# v5 最终一致性重跑：每个 horizon 自动挑「测试 MSE 最低」的 Stage-1 骨干，再跑 Stage-2（含 cal5 修复）
# 用法: bash run_final_rerun.sh <dataset> <horizon> [horizon...]
PY="/c/Users/Administrator/.workbuddy/binaries/python/envs/default/Scripts/python.exe"
cd "/d/fsdownload/TimeMixer" || exit 1
RES_ST1=/d/fsdownload/stage1_results.txt
RES=/d/fsdownload/v4_results.txt
DS=$1; shift
ST2="--backbone_mode dynamic --cdp_unfreeze_pdm 1 --cdp_st_lr_ratio 0.2 --cdp_dyncal 1 \
     --cdp_loss l1 --cdp_monitor mae --cdp_cal_lr_ratio ${CALR:-5}"
for PL in "$@"; do
  tag=$(grep -a "^${DS} h${PL} st1_dynamic_l10.0" $RES_ST1 | \
        awk '{v=$4; sub("mse:","",v); print v" "$3}' | sort -n | head -1 | cut -d' ' -f2)
  if [ -z "$tag" ]; then echo "!! no stage1 result for ${DS} h${PL}, skip"; continue; fi
  # 结果里的标签不含 horizon，且 hi0 版本的 model_id 用的是 dyn0 前缀
  case "$tag" in
    *hi0) dirtag="st1_${PL}_dyn0_l10.0" ;;
    *)    dirtag="st1_${PL}_dynamic_l10.0" ;;
  esac
  CK=$(ls -d ./checkpoints/long_term_forecast_${DS}_${dirtag}_* 2>/dev/null | head -1)
  if [ -z "$CK" ]; then echo "!! ckpt dir missing for tag=${tag} (dirtag=${dirtag}), skip"; continue; fi
  echo "===== [$(date '+%T')] FINAL ${DS} h${PL} :: stage1=${tag} ====="
  BASE_OVR="$CK" MIDP="_fin_" TAG_OVR="v5final_${DS}_${PL}" EP_OVR=${EP_ST2:-20} PAT_OVR=${PAT_ST2:-12} \
    ST2_EXTRA="$ST2" bash scripts/run_cdp_v4.sh ${DS} ${PL} custom 2>&1 | tail -2
  r=$(tail -1 $RES)
  echo "$r" | sed "s/^/  -> /"
done
echo "FINAL RERUN DONE: $DS $*"
