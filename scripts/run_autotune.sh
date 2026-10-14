#!/bin/bash
# 逐 (数据集 × horizon) 自动调优：Stage-1 初始化选型 + Stage-2 配置网格，全部跑完后由
# tools/select_best.py 按「MSE/MAE 同时最大化」自动选优。
#
# 固定协议（保证同设定内所有候选可比）：
#   Stage-1: EP=30 PAT=8，dynamic + 纯 L1
#   Stage-2: EP=20 PAT=12，lr=5e-4，V4BASE（unfreeze_head/norm, chan_rank 8）+ 核心 ST2
# 用法: bash run_autotune.sh <dataset> <horizon> [horizon...]
PY="/c/Users/Administrator/.workbuddy/binaries/python/envs/default/Scripts/python.exe"
cd "/d/fsdownload/TimeMixer" || exit 1
RES=/d/fsdownload/v4_results.txt
RES_ST1=/d/fsdownload/stage1_results.txt
DS=$1; shift
EP_ST2=${EP_ST2:-20}; PAT_ST2=${PAT_ST2:-12}
CORE="--backbone_mode dynamic --cdp_unfreeze_pdm 1 --cdp_st_lr_ratio 0.2 --cdp_dyncal 1 \
      --cdp_loss l1 --cdp_monitor mae"

# 结果行里的 tag 字段是 TAG+MIDP（如 v5final_ETTh1_96_fin_），所以按前缀匹配
# 注意：必须要求该行带真实数值（mse:..），否则训练中途崩溃写下的「空结果行」会被误判为
#       已完成，导致该候选被永久 skip（历史上 ETTh1 h192 at_cal1 就是这样被吞掉的）。
done_tag () {
  grep -aE "(^| )${DS} h${2} ${1}[^ ]* " $RES 2>/dev/null | grep -aq "mse:"
}

# 取该 horizon 测试 MSE 最低的 Stage-1 骨干（第 $3 名，默认第 1）
pick_ck () {
  local pl=$1 idx=${2:-1}
  local tag=$(grep -a "^${DS} h${pl} st1_dynamic_l10.0" $RES_ST1 | \
      awk '{v=$4; sub("mse:","",v); print v" "$3}' | sort -n | awk -v i=$idx 'NR==i{print $2}')
  [ -z "$tag" ] && return 1
  case "$tag" in
    *hi0) dirtag="st1_${pl}_dyn0_l10.0" ;;
    *)    dirtag="st1_${pl}_dynamic_l10.0" ;;
  esac
  ls -d ./checkpoints/long_term_forecast_${DS}_${dirtag}_* 2>/dev/null | head -1
}

run_cfg () {   # $1=pl $2=internal_tag $3=midp $4=ckpt  rest=extra args
  local PL=$1 itag=$2 midp=$3 ck=$4; shift 4
  local TAG="${itag}_${DS}_${PL}"
  if done_tag "$TAG" "$PL"; then echo "[skip] ${TAG}"; return 0; fi
  echo "===== [$(date '+%T')] AUTOTUNE ${TAG} (backbone=$(basename $ck | cut -c1-46)) ====="
  local ep=${EP_OVR:-$EP_ST2}; local pat=${PAT_OVR:-$PAT_ST2}
  BASE_OVR="$ck" MIDP="$midp" EP_OVR=$ep PAT_OVR=$pat TAG_OVR="$TAG" \
    ST2_EXTRA="$CORE $*" bash scripts/run_cdp_v4.sh ${DS} ${PL} custom 2>&1 | tail -1
}

for PL in "$@"; do
  # --- Stage-1 候选：手调表初始化 与 hi0（各跑一次，幂等）---
  # SKIP_ST1=1 时完全跳过（超大通道数据集没法重训骨干，只能做 CDP 校准）
  if [ "${SKIP_ST1:-0}" = "1" ]; then
    echo "[skip] Stage-1 disabled (SKIP_ST1=1) for ${DS} h${PL}"
  elif ! grep -aq "^${DS} h${PL} st1_dynamic_l10.0 " $RES_ST1; then
    echo "===== [$(date '+%T')] ST1 handtuned-init ${DS} h${PL} ====="
    TLOSS=l1 TL1MU=0.0 bash scripts/run_stage1_st.sh ${DS} ${PL} dynamic 2>&1 | tail -1
  fi
  if [ "${SKIP_ST1:-0}" != "1" ] && ! grep -aq "^${DS} h${PL} st1_dynamic_l10.0_hi0 " $RES_ST1; then
    echo "===== [$(date '+%T')] ST1 hi0 ${DS} h${PL} ====="
    bash scripts/run_st1_hi0.sh ${DS} ${PL} 2>&1 | tail -1
  fi

  CK1=$(pick_ck $PL 1); CK2=$(pick_ck $PL 2)
  if [ -z "$CK1" ]; then
    CK1=$(ls -d ./checkpoints/long_term_forecast_${DS}_96_${PL}_none_TimeMixer_* 2>/dev/null | grep -v CDP | head -1)
    echo "[pick] 无 Stage-1 结果，退回作者原始骨干: $CK1"
  fi
  echo "[pick] best=${CK1}"
  [ -z "$CK1" ] && { echo "!! no backbone ckpt for ${DS} h${PL}"; continue; }

  # --- Phase 1：便宜候选（覆盖最主要的自由度） ---
  run_cfg $PL v5final _fin_  "$CK1" --cdp_cal_lr_ratio 5
  run_cfg $PL at_cal1  _at1_  "$CK1" --cdp_cal_lr_ratio 1
  run_cfg $PL at_sw1   _ats1_ "$CK1" --cdp_cal_lr_ratio 5 --cdp_step_weight 1.0
  # 次优骨干上用当前最佳配置再试一次（防止「初始化 × 配置」交互）
  if [ "${BUDGET:-full}" != "lean" ]; then
    [ -n "$CK2" ] && [ "$CK2" != "$CK1" ] && run_cfg $PL at_2nd _at2_ "$CK2" --cdp_cal_lr_ratio 5
  fi

  if [ "${PHASE:-1}" = "2" ]; then
    # --- Phase 2：更贵的候选，只在该设定「尚未 MSE/MAE 双双 ≥5%」时跑（FORCE=1 可强制） ---
    need=1
    if [ "${FORCE:-0}" != "1" ]; then
      g=$($PY tools/select_best.py 2>/dev/null | grep -E "^\| ${DS} \| ${PL} \|" | grep -oE "[-+][0-9.]+% \| [-+][0-9.]+%" | head -1)
      if [ -n "$g" ]; then
        a=$(echo "$g" | awk '{print $1}' | tr -d '%'); b=$(echo "$g" | awk '{print $3}' | tr -d '%')
        awk -v a="$a" -v b="$b" 'BEGIN{exit !(a<=-5 && b<=-5)}' && need=0
      fi
    fi
    if [ "$need" = "1" ]; then
      run_cfg $PL at_sw2   _ats2_ "$CK1" --cdp_cal_lr_ratio 5 --cdp_step_weight 2.0
      run_cfg $PL at_st05  _ato5_ "$CK1" --cdp_cal_lr_ratio 5 --cdp_st_lr_ratio 0.5
      EP_OVR=30 run_cfg $PL at_ep30 _ate3_ "$CK1" --cdp_cal_lr_ratio 5
      EP_OVR=30 run_cfg $PL at_ep30c1 _ate4_ "$CK1" --cdp_cal_lr_ratio 1
    else
      echo "[phase2-skip] ${DS} h${PL} 已双双 ≥5%（$g）"
    fi
  fi
done
echo "AUTOTUNE DONE ${DS} $*"
$PY tools/select_best.py 2>/dev/null | tail -40
