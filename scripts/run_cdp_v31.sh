#!/bin/bash
# CDP v3.1 变体脚本（干净骨干 + 全部新杠杆）
# 用法: bash run_cdp_v31.sh <dataset> <horizon> <stage:screen|final> [variant...]
#   stage=screen → 15 epochs / patience 5（快筛）
#   stage=final  → 24 epochs / patience 7（正式）
# 可用环境变量: EP_OVR / PAT_OVR / V3LR / V3RANK 覆盖
PY="/c/Users/Administrator/.workbuddy/binaries/python/envs/default/Scripts/python.exe"
cd "/d/fsdownload/TimeMixer" || exit 1
RES="/d/fsdownload/v31_results.txt"

DS=$1; PL=$2; STAGE=${3:-screen}
shift 3 2>/dev/null
WANT="$*"          # 可选：只跑指定变体

case "$DS" in
  ETTh1)   DSARGS="--root_path ./dataset/ETT-small/ --data_path ETTh1.csv --data ETTh1 --enc_in 7 --dec_in 7 --c_out 7 --e_layers 2 --factor 1 --d_model 16 --d_ff 32 --batch_size 128" ;;
  ETTh2)   DSARGS="--root_path ./dataset/ETT-small/ --data_path ETTh2.csv --data ETTh2 --enc_in 7 --dec_in 7 --c_out 7 --e_layers 2 --factor 1 --d_model 16 --d_ff 32 --batch_size 128" ;;
  ETTm2)   DSARGS="--root_path ./dataset/ETT-small/ --data_path ETTm2.csv --data ETTm2 --enc_in 7 --dec_in 7 --c_out 7 --e_layers 3 --factor 3 --d_model 16 --d_ff 32 --batch_size 128" ;;
  ECL)     DSARGS="--root_path ./dataset/electricity/ --data_path electricity.csv --data custom --enc_in 321 --dec_in 321 --c_out 321 --e_layers 3 --factor 3 --d_model 16 --d_ff 32 --batch_size 32" ;;
  Traffic) DSARGS="--root_path ./dataset/traffic/ --data_path traffic.csv --data custom --enc_in 862 --dec_in 862 --c_out 862 --e_layers 3 --factor 3 --d_model 32 --d_ff 64 --batch_size 16" ;;
  weather) DSARGS="--root_path ./dataset/weather/ --data_path weather.csv --data custom --enc_in 21 --dec_in 21 --c_out 21 --e_layers 3 --factor 3 --d_model 16 --d_ff 32 --batch_size 128" ;;
  *) echo "unknown dataset $DS"; exit 1 ;;
esac

# 找冻结骨干 checkpoint（兼容目录名里的引号字符）
find_ckpt() {
  local prefix="$1" plen="$2" cand
  cand=$(ls -d ${prefix}_* 2>/dev/null | head -1)
  if [ -z "$cand" ]; then
    cand=$(ls -d ${prefix/_96_${plen}_none_TimeMixer/_96*${plen}_none_TimeMixer}_* 2>/dev/null | head -1)
  fi
  [ -n "$cand" ] && [ -d "$cand" ] && echo "$cand"
}
BASE=$(find_ckpt "./checkpoints/long_term_forecast_${DS}_96_${PL}_none_TimeMixer" ${PL})
if [ -z "$BASE" ]; then echo "!!! backbone ckpt not found for ${DS} h${PL}"; exit 1; fi
echo "backbone = $BASE"

if [ "$STAGE" = "screen" ]; then EP=${EP_OVR:-15}; PAT=${PAT_OVR:-5}; else EP=${EP_OVR:-24}; PAT=${PAT_OVR:-7}; fi

# 注意：--backbone_mode clean 是必须的（上游官方 out=season+trend）
COMMON="--task_name long_term_forecast --model TimeMixer_CDP --features M --seq_len 96 --label_len 0 \
  --d_layers 1 --des CDPv31 --itr 1 --learning_rate 0.01 --num_workers 0 \
  --down_sampling_layers 3 --down_sampling_method avg --down_sampling_window 2 \
  --backbone_mode clean --cdp_stage_w 1 \
  --cdp_M 4 --cdp_S 16 --cdp_err_decay 0.9 --cdp_lambda_t 0.01 \
  --train_epochs ${EP} --patience ${PAT} ${DSARGS}"

V3LR=${V3LR:-5e-4}
V3RANK=${V3RANK:-8}
V31BASE="--cdp_learning_rate ${V3LR} --cdp_loss mse_l1 --cdp_l1_mu 0.5 --cdp_monitor mae"

run_variant () {
  local tag="$1"; shift
  if [ -n "$WANT" ]; then
    case " $WANT " in *" $tag "*) ;; *) return 0 ;; esac
  fi
  echo "===== [$(date '+%T')] v3.1 variant ${tag} on ${DS} h${PL} (ep=${EP} pat=${PAT}) ====="
  local log="/d/fsdownload/v31_${DS}_${PL}_${tag}.log"
  "$PY" -u run.py --is_training 1 $COMMON --pred_len ${PL} \
     --model_id ${DS}_v31_${PL}_${tag} --cdp_backbone_checkpoint "$BASE" "$@" > "$log" 2>&1
  "$PY" -u run.py --is_training 0 $COMMON --pred_len ${PL} \
     --model_id ${DS}_v31_${PL}_${tag} --cdp_backbone_checkpoint "$BASE" "$@" >> "$log" 2>&1
  local r=$(grep -a -oE "mse:[0-9.]+, mae:[0-9.]+" "$log" | tail -1)
  echo "${DS} h${PL} ${tag}  ${r}" >> $RES
  echo "--> ${tag}: ${r}"
}

case "$STAGE" in
  screen)
    run_variant c1_v2ref  --cdp_gain 0 --cdp_out_affine 0 --cdp_noise_all 0 --cdp_stage_w 0 --cdp_loss mse --cdp_monitor mse --cdp_learning_rate 1e-4
    run_variant c1_full   ${V31BASE}
    run_variant c1_ha     ${V31BASE} --cdp_unfreeze_head 1
    run_variant c1_rank   ${V31BASE} --cdp_chan_rank ${V3RANK}
    run_variant c1_lsew   ${V31BASE} --backbone_mode learn --cdp_unfreeze_pdm 1
    run_variant c1_harank ${V31BASE} --cdp_unfreeze_head 1 --cdp_chan_rank ${V3RANK}
    run_variant c1_norm   ${V31BASE} --cdp_unfreeze_norm 1
    run_variant c1_normha ${V31BASE} --cdp_unfreeze_norm 1 --cdp_unfreeze_head 1
    run_variant c1_mul    ${V31BASE} --cdp_chan_rank_mul ${V3RANK}
    run_variant c1_all    ${V31BASE} --cdp_unfreeze_head 1 --cdp_unfreeze_norm 1 --cdp_chan_rank ${V3RANK} --cdp_chan_rank_mul ${V3RANK}
    ;;
  final)
    run_variant c1_full   ${V31BASE}
    run_variant c1_ha     ${V31BASE} --cdp_unfreeze_head 1
    run_variant c1_rank   ${V31BASE} --cdp_chan_rank ${V3RANK}
    run_variant c1_harank ${V31BASE} --cdp_unfreeze_head 1 --cdp_chan_rank ${V3RANK}
    run_variant c1_lsew   ${V31BASE} --backbone_mode learn --cdp_unfreeze_pdm 1
    run_variant c1_norm   ${V31BASE} --cdp_unfreeze_norm 1
    run_variant c1_normha ${V31BASE} --cdp_unfreeze_norm 1 --cdp_unfreeze_head 1
    run_variant c1_mul    ${V31BASE} --cdp_chan_rank_mul ${V3RANK}
    run_variant c1_all    ${V31BASE} --cdp_unfreeze_head 1 --cdp_unfreeze_norm 1 --cdp_chan_rank ${V3RANK} --cdp_chan_rank_mul ${V3RANK}
    ;;
  p2)
    # 第二轮：训练预算 + 部分骨干微调（以 c1_all 为基线做增量消融）
    V31ALL="${V31BASE} --cdp_unfreeze_head 1 --cdp_unfreeze_norm 1 --cdp_chan_rank ${V3RANK} --cdp_chan_rank_mul ${V3RANK}"
    run_variant c2_all    ${V31ALL}
    run_variant c2_hlr30  ${V31ALL} --cdp_head_lr_ratio 0.3
    run_variant c2_blk1   ${V31ALL} --cdp_unfreeze_blocks 1
    run_variant c2_blk1h  ${V31ALL} --cdp_unfreeze_blocks 1 --cdp_head_lr_ratio 0.3
    run_variant c2_blk2h  ${V31ALL} --cdp_unfreeze_blocks 2 --cdp_head_lr_ratio 0.3
    run_variant c2_blk3h  ${V31ALL} --cdp_unfreeze_blocks 3 --cdp_head_lr_ratio 0.3
    ;;
esac
echo "V31 DONE ${DS} h${PL} stage=${STAGE}"
