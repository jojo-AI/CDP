#!/bin/bash
# CDP v3 变体筛选脚本
# 用法: bash run_cdp_v3_screen.sh <dataset> <horizon> <stage:screen|final>
#   dataset: ETTh1 | ECL | Traffic | weather
#   stage=screen → 12 epochs / patience 4（快筛）; stage=final → 30 epochs / patience 8（正式）
PY="/c/Users/Administrator/.workbuddy/binaries/python/envs/default/Scripts/python.exe"
cd "/d/fsdownload/TimeMixer"
RES="/d/fsdownload/screen_results.txt"

DS=$1; PL=$2; STAGE=${3:-screen}

case "$DS" in
  ETTh1)   DSARGS="--root_path ./dataset/ETT-small/ --data_path ETTh1.csv --data ETTh1 --enc_in 7 --dec_in 7 --c_out 7 --e_layers 2 --factor 1 --d_model 16 --d_ff 32 --batch_size 128" ;;
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

if [ "$STAGE" = "screen" ]; then EP=12; PAT=4; else EP=${EP_OVR:-30}; PAT=${PAT_OVR:-8}; fi

COMMON="--task_name long_term_forecast --model TimeMixer_CDP --features M --seq_len 96 --label_len 0 \
  --d_layers 1 --des CDPv3 --itr 1 --learning_rate 0.01 --num_workers 0 \
  --down_sampling_layers 3 --down_sampling_method avg --down_sampling_window 2 \
  --cdp_M 4 --cdp_S 16 --cdp_err_decay 0.9 --cdp_lambda_t 0.01 \
  --train_epochs ${EP} --patience ${PAT} --num_workers 0 ${DSARGS}"

run_variant () {
  local tag="$1"; shift
  echo "===== [$(date '+%T')] variant ${tag} on ${DS} h${PL} ====="
  local log="/d/fsdownload/screen_${DS}_${PL}_${tag}.log"
  "$PY" -u run.py --is_training 1 $COMMON --pred_len ${PL} \
     --model_id ${DS}_v3_${PL}_${tag} --cdp_backbone_checkpoint "$BASE" "$@" > "$log" 2>&1
  "$PY" -u run.py --is_training 0 $COMMON --pred_len ${PL} \
     --model_id ${DS}_v3_${PL}_${tag} --cdp_backbone_checkpoint "$BASE" "$@" >> "$log" 2>&1
  local r=$(grep -a -oE "mse:[0-9.]+, mae:[0-9.]+" "$log" | tail -1)
  echo "${DS} h${PL} ${tag}  ${r}" >> $RES
  echo "--> ${tag}: ${r}"
}

case "$STAGE" in
  screen)
    run_variant v2ref   --cdp_gain 0 --cdp_out_affine 0 --cdp_noise_all 0 --cdp_loss mse --cdp_monitor mse --cdp_learning_rate 1e-4
    run_variant v3mse   --cdp_learning_rate 1e-4 --cdp_loss mse --cdp_monitor mse
    run_variant v3l1mae --cdp_learning_rate 1e-4 --cdp_loss mse_l1 --cdp_l1_mu 0.5 --cdp_monitor mae
    run_variant v3lr    --cdp_learning_rate 5e-4 --cdp_loss mse_l1 --cdp_l1_mu 0.5 --cdp_monitor mae
    run_variant v3lr_ema --cdp_learning_rate 5e-4 --cdp_loss mse_l1 --cdp_l1_mu 0.5 --cdp_monitor mae --cdp_ema_decay 0.9
    run_variant v3ha    --cdp_learning_rate 5e-4 --cdp_loss mse_l1 --cdp_l1_mu 0.5 --cdp_monitor mae --cdp_ema_decay 0.9 --cdp_unfreeze_head 1
    ;;
  final)
    run_variant v3best     --cdp_learning_rate ${V3LR:-5e-4} --cdp_loss mse_l1 --cdp_l1_mu 0.5 --cdp_monitor mae --cdp_ema_decay 0.9
    run_variant v3best_rank --cdp_learning_rate ${V3LR:-5e-4} --cdp_loss mse_l1 --cdp_l1_mu 0.5 --cdp_monitor mae --cdp_ema_decay 0.9 --cdp_chan_rank ${V3RANK:-8}
    run_variant v3best_ha   --cdp_learning_rate ${V3LR:-5e-4} --cdp_loss mse_l1 --cdp_l1_mu 0.5 --cdp_monitor mae --cdp_ema_decay 0.9 --cdp_unfreeze_head 1
    ;;
esac
echo "SCREEN DONE ${DS} h${PL} stage=${STAGE}"
