#!/bin/bash
# Stage-1（dynamic + 纯L1）但**不使用作者手调表初始化**（--handtune_init 0，从官方等价点 1,1,0 出发）
# 用法: bash run_st1_hi0.sh <dataset> <horizon>
PY="/c/Users/Administrator/.workbuddy/binaries/python/envs/default/Scripts/python.exe"
cd "/d/fsdownload/TimeMixer" || exit 1
DS=$1; PL=$2
case "$DS" in
  ETTh1)   DSARGS="--root_path ./dataset/ETT-small/ --data_path ETTh1.csv --data ETTh1 --enc_in 7 --dec_in 7 --c_out 7 --e_layers 2 --factor 1 --d_model 16 --d_ff 32 --batch_size 128" ;;
  weather) DSARGS="--root_path ./dataset/weather/ --data_path weather.csv --data custom --enc_in 21 --dec_in 21 --c_out 21 --e_layers 3 --factor 3 --d_model 16 --d_ff 32 --batch_size 128" ;;
  ETTh2)   DSARGS="--root_path ./dataset/ETT-small/ --data_path ETTh2.csv --data ETTh2 --enc_in 7 --dec_in 7 --c_out 7 --e_layers 2 --factor 1 --d_model 16 --d_ff 32 --batch_size 128" ;;
  ETTm1)   DSARGS="--root_path ./dataset/ETT-small/ --data_path ETTm1.csv --data ETTm1 --enc_in 7 --dec_in 7 --c_out 7 --e_layers 2 --factor 1 --d_model 16 --d_ff 32 --batch_size 128" ;;
  ETTm2)   DSARGS="--root_path ./dataset/ETT-small/ --data_path ETTm2.csv --data ETTm2 --enc_in 7 --dec_in 7 --c_out 7 --e_layers 2 --factor 1 --d_model 32 --d_ff 32 --batch_size 128" ;;
  *) echo "unknown $DS"; exit 1 ;;
esac
EP=${EP_OVR:-30}; PAT=${PAT_OVR:-8}
MID="${DS}_st1_${PL}_dyn0_l10.0"
log="/d/fsdownload/st1_${DS}_${PL}_dyn0_l10.0.log"
COMMON="--task_name long_term_forecast --model TimeMixer --features M --seq_len 96 --label_len 0 \
  --d_layers 1 --des ST1 --itr 1 --learning_rate 0.01 --num_workers 0 \
  --train_loss l1 --train_l1_mu 0.0 --handtune_init 0 \
  --down_sampling_layers 3 --down_sampling_method avg --down_sampling_window 2 \
  --train_epochs ${EP} --patience ${PAT} ${DSARGS}"
echo "===== [$(date '+%T')] stage1-hi0 on ${DS} h${PL} ====="
"$PY" -u run.py --is_training 1 $COMMON --pred_len ${PL} --model_id "${MID}" --backbone_mode dynamic > "$log" 2>&1
"$PY" -u run.py --is_training 0 $COMMON --pred_len ${PL} --model_id "${MID}" --backbone_mode dynamic >> "$log" 2>&1
r=$(grep -a -oE "mse:[0-9.]+, mae:[0-9.]+" "$log" | tail -1)
echo "${DS} h${PL} st1_dynamic_l10.0_hi0  ${r}" | tee -a /d/fsdownload/stage1_results.txt
