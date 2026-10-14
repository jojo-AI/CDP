#!/bin/bash
# 补跑各数据集「冻结骨干、无 CDP」的基线评估（配对参照）
# 用法: bash run_baseline_eval_all.sh <dataset> [horizons...]
PY="/c/Users/Administrator/.workbuddy/binaries/python/envs/default/Scripts/python.exe"
cd "/d/fsdownload/TimeMixer"
RES="/d/fsdownload/baseline_results.txt"
DS=$1; shift
HZ=${@:-"96 192 336 720"}

case "$DS" in
  ETTh1)   DATAKV="ETTh1";  DSARGS="--root_path ./dataset/ETT-small/ --data_path ETTh1.csv --data ETTh1 --enc_in 7 --dec_in 7 --c_out 7 --e_layers 2 --factor 1 --d_model 16 --d_ff 32 --batch_size 128" ;;
  ETTh2)   DATAKV="ETTh2";  DSARGS="--root_path ./dataset/ETT-small/ --data_path ETTh2.csv --data ETTh2 --enc_in 7 --dec_in 7 --c_out 7 --e_layers 2 --factor 1 --d_model 16 --d_ff 32 --batch_size 128" ;;
  ETTm1)   DATAKV="ETTm1";  DSARGS="--root_path ./dataset/ETT-small/ --data_path ETTm1.csv --data ETTm1 --enc_in 7 --dec_in 7 --c_out 7 --e_layers 2 --factor 1 --d_model 16 --d_ff 32 --batch_size 128" ;;
  ETTm2)   DATAKV="ETTm2"; DSARGS="--root_path ./dataset/ETT-small/ --data_path ETTm2.csv --data ETTm2 --enc_in 7 --dec_in 7 --c_out 7 --e_layers 2 --factor 1 --d_model 32 --d_ff 32 --batch_size 128" ;;
  ECL)     DATAKV="custom"; DSARGS="--root_path ./dataset/electricity/ --data_path electricity.csv --data custom --enc_in 321 --dec_in 321 --c_out 321 --e_layers 3 --factor 3 --d_model 16 --d_ff 32 --batch_size 32" ;;
  Traffic) DATAKV="custom"; DSARGS="--root_path ./dataset/traffic/ --data_path traffic.csv --data custom --enc_in 862 --dec_in 862 --c_out 862 --e_layers 3 --factor 3 --d_model 32 --d_ff 64 --batch_size 16" ;;
  weather) DATAKV="custom"; DSARGS="--root_path ./dataset/weather/ --data_path weather.csv --data custom --enc_in 21 --dec_in 21 --c_out 21 --e_layers 3 --factor 3 --d_model 16 --d_ff 32 --batch_size 128" ;;
  *) echo "unknown dataset $DS"; exit 1 ;;
esac

for PL in $HZ; do
  # 从实际 checkpoint 目录名反推 model_id / des（不同数据集命名带不带引号不一致）
  # 从实际 checkpoint 目录名反推（去引号）→ 软链接成 des=Exp 的标准名，避开引号匹配问题
  d=$(ls -d ./checkpoints/* 2>/dev/null | grep -F "_none_TimeMixer_${DATAKV}_sl96_pl${PL}_" \
      | grep -F "long_term_forecast_${DS}_" | grep -v CDP | head -1)
  if [ -z "$d" ]; then echo "${DS} h${PL}: NO BACKBONE CKPT, skip" | tee -a $RES; continue; fi
  want=$(basename "$d" | tr -d '\047')
  if [ "$(basename "$d")" != "$want" ]; then
    ln -sfn "$(basename "$d")" "./checkpoints/$want"   # 相对软链接
  fi
  log="/d/fsdownload/baseline_${DS}_${PL}.log"
  echo "--> ${DS} h${PL}: ckpt=$d -> link=$want"
  "$PY" -u run.py --task_name long_term_forecast --is_training 0 \
    --model_id "${DS}_96_${PL}" --model TimeMixer --features M --seq_len 96 --label_len 0 \
    --pred_len ${PL} --d_layers 1 --des Exp --itr 1 --learning_rate 0.01 --num_workers 0 \
    --down_sampling_layers 3 --down_sampling_method avg --down_sampling_window 2 \
    $DSARGS > "$log" 2>&1
  r=$(grep -a -oE "mse:[0-9.]+, mae:[0-9.]+" "$log" | tail -1)
  echo "${DS} h${PL} baseline  ${r}" >> $RES
  echo "${DS} h${PL}: ${r}"
done
echo "BASELINE EVAL DONE ${DS}"
