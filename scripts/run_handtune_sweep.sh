#!/bin/bash
# 手调「季节/趋势」权重 vs 官方 clean 的对照评估（仅推理，不重训）
#   clean     : out = season + trend            （官方/论文基线）
#   handtuned : out = w1·season + w2·trend      （w1/w2 按 数据集×horizon 查作者手调表）
#   handtuned_off : 连作者的 +0.3/+0.05 偏置一起复现（精确复现作者原始写法）
# 用法: bash run_handtune_sweep.sh <dataset> [horizons...]
PY="/c/Users/Administrator/.workbuddy/binaries/python/envs/default/Scripts/python.exe"
cd "/d/fsdownload/TimeMixer" || exit 1
RES="/d/fsdownload/handtune_results.txt"
DS=$1; shift
HZ=${@:-"96 192 336 720"}

case "$DS" in
  ETTh1)   DATAKV="ETTh1";  DSARGS="--root_path ./dataset/ETT-small/ --data_path ETTh1.csv --data ETTh1 --enc_in 7 --dec_in 7 --c_out 7 --e_layers 2 --factor 1 --d_model 16 --d_ff 32 --batch_size 128" ;;
  ETTh2)   DATAKV="ETTh2";  DSARGS="--root_path ./dataset/ETT-small/ --data_path ETTh2.csv --data ETTh2 --enc_in 7 --dec_in 7 --c_out 7 --e_layers 2 --factor 1 --d_model 16 --d_ff 32 --batch_size 128" ;;
  ETTm2)   DATAKV="ETTm2";  DSARGS="--root_path ./dataset/ETT-small/ --data_path ETTm2.csv --data ETTm2 --enc_in 7 --dec_in 7 --c_out 7 --e_layers 3 --factor 3 --d_model 16 --d_ff 32 --batch_size 128" ;;
  ECL)     DATAKV="custom"; DSARGS="--root_path ./dataset/electricity/ --data_path electricity.csv --data custom --enc_in 321 --dec_in 321 --c_out 321 --e_layers 3 --factor 3 --d_model 16 --d_ff 32 --batch_size 32" ;;
  Traffic) DATAKV="custom"; DSARGS="--root_path ./dataset/traffic/ --data_path traffic.csv --data custom --enc_in 862 --dec_in 862 --c_out 862 --e_layers 3 --factor 3 --d_model 32 --d_ff 64 --batch_size 16" ;;
  weather) DATAKV="custom"; DSARGS="--root_path ./dataset/weather/ --data_path weather.csv --data custom --enc_in 21 --dec_in 21 --c_out 21 --e_layers 3 --factor 3 --d_model 16 --d_ff 32 --batch_size 128" ;;
  *) echo "unknown dataset $DS"; exit 1 ;;
esac

for PL in $HZ; do
  d=$(ls -d ./checkpoints/* 2>/dev/null | grep -F "_none_TimeMixer_${DATAKV}_sl96_pl${PL}_" \
      | grep -F "long_term_forecast_${DS}_" | grep -v CDP | head -1)
  if [ -z "$d" ]; then echo "${DS} h${PL}: NO BACKBONE CKPT, skip" | tee -a $RES; continue; fi
  want=$(basename "$d" | tr -d '\047')
  if [ "$(basename "$d")" != "$want" ]; then
    ln -sfn "$(basename "$d")" "./checkpoints/$want" 2>/dev/null
  fi
  # 注意 model_id 必须与 checkpoint 的标准名一致（des=Exp），否则找不到 ckpt
  MID="${DS}_96_${PL}"
  for MODE in handtuned handtuned_off; do
    log="/d/fsdownload/ht_${DS}_${PL}_${MODE}.log"
    EXTRA=""
    [ "$MODE" = "handtuned_off" ] && EXTRA="--handtune_offset 1"
    "$PY" -u run.py --task_name long_term_forecast --is_training 0 \
      --model_id "${MID}" --model TimeMixer --features M --seq_len 96 --label_len 0 \
      --pred_len ${PL} --d_layers 1 --des Exp --itr 1 --learning_rate 0.01 --num_workers 0 \
      --down_sampling_layers 3 --down_sampling_method avg --down_sampling_window 2 \
      --backbone_mode handtuned $EXTRA $DSARGS > "$log" 2>&1
    r=$(grep -a -oE "mse:[0-9.]+, mae:[0-9.]+" "$log" | tail -1)
    echo "${DS} h${PL} ${MODE}  ${r}" | tee -a $RES
  done
done
echo "HANDTUNE SWEEP DONE ${DS}"
