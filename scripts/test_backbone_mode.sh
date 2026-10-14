#!/usr/bin/env bash
# 对照实验：同一 Traffic-96 checkpoint，仅切换骨干季节/趋势合成方式
# clean     = 上游官方 out = season + trend
# handtuned = 作者遗留手调 out = 0.75*season + 0.3 + 0.93*trend + 0.05
PY="/c/Users/Administrator/.workbuddy/binaries/python/envs/default/Scripts/python.exe"
RES="/d/fsdownload/backbone_mode_test.txt"
cd "D:/fsdownload/TimeMixer" || exit 1
: > "$RES"

for MODE in clean handtuned; do
  for PL in 96 192; do
    log="/d/fsdownload/bmode_${MODE}_h${PL}.log"
    "$PY" -u run.py --task_name long_term_forecast --is_training 0 \
      --root_path ./dataset/traffic/ --data_path traffic.csv \
      --model_id Traffic_96_${PL} --model TimeMixer --data custom --features M \
      --seq_len 96 --label_len 0 --pred_len ${PL} --e_layers 3 --d_layers 1 --factor 3 \
      --enc_in 862 --dec_in 862 --c_out 862 --des Exp --itr 1 \
      --d_model 32 --d_ff 64 --batch_size 8 --learning_rate 0.01 --num_workers 0 \
      --down_sampling_layers 3 --down_sampling_method avg --down_sampling_window 2 \
      --backbone_mode ${MODE} > "$log" 2>&1
    r=$(grep -a -oE "mse:[0-9.]+, mae:[0-9.]+" "$log" | tail -1)
    echo "backbone=${MODE}  Traffic h${PL}  ${r}" | tee -a "$RES"
  done
done
echo "=== DONE ==="
cat "$RES"
