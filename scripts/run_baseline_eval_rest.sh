#!/bin/bash
# 补跑 Traffic baseline(未加 CDP) 在 192/336/720 上的评估，用于与 CDP 结果做配对对比
PY="/c/Users/Administrator/.workbuddy/binaries/python/envs/default/Scripts/python.exe"
cd "/d/fsdownload/TimeMixer"

for pred_len in 192 336 720
do
  base_ckpt=$(ls -d ./checkpoints/long_term_forecast_Traffic_96_${pred_len}_none_TimeMixer_* 2>/dev/null | head -1)
  echo "===== BASELINE EVAL Traffic pred_len=${pred_len} ckpt=${base_ckpt} ====="
  "$PY" -u run.py --task_name long_term_forecast --is_training 0 \
    --root_path ./dataset/traffic/ --data_path traffic.csv \
    --model_id Traffic_96_${pred_len} --model TimeMixer \
    --data custom --features M --seq_len 96 --label_len 0 --pred_len ${pred_len} \
    --e_layers 3 --d_layers 1 --factor 3 --enc_in 862 --dec_in 862 --c_out 862 \
    --des Exp --itr 1 --d_model 32 --d_ff 64 --batch_size 8 --learning_rate 0.01 \
    --num_workers 0 --down_sampling_layers 3 --down_sampling_method avg --down_sampling_window 2 \
    > "/d/fsdownload/baseline_traffic${pred_len}.log" 2>&1
  tail -2 "/d/fsdownload/baseline_traffic${pred_len}.log"
done
echo "BASELINE EVAL DONE"
