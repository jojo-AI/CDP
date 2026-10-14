#!/bin/bash
# CDP Stage-2 训练：Traffic 4 horizons 顺序执行
PY="/c/Users/Administrator/.workbuddy/binaries/python/envs/default/Scripts/python.exe"
cd "/d/fsdownload/TimeMixer"

COMMON="--task_name long_term_forecast --root_path ./dataset/traffic/ --data_path traffic.csv --model TimeMixer_CDP --data custom --features M --seq_len 96 --label_len 0 --e_layers 3 --d_layers 1 --factor 3 --enc_in 862 --dec_in 862 --c_out 862 --des Exp --itr 1 --d_model 32 --d_ff 64 --batch_size 16 --learning_rate 0.01 --num_workers 0 --down_sampling_layers 3 --down_sampling_method avg --down_sampling_window 2 --cdp_M 4 --cdp_S 16 --cdp_err_decay 0.9 --cdp_lambda_t 0.01 --cdp_learning_rate 0.0001 --train_epochs 50 --patience 5"

for pred_len in 96 192 336 720
do
  base_ckpt="./checkpoints/long_term_forecast_Traffic_96_${pred_len}_none_TimeMixer_custom_sl96_pl${pred_len}_dm32_nh8_el3_dl1_df64_fc3_ebtimeF_dtTrue_Exp_0"
  echo "===== CDP TRAIN Traffic pred_len=${pred_len} ====="
  "$PY" -u run.py --is_training 1 $COMMON --pred_len ${pred_len} \
    --model_id Traffic_CDP_96_${pred_len} \
    --cdp_backbone_checkpoint "$base_ckpt" \
    > "/d/fsdownload/cdp_traffic_h${pred_len}.log" 2>&1
  echo "===== CDP TEST Traffic pred_len=${pred_len} ====="
  "$PY" -u run.py --is_training 0 $COMMON --pred_len ${pred_len} \
    --model_id Traffic_CDP_96_${pred_len} \
    >> "/d/fsdownload/cdp_traffic_h${pred_len}.log" 2>&1
done
echo "ALL TRAFFIC CDP RUNS DONE"
