#!/bin/bash
# 下一阶段：① 抢救 Traffic-336 v2 测试结果 ② 补三个数据集基线 ③ Traffic-96 v3 最优配置训练
PY="/c/Users/Administrator/.workbuddy/binaries/python/envs/default/Scripts/python.exe"
cd "/d/fsdownload/TimeMixer"

echo "===== [1] Traffic-336 v2 checkpoint 测试 ====="
"$PY" -u run.py --task_name long_term_forecast --is_training 0 \
  --root_path ./dataset/traffic/ --data_path traffic.csv --model TimeMixer_CDP \
  --data custom --features M --seq_len 96 --label_len 0 --pred_len 336 \
  --e_layers 3 --d_layers 1 --factor 3 --enc_in 862 --dec_in 862 --c_out 862 \
  --des Exp --itr 1 --d_model 32 --d_ff 64 --batch_size 16 --learning_rate 0.01 \
  --num_workers 0 --down_sampling_layers 3 --down_sampling_method avg --down_sampling_window 2 \
  --cdp_M 4 --cdp_S 16 --cdp_err_decay 0.9 --cdp_lambda_t 0.01 --cdp_learning_rate 1e-4 \
  --cdp_gain 0 --cdp_out_affine 0 --cdp_noise_all 0 \
  --model_id Traffic_CDP_96_336 >> /d/fsdownload/cdp_traffic_h336.log 2>&1
echo "Traffic h336 v2: $(grep -a -oE 'mse:[0-9.]+, mae:[0-9.]+' /d/fsdownload/cdp_traffic_h336.log | tail -1)"

echo "===== [2] 补基线评估（ETTh1 / ECL / weather） ====="
bash scripts/run_baseline_eval_all.sh ETTh1
bash scripts/run_baseline_eval_all.sh ECL
bash scripts/run_baseline_eval_all.sh weather

echo "===== [3] Traffic-96 v3 最优配置（mse_l1 + monitor mae + head adaptation）====="
EP_OVR=20 PAT_OVR=6 bash scripts/run_cdp_v3_screen.sh Traffic 96 final
echo "NEXT PHASE DONE"
