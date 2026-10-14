#!/bin/bash
# CDP 挂机队列：等待当前 Traffic 队列完成后，依次跑 ECL -> ETTh1 -> Weather
# 每个 horizon：CDP stage-2 训练（骨干冻结）+ 测试，结果追加到 result_cdp.txt
PY="/c/Users/Administrator/.workbuddy/binaries/python/envs/default/Scripts/python.exe"
cd "/d/fsdownload/TimeMixer"

echo "[QUEUE] $(date '+%F %T') start, waiting for Traffic queue to finish..."

# 等待 Traffic 全部完成（result_cdp.txt 出现 Traffic_CDP_96_720 的测试结果），最多等 10 小时
for i in $(seq 1 120); do
  if grep -q "Traffic_CDP_96_720" result_cdp.txt 2>/dev/null; then
    echo "[QUEUE] $(date '+%F %T') Traffic done, starting next datasets."
    break
  fi
  sleep 300
done

# 基线 checkpoint 路径解析：先精确前缀 glob，失败再用宽松模式（兼容目录名里的引号字符）
find_ckpt() {
  local prefix="$1"          # e.g. ./checkpoints/long_term_forecast_ECL_96_96_none_TimeMixer
  local plen="$2"            # pred_len
  local cand
  cand=$(ls -d ${prefix}_* 2>/dev/null | head -1)
  if [ -z "$cand" ]; then
    local loose="${prefix/_96_${plen}_none_TimeMixer/_96*${plen}_none_TimeMixer}"
    cand=$(ls -d ${loose}_* 2>/dev/null | head -1)
  fi
  if [ -n "$cand" ] && [ -d "$cand" ]; then
    echo "$cand"
  fi
}

run_dataset() {
  local name="$1"; shift
  local common="$*"
  for pred_len in 96 192 336 720; do
    local base_ckpt
    base_ckpt=$(find_ckpt "./checkpoints/long_term_forecast_${name}_96_${pred_len}_none_TimeMixer" ${pred_len})
    if [ -z "$base_ckpt" ]; then
      echo "[QUEUE] $(date '+%F %T') WARN: baseline ckpt not found for ${name} h${pred_len}, SKIP"
      continue
    fi
    local log="/d/fsdownload/cdp_${name}_h${pred_len}.log"
    echo "[QUEUE] $(date '+%F %T') ===== CDP TRAIN ${name} pred_len=${pred_len} ckpt=${base_ckpt} ====="
    "$PY" -u run.py --is_training 1 $common --pred_len ${pred_len} \
      --model_id ${name}_CDP_96_${pred_len} \
      --cdp_backbone_checkpoint "$base_ckpt" > "$log" 2>&1
    echo "[QUEUE] $(date '+%F %T') ===== CDP TEST ${name} pred_len=${pred_len} ====="
    "$PY" -u run.py --is_training 0 $common --pred_len ${pred_len} \
      --model_id ${name}_CDP_96_${pred_len} >> "$log" 2>&1
    grep "CDP RESULT" "$log" | tail -1 >> result_cdp_queue.txt
  done
  echo "[QUEUE] $(date '+%F %T') DATASET ${name} DONE"
}

# ECL: 321 通道, dm16 df32 el3 factor3
run_dataset ECL --task_name long_term_forecast \
  --root_path ./dataset/electricity/ --data_path electricity.csv --model TimeMixer_CDP \
  --data custom --features M --seq_len 96 --label_len 0 \
  --e_layers 3 --d_layers 1 --factor 3 --enc_in 321 --dec_in 321 --c_out 321 \
  --des Exp --itr 1 --d_model 16 --d_ff 32 \
  --batch_size 32 --learning_rate 0.01 --num_workers 0 \
  --down_sampling_layers 3 --down_sampling_method avg --down_sampling_window 2 \
  --cdp_M 4 --cdp_S 16 --cdp_err_decay 0.9 --cdp_lambda_t 0.01 \
  --cdp_learning_rate 0.0001 --train_epochs 50 --patience 5

# ETTh1: 7 通道, dm16 df32 el2 factor1
run_dataset ETTh1 --task_name long_term_forecast \
  --root_path ./dataset/ETT-small/ --data_path ETTh1.csv --model TimeMixer_CDP \
  --data ETTh1 --features M --seq_len 96 --label_len 0 \
  --e_layers 2 --d_layers 1 --factor 1 --enc_in 7 --dec_in 7 --c_out 7 \
  --des Exp --itr 1 --d_model 16 --d_ff 32 \
  --batch_size 128 --learning_rate 0.01 --num_workers 0 \
  --down_sampling_layers 3 --down_sampling_method avg --down_sampling_window 2 \
  --cdp_M 4 --cdp_S 16 --cdp_err_decay 0.9 --cdp_lambda_t 0.01 \
  --cdp_learning_rate 0.0001 --train_epochs 50 --patience 5

# Weather: 21 通道, dm16 df32 el3 factor3
run_dataset weather --task_name long_term_forecast \
  --root_path ./dataset/weather/ --data_path weather.csv --model TimeMixer_CDP \
  --data custom --features M --seq_len 96 --label_len 0 \
  --e_layers 3 --d_layers 1 --factor 3 --enc_in 21 --dec_in 21 --c_out 21 \
  --des Exp --itr 1 --d_model 16 --d_ff 32 \
  --batch_size 128 --learning_rate 0.01 --num_workers 0 \
  --down_sampling_layers 3 --down_sampling_method avg --down_sampling_window 2 \
  --cdp_M 4 --cdp_S 16 --cdp_err_decay 0.9 --cdp_lambda_t 0.01 \
  --cdp_learning_rate 0.0001 --train_epochs 50 --patience 5

echo "[QUEUE] $(date '+%F %T') ALL QUEUE DONE"
