#!/bin/bash
# Stage-1：从零训练骨干，对比「季节/趋势合成权重」的四种取值方式
#   clean     : out = season + trend                       （官方实现 = 论文基线）
#   handtuned : out = w1·season + w2·trend                 （作者手调表，按 数据集×horizon 查）
#   learn     : out = w1_k·season + w2_k·trend + b_k       （逐分辨率可学习，手调表初值）
#   dynamic   : learn + 逐样本权重偏移（由 patch 分布统计预测）
#
# ⚠️ model_id 带 st1 前缀，避免覆盖作者原始预训练 checkpoint（那些是 clean 基线的来源）。
# 用法: bash run_stage1_st.sh <dataset> <horizon> [modes...]
PY="/c/Users/Administrator/.workbuddy/binaries/python/envs/default/Scripts/python.exe"
cd "/d/fsdownload/TimeMixer" || exit 1
RES="/d/fsdownload/stage1_results.txt"
DS=$1; PL=$2
shift 2 2>/dev/null
MODES=${*:-"clean handtuned learn dynamic"}

case "$DS" in
  ETTh1)   DSARGS="--root_path ./dataset/ETT-small/ --data_path ETTh1.csv --data ETTh1 --enc_in 7 --dec_in 7 --c_out 7 --e_layers 2 --factor 1 --d_model 16 --d_ff 32 --batch_size 128" ;;
  ETTh2)   DSARGS="--root_path ./dataset/ETT-small/ --data_path ETTh2.csv --data ETTh2 --enc_in 7 --dec_in 7 --c_out 7 --e_layers 2 --factor 1 --d_model 16 --d_ff 32 --batch_size 128" ;;
  ETTm1)   DSARGS="--root_path ./dataset/ETT-small/ --data_path ETTm1.csv --data ETTm1 --enc_in 7 --dec_in 7 --c_out 7 --e_layers 2 --factor 1 --d_model 16 --d_ff 32 --batch_size 128" ;;
  ETTm2)   DSARGS="--root_path ./dataset/ETT-small/ --data_path ETTm2.csv --data ETTm2 --enc_in 7 --dec_in 7 --c_out 7 --e_layers 2 --factor 1 --d_model 32 --d_ff 32 --batch_size 128" ;;
  ECL)     DSARGS="--root_path ./dataset/electricity/ --data_path electricity.csv --data custom --enc_in 321 --dec_in 321 --c_out 321 --e_layers 3 --factor 3 --d_model 16 --d_ff 32 --batch_size 32" ;;
  Traffic) DSARGS="--root_path ./dataset/traffic/ --data_path traffic.csv --data custom --enc_in 862 --dec_in 862 --c_out 862 --e_layers 3 --factor 3 --d_model 32 --d_ff 64 --batch_size 16" ;;
  weather) DSARGS="--root_path ./dataset/weather/ --data_path weather.csv --data custom --enc_in 21 --dec_in 21 --c_out 21 --e_layers 3 --factor 3 --d_model 16 --d_ff 32 --batch_size 128" ;;
  *) echo "unknown dataset $DS"; exit 1 ;;
esac

EP=${EP_OVR:-30}
PAT=${PAT_OVR:-8}
TLOSS=${TLOSS:-mse}          # mse | mse_l1
TL1MU=${TL1MU:-0.5}
TAGSUF=""
[ "$TLOSS" != "mse" ] && TAGSUF="_${TLOSS}${TL1MU}"
COMMON="--task_name long_term_forecast --model TimeMixer --features M --seq_len 96 --label_len 0 \
  --d_layers 1 --des ST1 --itr 1 --learning_rate 0.01 --num_workers 0 \
  --train_loss ${TLOSS} --train_l1_mu ${TL1MU} \
  --down_sampling_layers 3 --down_sampling_method avg --down_sampling_window 2 \
  --train_epochs ${EP} --patience ${PAT} ${DSARGS}"

for MODE in $MODES; do
  MID="${DS}_st1_${PL}_${MODE}${TAGSUF}"
  log="/d/fsdownload/st1_${DS}_${PL}_${MODE}${TAGSUF}.log"
  echo "===== [$(date '+%T')] stage1 ${MODE}${TAGSUF} on ${DS} h${PL} (ep=${EP} pat=${PAT} loss=${TLOSS}) ====="
  "$PY" -u run.py --is_training 1 $COMMON --pred_len ${PL} --model_id "${MID}" \
      --backbone_mode ${MODE} > "$log" 2>&1
  "$PY" -u run.py --is_training 0 $COMMON --pred_len ${PL} --model_id "${MID}" \
      --backbone_mode ${MODE} >> "$log" 2>&1
  r=$(grep -a -oE "mse:[0-9.]+, mae:[0-9.]+" "$log" | tail -1)
  echo "${DS} h${PL} st1_${MODE}${TAGSUF}  ${r}" | tee -a $RES
done
echo "STAGE1 DONE ${DS} h${PL}"
