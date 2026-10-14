#!/bin/bash
# v4.1 长/中 horizon 探针：验证 (a) wd=0 修复 (b) 校准参数独立lr (c) 步级加权损失
# 用已有的 stage1 骨干，只重跑 Stage-2。CK_OVR 可指定骨干目录名。
PY="/c/Users/Administrator/.workbuddy/binaries/python/envs/default/Scripts/python.exe"
cd "/d/fsdownload/TimeMixer" || exit 1
DS=$1; PL=$2
case "$DS" in
  ETTh1)   DSARGS="--root_path ./dataset/ETT-small/ --data_path ETTh1.csv --data ETTh1 --enc_in 7 --dec_in 7 --c_out 7 --e_layers 2 --factor 1 --d_model 16 --d_ff 32 --batch_size 128" ;;
  weather) DSARGS="--root_path ./dataset/weather/ --data_path weather.csv --data custom --enc_in 21 --dec_in 21 --c_out 21 --e_layers 3 --factor 3 --d_model 16 --d_ff 32 --batch_size 128" ;;
  *) echo "unknown $DS"; exit 1 ;;
esac
CK=${CK_OVR:-$(ls -d ./checkpoints/long_term_forecast_${DS}_st1_${PL}_dynamic_l10.0_* 2>/dev/null | grep -v dyn0 | head -1)}
[ -z "$CK" ] && { echo "!! stage1 ckpt not found for $DS h$PL"; exit 1; }
echo "probe backbone = $CK"
COMMON="--task_name long_term_forecast --model TimeMixer_CDP --features M --seq_len 96 --label_len 0 \
  --d_layers 1 --des CDPv4 --itr 1 --num_workers 0 --train_epochs 20 --patience 12 \
  --down_sampling_layers 3 --down_sampling_method avg --down_sampling_window 2 \
  --cdp_backbone_checkpoint $CK ${DSARGS}"
WIN="--backbone_mode dynamic --cdp_unfreeze_pdm 1 --cdp_st_lr_ratio 0.2 --cdp_dyncal 1 \
     --cdp_loss l1 --cdp_monitor mae"
run_probe () {
  local tag="$1"; shift
  local log="/d/fsdownload/v4_${DS}_${PL}_lt_${tag}.log"
  echo "===== [$(date '+%T')] probe ${tag} on ${DS} h${PL} ====="
  "$PY" -u run.py --is_training 1 $COMMON --pred_len ${PL} \
    --model_id ${DS}_v4lt_${PL}_${tag} --cdp_learning_rate 5e-4 "$@" $WIN > "$log" 2>&1
  "$PY" -u run.py --is_training 0 $COMMON --pred_len ${PL} \
    --model_id ${DS}_v4lt_${PL}_${tag} --cdp_learning_rate 5e-4 "$@" $WIN >> "$log" 2>&1
  local r=$(grep -a -oE "mse:[0-9.]+, mae:[0-9.]+" "$log" | tail -1)
  echo "${DS} h${PL} lt_${tag}  ${r}" | tee -a /d/fsdownload/v4_results.txt
}
run_probe wd0
run_probe cal5  --cdp_cal_lr_ratio 5
run_probe sw    --cdp_step_weight 1.0
run_probe both  --cdp_cal_lr_ratio 5 --cdp_step_weight 1.0
echo "LT PROBE DONE ${DS} h${PL}"
