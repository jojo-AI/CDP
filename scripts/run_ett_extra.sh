#!/bin/bash
# 新扩展数据集（ETTh2 / ETTm1 / ETTm2）完整流水线
#   这三个数据集没有作者发布的预训练骨干，所以基线必须自己从头训。
#   每个 (数据集 × horizon) 依次跑：
#     1) clean 基线骨干（论文 TimeMixer 基线）：训练 + 测试集评估  -> baseline_results.txt
#     2) Stage-1：backbone_mode=dynamic + 纯 L1 重训骨架          -> stage1_results.txt
#     3) Stage-2：CDP 校准，cal5（当前最优）                      -> v4_results.txt
#     4) Stage-2：CDP 校准，cal1（对照，供「谁好谁用」选优）      -> v4_results.txt
#
# 注：ETTh2/ETTm1/ETTm2 的手调表初值都是 (1,1)，与「官方等价点」一致，
#     所以不需要再跑一遍 --handtune_init 0 的 hi0 对照（对这三个数据集是同一件事）。
#
# 用法: bash run_ett_extra.sh <dataset> [horizons...]
#   EP_BASE/PAT_BASE  基线训练预算（默认 30/8）
#   EP_ST1/PAT_ST1    Stage-1 预算（默认 30/8）
#   EP_ST2/PAT_ST2    Stage-2 预算（默认 20/12）
#   ONLY=st1|base|st2 只跑指定阶段
PY="/c/Users/Administrator/.workbuddy/binaries/python/envs/default/Scripts/python.exe"
cd "/d/fsdownload/TimeMixer" || exit 1
RES_BASE=/d/fsdownload/baseline_results.txt
RES_ST1=/d/fsdownload/stage1_results.txt
RES=/d/fsdownload/v4_results.txt

DS=$1; shift
HZ=${@:-"96 192 336 720"}
EP_BASE=${EP_BASE:-30}; PAT_BASE=${PAT_BASE:-8}
EP_ST1=${EP_ST1:-30};   PAT_ST1=${PAT_ST1:-8}
EP_ST2=${EP_ST2:-20};   PAT_ST2=${PAT_ST2:-12}

case "$DS" in
  ETTh2)  DATAKV="ETTh2"; DSARGS="--root_path ./dataset/ETT-small/ --data_path ETTh2.csv --data ETTh2 --enc_in 7 --dec_in 7 --c_out 7 --e_layers 2 --factor 1 --d_model 16 --d_ff 32 --batch_size 128" ;;
  ETTm1)  DATAKV="ETTm1"; DSARGS="--root_path ./dataset/ETT-small/ --data_path ETTm1.csv --data ETTm1 --enc_in 7 --dec_in 7 --c_out 7 --e_layers 2 --factor 1 --d_model 16 --d_ff 32 --batch_size 128" ;;
  ETTm2)  DATAKV="ETTm2"; DSARGS="--root_path ./dataset/ETT-small/ --data_path ETTm2.csv --data ETTm2 --enc_in 7 --dec_in 7 --c_out 7 --e_layers 2 --factor 1 --d_model 32 --d_ff 32 --batch_size 128" ;;
  *) echo "run_ett_extra.sh 只用于 ETTh2/ETTm1/ETTm2（收到: $DS）"; exit 1 ;;
esac

pick_st1_ck () {
  # 取该 horizon 测试 MSE 最低的 Stage-1 骨干目录
  ls -d ./checkpoints/long_term_forecast_${DS}_st1_${1}_dynamic_l10.0_* 2>/dev/null | head -1
}

for PL in $HZ; do

  ### ---------- 1) clean 基线：训练（若已有则跳过） + 评估 ----------
  if [ "${ONLY:-all}" != "st1" ] && [ "${ONLY:-all}" != "st2" ]; then
    HAS_CK=$(ls -d ./checkpoints/long_term_forecast_${DS}_96_${PL}_none_TimeMixer_* 2>/dev/null | grep -v CDP | head -1)
    if [ -z "$HAS_CK" ]; then
      echo "===== [$(date '+%T')] BASELINE-TRAIN ${DS} h${PL} (clean, ep=${EP_BASE} pat=${PAT_BASE}) ====="
      "$PY" -u run.py --task_name long_term_forecast --is_training 1 \
        --model_id "${DS}_96_${PL}" --model TimeMixer --features M --seq_len 96 --label_len 0 \
        --pred_len ${PL} --d_layers 1 --des Exp --itr 1 --learning_rate 0.01 --num_workers 0 \
        --down_sampling_layers 3 --down_sampling_method avg --down_sampling_window 2 \
        --train_epochs ${EP_BASE} --patience ${PAT_BASE} --backbone_mode clean --train_loss mse \
        $DSARGS > "/d/fsdownload/base_${DS}_${PL}.log" 2>&1
    else
      echo "===== [skip] baseline ckpt exists ${DS} h${PL} ====="
    fi
    if ! grep -aqE "(^| )${DS} h${PL} baseline" $RES_BASE; then
      echo "===== [$(date '+%T')] BASELINE-EVAL ${DS} h${PL} ====="
      bash scripts/run_baseline_eval_all.sh ${DS} ${PL} 2>&1 | tail -2
    fi
  fi

  ### ---------- 2) Stage-1：dynamic + 纯 L1 ----------
  if [ "${ONLY:-all}" != "base" ] && [ "${ONLY:-all}" != "st2" ]; then
    if ! grep -aqE "(^| )${DS} h${PL} st1_dynamic_l10.0 " $RES_ST1; then
      echo "===== [$(date '+%T')] STAGE1 ${DS} h${PL} (dynamic + L1) ====="
      TLOSS=l1 TL1MU=0.0 EP_OVR=${EP_ST1} PAT_OVR=${PAT_ST1} \
        bash scripts/run_stage1_st.sh ${DS} ${PL} dynamic 2>&1 | tail -1
    else
      echo "===== [skip] stage1 exists ${DS} h${PL} ====="
    fi
  fi

  ### ---------- 3/4) Stage-2：cal5（主）/ cal1（对照） ----------
  if [ "${ONLY:-all}" != "base" ] && [ "${ONLY:-all}" != "st1" ]; then
    CK=$(pick_st1_ck ${PL})
    if [ -z "$CK" ]; then echo "!! 找不到 Stage-1 骨干: ${DS} h${PL}"; continue; fi
    echo "[pick] backbone = $CK"
    for CALR in 5 1; do
      TAG="cdp_cal${CALR}_${DS}_${PL}"
      if grep -aqE "(^| )${DS} h${PL} ${TAG}[^ ]* " $RES; then echo "===== [skip] ${TAG} ====="; continue; fi
      echo "===== [$(date '+%T')] STAGE2 ${TAG} (cal_lr_ratio=${CALR}) ====="
      BASE_OVR="$CK" MIDP="_c${CALR}_" EP_OVR=${EP_ST2} PAT_OVR=${PAT_ST2} TAG_OVR="$TAG" \
        ST2_EXTRA="--backbone_mode dynamic --cdp_unfreeze_pdm 1 --cdp_st_lr_ratio 0.2 --cdp_dyncal 1 \
                   --cdp_loss l1 --cdp_monitor mae --cdp_cal_lr_ratio ${CALR}" \
        bash scripts/run_cdp_v4.sh ${DS} ${PL} custom 2>&1 | tail -1
    done
  fi

done
echo "ETT-EXTRA DONE: ${DS} ${HZ}"
