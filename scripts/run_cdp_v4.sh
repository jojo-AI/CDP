#!/bin/bash
# CDP v4：可学习 / 动态「季节-趋势」合成权重 + 架构优化
# ---------------------------------------------------------------
# 相对 v3.1 的新增/修正：
#   1) backbone_mode 变成四态：clean | handtuned | learn | dynamic
#        handtuned : w1/w2 按 (数据集, horizon) 查作者手调表（修掉了「全局硬编码 0.75/0.93」的 bug）
#        learn     : w1_k/w2_k/b_k 变成可学习参数（逐分辨率），从手调表初值出发
#        dynamic   : 在 learn 基础上，再由 per-patch 分布统计预测「逐样本」权重偏移
#   2) cdp_dyncal：输出级「输入自适应」校准，逐样本预测 (scale, bias)
#   3) 季节/趋势参数独立学习率组 --cdp_st_lr_ratio
#
# 用法: bash run_cdp_v4.sh <dataset> <horizon> <stage:screen|final|ablate> [variant...]
# 环境变量: EP_OVR PAT_OVR V4LR V4RANK V4STLR
PY="/c/Users/Administrator/.workbuddy/binaries/python/envs/default/Scripts/python.exe"
cd "/d/fsdownload/TimeMixer" || exit 1
RES="/d/fsdownload/v4_results.txt"

DS=$1; PL=$2; STAGE=${3:-screen}
shift 3 2>/dev/null
WANT="$*"

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

# BASE_OVR 可指定任意骨干 checkpoint 目录（例如 Stage-1 自己训练出来的 clean/dynamic 骨干）
if [ -n "$BASE_OVR" ]; then
  BASE="${BASE_OVR#./checkpoints/}"
  BASE="./checkpoints/$BASE"
else
  BASE=$(ls -d ./checkpoints/long_term_forecast_${DS}_96_${PL}_none_TimeMixer_* 2>/dev/null | grep -v CDP | head -1)
fi
if [ -z "$BASE" ]; then echo "!!! backbone ckpt not found for ${DS} h${PL}"; exit 1; fi
echo "backbone = $BASE"

if [ "$STAGE" = "screen" ]; then EP=${EP_OVR:-20}; PAT=${PAT_OVR:-6}; else EP=${EP_OVR:-30}; PAT=${PAT_OVR:-8}; fi

# 注意：骨干 checkpoint 是用 clean 混合练出来的，保持 --backbone_mode clean 才是公平基线
COMMON="--task_name long_term_forecast --model TimeMixer_CDP --features M --seq_len 96 --label_len 0 \
  --d_layers 1 --des CDPv4 --itr 1 --learning_rate 0.01 --num_workers 0 \
  --down_sampling_layers 3 --down_sampling_method avg --down_sampling_window 2 \
  --cdp_stage_w 1 --cdp_M 4 --cdp_S 16 --cdp_err_decay 0.9 --cdp_lambda_t 0.01 \
  --train_epochs ${EP} --patience ${PAT} ${DSARGS}"

V4LR=${V4LR:-5e-4}
V4RANK=${V4RANK:-8}
V4STLR=${V4STLR:-0.2}
MIDP=${MIDP:-}               # model_id 前缀后缀（用于区分不同骨干来源，避免 checkpoint 撞名）
# v3.1 里最好的那套「架构优化」配方（作为参照基线）
V4BASE="--cdp_learning_rate ${V4LR} --cdp_loss mse_l1 --cdp_l1_mu 0.5 --cdp_monitor mae \
  --cdp_unfreeze_head 1 --cdp_unfreeze_norm 1 --cdp_chan_rank ${V4RANK} --cdp_chan_rank_mul ${V4RANK}"

run_variant () {
  local tag="$1"; shift
  if [ -n "$WANT" ]; then
    case " $WANT " in *" $tag "*) ;; *) return 0 ;; esac
  fi
  echo "===== [$(date '+%T')] v4 ${tag} on ${DS} h${PL} (ep=${EP} pat=${PAT}) ====="
  local log="/d/fsdownload/v4_${DS}_${PL}${MIDP}_${tag}.log"
  "$PY" -u run.py --is_training 1 $COMMON --pred_len ${PL} \
     --model_id ${DS}_v4${MIDP}_${PL}_${tag} --cdp_backbone_checkpoint "$BASE" "$@" > "$log" 2>&1
  "$PY" -u run.py --is_training 0 $COMMON --pred_len ${PL} \
     --model_id ${DS}_v4${MIDP}_${PL}_${tag} --cdp_backbone_checkpoint "$BASE" "$@" >> "$log" 2>&1
  local r=$(grep -a -oE "mse:[0-9.]+, mae:[0-9.]+" "$log" | tail -1)
  echo "${DS} h${PL} ${tag}${MIDP}  ${r}" >> $RES
  echo "--> ${tag}: ${r}"
}

case "$STAGE" in
  # 快速筛选：先确认「哪些新杠杆真的有用」
  screen)
    run_variant r1_all       ${V4BASE}
    run_variant r2_ht        ${V4BASE} --backbone_mode handtuned
    run_variant r3_learn     ${V4BASE} --backbone_mode learn     --cdp_unfreeze_pdm 1 --cdp_st_lr_ratio ${V4STLR}
    run_variant r4_dyn       ${V4BASE} --backbone_mode dynamic   --cdp_unfreeze_pdm 1 --cdp_st_lr_ratio ${V4STLR}
    run_variant r5_dyncal    ${V4BASE} --cdp_dyncal 1
    run_variant r6_dyn_dc    ${V4BASE} --backbone_mode dynamic --cdp_unfreeze_pdm 1 --cdp_st_lr_ratio ${V4STLR} --cdp_dyncal 1
    ;;
  # 增量消融（以 screen 的赢家 r6_dyn_dc 为底座）
  ablate)
    V4WIN="--backbone_mode dynamic --cdp_unfreeze_pdm 1 --cdp_st_lr_ratio ${V4STLR} --cdp_dyncal 1"
    run_variant a_win        ${V4BASE} ${V4WIN}
    run_variant a_blk1       ${V4BASE} ${V4WIN} --cdp_unfreeze_blocks 1 --cdp_head_lr_ratio 0.3
    run_variant a_blk2       ${V4BASE} ${V4WIN} --cdp_unfreeze_blocks 2 --cdp_head_lr_ratio 0.3
    run_variant a_blk3       ${V4BASE} ${V4WIN} --cdp_unfreeze_blocks 3 --cdp_head_lr_ratio 0.3
    run_variant a_l1_1       ${V4BASE} ${V4WIN} --cdp_l1_mu 1.0
    run_variant a_l1_2       ${V4BASE} ${V4WIN} --cdp_l1_mu 2.0
    run_variant a_l1_2_blk1  ${V4BASE} ${V4WIN} --cdp_l1_mu 2.0 --cdp_unfreeze_blocks 1 --cdp_head_lr_ratio 0.3
    run_variant a_stlr_hi    ${V4BASE} ${V4WIN} --cdp_st_lr_ratio 1.0
    ;;
  # 通用单变体：参数完全由 ST2_EXTRA 指定，tag 由 TAG_OVR 指定
  # （给主队列/复现实验用，方便在不改脚本的情况下换配方）
  custom)
    run_variant ${TAG_OVR:-custom} ${V4BASE} ${ST2_EXTRA}
    ;;
  # 正式跑：赢家配方，长训练
  final)
    run_variant f_win        ${V4BASE} --backbone_mode dynamic --cdp_unfreeze_pdm 1 --cdp_st_lr_ratio ${V4STLR} --cdp_dyncal 1
    run_variant f_all        ${V4BASE} --backbone_mode dynamic --cdp_unfreeze_pdm 1 --cdp_st_lr_ratio ${V4STLR} --cdp_dyncal 1 --cdp_unfreeze_blocks ${V4BLK:-1} --cdp_head_lr_ratio 0.3
    ;;
esac
echo "V4 DONE ${DS} h${PL} stage=${STAGE}"
