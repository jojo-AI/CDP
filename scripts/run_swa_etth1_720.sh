#!/bin/bash
# =============================================================================
# SWA / 验证集平台期集成实验 —— ETTh1-720（2026-10-08 用户批准）
#
# 目的：ETTh1-720 换种子后最终 MSE 从 −5.75% 抖到 +5.28%（3 种子极差 18.4%）。
#   根因 = 平坦的 val 曲线 + argmin(val) 选点 = 抽签（见 multiseed_findings_20261008.md §7）。
#   本实验用「验证集平台期 checkpoint 集成」替换该抽签，看能否把跨种子抖动压进噪声下限。
#
# 做法：
#   1) 与既有协议完全相同的配方（ETTh1-720，Stage-1 dynamic+L1，Stage-2 lt_cal5），
#      仅额外加 --save_all_epochs 1（逐 epoch 存 checkpoint）。
#   2) 对 Stage-1 与 Stage-2 分别跑 tools/eval_ckpt_ensemble.py，比较四种口径：
#        ① 现有协议 argmin(val)  ② 平台期预测平均  ③ 平台期权重平均(SWA)  ④ 参考上界 argmin(test)
#   3) 跑 3 个种子（2021/2022/2023），比较「跨种子极差」是否下降。
#
# 排期：先等 /d/fsdownload/.multiseed.lock 释放（那里在跑 ETTm1-96/weather-96 的新种子），
#       本实验跑完后**自动释放锁并交棒**给 scripts/run_multiseed_traffic96.sh。
#
# 用法：bash scripts/run_swa_etth1_720.sh      （后台）
# 日志：/d/fsdownload/swa_etth1_720.log        结果：/d/fsdownload/swa_etth1_720_results.txt
# =============================================================================
PY="/c/Users/Administrator/.workbuddy/binaries/python/envs/default/Scripts/python.exe"
cd "/d/fsdownload/TimeMixer" || exit 1

LOG="/d/fsdownload/swa_etth1_720.log"
RES="/d/fsdownload/swa_etth1_720_results.txt"
LOCK="/d/fsdownload/.multiseed.lock"
SEEDS="${SEEDS_OVR:-2021 2022 2023}"
EPS="${EPS_OVR:-0.01}"

say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

# ---- 等前面的队列释放 GPU（绝不并发）----
acquired=0
for i in $(seq 1 2000); do
  if mkdir "$LOCK" 2>/dev/null; then
    echo $$ > "$LOCK/pid"
    acquired=1
    break
  fi
  [ $((i % 10)) -eq 1 ] && say "等待 GPU 队列释放（$LOCK，pid=$(cat "$LOCK/pid" 2>/dev/null))…"
  sleep 30
done
if [ "$acquired" -ne 1 ]; then say "!! 等锁超时，退出"; exit 1; fi
cleanup() { rm -rf "$LOCK"; }
trap cleanup EXIT

# ---- 公共参数（与 run_multiseed.sh 逐字段一致）----
ST1_COMMON="--task_name long_term_forecast --model TimeMixer --features M --seq_len 96 --label_len 0 \
  --d_layers 1 --des ST1 --itr 1 --learning_rate 0.01 --num_workers 0 \
  --train_loss l1 --train_l1_mu 0.0 \
  --down_sampling_layers 3 --down_sampling_method avg --down_sampling_window 2 \
  --train_epochs 30 --patience 8"
ST2_COMMON="--task_name long_term_forecast --model TimeMixer_CDP --features M --seq_len 96 --label_len 0 \
  --d_layers 1 --des CDPv4 --itr 1 --num_workers 0 \
  --down_sampling_layers 3 --down_sampling_method avg --down_sampling_window 2 \
  --cdp_stage_w 1 --cdp_M 4 --cdp_S 16 --cdp_err_decay 0.9 --cdp_lambda_t 0.01 \
  --train_epochs 20 --patience 12"
DSARGS="--root_path ./dataset/ETT-small/ --data_path ETTh1.csv --data ETTh1 \
  --enc_in 7 --dec_in 7 --c_out 7 --e_layers 2 --factor 1 --d_model 16 --d_ff 32 --batch_size 128 --pred_len 720"
ST1_EXTRA="--backbone_mode dynamic"
ST2_EXTRA="--backbone_mode dynamic --cdp_learning_rate 5e-4 --cdp_loss l1 --cdp_monitor mae \
  --cdp_dyncal 1 --cdp_unfreeze_pdm 1 --cdp_st_lr_ratio 0.2 --cdp_cal_lr_ratio 5"

say "########## SWA / 平台期集成实验 START (pid=$$) seeds='$SEEDS' eps=$EPS ##########"
touch "$RES"

for SEED in $SEEDS; do
  M1="ETTh1_st1_720_swa_s${SEED}"
  M2="ETTh1_swa_720_lt_cal5_s${SEED}"
  L1="/d/fsdownload/swa_st1_s${SEED}.log"
  L2="/d/fsdownload/swa_st2_s${SEED}.log"

  # ---------------- Stage-1（骨干，逐 epoch 存点） ----------------
  say "--- [seed=$SEED] Stage-1 训练（--save_all_epochs 1）→ $M1"
  "$PY" -u run.py --is_training 1 $ST1_COMMON $DSARGS $ST1_EXTRA \
      --model_id "$M1" --fix_seed "$SEED" --save_all_epochs 1 > "$L1" 2>&1
  D1=$(ls -dt ./checkpoints/*"$M1"* 2>/dev/null | head -1)
  if [ -z "$D1" ]; then say "!!! seed=$SEED Stage-1 找不到 checkpoint 目录 → 跳过"; continue; fi
  say "    Stage-1 目录 $D1"

  say "--- [seed=$SEED] Stage-1 集成评估"
  "$PY" -u tools/eval_ckpt_ensemble.py --log "$L1" --ckpt-dir "$D1" --eps "$EPS" \
      --out "/d/fsdownload/swa_ens_st1_s${SEED}.json" >> "$LOG" 2>&1 || say "!!! Stage-1 评估失败"

  # ---------------- Stage-2（CDP 校准，逐 epoch 存点） ----------------
  say "--- [seed=$SEED] Stage-2 训练（--save_all_epochs 1）→ $M2  backbone=$D1"
  "$PY" -u run.py --is_training 1 $ST2_COMMON --learning_rate 0.001 $DSARGS $ST2_EXTRA \
      --model_id "$M2" --cdp_backbone_checkpoint "$D1" --fix_seed "$SEED" --save_all_epochs 1 > "$L2" 2>&1
  D2=$(ls -dt ./checkpoints/*"$M2"* 2>/dev/null | head -1)
  if [ -z "$D2" ]; then say "!!! seed=$SEED Stage-2 找不到 checkpoint 目录 → 跳过"; continue; fi
  say "    Stage-2 目录 $D2"

  say "--- [seed=$SEED] Stage-2 集成评估"
  "$PY" -u tools/eval_ckpt_ensemble.py --log "$L2" --ckpt-dir "$D2" --eps "$EPS" \
      --out "/d/fsdownload/swa_ens_st2_s${SEED}.json" >> "$LOG" 2>&1 || say "!!! Stage-2 评估失败"

  echo "seed=$SEED st1_dir=$D1 st2_dir=$D2" >> "$RES"
done

# ---------------- 汇总 ----------------
say "--- 汇总（跨种子对比） ---"
"$PY" -u tools/swa_summary.py >> "$LOG" 2>&1 || say "!!! 汇总失败（工具缺失或报错，见上）"

say "########## SWA 实验 DONE ##########"

# ---------------- 交棒给 Traffic-96 队列（先释放锁，避免死锁） ----------------
trap - EXIT
rm -rf "$LOCK"
say "已释放 GPU 锁，交棒给 Traffic-96 队列（seed 2022，约 5.15 h）"
bash scripts/run_multiseed_traffic96.sh >> /d/fsdownload/multiseed_traffic96.log 2>&1
say "Traffic-96 队列结束"
