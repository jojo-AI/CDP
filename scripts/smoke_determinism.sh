#!/bin/bash
# =============================================================================
# 确定性冒烟测试 —— 多种子补跑（审稿 P1-2）的 P0 前置检查
#
# 要回答三个问题：
#   Q1 同一 seed 跑 N 次，结果是否 bit-identical？
#      → 若否，跨 seed 的 std 会被"非确定性噪声"污染，反而坐实 R3「不可复现」
#   Q2 同一 seed 的离散度有多大？（= 论文表格的最小分辨率下限）
#   Q3 不同 seed 结果是否真的不同？
#      → 反向对照。若不同 seed 结果相同，说明 seed 没生效，Q1 的"可复现"是假象。
#
# 被测对象：ETTh1-96 Stage-1（--backbone_mode dynamic --train_loss l1），
#           与最终协议 st1_dynamic_l10.0 完全同配方。
#           历史基准：mse 0.36919753838017955 / mae 0.3880593295540669
# 成本：单次 ~14 epoch × ~4.5 s ≈ 60 s；默认 4 次 ≈ 4.5 min。
#
# 用法:  bash scripts/smoke_determinism.sh [det_mode] [n_repeat]
#        det_mode 默认 0（原始行为）  n_repeat 默认 3
# =============================================================================
PY="/c/Users/Administrator/.workbuddy/binaries/python/envs/default/Scripts/python.exe"
cd "/d/fsdownload/TimeMixer" || exit 1

DET=${1:-0}
NREP=${2:-3}
OUT="/d/fsdownload/determinism_test_det${DET}"
rm -rf "$OUT"; mkdir -p "$OUT"

COMMON="--task_name long_term_forecast --model TimeMixer --features M --seq_len 96 --label_len 0 \
  --d_layers 1 --des DETSMOKE --itr 1 --learning_rate 0.01 --num_workers 0 \
  --train_loss l1 --train_l1_mu 0.0 \
  --down_sampling_layers 3 --down_sampling_method avg --down_sampling_window 2 \
  --train_epochs 30 --patience 8 \
  --root_path ./dataset/ETT-small/ --data_path ETTh1.csv --data ETTh1 \
  --enc_in 7 --dec_in 7 --c_out 7 --e_layers 2 --factor 1 --d_model 16 --d_ff 32 --batch_size 128 \
  --backbone_mode dynamic --det_mode ${DET}"

echo "########## 确定性冒烟测试  det_mode=${DET}  n_repeat=${NREP}  $(date '+%F %T') ##########"

run_one () {   # $1=标签  $2=seed
  local tag="$1" seed="$2"
  # ⚠️ model_id 必须包含 det_mode！否则不同 det_mode 的配置会共用同一批
  #    checkpoints/ 目录，后跑的覆盖先跑的，md5 比对直接失去意义（踩过这个坑）。
  local mid="ETTh1_detsmoke_d${DET}_${tag}"
  local log="$OUT/${tag}.log"
  echo "----- [$(date '+%T')] ${tag} (seed=${seed}) -----"
  "$PY" -u run.py --is_training 1 $COMMON --pred_len 96 --model_id "$mid" \
        --fix_seed "$seed" > "$log" 2>&1
  # 每 epoch 的 Train/Vali/Test Loss（7 位小数，足以判定一致性）
  grep -aoE "Epoch: [0-9]+, Steps: [0-9]+ \| Train Loss: [0-9.]+ Vali Loss: [0-9.]+ Test Loss: [0-9.]+" \
       "$log" > "$OUT/${tag}.epochs" 2>/dev/null
  # 最优 checkpoint 哈希（权重级比对，比 log 更严格）
  local ck
  ck=$(ls -d ./checkpoints/long_term_forecast_${mid}_* 2>/dev/null | head -1)
  if [ -n "$ck" ] && [ -f "$ck/checkpoint.pth" ]; then
    md5sum "$ck/checkpoint.pth" | awk '{print $1}' > "$OUT/${tag}.ckptmd5"
  else
    echo "NO_CKPT" > "$OUT/${tag}.ckptmd5"
  fi
  grep -aoE "mse:[0-9.]+, mae:[0-9.]+" "$log" 2>/dev/null | tail -1 > "$OUT/${tag}.final"
  echo "  ${tag}: eps=$(wc -l < "$OUT/${tag}.epochs")  ckpt=$(cat "$OUT/${tag}.ckptmd5")  $(cat "$OUT/${tag}.final")"
}

# --- 同 seed 重复 NREP 次 ---
for i in $(seq 1 $NREP); do
  run_one "same${i}" 2021
done
# --- 反向对照：另一个 seed ---
run_one "diff1" 2024

echo
echo "########## Q1/Q2 同 seed(2021) 重复 ${NREP} 次 ##########"
uniq_epoch=$(for i in $(seq 1 $NREP); do md5sum "$OUT/same${i}.epochs" | awk '{print $1}'; done | sort -u | wc -l)
uniq_ck=$(for i in $(seq 1 $NREP); do cat "$OUT/same${i}.ckptmd5"; done | sort -u | wc -l)
echo "  逐 epoch 轨迹的不同版本数: ${uniq_epoch} / ${NREP}   $([ "$uniq_epoch" = "1" ] && echo '✅ 完全一致' || echo '❌ 有分叉')"
echo "  checkpoint MD5 的不同版本数: ${uniq_ck} / ${NREP}   $([ "$uniq_ck" = "1" ] && echo '✅ 权重级一致' || echo '❌ 权重不同')"

"$PY" - "$OUT" "$NREP" <<'PYEOF'
import sys, os, re, statistics
out, nrep = sys.argv[1], int(sys.argv[2])
def load(tag):
    p = os.path.join(out, tag + '.final')
    if not os.path.exists(p): return None
    m = re.search(r'mse:([0-9.]+), mae:([0-9.]+)', open(p).read())
    return (float(m.group(1)), float(m.group(2))) if m else None
same = [load('same%d' % i) for i in range(1, nrep + 1)]
same = [s for s in same if s]
print()
print('  同 seed 各次最终指标:')
for i, s in enumerate(same, 1):
    print('    run%d  mse=%.8f  mae=%.8f' % (i, s[0], s[1]))
if len(same) >= 2:
    ms = [s[0] for s in same]; as_ = [s[1] for s in same]
    sp_m = (max(ms) - min(ms)) / statistics.mean(ms) * 100
    sp_a = (max(as_) - min(as_)) / statistics.mean(as_) * 100
    print('    MSE 极差 = %.6f  (%.4f%% 相对)' % (max(ms) - min(ms), sp_m))
    print('    MAE 极差 = %.6f  (%.4f%% 相对)' % (max(as_) - min(as_), sp_a))
    if sp_m < 1e-4 and sp_a < 1e-4:
        print('    → 判定: 数值等价（<1e-4%%），可视为 bit-identical 级别 ✅')
    elif sp_m < 0.05 and sp_a < 0.05:
        print('    → 判定: 准确定性（<0.05%%），对 1%% 级结论无影响 ⚠️可接受')
    else:
        print('    → 判定: 非确定性显著（>=0.05%%），std 会被污染 ❌')
d = load('diff1')
if d and same:
    print()
    print('  Q3 不同 seed 对照: seed2024 mse=%.8f mae=%.8f' % d)
    print('     vs 同 seed 均值 mse=%.8f mae=%.8f' % (statistics.mean([s[0] for s in same]), statistics.mean([s[1] for s in same])))
    dd = abs(d[0] - statistics.mean([s[0] for s in same])) / statistics.mean([s[0] for s in same]) * 100
    print('     seed 效应 = %.4f%% 相对 MSE' % dd)
    print('     → %s' % ('✅ seed 确实在起作用' if dd > 1e-3 else '❌ seed 无效，多种子实验无意义'))
PYEOF

echo
echo "DETERMINISM SMOKE DONE $(date '+%T')   原始数据: $OUT"
