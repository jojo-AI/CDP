#!/bin/bash
# =============================================================================
# 通用确定性冒烟测试（支持任意数据集）
#   2026-10-08 新增，用于把 ETTh1-96 的确定性结论外推到不同通道数的数据集。
#
# 背景：ETTh1-96（7 通道）实测
#   det_mode=0 → 同 seed 4 次 MSE 极差 0.89%，第 2 个 epoch 就分叉 ❌
#   det_mode=1 → 同 seed 4 次完全一致，分叉推迟到第 8 个 epoch ✅
# 但 21 通道的 weather 可能走不同 kernel/cuBLAS 分支，结论未必可外推 → 本脚本验证。
#
# 用法:  bash scripts/smoke_det_ds.sh <dataset> <det_mode> <n_repeat>
#   dataset: weather | ETTh1
# 例:    bash scripts/smoke_det_ds.sh weather 1 3
# =============================================================================
PY="/c/Users/Administrator/.workbuddy/binaries/python/envs/default/Scripts/python.exe"
cd "/d/fsdownload/TimeMixer" || exit 1

DS=${1:-weather}
DET=${2:-1}
NREP=${3:-3}
OUT="/d/fsdownload/det_${DS}_det${DET}"
rm -rf "$OUT"; mkdir -p "$OUT"

# --- 逐数据集配方（与历史 st1_<DS>_96_dynamic_l10.0 完全一致）---
case "$DS" in
  weather)
    DATA="custom"; ROOT="./dataset/weather/"; PATH_CSV="weather.csv"
    ENC=21; DM=16; EL=3; FACTOR=3; BATCH=128; ;;
  ETTh1)
    DATA="ETTh1"; ROOT="./dataset/ETT-small/"; PATH_CSV="ETTh1.csv"
    ENC=7; DM=16; EL=2; FACTOR=1; BATCH=128; ;;
  *)
    echo "未知数据集: $DS"; exit 1; ;;
esac

COMMON="--task_name long_term_forecast --model TimeMixer --features M --seq_len 96 --label_len 0 \
  --d_layers 1 --des DETSMOKE --itr 1 --learning_rate 0.01 --num_workers 0 \
  --train_loss l1 --train_l1_mu 0.0 \
  --down_sampling_layers 3 --down_sampling_method avg --down_sampling_window 2 \
  --train_epochs 30 --patience 8 \
  --root_path ${ROOT} --data_path ${PATH_CSV} --data ${DATA} \
  --enc_in ${ENC} --dec_in ${ENC} --c_out ${ENC} --e_layers ${EL} --factor ${FACTOR} \
  --d_model ${DM} --d_ff 32 --batch_size ${BATCH} \
  --backbone_mode dynamic --det_mode ${DET}"

echo "########## 通用确定性冒烟  ds=${DS}  det_mode=${DET}  n_repeat=${NREP}  $(date '+%F %T') ##########"

run_one () {   # $1=标签  $2=seed
  local tag="$1" seed="$2"
  # model_id 必须含 ds 与 det_mode，防止不同配置共用 checkpoint 目录互相覆盖
  local mid="${DS}_detsmoke_d${DET}_${tag}"
  local log="$OUT/${tag}.log"
  echo "----- [$(date '+%T')] ${tag} (seed=${seed}) -----"
  "$PY" -u run.py --is_training 1 $COMMON --pred_len 96 --model_id "$mid" \
        --fix_seed "$seed" > "$log" 2>&1
  grep -aoE "Epoch: [0-9]+, Steps: [0-9]+ \| Train Loss: [0-9.]+ Vali Loss: [0-9.]+ Test Loss: [0-9.]+" \
       "$log" > "$OUT/${tag}.epochs" 2>/dev/null
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

for i in $(seq 1 $NREP); do run_one "same${i}" 2021; done
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
        print('    → 判定: 数值等价（<1e-4%），可视为 bit-identical 级别 ✅')
    elif sp_m < 0.05 and sp_a < 0.05:
        print('    → 判定: 准确定性（<0.05%），对 1% 级结论无影响 ⚠️可接受')
    else:
        print('    → 判定: 非确定性显著（>=0.05%），std 会被污染 ❌')
d = load('diff1')
if d and same:
    print()
    print('  Q3 不同 seed 对照: seed2024 mse=%.8f mae=%.8f' % d)
    mm = statistics.mean([s[0] for s in same]); aa = statistics.mean([s[1] for s in same])
    print('     vs 同 seed 均值 mse=%.8f mae=%.8f' % (mm, aa))
    dd = abs(d[0] - mm) / mm * 100
    print('     seed 效应 = %.4f%% 相对 MSE' % dd)
    print('     → %s' % ('✅ seed 确实在起作用' if dd > 1e-3 else '❌ seed 无效，多种子实验无意义'))
PYEOF

# --- 分叉点分析：每 epoch 的 Train Loss 轨迹 ---
"$PY" - "$OUT" "$NREP" <<'PYEOF'
import sys, os, re
out, nrep = sys.argv[1], int(sys.argv[2])
print()
print('  --- 各次前 12 个 epoch 的 Train Loss（找分叉点）---')
trajs = {}
for tag in ['same%d' % i for i in range(1, nrep+1)] + ['diff1']:
    p = os.path.join(out, tag + '.epochs')
    if not os.path.exists(p): continue
    vals = []
    for line in open(p):
        m = re.search(r'Epoch: (\d+),.*Train Loss: ([0-9.]+)', line)
        if m: vals.append((int(m.group(1)), float(m.group(2))))
    trajs[tag] = dict(vals)
for tag, d in trajs.items():
    ks = sorted(d)[:12]
    print('    %-7s %s' % (tag, ' '.join('%.7f' % d[k] for k in ks)))
# 第一个出现差异的 epoch
ref = trajs.get('same1')
if ref:
    others = [t for t in trajs if t.startswith('same') and t != 'same1']
    if others:
        key = sorted(ref)
        diverge = None
        for k in key:
            vals = set(ref[k])
            for t in others:
                if k in trajs[t]: vals.add(trajs[t][k])
            if len(vals) > 1:
                diverge = k; break
        print('    → 同 seed 首次分叉 epoch = %s' % (diverge if diverge is not None else '无（全程一致）'))
PYEOF

echo
echo "SMOKE ${DS} DONE $(date '+%T')   原始数据: $OUT"
