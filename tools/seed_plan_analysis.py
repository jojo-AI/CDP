# -*- coding: utf-8 -*-
"""
多种子补跑成本估算 + 零成本替代统计
输出：
  1) 各数据集真实 epoch 耗时与"全量重放一个种子"的成本
  2) 多种子方案的墙钟时间（Tier 0/1/2/3）
  3) 符号检验（MAE / MSE 改善方向一致性）
  4) 候选一致性：同一设定下有多少个候选打败基线（替代种子实验的稳健性证据）
"""
import re, glob, statistics, collections, os, json
from math import comb

BASE = 'D:/fsdownload'
os.chdir(BASE)

# ---------- 1. 实测 epoch 耗时 ----------
pat_cost = re.compile(r'Epoch: \d+ cost time: ([0-9.]+)')
ep = collections.defaultdict(list)
for f in glob.glob('v4_*.log'):
    m = re.match(r'v4_([A-Za-z]+[0-9]?)_(\d+)_', f)
    if not m:
        continue
    ds, h = m.group(1), int(m.group(2))
    try:
        txt = open(f, encoding='utf-8', errors='ignore').read()
    except Exception:
        continue
    c = [float(x) for x in pat_cost.findall(txt)]
    if c:
        ep[(ds, h)].extend(c)

DS = ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'weather', 'ECL', 'Traffic']
HS = [96, 192, 336, 720]
EPOCHS_PER_STAGE = 20
STAGES = 2

# 用每格最小值作为"配方本征成本"（中位数被其它候选的 batch 设置污染）
cost = {}
print('=' * 78)
print('1) 实测 epoch 耗时（秒）与单设定成本   [Stage-1 + Stage-2 = %d epoch]' % (EPOCHS_PER_STAGE * STAGES))
print('=' * 78)
print('%-9s %8s %8s %8s %8s | %10s' % ('dataset', 'h96', 'h192', 'h336', 'h720', '4档合计(小时)'))
per_ds_hours = {}
for ds in DS:
    row = []
    for h in HS:
        v = ep.get((ds, h))
        row.append(min(v) if v else None)
    hrs = sum(r * EPOCHS_PER_STAGE * STAGES for r in row if r) / 3600
    per_ds_hours[ds] = hrs
    print('%-9s %8s %8s %8s %8s | %10.2f' % (
        ds,
        *['%.0f' % r if r else '-' for r in row],
        hrs))
    for h, r in zip(HS, row):
        if r:
            cost[(ds, h)] = r * EPOCHS_PER_STAGE * STAGES / 3600

total_all = sum(per_ds_hours.values())
print('-' * 78)
print('全部 28 个设定各跑 1 次（= 1 个种子）总成本: %.1f 小时' % total_all)
print()

# ---------- 2. 方案成本 ----------
print('=' * 78)
print('2) 多种子方案墙钟成本（现有 seed=2021 算 1 个种子，需再补 2 个）')
print('=' * 78)
plans = {
    'Tier 3  全 7 数据集 × 4 档 × 2 新种子（方案书原文级别）': [(d, h) for d in DS for h in HS],
    'Tier 2b 5 数据集(4×ETT+weather) × 4 档 + Traffic h96 × 2 新种子':
        [(d, h) for d in ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'weather'] for h in HS] + [('Traffic', 96)],
    'Tier 2  5 数据集(4×ETT+weather) × 4 档 × 2 新种子':
        [(d, h) for d in ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'weather'] for h in HS],
    'Tier 1b 只 4 个 ETT × 4 档 × 2 新种子':
        [(d, h) for d in ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2'] for h in HS],
    'Tier 1a ETTh1+ETTm1 × 4 档 × 2 新种子':
        [(d, h) for d in ['ETTh1', 'ETTm1'] for h in HS],
}
for name, cells in plans.items():
    h_ = sum(cost.get(c, 0) for c in cells) * 2
    print('%-58s %7.1f h  (%4.1f 天)  %2d 格' % (name, h_, h_ / 24, len(cells)))
print()

# ---------- 3. 解析结果文件 ----------
pat = re.compile(r'^([A-Za-z]+[0-9]?)\s+h(\d+)\s+(\S+)\s+mse:([0-9.]+),\s*mae:([0-9.]+)')
base, cand = {}, collections.defaultdict(list)
for fn in ['baseline_results.txt', 'stage1_results.txt', 'v4_results.txt', 'handtune_results.txt']:
    if not os.path.exists(fn):
        continue
    for line in open(fn, encoding='utf-8', errors='ignore'):
        m = pat.match(line.strip())
        if not m:
            continue
        ds, h, tag, mse, mae = m.group(1), int(m.group(2)), m.group(3), float(m.group(4)), float(m.group(5))
        if tag == 'baseline':
            base[(ds, h)] = (mse, mae)
        else:
            cand[(ds, h)].append((tag, mse, mae, fn))

# ---------- 4. 符号检验 ----------
print('=' * 78)
print('3) 符号检验（H0: 改善方向随机 p=0.5，单侧）—— 零成本统计')
print('=' * 78)


def sign_p(k, n):
    return sum(comb(n, i) * 0.5 ** n for i in range(k, n + 1))


n_mae = n_mse = n_both = n_set = 0
for k, v in cand.items():
    if k not in base:
        continue
    bm, ba = base[k]
    best = min(v, key=lambda x: (x[2], x[1]))
    n_set += 1
    if best[2] < ba:
        n_mae += 1
    if best[1] < bm:
        n_mse += 1
    if best[1] < bm and best[2] < ba:
        n_both += 1
for nm, k in [('MAE 改善', n_mae), ('MSE 改善', n_mse), ('MAE+MSE 双改善', n_both)]:
    print('%-16s %2d/%2d  单侧 p = %.5f   %s' % (nm, k, n_set, sign_p(k, n_set),
                                                'p<0.001 ✅' if sign_p(k, n_set) < 0.001 else ('p<0.05 ✅' if sign_p(k, n_set) < 0.05 else '不显著 ❌')))
print()

# ---------- 5. 候选一致性 ----------
print('=' * 78)
print('4) 候选一致性：同一设定下「多数候选是否都能打败基线」')
print('   （若多个结构不同的候选都赢，说明增益不是靠单点调参运气）')
print('=' * 78)
print('%-9s %4s %5s | %7s %7s | %8s %8s' % ('dataset', 'h', 'n候选', 'MAE赢', 'MSE赢', 'MAE最好', 'MAE最差'))
print('-' * 78)
rows = []
for k in sorted(cand, key=lambda x: (DS.index(x[0]) if x[0] in DS else 99, x[1])):
    if k not in base:
        continue
    bm, ba = base[k]
    v = cand[k]
    wm = sum(1 for _, m, a, _ in v if a < ba)
    wms = sum(1 for _, m, a, _ in v if m < bm)
    gains = sorted((a - ba) / ba * 100 for _, m, a, _ in v)
    rows.append((k, len(v), wm, wms, gains[0], gains[-1]))
    print('%-9s %4d %5d | %7d %7d | %+7.2f%% %+7.2f%%' % (k[0], k[1], len(v), wm, wms, gains[0], gains[-1]))
print()
ok = [r for r in rows if r[1] >= 3]
maj = sum(1 for r in ok if r[2] > r[1] / 2)
print('在有 >=3 个候选的 %d 个设定中，有 %d 个（%.0f%%）是「多数候选都打败基线 MAE」' % (len(ok), maj, 100 * maj / max(1, len(ok))))
json.dump({'per_ds_hours': per_ds_hours, 'total_one_seed_hours': total_all,
           'sign_test': {'mae': [n_mae, n_set], 'mse': [n_mse, n_set], 'both': [n_both, n_set]}},
          open('seed_plan_analysis.json', 'w'), indent=2)
