#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""统计「每个 (数据集, horizon) 的第一名归属」：CDP 校准栈 / 纯 Stage-1 骨干 / 手调推理占位。

用法:
    python tools/count_wins.py                 # 汇总表 + 计数 + 最佳 MSE/MAE
    python tools/count_wins.py --detail        # 附带每个设定的归属清单

归属判定（与 select_best.py 同一套候选池与打分准则）：
    CDP     : Stage-2 校准栈候选（at_* / cdp_* / v5final / lt_*）
    STAGE1  : 只重训骨干、未上 CDP（st1_*）
    INFER   : 手调权重纯推理（handtuned / handtuned_off），即占位值
    EXPLORE : 早期探索候选（仅当它意外成为第一名时才标注）
"""
import os
import re
import sys
from collections import defaultdict

ROOT = os.environ.get('CDP_RESULT_DIR', 'D:/fsdownload')
BASE_FILE = os.path.join(ROOT, 'baseline_results.txt')
FILES = [
    os.path.join(ROOT, 'v4_results.txt'),
    os.path.join(ROOT, 'stage1_results.txt'),
    os.path.join(ROOT, 'handtune_results.txt'),
]

LINE = re.compile(r'^(\S+)\s+h(\d+)\s+(\S+)\s+mse:([\d.]+),\s*mae:([\d.]+)')
BASE = re.compile(r'^(\S+)\s+h(\d+)\s+baseline\s+mse:([\d.]+),\s*mae:([\d.]+)')

DS_ORDER = {'ETTh1': 0, 'ETTh2': 1, 'ETTm1': 2, 'ETTm2': 3, 'weather': 4, 'ECL': 5, 'Traffic': 6}
CDP_PREFIX = ('at_', 'cdp_', 'v5final', 'lt_')
STAGE1_PREFIX = ('st1_',)
INFER_TAGS = ('handtuned', 'handtuned_off')


def owner(tag):
    """候选 -> 归属类别"""
    if tag in INFER_TAGS:
        return 'INFER'
    if tag.startswith(STAGE1_PREFIX):
        return 'STAGE1'
    if tag.startswith(CDP_PREFIX):
        return 'CDP'
    return 'EXPLORE'


def load():
    base = {}
    for ln in open(BASE_FILE, encoding='utf-8', errors='ignore'):
        m = BASE.match(ln.strip())
        if m:
            base[(m.group(1), int(m.group(2)))] = (float(m.group(3)), float(m.group(4)))
    cand = defaultdict(list)
    for path in FILES:
        if not os.path.exists(path):
            continue
        for ln in open(path, encoding='utf-8', errors='ignore'):
            m = LINE.match(ln.strip())
            if m:
                cand[(m.group(1), int(m.group(2)))].append(
                    (m.group(3), float(m.group(4)), float(m.group(5))))
    return base, cand


def main():
    detail = '--detail' in sys.argv
    base, cand = load()

    keys = sorted(set(list(base) + list(cand)),
                  key=lambda k: (DS_ORDER.get(k[0], 9), k[1]))

    wins = {'CDP': 0, 'STAGE1': 0, 'INFER': 0, 'EXPLORE': 0}
    no_cand = []
    best_mse_imp = None      # (Δ%, tag, ds, h, mse)
    best_mae_imp = None
    best_mse_abs = None      # (mse, tag, ds, h)
    best_mae_abs = None
    cdp_rows = []

    print('%-8s %4s %-34s %-7s %9s %9s %8s %8s' %
          ('DS', 'h', '选定候选', '归属', 'MSE', 'MAE', 'ΔMSE', 'ΔMAE'))
    print('-' * 100)

    for k in keys:
        ds, h = k
        b = base.get(k)
        rs = [r for r in cand.get(k, []) if b]
        if not b:
            continue
        if not rs:
            no_cand.append(k)
            print('%-8s %4d %-34s %-7s' % (ds, h, '(无候选)', '—'))
            continue
        scored = []
        for tag, mse, mae in rs:
            dm = (mse / b[0] - 1) * 100
            da = (mae / b[1] - 1) * 100
            # score = min(-dm, -da) = -max(dm, da)：先最大化「较差的那一项」的提升量
            # （负值 = 至少有一项退步；与 select_best.py 的准则一致）
            scored.append((tag, mse, mae, dm, da, -max(dm, da), (dm + da) / 2))
        scored.sort(key=lambda r: (-r[5], r[6]))
        t = scored[0]
        own = owner(t[0])
        wins[own] += 1
        print('%-8s %4d %-34s %-7s %9.5f %9.5f %+8.2f %+8.2f' %
              (ds, h, t[0], own, t[1], t[2], t[3], t[4]))

        if t[3] < 0:
            if best_mse_imp is None or t[3] < best_mse_imp[0]:
                best_mse_imp = (t[3], t[0], ds, h, t[1])
        if t[4] < 0:
            if best_mae_imp is None or t[4] < best_mae_imp[0]:
                best_mae_imp = (t[4], t[0], ds, h, t[2])
        if best_mse_abs is None or t[1] < best_mse_abs[0]:
            best_mse_abs = (t[1], t[0], ds, h)
        if best_mae_abs is None or t[2] < best_mae_abs[0]:
            best_mae_abs = (t[2], t[0], ds, h)

        # CDP 在该设定内相对「非 CDP 最佳」的额外增益（正=CDP 更好）
        non_cdp = [r for r in scored if owner(r[0]) != 'CDP']
        if own == 'CDP' and non_cdp:
            n = non_cdp[0]
            cdp_rows.append((ds, h, t[3] - n[3], t[4] - n[4]))

    if detail:
        print('\n=== 明细 ===')
        for ds, h, dm, da in cdp_rows:
            print('  %-8s h%-4d CDP 相对该设定内最佳非CDP候选: ΔMSE %+.2fpp  ΔMAE %+.2fpp' % (ds, h, dm, da))

    total = sum(wins.values())
    # 「有真实新协议候选」= 该设定至少有一个非手调推理候选（st1/cdp 任一）
    real_set = 0
    for k in keys:
        if not base.get(k):
            continue
        rs = cand.get(k, [])
        if any(owner(t) != 'INFER' for t in [r[0] for r in rs]):
            real_set += 1

    print('\n=== 第一名归属统计 ===')
    print('  已完成设定数: %d （另有 %d 个设定尚无任何候选）' % (total, len(no_cand)))
    for k in ('CDP', 'STAGE1', 'INFER', 'EXPLORE'):
        if wins[k]:
            print('  %-8s %2d 个  (%.0f%%)' % (k, wins[k], 100.0 * wins[k] / total))
    print('  -> 其中「跑过新标准候选」的设定 = %d 个，CDP 拿下 %d 个 (%.0f%%)'
          % (real_set, wins['CDP'], 100.0 * wins['CDP'] / real_set))
    print('  -> 仅有手调推理占位的设定 = %d 个（CDP 尚未获得真实对标机会）'
          % (total - real_set))
    if no_cand:
        print('  无候选设定: ' + ', '.join('%s-%d' % (d, h) for d, h in no_cand))

    print('\n=== 极值 ===')
    if best_mse_imp:
        print('  最佳 MSE 提升: %+.2f%%  -> %s (%s h%d, MSE %.5f)' % best_mse_imp)
    if best_mae_imp:
        print('  最佳 MAE 提升: %+.2f%%  -> %s (%s h%d, MAE %.5f)' % best_mae_imp)
    if best_mse_abs:
        print('  最低 MSE 绝对值: %.5f  -> %s (%s h%d)' % best_mse_abs)
    if best_mae_abs:
        print('  最低 MAE 绝对值: %.5f  -> %s (%s h%d)' % best_mae_abs)


if __name__ == '__main__':
    main()
