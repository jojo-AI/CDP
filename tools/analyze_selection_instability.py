#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
「同种子散度」归因分析：训练噪声 vs 选优不稳定（2026-10-08 新增）

动机：
  确定性冒烟测试里，同 seed 重复的最终 MSE 极差 0.7%~1.2%，看起来像「训练不可复现」。
  但逐 epoch 日志显示：
    - 第 1 个 epoch 的 loss 只差 1e-5（FP 噪声量级）
    - 中后期各 run 的 val 曲线互相交错，best-val 值近乎平局
    - 落在不同的 best epoch，而 val 与 test 的相关性很弱
  ⇒ 真正放大器可能是「按 val 选 best epoch」这一选择步骤，而不是训练本身。

本脚本对一批冒烟实验目录做定量拆解，逐 run 输出：
    ep1 相对差、val 轨迹分叉点、best-val epoch/值、val plateau 宽度、
    「val 最优 epoch」与「test 最优 epoch」是否一致、final MSE。
并把长表写成 CSV，便于写进论文/回复审稿人。

用法：
  python tools/analyze_selection_instability.py --dirs \
      D:/fsdownload/determinism_test_det0 D:/fsdownload/det_weather_det1 \
      --out D:/fsdownload/selection_instability.csv
"""
import os
import re
import csv
import glob
import argparse
import statistics


EP_RE = re.compile(
    r'Epoch: (\d+), Steps: (\d+) \| Train Loss: ([\d.]+) Vali Loss: ([\d.]+) Test Loss: ([\d.]+)')


def parse_run(log_path):
    txt = open(log_path, encoding='utf-8', errors='ignore').read()
    rows = []
    for a, b, c, d, e in EP_RE.findall(txt):
        rows.append({'epoch': int(a), 'steps': int(b),
                     'train': float(c), 'val': float(d), 'test': float(e)})
    return rows


def final_mse(f):
    """f = .final 文件的完整路径"""
    if not os.path.exists(f):
        return None
    m = re.search(r'mse:([\d.]+)', open(f, encoding='utf-8', errors='ignore').read())
    return float(m.group(1)) if m else None


def ckpt_md5(f):
    """f = .ckptmd5 文件的完整路径"""
    if not os.path.exists(f):
        return None
    v = open(f, encoding='utf-8', errors='ignore').read().strip()
    return v if v and v != 'NO_CKPT' else None


def rel(a, b):
    return abs(a - b) / abs(b) * 100 if b else 0.0


def analyze_dir(base, out_rows):
    tags = sorted(os.path.basename(p).replace('.log', '')
                  for p in glob.glob(os.path.join(base, 'same*.log')))
    runs = {}
    for t in tags:
        rows = parse_run(os.path.join(base, t + '.log'))
        if not rows:
            continue
        runs[t] = {
            'rows': rows,
            'mse': final_mse(os.path.join(base, t + '.final')),
            'ckpt': ckpt_md5(os.path.join(base, t + '.ckptmd5')),
        }
    if len(runs) < 2:
        print('  [%s] 有效 same* 运行 < 2，跳过' % os.path.basename(base))
        return
    ref = 'same1'
    base_rows = runs[ref]['rows']
    n = min(len(runs[t]['rows']) for t in runs)
    # 首个 val 相对差 > 0.1% 的 epoch（视为实质分叉）
    fork_ep = None
    for i in range(n):
        v0 = base_rows[i]['val']
        if any(rel(runs[t]['rows'][i]['val'], v0) > 0.1 for t in runs if t != ref):
            fork_ep = i + 1
            break
    print('=' * 92)
    print('### %s   (%d 个同 seed run)' % (base, len(runs)))
    for t in sorted(runs):
        r = runs[t]
        best = min(r['rows'], key=lambda x: x['val'])
        test_best = min(r['rows'], key=lambda x: x['test'])
        # val plateau 宽度：best 之后 val 仍在 best*1.003 内的 epoch 数
        plateau = [x['epoch'] for x in r['rows'] if x['val'] <= best['val'] * 1.003]
        row = {
            'dir': base,
            'tag': t,
            'epochs': len(r['rows']),
            'ep1_train': r['rows'][0]['train'],
            'ep1_val': r['rows'][0]['val'],
            'best_val_epoch': best['epoch'],
            'best_val': best['val'],
            'best_val_testloss': best['test'],
            'test_best_epoch': test_best['epoch'],
            'test_best_loss': test_best['test'],
            'plateau_epochs': len(plateau),
            'final_mse': r['mse'],
            'ckpt_md5': (r['ckpt'] or '')[:10],
            'same_ckpt_as_ref': (r['ckpt'] == runs[ref]['ckpt']) if r['ckpt'] else None,
        }
        out_rows.append(row)
        print('  %-7s ep=%2d | ep1: tr=%.7f val=%.7f | best(val)ep=%2d val=%.7f '
              '| test最优ep=%2d | plateau=%d ep | MSE=%s | ck=%s'
              % (t, len(r['rows']), row['ep1_train'], row['ep1_val'],
                 best['epoch'], best['val'], test_best['epoch'], len(plateau),
                 ('%.5f' % r['mse']) if r['mse'] else 'NA', row['ckpt_md5']))

    ms = [runs[t]['mse'] for t in runs if runs[t]['mse']]
    cks = {runs[t]['ckpt'] for t in runs if runs[t]['ckpt']}
    cks.discard('NO_CKPT')
    print('  ---------- 汇总 ----------')
    if ms:
        print('  final MSE: %s  极差 = %.4f%%   (max-min)/min'
              % ([round(x, 5) for x in ms], (max(ms) - min(ms)) / min(ms) * 100))
    print('  checkpoint 版本数: %s / %d  %s'
          % (len(cks), len(runs), '✅ 权重完全一致' if len(cks) == 1 else '❌ 权重不同'))
    print('  实质分叉 epoch（val 相对差首超 0.1%%）: %s' % (fork_ep or '未出现'))
    plateau_eps = sorted({min(runs[t]['rows'], key=lambda x: x['val'])['epoch'] for t in runs})
    print('  各 run 的 best-val epoch: %s  ⇒ %s'
          % (plateau_eps,
             '选择一致' if len(plateau_eps) == 1 else '选择落在不同 epoch（选优不稳定）'))
    spread_val = max(rel(min(runs[t]['rows'], key=lambda x: x['val'])['val'],
                         min(runs[ref]['rows'], key=lambda x: x['val'])['val']) for t in runs)
    print('  各 run 的 best-val 值极差: %.4f%%  ← 若与 final MSE 极差同量级，说明散度主要由选优解释'
          % spread_val)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dirs', nargs='*', default=[
        'D:/fsdownload/determinism_test_det0',
        'D:/fsdownload/determinism_test_det1',
        'D:/fsdownload/det_weather_det0',
        'D:/fsdownload/det_weather_det1',
    ])
    ap.add_argument('--out', default='D:/fsdownload/selection_instability.csv')
    a = ap.parse_args()
    rows = []
    for d in a.dirs:
        if os.path.isdir(d):
            analyze_dir(d, rows)
        else:
            print('跳过不存在的目录: %s' % d)
    if rows:
        with open(a.out, 'w', newline='', encoding='utf-8-sig') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print()
        print('已写出长表: %s（%d 行）' % (a.out, len(rows)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
