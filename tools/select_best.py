#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""逐 (数据集 × horizon) 自动选优：汇总所有候选结果，按「MSE 与 MAE 同时最大化」排序。

打分准则（用户目标：两个指标都要提升）：
    score = min(dMSE%, dMAE%)        # 首先最大化「较差的那一项」
    并列时按 mean(dMSE%, dMAE%) 排序
    score < 0 表示至少有一项相对基线退步（不合格，仅作参考列出）

协议分组（只有同协议候选才可直接比较）：
    final   : Stage-2 EP=20 PAT=12（cal5 及之后的所有调优候选）
    old     : Stage-2 EP=20 PAT=8 （主队列早期配方）
    explore : 早期探索（EP=30 PAT=8 等），仅作历史参考
输出：终端摘要 + /d/fsdownload/leaderboard.md
"""
import re
import os
import sys
from collections import defaultdict

ROOT = os.environ.get('CDP_RESULT_DIR', 'D:/fsdownload')
BASE_FILE = os.path.join(ROOT, 'baseline_results.txt')
FILES = [
    (os.path.join(ROOT, 'v4_results.txt'), ''),
    (os.path.join(ROOT, 'stage1_results.txt'), ''),
    (os.path.join(ROOT, 'handtune_results.txt'), ''),
]

LINE = re.compile(r'^(\S+)\s+h(\d+)\s+(\S+)\s+mse:([\d.]+),\s*mae:([\d.]+)')
BASE = re.compile(r'^(\S+)\s+h(\d+)\s+baseline\s+mse:([\d.]+),\s*mae:([\d.]+)')

# 标签 -> (协议, 说明)
METADATA = [
    ('v5final',  'final',   'Stage-2 cal5 (最终配方)'),
    ('at_cal1',  'final',   'Stage-2 cal1 (对照)'),
    ('at_sw1',   'final',   'cal5 + step-weight 1.0'),
    ('at_sw2',   'final',   'cal5 + step-weight 2.0'),
    ('at_st05',  'final',   'cal5 + st_lr_ratio 0.5'),
    ('at_st10',  'final',   'cal5 + st_lr_ratio 1.0'),
    ('at_2nd',   'final',   'cal5 @ 次优骨干'),
    ('lt_both',  'final',   'cal5 + step-weight 1.0 (探针)'),
    ('lt_cal5',  'final',   'cal5 (探针)'),
    ('lt_sw',    'final',   'step-weight 1.0 (探针)'),
    ('lt_wd0',   'final',   'wd=0 (探针)'),
    ('cdp_on_st1', 'old',   '两阶段 (早期配方)'),
    ('cdp_on_clean', 'old', 'CDP @ 作者骨干'),
    ('stack_l1', 'explore', 'dynamic+纯L1 骨干 + CDP(L1)'),
    ('stack_',   'explore', '叠加实验'),
    ('r1_',      'explore', 'v4 筛选'),
    ('r2_',      'explore', 'v4 筛选'),
    ('r3_',      'explore', 'v4 筛选'),
    ('r4_',      'explore', 'v4 筛选'),
    ('r5_',      'explore', 'v4 筛选'),
    ('r6_',      'explore', 'v4 筛选'),
    ('a_',       'explore', 'v4 消融'),
    ('f_win',    'explore', 'v4 final 旧配方 f_win'),
    ('f_all',    'explore', 'v4 final 旧配方 f_all'),
    ('st1_',     'stage1',  'Stage-1 骨干（无 CDP）'),
    ('handtuned', 'infer',  '手调权重纯推理'),
]


def classify(tag):
    for key, proto, desc in METADATA:
        if tag.startswith(key):
            return proto, desc
    return 'unknown', ''


def read_baselines():
    out = {}
    if not os.path.exists(BASE_FILE):
        return out
    for ln in open(BASE_FILE, encoding='utf-8', errors='ignore'):
        m = BASE.match(ln.strip())
        if m:
            out[(m.group(1), int(m.group(2)))] = (float(m.group(3)), float(m.group(4)))
    return out


def read_candidates():
    out = []
    for path, _ in FILES:
        if not os.path.exists(path):
            continue
        for ln in open(path, encoding='utf-8', errors='ignore'):
            m = LINE.match(ln.strip())
            if not m:
                continue
            ds, h, tag = m.group(1), int(m.group(2)), m.group(3)
            # 同一个 tag 可能被重复写入，保留最后一次（覆盖式）
            out = [c for c in out if not (c['ds'] == ds and c['h'] == h and c['tag'] == tag)]
            out.append(dict(ds=ds, h=h, tag=tag,
                            mse=float(m.group(4)), mae=float(m.group(5))))
    return out


def main():
    base = read_baselines()
    cands = read_candidates()
    groups = defaultdict(list)
    for c in cands:
        proto, desc = classify(c['tag'])
        c['proto'], c['desc'] = proto, desc
        b = base.get((c['ds'], c['h']))
        if b:
            c['dmse'] = (c['mse'] / b[0] - 1) * 100
            c['dmae'] = (c['mae'] / b[1] - 1) * 100
        else:
            c['dmse'] = c['dmae'] = None
        # score/avg 用「提升量」（正数=比基线好），越大越好
        c['score'] = min(-c['dmse'], -c['dmae']) if c['dmse'] is not None else -1e9
        c['avg'] = (-c['dmse'] - c['dmae']) / 2 if c['dmse'] is not None else -1e9
        groups[(c['ds'], c['h'])].append(c)

    order = {'ETTh1': 0, 'ETTh2': 1, 'ETTm1': 2, 'ETTm2': 3, 'weather': 4, 'ECL': 5, 'Traffic': 6}
    lines = ['# CDP × TimeMixer++ 逐设定选优榜', '',
             '打分准则：score = min(ΔMSE%, ΔMAE%)（先保证两项都不拖后腿），并列看均值。',
             '负值代表该项相对论文干净基线退步。', '']
    sel = []
    for (ds, h) in sorted(groups, key=lambda k: (order.get(k[0], 9), k[1])):
        b = base.get((ds, h))
        rows = [c for c in groups[(ds, h)] if c['dmse'] is not None]
        if not rows:
            continue
        # 全部候选一起比（不区分协议）：用户要求「谁效果最好就用谁」。
        # 旧协议候选若分数更高，照样本着选定，仅用标记提示协议差异。
        ranked = sorted(rows, key=lambda c: (-c['score'], -c['avg']))
        best = ranked[0]
        fallback = ''
        if best['proto'] != 'final':
            fallback = '（注意：旧协议候选，最终表前需在固定协议下复跑确认）'
        fin = ranked
        other = []
        sel.append((ds, h, b, best))
        lines.append('## %s h%d' % (ds, h))
        if b:
            lines.append('基线(论文)：MSE %.5f / MAE %.5f' % (b[0], b[1]))
        lines.append('')
        lines.append('| 候选 | 协议 | MSE | MAE | ΔMSE | ΔMAE | score |')
        lines.append('|---|---|---|---|---|---|---|')
        for c in fin[:6]:
            lines.append('| %s | %s | %.5f | %.5f | %+.2f%% | %+.2f%% | %+.2f |' %
                         (c['tag'], c['proto'], c['mse'], c['mae'], c['dmse'], c['dmae'], c['score']))
        lines.append('')
        lines.append('**选定：%s（%s）%s**' % (best['tag'], best['desc'], fallback))
        lines.append('')

    # 汇总
    lines.append('## 汇总')
    lines.append('')
    lines.append('| 数据集 | h | 基线 MSE/MAE | 选定候选 | 最佳 MSE/MAE | ΔMSE | ΔMAE |')
    lines.append('|---|---|---|---|---|---|---|')
    n = n5 = 0
    s_mse = s_mae = 0.0
    for ds, h, b, c in sel:
        n += 1
        s_mse += c['dmse']
        s_mae += c['dmae']
        if c['dmse'] <= -5 and c['dmae'] <= -5:
            n5 += 1
        lines.append('| %s | %d | %.5f / %.5f | %s | %.5f / %.5f | %+.2f%% | %+.2f%% |' %
                     (ds, h, b[0], b[1], c['tag'], c['mse'], c['mae'], c['dmse'], c['dmae']))
    if n:
        lines.append('')
        lines.append('平均：ΔMSE %+.2f%% / ΔMAE %+.2f%%；'
                     '两项同时≥5%% 的设定：%d/%d' % (s_mse / n, s_mae / n, n5, n))
    txt = '\n'.join(lines)
    out = os.path.join(ROOT, 'leaderboard.md')
    with open(out, 'w', encoding='utf-8') as f:
        f.write(txt + '\n')
    print(txt)
    print('\n[saved] ' + out)


if __name__ == '__main__':
    main()
