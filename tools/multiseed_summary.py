#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
多种子补跑结果汇总（2026-10-08 新增）

把「已有的 seed-2021 单次结果」与「新补的 seed 2022/2023」拼成 mean ± std 表，
并逐设定判定：相对论文干净基线的提升是否还站得住。

数据来源：
  - D:/fsdownload/multiseed_results.txt        新种子（本脚本只读，缺数据也能跑）
  - D:/fsdownload/v4_results.txt               seed-2021 的 Stage-2 winner 行
  - D:/fsdownload/stage1_results.txt           seed-2021 的 Stage-1 骨干行

用法：python tools/multiseed_summary.py [--out D:/fsdownload/multiseed_summary.md]
"""
import os
import re
import csv
import argparse
import statistics

MS = 'D:/fsdownload/multiseed_results.txt'
V4 = 'D:/fsdownload/v4_results.txt'
ST1 = 'D:/fsdownload/stage1_results.txt'

# 逐设定配置：new_tag = multiseed_results.txt 里的 tag；
#            s2021 = (文件, 匹配该文件里 seed-2021 行的正则)
SETTINGS = [
    {
        'key': 'ETTh1-720', 'ds': 'ETTh1', 'h': 720, 'cand': 'lt_cal5',
        'base': (0.49983, 0.47896),
        'new_tag': 'ms_lt_cal5',
        'st1_new_tag': 'ms_ETTh1_720_st1',
        's2021_file': V4, 's2021_re': r'^ETTh1 h720 lt_cal5 ',
        's2021_st1_file': ST1, 's2021_st1_re': r'^ETTh1 h720 st1_dynamic_l10\.0 ',
    },
    {
        'key': 'ETTm1-96', 'ds': 'ETTm1', 'h': 96, 'cand': 'cdp_cal5',
        'base': (0.33240, 0.36917),
        'new_tag': 'ms_cdp_cal5',
        'st1_new_tag': 'ms_ETTm1_96_st1',
        's2021_file': V4, 's2021_re': r'^ETTm1 h96 cdp_cal5_ETTm1_96_c5_ ',
        's2021_st1_file': ST1, 's2021_st1_re': r'^ETTm1 h96 st1_dynamic_l10\.0 ',
    },
    {
        'key': 'weather-96', 'ds': 'weather', 'h': 96, 'cand': 'at_sw1',
        'base': (0.16853, 0.21496),
        'new_tag': 'ms_at_sw1',
        'st1_new_tag': 'ms_weather_96_st1',
        's2021_file': V4, 's2021_re': r'^weather h96 at_sw1_weather_96_ats1_ ',
        # weather-96 winner 用的是 handtune_init 0 的骨干
        's2021_st1_file': ST1, 's2021_st1_re': r'^weather h96 st1_dynamic_l10\.0_hi0 ',
    },
]

LINE_RE = re.compile(r'^(\S+)\s+h(\d+)\s+(\S+)\s+mse:([\d.]+),\s*mae:([\d.]+)')


def load(path):
    rows = []
    if not os.path.exists(path):
        return rows
    for ln in open(path, encoding='utf-8', errors='ignore'):
        m = LINE_RE.match(ln.strip())
        if m:
            rows.append({'ds': m.group(1), 'h': int(m.group(2)), 'tag': m.group(3),
                         'mse': float(m.group(4)), 'mae': float(m.group(5))})
    return rows


def find(rows, regex):
    p = re.compile(regex)
    hits = [r for r in rows if p.match('%s h%d %s ' % (r['ds'], r['h'], r['tag']))]
    return hits[-1] if hits else None


def seed_of(tag):
    m = re.search(r'_s(\d{4})$', tag)
    return int(m.group(1)) if m else 2021


def pct(a, b):
    return (a - b) / b * 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='D:/fsdownload/multiseed_summary.md')
    a = ap.parse_args()

    new_rows = load(MS)
    v4_rows = load(V4)
    st1_rows = load(ST1)

    L = ['# 多种子结果汇总（Headline 设定）\n',
         '生成时间：%s\n' % __import__('datetime').datetime.now().strftime('%F %T'),
         '协议：与既有 28 设定完全一致（`det_mode=0`），只追加新种子；'
         '每设定完整重跑两阶段 Stage-1 → Stage-2。\n']

    summary = []
    for cfg in SETTINGS:
        L.append('\n## %s　（winner = %s）\n' % (cfg['key'], cfg['cand']))

        # ---------- Stage-2 逐种子 ----------
        runs = []
        s21 = find(v4_rows, cfg['s2021_re'])
        if s21:
            runs.append({'seed': 2021, 'mse': s21['mse'], 'mae': s21['mae'], 'src': 'v4_results.txt'})
        for r in new_rows:
            if r['tag'].startswith(cfg['new_tag']) and r['ds'] == cfg['ds'] and r['h'] == cfg['h']:
                runs.append({'seed': seed_of(r['tag']), 'mse': r['mse'], 'mae': r['mae'],
                             'src': 'multiseed_results.txt'})
        runs.sort(key=lambda x: x['seed'])

        L.append('| 种子 | MSE | MAE | ΔMSE vs 基线 | ΔMAE vs 基线 |')
        L.append('|---|---|---|---|---|')
        for r in runs:
            L.append('| %d | %.5f | %.5f | %+.2f%% | %+.2f%% |'
                     % (r['seed'], r['mse'], r['mae'],
                        pct(r['mse'], cfg['base'][0]), pct(r['mae'], cfg['base'][1])))

        if len(runs) >= 2:
            ms = [r['mse'] for r in runs]
            ma = [r['mae'] for r in runs]
            L.append('')
            L.append('**跨 %d 个种子：MSE %.5f ± %.5f（极差 %.2f%%）｜MAE %.5f ± %.5f（极差 %.2f%%）**'
                     % (len(runs), statistics.mean(ms), statistics.stdev(ms),
                        (max(ms) - min(ms)) / min(ms) * 100,
                        statistics.mean(ma), statistics.stdev(ma),
                        (max(ma) - min(ma)) / min(ma) * 100))
            L.append('')
            L.append('- 相对基线：ΔMSE 均值 %+.2f%%（各种子 %+.2f%% ~ %+.2f%%）｜'
                     'ΔMAE 均值 %+.2f%%（%+.2f%% ~ %+.2f%%）'
                     % (pct(statistics.mean(ms), cfg['base'][0]),
                        pct(max(ms), cfg['base'][0]), pct(min(ms), cfg['base'][0]),
                        pct(statistics.mean(ma), cfg['base'][1]),
                        pct(max(ma), cfg['base'][1]), pct(min(ma), cfg['base'][1])))
            if min(ms) < cfg['base'][0] > max(ms):
                L.append('- ⚠️ **所有种子都优于基线** → MSE 侧结论稳健')
            elif max(ms) < cfg['base'][0]:
                L.append('- ⚠️ 部分种子劣于基线 → MSE 侧结论**不稳**，需谨慎宣称')
            if max(ma) < cfg['base'][1]:
                L.append('- ✅ **所有种子都优于基线** → MAE 侧结论稳健')
            elif min(ma) > cfg['base'][1]:
                L.append('- ❌ MAE 侧全部劣于基线')
            else:
                L.append('- ⚠️ MAE 侧部分种子劣于基线 → 需谨慎宣称')
        else:
            L.append('')
            L.append('（新种子结果尚未产出，待 `%s` 出现后重跑本脚本）' % cfg['new_tag'])

        # ---------- Stage-1 骨干（增益拆解用）----------
        s1runs = []
        if cfg['s2021_st1_file']:
            r = find(st1_rows, cfg['s2021_st1_re'])
            if r:
                s1runs.append({'seed': 2021, 'mse': r['mse'], 'mae': r['mae'], 'src': 'stage1_results.txt'})
        for r in new_rows:
            if r['tag'].startswith(cfg['st1_new_tag']) and r['ds'] == cfg['ds'] and r['h'] == cfg['h']:
                s1runs.append({'seed': seed_of(r['tag']), 'mse': r['mse'], 'mae': r['mae'],
                               'src': 'multiseed_results.txt'})
        s1runs.sort(key=lambda x: x['seed'])
        if s1runs:
            L.append('')
            L.append('Stage-1 骨干（无 CDP）：' +
                     '｜'.join('seed%d %.5f/%.5f' % (r['seed'], r['mse'], r['mae']) for r in s1runs))
            if len(s1runs) >= 2:
                ms = [r['mse'] for r in s1runs]
                L.append('　→ 骨干 MSE 极差 **%.2f%%**（%.5f ~ %.5f）'
                         % ((max(ms) - min(ms)) / min(ms) * 100, min(ms), max(ms)))
        summary.append(cfg['key'])

    L.append('\n---\n\n## 读法\n')
    L.append('- 「跨种子极差」>> 「同种子噪声下限（MSE 1.0–1.2% / MAE 0.4–0.75%）」'
             '说明种子效应占主导，逐设定结论必须用多种子口径。')
    L.append('- 若某设定在部分种子上劣于基线，则该设定的单次结果属于**运气**，'
             '不能写进主表当卖点（改口径：写「在 N 个种子中的 M 个上优于基线」）。')
    L.append('- Stage-1 骨干的跨种子极差若已经很大，说明**增益拆解**本身也不稳：'
             'CDP 的贡献必须在同一骨干内比较，不能跨种子比较。')

    with open(a.out, 'w', encoding='utf-8') as f:
        f.write('\n'.join(L) + '\n')
    print('\n'.join(L))
    print('\n已写出: %s' % a.out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
