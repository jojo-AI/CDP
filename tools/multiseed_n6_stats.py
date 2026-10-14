# -*- coding: utf-8 -*-
"""n=6 多种子完整统计：逐设定 mean±sd、within-backbone CDP 贡献、符号检验、可分辨性。"""
import re, os, statistics as st, math
import sys

# 2026-10-08 修复：Windows 计划任务/非交互环境下 stdout 默认编码是 GBK，
# 打印 ✅/❌ (U+2705/U+274C) 会抛 UnicodeEncodeError 直接崩掉整个脚本
# （表现：收尾脚本报 "ERR n6 统计刷新失败"，进而不关机）。
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

RES = 'D:/fsdownload/multiseed_results.txt'
BASE = {  # 作者发布 ckpt 评估的干净基线（MSE, MAE）
    ('ETTh1', '720'): (0.49983, 0.47896),
    ('ETTm1', '96'): (0.3324017309134755, 0.36916656580884677),
    ('weather', '96'): (0.16853, 0.21496),
}
# Stage-1 / Stage-2 的 tag 模板（results 文件里的写法）
CFG = {
    'ETTh1-720': dict(ds='ETTh1', h='720', st1='ms_ETTh1_720_st1_s%d', st2='ms_lt_cal5_s%d'),
    'ETTm1-96':  dict(ds='ETTm1', h='96',  st1='ms_ETTm1_96_st1_s%d',  st2='ms_cdp_cal5_s%d'),
    'weather-96':dict(ds='weather', h='96',st1='ms_weather_96_st1_s%d',st2='ms_at_sw1_s%d'),
}
SEEDS = [2021, 2022, 2023, 2024, 2025, 2026]

# seed 2021 = 原始 28 设定那一轮，不在 multiseed_results.txt 里，从原始结果文件取
SEED2021 = {
    'ETTh1-720':  ((0.4820203236171177, 0.46850646488250247),  (0.4710809662228539, 0.45393077305385043)),
    'ETTm1-96':   ((0.31407059788895264, 0.3430459408788135),  (0.31257213731064654, 0.34163291317820865)),
    'weather-96': ((0.16441425636144985, 0.20034302100933418), (0.1620854895511562, 0.19846980655377877)),
}

def load():
    d = {}
    for ln in open(RES, encoding='utf-8', errors='ignore'):
        m = re.match(r'^(\S+) h(\d+) (\S+)\s+mse:([\d.]+), mae:([\d.]+)', ln)
        if m:
            d.setdefault((m.group(1), m.group(2)), {})[m.group(3)] = (float(m.group(4)), float(m.group(5)))
    return d

D = load()

def get_pair(name, c, s):
    """返回 (stage1, stage2)；seed 2021 走原始结果文件的硬编码值。"""
    if s == 2021:
        return SEED2021[name]
    k = (c['ds'], c['h'])
    return D[k].get(c['st1'] % s), D[k].get(c['st2'] % s)

print('=' * 100)
print('一、逐设定 × 6 种子：最终指标与相对基线')
print('=' * 100)
all_dm, all_da = [], []
for name, c in CFG.items():
    b = BASE[(c['ds'], c['h'])]
    print('\n### %s   （基线 %.5f / %.5f）' % (name, b[0], b[1]))
    print('%-6s %-22s %-22s %-24s' % ('种子', 'Stage-1 骨干', 'Stage-2 最终', 'ΔMSE / ΔMAE'))
    fins, dm, da, st1s, fina = [], [], [], [], []
    for s in SEEDS:
        s1, s2 = get_pair(name, c, s)
        if s2 is None:
            print('%-6d 缺失' % s); continue
        st1s.append(s1[0])
        fins.append(s2[0]); fina.append(s2[1])
        m = (s2[0] - b[0]) / b[0] * 100
        a = (s2[1] - b[1]) / b[1] * 100
        dm.append(m); da.append(a)
        print('%-6d %-22s %-22s %+8.2f%% / %+8.2f%%%s' % (
            s, '%.5f / %.5f' % s1, '%.5f / %.5f' % s2, m, a,
            '' if m < 0 else '   ← 劣于基线'))
    n = len(dm)
    print('%-6s %-22s %-22s %+8.2f%% / %+8.2f%%   ← 均值(n=%d)' % (
        '均值', '—', '%.5f / %.5f' % (st.mean(fins), st.mean(fina)), st.mean(dm), st.mean(da), n))
    print('   sd(pp)：ΔMSE %.2f  ΔMAE %.2f ｜ 最终 MSE 极差 %.2f%%  CV %.2f%%' % (
        st.stdev(dm), st.stdev(da), (max(fins) - min(fins)) / min(fins) * 100,
        st.stdev(fins) / st.mean(fins) * 100))
    print('   Stage-1 骨干 MSE 极差 %.2f%%（CV %.2f%%）  →  校准后 %.2f%%（CV %.2f%%）  %s' % (
        (max(st1s) - min(st1s)) / min(st1s) * 100, st.stdev(st1s) / st.mean(st1s) * 100,
        (max(fins) - min(fins)) / min(fins) * 100, st.stdev(fins) / st.mean(fins) * 100,
        '✅ 骨干稳' if (max(st1s) - min(st1s)) / min(st1s) * 100 < 1.5 else '❌ 骨干抖'))
    print('   优于基线的种子数： MSE %d/%d   MAE %d/%d' % (
        sum(1 for x in dm if x < 0), n, sum(1 for x in da if x < 0), n))
    all_dm += dm; all_da += da

print('\n' + '=' * 100)
print('二、within-backbone CDP 贡献（同种子 Stage-1 → Stage-2，逐配对）')
print('=' * 100)
pairs = []
for name, c in CFG.items():
    print('\n### %s' % name)
    gm, ga = [], []
    for s in SEEDS:
        s1, s2 = get_pair(name, c, s)
        if not s1 or not s2: continue
        m = (s2[0] - s1[0]) / s1[0] * 100
        a = (s2[1] - s1[1]) / s1[1] * 100
        gm.append(m); ga.append(a); pairs.append((name, s, s1[0], s2[0], m, a))
        print('   seed %d  %s → %s   ΔMSE %+7.2f%%  ΔMAE %+7.2f%%%s' % (
            s, '%.5f' % s1[0], '%.5f' % s2[0], m, a, '' if m < 0 else '   ← 变差'))
    print('   均值 ΔMSE %+.2f%%  ΔMAE %+.2f%%' % (st.mean(gm), st.mean(ga)))

pm = [p[4] for p in pairs]; pa = [p[5] for p in pairs]
print('\n★ 全 %d 配对：MSE 改善 %d/%d ｜ MAE 改善 %d/%d' % (
    len(pairs), sum(1 for x in pm if x < 0), len(pm), sum(1 for x in pa if x < 0), len(pa)))

def sign_p(k, n):
    """双尾符号检验（p0=0.5，精确枚举）。k = 改善（Δ<0）的个数。"""
    if n == 0:
        return float('nan')
    tail = sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n   # P(X >= k)
    return min(1.0, 2 * tail)

print('   符号检验（双尾）：MSE p = %.4g ｜ MAE p = %.4g' % (
    sign_p(sum(1 for x in pm if x < 0), len(pm)), sign_p(sum(1 for x in pa if x < 0), len(pa))))

print('\n' + '=' * 100)
print('三、全 %d 个「设定×种子」配对的整体口径' % len(all_dm))
print('=' * 100)
print('   ΔMSE：均值 %+.2f%%  中位 %+.2f%%  sd %.2fpp  ｜ 优于基线 %d/%d' % (
    st.mean(all_dm), st.median(all_dm), st.stdev(all_dm), sum(1 for x in all_dm if x < 0), len(all_dm)))
print('   ΔMAE：均值 %+.2f%%  中位 %+.2f%%  sd %.2fpp  ｜ 优于基线 %d/%d' % (
    st.mean(all_da), st.median(all_da), st.stdev(all_da), sum(1 for x in all_da if x < 0), len(all_da)))
print('   符号检验（双尾）：MSE p = %.4g ｜ MAE p = %.4g' % (
    sign_p(sum(1 for x in all_dm if x < 0), len(all_dm)), sign_p(sum(1 for x in all_da if x < 0), len(all_da))))
print('\n   噪声下限（同种子复现，实测）：MSE 1.0–1.2% / MAE 0.4–0.75%')
print('   |ΔMSE| 超出 1.2%% 的配对：%d/%d ｜ |ΔMAE| 超出 0.75%% 的配对：%d/%d' % (
    sum(1 for x in all_dm if abs(x) >= 1.2), len(all_dm),
    sum(1 for x in all_da if abs(x) >= 0.75), len(all_da)))

# --------------------------------------------------------------------------- #
print('\n' + '=' * 100)
print('四、机制：CDP 是"修补层"吗？—— 骨干质量 vs CDP 相对收益')
print('=' * 100)

def corr(xs, ys):
    n = len(xs); mx = sum(xs) / n; my = sum(ys) / n
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs)); sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if sx == 0 or sy == 0: return float('nan')
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (sx * sy)

def rankcorr(xs, ys):
    def rank(v):
        s = sorted(range(len(v)), key=lambda i: v[i]); r = [0] * len(v)
        for pos, i in enumerate(s): r[i] = pos + 1
        return r
    return corr(rank(xs), rank(ys))

st1v = [p[2] for p in pairs]; gain = [p[4] for p in pairs]
print('  全 %d 配对：corr(骨干 MSE, CDP ΔMSE) = %+.3f ｜ Spearman = %+.3f' % (
    len(pairs), corr(st1v, gain), rankcorr(st1v, gain)))

# 只在骨干波动最大的 ETTh1-720 上单看
for nm in ('ETTh1-720', 'ETTm1-96', 'weather-96'):
    sub = [p for p in pairs if p[0] == nm]
    if len(sub) < 3: continue
    print('   %-11s corr = %+.3f ｜ Spearman = %+.3f  （n=%d）' % (
        nm, corr([p[2] for p in sub], [p[4] for p in sub]),
        rankcorr([p[2] for p in sub], [p[4] for p in sub]), len(sub)))

print('\n  按骨干好坏分档（全 18 配对）：')
med = st.median(st1v)
for lab, sel in (('骨干好（≤ 中位）', [p for p in pairs if p[2] <= med]),
                 ('骨干差（> 中位）', [p for p in pairs if p[2] > med])):
    if not sel: continue
    print('   %-16s n=%2d  CDP ΔMSE 均值 %+6.2f%%   中位 %+6.2f%%' % (
        lab, len(sel), st.mean([p[4] for p in sel]), st.median([p[4] for p in sel])))
print('  中位骨干 MSE = %.5f' % med)

print('\n  逐设定：骨干 CV → 校准后 CV（校准是否收窄跨种子离散）')
for nm in ('ETTh1-720', 'ETTm1-96', 'weather-96'):
    sub = [p for p in pairs if p[0] == nm]
    a = [p[2] for p in sub]; b = [p[3] for p in sub]
    ca, cb = st.stdev(a) / st.mean(a) * 100, st.stdev(b) / st.mean(b) * 100
    print('   %-11s %.2f%% → %.2f%%  %s' % (nm, ca, cb, '✅ 收窄' if cb < ca else '⚠️ 未收窄'))
