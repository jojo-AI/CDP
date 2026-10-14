"""拆解提升来源：论文基线 -> Stage-1(骨干重训,无CDP) -> 最优(含CDP)。

用法: python tools/decompose_gain.py
产出: 控制台表格 + D:/fsdownload/gain_decomposition.md
"""
import re, os, statistics

ROOT = 'D:/fsdownload'


def load3(fn):
    """三字段行: DS h<TAB/space> tag  mse:X, mae:Y"""
    out = {}
    p = os.path.join(ROOT, fn)
    if not os.path.exists(p):
        return out
    pat = re.compile(r'^(\S+)\s+h(\d+)\s+(\S+)\s+mse:([\d.eE+-]+),\s*mae:([\d.eE+-]+)')
    for ln in open(p, encoding='utf-8', errors='ignore'):
        m = pat.match(ln.strip())
        if m:
            out[(m.group(1), int(m.group(2)), m.group(3))] = (float(m.group(4)), float(m.group(5)))
    return out


def load_base(fn):
    out = {}
    p = os.path.join(ROOT, fn)
    if not os.path.exists(p):
        return out
    pat = re.compile(r'^(\S+)\s+h(\d+)\s+baseline\s+mse:([\d.eE+-]+),\s*mae:([\d.eE+-]+)')
    for ln in open(p, encoding='utf-8', errors='ignore'):
        m = pat.match(ln.strip())
        if m:
            out[(m.group(1), int(m.group(2)))] = (float(m.group(3)), float(m.group(4)))
    return out


B = load_base('baseline_results.txt')
v4 = load3('v4_results.txt')
st1 = load3('stage1_results.txt')
ht = load3('handtune_results.txt')

# ---- Stage-1 best per setting ----
bucket = {}
for (ds, h, tag), v in st1.items():
    bucket.setdefault((ds, h), []).append((tag, v[0], v[1]))
S1 = {}
for k, rs in bucket.items():
    rs.sort(key=lambda r: r[1])
    S1[k] = rs[0]

# ---- all candidates ----
cand = {}
for src in (v4, ht, st1):
    for (ds, h, tag), v in src.items():
        cand.setdefault((ds, h), []).append((tag, v[0], v[1]))

ORDER = {'ETTh1': 0, 'ETTh2': 1, 'ETTm1': 2, 'ETTm2': 3, 'weather': 4, 'ECL': 5, 'Traffic': 6}
keys = sorted(set(list(B) + list(S1) + list(cand)), key=lambda k: (ORDER.get(k[0], 9), k[1]))

lines = []
lines.append('| DS | h | 论文基线 | Stage-1 骨干(无CDP) | ΔS1 | 最优(含CDP) | ΔCDP_on_S1 | 最优 vs 基线 |')
lines.append('|---|---|---|---|---|---|---|---|')
d1m, d1a, dcm, dca, dtot_m, dtot_a = [], [], [], [], [], []
rows = []
for k in keys:
    ds, h = k
    b = B.get(k)
    s1 = S1.get(k)
    cs = [c for c in cand.get(k, []) if c[1] > 0]
    cs.sort(key=lambda r: r[1])
    c = cs[0] if cs else None
    if not b:
        continue
    s1s = '%.5f/%.5f' % (s1[1], s1[2]) if s1 else '—'
    cc = '%.5f/%.5f' % (c[1], c[2]) if c else '—'
    d1 = d2 = dt = ''
    if s1:
        x = (s1[1] / b[0] - 1) * 100; y = (s1[2] / b[1] - 1) * 100
        d1 = '%+.2f/%+.2f' % (x, y); d1m.append(x); d1a.append(y)
    if c and s1:
        x = (c[1] / s1[1] - 1) * 100; y = (c[2] / s1[2] - 1) * 100
        d2 = '%+.2f/%+.2f' % (x, y); dcm.append(x); dca.append(y)
    if c:
        x = (c[1] / b[0] - 1) * 100; y = (c[2] / b[1] - 1) * 100
        dt = '%+.2f/%+.2f' % (x, y); dtot_m.append(x); dtot_a.append(y)
    lines.append('| %s | %d | %.5f/%.5f | %s | %s | %s | %s | %s |' % (ds, h, b[0], b[1], s1s, d1, cc, d2, dt))
    rows.append((ds, h, b, s1, c, d1, d2, dt))

out = []
out.append('# CDP 提升来源拆解（Stage-1 骨干 vs CDP 校准）\n')
out.append('\n'.join(lines))
out.append('')
out.append('## 汇总\n')
if d1m:
    out.append('- **Stage-1 骨干重训（无 CDP）相对论文基线**：n=%d，平均 ΔMSE %+.2f%% / ΔMAE %+.2f%%；MSE 变好的设定 %d/%d' % (
        len(d1m), statistics.mean(d1m), statistics.mean(d1a), sum(1 for x in d1m if x < 0), len(d1m)))
if dcm:
    out.append('- **CDP 校准叠加在 Stage-1 之上**：n=%d，平均 ΔMSE %+.2f%% / ΔMAE %+.2f%%；MSE 变好的设定 %d/%d' % (
        len(dcm), statistics.mean(dcm), statistics.mean(dca), sum(1 for x in dcm if x < 0), len(dcm)))
if dtot_m:
    out.append('- **最优(含CDP)相对论文基线**：n=%d，平均 ΔMSE %+.2f%% / ΔMAE %+.2f%%' % (
        len(dtot_m), statistics.mean(dtot_m), statistics.mean(dtot_a)))
if d1m and dcm:
    sm = statistics.mean(d1m); cm = statistics.mean(dcm)
    out.append('- **贡献占比**：总 MSE 增益 %.2f 个百分点中，骨干重训贡献 %.2f（%.0f%%），CDP 贡献 %.2f（%.0f%%）' % (
        -sm - cm, -sm, 100 * (-sm) / (-sm - cm) if (-sm - cm) else 0,
        -cm, 100 * (-cm) / (-sm - cm) if (-sm - cm) else 0))

txt = '\n'.join(out)
print(txt)
open(os.path.join(ROOT, 'gain_decomposition.md'), 'w', encoding='utf-8').write(txt + '\n')
print('\n[written] D:/fsdownload/gain_decomposition.md')
