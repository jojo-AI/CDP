"""列出指定数据集每个 (DS,h) 的全部候选及其相对基线的提升幅度。"""
import re, os, sys

ROOT = 'D:/fsdownload'


def load(fn, pat, ngroup=5):
    out = {}
    p = os.path.join(ROOT, fn)
    if not os.path.exists(p):
        return out
    for ln in open(p, encoding='utf-8', errors='ignore'):
        m = re.match(pat, ln.strip())
        if m:
            out[(m.group(1), int(m.group(2)), m.group(3))] = (float(m.group(4)), float(m.group(5)))
    return out


def load_base(fn):
    out = {}
    p = os.path.join(ROOT, fn)
    if not os.path.exists(p):
        return out
    for ln in open(p, encoding='utf-8', errors='ignore'):
        m = re.match(r'^(\S+)\s+h(\d+)\s+\S+\s+mse:([\d.]+),\s*mae:([\d.]+)', ln.strip())
        if m:
            out[(m.group(1), int(m.group(2)))] = (float(m.group(3)), float(m.group(4)))
    return out


base = load_base('baseline_results.txt')
cands = {}
for fn in ('v4_results.txt', 'stage1_results.txt'):
    for (ds, h, tag), v in load(fn, r'^(\S+)\s+h(\d+)\s+(\S+)\s+mse:([\d.]+),\s*mae:([\d.]+)').items():
        cands.setdefault((ds, h), []).append((tag, v[0], v[1]))

want = sys.argv[1:] or None
for k in sorted(base, key=lambda x: (x[0], x[1])):
    if want and k[0] not in want:
        continue
    b = base[k]
    print('%-8s h%-4d baseline %.5f/%.5f' % (k[0], k[1], b[0], b[1]))
    rs = cands.get(k, [])
    rs.sort(key=lambda r: -min(-(r[1] / b[0] - 1) * 100, -(r[2] / b[1] - 1) * 100))
    for tag, mse, mae in rs:
        dm = (mse / b[0] - 1) * 100
        da = (mae / b[1] - 1) * 100
        mark = '  <== BEST' if (tag, mse, mae) == rs[0] else ''
        print('        %-34s %.5f/%.5f  %+7.2f%% %+7.2f%%%s' % (tag[:34], mse, mae, dm, da, mark))
    if not rs:
        print('        (无候选)')
