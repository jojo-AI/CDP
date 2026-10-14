# -*- coding: utf-8 -*-
"""汇总 SWA / 平台期集成实验的结果，对比「跨种子抖动」是否下降。

读取 D:/fsdownload/swa_ens_st{1,2}_s<seed>.json（由 tools/eval_ckpt_ensemble.py 产出），
对每种选点口径算：逐种子 test MSE / MAE、Δ% vs 官方基线、跨种子极差与 CV。
"""
import glob
import json
import os
import re

BASE = {"ETTh1-720": (0.4998258257669116, 0.4789550054640997)}
OUT_MD = "D:/fsdownload/swa_etth1_720_summary.md"

PROTOCOLS = [
    ("protocol_argmin_val", "① 现有协议 argmin(val)"),
    ("protocol_ens_pred", "② 平台期预测平均"),
    ("protocol_ens_weight", "③ 平台期权重平均 SWA"),
    ("ref_oracle_argmin_test", "④ 参考上界 argmin(test)"),
]


def load(stage):
    rows = {}
    for f in sorted(glob.glob("D:/fsdownload/swa_ens_st%d_s*.json" % stage)):
        seed = int(re.search(r"_s(\d+)\.json", f).group(1))
        rows[seed] = json.load(open(f, encoding='utf-8'))
    return rows


def fmt_delta(v, b):
    return "%+.2f%%" % ((v - b) / b * 100.0)


def spread(xs):
    if len(xs) < 2:
        return float('nan'), float('nan')
    return (max(xs) - min(xs)) / min(xs) * 100.0, (max(xs) - min(xs)) / min(xs) * 100.0


lines = ["# SWA / 验证集平台期集成实验 · 汇总", ""]
bm, ba = BASE["ETTh1-720"]
lines.append("官方基线（作者发布 ckpt 评估）：MSE %.5f / MAE %.5f" % (bm, ba))
lines.append("")

for stage, sname in ((2, "Stage-2（CDP 校准，最终指标）"), (1, "Stage-1（骨干）")):
    rows = load(stage)
    if not rows:
        lines.append("## %s —— 尚无结果" % sname)
        lines.append("")
        continue
    seeds = sorted(rows)
    lines.append("## %s" % sname)
    lines.append("")
    lines.append("| 口径 | " + " | ".join("seed %d" % s for s in seeds) + " | 跨种子极差 | CV |")
    lines.append("|---" * (len(seeds) + 3) + "|")
    for key, label in PROTOCOLS:
        vals = []
        for s in seeds:
            r = rows[s].get(key)
            if not r:
                vals.append("—")
                continue
            vals.append("%.5f (%s)" % (r['test_mse'], fmt_delta(r['test_mse'], bm)))
        nums = [rows[s][key]['test_mse'] for s in seeds if key in rows[s]]
        sp, cv = spread(nums)
        lines.append("| %s | %s | %.2f%% | %.2f%% |" % (label, " | ".join(vals), sp, cv))
    lines.append("")
    # MAE 侧
    lines.append("MAE 侧（基线 %.5f）：" % ba)
    lines.append("")
    lines.append("| 口径 | " + " | ".join("seed %d" % s for s in seeds) + " | 跨种子极差 |")
    lines.append("|---" * (len(seeds) + 2) + "|")
    for key, label in PROTOCOLS:
        vals = []
        for s in seeds:
            r = rows[s].get(key)
            vals.append("—" if not r else "%.5f (%s)" % (r['test_mae'], fmt_delta(r['test_mae'], ba)))
        nums = [rows[s][key]['test_mae'] for s in seeds if key in rows[s]]
        sp, _ = spread(nums)
        lines.append("| %s | %s | %.2f%% |" % (label, " | ".join(vals), sp))
    lines.append("")
    # 平台期长度
    lines.append("平台期 epoch 数：" + "、".join(
        "seed %d → %d 个（epoch %s）" % (s, len(rows[s]['plateau_epochs']), rows[s]['plateau_epochs'])
        for s in seeds))
    lines.append("")

md = "\n".join(lines)
print(md)
with open(OUT_MD, 'w', encoding='utf-8') as f:
    f.write(md + "\n")
print("\n已写出 %s" % OUT_MD)
