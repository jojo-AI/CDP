# -*- coding: utf-8 -*-
"""扫描所有训练日志，统计每个数据/预测长度设定的「每 epoch 步数」。

用途：判定哪些设定属于「短 epoch → val 曲线噪声大 → 选优抖动大」的高风险档
（根因见 D:/fsdownload/determinism_rootcause_report_20261008.md）。
"""
import glob
import os
import re

PAT_STEP = re.compile(r"Epoch: 1, Steps: (\d+)")
PAT_KEY = re.compile(r"(ETTh1|ETTh2|ETTm1|ETTm2|weather|ECL|Traffic)\D*?(96|192|336|720)")

BASE = "D:/fsdownload"


def main():
    best = {}
    for f in glob.glob(os.path.join(BASE, "*.log")):
        b = os.path.basename(f)
        if not b.startswith(("v4_", "st1_", "ms_")):
            continue
        m = PAT_KEY.search(b)
        if not m:
            continue
        try:
            txt = open(f, encoding="utf-8", errors="ignore").read()
        except OSError:
            continue
        s = PAT_STEP.findall(txt)
        if not s:
            continue
        key = (m.group(1), int(m.group(2)))
        best[key] = max(best.get(key, 0), int(s[0]))

    print("%-10s %-6s %s" % ("数据集", "pred_len", "steps/epoch   风险"))
    print("-" * 52)
    for (ds, h), st in sorted(best.items(), key=lambda kv: kv[1]):
        if st < 100:
            flag = "🔴 高风险（<100）"
        elif st < 270:
            flag = "🟡 注意（<270）"
        else:
            flag = "✅ 稳"
        print("%-10s h%-5d %-13d %s" % (ds, h, st, flag))

    vals = sorted(best.values())
    print("\n共 %d 个设定；最小值 %d，中位数 %d，最大值 %d" % (len(vals), vals[0], vals[len(vals) // 2], vals[-1]))
    print("低于 100 步的设定数：%d" % sum(1 for v in best.values() if v < 100))
    print("低于 270 步的设定数：%d" % sum(1 for v in best.values() if v < 270))


if __name__ == "__main__":
    main()
