# -*- coding: utf-8 -*-
"""汇总 mk_results/*.json 为对照表（只读，可在任意机器跑）。"""
import glob
import json
import os
import sys

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(PKG, "mk_results")

rows = []
for p in glob.glob(os.path.join(RES, "mk_*.json")):
    try:
        j = json.load(open(p, encoding="utf-8"))
    except Exception:
        continue
    if j.get("status") != "ok" or j.get("mse") is None:
        continue
    if j["axis"] == "M":
        rows.append(("M", float(j["seg_ratio"]), j.get("cdp_M"), j["mse"], j["mae"], p))
    else:
        rows.append(("K", float(j["K"]), j.get("cdp_M"), j["mse"], j["mae"], p))

rows.sort(key=lambda t: (t[0], t[1]))
print("axis  value  cdp_M  mse        mae        file")
for ax, v, m, mse, mae, p in rows:
    print("%-4s  %-5g  %-6s  %-10s %-10s %s" %
          (ax, v, m, mse, mae, os.path.basename(p)))
print("\n共 %d 个有效点" % len(rows))
if not rows:
    print("mk_results 下暂无有效结果。", file=sys.stderr)
