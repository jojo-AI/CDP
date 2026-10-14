# -*- coding: utf-8 -*-
"""ETTh1-720 多种子分布 + 需要多少种子才能把均值误差压到目标范围。

输入：multiseed_results.txt（含 seed2021 原始值，由 --seed-2021 手动注入）
"""
import math
import re

BASE_MSE, BASE_MAE = 0.4998258257669116, 0.4789550054640997

# (seed, Stage-1 骨干, Stage-2 最终)
DATA = {
    "ETTh1-720": [
        (2021, (0.4820203236171177, 0.46850646488250247), (0.4710809662228539, 0.45393077305385043)),
        (2022, (0.552621587117513, 0.4916414836096385), (0.5262056456671821, 0.4818116142636254)),
        (2023, (0.4667037025330559, 0.4586229929848323), (0.48403840594821507, 0.46213650779118615)),
    ],
    "ETTm1-96": [
        (2021, (0.31407059788895264, 0.3430459408788135), (0.31257213731064654, 0.34163291317820865)),
        (2022, (0.31798844480335936, 0.34280015084117155), (0.3132224919771753, 0.3403794736132512)),
    ],
    "weather-96": [
        (2021, (0.16441425636144985, 0.20034302100933418), (0.1620854895511562, 0.19846980655377877)),
        (2022, (0.16427346409336624, 0.19972603082329812), (0.16461510675083812, 0.19986093247518477)),
    ],
}
BASE = {"ETTh1-720": (BASE_MSE, BASE_MAE),
        "ETTm1-96": (0.3324017309134755, 0.36916656580884677),
        "weather-96": (0.16853407913625135, 0.21495861505354805)}


def pct(v, b):
    return (v - b) / b * 100.0


def mean(xs):
    return sum(xs) / len(xs)


def sd(xs):
    if len(xs) < 2:
        return float("nan")
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


for name, rows in DATA.items():
    bm, ba = BASE[name]
    print("=" * 84)
    print("### %s   （基线 %.5f / %.5f）" % (name, bm, ba))
    fms, fas, s1 = [], [], []
    for seed, st1, st2 in rows:
        dm, da = pct(st2[0], bm), pct(st2[1], ba)
        cdm, cda = pct(st2[0], st1[0]), pct(st2[1], st1[1])
        fms.append(st2[0]); fas.append(st2[1]); s1.append(st1[0])
        print("   seed %d  骨干 %.5f → 最终 %.5f/%.5f   ΔMSE %+6.2f%%  ΔMAE %+6.2f%%   | CDP同骨干 ΔMSE %+5.2f%% ΔMAE %+5.2f%%"
              % (seed, st1[0], st2[0], st2[1], dm, da, cdm, cda))
    n = len(fms)
    print("   ---- n=%d ----" % n)
    print("   最终 MSE 均值 %.5f (Δ%+.2f%%)   sd=%.5f  CV=%.2f%%"
          % (mean(fms), pct(mean(fms), bm), sd(fms), sd(fms) / mean(fms) * 100))
    print("   最终 MAE 均值 %.5f (Δ%+.2f%%)   sd=%.5f  CV=%.2f%%"
          % (mean(fas), pct(mean(fas), ba), sd(fas), sd(fas) / mean(fas) * 100))
    print("   骨干 MSE 跨种子：min %.5f max %.5f  极差 %.2f%%" % (min(s1), max(s1), (max(s1) - min(s1)) / min(s1) * 100))
    pos = sum(1 for x in fms if x < bm)
    print("   最终 MSE 优于基线的种子数：%d/%d" % (pos, n))
    cdm = [pct(r[2][0], r[1][0]) for r in rows]
    cda = [pct(r[2][1], r[1][1]) for r in rows]
    print("   ★ CDP 同骨干贡献（%d/%d 为改善）：ΔMSE 均 %+.2f%%  ΔMAE 均 %+.2f%%"
          % (sum(1 for x in cdm if x < 0), n, mean(cdm), mean(cda)))

# 需要多少种子才能把均值的标准误压到目标
print()
print("=" * 84)
print("### 要把 ETTh1-720 的『均值的标准误』压到目标，需要多少种子？")
cv_m = sd([r[2][0] for r in DATA["ETTh1-720"]]) / mean([r[2][0] for r in DATA["ETTh1-720"]])
cv_a = sd([r[2][1] for r in DATA["ETTh1-720"]]) / mean([r[2][1] for r in DATA["ETTh1-720"]])
print("   实测（n=3）单次 CV：MSE %.2f%% / MAE %.2f%%" % (cv_m * 100, cv_a * 100))
print("   %-14s %-12s %-12s %s" % ("目标 SE", "需 n (MSE)", "需 n (MAE)", "估算耗时(ETTh1-720)"))
for tgt in (2.0, 1.5, 1.0, 0.5):
    nm = (cv_m / (tgt / 100)) ** 2
    na = (cv_a / (tgt / 100)) ** 2
    print("   ±%.1f%%         %-12s %-12s %.1f h（3.4 min/种子）" % (tgt, "%.0f" % nm, "%.0f" % na, max(nm, na) * 3.4 / 60))
