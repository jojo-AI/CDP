# -*- coding: utf-8 -*-
"""
TimeKAN 侧 cdp_loss 敏感性对照实验（2026-10-15）

设计要点
--------
* 只扫描 **一个因子**：stage-2 训练损失 cdp_loss ∈ {l1(已有), mse(本次新增)}。
  理由：CycleNet 侧三因子扫描已证明 cdp_lr / feat_k 均不敏感，
  且 TimeKAN 的 PSC 特征窗口硬编码 k=min(16, seq_len-1)=16，与 CycleNet 默认值相同
  ⇒ 那两项结论已覆盖 TimeKAN，无需重测。
* 对照组 **零成本**：TimeKAN 主实验已有 8 条 weather/traffic × 4 horizon 的
  head（L1）结果（seed2021），同数据集/同 horizon/同 seed/同骨干 ckpt 严格配对。
* 骨干严格复用：head 模式加载 cdp_ckpt/ 下既有的 baseline ckpt，不重训、不覆盖。
* 结果落到 cdp_results/ 并带后缀 _lossMSE，**不会覆盖任何既有结果**。
* 幂等：结果已存在则跳过（可安全续跑）。
"""
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
RES = os.path.join(ROOT, "cdp_results")

# (data, enc_in)：weather 21ch 快，traffic 862ch 慢
JOBS = [("weather", 21), ("traffic", 862)]
HORIZONS = (96, 192, 336, 720)
SEED = 2021
MODE = "head"
SUFFIX = "_lossMSE"   # 与 CycleNet 侧命名口径一致


def log(msg):
    line = "[%s] %s" % (time.strftime("%H:%M:%S"), msg)
    print(line, flush=True)
    with open(os.path.join(ROOT, "batch_log_loss.txt"), "a", encoding="utf-8") as f:
        f.write(line + "\n")


def result_path(data, h):
    return os.path.join(RES, "cdp_%s_%s_%d_seed%d%s.json" % (MODE, data, h, SEED, SUFFIX))


def run_one(data, enc_in, h, idx, total):
    out = result_path(data, h)
    if os.path.exists(out):
        log("SKIP %s-%d (exists)" % (data, h))
        return True
    logfile = os.path.join(ROOT, "run_loss_%02d_%s_%d.log" % (idx, data, h))
    cmd = [PY, os.path.join(ROOT, "cdp_timekan_gpu.py"),
           "--data", data, "--pred_len", str(h), "--enc_in", str(enc_in),
           "--mode", MODE, "--seed", str(SEED),
           "--cdp_loss", "mse", "--suffix", SUFFIX]
    log("START [%d/%d] %s-%d" % (idx, total, data, h))
    t0 = time.time()
    with open(logfile, "w", encoding="utf-8", errors="replace") as lf:
        rc = subprocess.call(cmd, cwd=ROOT, stdout=lf, stderr=subprocess.STDOUT)
    dt = (time.time() - t0) / 60.0
    if rc == 0 and os.path.exists(out):
        log("DONE  [%d/%d] %s-%d  %.1f min" % (idx, total, data, h, dt))
        return True
    log("FAIL  [%d/%d] %s-%d  rc=%d  %.1f min  (see %s)" % (
        idx, total, data, h, rc, dt, os.path.basename(logfile)))
    return False


def main():
    tasks = [(d, c, h) for d, c in JOBS for h in HORIZONS]
    total = len(tasks)
    log("=" * 60)
    log("TimeKAN cdp_loss=mse sensitivity: %d runs (seed %d)" % (total, SEED))
    log("=" * 60)
    ok = fail = 0
    for i, (d, c, h) in enumerate(tasks, 1):
        if run_one(d, c, h, i, total):
            ok += 1
        else:
            fail += 1
    log("=== ALL DONE === ok=%d fail=%d skipped_or_done=%d" % (ok, fail, total - ok - fail))
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
