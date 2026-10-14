# -*- coding: utf-8 -*-
"""
TimeKAN + CDP overnight batch runner (local CPU, 8 cores).

Design:
  - 2 parallel workers x 4 torch threads each (8-core box).
  - One job = (seed, dataset, horizon); a job runs its modes sequentially
    (baseline before head/cao/cdp so the checkpoint dependency holds).
  - Skip-any-result-that-exists -> safe to re-run after reboot/interrupt.
  - Phase 1: baseline+head for seeds 2021..2026 (main table + error bars)
    Phase 2: cao+cdp ablation for seed 2021
    Phase 3: cao+cdp ablation for seeds 2022..2026 (bonus, runs if time allows)
  - Lock file prevents two concurrent runners; STOP file halts between jobs.
"""
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
SCRIPT = os.path.join(ROOT, "cdp_timekan_cpu.py")
RES = os.path.join(ROOT, "cdp_results")
os.makedirs(RES, exist_ok=True)
LOG = os.path.join(RES, "batch_log.txt")
LOCK = os.path.join(RES, "batch.lock")
STOP = os.path.join(RES, "STOP")
STATUS = os.path.join(RES, "batch_status.json")

SEEDS = [2021, 2022, 2023, 2024]
SETTINGS = [(d, h, 7) for d in ("ETTh1", "ETTh2", "ETTm1", "ETTm2")
            for h in (96, 192, 336, 720)]
THREADS = "4"
WORKERS = 2


def log(msg):
    line = time.strftime("[%m-%d %H:%M:%S] ") + msg
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)


def result_exists(data, plen, seed, mode):
    tag = f"{data}_{plen}_seed{seed}"
    name = f"baseline_{tag}.json" if mode == "baseline" else f"cdp_{mode}_{tag}.json"
    return os.path.exists(os.path.join(RES, name))


def build_phase(modes, seeds, sets=None):
    sets = sets if sets is not None else SETTINGS
    jobs = []
    for seed in seeds:
        for (d, h, ch) in sets:
            ms = [m for m in modes if not result_exists(d, h, seed, m)]
            if ms:
                jobs.append({"seed": seed, "data": d, "plen": h, "enc_in": ch, "modes": ms})
    return jobs


def run_one(seed, data, plen, enc_in, mode):
    cmd = [PY, SCRIPT, "--data", data, "--pred_len", str(plen),
           "--mode", mode, "--seed", str(seed), "--enc_in", str(enc_in)]
    env = dict(os.environ)
    env["CDP_THREADS"] = THREADS
    env["OMP_NUM_THREADS"] = THREADS
    env["MKL_NUM_THREADS"] = THREADS
    t0 = time.time()
    p = subprocess.run(cmd, cwd=ROOT, env=env,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       text=True, encoding="utf-8", errors="replace")
    dt = time.time() - t0
    tail = p.stdout.strip().splitlines()[-1][:120] if p.stdout.strip() else ""
    ok = (p.returncode == 0) and result_exists(data, plen, seed, mode)
    log(f"{'DONE' if ok else 'FAIL'} {data}-{plen} {mode} seed{seed} "
        f"({dt/60:.1f} min) {tail}")
    if not ok:
        with open(LOG, "a", encoding="utf-8") as f:
            f.write("---- stderr tail ----\n" + p.stdout[-2000:] + "\n")
    return ok


def do_job(job):
    if os.path.exists(STOP):
        return False
    tag = f"{job['data']}-{job['plen']} seed{job['seed']} [{','.join(job['modes'])}]"
    log(f"START job {tag}")
    for mode in job["modes"]:
        if result_exists(job["data"], job["plen"], job["seed"], mode):
            continue
        if not run_one(job["seed"], job["data"], job["plen"], job["enc_in"], mode):
            job["failed"] = True
            break
    with open(LOCK, "a") as f:
        f.write(".")  # refresh lock mtime
    return True


def run_phase(jobs, phase_name, pool):
    total = len(jobs)
    log(f"=== PHASE {phase_name}: {total} jobs ===")
    done = 0
    for _ in pool.map(do_job, jobs):
        done += 1
        json.dump({"phase": phase_name, "done": done, "total": total,
                   "ts": time.strftime("%m-%d %H:%M:%S")},
                  open(STATUS, "w"), indent=1)
        if os.path.exists(STOP):
            log("STOP file detected - halting between jobs")
            break


def main():
    if os.path.exists(LOCK) and (time.time() - os.path.getmtime(LOCK)) < 600:
        print("Another batch appears to be running (fresh batch.lock). "
              "Delete cdp_results/batch.lock if this is wrong.", flush=True)
        sys.exit(1)
    open(LOCK, "w").write(time.strftime("%m-%d %H:%M:%S") + "\n")
    if os.path.exists(STOP):
        os.remove(STOP)

    log(f"batch start | settings={len(SETTINGS)} seeds={SEEDS} "
        f"workers={WORKERS} threads={THREADS}")
    pool = ThreadPoolExecutor(max_workers=WORKERS)

    # Priority order (m-data runs are ~5x slower than h-data):
    #   1a) seed2021 all settings  -> main-table numbers ASAP
    #   1b) seeds 2022-2026, ETTh1/ETTh2 -> fast, completes 6-seed error bars on h
    #   1c) ablation seed2021 (cao+cdp)  -> paper ablation table
    #   1d) seeds 2022-2026, ETTm1/ETTm2 -> slow, trails by ~1.5 days
    #   1e) ablation seeds 2022-2026 (bonus)
    h_sets = [(d, h, ch) for (d, h, ch) in SETTINGS if d.startswith("ETTh")]
    m_sets = [(d, h, ch) for (d, h, ch) in SETTINGS if d.startswith("ETTm")]

    p1a = build_phase(("baseline", "head"), SEEDS[:1])
    run_phase(p1a, "1a seed2021 all (baseline+head)", pool)
    if not os.path.exists(STOP):
        p1b = build_phase(("baseline", "head"), SEEDS[1:], h_sets)
        run_phase(p1b, "1b ETTh seeds2022-2026 (baseline+head)", pool)
    if not os.path.exists(STOP):
        p1c = build_phase(("cao", "cdp"), SEEDS[:1])
        run_phase(p1c, "1c ablation seed2021 (cao+cdp)", pool)
    if not os.path.exists(STOP):
        p1d = build_phase(("baseline", "head"), SEEDS[1:], m_sets)
        run_phase(p1d, "1d ETTm seeds2022-2026 (baseline+head)", pool)
    if not os.path.exists(STOP):
        p1e = build_phase(("cao", "cdp"), SEEDS[1:], h_sets + m_sets)
        run_phase(p1e, "1e ablation seeds2022-2026 (bonus)", pool)

    pool.shutdown()
    log("batch finished (or stopped)")
    json.dump({"phase": "finished", "ts": time.strftime("%m-%d %H:%M:%S")},
              open(STATUS, "w"), indent=1)


if __name__ == "__main__":
    main()
