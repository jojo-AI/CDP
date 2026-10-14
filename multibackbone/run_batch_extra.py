# -*- coding: utf-8 -*-
"""
Third-backbone batch runner: CycleNet + iTransformer, both with CDP.

Queue (per repo, CycleNet first):
  Phase 1: baseline + head(CAO+PSC) x 16 ETT settings x seeds 2021-2026
  Phase 2: cao + cdp ablation, seed 2021
Conventions identical to TimeKAN run_batch.py:
  - skip-if-exists json -> safe to re-run / resume after any interruption
  - STOP file inside <repo>/cdp_results/ halts gracefully after current run
  - batch.lock freshness check prevents double launch
  - workers via env CDP_WORKERS (default 2), threads per run via CDP_THREADS
    (default cpu_count-1; use 3 when sharing the box with another batch)

Usage:
  python run_batch_extra.py
"""
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

BASE = os.path.dirname(os.path.abspath(__file__))
REPOS = [("CycleNet", os.path.join(BASE, "CycleNet-main"),
          "cdp_cyclenet_cpu.py", "cyclenet"),
         ("iTransformer", os.path.join(BASE, "iTransformer-main"),
          "cdp_itransformer_cpu.py", "itransformer")]
PY = sys.executable
SEEDS = [2021, 2022, 2023, 2024]
SETTINGS = [(d, h) for d in ("ETTh1", "ETTh2", "ETTm1", "ETTm2",
                             "weather", "electricity", "traffic")
            for h in (96, 192, 336, 720)]
WORKERS = int(os.environ.get("CDP_WORKERS", "2"))
# Threads per run: set CDP_THREADS (e.g. "2") when sharing the box with the
# TimeKAN batch; leave empty to use each script's default (cpu_count-1).
THREADS = os.environ.get("CDP_THREADS", "")
# Restrict repos via CDP_REPOS (comma-separated, e.g. "CycleNet").
_REPO_ENV = os.environ.get("CDP_REPOS", "")
if _REPO_ENV:
    _want = {s.strip().lower() for s in _REPO_ENV.split(",") if s.strip()}
    REPOS = [r for r in REPOS if r[0].lower() in _want]


def log(repo, msg):
    line = time.strftime("[%m-%d %H:%M:%S] ") + f"[{repo}] " + msg
    print(line, flush=True)
    logp = os.path.join(BASE, repo + "-main", "cdp_results", "batch_log.txt")
    os.makedirs(os.path.dirname(logp), exist_ok=True)
    with open(logp, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def result_path(repo_dir, mode, data, plen, seed):
    tag = f"{data}_{plen}_seed{seed}"
    name = f"baseline_{tag}.json" if mode == "baseline" else f"cdp_{mode}_{tag}.json"
    return os.path.join(repo_dir, "cdp_results", name)


def run_one(repo, repo_dir, script, ck_prefix, mode, data, plen, seed):
    if os.path.exists(result_path(repo_dir, mode, data, plen, seed)):
        return True
    if os.path.exists(os.path.join(repo_dir, "cdp_results", "STOP")):
        return False
    env = dict(os.environ)
    if THREADS:
        env["CDP_THREADS"] = THREADS
        env["OMP_NUM_THREADS"] = THREADS
        env["MKL_NUM_THREADS"] = THREADS

    def _exec(cmd):
        t0 = time.time()
        p = subprocess.run(cmd, cwd=repo_dir, env=env,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           text=True, encoding="utf-8", errors="replace")
        return p, time.time() - t0

    def _log_fail(cmd, p, dt):
        with open(os.path.join(repo_dir, "cdp_results", "batch_log.txt"),
                  "a", encoding="utf-8") as f:
            f.write("---- tail ----\n" + p.stdout[-2000:] + "\n")

    # Two-stage modes load the frozen baseline checkpoint. With >1 worker a
    # head job may start before its baseline job finished; regenerate the
    # checkpoint by (re)running the baseline stage first if it is missing.
    if mode != "baseline":
        ck = os.path.join(repo_dir, "cdp_ckpt",
                          f"{ck_prefix}_{data}_{plen}_seed{seed}.pth")
        if not os.path.exists(ck):
            bcmd = [PY, script, "--data", data, "--pred_len", str(plen),
                    "--mode", "baseline", "--seed", str(seed)]
            log(repo, f"ckpt missing for {data}-{plen} seed{seed} - "
                      f"retraining baseline first")
            p, dt = _exec(bcmd)
            okb = (p.returncode == 0) and os.path.exists(ck)
            log(repo, f"{'DONE' if okb else 'FAIL'} {data}-{plen} "
                      f"baseline(ckpt-rebuild) seed{seed} ({dt/60:.1f} min)")
            if not okb:
                _log_fail(bcmd, p, dt)
                return False

    cmd = [PY, script, "--data", data, "--pred_len", str(plen),
           "--mode", mode, "--seed", str(seed)]
    p, dt = _exec(cmd)
    tail = p.stdout.strip().splitlines()[-1][:120] if p.stdout.strip() else ""
    ok = (p.returncode == 0) and os.path.exists(
        result_path(repo_dir, mode, data, plen, seed))
    log(repo, f"{'DONE' if ok else 'FAIL'} {data}-{plen} {mode} seed{seed} "
              f"({dt/60:.1f} min) {tail}")
    if not ok:
        _log_fail(cmd, p, dt)
    return ok


def build_jobs(repo, repo_dir, script, ck):
    # Priority: seed2021 custom (weather/ECL/traffic, user priority) ->
    # seed2021 remaining ETT -> seeds2022-26 ETTh (fast) ->
    # seeds2022-26 custom -> seeds2022-26 ETTm (slow) -> ablation seed2021.
    jobs = []
    h_sets = [s for s in SETTINGS if s[0].startswith("ETTh")]
    m_sets = [s for s in SETTINGS if s[0].startswith("ETTm")]
    c_sets = [s for s in SETTINGS
              if s[0] in ("weather", "electricity", "traffic")]

    def add(seeds, sets, modes):
        for seed in seeds:
            for (d, h) in sets:
                for mode in modes:
                    if not os.path.exists(result_path(repo_dir, mode, d, h, seed)):
                        jobs.append((repo, repo_dir, script, ck, mode, d, h, seed))

    add(SEEDS[:1], c_sets, ("baseline", "head"))
    add(SEEDS[:1], h_sets + m_sets, ("baseline", "head"))
    add(SEEDS[1:], h_sets, ("baseline", "head"))
    add(SEEDS[1:], c_sets, ("baseline", "head"))
    add(SEEDS[1:], m_sets, ("baseline", "head"))
    add(SEEDS[:1], SETTINGS, ("cao", "cdp"))
    return jobs


def stop_requested(repo, repo_dir):
    return os.path.exists(os.path.join(repo_dir, "cdp_results", "STOP"))


def main():
    all_jobs = []
    for repo, repo_dir, script, ck in REPOS:
        lock = os.path.join(repo_dir, "cdp_results", "batch.lock")
        os.makedirs(os.path.dirname(lock), exist_ok=True)
        if os.path.exists(lock) and (time.time() - os.path.getmtime(lock)) < 600:
            print(f"[{repo}] another batch appears to be running "
                  f"(fresh batch.lock). Delete it if wrong.", flush=True)
            sys.exit(1)
        open(lock, "w").write(time.strftime("%m-%d %H:%M:%S") + "\n")
        jobs = build_jobs(repo, repo_dir, script, ck)
        log(repo, f"queued {len(jobs)} runs (Phase1 baseline+head x6 seeds, "
                  f"Phase2 cao+cdp seed2021)")
        all_jobs += jobs

    if not all_jobs:
        print("nothing to run - all results already exist")
        return

    log("BATCH", f"start | {len(all_jobs)} runs total | workers={WORKERS}")
    done = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futures = [ex.submit(run_one, *j) for j in all_jobs]
        for repo, repo_dir, _s, _m, _d, _p, _sd in all_jobs:
            pass
        for j, fut in zip(all_jobs, futures):
            fut.result()
            done += 1
            json.dump({"done": done, "total": len(all_jobs),
                       "ts": time.strftime("%m-%d %H:%M:%S")},
                      open(os.path.join(BASE, "batch_status.json"), "w"),
                      indent=1)
            if stop_requested(j[1], j[1]):
                log("BATCH", "STOP requested - draining remaining queued "
                             "runs of other repo only")
    log("BATCH", "finished (or stopped)")
    json.dump({"phase": "finished", "ts": time.strftime("%m-%d %H:%M:%S")},
              open(os.path.join(BASE, "batch_status.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
