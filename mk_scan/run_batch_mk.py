# -*- coding: utf-8 -*-
"""
CDP M/K 扫描批量 runner —— A100/Linux 自包含版（20261015）
==========================================================
目标：ETTh1-96，TimeMixer++ CDP 两阶段，seed 固定 2021

双模式（自动检测）：
  严格模式  repo/checkpoints/ 下存在归档 Stage-1 骨干
            long_term_forecast_ETTh1_st1_96_dynamic_l10.0_*
            → M 扫描 4 点 {0.6,0.8,1.2,1.5} 复用归档骨干（只跑 Stage-2），
              归档锚点（M r=1.0 / K=4）从 archived_reference/ 复制后跳过；
            → K 扫描 {2,3,5,6}（Stage-1 重训 + Stage-2）。
  自足模式  无归档骨干 → 先在本机重训默认骨干（task_K(4) 内完成 Stage-1），
            M 扫描 5 点 {0.6,0.8,1.0,1.2,1.5}（含本地锚点 r=1.0）全部挂在该骨干上，
            K 扫描 {2,3,5,6} 同上。JSON 记 backbone_source=local_retrain。

特性：skip-if-exists + 断点续跑；结束自动打包 回传_MK扫描结果_*.zip。
用法：python mk_scan/run_batch_mk.py     （重跑即续跑）
环境变量（可选）：CDP_PY 训练用 python；CUDA_VISIBLE_DEVICES 选卡。
"""
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time
import zipfile

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # 包根
REPO = os.path.join(PKG, "repo")
RES = os.path.join(PKG, "mk_results")
LOGD = os.path.join(PKG, "mk_logs")
ARCH = os.path.join(PKG, "archived_reference")
SEED = 2021
PY = os.environ.get("CDP_PY", sys.executable)

ETTH1_DSARGS = ("--root_path ./dataset/ETT-small/ --data_path ETTh1.csv --data ETTh1 "
                "--enc_in 7 --dec_in 7 --c_out 7 --e_layers 2 --factor 1 "
                "--d_model 16 --d_ff 32 --batch_size 128")

ST1_COMMON = ("--task_name long_term_forecast --model TimeMixer --features M --seq_len 96 "
              "--label_len 0 --d_layers 1 --des ST1 --itr 1 --learning_rate 0.01 "
              "--num_workers 0 --train_loss l1 --train_l1_mu 0.0 "
              "--down_sampling_method avg --down_sampling_window 2 "
              "--train_epochs 30 --patience 8")

ST2_COMMON = ("--task_name long_term_forecast --model TimeMixer_CDP --features M --seq_len 96 "
              "--label_len 0 --d_layers 1 --des MK --itr 1 --learning_rate 0.01 "
              "--num_workers 0 --down_sampling_method avg --down_sampling_window 2 "
              "--cdp_stage_w 1 --cdp_S 16 --cdp_err_decay 0.9 --cdp_lambda_t 0.01 "
              "--train_epochs 20 --patience 12")

# ETTh1-96 论文 winner 配置（= 归档 at_cal1 的同一配方）
ST2_WINNER = ("--cdp_learning_rate 5e-4 --cdp_loss mse_l1 --cdp_l1_mu 0.5 --cdp_monitor mae "
              "--cdp_unfreeze_head 1 --cdp_unfreeze_norm 1 "
              "--cdp_chan_rank 8 --cdp_chan_rank_mul 8 "
              "--backbone_mode dynamic --cdp_unfreeze_pdm 1 --cdp_st_lr_ratio 0.2 "
              "--cdp_dyncal 1 --cdp_loss l1 --cdp_monitor mae --cdp_cal_lr_ratio 1")


def log(msg):
    print("[%s] %s" % (time.strftime("%F %T"), msg), flush=True)


def ensure_run_mk():
    """run.py 单点补丁：--cdp_M type=int→float，其余零改动。"""
    src = os.path.join(REPO, "run.py")
    dst = os.path.join(REPO, "run_mk.py")
    with open(src, encoding="utf-8") as f:
        s = f.read()
    key = "parser.add_argument('--cdp_M', type=int"
    n = s.count(key)
    if n != 1:
        raise SystemExit("!! run.py 中 cdp_M 定义行匹配数=%d（期望 1）→ run.py 已被改动，"
                         "停止并回报，不要自行修改。" % n)
    patched = s.replace(key, "parser.add_argument('--cdp_M', type=float")
    if os.path.exists(dst):
        with open(dst, encoding="utf-8") as f:
            if f.read() == patched:
                return dst
    with open(dst, "w", encoding="utf-8", newline="\n") as f:
        f.write(patched)
    log("已生成 %s（run.py 单点补丁：--cdp_M type=int→float）" % dst)
    return dst


def find_stage1_ckpt_default():
    """归档/锚点 Stage-1 骨干（ETTh1-96, dynamic+l1, seed2021, 无 K4 后缀）。"""
    pats = glob.glob(os.path.join(
        REPO, "checkpoints", "long_term_forecast_ETTh1_st1_96_dynamic_l10.0_*"))
    pats = [p for p in pats if "CDP" not in os.path.basename(p)]
    if not pats:
        return None
    pref = [p for p in pats if p.endswith("_2021") or "_s2021" in os.path.basename(p)]
    return sorted(pref or pats)[0]


def find_k4_ckpt():
    """自足模式：K=4 锚点骨干（task_K(4) 重训所得）。"""
    pats = glob.glob(os.path.join(
        REPO, "checkpoints", "long_term_forecast_ETTh1_st1_96_K4_dynamic_l10.0_*"))
    pats = [p for p in pats if "CDP" not in os.path.basename(p)]
    return sorted(pats)[0] if pats else None


def run_one(iso, runner, common, extra, mid, ck, logfile):
    toks = ["-u", runner, "--is_training", str(iso)] + common.split() + extra.split()
    toks += ["--model_id", mid, "--fix_seed", str(SEED)]
    if ck:
        toks += ["--cdp_backbone_checkpoint", ck]
    with open(logfile, "w", encoding="utf-8", errors="replace") as f:
        rc = subprocess.call([PY, "-X", "utf8"] + toks, cwd=REPO, stdout=f,
                             stderr=subprocess.STDOUT)
    with open(logfile, encoding="utf-8", errors="replace") as f:
        txt = f.read()
    ms = re.findall(r"mse:([0-9.]+), mae:([0-9.]+)", txt)
    if rc != 0:
        return None, None, "exit=%d" % rc
    if not ms:
        return None, None, "no-mse-line"
    return float(ms[-1][0]), float(ms[-1][1]), "ok"


def done(name):
    p = os.path.join(RES, name)
    if not os.path.exists(p):
        return False
    try:
        with open(p, encoding="utf-8") as f:
            j = json.load(f)
        return j.get("status") == "ok" and j.get("mse") is not None
    except Exception:
        return False


def save(name, obj):
    obj["finished_at"] = time.strftime("%F %T")
    with open(os.path.join(RES, name), "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    log("  -> %s  mse=%s mae=%s" % (name, obj.get("mse"), obj.get("mae")))


FAILS = []


def guarded(name, fn, *a):
    if done(name):
        log("[skip] %s（结果已存在）" % name)
        return
    log("===== 任务 %s =====" % name)
    try:
        fn(*a)
    except Exception as e:
        FAILS.append((name, repr(e)))
        log("!! %s 异常：%r" % (name, e))


def task_M(r, runmk, ck, backbone_source):
    m = round(4.0 * r, 2)
    name = "mk_M_r%g_seed%d.json" % (r, SEED)
    mid = "ETTh1_mk_M%g_s%d" % (r, SEED)
    lf = os.path.join(LOGD, "M_r%g_s%d.log" % (r, SEED))
    extra = ST2_WINNER + " --cdp_M %g --down_sampling_layers 3 " % m + ETTH1_DSARGS
    mse = mae = None
    for iso in (1, 0):
        mse, mae, st = run_one(iso, runmk, ST2_COMMON, extra, mid, ck, lf)
        if st != "ok":
            FAILS.append((name, "stage is_training=%d: %s" % (iso, st)))
            log("!! %s is_training=%d 失败（%s）" % (name, iso, st))
            return
    save(name, {"axis": "M", "seg_ratio": r, "cdp_M": m, "down_sampling_layers": 3,
                "model_id": mid, "seed": SEED, "mse": mse, "mae": mae,
                "backbone_source": backbone_source,
                "stage1_ckpt": os.path.basename(ck), "status": "ok", "log": lf})


def task_K(K, runmk):
    dsl = K - 1
    st1_name = "st1_K%d_seed%d.json" % (K, SEED)
    st1_mid = "ETTh1_st1_96_K%d_dynamic_l10.0_s%d" % (K, SEED)
    lf1 = os.path.join(LOGD, "K%d_stage1_s%d.log" % (K, SEED))

    def ck_dirs():
        p = glob.glob(os.path.join(REPO, "checkpoints",
                                   "long_term_forecast_%s_*" % st1_mid))
        return [x for x in p if "CDP" not in os.path.basename(x)]

    if not done(st1_name):
        if not ck_dirs():
            log("  Stage-1 (K=%d, down_sampling_layers=%d) 重训…" % (K, dsl))
            extra = ST1_COMMON + " --down_sampling_layers %d " % dsl + \
                ETTH1_DSARGS + " --backbone_mode dynamic"
            mse, mae, st = run_one(1, "run.py", "", extra, st1_mid, None, lf1)
            if st != "ok":
                FAILS.append((st1_name, "stage1 train: %s" % st))
                log("!! K=%d Stage-1 训练失败（%s）→ 跳过该点" % (K, st))
                return
        ck = sorted(ck_dirs())[0]
        extra = ST1_COMMON + " --down_sampling_layers %d " % dsl + \
            ETTH1_DSARGS + " --backbone_mode dynamic"
        mse, mae, st = run_one(0, "run.py", "", extra, st1_mid, None, lf1)
        if st != "ok":
            FAILS.append((st1_name, "stage1 eval: %s" % st))
            log("!! K=%d Stage-1 评测失败（%s）→ 跳过该点" % (K, st))
            return
        save(st1_name, {"axis": "K", "K": K, "down_sampling_layers": dsl,
                        "model_id": st1_mid, "seed": SEED,
                        "mse": mse, "mae": mae, "stage": "stage1",
                        "backbone_source": "local_retrain",
                        "status": "ok", "log": lf1})
    else:
        ck_list = ck_dirs()
        if not ck_list:
            FAILS.append((st1_name, "json 已存在但 checkpoint 丢失"))
            log("!! K=%d 的 Stage-1 JSON 已存在但 checkpoint 目录丢失 → 跳过该点并回报" % K)
            return
        ck = sorted(ck_list)[0]

    name = "mk_K%d_seed%d.json" % (K, SEED)
    mid = "ETTh1_mk_K%d_s%d" % (K, SEED)
    lf2 = os.path.join(LOGD, "K%d_stage2_s%d.log" % (K, SEED))
    extra = ST2_WINNER + " --cdp_M 4 --down_sampling_layers %d " % dsl + ETTH1_DSARGS
    mse = mae = None
    for iso in (1, 0):
        mse, mae, st = run_one(iso, runmk, ST2_COMMON, extra, mid, ck, lf2)
        if st != "ok":
            FAILS.append((name, "stage is_training=%d: %s" % (iso, st)))
            log("!! %s is_training=%d 失败（%s）" % (name, iso, st))
            return
    with open(os.path.join(RES, st1_name), encoding="utf-8") as f:
        st1 = json.load(f)
    save(name, {"axis": "K", "K": K, "cdp_M": 4.0, "down_sampling_layers": dsl,
                "model_id": mid, "seed": SEED, "mse": mse, "mae": mae,
                "stage1_mse": st1.get("mse"), "stage1_mae": st1.get("mae"),
                "backbone_source": "archived" if K == 4 and st1.get("source") == "archived"
                                   else "local_retrain",
                "stage1_ckpt": os.path.basename(ck), "status": "ok", "log": lf2})


def summarize():
    rows = []
    for p in glob.glob(os.path.join(RES, "mk_*.json")):
        with open(p, encoding="utf-8") as f:
            j = json.load(f)
        if j.get("status") != "ok":
            continue
        if j["axis"] == "M":
            rows.append(("M", j["seg_ratio"], j["mse"], j["mae"]))
        else:
            rows.append(("K", j["K"], j["mse"], j["mae"]))
    rows.sort(key=lambda t: (t[0], t[1]))
    log("======== 汇总（ETTh1-96, seed%d）========" % SEED)
    for ax, v, mse, mae in rows:
        log("  %s=%-4g  mse=%-12s mae=%s" % (ax, v, mse, mae))
    return rows


def pack():
    zpath = os.path.join(PKG, "回传_MK扫描结果_%s.zip" % time.strftime("%Y%m%d_%H%M"))
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for d in (RES, LOGD):
            for root, _dirs, files in os.walk(d):
                for fn in files:
                    fp = os.path.join(root, fn)
                    z.write(fp, os.path.relpath(fp, PKG))
        console = os.path.join(PKG, "mk_scan_console.log")
        if os.path.isfile(console):
            z.write(console, "mk_scan_console.log")
    log("回传包已生成：%s" % zpath)
    return zpath


def main():
    for d in (RES, LOGD):
        os.makedirs(d, exist_ok=True)
    lock = os.path.join(PKG, ".mk_batch.lock")
    if os.path.exists(lock):
        log("!! 检测到 .mk_batch.lock：上一批次可能仍在运行。确认没有并发后删除该文件再重试。")
        sys.exit(1)
    open(lock, "w").write(str(os.getpid()))
    try:
        if not os.path.isdir(REPO):
            raise SystemExit("!! 仓库不存在：%s" % REPO)
        if not os.path.isfile(os.path.join(REPO, "run.py")):
            raise SystemExit("!! %s 下没有 run.py" % REPO)
        if not os.path.isfile(os.path.join(REPO, "dataset", "ETT-small", "ETTh1.csv")):
            raise SystemExit("!! 缺少数据集 dataset/ETT-small/ETTh1.csv")

        # torch/CUDA 自检（给出清晰报错，而不是训练中途崩）
        try:
            import torch  # noqa
            assert torch.cuda.is_available()
            log("CUDA OK: %s" % torch.cuda.get_device_name(0))
        except Exception as e:
            raise SystemExit("!! 当前 python 无可用 CUDA torch（%r）→ "
                             "请先用集群环境 python 运行，或设置 CDP_PY 指向它。" % e)

        arch_ck = find_stage1_ckpt_default()
        runmk = ensure_run_mk()

        if arch_ck is not None:
            MODE = "strict"
            log("【严格模式】检测到归档骨干：%s" % os.path.basename(arch_ck))
            os.makedirs(ARCH, exist_ok=False) if not os.path.isdir(ARCH) else None
            for fn in glob.glob(os.path.join(ARCH, "*.json")):
                dst = os.path.join(RES, os.path.basename(fn))
                if not os.path.exists(dst):
                    shutil.copyfile(fn, dst)
                    log("  归档锚点已引用：%s" % os.path.basename(fn))
            M_RATIOS = [0.6, 0.8, 1.2, 1.5]
            K_LIST = [2, 3, 5, 6]
            for r in M_RATIOS:
                guarded("mk_M_r%g_seed%d.json" % (r, SEED),
                        task_M, r, runmk, arch_ck, "archived")
            for K in K_LIST:
                guarded("mk_K%d_seed%d.json" % (K, SEED), task_K, K, runmk)
        else:
            MODE = "self"
            log("【自足模式】无归档骨干 → 先重训默认骨干（K=4 任务内完成），"
                "M 扫描含本地锚点 r=1.0，共 5 点。")
            M_RATIOS = [0.6, 0.8, 1.0, 1.2, 1.5]
            K_LIST = [2, 3, 5, 6]
            guarded("mk_K4_seed%d.json" % SEED, task_K, 4, runmk)
            ck = find_k4_ckpt()
            if ck is None:
                log("!! K=4 锚点骨干缺失 → M 扫描无法保证同骨干可比，全部跳过并回报。")
                FAILS.append(("mk_M_*", "anchor backbone missing"))
            else:
                log("M 扫描复用本地锚点骨干：%s" % os.path.basename(ck))
                for r in M_RATIOS:
                    guarded("mk_M_r%g_seed%d.json" % (r, SEED),
                            task_M, r, runmk, ck, "local_retrain")
            for K in K_LIST:
                guarded("mk_K%d_seed%d.json" % (K, SEED), task_K, K, runmk)

        log("模式：%s" % MODE)
        summarize()
        pack()
        if FAILS:
            log("!! 本批次存在 %d 个失败点：%s" % (len(FAILS), FAILS))
            log("!! 重跑 run_all.sh（或本脚本）即可断点续跑；连续 2 次同点失败则停下回报日志。")
        else:
            log("全部任务完成。")
    finally:
        if os.path.exists(lock):
            os.remove(lock)


if __name__ == "__main__":
    main()
