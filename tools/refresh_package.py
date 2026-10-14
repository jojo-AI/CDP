#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""增量刷新 CDP 交付包目录（只增不删，幂等）。

背景
----
包目录 `CDP_实验结果完整包_20261008/` 是「build_package.py 骨架 + 人工整理」的混合体
（例如 `02_超参与协议/` 下有 determinism / multiseed / noise_floor 等人工放入的报告，
 而 build_package.py 里并没有这些文件的列表）。因此**不能**重跑 build_package.py
（它会 shutil.rmtree 整个目录 → 人工整理的部分全部丢失）。
本脚本采用只增不删的定向同步：

  规则 A：源文件在包内任一子目录已存在同名 → 原地覆盖更新（保持既有布局）
  规则 B：包内不存在 → 按扩展名默认映射放到合理子目录
  规则 C：`04_源码/` `05_论文工作区/` 做目录级增量同步（覆盖，不删）

用法
----
  python tools/refresh_package.py                 # 默认包目录
  python tools/refresh_package.py <PKG_DIR>       # 指定包目录
"""
import os
import sys
import shutil

SRC = 'D:/fsdownload'
PKG = sys.argv[1] if len(sys.argv) > 1 else 'D:/fsdownload/CDP_实验结果完整包_20261008'
T = 'D:/fsdownload/TimeMixer'
KW = 'D:/fsdownload/TimeMixer/实验进度记录/CDP_workspace'

IGNORE = shutil.ignore_patterns('__pycache__', '*.pyc', '*.pyo', '.ipynb_checkpoints')

# 顶层目录不进「顶层文件同步」
SKIP_DIRS = {'CDP_实验结果完整包_20261008', 'CDP_paper_package_20260929', 'TimeMixer',
             '.venv', '.idea', '__pycache__', 'Output'}

# 规则 B 的默认落点
DEFAULT_DIR = {
    '.log': '07_训练日志',
    '.md': '06_分析报告',
    '.txt': '01_最终结果',
    '.csv': '02_超参与协议',
    '.json': '02_超参与协议',
    '.pdf': '06_分析报告',
}
# 明确不打包的扩展名（大文件 / 无关）
SKIP_EXT = {'.pth', '.pt', '.zip', '.npy', '.npz', '.ckpt', '.tar', '.gz', '.png', '.jpg'}


def index_pkg():
    """建立 包内文件名 -> [绝对路径] 索引。"""
    idx = {}
    for root, _dirs, files in os.walk(PKG):
        for fn in files:
            idx.setdefault(fn, []).append(os.path.join(root, fn))
    return idx


def sync_source_tree():
    """规则 C：源码 + 论文工作区增量同步。"""
    n = 0
    src_root = os.path.join(PKG, '04_源码')
    os.makedirs(src_root, exist_ok=True)
    for f in ['run.py', 'requirements.txt',
              'exp/exp_long_term_forecasting.py', 'exp/exp_long_term_forecasting_cdp.py']:
        s = os.path.join(T, f)
        if os.path.exists(s):
            shutil.copy2(s, os.path.join(src_root, os.path.basename(f)))
            n += 1
    for d in ['models', 'exp', 'utils', 'data_provider', 'scripts', 'tools']:
        s = os.path.join(T, d)
        if os.path.isdir(s):
            shutil.copytree(s, os.path.join(src_root, d), dirs_exist_ok=True, ignore=IGNORE)
            n += 1
    if os.path.isdir(KW):
        for name in os.listdir(KW):
            s = os.path.join(KW, name)
            d = os.path.join(PKG, '05_论文工作区', name)
            if os.path.isdir(s):
                shutil.copytree(s, d, dirs_exist_ok=True, ignore=IGNORE)
            else:
                os.makedirs(os.path.dirname(d), exist_ok=True)
                shutil.copy2(s, d)
            n += 1
    return n


def main():
    if not os.path.isdir(PKG):
        print('!! 包目录不存在: %s' % PKG)
        return 3

    idx = index_pkg()
    updated, added, skipped = [], [], []

    for fn in sorted(os.listdir(SRC)):
        sp = os.path.join(SRC, fn)
        if os.path.isdir(sp):
            continue
        # 跳过临时/隐藏文件（如 _task_check.txt、.packed_ready），避免污染交付包
        if fn.startswith('_') or fn.startswith('.'):
            skipped.append(fn)
            continue
        ext = os.path.splitext(fn)[1].lower()
        if ext in SKIP_EXT or ext == '':
            skipped.append(fn)
            continue

        hits = idx.get(fn)          # 规则 A
        if hits:
            for h in hits:
                shutil.copy2(sp, h)
                updated.append(os.path.relpath(h, PKG))
        else:                        # 规则 B
            sub = DEFAULT_DIR.get(ext)
            if not sub:
                skipped.append(fn)
                continue
            dst = os.path.join(PKG, sub, fn)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(sp, dst)
            added.append(os.path.relpath(dst, PKG))
            idx.setdefault(fn, []).append(dst)

    n_src = sync_source_tree()

    print('[refresh] 更新 %d 处 / 新增 %d 个 / 跳过 %d 个 / 源码+工作区 %d 项'
          % (len(updated), len(added), len(skipped), n_src))
    if added:
        print('[新增]')
        for a in added[:40]:
            print('   +', a)
    if updated:
        print('[更新]')
        for u in updated[:40]:
            print('   ~', u)
    return 0


if __name__ == '__main__':
    sys.exit(main())
