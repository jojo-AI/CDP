#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
捕捉被 warnings.filterwarnings('ignore') 吞掉的非确定性警告。

背景（2026-10-08）：
  exp/exp_long_term_forecasting.py 第 15 行执行
      warnings.filterwarnings('ignore')
  而 PyTorch 在 `use_deterministic_algorithms(True, warn_only=True)` 模式下，
  是用 **Python warning** 报告「某算子没有确定性实现、已回退到非确定版本」的。
  于是 det_mode=1 下所有这类警告都被静默吞掉 —— 看起来"一切正常"，
  实际训练却在第 1 个 epoch 就分叉（weather-96 实测：同 seed 3 次 MSE 极差 1.13%）。

本脚本先让 filterwarnings 失效，再以 det_mode=1 等价配置跑一次训练，
把 PyTorch 抛出的每条警告原样打印（含算子名与触发位置），用于定位根因。

用法:
  python tools/probe_det_warnings.py --log D:/fsdownload/st1_weather_96_dynamic_l10.0.log \
      --epochs 1 --det 1
"""
import os
import sys
import re
import ast
import types
import random
import argparse
import warnings

# --------------------------------------------------------------------------- #
# 1) 必须在下游 exp 模块 import 之前劫持 filterwarnings，
#    否则 exp 里的 warnings.filterwarnings('ignore') 会掩盖掉所有非确定性警告。
# --------------------------------------------------------------------------- #
_orig_filterwarnings = warnings.filterwarnings


def _noop_filterwarnings(*args, **kwargs):
    """把下游的 filterwarnings('ignore') 变成空操作。"""
    pass


warnings.filterwarnings = _noop_filterwarnings
_orig_filterwarnings('always')      # 用原函数设定：任何警告都显示
warnings.simplefilter('always')


# 2) 自定义 showwarning：打上可 grep 的前缀，并显示类别与位置
def _showwarning(message, category, filename, lineno, file=None, line=None):
    sys.stderr.write('[DETWARN] %s: %s  (%s:%s)\n'
                     % (category.__name__, message, filename, lineno))
    sys.stderr.flush()


warnings.showwarning = _showwarning

import numpy as np          # noqa: E402
import torch                # noqa: E402


def parse_real_args(log_path):
    txt = open(log_path, encoding='utf-8', errors='ignore').read()
    ns_str = re.search(r'Namespace\(([^\n]*)\)', txt).group(1)
    tree = ast.parse('dict(' + ns_str + ')', mode='eval')
    kw = {n.arg: ast.literal_eval(n.value) for n in tree.body.keywords}
    return types.SimpleNamespace(**kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--log', default='D:/fsdownload/st1_weather_96_dynamic_l10.0.log')
    ap.add_argument('--seed', type=int, default=2021)
    ap.add_argument('--det', type=int, default=1, help='1=warn_only=True（只警告）2=严格（抛错）')
    ap.add_argument('--epochs', type=int, default=1)
    a = ap.parse_args()

    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(a.seed)
        torch.cuda.manual_seed_all(a.seed)

    if a.det in (1, 2):
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=(a.det == 1))

    args = parse_real_args(a.log)
    args.train_epochs = a.epochs
    args.itr = 1
    args.use_multi_gpu = False
    args.use_gpu = bool(torch.cuda.is_available())

    print('[probe] 配置来源: %s' % a.log)
    print('[probe] data=%s enc_in=%s pred_len=%s batch=%s det=%s epochs=%s'
          % (args.data, args.enc_in, args.pred_len, args.batch_size, a.det, args.train_epochs))
    print('[probe] cudnn.deterministic=%s benchmark=%s cudnn.allow_tf32=%s '
          'matmul.allow_tf32=%s cublas_ws=%s det_algos=%s'
          % (torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark,
             torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32,
             os.environ.get('CUBLAS_WORKSPACE_CONFIG', '<unset>'),
             torch.are_deterministic_algorithms_enabled()))
    print('[probe] warnings 已恢复显示（下游 filterwarnings 已被置为 no-op）')
    sys.stdout.flush()

    sys.path.insert(0, os.getcwd())
    from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast

    exp = Exp_Long_Term_Forecast(args)
    setting = 'detwarn_%s' % os.path.basename(a.log).replace('.log', '')

    try:
        exp.train(setting)
    except RuntimeError as e:
        print()
        print('=' * 76)
        print('[probe] 🎯 严格模式抛错，这就是非确定性算子：')
        print(e)
        print('=' * 76)
        return 0

    print()
    print('[probe] 训练完成，未抛错。请检查上方带 [DETWARN] 前缀的警告。')
    return 0


if __name__ == '__main__':
    sys.exit(main())
