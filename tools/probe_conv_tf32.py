#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
针对性验证：cuDNN 卷积 + TF32 是否是跨数据集非确定性的根因。

线索（2026-10-08）：
  trace 记录到模型有一个被忽略的卷积层 —— `enc_embedding.value_embedding.tokenConv`
  （nn.Conv1d，in_channels = enc_in，kernel_size=3，padding_mode='circular'）。
  而 `torch.backends.cudnn.allow_tf32` 的 PyTorch 默认值是 **True**，
  且 run.py 的 det_mode 分支只设了 cudnn.deterministic / benchmark，**从未关闭 TF32**。

  这可以解释为什么确定性开关对 ETTh1（7 通道）有效、对 weather（21 通道）失效：
  Conv1d 的 in_channels 不同 → cuDNN 选择的算法不同 → 其中一条走 TF32 非确定路径。

本脚本用同一份权重/输入，重复 N 次前向+反向，比较输出与输入的梯度哈希：
  哈希全部相同 ⇒ 该配置确定；出现多版本 ⇒ 非确定。

用法: python tools/probe_conv_tf32.py
"""
import hashlib

import torch
import torch.nn as nn


def h(t):
    if t is None:
        return 'None'
    return hashlib.md5(
        t.detach().to('cpu', torch.float32).contiguous().numpy().tobytes()
    ).hexdigest()[:12]


def test_conv(c_in, d_model=16, t_len=96, batch=128, tf32=True, iters=4,
              deterministic=True, benchmark=False):
    """在指定 cudnn 配置下重复 iters 次 Conv1d 前向+反向，返回哈希集合。"""
    torch.backends.cudnn.allow_tf32 = tf32
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = benchmark

    outs, grads = [], []
    for _ in range(iters):
        torch.manual_seed(2021)
        conv = nn.Conv1d(c_in, d_model, kernel_size=3, padding=1,
                         padding_mode='circular').cuda().float()
        base = torch.randn(batch, c_in, t_len, device='cuda')
        x = base.clone().detach().requires_grad_(True)
        y = conv(x)
        y.pow(2).mean().backward()
        outs.append(h(y))
        grads.append(h(x.grad))
    return outs, grads


def main():
    if not torch.cuda.is_available():
        print('需要 CUDA')
        return 1
    print('torch %s | GPU %s | cudnn.version %s'
          % (torch.__version__, torch.cuda.get_device_name(0), torch.backends.cudnn.version()))
    print()
    print('%-6s %-7s %-8s | %-16s %-16s' % ('c_in', 'tf32', 'det', 'out 版本数', 'grad 版本数'))
    print('-' * 68)

    results = {}
    for c_in in (7, 21, 321, 862):
        for tf32 in (True, False):
            o, g = test_conv(c_in, tf32=tf32)
            no, ng = len(set(o)), len(set(g))
            results[(c_in, tf32)] = (no, ng)
            flag = '  ✅ 确定' if (no == 1 and ng == 1) else '  ❌ 非确定'
            print('%-6d %-7s %-8s | %-16s %-16s%s'
                  % (c_in, tf32, 'True', '%d / %d' % (no, len(o)),
                     '%d / %d' % (ng, len(g)), flag))

    print()
    print('=' * 68)
    print('结论：')
    bad_tf32 = [k for k, v in results.items() if v[0] > 1 or v[1] > 1]
    if not bad_tf32:
        print('  所有配置均确定 ⇒ 卷积/TF32 不是根因，需继续追查。')
    else:
        print('  以下配置非确定:')
        for k in bad_tf32:
            print('    c_in=%d  tf32=%s  → out %d 版本 / grad %d 版本'
                  % (k[0], k[1], results[k][0], results[k][1]))
        all_tf32_on = all(k[1] for k in bad_tf32)
        if all_tf32_on:
            print()
            print('  🎯 全部集中在 tf32=True ⇒ **根因 = cuDNN 卷积 + TF32**。')
            print('     修复：在 det_mode 分支里加 torch.backends.cudnn.allow_tf32 = False')
            print('     （以及 torch.backends.cuda.matmul.allow_tf32 = False）。')
        else:
            print()
            print('  ⚠️ 非确定性不限于 tf32=True，还有其他来源。')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
