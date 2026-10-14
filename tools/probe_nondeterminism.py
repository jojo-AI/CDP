# -*- coding: utf-8 -*-
"""
非确定性算子隔离测试
====================
背景：ETTh1-96 同 seed 重复运行，默认配置下 MSE 波动最高 2.65%；
      加上 cuDNN/cuBLAS 确定性开关后波动 <1e-4%。但严格模式
      (torch.use_deterministic_algorithms(True)) 不报任何错，
      说明元凶是 PyTorch「不认识」的静默非确定性算子。

本脚本逐个测试 TimeMixer 用到的候选算子，定位到底是谁：
  1. nn.Linear 的 forward+backward           （cuBLAS batched GEMM）
  2. nn.AvgPool1d 的 forward+backward        （疑似 atomicAdd）
  3. nn.Conv1d  的 forward+backward          （cuDNN）
  4. 完整 Model 的 forward+backward          （端到端）

判定：同一输入跑两次，比较梯度张量的字节哈希。
"""
import os
import sys
import hashlib

# 与被测训练保持一致的确定性设置
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import torch
import torch.nn as nn

torch.manual_seed(2021)
torch.cuda.manual_seed_all(2021)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
try:
    torch.use_deterministic_algorithms(True, warn_only=True)
except Exception:
    pass

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'


def thash(t):
    return hashlib.md5(t.detach().cpu().numpy().tobytes()).hexdigest()[:16]


def probe(name, build, inputs):
    """build() 新建模块（固定权重），inputs 固定输入；跑两次 fwd+bwd 比对梯度。"""
    hashes = []
    for _ in range(2):
        torch.manual_seed(2021)          # 每次都重置，确保初始权重一致
        m = build().to(DEV)
        m.zero_grad(set_to_none=True)
        # 无参模块（池化/GELU/Dropout/LayerNorm）必须让「输入」本身需要梯度，
        # 否则 backward 会因为图中没有可训练张量而报错。
        x = inputs()
        if isinstance(x, tuple):
            x = tuple(t.clone().to(DEV).requires_grad_(True) for t in x)
            y = m(*x)
        else:
            x = x.clone().to(DEV).requires_grad_(True)
            y = m(x)
        loss = y[0].pow(2).mean() if isinstance(y, (tuple, list)) else y.pow(2).mean()
        loss.backward()
        grads = [thash(p.grad) for p in m.parameters() if p.grad is not None]
        if not grads:
            xs = x if isinstance(x, tuple) else (x,)
            grads = [thash(t.grad) for t in xs if t.grad is not None]
        hashes.append((thash(y[0] if isinstance(y, (tuple, list)) else y), grads))
    same_fwd = hashes[0][0] == hashes[1][0]
    same_bwd = hashes[0][1] == hashes[1][1]
    print('  %-34s forward: %-12s backward: %s'
          % (name, '✅ 一致' if same_fwd else '❌ 不一致',
             '✅ 一致' if same_bwd else '❌ 不一致 (%d/%d 个梯度不同)'
             % (sum(1 for a, b in zip(hashes[0][1], hashes[1][1]) if a != b),
                len(hashes[0][1]))))
    return same_fwd and same_bwd


print('device = %s' % DEV)
print('=' * 74)
print('非确定性算子隔离测试')
print('=' * 74)

B, T, C = 128, 96, 7

probe('nn.Linear (B,T,C)->(B,T,32)',
      lambda: nn.Linear(C, 32),
      lambda: torch.randn(B, T, C, generator=torch.Generator().manual_seed(0)))

probe('nn.AvgPool1d(kernel=2, stride=2)',
      lambda: nn.AvgPool1d(2, 2),
      lambda: torch.randn(B, 64, T, generator=torch.Generator().manual_seed(0)))

probe('nn.AvgPool1d(kernel=2, stride=2, pad=1, count_include_pad=False)',
      lambda: nn.AvgPool1d(2, 2, padding=1, count_include_pad=False),
      lambda: torch.randn(B, 64, T, generator=torch.Generator().manual_seed(0)))

probe('nn.MaxPool1d(kernel=2, stride=2)',
      lambda: nn.MaxPool1d(2, 2),
      lambda: torch.randn(B, 64, T, generator=torch.Generator().manual_seed(0)))

probe('nn.Conv1d(C=7->16, kernel=3)',
      lambda: nn.Conv1d(C, 16, 3, padding=1),
      lambda: torch.randn(B, C, T, generator=torch.Generator().manual_seed(0)))

probe('nn.GELU',
      lambda: nn.GELU(),
      lambda: torch.randn(B, T, 32, generator=torch.Generator().manual_seed(0)))

probe('nn.LayerNorm(32)',
      lambda: nn.LayerNorm(32),
      lambda: torch.randn(B, T, 32, generator=torch.Generator().manual_seed(0)))

probe('nn.Dropout(p=0.1) [train mode]',
      lambda: nn.Dropout(0.1).train(),
      lambda: torch.randn(B, T, 32, generator=torch.Generator().manual_seed(0)))

# --- 端到端：真实 Model（配置直接从真实训练日志里的 Namespace 还原，保证与实验一致）---
import re
import ast
import types

for _p in (os.getcwd(), os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'TimeMixer')):
    if os.path.isdir(os.path.join(_p, 'models')):
        sys.path.insert(0, _p)
        break
else:
    print('  ⚠️ 未找到 models 包，请从 TimeMixer 根目录运行')

LOG = None
for _cand in ('D:/fsdownload/determinism_test_det1/same1.log',
              os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           'determinism_test_det1', 'same1.log')):
    # 注意：不能用 /d/... 这种 MSYS 路径 —— 原生 Windows Python 解析不了
    #（bash 只会在「命令行参数」上做自动路径转换，脚本内部硬编码的不会）。
    if os.path.exists(_cand):
        LOG = _cand
        break
try:
    txt = open(LOG, encoding='utf-8', errors='ignore').read()
    ns_str = re.search(r'Namespace\(([^\n]*)\)', txt).group(1)
    # 不能用 ast.literal_eval('dict(...)')：literal_eval 禁止函数调用（Call 节点）。
    # 改成解析关键字参数，逐个 literal_eval，安全且不依赖 eval。
    _tree = ast.parse('dict(' + ns_str + ')', mode='eval')
    _kw = {n.arg: ast.literal_eval(n.value) for n in _tree.body.keywords}
    Cfg = types.SimpleNamespace(**_kw)
    print('  配置来源: %s' % LOG)
except Exception as e:
    print('  ⚠️ 无法还原配置(%s)，退回内置默认值' % e)
    ns = dict(seq_len=96, label_len=0, pred_len=96, enc_in=7, dec_in=7, c_out=7,
              d_model=16, n_heads=8, e_layers=2, d_layers=1, d_ff=32, moving_avg=25,
              factor=1, distil=True, dropout=0.1, embed='timeF', activation='gelu',
              output_attention=False, channel_independence=1, decomp_method='moving_avg',
              use_norm=1, down_sampling_layers=3, down_sampling_window=2,
              down_sampling_method='avg', top_k=5, freq='h', task_name='long_term_forecast',
              backbone_mode='dynamic', handtune_w1=-1.0, handtune_w2=-1.0,
              handtune_init=1, handtune_offset=0, handtune_dyn_scale=0.5)
    Cfg = types.SimpleNamespace(**ns)


def make_model():
    """每次新建 Model；内部会消耗 RNG，所以外层 probe 会先重设种子。"""
    from models.TimeMixer import Model
    return Model(Cfg)


probe('完整 Model (真实实验配置)',
      make_model,
      lambda: (torch.randn(B, 96, 7, generator=torch.Generator().manual_seed(0)),
               torch.randn(B, 96, 4, generator=torch.Generator().manual_seed(1)),
               torch.randn(B, 96, 7, generator=torch.Generator().manual_seed(2)),
               torch.randn(B, 96, 4, generator=torch.Generator().manual_seed(3))))

print('-' * 74)
print('A) 优化器 / 多步训练 级别')
print('-' * 74)


def probe_train_steps(name, n_steps=3, use_loader=False, det=True):
    """完整复现「前向→损失→反向→优化器更新」n 步，比对最终参数。
    det=False 时模拟默认配置（cudnn.deterministic=False），
    用来验证 cuDNN 非确定性卷积算子是否就是训练分叉的主因。"""
    runs = []
    for _ in range(2):
        torch.backends.cudnn.deterministic = det
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(2021)
        torch.cuda.manual_seed_all(2021)
        model = make_model().to(DEV)
        opt = torch.optim.Adam(model.parameters(), lr=0.01)
        crit = nn.L1Loss()
        # 固定数据：生成 n_steps 个互不相同的批次
        gen = torch.Generator().manual_seed(7)
        xs = [torch.randn(B, 96, 7, generator=gen) for _ in range(n_steps)]
        ys = [torch.randn(B, 96, 7, generator=gen) for _ in range(n_steps)]
        marks = torch.zeros(B, 96, 4)
        if use_loader:
            # 走一次 DataLoader，检验 shuffle 是否确定
            from torch.utils.data import TensorDataset, DataLoader
            ds = TensorDataset(torch.stack(xs), torch.stack(ys))
            dl = DataLoader(ds, batch_size=B, shuffle=True, num_workers=0)
            batches = list(dl)
            xs = [b[0][0] for b in batches][:n_steps]
            ys = [b[1][0] for b in batches][:n_steps]
        model.train()
        for i in range(min(n_steps, len(xs))):
            opt.zero_grad()
            out = model(xs[i].to(DEV), marks.to(DEV), ys[i].to(DEV), marks.to(DEV))[0]
            loss = crit(out, ys[i].to(DEV))
            loss.backward()
            opt.step()
        runs.append([thash(p) for p in model.parameters()])
    same = runs[0] == runs[1]
    n_diff = sum(1 for a, b in zip(runs[0], runs[1]) if a != b)
    print('  %-34s %s' % (name, '✅ 一致' if same else '❌ 不一致 (%d/%d 参数不同)' % (n_diff, len(runs[0]))))


NSTEP = 200
print('  [对照实验：cudnn.deterministic 开/关]')
probe_train_steps('Adam %d 步  deterministic=True' % NSTEP, NSTEP, False, det=True)
probe_train_steps('Adam %d 步  deterministic=False（默认）' % NSTEP, NSTEP, False, det=False)
probe_train_steps('Adam %d 步  deterministic=True +Loader' % NSTEP, NSTEP, True, det=True)
probe_train_steps('Adam %d 步  deterministic=False +Loader' % NSTEP, NSTEP, True, det=False)

print()
print('-' * 74)
print('B) DataLoader shuffle 确定性')
print('-' * 74)


def probe_loader():
    from torch.utils.data import TensorDataset, DataLoader
    gen = torch.Generator().manual_seed(7)
    x = torch.randn(256, 96, 7, generator=gen)
    y = torch.randn(256, 96, 7, generator=gen)
    ds = TensorDataset(x, y)
    outs = []
    for _ in range(2):
        torch.manual_seed(2021)
        dl = DataLoader(ds, batch_size=128, shuffle=True, num_workers=0)
        outs.append([thash(b[0]) for b in dl])
    ok = outs[0] == outs[1]
    print('  %-34s %s' % ('shuffle=True 的批次顺序',
                          '✅ 一致' if ok else '❌ 不一致（说明有 RNG 未固定）'))


probe_loader()

print('=' * 74)
print('结论：标 ❌ 的层级即为训练非确定性的来源')
print('（模型层已知确定性 ✓；若仅在多步训练处 ❌，说明问题在优化器/循环）')
