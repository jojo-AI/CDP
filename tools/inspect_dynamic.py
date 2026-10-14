"""诊断 dynamic 模式实际学到了什么：读取各 Stage-1 checkpoint 的 season_w/trend_w/bias_w 与 dyn_net。

用法: python tools/inspect_dynamic.py
"""
import torch, glob, os, sys

CK = 'checkpoints'
RUNS = [
    ('ETTh1-96', 'dynamic', 'long_term_forecast_ETTh1_st1_96_dynamic_none_*'),
    ('ETTh1-96', 'dynamic(L1)', 'long_term_forecast_ETTh1_st1_96_dynamic_l10.0_*'),
    ('ETTh1-96', 'learn', 'long_term_forecast_ETTh1_st1_96_learn_none_*'),
    ('ETTh1-96', 'handtuned', 'long_term_forecast_ETTh1_st1_96_handtuned_none_*'),
    ('ETTh1-96', 'clean', 'long_term_forecast_ETTh1_st1_96_clean_none_*'),
    ('ETTh1-192', 'dynamic', 'long_term_forecast_ETTh1_st1_192_dynamic_l10.0_*'),
    ('ETTh1-336', 'dynamic', 'long_term_forecast_ETTh1_st1_336_dynamic_l10.0_*'),
    ('ETTh1-720', 'dynamic', 'long_term_forecast_ETTh1_st1_720_dynamic_l10.0_*'),
    ('weather-96', 'dynamic', 'long_term_forecast_weather_st1_96_dynamic_none_*'),
    ('weather-96', 'learn', 'long_term_forecast_weather_st1_96_learn_none_*'),
    ('weather-96', 'handtuned', 'long_term_forecast_weather_st1_96_handtuned_none_*'),
]

TABLE_HINT = {
    ('ETTh1', 96): (1.024, 0.87), ('ETTh1', 192): (1.11, 0.8),
    ('ETTh1', 336): (1.228, 0.6), ('ETTh1', 720): (1.126, 0.497),
    ('weather', 96): (1.3, 0.55),
}


def load(pat):
    dirs = sorted(glob.glob(os.path.join(CK, pat)))
    for d in dirs:
        f = os.path.join(d, 'checkpoint.pth')
        if os.path.exists(f):
            o = torch.load(f, map_location='cpu', weights_only=False)
            if isinstance(o, dict) and 'model' in o and isinstance(o['model'], dict):
                o = o['model']
            return o, os.path.basename(d)
    return None, None


print('%-11s %-12s %-42s %-42s %s' % ('设定', '模式', 'season_w (逐分辨率)', 'trend_w (逐分辨率)', 'bias_w'))
print('-' * 150)
records = []
for ds, mode, pat in RUNS:
    sd, dname = load(pat)
    if sd is None:
        print('%-11s %-12s %s' % (ds, mode, '未找到 checkpoint'))
        continue
    sw = [k for k in sd if k.endswith('season_w')]
    if not sw:
        print('%-11s %-12s %s' % (ds, mode, '无权重参数（clean 官方公式 out=season+trend）'))
        continue
    k = sw[0]
    s = [round(v, 4) for v in sd[k].float().tolist()]
    t = [round(v, 4) for v in sd[k.replace('season_w', 'trend_w')].float().tolist()]
    b = [round(v, 4) for v in sd[k.replace('season_w', 'bias_w')].float().tolist()]
    print('%-11s %-12s %-42s %-42s %s' % (ds, mode, s, t, b))
    records.append((ds, mode, s, t, b, sd))

print()
print('=== dyn_net（超网络）是否真的学动了 ===')
for ds, mode, s, t, b, sd in records:
    wk = [k for k in sd if k.endswith('dyn_net.2.weight')]
    if not wk:
        print('%-11s %-12s 无 dyn_net' % (ds, mode))
        continue
    W = sd[wk[0]].float()
    bk = wk[0].replace('.weight', '.bias')
    print('%-11s %-12s 末层|W|均值 %.3e  最大 %.3e  末层bias %s  (初始 σ=1e-4)'
          % (ds, mode, W.abs().mean(), W.abs().max(), [round(v, 6) for v in sd[bk].float().tolist()]))

print()
print('=== 参数规模 ===')
sample, _ = load(RUNS[0][2])
if sample:
    total = sum(v.numel() for v in sample.values() if hasattr(v, 'numel'))
    newp = sum(v.numel() for k, v in sample.items()
               if hasattr(v, 'numel') and ('season_w' in k or 'trend_w' in k or 'bias_w' in k or 'dyn_net' in k))
    print('ETTh1-96 dynamic 总参数 %d ；其中新增（权重合成+超网络）%d = 占比 %.3f%%' % (total, newp, 100.0 * newp / total))
