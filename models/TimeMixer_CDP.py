# =============================================================================
# CDP: Calibration-based Dynamic Patch (reference reconstruction)
#
# 依据论文 CDP_revised_v2.tex §III 重建（原源码丢失）。三模块：
#   DPS  —— training-free 动态分块（§III-A, Alg.1）：
#           τ_min 由归一化自相关 <0.2 / 主周期确定；L_seg = max(1, ⌊τ_min/M⌋)；
#           段趋势由三重指数平滑(TES)估计；|Δtrend| ≥ errMat 则切分，否则合并。
#   MA   —— 跨 K 分辨率对齐（§III-B）：
#           C_{k,i,j,s} = w_s^T D · exp(−λ‖ΔD‖²)；ŷ = Σ a_{k,j} β_s C；
#           横向项 L_trans = Σ‖D'' − γ·sign(ΔD)‖²。
#   CE   —— 推理阶段逐 stage 加性校准（§III-C）：
#           ŷ_i^cal = ŷ_base,i + A_i⊙trend_i + B_i⊙noise_i + C_i⊙freq_i + d_i
# 训练：两阶段。骨干(TimeMixer)预训练并冻结，只训 CDP 参数(MA + A/B/C/d)。
#
# 超参默认值来源：论文 §III（M=4, K=down_sampling_layers+1, S=16）与
# errMat EMA 衰减 0.9（复现计划 §3.1 待确认项，先用 0.9 占位）。
# λ_t（L_trans 权重）、α/β/γ(TES) 为待回填超参，见 CDP_hyperparam_checklist.md。
# =============================================================================

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.TimeMixer import Model as TimeMixerBackbone


# -----------------------------------------------------------------------------
# DPS: training-free dynamic patch segmentation
# -----------------------------------------------------------------------------
class DPS(nn.Module):
    """对每个分辨率的输入序列做动态分块，输出 per-patch 统计特征。

    输出（per 分辨率）:
        trend_stats  [B*N, P]  每段 TES 趋势斜率（段末 trend 分量 b_t）
        noise_stats  [B*N, P]  每段残差标准差（噪声分量）
        freq_stats   [B*N, P]  每段季节分量幅度（TES seasonal 的 std）
        diff_desc    [B*N, P, S] 段内一阶差分 D 的 S 维描述子（供 MA 的 w_s/β_s）
        diff_energy  [B*N, P]  ‖ΔD‖²（二阶差分能量，供 exp(−λ‖ΔD‖²) 门控）
        transverse   标量        L_transverse（横向趋势项，γ 可学）
        L_seg / P    段长 / 段数
    """

    def __init__(self, M=4, S=16, err_decay=0.9, err_init=0.1,
                 tes_alpha=0.5, tes_beta=0.5, tes_gamma=0.5, noise_all=False):
        super().__init__()
        self.M = M                # 动态分块 ratio（论文默认 4，扫 {2,4,8}）
        self.S = S                # 特征维数（论文默认 16）
        self.err_decay = err_decay
        self.noise_all = noise_all   # v3: 粗分辨率(L_seg=1)也用高通残差定义噪声
        self.tes_alpha, self.tes_beta, self.tes_gamma = tes_alpha, tes_beta, tes_gamma
        # errMat: per-feature 误差阈值，EMA 在线更新（衰减 err_decay）
        self.register_buffer('errMat', torch.tensor(float(err_init)))

    # ---- 归一化自相关 → τ_min（主周期） ----
    @torch.no_grad()
    def _dominant_period(self, x):
        # x: [B*N, T]  返回每样本 τ_min [B*N]（float）
        BN, T = x.shape
        y = x - x.mean(dim=1, keepdim=True)
        nfft = 2 ** math.ceil(math.log2(2 * T))
        f = torch.fft.rfft(y, n=nfft, dim=1)
        acf = torch.fft.irfft(f * torch.conj(f), n=nfft, dim=1)[:, :T].real
        acf = acf / (acf[:, :1].clamp_min(1e-8))          # 归一化，lag0=1
        acf = acf[:, 1:]                                   # 去掉 lag0
        # 主周期 = lag>=2 内第一个局部极大值（周期峰）；无峰则取自相关首次跌破 0.2 的 lag
        rising = torch.zeros_like(acf, dtype=torch.bool)
        rising[:, 1:] = acf[:, 1:] > acf[:, :-1]           # 局部上升沿（峰的候选）
        has_peak = rising.any(dim=1)
        first_peak = torch.argmax(rising.float(), dim=1).float() + 2.0  # lag>=2
        below = acf < 0.2
        has_below = below.any(dim=1)
        first_below = torch.argmax(below.float(), dim=1).float() + 1.0
        tau = torch.where(has_peak, first_peak, torch.where(has_below, first_below,
                                                            torch.full_like(first_below, T / 4.0)))
        tau = tau.clamp(2.0, T / 2.0)
        return tau

    # ---- 三重指数平滑（加性）→ level/trend/seasonal ----
    def _tes(self, x, period_vec):
        # x: [B*N, T]，period_vec: [B*N] 主周期（float）
        BN, T = x.shape
        a, b, g = self.tes_alpha, self.tes_beta, self.tes_gamma
        L = period_vec.clamp(2.0, max(2.0, T / 2.0)).long().clamp(2, T)  # [BN]
        l = x[:, 0].clone()
        b_t = (x[:, 1] - x[:, 0]).clone() if T > 1 else torch.zeros_like(l)
        s = torch.zeros(BN, T + 4, device=x.device, dtype=x.dtype)
        # 初始化季节索引：前 L 个点减去首个 level
        l_init = x[:, 0]
        idx0 = torch.arange(T, device=x.device).unsqueeze(0) % L.unsqueeze(1)  # [BN, T]
        s_init = x - l_init.unsqueeze(1)
        s_first = torch.zeros(BN, int(L.max().item()) + 1, device=x.device, dtype=x.dtype)
        s_first.scatter_add_(1, idx0, s_init)
        cnt = torch.zeros(BN, int(L.max().item()) + 1, device=x.device, dtype=x.dtype)
        cnt.scatter_add_(1, idx0, torch.ones_like(s_init))
        s_first = s_first / cnt.clamp_min(1.0)
        trend_hist = []
        resid_hist = []
        for t in range(T):
            # seasonal 缓存取值：s_{t-L}
            s_full = torch.gather(s_first, 1, idx0[:, t].unsqueeze(1)).squeeze(1)
            l_new = a * (x[:, t] - s_full) + (1 - a) * (l + b_t)
            b_new = b * (l_new - l) + (1 - b) * b_t
            s_new = g * (x[:, t] - l_new) + (1 - g) * s_full
            trend_hist.append(b_new)
            resid_hist.append(x[:, t] - l_new - s_full)
            # 将 s_new 写回 s_first 对应槽位，供 t+L 使用
            s_first.scatter_(1, idx0[:, t].unsqueeze(1), s_new.unsqueeze(1))
            l, b_t = l_new, b_new
        trend_hist = torch.stack(trend_hist, dim=1)      # [BN, T]
        resid_hist = torch.stack(resid_hist, dim=1)      # [BN, T]
        return trend_hist, resid_hist, idx0, s_first

    def forward(self, x):
        # x: [BN, T, 1] 单通道（channel_independence=1）或 [BN, T, C]
        # 描述子统计不需要对骨干求梯度（DPS 内使用了 scatter_ 等 in-place 操作，
        # 若保留计算图会在 unfreeze_norm/head 模式下触发 autograd in-place 冲突）；
        # MA/CE 自身的可学参数仍能正常拿到梯度。
        x = x.detach()
        BN, T, C = x.shape
        x2 = x.squeeze(-1) if C == 1 else x.mean(dim=-1)   # 统计用主通道均值
        tau = self._dominant_period(x2)                    # [BN]
        tau_med = tau.median()
        L_seg = max(1, int(math.floor(tau_med.item() / self.M)))
        L_seg = min(L_seg, T)
        P = T // L_seg                                     # 固定段数（去尾）
        if P == 0:
            P, L_seg = 1, T

        # 段统计（基于 TES 的段末趋势/残差/季节幅度）
        trend_hist, resid_hist, idx0, s_first = self._tes(x2, tau)
        Tr = trend_hist[:, :P * L_seg].reshape(BN, P, L_seg)
        Rs = resid_hist[:, :P * L_seg].reshape(BN, P, L_seg)
        trend_stats = Tr[:, :, -1]                          # 段末趋势 [BN, P]

        def _safe_std(v):
            # 段长为 1 时 torch.std 会产生 NaN，改用手动 var(correction=0)
            # 注意：sqrt 在 0 处反向为 inf，必须加 eps 后再开方（否则 NaN 污染梯度）
            var = ((v - v.mean(dim=2, keepdim=True)) ** 2).mean(dim=2)
            return (var + 1e-8).sqrt()

        noise_stats = _safe_std(Rs)                         # 段内残差 std
        # v3: 段长=1 时 std 恒为 0（噪声校准失效）→ 改用高通残差幅度（x − 移动平均）
        if self.noise_all and L_seg == 1:
            kk = 5
            xpad = F.pad(x2.unsqueeze(1), (kk // 2, kk // 2), mode='replicate')
            ma_s = F.avg_pool1d(xpad, kernel_size=kk, stride=1).squeeze(1)   # [BN, T]
            hf = (x2 - ma_s).abs()[:, :P * L_seg].reshape(BN, P, L_seg)
            noise_stats = hf.mean(dim=2)                    # [BN, P]
        # 季节分量幅度：段内 seasonal 索引的 std
        seas = torch.gather(s_first, 1, idx0)               # [BN, T] seasonal 序列
        Seas = seas[:, :P * L_seg].reshape(BN, P, L_seg)
        freq_stats = _safe_std(Seas)

        # 一阶差分描述子（D）与二阶差分能量（ΔD）
        D = x2[:, 1:] - x2[:, :-1]                          # [BN, T-1]
        if L_seg > 1:
            # 每 patch 贡献 L_seg-1 个差分：补齐到 P*L_seg 后重排，切掉每段末尾
            need = P * L_seg
            D_pad = F.pad(D.unsqueeze(1), (0, max(0, need - D.shape[1])),
                          mode='replicate').squeeze(1)      # [BN, P*L_seg]
            Dp = D_pad[:, :need].reshape(BN, P, L_seg)[:, :, :-1]   # [BN, P, L_seg-1]
        else:
            # L_seg=1：每 patch 一个差分（含跨界差分），补齐到 P 个
            D_pad = F.pad(D.unsqueeze(1), (0, max(0, P - D.shape[1])),
                          mode='replicate').squeeze(1)      # [BN, P]
            Dp = D_pad.unsqueeze(2)                         # [BN, P, 1]
        # S 维描述子：等分 S 个子窗均值
        if L_seg - 1 >= self.S:
            bins = torch.chunk(Dp, self.S, dim=2)
            desc = torch.stack([b.mean(dim=2) for b in bins], dim=2)   # [BN, P, S]
        else:
            desc = F.interpolate(Dp, size=self.S, mode='linear', align_corners=False)
        d2 = Dp[:, :, 1:] - Dp[:, :, :-1] if Dp.shape[2] > 1 else Dp
        diff_energy = (d2 ** 2).mean(dim=2)                 # ‖ΔD‖² 均值 [BN, P]

        # 横向趋势项 L_trans = Σ‖D'' − γ·sign(ΔD)‖²（γ 由 MA 持有，此处先返回 D''）
        transverse = None

        # errMat EMA 更新（逐 epoch 在线统计量）
        with torch.no_grad():
            med = D.abs().median()
            self.errMat.mul_(self.err_decay).add_(med * (1 - self.err_decay))

        # Alg.1 合并判定（段级趋势差 ≥ errMat 则保留边界）
        seg_trend_diff = (trend_stats[:, 1:] - trend_stats[:, :-1]).abs()  # [BN, P-1]
        boundary = (seg_trend_diff >= self.errMat).float()                  # 1=切分

        return {
            'L_seg': L_seg, 'P': P, 'tau': tau_med,
            'trend': trend_stats, 'noise': noise_stats, 'freq': freq_stats,
            'desc': desc, 'diff_energy': diff_energy,
            'boundary': boundary, 'D': Dp, 'transverse': transverse,
        }


# -----------------------------------------------------------------------------
# MA: multi-resolution alignment
# -----------------------------------------------------------------------------
class MA(nn.Module):
    """跨 K 分辨率对齐 DPS patch 特征，产出 per-stage 校准分量张量。

    论文公式映射：
      C_{k,j,s} = w_s · desc_s · exp(−λ‖ΔD‖²)          (趋势敏感门控)
      patch 校准 = Σ_s β_s C_{k,j,s}，a_{k,j} 为 patch 权重
      对齐到预测维：patch 级标量/分量 → 线性插值到 pred_len
      L_transverse = Σ‖D'' − γ·sign(ΔD)‖²             (γ 可学)
    """

    def __init__(self, K, S=16, pred_len=96, max_patches=128):
        super().__init__()
        self.K, self.S, self.pred_len = K, S, pred_len
        self.w = nn.Parameter(torch.ones(S) / S)              # w_s
        self.beta = nn.Parameter(torch.ones(S) / S)           # β_s
        self.lam = nn.Parameter(torch.tensor(0.5))            # λ
        self.gamma = nn.Parameter(torch.tensor(0.1))          # γ
        # a_{k,j}: 每分辨率每 patch 权重（初始化 1 → 从 baseline 出发）
        # 预分配 max_patches，按实际 P 切片（P ≤ T_k ≤ seq_len）
        self.a = nn.ParameterList([nn.Parameter(torch.ones(max_patches)) for _ in range(K)])

    def _gate(self, diff_energy):
        # exp(−λ‖ΔD‖²)：λ 经 softplus 保持正
        lam = F.softplus(self.lam)
        return torch.exp(-lam * diff_energy.clamp_min(0))

    def forward(self, dps_out, k, BN):
        P = dps_out['P']
        a = self.a[k][:P]                                     # [P]
        gate = self._gate(dps_out['diff_energy'])             # [BN, P]
        # C_{k,j,s} = w_s·desc_s·gate  →  patch 值 = Σ_s β_s C
        C = dps_out['desc'] * self.w.view(1, 1, -1) * gate.unsqueeze(-1)  # [BN,P,S]
        patch_cal = (C * self.beta.view(1, 1, -1)).sum(-1)    # [BN, P]

        def to_pred_len(vals):
            # vals [BN, P] → 线性插值到 pred_len；a_{k,j} 加权
            v = vals * a.view(1, P)
            if P == 1:
                return v.expand(BN, self.pred_len)
            return F.interpolate(v.unsqueeze(1), size=self.pred_len,
                                 mode='linear', align_corners=False).squeeze(1)

        cal_freq = to_pred_len(patch_cal)                     # MA 主校准分量
        cal_trend = to_pred_len(dps_out['trend'] * gate)      # 趋势分量（门控后）
        cal_noise = to_pred_len(dps_out['noise'])             # 噪声分量

        # L_transverse: D'' − γ·sign(ΔD)
        D = dps_out['D']                                      # [BN, P, L_seg-1]
        if D.shape[2] > 2:
            d1 = D[:, :, 1:] - D[:, :, :-1]
            d2 = d1[:, :, 1:] - d1[:, :, :-1]
            trans = ((d2 - F.softplus(self.gamma) * torch.sign(d1[:, :, 1:])) ** 2).mean()
        else:
            trans = torch.zeros(1, device=D.device).squeeze()
        return cal_trend, cal_noise, cal_freq, trans


# -----------------------------------------------------------------------------
# CE: per-stage additive calibration engine
# -----------------------------------------------------------------------------
class CE(nn.Module):
    """ŷ_k^cal = ŷ_k + A_k⊙trend_k + B_k⊙noise_k + C_k⊙freq_k + d_k

    A/B/C: per-stage 长度 pred_len 的可学权重向量；d: per-stage 标量。
    初始化：A=B=d=0（trend/noise 校准从 baseline 出发）；
    C=1e-2（小非零初值，保证 MA 参数从第一步就有梯度，避免 0×链式死锁；
    对 baseline 的偏移量级 ~1e-3，可忽略）。
    """

    def __init__(self, K, pred_len=96, use_gain=True):
        super().__init__()
        self.K, self.pred_len = K, pred_len
        self.use_gain = use_gain
        self.A = nn.ParameterList([nn.Parameter(torch.zeros(pred_len)) for _ in range(K)])
        self.B = nn.ParameterList([nn.Parameter(torch.zeros(pred_len)) for _ in range(K)])
        self.C = nn.ParameterList([nn.Parameter(torch.full((pred_len,), 1e-2)) for _ in range(K)])
        self.d = nn.ParameterList([nn.Parameter(torch.zeros(1)) for _ in range(K)])
        # v3: 每 stage 每 step 的乘性增益 G_k（init 1）——直接对应手调实验
        #     out = 0.75*out_season + 0.93*out_trend 这类缩放；同时兼作 per-stage 混合权重
        if use_gain:
            self.gain = nn.ParameterList([nn.Parameter(torch.ones(pred_len)) for _ in range(K)])
        else:
            self.gain = None

    def forward(self, k, dec_k, trend_k, noise_k, freq_k):
        # dec_k/trend_k/noise_k/freq_k: [BN, pred_len]
        base = dec_k if self.gain is None else dec_k * self.gain[k].unsqueeze(0)
        return base + self.A[k].unsqueeze(0) * trend_k \
                    + self.B[k].unsqueeze(0) * noise_k \
                    + self.C[k].unsqueeze(0) * freq_k \
                    + self.d[k]


# -----------------------------------------------------------------------------
# v3: 输出级校准（逐 horizon-step 仿射 + 逐通道低秩偏置）
# -----------------------------------------------------------------------------
class FinalCal(nn.Module):
    """ŷ = a ⊙ ŷ_sum + b（a init 1，b init 0，长度 pred_len）
        + 逐通道低秩偏置 (U @ V)（rank=0 关闭；U init 0 → 起点严格等于 baseline）

    动机：Traffic/ECL 这类高通道数据集上，各通道存在稳定但不同的系统性偏移，
    逐 horizon-step 的仿射项可以一次性纠掉「随预测步长增长的偏差」，
    这是纯加性 per-stage 校准做不到的。
    """

    def __init__(self, pred_len, n_channels, rank=0, rank_mul=0):
        super().__init__()
        self.pred_len, self.n_channels, self.rank = pred_len, n_channels, rank
        self.rank_mul = rank_mul
        self.a = nn.Parameter(torch.ones(pred_len))
        self.b = nn.Parameter(torch.zeros(pred_len))
        # 低秩项初值：U 取小随机、V 取 0 → 乘积恒为 0（起点严格等于 baseline），
        # 但 dL/dV = U^T·grad ≠ 0，梯度可流动（若 U/V 同时取 0 会造成梯度死锁，永远学不动）
        if rank > 0:
            self.U = nn.Parameter(torch.randn(n_channels, rank) * 0.01)
            self.V = nn.Parameter(torch.zeros(rank, pred_len))
        # v3.1: 逐通道「乘性」低秩校准 y = (1 + Ua@Va) ⊙ y
        #   不同通道的尺度偏差各异，共享标量 a 无法刻画；低秩分解让它可学且参数少
        if rank_mul > 0:
            self.Ua = nn.Parameter(torch.randn(n_channels, rank_mul) * 0.01)
            self.Va = nn.Parameter(torch.zeros(rank_mul, pred_len))

    def forward(self, y):
        # y: [B, pred_len, N]
        if self.rank_mul > 0:
            a_eff = self.a.view(1, -1, 1) + (self.Ua @ self.Va).t().unsqueeze(0)
            y = a_eff * y + self.b.view(1, -1, 1)
        else:
            y = self.a.view(1, -1, 1) * y + self.b.view(1, -1, 1)
        if self.rank > 0:
            y = y + (self.U @ self.V).t().unsqueeze(0)      # [1, pred_len, N]
        return y


# -----------------------------------------------------------------------------
# v4: 输入自适应输出校准（把「拟合 patch 分布」用到输出级）
# -----------------------------------------------------------------------------
class DynFinalCal(nn.Module):
    """ŷ_i = (1 + s_i) ⊙ ŷ + b_i，其中 (s_i, b_i) 由「该样本自己的 patch 分布统计」预测。

    与 FinalCal 的区别：FinalCal 的 a/b 是全局共享的常数（+ 静态逐通道低秩），
    对同一条预测曲线只做一次固定仿射；DynFinalCal 让校准量随输入序列的
    trend/noise/freq/差分分布变化 —— 同一 batch 内不同序列拿到各自的校准量，
    这正是 CDP 论文「拟合不同 patch 的分布模式」在输出端最直接的实现。

    末层小随机初始化（σ=1e-4）：起点等价于 baseline（偏移量级 ~1e-4），
    同时保证梯度可流过整个 MLP（零初始化会造成梯度死锁）。
    """

    def __init__(self, K, pred_len, n_feat=5, hidden=64):
        super().__init__()
        self.pred_len, self.n_feat = pred_len, n_feat
        in_dim = K * n_feat
        self.norm = nn.LayerNorm(in_dim)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, pred_len * 2))
        nn.init.normal_(self.net[-1].weight, mean=0.0, std=1e-4)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, y_flat, feats):
        # y_flat: [BN, pred_len]；feats: [BN, K*n_feat]
        z = self.net(self.norm(feats))
        s = z[:, :self.pred_len]
        b = z[:, self.pred_len:]
        return (1.0 + s) * y_flat + b


# -----------------------------------------------------------------------------
# CDP container
# -----------------------------------------------------------------------------
class CDP(nn.Module):
    def __init__(self, configs, n_channels=None):
        super().__init__()
        K = configs.down_sampling_layers + 1
        self.K = K
        self.pred_len = configs.pred_len
        self.dps = DPS(M=getattr(configs, 'cdp_M', 4),
                       S=getattr(configs, 'cdp_S', 16),
                       err_decay=getattr(configs, 'cdp_err_decay', 0.9),
                       noise_all=bool(getattr(configs, 'cdp_noise_all', 0)))
        self.ma = MA(K=K, S=getattr(configs, 'cdp_S', 16), pred_len=configs.pred_len,
                     max_patches=max(128, configs.seq_len))
        self.ce = CE(K=K, pred_len=configs.pred_len,
                     use_gain=bool(getattr(configs, 'cdp_gain', 1)))
        # v3: 输出级校准（逐 step 仿射 + 逐通道低秩偏置）
        if getattr(configs, 'cdp_out_affine', 1) and n_channels:
            self.final_cal = FinalCal(pred_len=configs.pred_len, n_channels=n_channels,
                                      rank=int(getattr(configs, 'cdp_chan_rank', 0)),
                                      rank_mul=int(getattr(configs, 'cdp_chan_rank_mul', 0)))
        else:
            self.final_cal = None
        # v4: 输入自适应输出校准（由 patch 分布统计预测逐样本 (scale, bias)）
        self.dyn_final_cal = (DynFinalCal(K, configs.pred_len)
                              if int(getattr(configs, 'cdp_dyncal', 0) or 0) else None)
        # v3.1: per-stage 可学习混合权重（原版是各 stage 等权相加）
        #   alpha_k init 1 → 起点严格等于 baseline；学习各分辨率预测的贡献度
        if int(getattr(configs, 'cdp_stage_w', 1) or 0):
            self.stage_w = nn.Parameter(torch.ones(K))
            self.stage_b = nn.Parameter(torch.zeros(K))
        else:
            self.stage_w = None
            self.stage_b = None
        self.last_transverse = torch.tensor(0.0)
        self._B, self._N = None, None

    def forward(self, x_list, dec_out_list, B, N):
        # x_list: list of [B*N, T_k, 1]；dec_out_list: list of [B, pred_len, N]
        self._B, self._N = B, N
        cal_stages = []
        feats = []
        trans_total = 0.0
        for k in range(self.K):
            dps_out = self.dps(x_list[k])
            trend_k, noise_k, freq_k, trans = self.ma(dps_out, k, x_list[k].shape[0])
            dec_k = dec_out_list[k].permute(0, 2, 1).reshape(-1, self.pred_len)  # [BN, pred]
            cal_k = self.ce(k, dec_k, trend_k, noise_k, freq_k)
            cal_stages.append(cal_k.reshape(B, N, self.pred_len).permute(0, 2, 1))
            trans_total = trans_total + trans
            if self.dyn_final_cal is not None:
                feats.append(torch.stack([
                    dps_out['trend'].abs().mean(dim=1),
                    dps_out['noise'].mean(dim=1),
                    dps_out['freq'].mean(dim=1),
                    dps_out['desc'].abs().mean(dim=(1, 2)),
                    dps_out['diff_energy'].mean(dim=1),
                ], dim=1))                                     # [BN, 5]
        self._feats = torch.cat(feats, dim=1) if feats else None  # [BN, 5K]
        self.last_transverse = trans_total
        return cal_stages

    def combine(self, cal_stages):
        """各 stage 加权求和 + 输入自适应校准 + 输出级校准（v3.1 / v4）。"""
        if self.stage_w is not None:
            dec_out = 0.0
            for k, cs in enumerate(cal_stages):
                dec_out = dec_out + self.stage_w[k] * cs + self.stage_b[k] / max(1, len(cal_stages))
        else:
            dec_out = torch.stack(cal_stages, dim=-1).sum(-1)
        # v4: 先做输入自适应校准（逐样本 scale/bias），再做全局静态校准
        if self.dyn_final_cal is not None and self._feats is not None:
            B, N = self._B, self._N
            y_flat = dec_out.permute(0, 2, 1).reshape(-1, self.pred_len)   # [BN, pred]
            y_flat = self.dyn_final_cal(y_flat, self._feats)
            dec_out = y_flat.reshape(B, N, self.pred_len).permute(0, 2, 1)
        if self.final_cal is not None:
            dec_out = self.final_cal(dec_out)
        return dec_out


# -----------------------------------------------------------------------------
# Model: TimeMixer backbone (frozen) + CDP plugin
# -----------------------------------------------------------------------------
class Model(TimeMixerBackbone):
    """TimeMixer + CDP（两阶段训练：骨干冻结，只训 CDP）。"""

    def __init__(self, configs):
        super(Model, self).__init__(configs)
        self.cdp = CDP(configs, n_channels=configs.enc_in)
        # v3: 可选解冻输出头（predict_layers/projection_layer）——「CDP + Head Adaptation」
        self.unfreeze_head = bool(int(getattr(configs, 'cdp_unfreeze_head', 0) or 0))
        # v3.1: 可选解冻 PDM 的季节/趋势合成权重（backbone_mode=learn 时才存在）——「LSEW」
        self.unfreeze_pdm = bool(int(getattr(configs, 'cdp_unfreeze_pdm', 0) or 0))
        # v3.1: 可选解冻各分辨率的归一化仿射参数（逐通道 scale/shift）——「归一化校准」
        #   Traffic/ECL 这类高通道数据上，逐通道的分布漂移是主要误差来源之一
        self.unfreeze_norm = bool(int(getattr(configs, 'cdp_unfreeze_norm', 0) or 0))
        # v3.1: 可选解冻最后 N 个 PDM 块（部分骨干微调）——CDP 只做输出校准时提升有天花板，
        #       放开最后若干层的容量是拿到更大增益的关键（学习率由 cdp_head_lr_ratio 控制）
        self.unfreeze_blocks = int(getattr(configs, 'cdp_unfreeze_blocks', 0) or 0)
        self.freeze_backbone()
        if self.unfreeze_head:
            n_head = 0
            for name, p in self.named_parameters():
                if ('predict_layers' in name) or ('projection_layer' in name):
                    p.requires_grad = True
                    n_head += p.numel()
            print('[CDP] head adaptation enabled, trainable head params =', n_head)
        if self.unfreeze_norm:
            n_norm = 0
            for name, p in self.named_parameters():
                if 'normalize_layers' in name and ('affine_weight' in name or 'affine_bias' in name):
                    p.requires_grad = True
                    n_norm += p.numel()
            print('[CDP] normalization calibration enabled, trainable norm params =', n_norm)
        if self.unfreeze_blocks > 0:
            e = int(getattr(self.configs, 'e_layers', 1))
            keep = set(range(max(0, e - self.unfreeze_blocks), e))
            n_blk = 0
            for name, p in self.named_parameters():
                if name.startswith('pdm_blocks.'):
                    try:
                        idx = int(name.split('.')[1])
                    except (IndexError, ValueError):
                        continue
                    if idx in keep:
                        p.requires_grad = True
                        n_blk += p.numel()
            print('[CDP] partial backbone fine-tune: last {} PDM blocks, trainable params = {}'.format(
                self.unfreeze_blocks, n_blk))
        if self.unfreeze_pdm:
            n_pdm = 0
            for name, p in self.named_parameters():
                if name.endswith(('season_w', 'trend_w', 'bias_w')) or ('.dyn_net.' in name):
                    p.requires_grad = True
                    n_pdm += p.numel()
            print('[CDP] learnable season/trend mixing (w1_k/w2_k/b_k + dynamic hyper-net) '
                  'enabled, params =', n_pdm)

    def freeze_backbone(self):
        for name, p in self.named_parameters():
            if not name.startswith('cdp.'):
                p.requires_grad = False

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # ---- 以下复制自 TimeMixer.forecast，但整体在 no_grad 下执行（骨干冻结），
        #      并捕获 x_list（各分辨率输入）与 dec_out_list（各 stage 预测输出）----
        #  v3: 若启用 head adaptation，则保留计算图（输出头需要梯度）
        import contextlib
        _need_grad = (self.unfreeze_head or self.unfreeze_pdm or self.unfreeze_norm
                      or self.unfreeze_blocks > 0)
        _ctx = contextlib.nullcontext() if _need_grad else torch.no_grad()
        with _ctx:
            if self.use_future_temporal_feature:
                if self.channel_independence == 1:
                    B, T, N = x_enc.size()
                    x_mark_dec = x_mark_dec.repeat(N, 1, 1)
                    self.x_mark_dec = self.enc_embedding(None, x_mark_dec)
                else:
                    self.x_mark_dec = self.enc_embedding(None, x_mark_dec)

            x_enc_, x_mark_enc_ = self.__multi_scale_process_inputs(x_enc, x_mark_enc)

            x_list, x_mark_list = [], []
            if x_mark_enc_ is not None:
                for i, x, x_mark in zip(range(len(x_enc_)), x_enc_, x_mark_enc_):
                    B, T, N = x.size()
                    x = self.normalize_layers[i](x, 'norm')
                    if self.channel_independence == 1:
                        x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
                        x_mark = x_mark.repeat(N, 1, 1)
                    x_list.append(x)
                    x_mark_list.append(x_mark)
            else:
                for i, x in zip(range(len(x_enc_)), x_enc_):
                    B, T, N = x.size()
                    x = self.normalize_layers[i](x, 'norm')
                    if self.channel_independence == 1:
                        x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
                    x_list.append(x)

            enc_out_list = []
            x_list2 = self.pre_enc(x_list)
            if x_mark_enc_ is not None:
                for i, x, x_mark in zip(range(len(x_list2[0])), x_list2[0], x_mark_list):
                    enc_out_list.append(self.enc_embedding(x, x_mark))
            else:
                for i, x in zip(range(len(x_list2[0])), x_list2[0]):
                    enc_out_list.append(self.enc_embedding(x, None))

            for i in range(self.layer):
                enc_out_list = self.pdm_blocks[i](enc_out_list)

            dec_out_list = self.future_multi_mixing(B, enc_out_list, x_list2)

        # ---- CDP 校准（唯一可训练路径）----
        cal_stages = self.cdp(x_list, dec_out_list, B, x_enc.shape[-1])

        dec_out = self.cdp.combine(cal_stages)
        dec_out = self.normalize_layers[0](dec_out, 'denorm')
        return dec_out
