import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Autoformer_EncDec import series_decomp
from layers.Embed import DataEmbedding_wo_pos
from layers.StandardNorm import Normalize

# =============================================================================
# 作者手调「季节 / 趋势」合成权重表
#
# 出处：本文件 PastDecomposableMixing.forward 原始注释里记录的网格搜索结果。
#   骨干原本是 out = out_season + out_trend（官方实现）。
#   作者在每个 (数据集, horizon) 上单独搜索了 w1/w2 的最优值：
#       ETTh1   96:(1.024, 0.87)  192:(1.11, 0.8)  336:(1.228, 0.6)  720:(1.126, 0.497)
#       weather 96:(1.3, 0.55)    192:(0.85, 1.35) 336:(0.95, 1.0)    720:(0.75, 0.93)
#       ECL     96:(0.97, 0.98)   192:(0.95, 0.98) 336:(0.95, 0.9)    720:(0.93, 0.97)
#       Traffic 96:(0.925, 0.98)  192:(0.87, 0.95) 336:(0.8, 0.92)
#
# ⚠️ 关键教训：0.75/0.93 只是 weather-720 的那一组，是一个「数据集×horizon 专用」
#   常数。把它当全局常数硬编码进骨干 = 给所有实验注入错误偏差（曾污染全部结果）。
#   因此这里改成查表，并且支持把 w1/w2 变成可学习参数（见 backbone_mode）。
# =============================================================================
HANDTUNED_TABLE = {
    ('ETTh1', 96): (1.024, 0.87),
    ('ETTh1', 192): (1.11, 0.8),
    ('ETTh1', 336): (1.228, 0.6),
    ('ETTh1', 720): (1.126, 0.497),
    ('weather', 96): (1.3, 0.55),
    ('weather', 192): (0.85, 1.35),
    ('weather', 336): (0.95, 1.0),
    ('weather', 720): (0.75, 0.93),
    ('ECL', 96): (0.97, 0.98),
    ('ECL', 192): (0.95, 0.98),
    ('ECL', 336): (0.95, 0.9),
    ('ECL', 720): (0.93, 0.97),
    ('Traffic', 96): (0.925, 0.98),
    ('Traffic', 192): (0.87, 0.95),
    ('Traffic', 336): (0.8, 0.92),
    ('Traffic', 720): (0.87, 0.95),   # 作者未给 720，沿用 192 一组占位
    ('ETTh2', 96): (1.0, 1.0),
    ('ETTh2', 192): (1.0, 1.0),
    ('ETTh2', 336): (1.0, 1.0),
    ('ETTh2', 720): (1.0, 1.0),
    ('ETTm2', 96): (1.0, 1.0),
    ('ETTm2', 192): (1.0, 1.0),
    ('ETTm2', 336): (1.0, 1.0),
    ('ETTm2', 720): (1.0, 1.0),
}


def resolve_dataset_name(configs):
    """从 root_path / data_path / data 里稳健地识别数据集名（Traffic/ECL 的 data=custom）。"""
    s = '{} {} {}'.format(getattr(configs, 'root_path', ''),
                          getattr(configs, 'data_path', ''),
                          getattr(configs, 'data', '')).lower()
    for key, name in (('traffic', 'Traffic'), ('electricity', 'ECL'), ('ecl', 'ECL'),
                      ('weather', 'weather'), ('etth1', 'ETTh1'), ('etth2', 'ETTh2'),
                      ('ettm1', 'ETTm1'), ('ettm2', 'ETTm2')):
        if key in s:
            return name
    return None


def resolve_handtuned_pair(configs, default=(1.0, 1.0)):
    """返回该 (数据集, horizon) 的手调 (w1, w2)；命令行 --handtune_w1/--handtune_w2 优先。"""
    w1 = float(getattr(configs, 'handtune_w1', -1.0) or -1.0)
    w2 = float(getattr(configs, 'handtune_w2', -1.0) or -1.0)
    if w1 >= 0.0 and w2 >= 0.0:
        return (w1, w2)
    name = resolve_dataset_name(configs)
    if name is not None:
        pair = HANDTUNED_TABLE.get((name, int(getattr(configs, 'pred_len', 96))))
        if pair is not None:
            return pair
    return default


class DFT_series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, top_k=5):
        super(DFT_series_decomp, self).__init__()
        self.top_k = top_k

    def forward(self, x):
        xf = torch.fft.rfft(x)
        freq = abs(xf)
        freq[0] = 0
        top_k_freq, top_list = torch.topk(freq, self.top_k)
        xf[freq <= top_k_freq.min()] = 0
        x_season = torch.fft.irfft(xf)
        x_trend = x - x_season
        return x_season, x_trend


class MultiScaleSeasonMixing(nn.Module):
    """
    Bottom-up mixing season pattern
    """

    def __init__(self, configs):
        super(MultiScaleSeasonMixing, self).__init__()

        self.down_sampling_layers = torch.nn.ModuleList(
            [
                nn.Sequential(
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** i),
                        configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                    ),
                    nn.GELU(),
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                        configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                    ),

                )
                for i in range(configs.down_sampling_layers)
            ]
        )

    def forward(self, season_list):

        # mixing high->low
        out_high = season_list[0]
        out_low = season_list[1]
        out_season_list = [out_high.permute(0, 2, 1)]

        for i in range(len(season_list) - 1):
            out_low_res = self.down_sampling_layers[i](out_high)
            out_low = out_low + out_low_res
            out_high = out_low
            if i + 2 <= len(season_list) - 1:
                out_low = season_list[i + 2]
            out_season_list.append(out_high.permute(0, 2, 1))

        return out_season_list


class MultiScaleTrendMixing(nn.Module):
    """
    Top-down mixing trend pattern
    """

    def __init__(self, configs):
        super(MultiScaleTrendMixing, self).__init__()

        self.up_sampling_layers = torch.nn.ModuleList(
            [
                nn.Sequential(
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                        configs.seq_len // (configs.down_sampling_window ** i),
                    ),
                    nn.GELU(),
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** i),
                        configs.seq_len // (configs.down_sampling_window ** i),
                    ),
                )
                for i in reversed(range(configs.down_sampling_layers))
            ])

    def forward(self, trend_list):

        # mixing low->high
        trend_list_reverse = trend_list.copy()
        trend_list_reverse.reverse()
        out_low = trend_list_reverse[0]
        out_high = trend_list_reverse[1]
        out_trend_list = [out_low.permute(0, 2, 1)]

        for i in range(len(trend_list_reverse) - 1):
            out_high_res = self.up_sampling_layers[i](out_low)
            out_high = out_high + out_high_res
            out_low = out_high
            if i + 2 <= len(trend_list_reverse) - 1:
                out_high = trend_list_reverse[i + 2]
            out_trend_list.append(out_low.permute(0, 2, 1))

        out_trend_list.reverse()
        return out_trend_list


class PastDecomposableMixing(nn.Module):
    def __init__(self, configs):
        super(PastDecomposableMixing, self).__init__()
        self.configs = configs
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.down_sampling_window = configs.down_sampling_window

        self.layer_norm = nn.LayerNorm(configs.d_model)
        self.dropout = nn.Dropout(configs.dropout)
        self.channel_independence = configs.channel_independence

        if configs.decomp_method == 'moving_avg':
            self.decompsition = series_decomp(configs.moving_avg)
        elif configs.decomp_method == "dft_decomp":
            self.decompsition = DFT_series_decomp(configs.top_k)
        else:
            raise ValueError('decompsition is error')

        if configs.channel_independence == 0:
            self.cross_layer = nn.Sequential(
                nn.Linear(in_features=configs.d_model, out_features=configs.d_ff),
                nn.GELU(),
                nn.Linear(in_features=configs.d_ff, out_features=configs.d_model),
            )

        # Mixing season
        self.mixing_multi_scale_season = MultiScaleSeasonMixing(configs)

        # Mxing trend
        self.mixing_multi_scale_trend = MultiScaleTrendMixing(configs)

        self.out_cross_layer = nn.Sequential(
            nn.Linear(in_features=configs.d_model, out_features=configs.d_ff),
            nn.GELU(),
            nn.Linear(in_features=configs.d_ff, out_features=configs.d_model),
        )

        # ------------------------------------------------------------------
        # 季节/趋势合成权重（backbone_mode = learn / dynamic 时注册为可学习参数）
        #   learn   : out = w1_k·season + w2_k·trend + b_k     （逐分辨率 k 各一组）
        #   dynamic : 在 learn 的基础上，再由 per-patch 分布统计预测「逐样本」的权重偏移
        #   初始化：默认用作者手调表（--handtune_init 1），即从作者搜到的好点出发；
        #           查不到表则退化为 (1,1,0)，严格等于 clean 官方实现。
        # 仅在 learn/dynamic 下注册，保证 clean/handtuned 的 state_dict key 与官方一致。
        # ------------------------------------------------------------------
        self.bb_mode = getattr(configs, 'backbone_mode', 'clean')
        self.n_res = int(configs.down_sampling_layers) + 1
        if self.bb_mode in ('learn', 'dynamic'):
            if int(getattr(configs, 'handtune_init', 1) or 0):
                w1_0, w2_0 = resolve_handtuned_pair(configs)
            else:
                w1_0, w2_0 = 1.0, 1.0
            self.season_w = nn.Parameter(torch.full((self.n_res,), float(w1_0)))
            self.trend_w = nn.Parameter(torch.full((self.n_res,), float(w2_0)))
            self.bias_w = nn.Parameter(torch.zeros(self.n_res))
            print('[PDM] backbone_mode={} | init season_w={} trend_w={}'.format(
                self.bb_mode, [round(v, 4) for v in self.season_w.detach().tolist()],
                [round(v, 4) for v in self.trend_w.detach().tolist()]))
        # dynamic：超网络把「该样本的 season/trend 分布统计」映射成权重偏移
        #   末层用小随机初始化（σ=1e-4）而非零初始化：零初始化会让 tanh 前的隐层
        #   拿不到梯度（grad_h = W2^T·grad = 0）→ 超网络永远学不动；
        #   1e-4 的随机初值带来的输出偏移量级 ~5e-5，起点仍等价于 clean/手调初值。
        if self.bb_mode == 'dynamic':
            self.dyn_scale = float(getattr(configs, 'handtune_dyn_scale', 0.5) or 0.5)
            self.dyn_net = nn.Sequential(
                nn.Linear(6, 32), nn.GELU(), nn.Linear(32, 3))
            nn.init.normal_(self.dyn_net[2].weight, mean=0.0, std=1e-4)
            nn.init.zeros_(self.dyn_net[2].bias)

    def _patch_weight_delta(self, out_season, out_trend):
        """CDP 精髓：从 per-patch（本例为 per-series）的 season/trend 分布统计，
        预测该样本专属的 (Δw1, Δw2, Δb)。tanh 有界 → 权重不会跑飞。"""
        s, t = out_season, out_trend
        stats = torch.stack([
            s.std(dim=(1, 2)), t.std(dim=(1, 2)),
            s.abs().mean(dim=(1, 2)), t.abs().mean(dim=(1, 2)),
            s.abs().amax(dim=(1, 2)), t.abs().amax(dim=(1, 2)),
        ], dim=1)                                          # [B, 6]
        stats = torch.log(stats + 1e-6)
        return torch.tanh(self.dyn_net(stats)) * self.dyn_scale    # [B, 3]

    def forward(self, x_list):
        length_list = []
        for x in x_list:
            _, T, _ = x.size()
            length_list.append(T)

        # Decompose to obtain the season and trend
        season_list = []
        trend_list = []
        for x in x_list:
            season, trend = self.decompsition(x)
            if self.channel_independence == 0:
                season = self.cross_layer(season)
                trend = self.cross_layer(trend)
            season_list.append(season.permute(0, 2, 1))
            trend_list.append(trend.permute(0, 2, 1))

        # bottom-up season mixing
        out_season_list = self.mixing_multi_scale_season(season_list)
        # top-down trend mixing
        out_trend_list = self.mixing_multi_scale_trend(trend_list)

        out_list = []
        for k, (ori, out_season, out_trend, length) in enumerate(
                zip(x_list, out_season_list, out_trend_list, length_list)):
            # TODO: 关键代码：季节+趋势 ,改进：为季节性/趋势分量引入可学习权重（如out = w1*out_season + w2*out_trend），适配不同数据特性。
            # ETTh1预测96最佳参数：1.024; 0.87
            # ETTh1预测192最佳参数：1.11; 0.8
            # ETTh1预测336最佳参数：1.228; 0.6
            # ETTh1预测720最佳参数：1.126; 0.497
            # WEATHER_96_1.3_0.55
            # WEATHER_192_0.85_1.35
            # WEATHER_336_0.95_1
            # WEATHER_720_0.75_0.93
            # ECL_96_0.97_0.98
            # ECL_192_0.95_0.98
            # ECL_336_0.95_0.9
            # ECL_720_0.93_0.97
            # Traffic_96_0.925_0.98
            # Traffic_192_0.87_0.95
            # Traffic_336_0.8_0.92
            # ---- 骨干季节/趋势合成方式（可切换）----
            # clean     : 上游官方实现 out = out_season + out_trend（论文基线必须用这个）
            # handtuned : 作者手调常数「按 (数据集, horizon) 查表」（--handtune_w1/w2 可覆盖）
            # learn     : 可学习合成 out = w1_k*season + w2_k*trend + b_k（逐分辨率各一组）
            # dynamic   : learn + 由 per-patch 分布统计预测逐样本权重偏移（CDP 精髓）
            _mode = getattr(self.configs, 'backbone_mode', 'clean')
            if _mode == 'handtuned':
                w1, w2 = resolve_handtuned_pair(self.configs)
                if int(getattr(self.configs, 'handtune_offset', 0) or 0):
                    # 精确复现作者示例写法：w1*season + 0.3 + w2*trend + 0.05
                    out = w1 * out_season + 0.3 + w2 * out_trend + 0.05
                else:
                    out = w1 * out_season + w2 * out_trend
            elif _mode in ('learn', 'dynamic'):
                ki = min(k, self.n_res - 1)
                w1 = self.season_w[ki]
                w2 = self.trend_w[ki]
                b = self.bias_w[ki]
                if _mode == 'dynamic':
                    d = self._patch_weight_delta(out_season, out_trend)      # [B, 3]
                    w1 = w1 + d[:, 0:1].unsqueeze(-1)
                    w2 = w2 + d[:, 1:2].unsqueeze(-1)
                    b = b + d[:, 2:3].unsqueeze(-1)
                out = w1 * out_season + w2 * out_trend + b
            else:
                out = out_season + out_trend
            # print("季节性:", out_season, "趋势：", out_trend * 0.2)
            if self.channel_independence:
                out = ori + self.out_cross_layer(out)
            out_list.append(out[:, :length, :])
        return out_list


class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.down_sampling_window = configs.down_sampling_window
        self.channel_independence = configs.channel_independence
        # 通过configs.e_layers动态决定层数，实现模型容量与计算资源的灵活配置。将复杂算子封装为独立模块（PastDecomposableMixing），提升代码可读性和可维护性。
        self.pdm_blocks = nn.ModuleList([PastDecomposableMixing(configs)
                                         for _ in range(configs.e_layers)])

        self.preprocess = series_decomp(configs.moving_avg)
        self.enc_in = configs.enc_in
        self.use_future_temporal_feature = configs.use_future_temporal_feature

        if self.channel_independence == 1:
            self.enc_embedding = DataEmbedding_wo_pos(1, configs.d_model, configs.embed, configs.freq,
                                                      configs.dropout)
        else:
            self.enc_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                      configs.dropout)

        self.layer = configs.e_layers

        self.normalize_layers = torch.nn.ModuleList(
            [
                Normalize(self.configs.enc_in, affine=True, non_norm=True if configs.use_norm == 0 else False)
                for i in range(configs.down_sampling_layers + 1)
            ]
        )

        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.predict_layers = torch.nn.ModuleList(
                [
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** i),
                        configs.pred_len,
                    )
                    for i in range(configs.down_sampling_layers + 1)
                ]
            )

            if self.channel_independence == 1:
                self.projection_layer = nn.Linear(
                    configs.d_model, 1, bias=True)
            else:
                self.projection_layer = nn.Linear(
                    configs.d_model, configs.c_out, bias=True)

                self.out_res_layers = torch.nn.ModuleList([
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** i),
                        configs.seq_len // (configs.down_sampling_window ** i),
                    )
                    for i in range(configs.down_sampling_layers + 1)
                ])

                self.regression_layers = torch.nn.ModuleList(
                    [
                        torch.nn.Linear(
                            configs.seq_len // (configs.down_sampling_window ** i),
                            configs.pred_len,
                        )
                        for i in range(configs.down_sampling_layers + 1)
                    ]
                )
        if self.task_name == 'imputation' or self.task_name == 'anomaly_detection':
            if self.channel_independence == 1:
                self.projection_layer = nn.Linear(
                    configs.d_model, 1, bias=True)
            else:
                self.projection_layer = nn.Linear(
                    configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(
                configs.d_model * configs.seq_len, configs.num_class)

    def out_projection(self, dec_out, i, out_res):
        dec_out = self.projection_layer(dec_out)
        out_res = out_res.permute(0, 2, 1)
        out_res = self.out_res_layers[i](out_res)
        out_res = self.regression_layers[i](out_res).permute(0, 2, 1)
        dec_out = dec_out + out_res
        return dec_out

    def pre_enc(self, x_list):
        if self.channel_independence == 1:
            return (x_list, None)
        else:
            out1_list = []
            out2_list = []
            for x in x_list:
                # x1是残差，x2是移动平均
                x_1, x_2 = self.preprocess(x)
                out1_list.append(x_1)
                out2_list.append(x_2)
            return (out1_list, out2_list)

    def __multi_scale_process_inputs(self, x_enc, x_mark_enc):
        # 编码器输入x_enc（形状[batch_size, sequence_length, features]）和时间标记x_mark_enc（可选，如时间戳、季节性编码）。
        # Max Pooling：torch.nn.MaxPool1d，取窗口内最大值，保留显著特征。
        if self.configs.down_sampling_method == 'max':
            down_pool = torch.nn.MaxPool1d(self.configs.down_sampling_window, return_indices=False)
        # Average Pooling：torch.nn.AvgPool1d，取窗口内均值，平滑噪声。
        elif self.configs.down_sampling_method == 'avg':
            down_pool = torch.nn.AvgPool1d(self.configs.down_sampling_window)
        # Convolutional Downsampling：通过nn.Conv1d实现，使用stride=down_sampling_window实现下采样，padding_mode='circular'保持序列周期性（避免边界效应）。
        elif self.configs.down_sampling_method == 'conv':
            padding = 1 if torch.__version__ >= '1.5.0' else 2
            down_pool = nn.Conv1d(in_channels=self.configs.enc_in, out_channels=self.configs.enc_in,
                                  kernel_size=3, padding=padding,
                                  stride=self.configs.down_sampling_window,
                                  padding_mode='circular',
                                  bias=False)
        else:
            return x_enc, x_mark_enc
        # B,T,C -> B,C,T x_enc.permute(0,2,1)将输入从[B,T,C]转为[B,C,T]，适配PyTorch的池化/卷积层（要求输入为[batch, channels, length]）。
        x_enc = x_enc.permute(0, 2, 1)

        x_enc_ori = x_enc
        x_mark_enc_mark_ori = x_mark_enc

        x_enc_sampling_list = []
        x_mark_sampling_list = []
        x_enc_sampling_list.append(x_enc.permute(0, 2, 1))
        x_mark_sampling_list.append(x_mark_enc)
        # 循环down_sampling_layers次，每次对上一尺度的输出进行下采样，形成金字塔结构（如原始序列→下采样1次→下采样2次→...）。
        for i in range(self.configs.down_sampling_layers):
            x_enc_sampling = down_pool(x_enc_ori)

            x_enc_sampling_list.append(x_enc_sampling.permute(0, 2, 1))
            x_enc_ori = x_enc_sampling

            if x_mark_enc_mark_ori is not None:
                x_mark_sampling_list.append(x_mark_enc_mark_ori[:, ::self.configs.down_sampling_window, :])
                x_mark_enc_mark_ori = x_mark_enc_mark_ori[:, ::self.configs.down_sampling_window, :]

        x_enc = x_enc_sampling_list
        if x_mark_enc_mark_ori is not None:
            x_mark_enc = x_mark_sampling_list
        else:
            x_mark_enc = x_mark_enc

        return x_enc, x_mark_enc

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.use_future_temporal_feature:
            if self.channel_independence == 1:
                B, T, N = x_enc.size()
                x_mark_dec = x_mark_dec.repeat(N, 1, 1)
                self.x_mark_dec = self.enc_embedding(None, x_mark_dec)
            else:
                self.x_mark_dec = self.enc_embedding(None, x_mark_dec)
        # 多尺度处理输入数据，平滑噪音，做下采样
        x_enc, x_mark_enc = self.__multi_scale_process_inputs(x_enc, x_mark_enc)

        x_list = []
        x_mark_list = []
        if x_mark_enc is not None:
            for i, x, x_mark in zip(range(len(x_enc)), x_enc, x_mark_enc):
                B, T, N = x.size()
                x = self.normalize_layers[i](x, 'norm')
                if self.channel_independence == 1:
                    x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
                    x_mark = x_mark.repeat(N, 1, 1)
                x_list.append(x)
                x_mark_list.append(x_mark)
        else:
            for i, x in zip(range(len(x_enc)), x_enc, ):
                B, T, N = x.size()
                x = self.normalize_layers[i](x, 'norm')
                if self.channel_independence == 1:
                    x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
                x_list.append(x)

        # embedding
        enc_out_list = []
        # 对经过平滑去燥和下采样后的数据做提取 # x1是残差，x2是移动平均
        x_list = self.pre_enc(x_list)
        if x_mark_enc is not None:
            for i, x, x_mark in zip(range(len(x_list[0])), x_list[0], x_mark_list):
                enc_out = self.enc_embedding(x, x_mark)  # [B,T,C]
                enc_out_list.append(enc_out)
        else:
            for i, x in zip(range(len(x_list[0])), x_list[0]):
                enc_out = self.enc_embedding(x, None)  # [B,T,C]
                enc_out_list.append(enc_out)

        # Past Decomposable Mixing as encoder for past，做Encode预测
        for i in range(self.layer):
            enc_out_list = self.pdm_blocks[i](enc_out_list)

        # 调用future_multi_mixing方法生成多步预测结果列表。
        # B：通常表示批量大小（batch size），用于控制并行计算维度。
        # enc_out_list：编码器（如PastDecomposableMixing模块）的输出列表，包含历史序列的特征表示。
        # x_list：原始输入序列或中间特征，可能包含当前步的输入特征。
        dec_out_list = self.future_multi_mixing(B, enc_out_list, x_list)
        # print("返回：", dec_out_list)
        # # TODO: (1)加权求和（手动定义权重）
        # weights = torch.tensor([0.5, 0.3, 0.2])  # 假设3个尺度的权重
        # dec_out_stacked = torch.stack(dec_out_list, dim=-1)  # [B, T, C, S]
        # dec_out = (dec_out_stacked * weights).sum(-1)  # 加权求和
        # # TODO: (2)注意力融合（动态学习权重）
        # attention_weights = torch.softmax(torch.randn(S), dim=-1)  # 生成随机注意力权重
        # dec_out = (dec_out_stacked * attention_weights).sum(-1)
        # torch.stack在dim=-1维度（最后一个维度）堆叠列表中的张量。
        dec_out = torch.stack(dec_out_list, dim=-1)


        # 定义时间步权重 [96个时间步]
        # time_weights = nn.Parameter(torch.linspace(0.5, 1.5, 96), requires_grad=True)
        # # 时间步维度加权
        # weighted_dec_out = dec_out * time_weights.view(1, -1, 1)  # 广播到[128,96,7]
        # dec_out = weighted_dec_out.sum(-1)


        # 砍掉最后一个维度（原代码）
        dec_out = dec_out.sum(-1)
        dec_out = self.normalize_layers[0](dec_out, 'denorm')
        # print("返回的预测结果1：", dec_out)
        return dec_out

    def future_multi_mixing(self, B, enc_out_list, x_list):
        dec_out_list = []
        if self.channel_independence == 1:
            x_list = x_list[0]
            for i, enc_out in zip(range(len(x_list)), enc_out_list):
                # 经过训练好的layers，乘以对应的权重参数
                dec_out = self.predict_layers[i](enc_out.permute(0, 2, 1)).permute(
                    0, 2, 1)  # align temporal dimension
                if self.use_future_temporal_feature:
                    dec_out = dec_out + self.x_mark_dec
                    dec_out = self.projection_layer(dec_out)
                else:
                    # print("进来")
                    dec_out = self.projection_layer(dec_out)
                dec_out = dec_out.reshape(B, self.configs.c_out, self.pred_len).permute(0, 2, 1).contiguous()
                dec_out_list.append(dec_out)

        else:
            for i, enc_out, out_res in zip(range(len(x_list[0])), enc_out_list, x_list[1]):
                dec_out = self.predict_layers[i](enc_out.permute(0, 2, 1)).permute(
                    0, 2, 1)  # align temporal dimension
                dec_out = self.out_projection(dec_out, i, out_res)
                dec_out_list.append(dec_out)

        return dec_out_list

    def classification(self, x_enc, x_mark_enc):
        x_enc, _ = self.__multi_scale_process_inputs(x_enc, None)
        x_list = x_enc

        # embedding
        enc_out_list = []
        for x in x_list:
            enc_out = self.enc_embedding(x, None)  # [B,T,C]
            enc_out_list.append(enc_out)

        # MultiScale-CrissCrossAttention  as encoder for past
        for i in range(self.layer):
            enc_out_list = self.pdm_blocks[i](enc_out_list)

        enc_out = enc_out_list[0]
        # Output
        # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.act(enc_out)
        output = self.dropout(output)
        # zero-out padding embeddings
        output = output * x_mark_enc.unsqueeze(-1)
        # (batch_size, seq_length * d_model)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)  # (batch_size, num_classes)
        return output

    def anomaly_detection(self, x_enc):
        B, T, N = x_enc.size()
        x_enc, _ = self.__multi_scale_process_inputs(x_enc, None)

        x_list = []

        for i, x in zip(range(len(x_enc)), x_enc, ):
            B, T, N = x.size()
            x = self.normalize_layers[i](x, 'norm')
            if self.channel_independence == 1:
                x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
            x_list.append(x)

        # embedding
        enc_out_list = []
        for x in x_list:
            enc_out = self.enc_embedding(x, None)  # [B,T,C]
            enc_out_list.append(enc_out)

        # MultiScale-CrissCrossAttention  as encoder for past
        for i in range(self.layer):
            enc_out_list = self.pdm_blocks[i](enc_out_list)

        dec_out = self.projection_layer(enc_out_list[0])
        dec_out = dec_out.reshape(B, self.configs.c_out, -1).permute(0, 2, 1).contiguous()

        dec_out = self.normalize_layers[0](dec_out, 'denorm')
        return dec_out

    def imputation(self, x_enc, x_mark_enc, mask):
        means = torch.sum(x_enc, dim=1) / torch.sum(mask == 1, dim=1)
        means = means.unsqueeze(1).detach()
        x_enc = x_enc - means
        x_enc = x_enc.masked_fill(mask == 0, 0)
        stdev = torch.sqrt(torch.sum(x_enc * x_enc, dim=1) /
                           torch.sum(mask == 1, dim=1) + 1e-5)
        stdev = stdev.unsqueeze(1).detach()
        x_enc /= stdev

        B, T, N = x_enc.size()
        x_enc, x_mark_enc = self.__multi_scale_process_inputs(x_enc, x_mark_enc)

        x_list = []
        x_mark_list = []
        if x_mark_enc is not None:
            for i, x, x_mark in zip(range(len(x_enc)), x_enc, x_mark_enc):
                B, T, N = x.size()
                if self.channel_independence == 1:
                    x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
                x_list.append(x)
                x_mark = x_mark.repeat(N, 1, 1)
                x_mark_list.append(x_mark)
        else:
            for i, x in zip(range(len(x_enc)), x_enc, ):
                B, T, N = x.size()
                if self.channel_independence == 1:
                    x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
                x_list.append(x)

        # embedding
        enc_out_list = []
        for x in x_list:
            enc_out = self.enc_embedding(x, None)  # [B,T,C]
            enc_out_list.append(enc_out)

        # MultiScale-CrissCrossAttention  as encoder for past
        for i in range(self.layer):
            enc_out_list = self.pdm_blocks[i](enc_out_list)

        dec_out = self.projection_layer(enc_out_list[0])
        dec_out = dec_out.reshape(B, self.configs.c_out, -1).permute(0, 2, 1).contiguous()

        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':

            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out
        if self.task_name == 'imputation':
            dec_out = self.imputation(x_enc, x_mark_enc, mask)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        else:
            raise ValueError('Other tasks implemented yet')
