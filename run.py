import argparse
import torch

from exp.exp_anomaly_detection import Exp_Anomaly_Detection
from exp.exp_classification import Exp_Classification
from exp.exp_imputation import Exp_Imputation
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from exp.exp_long_term_forecasting_cdp import Exp_Long_Term_Forecast_CDP
from exp.exp_short_term_forecasting import Exp_Short_Term_Forecast
import random
import numpy as np

# ===========================================================================
# 随机种子 / 确定性开关（2026-10-08 新增，用于多种子补跑，审稿 P1-2）
#
# 顺序在这里至关重要，踩过一次坑：
#   `CUBLAS_WORKSPACE_CONFIG` 必须在 cuBLAS handle 创建之前设置，
#   而 torch.cuda.is_available() / torch.cuda.manual_seed() 就会初始化 CUDA。
#   所以环境变量必须放在 `import torch` 之前 —— 放在 seeding 之后再设是无效的
#   （实测：那样做同 seed 两次的 MSE 仍差 0.22%）。
# ===========================================================================
import os as _os
import argparse as _argparse_pre

_pre_parser = _argparse_pre.ArgumentParser(add_help=False)
_pre_parser.add_argument('--fix_seed', type=int, default=2021,
                         help='random seed (default 2021, 保持历史结果可复现)')
_pre_parser.add_argument('--det_mode', type=int, default=0,
                         help='1 = 启用 cuDNN/cuBLAS 确定性算子（多种子实验的确定性前提）')
_pre_args, _ = _pre_parser.parse_known_args()

if _pre_args.det_mode in (1, 2):
    # 必须在 import torch 之前！否则 cuBLAS GEMM 的非确定性无法消除。
    _os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import argparse
import torch

from exp.exp_anomaly_detection import Exp_Anomaly_Detection
from exp.exp_classification import Exp_Classification
from exp.exp_imputation import Exp_Imputation
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from exp.exp_long_term_forecasting_cdp import Exp_Long_Term_Forecast_CDP
from exp.exp_short_term_forecasting import Exp_Short_Term_Forecast
import random
import numpy as np

fix_seed = _pre_args.fix_seed
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(fix_seed)
    torch.cuda.manual_seed_all(fix_seed)

if _pre_args.det_mode == 1:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception as _e:  # 旧版本 torch 没有该 API
        print('[seed] use_deterministic_algorithms unavailable: %s' % _e)
elif _pre_args.det_mode == 2:
    # det_mode=2 = 严格确定性：遇到未实现确定性版本的算子直接抛错，
    # 用来定位究竟是哪个 op 在破坏可复现性（诊断用，跑通后可回到 det_mode=1）。
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

print('[seed] fix_seed=%d det_mode=%d cuda=%s cublas_ws=%s det_algos=%s'
      % (fix_seed, _pre_args.det_mode, torch.cuda.is_available(),
         _os.environ.get('CUBLAS_WORKSPACE_CONFIG', '<unset>'),
         torch.are_deterministic_algorithms_enabled()))

parser = argparse.ArgumentParser(description='TimeMixer')

# basic config
parser.add_argument('--task_name', type=str, required=True, default='long_term_forecast',
                    help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
parser.add_argument('--model', type=str, required=True, default='Autoformer',
                    help='model name, options: [Autoformer, Transformer, TimesNet]')

# data loader
parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type')
parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly, ms:milliseconds], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

# forecasting task
parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
parser.add_argument('--label_len', type=int, default=48, help='start token length')
parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)

# model define
parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
parser.add_argument('--c_out', type=int, default=7, help='output size')
parser.add_argument('--d_model', type=int, default=16, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=32, help='dimension of fcn')
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
parser.add_argument('--factor', type=int, default=1, help='attn factor')
parser.add_argument('--distil', action='store_false',
                    help='whether to use distilling in encoder, using this argument means not using distilling',
                    default=True)
parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
parser.add_argument('--channel_independence', type=int, default=1,
                    help='0: channel dependence 1: channel independence for FreTS model')
parser.add_argument('--decomp_method', type=str, default='moving_avg',
                    help='method of series decompsition, only support moving_avg or dft_decomp')
parser.add_argument('--use_norm', type=int, default=1, help='whether to use normalize; True 1 False 0')
parser.add_argument('--down_sampling_layers', type=int, default=0, help='num of down sampling layers')
parser.add_argument('--down_sampling_window', type=int, default=1, help='down sampling window size')
parser.add_argument('--down_sampling_method', type=str, default='avg',
                    help='down sampling method, only support avg, max, conv')
parser.add_argument('--use_future_temporal_feature', type=int, default=0,
                    help='whether to use future_temporal_feature; True 1 False 0')

# imputation task
parser.add_argument('--mask_rate', type=float, default=0.125, help='mask ratio')

# anomaly detection task
parser.add_argument('--anomaly_ratio', type=float, default=0.25, help='prior anomaly ratio (%)')

# optimization
parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=100, help='train epochs')
parser.add_argument('--batch_size', type=int, default=16, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=15, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='MSE', help='loss function')
parser.add_argument('--drop_last', type=bool, default=True, help='drop last')
parser.add_argument('--lradj', type=str, default='TST', help='adjust learning rate')
parser.add_argument('--pct_start', type=float, default=0.2, help='pct_start')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
parser.add_argument('--comment', type=str, default='none', help='com')
# 随机种子 / 确定性开关 —— 实际的 seeding 发生在文件顶部的预解析块（必须早于下游随机初始化）。
# 这里重复注册只是为了让 parse_args() 不报 "unrecognized arguments"。
parser.add_argument('--fix_seed', type=int, default=2021, help='random seed (pre-parsed at top of run.py)')
parser.add_argument('--det_mode', type=int, default=0, help='1 = deterministic cudnn/algorithms (pre-parsed at top of run.py)')
# SWA 实验专用：逐 epoch 保存 checkpoint（默认关）。用于事后做「验证集平台期集成」，
# 以替换对噪声极敏感的 argmin(val) 选点。见 tools/eval_ckpt_ensemble.py。
parser.add_argument('--save_all_epochs', type=int, default=0, help='1 = save checkpoint after every epoch (SWA experiment)')

# GPU
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1', help='device ids of multile gpus')

# de-stationary projector params
parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                    help='hidden layer dimensions of projector (List)')
parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')

# CDP plugin params（超参默认值见 CDP_hyperparam_checklist.md，待作者回填确认）
parser.add_argument('--cdp_M', type=int, default=4, help='CDP DPS resolution factor M (paper default 4, sweep {2,4,8})')
parser.add_argument('--cdp_S', type=int, default=16, help='CDP MA feature dim S (paper default 16)')
parser.add_argument('--cdp_err_decay', type=float, default=0.9, help='CDP errMat EMA decay (0.9)')
parser.add_argument('--cdp_lambda_t', type=float, default=0.01, help='CDP L_transverse weight lambda_t')
parser.add_argument('--cdp_learning_rate', type=float, default=1e-4, help='CDP stage-2 learning rate (paper 1e-4)')
parser.add_argument('--cdp_backbone_checkpoint', type=str, default='',
                    help='frozen backbone checkpoint dir for CDP stage-2 training')
# ---- CDP v3 增强开关（默认全部开启；设为 0 可回退到 v2 行为，便于消融）----
parser.add_argument('--cdp_gain', type=int, default=1,
                    help='v3: per-stage per-step multiplicative gain on dec_k (init 1)')
parser.add_argument('--cdp_out_affine', type=int, default=1,
                    help='v3: output-level per-step affine a*y+b (a init 1, b init 0)')
parser.add_argument('--cdp_chan_rank', type=int, default=0,
                    help='v3: per-channel low-rank per-step bias rank (0=off, try 4/8)')
parser.add_argument('--cdp_noise_all', type=int, default=1,
                    help='v3: define noise at coarse resolutions via high-pass residual (L_seg=1)')
parser.add_argument('--cdp_loss', type=str, default='mse', choices=['mse', 'mse_l1', 'l1'],
                    help='stage-2 loss: mse | mse_l1 (MSE + mu*L1)')
parser.add_argument('--cdp_l1_mu', type=float, default=0.5, help='weight of L1 term in mse_l1 loss')
parser.add_argument('--cdp_monitor', type=str, default='mse', choices=['mse', 'mae'],
                    help='early-stopping / best-checkpoint monitor metric')
parser.add_argument('--cdp_ema_decay', type=float, default=0.0,
                    help='CDP param EMA decay (0=off, e.g. 0.9)')
parser.add_argument('--cdp_unfreeze_head', type=int, default=0,
                    help='v3: unfreeze output heads (predict_layers/projection_layer) for head adaptation')
parser.add_argument('--cdp_head_lr_ratio', type=float, default=0.1,
                    help='lr ratio of head-adaptation params vs cdp lr (when cdp_unfreeze_head=1)')
parser.add_argument('--cdp_st_lr_ratio', type=float, default=1.0,
                    help='v4: lr ratio of season/trend mixing params vs cdp lr (backbone_mode=learn/dynamic)')
parser.add_argument('--cdp_cal_lr_ratio', type=float, default=1.0,
                    help='v4.1: lr ratio of final_cal/stage_w/dyn_final_cal vs cdp lr; '
                         'these params now always use weight_decay=0')
parser.add_argument('--cdp_step_weight', type=float, default=0.0,
                    help='v4.1: step-wise loss weighting w_t = 1 + a*t/(T-1) (0=off); '
                         'helps long horizons where far steps dominate the error')
parser.add_argument('--cdp_dyncal', type=int, default=0,
                    help='v4: 输入自适应输出校准——由 patch 分布统计预测逐样本 (scale,bias)，'
                         '对应 CDP「拟合不同 patch 分布模式」的动机')
# ---- v4: Stage-1（骨干训练）损失 ----
parser.add_argument('--train_loss', type=str, default='mse', choices=['mse', 'mse_l1', 'l1'],
                    help='v4: 骨干训练损失；mse_l1 = MSE + mu*L1；l1 = 纯 L1（直接优化 MAE）')
parser.add_argument('--train_l1_mu', type=float, default=0.5,
                    help='v4: train_loss=mse_l1 时的 L1 权重')
parser.add_argument('--cdp_log_test', type=int, default=0,
                    help='also evaluate test set each epoch (slower; 0=skip)')
# ---- CDP v3.1 增强开关 ----
parser.add_argument('--cdp_stage_w', type=int, default=1,
                    help='v3.1: learnable per-stage mixing weight alpha_k (init 1 = 等权求和)')
parser.add_argument('--cdp_unfreeze_pdm', type=int, default=0,
                    help='v3.1: 解冻骨干 PDM 的季节/趋势合成权重（需 --backbone_mode learn）')
parser.add_argument('--cdp_unfreeze_norm', type=int, default=0,
                    help='v3.1: 解冻各分辨率归一化层的逐通道仿射参数（归一化校准）')
parser.add_argument('--cdp_chan_rank_mul', type=int, default=0,
                    help='v3.1: 逐通道「乘性」低秩校准的秩（0=off，高通道数据集建议 4/8）')
parser.add_argument('--cdp_unfreeze_blocks', type=int, default=0,
                    help='v3.1: 解冻最后 N 个 PDM 块（部分骨干微调，lr 由 cdp_head_lr_ratio 控制）')

# ---- 骨干季节/趋势合成方式（关键：默认 clean = 上游官方实现）----
# 背景：本地 models/TimeMixer.py 的 PastDecomposableMixing.forward 曾被手工改成
#   out = 0.75*out_season + 0.3 + 0.93*out_trend + 0.05（WEATHER_720 的调参值），
# 那是一个「数据集×horizon 专用」的常数，绝不可作为通用骨干。正式实验必须用 clean。
# 现在改为：handtuned 走 (数据集, horizon) 查表；learn/dynamic 把 w1/w2 变成可学习参数。
parser.add_argument('--backbone_mode', type=str, default='clean',
                    choices=['clean', 'handtuned', 'learn', 'dynamic'],
                    help='clean=官方 out=season+trend（论文基线）; '
                         'handtuned=按(数据集,horizon)查作者手调表; '
                         'learn=可学习 w1_k*season+w2_k*trend+b_k（逐分辨率）; '
                         'dynamic=learn + 由 per-patch 分布统计预测逐样本权重偏移（CDP 精髓）')
parser.add_argument('--handtune_w1', type=float, default=-1.0,
                    help='手调季节权重覆盖（<0 = 用作者表）；配合 --backbone_mode handtuned')
parser.add_argument('--handtune_w2', type=float, default=-1.0,
                    help='手调趋势权重覆盖（<0 = 用作者表）')
parser.add_argument('--handtune_init', type=int, default=1,
                    help='learn/dynamic 模式下用作者手调表初始化 w1/w2（1=是，0=用 1.0）')
parser.add_argument('--handtune_offset', type=int, default=0,
                    help='handtuned 模式是否连同 +0.3/+0.05 偏置一起复现（1=精确复现作者示例）')
parser.add_argument('--handtune_dyn_scale', type=float, default=0.5,
                    help='dynamic 模式权重偏移的幅度上限（tanh 缩放，默认 0.5）')

if __name__ == '__main__':
    args = parser.parse_args()
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print(args)

    if args.task_name == 'long_term_forecast':
        Exp = Exp_Long_Term_Forecast_CDP if args.model == 'TimeMixer_CDP' else Exp_Long_Term_Forecast
    elif args.task_name == 'short_term_forecast':
        Exp = Exp_Short_Term_Forecast
    elif args.task_name == 'imputation':
        Exp = Exp_Imputation
    elif args.task_name == 'anomaly_detection':
        Exp = Exp_Anomaly_Detection
    elif args.task_name == 'classification':
        Exp = Exp_Classification
    else:
        Exp = Exp_Long_Term_Forecast

    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            setting = '{}_{}_{}_{}_{}_sl{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
                args.task_name,
                args.model_id,
                args.comment,
                args.model,
                args.data,
                args.seq_len,
                args.pred_len,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.factor,
                args.embed,
                args.distil,
                args.des, ii)

            exp = Exp(args)  # set experiments
            print('======================',torch.cuda.is_available())
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)
            torch.cuda.empty_cache()
    else:
        ii = 0
        setting = '{}_{}_{}_{}_{}_sl{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
            args.task_name,
            args.model_id,
            args.comment, 
            args.model,
            args.data,
            args.seq_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.factor,
            args.embed,
            args.distil,
            args.des, ii)

        exp = Exp(args)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()
