# =============================================================================
# CDP 两阶段训练/测试流程（继承自 Exp_Long_Term_Forecast，最小改动）：
#   Stage 1（已有）：骨干 TimeMixer baseline 预训练 → checkpoint.pth
#   Stage 2（本类）：加载冻结骨干 checkpoint，只训 CDP（MA + CE 的 A/B/C/d），
#                     损失 L = L_MSE + λ_t·L_transverse，AdamW lr=1e-4（论文 Impl）
# =============================================================================
from torch.optim import lr_scheduler

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np

warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast_CDP(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast_CDP, self).__init__(args)
        # ---- Stage 2：加载冻结骨干权重 ----
        if getattr(self.args, 'cdp_backbone_checkpoint', ''):
            ckpt = os.path.join(self.args.cdp_backbone_checkpoint, 'checkpoint.pth')
            state = torch.load(ckpt, map_location='cpu')
            missing, unexpected = self.model.load_state_dict(state, strict=False)
            # 骨干权重应全部命中；missing 只应含 cdp.*（CDP 新参数），
            # 以及 backbone_mode=learn/dynamic 时新增的 PDM 季节/趋势合成权重与动态超网络
            def _ok(m):
                if m.startswith('cdp.'):
                    return True
                if '.dyn_net.' in m:
                    return True
                return m.endswith(('season_w', 'trend_w', 'bias_w'))
            bad = [m for m in missing if not _ok(m)]
            assert not bad, '骨干 checkpoint 不匹配，missing keys: {}'.format(bad[:5])
            print('[CDP] loaded frozen backbone from', ckpt)
            print('[CDP] missing (expected cdp.* / learn-mode weights only):', len(missing), 'params')

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        # 论文：AdamW，只优化 CDP 可学参数，lr=cdp_learning_rate（默认 1e-4）
        # v3 : 若解冻输出头，则 head 用更小学习率（cdp_head_lr_ratio，默认 0.1）
        # v4 : 季节/趋势合成权重（w1_k/w2_k/b_k）与动态超网络单独一组，
        #      可用 --cdp_st_lr_ratio 单独给 lr（默认 1.0 = 与 CDP 同 lr）
        # v4.1: 校准类参数（final_cal/stage_w/dyn_final_cal）单独一组——
        #   诊断发现 AdamW 默认 weight_decay=0.01 会把这些 init≈0/1 的小参数
        #   钉死在初始值附近（h336 上 final_cal.a std 仅 0.0007），必须 wd=0
        cdp_params, cal_params, head_params, st_params = [], [], [], []
        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if n.startswith('cdp.'):
                if ('final_cal' in n) or ('.stage_w' in n) or ('.stage_b' in n):
                    cal_params.append(p)
                else:
                    cdp_params.append(p)
            elif n.endswith(('season_w', 'trend_w', 'bias_w')) or ('.dyn_net.' in n):
                st_params.append(p)
            else:
                head_params.append(p)
        groups = [{'params': cdp_params, 'lr': self.args.cdp_learning_rate, 'weight_decay': 0.0}]
        self._max_lr = [self.args.cdp_learning_rate]
        clr = self.args.cdp_learning_rate * float(getattr(self.args, 'cdp_cal_lr_ratio', 1.0))
        groups.append({'params': cal_params, 'lr': clr, 'weight_decay': 0.0})
        self._max_lr.append(clr)
        print('[CDP] param groups: cdp={} tensors lr={} wd=0 | cal={} tensors lr={} wd=0'.format(
            len(cdp_params), self.args.cdp_learning_rate, len(cal_params), clr))
        if st_params:
            slr = self.args.cdp_learning_rate * float(getattr(self.args, 'cdp_st_lr_ratio', 1.0))
            groups.append({'params': st_params, 'lr': slr})
            self._max_lr.append(slr)
            print('[CDP] param groups: cdp={} tensors lr={} | season/trend={} tensors lr={}'.format(
                len(cdp_params), self.args.cdp_learning_rate, len(st_params), slr))
        if head_params:
            hlr = self.args.cdp_learning_rate * float(getattr(self.args, 'cdp_head_lr_ratio', 0.1))
            groups.append({'params': head_params, 'lr': hlr})
            self._max_lr.append(hlr)
            print('[CDP] param groups: cdp={} tensors lr={} | head={} tensors lr={}'.format(
                len(cdp_params), self.args.cdp_learning_rate, len(head_params), hlr))
        return optim.AdamW(groups)

    def _select_criterion(self):
        # v3: 支持 MSE+μ·L1 混合损失（表格同时报 MAE，L1 项直接优化 MAE）
        _l = str(getattr(self.args, 'cdp_loss', 'mse'))
        # v4.1: 步级加权——长 horizon 上远端 step 误差大且难学，
        #   w_t = 1 + α·t/(T-1)（按均值归一），让校准向远端倾斜
        sw = float(getattr(self.args, 'cdp_step_weight', 0) or 0)
        step_w = None
        if sw > 0 and self.args.pred_len > 1:
            _w = torch.linspace(1.0, 1.0 + sw, self.args.pred_len)
            step_w = (_w / _w.mean()).view(1, self.args.pred_len, 1)
            print('[CDP] step-weighted loss: alpha={}, w[0]={:.3f} w[-1]={:.3f}'.format(
                sw, step_w[0, 0, 0].item(), step_w[0, -1, 0].item()))

        def _red(err):
            if step_w is None:
                return err.mean()
            return (err * step_w.to(err.device)).mean()

        if _l == 'mse_l1':
            mu = float(getattr(self.args, 'cdp_l1_mu', 0.5))
            def _crit(pred, true):
                d = pred - true
                return _red(d.pow(2)) + mu * _red(d.abs())
            return _crit
        if _l == 'l1':
            # v4: 纯 L1 —— MAE 用一个「对不同样本一视同仁」的绝对误差目标，
            #     当 MAE 是主要瓶颈时比 MSE 主导的损失更对口
            def _crit(pred, true):
                return _red((pred - true).abs())
            return _crit
        def _crit(pred, true):
            return _red((pred - true).pow(2))
        return _crit

    def vali(self, vali_data, vali_loader, criterion):
        # 返回 (mse, mae)：流式累计，避免拼接大数组
        total_sq, total_abs, n = 0.0, 0.0, 0
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                if 'PEMS' == self.args.data or 'Solar' == self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                dec_inp = None
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                diff = outputs.detach() - batch_y.detach()
                total_sq += float((diff ** 2).sum())
                total_abs += float(diff.abs().sum())
                n += diff.numel()
        self.model.train()
        return total_sq / max(n, 1), total_abs / max(n, 1)

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                                            steps_per_epoch=train_steps,
                                            pct_start=self.args.pct_start,
                                            epochs=self.args.train_epochs,
                                            max_lr=self._max_lr)

        # v3: CDP 参数 EMA（权重平均，抑制后期震荡；decay=0 关闭）
        ema_decay = float(getattr(self.args, 'cdp_ema_decay', 0.0) or 0.0)
        ema_state = None
        if ema_decay > 0:
            ema_state = {n: p.detach().clone() for n, p in self.model.named_parameters()
                         if n.startswith('cdp.')}
            print('[CDP] EMA enabled, decay={}'.format(ema_decay))
        monitor = str(getattr(self.args, 'cdp_monitor', 'mse'))
        print('[CDP] loss={}, monitor={}'.format(getattr(self.args, 'cdp_loss', 'mse'), monitor))

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                if 'PEMS' == self.args.data or 'Solar' == self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                dec_inp = None

                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                loss = criterion(outputs, batch_y)
                # L = L_MSE + λ_t · L_transverse
                trans = getattr(self.model, 'cdp', None)
                if trans is not None and isinstance(getattr(trans, 'last_transverse', None), torch.Tensor):
                    loss = loss + self.args.cdp_lambda_t * trans.last_transverse
                train_loss.append(loss.item())

                if (i + 1) % 200 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(
                        i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                model_optim.step()
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
                scheduler.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_mse, vali_mae = self.vali(vali_data, vali_loader, criterion)
            if int(getattr(self.args, 'cdp_log_test', 0) or 0):
                test_mse, test_mae = self.vali(test_data, test_loader, criterion)
            else:
                test_mse, test_mae = float('nan'), float('nan')

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali MSE: {3:.7f} Vali MAE: {4:.7f} "
                  "Test MSE: {5:.7f} Test MAE: {6:.7f}".format(
                      epoch + 1, train_steps, train_loss, vali_mse, vali_mae, test_mse, test_mae))
            # SWA 实验用：逐 epoch 存 checkpoint（默认关，绝不影响既有实验）
            # 与 exp_long_term_forecasting.py 完全对应；Stage-2 的集成评估依赖它。
            if int(getattr(self.args, 'save_all_epochs', 0)):
                torch.save(self.model.state_dict(),
                           os.path.join(path, 'epoch_%03d.pth' % (epoch + 1)))
            score = vali_mse if monitor == 'mse' else vali_mae
            early_stopping(score, self.model, path)
            if ema_state is not None:
                for n, p in self.model.named_parameters():
                    if n.startswith('cdp.'):
                        ema_state[n].mul_(ema_decay).add_(p.detach(), alpha=1 - ema_decay)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path, map_location='cpu'))
        if ema_state is not None:
            sd = self.model.state_dict()
            sd.update(ema_state)
            self.model.load_state_dict(sd)
            torch.save(self.model.state_dict(), best_model_path)
            print('[CDP] EMA weights applied and written back to checkpoint')
        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model=================================')
            self.model.load_state_dict(torch.load(
                os.path.join('./checkpoints/' + setting, 'checkpoint.pth'),
                map_location='cpu'))
        # 流式累计指标（避免拼接大数组：Traffic-720 需要 ~13GB 会 OOM）
        n_elem = 0
        sum_abs = 0.0
        sum_sq = 0.0
        sum_mape = 0.0
        sum_mspe = 0.0
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                if 'PEMS' == self.args.data or 'Solar' == self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                dec_inp = None
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs[:, -self.args.pred_len:, f_dim:]
                true = batch_y[:, -self.args.pred_len:, f_dim:]
                # 累计误差统计量（不保留完整数组）
                diff = pred - true
                n_elem += int(diff.size)
                sum_abs += float(np.abs(diff).sum())
                sum_sq += float((diff ** 2).sum())
                with np.errstate(divide='ignore', invalid='ignore'):
                    ratio = np.abs(diff / true)
                    sum_mape += float(np.where(ratio > 5, 0, ratio).sum())
                    sum_mspe += float((ratio ** 2).sum())
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        mae = sum_abs / n_elem
        mse = sum_sq / n_elem
        rmse = float(np.sqrt(mse))
        mape = sum_mape / n_elem
        mspe = sum_mspe / n_elem
        print('[CDP RESULT]', setting)
        print('mse:{}, mae:{}'.format(mse, mae))
        with open('./result_cdp.txt', 'a') as f:
            f.write(setting + "  mse:{} mae:{}\n".format(mse, mae))
        return
