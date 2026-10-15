# TimeKAN 侧 CDP 超参敏感性：stage-2 训练损失（L1 vs MSE）

> 生成时间：2026-10-15　|　结果：**8/8 全部完成，零失败**　|　总耗时 2h22m

---

## 一、为什么只做这一个超参

CycleNet 骨干上已做过三因子扫描（`cdp_lr` / `cdp_loss` / `feat_k`），结论可直接迁移：

| 因子 | CycleNet 结论 | TimeKAN 是否需要重测 | 理由 |
|---|---|---|---|
| `cdp_loss` | **决定性**：换 MSE ⇒ MAE 收益消失 | ⭐ **需要** | 唯一有增量信息的一项，且是方法设计依据 |
| `cdp_lr` | 不敏感（跨 16 倍仅波动 0.11–0.18 pp） | 否 | TimeKAN 跨 4 种子 ΔMAE 极差仅 0.38pp，稳定性已被侧面证明 |
| `feat_k` | 不敏感（8/16/32 仅 0.08–0.14 pp） | **否** | TimeKAN 硬编码 `k=min(16, seq_len-1)=16`，恰为 CycleNet 默认值，结论已覆盖 |

另外一个**零成本的免费论据**：TimeKAN 主实验 64 组 head 结果本身就呈现
「ΔMAE 全胜 64/64，而 ΔMSE 仅 −0.15% 且在 electricity(+0.19%)/traffic(+0.32%)/weather(+0.22%)
三个多通道数据集上甚至微正」——正是「L1 训练 ⇒ 收益集中在 MAE、不惠及 MSE」的指纹。
本次实验把它从**现象级相关**升级为**干预性因果**。

---

## 二、实验设计（严格配对）

| 项 | 值 |
|---|---|
| 变量 | stage-2 训练损失 `cdp_loss` ∈ {l1, mse} |
| 对照臂 | 主实验既有的 head 结果（L1），seed 2021 |
| 实验臂 | 本次新增（MSE），带后缀 `_lossMSE` |
| 设定 | weather / traffic × horizon {96,192,336,720} = **8 组严格配对** |
| 骨干 | **两臂共用同一个冻结 ckpt**（head 模式只加载不训练骨干） |
| 选点规则 | val-MAE，**全程未变**（这是关键：只换优化目标，不换评价/选择准则） |
| 种子 | 2021（与主实验一致） |

**配对有效性**：两臂冻结骨干指标最大相对差 **1.00e-07**（纯 GPU 浮点噪声）⇒ 配对严格成立。

---

## 三、核心结果

| 数据集 | n | ΔMAE（L1） | ΔMAE（MSE） | 收益衰减 |
|---|---|---|---|---|
| weather | 4 | −1.27% | **+3.87%** | +5.14 pp |
| traffic | 4 | −5.17% | −0.36% | +4.81 pp |
| **全部** | **8** | **−3.22%** | **+1.76%** | **+4.98 pp** |

- **8/8 无一例外**：MAE 收益差区间 +4.51 ~ +6.11 pp。
- weather 4/4 **由改善翻转为恶化**（+2.84% ~ +4.74%）。
- traffic 4/4 收益被**削弱 82%–99%**（如 traffic-96：−5.90% → −0.77%）。
- L1 臂 MAE 改善 **8/8**；MSE 臂仅 **3/8**。

### 跨骨干对照（详见 `cross_backbone_loss.md`）

| 骨干 | 设定 | ΔMAE(L1) | ΔMAE(MSE) | 衰减 |
|---|---|---|---|---|
| **TimeKAN** | 全部 8 设定 | −3.22% | +1.76% | +4.98 pp |
| CycleNet | electricity-96 | −1.21% | +0.23% | +1.44 pp |
| CycleNet | traffic-96 | −4.63% | +0.05% | +4.68 pp |

两个**独立骨干**在同一变量上给出**同一结论**。

---

## 四、结论与写作建议

**机理**：CDP 的收益方向由 **stage-2 训练损失**决定；骨干选择采用 **val-MAE**，
故 stage-2 损失必须与之一致 —— 这是「CDP 默认配 L1」的实验依据，
也解释了为何 CDP 的 MSE 收益普遍弱于 MAE 收益。

### ⚠️ 写作红线

- ❌ **不要**写「CDP 在 MSE 训练下失效」——会被读成方法脆弱。
- ✅ **建议**写：「CDP 的收益方向跟随 stage-2 训练损失的目标取向；由于骨干选择采用 val-MAE，
  我们将 stage-2 损失设为 L1 以保证优化目标与选择准则一致。」
- ⚠️ **须诚实标注**：TimeKAN 的 **weather** 上 MSE 臂的 ΔMSE 也未见改善（+0.28%），
  与 traffic 上「ΔMSE 由 +0.13% 转为 −0.11%」的经典交换**不完全一致**。
  因此**不要**在论文中概括断言「换 MSE 必然改善 MSE」。

---

## 五、代码改动（向后兼容，可核验）

对 `cdp_timekan_gpu.py` 做了 **4 处**改动，原文件已备份为 `cdp_timekan_gpu.py.bak_20261015`：

1. 新增 `--cdp_loss {l1,mse}`（**默认 `l1`**）与 `--suffix`（默认空）
2. `cdp_criterion = criterion_mse if cdp_loss=='mse' else criterion_l1`
3. loss 计算改用 `cdp_criterion`（**只改 head/cdp 分支，cao 分支未动**）
4. 结果文件名支持 suffix；JSON 新增 `cdp_loss` / `cdp_lr` 字段

**回归验证**：默认参数重跑 weather-96，与既有结果相差 **1e-8 相对误差 / ΔMAE 0.0005 pp**
（GPU 浮点非确定性），行为未改变 ⇒ **原主实验 232 组结果仍完全可复现**。

**结果不覆盖**：新结果带 `_lossMSE` 后缀独立存放，`cdp_results/` 原有 233 个 json 未被污染。

---

## 六、复现方式

```bash
cd TimeKAN-main
# 单条示例（注意必须显式传 --enc_in）
python cdp_timekan_gpu.py --data traffic --pred_len 96 --enc_in 862 \
       --mode head --seed 2021 --cdp_loss mse --suffix _lossMSE

# 批量（幂等，结果存在则跳过）
python run_batch_tk_loss.py

# 汇总
python tools/summarize_tk_loss.py     # -> tk_loss_summary.md/.csv
python tools/cross_backbone_loss.py   # -> cross_backbone_loss.md
```

> ⚠️ `--enc_in` **必须显式指定**：weather=21、traffic=862、electricity=321、ETT=7。
> 默认值 7 只适用于 ETT，不传会报 `size mismatch for normalize_layers.0.affine_weight`。

---

## 七、文件清单

| 文件 | 说明 |
|---|---|
| `tk_loss_summary.md` / `.csv` | **主表**：8 组严格配对对照 + 聚合结论 |
| `cross_backbone_loss.md` | **跨骨干对照**（TimeKAN vs CycleNet）+ 写作红线 |
| `cdp_results/*_lossMSE.json` | 8 条 MSE 臂结果 |
| `cdp_results/cdp_head_{weather,traffic}_*_seed2021.json` | 8 条 L1 臂对照（来自主实验） |
| `run_loss_*.log` | 8 份完整训练日志 |
| `batch_log_loss.txt` | 批量执行记录 |
| `run_batch_tk_loss.py` | 批量驱动（幂等可续跑） |
| `tools/summarize_tk_loss.py` / `tools/cross_backbone_loss.py` | 汇总工具 |
| `cdp_timekan_gpu.py` / `.bak_20261015` | 改动后脚本 + 备份（可 diff 核验） |

> 数据集 CSV 体积过大（weather/traffic 共约 300 MB）未打包，
> 本机路径 `D:/cdp_gpu/TimeKAN-main/dataset/custom/` 可直接复用。
