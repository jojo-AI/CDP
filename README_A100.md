# CDP M/K 扫描 · A100 一键包（20261015）

## 这是什么
为论文敏感性图补真实 M/K 扫描数据的自包含实验包：TimeMixer++ CDP 两阶段协议，
ETTh1-96、seed 2021，与归档主实验逐字段同配方。数据集(ETTh1.csv)与全部代码已内置，
**解压 → `bash run_all.sh` → 结束后在启动目录拿 `回传_MK扫描结果_*.zip`**。

## 两种模式（自动检测，无需配置）
| 模式 | 触发条件 | 行为 |
|---|---|---|
| **严格模式** | `repo/checkpoints/` 下存在归档骨干 `long_term_forecast_ETTh1_st1_96_dynamic_l10.0_*` | M 扫描 {0.6,0.8,1.2,1.5} 复用归档骨干只跑 Stage-2；M r=1.0 与 K=4 直接引用归档锚点；K 扫描 {2,3,5,6} 正常重训。结果与主实验**严格可比** |
| **自足模式**（默认，包内无 ckpt） | 上面的目录为空 | 先重训一次默认骨干（K=4 任务内），M 扫描 5 点（含本地锚点 r=1.0）全部挂在该骨干上，曲线内部自洽；K 扫描 {2,3,5,6}。结果与归档 winner 数字不直接可比，但敏感性形状有效 |

如果你能从旧 GPU 服务器把归档骨干整目录 scp 进 `repo/checkpoints/`，包会自动切换
严格模式，**强烈推荐**（一条 scp 的事）：`scp -r 旧服务器:D:\fsdownload\TimeMixer\checkpoints\long_term_forecast_ETTh1_st1_96_dynamic_l10.0_*  repo/checkpoints/`

## 运行
```bash
unzip CDP_MK_A100_20261015.zip
cd CDP_MK_A100_20261015
CUDA_VISIBLE_DEVICES=0 bash run_all.sh        # 选卡可省略
```
- python 自动选择：环境变量 `CDP_PY` > `python3` > `python`，要求该解释器有 CUDA torch；
  都没有才建 venv 装 `requirements_a100.txt`（**不含 torch**，torch 请用集群自带）。
- 时长（A100 单卡）：严格模式 ≈ 2.5~3.5 h；自足模式 ≈ 3~4 h。
- 断点续跑：中断后重新 `bash run_all.sh`，已完成的点自动跳过。
- 并发保护：包根的 `.mk_batch.lock`；确认无并发后可删。

## 红线（沿用原任务包）
1. 只运行 `run_all.sh` / `mk_scan/run_batch_mk.py`，不得手改任何文件；
   `run_mk.py` 由 runner 自动生成（run.py 的 `--cdp_M` int→float 单点补丁）。
2. 严格模式下 M 扫描严禁重训 Stage-1——必须复用归档骨干 ckpt。
3. 不得改 seed(2021)、不改任何超参/协议/选点规则（Stage-2 = 归档 winner at_cal1 配方）。
4. 不得动 `repo/checkpoints/` 已有目录；不得删 JSON。
5. 某 run 失败：重跑 `bash run_all.sh`；连续 2 次同点失败 → 停，把
   `mk_logs/<对应>.log` 和 `mk_scan_console.log` 一起发回。

## 回传包内容
```
回传_MK扫描结果_YYYYMMDD_HHMM.zip
├── mk_results/    mk_M_r*.json, mk_K*.json, st1_K*.json（字段含 axis/seg_ratio|K/cdp_M/
│                  down_sampling_layers/mse/mae/stage1_mse/backbone_source/status/log）
├── mk_logs/       每个 run 的完整训练日志
└── mk_scan_console.log
```

## 目录结构
```
CDP_MK_A100_20261015/
├── run_all.sh                  # 一键入口
├── requirements_a100.txt       # 仅 venv 兜底用（无 torch）
├── repo/                       # TimeMixer++ CDP 完整代码（含 run.py/models/exp/...）
│   ├── dataset/ETT-small/ETTh1.csv
│   └── checkpoints/            # 可选：放归档骨干→严格模式
├── mk_scan/run_batch_mk.py     # 扫描 runner（双模式、断点续跑、自动打包）
├── mk_scan/summarize_mk.py     # 结果汇总表
└── archived_reference/         # 归档锚点 JSON（M r=1.0 / K=4，严格模式引用）
```
