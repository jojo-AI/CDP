# Plug-in transfer on the ETT benchmarks (per-setting deltas)

This table corresponds to `tab:multibackbone` in an earlier draft of the paper.
The absolute per-setting MSE/MAE values for every backbone and `backbone + CDP`
are in Table `tab:main_perhorizon` of the paper; this file adds the per-setting
relative deltas with per-seed dispersion.

Relative change (ΔMSE% / ΔMAE%, **negative = improvement**) over pre-fixed seeds,
relative to each backbone's reproduced baseline under the identical protocol
(seeds 2021–2024; TimeKAN on the ETT family: single pre-fixed seed 2021, no sd).

| Setting    | TimeKAN+CDP ΔMSE% | TimeKAN+CDP ΔMAE% | CycleNet+CDP ΔMSE% | CycleNet+CDP ΔMAE% | iTransformer+CDP ΔMSE% | iTransformer+CDP ΔMAE% |
|------------|-------------------|-------------------|--------------------|--------------------|------------------------|------------------------|
| ETTh1-96   | −0.11 | −1.92 | −1.46±0.93 | −2.33±0.82 | +0.04±0.40 | −0.93±0.30 |
| ETTh1-192  | −0.37 | −1.94 | −0.18±1.14 | −1.33±0.65 | −0.53±0.15 | −1.34±0.05 |
| ETTh1-336  | −0.58 | −1.63 | +0.51±1.80 | −1.07±0.89 | +0.85±1.26 | −0.45±0.78 |
| ETTh1-720  | +0.29 | −1.66 | +1.71±1.02 | −0.41±0.63 | +4.05±1.33 | +0.65±0.93 |
| ETTh2-96   | −2.32 | −2.87 | −1.45±0.82 | −1.63±0.44 | −2.61±1.17 | −2.62±0.62 |
| ETTh2-192  | −1.97 | −1.92 | −2.46±0.65 | −2.08±0.15 | −2.21±0.39 | −2.31±0.10 |
| ETTh2-336  | −3.53 | −3.11 | −2.37±0.86 | −2.53±0.36 | +0.55±0.66 | −1.19±0.61 |
| ETTh2-720  | −6.39 | −4.31 | −3.28±1.53 | −2.56±0.81 | +1.60±2.83 | −0.46±1.27 |
| ETTm1-96   | −2.22 | −2.37 | −2.02±0.59 | −4.12±0.38 | −1.42±0.57 | −2.45±0.40 |
| ETTm1-192  | −0.81 | −1.71 | −0.13±0.30 | −3.14±0.17 | +0.48±0.55 | −1.90±0.54 |
| ETTm1-336  | −0.96 | −1.85 | −0.00±0.40 | −2.77±0.16 | +0.61±1.08 | −1.38±0.63 |
| ETTm1-720  | −1.37 | −1.99 | +1.24±0.13 | −1.67±0.21 | +1.46±1.18 | −1.06±0.71 |
| ETTm2-96   | −1.19 | −1.68 | −1.68±0.65 | −1.62±0.31 | −0.95±0.22 | −2.02±0.25 |
| ETTm2-192  | −0.03 | −1.09 | −1.46±0.38 | −1.32±0.23 | −2.24±0.71 | −2.51±0.32 |
| ETTm2-336  | −1.33 | −1.89 | −1.34±0.15 | −1.39±0.21 | −1.52±0.57 | −1.94±0.37 |
| ETTm2-720  | +1.51 | −0.08 | −1.13±0.21 | −1.13±0.22 | −0.84±0.70 | −1.52±0.50 |

Summary (paper, "Four observations"): on the ETT family, CycleNet+CDP improves
MAE in 16/16 settings (mean −1.94%) and MSE in 13/16; TimeKAN+CDP improves MAE
in 16/16 (mean −2.00%) and MSE in 14/16, largest single gain −6.39% MSE /
−4.31% MAE on ETTh2-720; iTransformer+CDP improves MAE in 15/16 (58/64
seed-level runs, mean −1.46%). Pooled significance over 68 settings / 272
paired runs: MAE improves in 265/272 (97.4%, mean −2.19%, p < 1e-62).

Multi-seed evidence on the large multivariate benchmarks is in the paper's
`tab:itransformer_big`.
