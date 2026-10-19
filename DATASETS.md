# Dataset Details (archived from the paper main text, page-budget cut 2026-10-19)

Evaluation benchmarks used in the paper (lookback 96; horizons {96, 192, 336, 720}; chronological train/val/test splits). *Forecastability: one minus the entropy of the Fourier decomposition (Goerg, 2012); higher values imply better predictability.

| Tasks | Dataset | Dim | Prediction Lengths | Dataset Size (train, val, test) | Frequency | Forecastability* |
|---|---|---|---|---|---|---|
| Forecast | ETTm1 | 7 | {96, 192, 336, 720} | (34465, 11521, 11521) | 15min | 0.46 |
| Forecast | ETTm2 | 7 | {96, 192, 336, 720} | (34465, 11521, 11521) | 15min | 0.55 |
| Forecast | ETTh1 | 7 | {96, 192, 336, 720} | (8545, 2881, 2881) | Hourly | 0.38 |
| Forecast | ETTh2 | 7 | {96, 192, 336, 720} | (8545, 2881, 2881) | Hourly | 0.45 |
| Forecast | Electricity | 321 | {96, 192, 336, 720} | (18317, 2633, 5261) | Hourly | 0.77 |
| Forecast | Traffic | 862 | {96, 192, 336, 720} | (12185, 1757, 3509) | Hourly | 0.68 |
| Forecast | Weather | 21 | {96, 192, 336, 720} | (36792, 5271, 10540) | 10 min | 0.75 |

Note: archived from the DASFAA 2027 submission's Sec. "Experiments — Dataset" to meet the 16-page limit; referenced from the paper as "full statistics are provided in the repository".
