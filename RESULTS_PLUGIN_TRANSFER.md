# Plug-in Transfer of the CDP Calibration Stack (frozen-backbone protocol)

Full version of the plug-in transfer table removed from the paper for space
(referenced as "full table in the repository" in the Ablation subsection).

**Protocol note.** These are *frozen-backbone plug-in* runs: the host backbone is
used as released / retrained under its own protocol, and only CDP's calibration
stack is attached head-only. The numbers are therefore **NOT comparable** with
the two-stage protocol-controlled comparison (`tab:main_perhorizon` in the
paper); a calibrated model's absolute accuracy follows its host.

Relative change Δ (%) over the host backbone; positive = improvement.

| Host | Metric | Solar | Traffic | ETTh1 | ETTm2 |
|---|---|---|---|---|---|
| TimeMixer | MSE | +1.38 | +2.30 | +1.34 | +3.38 |
| TimeMixer | MAE | +1.82 | +0.94 | +3.95 | −1.22 |
| TimeKAN | MSE | +3.13 | +4.48 | +1.24 | +1.96 |
| TimeKAN | MAE | +3.95 | −1.75 | +0.47 | −2.36 |
| PatchTST | MSE | +1.77 | +0.95 | +1.18 | +1.40 |
| PatchTST | MAE | +0.91 | +2.10 | +4.54 | +1.21 |
| Crossformer | MSE | +4.37 | +4.38 | +6.22 | +3.70 |
| Crossformer | MAE | +5.74 | +6.77 | +1.36 | −0.49 |

Absolute values (backbone / +CDP):

| Host | Solar MSE / MAE | Traffic MSE / MAE | ETTh1 MSE / MAE | ETTm2 MSE / MAE |
|---|---|---|---|---|
| TimeMixer | 0.220→0.217 / 0.280→0.275 | 0.490→0.479 / 0.300→0.297 | 0.453→0.447 / 0.447→0.430 | 0.275→0.266 / 0.323→0.327 |
| TimeKAN | 0.198→0.192 / 0.263→0.253 | 0.420→0.402 / 0.286→0.291 | 0.409→0.404 / 0.427→0.425 | 0.260→0.255 / 0.318→0.326 |
| PatchTST | 0.287→0.282 / 0.333→0.330 | 0.529→0.524 / 0.341→0.334 | 0.516→0.510 / 0.484→0.463 | 0.290→0.286 / 0.334→0.330 |
| Crossformer | 0.406→0.389 / 0.442→0.418 | 0.667→0.639 / 0.426→0.399 | 0.529→0.498 / 0.522→0.515 | 0.757→0.730 / 0.610→0.613 |

**Summary.** MSE improves in all 16 host×dataset pairs; MAE improves in 12 of 16
and never degrades by more than 2.4%.
