# Multi-backbone CDP adapters (TimeKAN / CycleNet / iTransformer)

The main codebase at the repository root implements CDP on the primary
backbone (TimeMixer++). The generalization experiments in the paper
(Sec. "Generalization to Additional Backbones") apply the same calibration
stack to three additional backbones. Instead of forking each codebase, we
drop a self-contained adapter script into each **official upstream
repository** and drive the runs with the batch runners in this folder.

## Files

| File | Purpose |
|---|---|
| `cdp_timekan_cpu.py` | CDP adapter for the [official TimeKAN repo](https://github.com/EmorZz1G/TimeKAN) (imports `models.TimeKAN`). CPU/GPU auto. |
| `cdp_cyclenet_cpu.py` | CDP adapter for the [official CycleNet repo](https://github.com/ACAT-SCUT/CycleNet) (imports `models.CycleNet`). Protocol mirrors the NeurIPS 2024 official scripts (Linear-Input-96). |
| `run_batch.py` | TimeKAN overnight batch runner (2 workers x 4 torch threads, 8-core CPU box). |
| `run_batch_extra.py` | CycleNet + iTransformer batch runner (drives `cdp_cyclenet_cpu.py` / `cdp_itransformer_cpu.py` inside `CycleNet-main/` and `iTransformer-main/`). |
| `start_extra_backbones.bat` | Windows launcher for `run_batch_extra.py`. |

## Reproduction protocol

1. Clone the three official upstream repositories next to this folder:
   `TimeKAN-main/`, `CycleNet-main/`, `iTransformer-main/`.
2. Copy the corresponding adapter script into the repository root, e.g.
   `cp cdp_timekan_cpu.py TimeKAN-main/`.
3. Run one adapter per (dataset, horizon, seed, mode) pair, or use the batch
   runners which add skip-if-exists resumption, lock/STOP files, and phased
   scheduling (Phase 1: baseline + head for seeds 2021-2024; Phase 2/3:
   cao/cdp ablation).
4. The protocol mirrors each backbone's official scripts (see the header of
   each adapter for the exact hyperparameters: epochs, early stopping,
   model-selection metric, learning rate, cycle length, etc.).

## Known gaps

- `cdp_itransformer_cpu.py` (iTransformer adapter) and `run_batch_gpu.py`
  (GPU runner for iTransformer, RTX 3060) executed on the experiment server
  and are not yet in this snapshot; they will be added after recovery.
  The iTransformer run log (280/280 runs, 0 unresolved failures) is archived
  with the paper's experiment results.
