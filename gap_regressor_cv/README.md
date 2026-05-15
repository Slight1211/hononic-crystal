# Gap-Edge Regressor Cross-Validation

This directory contains the code, configuration files, and saved results used to tune branch-specific lower/upper band-edge regressors for the square-bounded PWE dataset.

## Task

The regressor is trained after the gap-index classification stage. For a selected branch, it predicts the lower and upper complete band-gap edge frequencies:

```text
[f_low, f_up]
```

The current completed CV run is for `target_gap_index = 3`.

## Data

The full dataset and graph cache are not committed to this repository because they are large local artifacts:

- `E:\datasets`
- `E:\datasets_graph_cache_novel109_top_batches001_600_20260512\source_000`

The completed gap3 CV run used:

```text
available gap3 samples = 1,596,663
sample fraction = 0.20
selected samples = 319,333
folds = 5
epochs per fold = 5
batch size = 256
```

## Files

- `code/train_gap_regressor_cv.py`: k-fold CV driver for gap-index-specific lower/upper band-edge regression.
- `code/run_gap_regressor_cv.ps1`: PowerShell launcher.
- `code/train_gap_index_then_regress.py`, `code/train_first_band_gap_multitask.py`, `code/train_two_stage_gap_gnn.py`, `code/node_pwe_repro.py`: model and data-loading dependencies.
- `configs/gap3_regressor_cv_configs.json`: 12 candidate hyperparameter configurations.
- `results/gap3_cv_summary_20260515_040724.json`: full saved CV summary for the 20% gap3 run.
- `results/gap3_cv_scores_summary.md`: compact score table.

## Selection Metric

The primary selection metric is the mean MAE of the lower and upper gap-edge frequencies:

```text
mean_bound_mae_khz = (lower_mae_khz + upper_mae_khz) / 2
```

Width MAE is reported as a derived metric from the predicted lower and upper edges.

## Recommended Gap3 Configuration

The best 20% subset, five-fold CV result was obtained with:

```text
name = lr0005
hidden_dim = 224
layers = 8
dropout = 0.10
lr = 5e-4
weight_decay = 1e-5
smooth_l1_beta = 1.0
```

Mean CV metrics:

```text
mean_bound_mae_khz = 0.164867
lower_mae_khz = 0.078144
upper_mae_khz = 0.251590
width_mae_khz = 0.260491
lower_r2 = 0.995967
upper_r2 = 0.997511
```

## Example

From the original project root:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\gap_regressor_cv\code\run_gap_regressor_cv.ps1 `
  -PreprocessedRoot "E:\datasets_graph_cache_novel109_top_batches001_600_20260512\source_000" `
  -TargetGapIndex 3 `
  -OutputDir ".\gap3_regressor_cv_runs" `
  -SampleFraction 0.20 `
  -Folds 5 `
  -Epochs 5 `
  -BatchSize 256 `
  -NumWorkers 4 `
  -ConfigsJson ".\gap_regressor_cv\configs\gap3_regressor_cv_configs.json"
```
