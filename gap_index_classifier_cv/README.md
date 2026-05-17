# Gap-Index Classifier Cross-Validation

This directory contains the code and configuration files used to tune the four-class gap-index classifier for the square-bounded PWE dataset.

## Task

The classifier predicts one of four classes:

- `no_gap`
- `gap2`
- `gap3`
- `other`

The regression models for gap-edge prediction are trained separately and are not part of this cross-validation run.

## Data

The full dataset and graph cache are not committed to this repository because they are large local artifacts:

- `E:\datasets`
- `E:\datasets_graph_cache_novel109_top_batches001_600_20260512\source_000`

The CV scripts expect the preprocessed graph cache path to be available locally.

## Files

- `code/train_gap_index_classifier_cv.py`: stratified k-fold cross-validation driver.
- `code/run_gap_index_classifier_cv.ps1`: PowerShell launcher.
- `code/run_gap_index_classifier_improved.ps1`: full-data training launcher for the selected classifier configuration.
- `code/train_gap_index_classifier_improved.py`: four-class indexing, focal loss, weighting, and evaluation helpers.
- `code/train_gap_index_then_regress.py`, `code/train_first_band_gap_multitask.py`, `code/train_two_stage_gap_gnn.py`: model and data-loading dependencies.
- `configs/*.json`: candidate hyperparameter configurations.
- `results/*.json`: saved CV summaries.
- `results/cv_scores_summary.md`: compact score table.
- `results/full_classifier_20260515_212410/`: final full-data classifier metadata, epoch history, and test summary.
- `figures/ml_classifier_current_results.*`: validation history and test confusion-matrix figure used in the manuscript.

## Recommended Configuration

The best 20% subset, stratified 5-fold CV result was obtained with:

```text
name = gamma100_sampler025
hidden_dim = 224
layers = 8
dropout = 0.08
lr = 3e-4
weight_decay = 1e-5
class_weight_scheme = sqrt_inverse
sampler = weighted
sampler_power = 0.25
focal_gamma = 1.0
label_smoothing = 0.01
```

The final model should be retrained on the full dataset using this selected configuration and evaluated on an independent stratified test split.

## Full-Data Classifier Training

The final classifier was trained on all `1,746,000` unique geometries with an `80/10/10` train--validation--test split.

```text
epochs = 20
batch_size = 320
hidden_dim = 224
layers = 8
dropout = 0.08
lr = 3e-4
weight_decay = 1e-5
class_weight_scheme = sqrt_inverse
sampler = weighted
sampler_power = 0.25
focal_gamma = 1.0
label_smoothing = 0.01
best_metric = macro_f1
```

Final held-out test metrics:

```text
macro_f1 = 0.854428
balanced_accuracy = 0.893322
accuracy = 0.977234
```

The selected epoch was epoch 19. Full metadata and the epoch-by-epoch training history are stored in `results/full_classifier_20260515_212410/`.

## Example

From the project root:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\gap_index_classifier_cv\code\run_gap_index_classifier_cv.ps1 `
  -PreprocessedRoot "E:\datasets_graph_cache_novel109_top_batches001_600_20260512\source_000" `
  -SampleFraction 0.20 `
  -Folds 5 `
  -Epochs 5 `
  -BatchSize 160 `
  -NumWorkers 3 `
  -ConfigsJson ".\gap_index_classifier_cv\configs\gap_index_classifier_cv_refined_A_configs.json"
```

Final full-data training can be launched with:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\gap_index_classifier_cv\code\run_gap_index_classifier_improved.ps1 `
  -PreprocessedRoot "E:\datasets_graph_cache_novel109_top_batches001_600_20260512\source_000" `
  -OutputDir ".\gap_index_classifier_full_60pct_runs" `
  -Epochs 20 `
  -BatchSize 320 `
  -NumWorkers 3 `
  -Dropout 0.08 `
  -SamplerPower 0.25 `
  -FocalGamma 1.0 `
  -LabelSmoothing 0.01
```
