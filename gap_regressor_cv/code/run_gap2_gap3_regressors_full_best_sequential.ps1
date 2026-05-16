param(
    [string]$PreprocessedRoot = "E:\datasets_graph_cache_novel109_top_batches001_600_20260512\source_000",
    [string]$ProjectRoot = "C:\Users\Sligh\Desktop\Npw",
    [int]$RegressorEpochs = 20,
    [int]$BatchSize = 256
)

$ErrorActionPreference = "Stop"

$Runner = Join-Path $ProjectRoot "run_gap_regressors_only_parallel.ps1"
if (-not (Test-Path $Runner)) {
    throw "Runner not found: $Runner"
}
if (-not (Test-Path $PreprocessedRoot)) {
    throw "Preprocessed root not found: $PreprocessedRoot"
}

$env:PYTHONUNBUFFERED = "1"

Write-Host "[sequential] Starting gap2 regressor with baseline best full-fold settings..."
$env:OMP_NUM_THREADS = "3"
$env:MKL_NUM_THREADS = "3"
$env:NUMEXPR_NUM_THREADS = "3"
$env:OPENBLAS_NUM_THREADS = "3"
& $Runner `
    -PreprocessedRoot $PreprocessedRoot `
    -OutputDir (Join-Path $ProjectRoot "gap2_regressor_full_best_runs") `
    -RegressorEpochs $RegressorEpochs `
    -BatchSize $BatchSize `
    -HiddenDim 224 `
    -Layers 8 `
    -Dropout 0.10 `
    -LearningRate 0.0003 `
    -WeightDecay 0.00001 `
    -NumWorkers 3 `
    -RegressorNames "gap2_expert"

if ($LASTEXITCODE -ne 0) {
    throw "gap2 regressor failed with exit code $LASTEXITCODE"
}

Write-Host "[sequential] Starting gap3 regressor with CV-selected settings..."
$env:OMP_NUM_THREADS = "4"
$env:MKL_NUM_THREADS = "4"
$env:NUMEXPR_NUM_THREADS = "4"
$env:OPENBLAS_NUM_THREADS = "4"
& $Runner `
    -PreprocessedRoot $PreprocessedRoot `
    -OutputDir (Join-Path $ProjectRoot "gap3_regressor_full_best_runs") `
    -RegressorEpochs $RegressorEpochs `
    -BatchSize $BatchSize `
    -HiddenDim 224 `
    -Layers 8 `
    -Dropout 0.10 `
    -LearningRate 0.0005 `
    -WeightDecay 0.00001 `
    -NumWorkers 4 `
    -RegressorNames "gap3_expert"

if ($LASTEXITCODE -ne 0) {
    throw "gap3 regressor failed with exit code $LASTEXITCODE"
}

Write-Host "[sequential] gap2 and gap3 regressor training completed."
