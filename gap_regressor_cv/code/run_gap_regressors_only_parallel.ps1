param(
    [string]$PreprocessedRoot = "E:\datasets_graph_cache_novel109_top_batches001_600_20260512\source_000",
    [string]$OutputDir = "C:\Users\Sligh\Desktop\Npw\gnn_gap_regressors_parallel_runs",
    [int]$RegressorEpochs = 20,
    [int]$BatchSize = 96,
    [int]$HiddenDim = 224,
    [int]$Layers = 8,
    [double]$Dropout = 0.10,
    [double]$LearningRate = 3e-4,
    [double]$WeightDecay = 1e-5,
    [int]$NumWorkers = 0,
    [string]$GraphFeatureMode = "novel105_phase_edge_shell",
    [string]$RegressorNames = ""
)

$ErrorActionPreference = "Stop"
$env:PYTHONUNBUFFERED = "1"

$ProjectRoot = "C:\Users\Sligh\Desktop\Npw"
$Python = Join-Path $ProjectRoot ".venv-gnn-cu128\Scripts\python.exe"
$LogDir = Join-Path $ProjectRoot "overnight_training_logs\gap_regressors_parallel_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$MasterLog = Join-Path $LogDir "master.log"
$TrainLog = Join-Path $LogDir "train.log"
$StatusPath = Join-Path $LogDir "STATUS.txt"

function Write-Log {
    param([string]$Message)
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    $line | Tee-Object -FilePath $MasterLog -Append
}

try {
    Write-Log "Parallel regressor-only job started."
    Write-Log "PreprocessedRoot=$PreprocessedRoot"
    Write-Log "OutputDir=$OutputDir"
    Write-Log "RegressorEpochs=$RegressorEpochs BatchSize=$BatchSize"
    Write-Log "NumWorkers=$NumWorkers RegressorNames=$RegressorNames"

    if (-not (Test-Path $Python)) {
        throw "Python environment not found: $Python"
    }
    if (-not (Test-Path $PreprocessedRoot)) {
        throw "Preprocessed root not found: $PreprocessedRoot"
    }

    "RUNNING: regressor-only parallel" | Set-Content -Path $StatusPath -Encoding UTF8
    $TrainArgs = @(
        (Join-Path $ProjectRoot "train_gap_regressors_only.py"),
        "--preprocessed-root", $PreprocessedRoot,
        "--output-dir", $OutputDir,
        "--max-batches", "0",
        "--max-samples", "0",
        "--min-sample-index", "0",
        "--max-sample-index", "0",
        "--regressor-epochs", "$RegressorEpochs",
        "--batch-size", "$BatchSize",
        "--hidden-dim", "$HiddenDim",
        "--layers", "$Layers",
        "--dropout", "$Dropout",
        "--lr", "$LearningRate",
        "--weight-decay", "$WeightDecay",
        "--num-workers", "$NumWorkers",
        "--graph-feature-mode", $GraphFeatureMode,
        "--warmup-fraction", "0.08",
        "--min-lr-scale", "0.10",
        "--amp-mode", "bf16",
        "--device", "auto",
        "--min-group-train-samples", "200",
        "--major-gap-indices", "2", "3"
    )
    if ($RegressorNames.Trim().Length -gt 0) {
        $TrainArgs += "--regressor-filter"
        $TrainArgs += ($RegressorNames -split "," | ForEach-Object { $_.Trim() } | Where-Object { $_.Length -gt 0 })
    }

    & $Python @TrainArgs 2>&1 | Tee-Object -FilePath $TrainLog
    if ($LASTEXITCODE -ne 0) {
        throw "Regressor-only training failed with exit code $LASTEXITCODE"
    }

    "DONE" | Set-Content -Path $StatusPath -Encoding UTF8
    Write-Log "Parallel regressor-only job finished successfully."
}
catch {
    "FAILED: $($_.Exception.Message)" | Set-Content -Path $StatusPath -Encoding UTF8
    Write-Log "FAILED: $($_.Exception.Message)"
    exit 1
}
