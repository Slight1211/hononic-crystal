param(
    [string]$PreprocessedRoot = "E:\datasets_graph_cache_novel109_top_batches001_600_20260512\source_000",
    [string]$OutputDir = "C:\Users\Sligh\Desktop\Npw\gap_regressor_cv_runs",
    [int]$TargetGapIndex = 3,
    [double]$SampleFraction = 0.20,
    [int]$Folds = 5,
    [int]$Epochs = 6,
    [int]$BatchSize = 256,
    [int]$NumWorkers = 4,
    [string]$GraphFeatureMode = "novel105_phase_edge_shell",
    [string]$ConfigsJson = ""
)

$ErrorActionPreference = "Stop"
$env:PYTHONUNBUFFERED = "1"

$ProjectRoot = "C:\Users\Sligh\Desktop\Npw"
$Python = Join-Path $ProjectRoot ".venv-gnn-cu128\Scripts\python.exe"
$LogDir = Join-Path $ProjectRoot "overnight_training_logs\gap$($TargetGapIndex)_regressor_cv_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
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
    Write-Log "Gap$TargetGapIndex lower/upper regressor CV started."
    Write-Log "PreprocessedRoot=$PreprocessedRoot"
    Write-Log "OutputDir=$OutputDir"
    Write-Log "TargetGapIndex=$TargetGapIndex SampleFraction=$SampleFraction Folds=$Folds Epochs=$Epochs BatchSize=$BatchSize NumWorkers=$NumWorkers"
    if ($ConfigsJson -ne "") {
        Write-Log "ConfigsJson=$ConfigsJson"
    }

    if (-not (Test-Path $Python)) {
        throw "Python environment not found: $Python"
    }
    if (-not (Test-Path $PreprocessedRoot)) {
        throw "Preprocessed root not found: $PreprocessedRoot"
    }

    $cvArgs = @(
        (Join-Path $ProjectRoot "train_gap_regressor_cv.py"),
        "--preprocessed-root", $PreprocessedRoot,
        "--output-dir", $OutputDir,
        "--target-gap-index", "$TargetGapIndex",
        "--sample-fraction", "$SampleFraction",
        "--folds", "$Folds",
        "--epochs", "$Epochs",
        "--batch-size", "$BatchSize",
        "--num-workers", "$NumWorkers",
        "--graph-feature-mode", $GraphFeatureMode,
        "--amp-mode", "bf16",
        "--device", "auto",
        "--best-metric", "mean_bound_mae_khz"
    )
    if ($ConfigsJson -ne "") {
        if (-not (Test-Path $ConfigsJson)) {
            throw "Configs JSON not found: $ConfigsJson"
        }
        $cvArgs += @("--configs-json", $ConfigsJson)
    }

    "RUNNING: gap$TargetGapIndex lower/upper regressor CV" | Set-Content -Path $StatusPath -Encoding UTF8
    & $Python @cvArgs 2>&1 | Tee-Object -FilePath $TrainLog
    if ($LASTEXITCODE -ne 0) {
        throw "Gap$TargetGapIndex regressor CV failed with exit code $LASTEXITCODE"
    }

    "DONE" | Set-Content -Path $StatusPath -Encoding UTF8
    Write-Log "Gap$TargetGapIndex regressor CV finished successfully."
}
catch {
    "FAILED: $($_.Exception.Message)" | Set-Content -Path $StatusPath -Encoding UTF8
    Write-Log "FAILED: $($_.Exception.Message)"
    exit 1
}
