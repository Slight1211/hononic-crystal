param(
    [string]$PreprocessedRoot = "E:\datasets_graph_cache_novel109_top_batches001_600_20260512\source_000",
    [string]$OutputDir = "",
    [double]$SampleFraction = 0.20,
    [int]$Folds = 5,
    [int]$Epochs = 5,
    [int]$BatchSize = 192,
    [int]$NumWorkers = 4,
    [string]$GraphFeatureMode = "novel105_phase_edge_shell",
    [string]$ConfigsJson = "",
    [string]$Python = ""
)

$ErrorActionPreference = "Stop"
$env:PYTHONUNBUFFERED = "1"

$CodeRoot = $PSScriptRoot
$CvRoot = Split-Path -Parent $CodeRoot
if ($OutputDir -eq "") {
    $OutputDir = Join-Path $CvRoot "runs"
}
if ($Python -eq "") {
    $candidate = Join-Path (Split-Path -Parent $CvRoot) ".venv-gnn-cu128\Scripts\python.exe"
    if (Test-Path $candidate) {
        $Python = $candidate
    }
    else {
        $Python = "python"
    }
}
$LogDir = Join-Path $CvRoot "logs\gap_index_classifier_cv_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
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
    Write-Log "Four-class gap-index classifier CV started."
    Write-Log "PreprocessedRoot=$PreprocessedRoot"
    Write-Log "OutputDir=$OutputDir"
    Write-Log "SampleFraction=$SampleFraction Folds=$Folds Epochs=$Epochs BatchSize=$BatchSize NumWorkers=$NumWorkers"
    if ($ConfigsJson -ne "") {
        Write-Log "ConfigsJson=$ConfigsJson"
    }

    if (-not (Test-Path $Python)) {
        throw "Python environment not found: $Python"
    }
    if (-not (Test-Path $PreprocessedRoot)) {
        throw "Preprocessed root not found: $PreprocessedRoot"
    }

    "RUNNING: four-class classifier CV" | Set-Content -Path $StatusPath -Encoding UTF8
    $cvArgs = @(
        (Join-Path $CodeRoot "train_gap_index_classifier_cv.py"),
        "--preprocessed-root", $PreprocessedRoot,
        "--output-dir", $OutputDir,
        "--sample-fraction", "$SampleFraction",
        "--folds", "$Folds",
        "--epochs", "$Epochs",
        "--batch-size", "$BatchSize",
        "--num-workers", "$NumWorkers",
        "--graph-feature-mode", $GraphFeatureMode,
        "--amp-mode", "bf16",
        "--device", "auto",
        "--best-metric", "val_macro_f1"
    )
    if ($ConfigsJson -ne "") {
        if (-not (Test-Path $ConfigsJson)) {
            throw "Configs JSON not found: $ConfigsJson"
        }
        $cvArgs += @("--configs-json", $ConfigsJson)
    }

    & $Python @cvArgs 2>&1 | Tee-Object -FilePath $TrainLog
    if ($LASTEXITCODE -ne 0) {
        throw "Classifier CV failed with exit code $LASTEXITCODE"
    }

    "DONE" | Set-Content -Path $StatusPath -Encoding UTF8
    Write-Log "Classifier CV finished successfully."
}
catch {
    "FAILED: $($_.Exception.Message)" | Set-Content -Path $StatusPath -Encoding UTF8
    Write-Log "FAILED: $($_.Exception.Message)"
    exit 1
}
