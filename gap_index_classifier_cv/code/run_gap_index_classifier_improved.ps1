param(
    [string]$PreprocessedRoot = "E:\datasets_graph_cache_novel109_top_batches001_600_20260512\source_000",
    [string]$ProjectRoot = "",
    [string]$OutputDir = "",
    [int]$Epochs = 16,
    [int]$BatchSize = 448,
    [int]$HiddenDim = 224,
    [int]$Layers = 8,
    [double]$Dropout = 0.10,
    [double]$LearningRate = 3e-4,
    [double]$WeightDecay = 1e-5,
    [int]$NumWorkers = 4,
    [string]$GraphFeatureMode = "novel105_phase_edge_shell",
    [string]$ClassWeightScheme = "sqrt_inverse",
    [string]$Sampler = "weighted",
    [double]$SamplerPower = 0.50,
    [double]$FocalGamma = 1.5,
    [double]$LabelSmoothing = 0.02
)

$ErrorActionPreference = "Stop"
$env:PYTHONUNBUFFERED = "1"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
if ($ProjectRoot -eq "") {
    $ProjectRoot = (Resolve-Path (Join-Path $ScriptDir "..\..")).Path
}
if ($OutputDir -eq "") {
    $OutputDir = Join-Path $ProjectRoot "gap_index_classifier_full_runs"
}

$Python = Join-Path $ProjectRoot ".venv-gnn-cu128\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    $Python = "python"
}
$LogDir = Join-Path $ProjectRoot "overnight_training_logs\gap_index_classifier_improved_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
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
    Write-Log "Improved four-class gap-index classifier job started."
    Write-Log "PreprocessedRoot=$PreprocessedRoot"
    Write-Log "OutputDir=$OutputDir"
    Write-Log "Epochs=$Epochs BatchSize=$BatchSize NumWorkers=$NumWorkers"
    Write-Log "ClassWeightScheme=$ClassWeightScheme Sampler=$Sampler SamplerPower=$SamplerPower FocalGamma=$FocalGamma LabelSmoothing=$LabelSmoothing"

    if (-not (Test-Path $PreprocessedRoot)) {
        throw "Preprocessed root not found: $PreprocessedRoot"
    }

    "RUNNING: improved four-class classifier" | Set-Content -Path $StatusPath -Encoding UTF8
    & $Python (Join-Path $ScriptDir "train_gap_index_classifier_improved.py") `
        --preprocessed-root $PreprocessedRoot `
        --output-dir $OutputDir `
        --class-mode four-class `
        --max-batches 0 `
        --max-samples 0 `
        --min-sample-index 0 `
        --max-sample-index 0 `
        --epochs $Epochs `
        --batch-size $BatchSize `
        --hidden-dim $HiddenDim `
        --layers $Layers `
        --dropout $Dropout `
        --lr $LearningRate `
        --weight-decay $WeightDecay `
        --num-workers $NumWorkers `
        --graph-feature-mode $GraphFeatureMode `
        --warmup-fraction 0.08 `
        --min-lr-scale 0.10 `
        --amp-mode bf16 `
        --device auto `
        --class-weight-scheme $ClassWeightScheme `
        --sampler $Sampler `
        --sampler-power $SamplerPower `
        --focal-gamma $FocalGamma `
        --label-smoothing $LabelSmoothing `
        --best-metric macro_f1 2>&1 | Tee-Object -FilePath $TrainLog
    if ($LASTEXITCODE -ne 0) {
        throw "Improved classifier training failed with exit code $LASTEXITCODE"
    }

    "DONE" | Set-Content -Path $StatusPath -Encoding UTF8
    Write-Log "Improved classifier job finished successfully."
}
catch {
    "FAILED: $($_.Exception.Message)" | Set-Content -Path $StatusPath -Encoding UTF8
    Write-Log "FAILED: $($_.Exception.Message)"
    exit 1
}
