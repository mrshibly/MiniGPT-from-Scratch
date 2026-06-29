# PowerShell Script to run MiniGPT 124M parameter training on RTX 5060 Ti GPU

Write-Host "=================================================" -ForegroundColor Cyan
Write-Host " Starting MiniGPT 124M (Stretch Config) Training " -ForegroundColor Cyan
Write-Host " Target Device: NVIDIA GeForce RTX 5060 Ti (16GB) " -ForegroundColor Cyan
Write-Host "=================================================" -ForegroundColor Cyan

# Set environment variables for PyTorch / CUDA optimization
$env:CUDA_LAUNCH_BLOCKING = "0"
$env:PYTHONUNBUFFERED = "1"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"

$venvPython = "$PSScriptRoot\.venv\Scripts\python.exe"

if (-not (Test-Path $venvPython)) {
    Write-Host "[INFO] Virtual environment not found. Creating .venv..." -ForegroundColor Yellow
    & "C:\Program Files\Autodesk\3ds Max 2025\Python\python.exe" -m venv "$PSScriptRoot\.venv"
}

Write-Host "[INFO] Using Python: $venvPython" -ForegroundColor Green

# Verify processed data and tokenizer
if (-not (Test-Path "data\processed\train.bin") -or -not (Test-Path "data\tokenizer\tokenizer.json")) {
    Write-Host "[WARNING] Missing processed binary dataset or tokenizer." -ForegroundColor Yellow
    Write-Host "[INFO] Running data preparation scripts..." -ForegroundColor Yellow
    & $venvPython src\datasets\download_fineweb.py
    & $venvPython src\datasets\clean_text.py
    & $venvPython src\tokenizer\train_tokenizer.py
    & $venvPython src\datasets\prepare_data.py
} else {
    Write-Host "[INFO] Processed dataset (train.bin, val.bin) and tokenizer verified!" -ForegroundColor Green
}

# Ensure checkpoints directory exists
if (-not (Test-Path "checkpoints")) {
    New-Item -ItemType Directory -Path "checkpoints" | Out-Null
}

Write-Host "`n[INFO] Launching 124M Parameter Training Loop..." -ForegroundColor Cyan
& $venvPython src\train\train.py
