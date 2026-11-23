$ErrorActionPreference = "Stop"
$RootDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $RootDir

$BinDir = Join-Path $RootDir "bin"
New-Item -ItemType Directory -Force -Path $BinDir | Out-Null

function Refresh-Path {
    $machinePath = [System.Environment]::GetEnvironmentVariable("Path", "Machine")
    $userPath = [System.Environment]::GetEnvironmentVariable("Path", "User")
    $env:Path = "$BinDir;$machinePath;$userPath;C:\Program Files\Tesseract-OCR"
}
Refresh-Path

Write-Host "[1/5] Kiem tra Python..." -ForegroundColor Cyan
$pyCmd = Get-Command python -ErrorAction SilentlyContinue
if (-not $pyCmd) {
    if (Get-Command winget -ErrorAction SilentlyContinue) {
        Write-Host "Dang cai Python 3.12 qua winget..."
        winget install -e --id Python.Python.3.12 --accept-package-agreements --accept-source-agreements
        Refresh-Path
    } else {
        throw "Chua co Python va khong tim thay winget. Vui long cai Python 3.10-3.12 tu python.org roi chay lai."
    }
}

Write-Host "[2/5] Kiem tra FFmpeg & Tesseract OCR..." -ForegroundColor Cyan
if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
    if (Get-Command winget -ErrorAction SilentlyContinue) {
        Write-Host "Dang cai FFmpeg qua winget..."
        winget install -e --id Gyan.FFmpeg --accept-package-agreements --accept-source-agreements
        Refresh-Path
    } else {
        Write-Warning "Chua co FFmpeg. Vui long cai FFmpeg va them vao PATH."
    }
}
if (-not (Get-Command tesseract -ErrorAction SilentlyContinue)) {
    if (Get-Command winget -ErrorAction SilentlyContinue) {
        Write-Host "Dang cai Tesseract OCR (doc bien so) qua winget..."
        winget install -e --id UB-Mannheim.TesseractOCR --accept-package-agreements --accept-source-agreements
        Refresh-Path
    }
}

Write-Host "[3/5] Kiem tra Cloudflared & MediaMTX..." -ForegroundColor Cyan
$CfExe = Join-Path $BinDir "cloudflared.exe"
if (-not (Test-Path $CfExe)) {
    Write-Host "Dang tai cloudflared.exe..."
    Invoke-WebRequest -Uri "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe" -OutFile $CfExe
}

$MtxExe = Join-Path $BinDir "mediamtx.exe"
if (-not (Test-Path $MtxExe)) {
    Write-Host "Dang tai mediamtx.exe..."
    $MtxZip = Join-Path $BinDir "mediamtx.zip"
    try {
        Invoke-WebRequest -Uri "https://github.com/bluenviron/mediamtx/releases/download/v1.9.3/mediamtx_v1.9.3_windows_amd64.zip" -OutFile $MtxZip
        Expand-Archive -Path $MtxZip -DestinationPath $BinDir -Force
        Remove-Item $MtxZip -Force -ErrorAction SilentlyContinue
    } catch {
        Write-Warning "Khong tai duoc MediaMTX (khong bat buoc neu dung Yoosee RTSP truc tiep)."
    }
}

Write-Host "[4/5] Tao moi truong ao .venv va cai thu vien Python..." -ForegroundColor Cyan
$VenvDir = Join-Path $RootDir ".venv"
if (-not (Test-Path $VenvDir)) {
    python -m venv $VenvDir
}
$VenvPy = Join-Path $VenvDir "Scripts\python.exe"
& $VenvPy -m pip install --upgrade pip
& $VenvPy -m pip install -r (Join-Path $RootDir "requirements.txt")

# Thu cai MediaPipe Pose neu Python <= 3.12 de bat khung xuong day du
$pyVer = & $VenvPy -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
if ([version]$pyVer -le [version]"3.12") {
    Write-Host "Thu cai MediaPipe Pose cho khung xuong day du (Python $pyVer)..." -ForegroundColor Cyan
    try {
        & $VenvPy -m pip install "mediapipe>=0.10.14"
    } catch {
        Write-Warning "Khong cai duoc mediapipe, he thong se dung OpenCV Person + Fall AI."
    }
}

Write-Host "[5/5] Tao cau hinh mac dinh..." -ForegroundColor Cyan
$MtxYml = Join-Path $RootDir "mediamtx.yml"
if (-not (Test-Path $MtxYml)) {
    Set-Content -Path $MtxYml -Value "paths:`n  all:`n    source: publisher`n" -Encoding UTF8
}
New-Item -ItemType Directory -Force -Path (Join-Path $RootDir "snapshots") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $RootDir "recordings") | Out-Null

Write-Host "Cai dat CamHub AI Pro cho Windows thanh cong!" -ForegroundColor Green
