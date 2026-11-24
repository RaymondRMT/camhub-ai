$ErrorActionPreference = "Stop"
$RootDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $RootDir

[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$BinDir = Join-Path $RootDir "bin"
New-Item -ItemType Directory -Force -Path $BinDir | Out-Null

function Refresh-Path {
    $machinePath = [System.Environment]::GetEnvironmentVariable("Path", "Machine")
    $userPath = [System.Environment]::GetEnvironmentVariable("Path", "User")
    $pyLocal = Join-Path $env:LocalAppData "Programs\Python\Python312"
    $pyLocalScripts = Join-Path $pyLocal "Scripts"
    $env:Path = "$BinDir;$pyLocal;$pyLocalScripts;$machinePath;$userPath;C:\Program Files\Python312;C:\Program Files\Tesseract-OCR"
}
Refresh-Path

Write-Host "[1/5] Kiem tra Python..." -ForegroundColor Cyan
$pyCmd = Get-Command python -ErrorAction SilentlyContinue
if (-not $pyCmd -or ($pyCmd.Source -like "*WindowsApps\python.exe")) {
    if (Get-Command winget -ErrorAction SilentlyContinue) {
        Write-Host "Dang cai Python 3.12 qua winget..."
        winget install -e --id Python.Python.3.12 --accept-package-agreements --accept-source-agreements
        Refresh-Path
    } else {
        Write-Host "Khong co winget - Dang tai Python 3.12 truc tiep tu python.org..." -ForegroundColor Yellow
        $PyInstaller = Join-Path $env:TEMP "python-3.12.7-amd64.exe"
        Invoke-WebRequest -Uri "https://www.python.org/ftp/python/3.12.7/python-3.12.7-amd64.exe" -OutFile $PyInstaller
        Write-Host "Dang cai dat Python 3.12 (vui long cho 30-60 giay)..."
        Start-Process -FilePath $PyInstaller -ArgumentList "/quiet InstallAllUsers=0 PrependPath=1 Include_test=0" -Wait
        Remove-Item $PyInstaller -Force -ErrorAction SilentlyContinue
        Refresh-Path
    }
}
$pyCmd = Get-Command python -ErrorAction SilentlyContinue
if (-not $pyCmd -or ($pyCmd.Source -like "*WindowsApps\python.exe")) {
    throw "Khong the tu dong cai Python 3.12. Vui long cai thu cong tu https://www.python.org/downloads/ (tich Add python.exe to PATH)."
}

Write-Host "[2/5] Kiem tra FFmpeg & Tesseract OCR..." -ForegroundColor Cyan
if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
    if (Get-Command winget -ErrorAction SilentlyContinue) {
        Write-Host "Dang cai FFmpeg qua winget..."
        winget install -e --id Gyan.FFmpeg --accept-package-agreements --accept-source-agreements
        Refresh-Path
    } else {
        Write-Host "Dang tai FFmpeg truc tiep vao thu muc bin..." -ForegroundColor Yellow
        $FfZip = Join-Path $env:TEMP "ffmpeg-essentials.zip"
        $FfTmp = Join-Path $env:TEMP "ffmpeg_extract"
        try {
            Invoke-WebRequest -Uri "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip" -OutFile $FfZip
            if (Test-Path $FfTmp) { Remove-Item $FfTmp -Recurse -Force }
            Expand-Archive -Path $FfZip -DestinationPath $FfTmp -Force
            Get-ChildItem -Path $FfTmp -Recurse -Include "ffmpeg.exe","ffprobe.exe" | ForEach-Object {
                Copy-Item $_.FullName -Destination (Join-Path $BinDir $_.Name) -Force
            }
            Remove-Item $FfZip,$FfTmp -Recurse -Force -ErrorAction SilentlyContinue
            Refresh-Path
        } catch {
            Write-Warning "Khong tai duoc FFmpeg tu dong. Vui long tai FFmpeg va chep ffmpeg.exe vao thu muc bin."
        }
    }
}
if (-not (Get-Command tesseract -ErrorAction SilentlyContinue)) {
    if (Get-Command winget -ErrorAction SilentlyContinue) {
        Write-Host "Dang cai Tesseract OCR (doc bien so) qua winget..."
        winget install -e --id UB-Mannheim.TesseractOCR --accept-package-agreements --accept-source-agreements
        Refresh-Path
    } else {
        Write-Host "Bo qua Tesseract OCR (khong bat buoc; co the cai sau neu can doc chu bien so)." -ForegroundColor DarkGray
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
