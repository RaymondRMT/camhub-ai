@echo off
title CamHub AI Pro Server (Windows)
cd /d "%~dp0"
set "PATH=%~dp0bin;C:\Program Files\Tesseract-OCR;%PATH%"

if not exist ".venv\Scripts\python.exe" (
    echo [LOI] Chua cai dat moi truong .venv. Vui long chay setup_windows.bat truoc!
    pause
    exit /b 1
)

if exist "bin\mediamtx.exe" (
    tasklist /FI "IMAGENAME eq mediamtx.exe" 2>NUL | find /I /N "mediamtx.exe" >NUL
    if errorlevel 1 (
        start "CamHub MediaMTX" /MIN "%~dp0bin\mediamtx.exe" "%~dp0mediamtx.yml"
    )
)

echo ==================================================
echo   CamHub AI Pro dang chay tai: http://127.0.0.1:8000
echo   Dong cua so nay hoac chay stop_windows.bat de tat
echo ==================================================
".venv\Scripts\python.exe" -m uvicorn main:app --host 0.0.0.0 --port 8000
