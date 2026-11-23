@echo off
title CamHub AI Pro - Cai dat cho Windows
cd /d "%~dp0"
echo ==================================================
echo   CamHub AI Pro - Tu dong cai dat tren Windows
echo ==================================================
echo.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup_windows.ps1"
if errorlevel 1 (
    echo.
    echo [LOI] Qua trinh cai dat gap su co. Vui long kiem tra thong bao o tren.
    pause
    exit /b 1
)
echo.
echo [HOAN TAT] Ban co the chay file start_windows.bat de khoi dong he thong!
pause
