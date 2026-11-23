@echo off
cd /d "%~dp0"
echo Dang dung CamHub AI Pro tren Windows...
taskkill /F /IM cloudflared.exe >NUL 2>&1
taskkill /F /IM mediamtx.exe >NUL 2>&1
taskkill /F /IM ffmpeg.exe >NUL 2>&1
wmic process where "name='python.exe' and commandline like '%%uvicorn main:app%%'" call terminate >NUL 2>&1
echo Da dung toan bo tien trinh CamHub AI Pro.
pause
