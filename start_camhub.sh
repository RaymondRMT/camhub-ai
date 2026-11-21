#!/usr/bin/env bash
set -e
APP_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$APP_DIR"

pkill -f "./mediamtx" 2>/dev/null || true
pkill -f "uvicorn main:app" 2>/dev/null || true
sleep 1

if [ -x "$APP_DIR/mediamtx" ] && [ -f "$APP_DIR/mediamtx.yml" ]; then
  nohup "$APP_DIR/mediamtx" "$APP_DIR/mediamtx.yml" > "$APP_DIR/mediamtx.log" 2>&1 &
  sleep 2
fi

if [ -x "$APP_DIR/.venv/bin/uvicorn" ]; then
  UVICORN_BIN="$APP_DIR/.venv/bin/uvicorn"
else
  UVICORN_BIN="uvicorn"
fi

nohup "$UVICORN_BIN" main:app --host 0.0.0.0 --port 8000 > "$APP_DIR/server.log" 2>&1 &
sleep 2

echo "CamHub AI Pro đã khởi động tại: http://localhost:8000"
echo "Xem log trực tiếp: tail -f $APP_DIR/server.log"
