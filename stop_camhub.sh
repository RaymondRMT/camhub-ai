#!/usr/bin/env bash
pkill -f "./mediamtx" 2>/dev/null || true
pkill -f "uvicorn main:app" 2>/dev/null || true
pkill -f "cloudflared tunnel" 2>/dev/null || true
echo "Đã dừng toàn bộ dịch vụ CamHub AI Pro."
