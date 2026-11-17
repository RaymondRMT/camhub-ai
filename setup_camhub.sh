#!/data/data/com.termux/files/usr/bin/bash
set -e
clear
echo "=========================================================="
echo "   CAMHUB AI PRO - DANG TAI & CAI DAT TOAN BO HE THONG    "
echo "=========================================================="

# Thay thế bằng URL kho GitHub của bạn
REPO_URL="https://github.com/RaymondRMT/camhub-ai.git"
APP_DIR="$HOME/camhub_server"

# 1. Cấp quyền truy cập bộ nhớ & giữ CPU hoạt động nền
termux-wake-lock || true
termux-setup-storage || true
sleep 2

# Cho phép ứng dụng Launcher APK gọi lệnh Termux không bị chặn
mkdir -p ~/.termux
grep -q "allow-external-apps" ~/.termux/termux.properties 2>/dev/null || echo "allow-external-apps = true" >> ~/.termux/termux.properties
termux-reload-settings || true

echo "[1/5] Cap nhat he thong & tai cac goi nen tang..."
pkg update -y && pkg upgrade -y
pkg install -y git python python-numpy python-opencv ffmpeg wget tar openssl-tool termux-api libjpeg-turbo libpng cloudflared || \
pkg install -y git python ffmpeg wget tar openssl-tool termux-api libjpeg-turbo libpng cloudflared

echo "[2/5] Tai ma nguon & du lieu ung dung ve thiet bi..."
if [ -d "$APP_DIR/.git" ]; then
    cd "$APP_DIR" && git pull origin main
else
    rm -rf "$APP_DIR"
    git clone "$REPO_URL" "$APP_DIR"
    cd "$APP_DIR"
fi

# Liên kết thư mục lưu video ra bộ nhớ chung điện thoại (/sdcard/CamHub_Videos)
mkdir -p /sdcard/CamHub_Videos "$APP_DIR/static" "$APP_DIR/snapshots"
rm -rf "$APP_DIR/recordings"
ln -s /sdcard/CamHub_Videos "$APP_DIR/recordings"

echo "[3/5] Tai MediaMTX (Engine phat & ghi hinh ARM64)..."
if [ ! -f "./mediamtx" ]; then
    wget -q https://github.com/bluenviron/mediamtx/releases/download/v1.9.3/mediamtx_v1.9.3_linux_arm64v8.tar.gz
    tar -xzf mediamtx_v1.9.3_linux_arm64v8.tar.gz mediamtx
    rm -f mediamtx_v1.9.3_linux_arm64v8.tar.gz
    chmod +x mediamtx
fi

echo "[4/5] Cai dat thu vien AI & Web Server..."
pip install --upgrade pip
pip install --no-cache-dir fastapi uvicorn aiofiles psutil pyyaml httpx PyJWT python-multipart onvif-zeep
# Cài đặt thêm mediapipe / opencv-python-headless nếu kiến trúc thiết bị hỗ trợ wheel
pip install --no-cache-dir opencv-python-headless || true
pip install --no-cache-dir mediapipe || true

cat << 'EOF' > "$APP_DIR/mediamtx.yml"
webrtcAddress: :8889
hlsAddress: :8888
api: yes
apiAddress: 127.0.0.1:9997
paths:
  all_others:
    record: yes
    recordPath: ./recordings/%path/%Y-%m-%d_%H-%M-%S
    recordFormat: fmp4
    recordPartDuration: 1s
    recordSegmentDuration: 5m
EOF

echo "[5/5] Tao nut khoi dong nhanh & tu dong chay khi bat may..."
mkdir -p ~/.shortcuts ~/.termux/boot

cat << 'EOF' > "$APP_DIR/start_server.sh"
#!/data/data/com.termux/files/usr/bin/bash
termux-wake-lock || true
cd ~/camhub_server
pkill -f mediamtx || true
pkill -f "uvicorn main:app" || true
nohup ./mediamtx mediamtx.yml > mediamtx.log 2>&1 &
sleep 2
nohup uvicorn main:app --host 0.0.0.0 --port 8000 > server.log 2>&1 &
sleep 2
termux-open-url "http://localhost:8000" || true
EOF

chmod +x "$APP_DIR/start_server.sh"
cp "$APP_DIR/start_server.sh" ~/.shortcuts/"CamHub AI Pro.sh"
cp "$APP_DIR/start_server.sh" ~/.termux/boot/start_camhub.sh

echo "=========================================================="
echo " HOAN TAT! DANG MO GIAO DIEN QUAN LY CAMERA...            "
echo "=========================================================="
bash "$APP_DIR/start_server.sh"
