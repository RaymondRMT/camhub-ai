#!/data/data/com.termux/files/usr/bin/bash
set -e
clear
echo "=========================================================="
echo "   CAMHUB AI PRO - CAI DAT TU DONG TREN TERMUX + UBUNTU   "
echo "=========================================================="

REPO_URL="https://github.com/RaymondRMT/camhub-ai.git"
APP_DIR="$HOME/camhub-ai"

# Buoc 1: Chuan bi moi truong Termux & tai ma nguon
termux-wake-lock || true
termux-setup-storage || true
mkdir -p ~/.termux
grep -q "allow-external-apps" ~/.termux/termux.properties 2>/dev/null || echo "allow-external-apps = true" >> ~/.termux/termux.properties
termux-reload-settings || true

echo "[1/5] Cap nhat Termux & cai dat git, proot-distro..."
pkg update -y && pkg upgrade -y
pkg install -y git proot-distro termux-api wget tar || true

if [ -d "$APP_DIR/.git" ]; then
    cd "$APP_DIR" && git pull origin main
else
    rm -rf "$APP_DIR"
    git clone "$REPO_URL" "$APP_DIR"
fi

# Lien ket thu muc luu video ra bo nho chung dien thoai
mkdir -p /sdcard/CamHub_Videos "$APP_DIR/static" "$APP_DIR/snapshots"
rm -rf "$APP_DIR/recordings"
ln -s /sdcard/CamHub_Videos "$APP_DIR/recordings"

# Buoc 2: Cai dat Ubuntu thu gon ben trong Termux
echo "[2/5] Kiem tra & cai dat Ubuntu (proot-distro)..."
if [ ! -d "$PREFIX/var/lib/proot-distro/installed-rootfs/ubuntu" ]; then
    proot-distro install ubuntu
fi

# Buoc 3 & 4: Cai dat cong cu nen tang & thu vien Python trong Ubuntu (khong hoi tzdata)
echo "[3/5] & [4/5] Cai dat Python, FFmpeg va thu vien AI trong Ubuntu..."
proot-distro login ubuntu -- bash -c '
set -e
export DEBIAN_FRONTEND=noninteractive
ln -fs /usr/share/zoneinfo/Asia/Ho_Chi_Minh /etc/localtime
apt-get update && apt-get install -y python3 python3-pip python3-venv ffmpeg libgl1 libglib2.0-0 tzdata
dpkg-reconfigure -f noninteractive tzdata

if [ ! -d /root/venv ]; then
    python3 -m venv /root/venv
fi

/root/venv/bin/pip install --upgrade pip
/root/venv/bin/pip install --prefer-binary \
  fastapi uvicorn aiofiles psutil onvif-zeep \
  opencv-python-headless numpy httpx websockets \
  python-multipart PyJWT passlib bcrypt python-dotenv

/root/venv/bin/python -c "import cv2, fastapi, onvif, httpx, jwt; print(\"Cai dat thanh cong! OpenCV:\", cv2.__version__)"
'

# Buoc 5: Tao lenh khoi dong nhanh 1 cham (camhub)
echo "[5/5] Tao lenh tat camhub & tu dong khoi chay..."
SERVER_DIR=$(dirname "$(find "$HOME/camhub-ai" -name "main.py" | head -n 1)")

cat << EOF > "$HOME/start_camhub.sh"
#!/data/data/com.termux/files/usr/bin/bash
termux-wake-lock || true
proot-distro login ubuntu -- bash -c "cd $SERVER_DIR && /root/venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000"
EOF
chmod +x "$HOME/start_camhub.sh"

grep -q 'alias camhub=' ~/.bashrc 2>/dev/null || echo 'alias camhub="bash ~/start_camhub.sh"' >> ~/.bashrc

mkdir -p ~/.shortcuts ~/.termux/boot
cp "$HOME/start_camhub.sh" ~/.shortcuts/"CamHub AI Pro.sh"
cp "$HOME/start_camhub.sh" ~/.termux/boot/start_camhub.sh

echo "=========================================================="
echo " HOAN TAT! Go lenh 'camhub' de bat dau chay may chu.      "
echo "=========================================================="
bash "$HOME/start_camhub.sh"
