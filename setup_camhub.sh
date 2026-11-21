#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
#  CAMHUB AI PRO — TRÌNH CÀI ĐẶT TỰ ĐỘNG (ANDROID TERMUX + UBUNTU PROOT / LINUX)
#  Repository: https://github.com/RaymondRMT/camhub-ai
# ==============================================================================

REPO_URL="${CAMHUB_REPO_URL:-https://github.com/RaymondRMT/camhub-ai.git}"
SCRIPT_RAW_URL="${CAMHUB_SCRIPT_URL:-https://raw.githubusercontent.com/RaymondRMT/camhub-ai/main/setup_camhub.sh}"
MEDIAMTX_VERSION="v1.9.3"

print_banner() {
  echo "=================================================================="
  echo "   CAMHUB AI PRO — TRẠM CHỦ CAMERA AI PHÁT HIỆN TÉ NGÃ (UBUNTU)   "
  echo "=================================================================="
}

detect_arch_tag() {
  local arch
  arch="$(uname -m)"
  case "$arch" in
    aarch64|arm64) echo "arm64v8" ;;
    armv7l|armv7*) echo "armv7" ;;
    x86_64|amd64)  echo "amd64" ;;
    *)             echo "arm64v8" ;;
  esac
}

detect_cf_arch() {
  local arch
  arch="$(uname -m)"
  case "$arch" in
    aarch64|arm64) echo "arm64" ;;
    armv7l|armv7*) echo "arm" ;;
    x86_64|amd64)  echo "amd64" ;;
    *)             echo "arm64" ;;
  esac
}

install_inside_ubuntu() {
  local APP_DIR="${CAMHUB_APP_DIR:-$HOME/camhub_server}"
  local ARCH_TAG
  local CF_ARCH
  ARCH_TAG="$(detect_arch_tag)"
  CF_ARCH="$(detect_cf_arch)"

  echo "[Ubuntu 1/5] Cập nhật gói hệ thống & thư viện nền AI..."
  export DEBIAN_FRONTEND=noninteractive
  dpkg --configure -a --force-confold 2>/dev/null || true
  apt-get update -y
  apt-get install -y --no-install-recommends \
    -o Dpkg::Options::="--force-confdef" \
    -o Dpkg::Options::="--force-confold" \
    git curl wget tar ca-certificates \
    python3 python3-pip python3-venv \
    ffmpeg libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    procps iproute2

  echo "[Ubuntu 2/5] Tải mã nguồn CamHub AI Pro từ GitHub..."
  if [ -d "$APP_DIR/.git" ]; then
    cd "$APP_DIR"
    git fetch origin main && git reset --hard origin/main
  else
    rm -rf "$APP_DIR"
    git clone "$REPO_URL" "$APP_DIR"
    cd "$APP_DIR"
  fi

  mkdir -p "$APP_DIR/static" "$APP_DIR/snapshots" "$APP_DIR/recordings"

  # Liên kết thư mục lưu video sang /sdcard/CamHub_Videos nếu đang chạy trong Android proot
  if [ -d "/sdcard" ] && [ -w "/sdcard" ]; then
    mkdir -p /sdcard/CamHub_Videos
    if [ ! -L "$APP_DIR/recordings" ]; then
      rm -rf "$APP_DIR/recordings"
      ln -s /sdcard/CamHub_Videos "$APP_DIR/recordings"
    fi
  fi

  echo "[Ubuntu 3/5] Cài đặt MediaMTX ($ARCH_TAG) & Cloudflare Tunnel ($CF_ARCH)..."
  if [ ! -x "$APP_DIR/mediamtx" ]; then
    local MTX_TAR="mediamtx_${MEDIAMTX_VERSION}_linux_${ARCH_TAG}.tar.gz"
    wget -q "https://github.com/bluenviron/mediamtx/releases/download/${MEDIAMTX_VERSION}/${MTX_TAR}" -O "$MTX_TAR"
    tar -xzf "$MTX_TAR" mediamtx
    rm -f "$MTX_TAR"
    chmod +x "$APP_DIR/mediamtx"
  fi

  if ! command -v cloudflared >/dev/null 2>&1; then
    wget -q "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-${CF_ARCH}" -O /usr/local/bin/cloudflared
    chmod +x /usr/local/bin/cloudflared
  fi

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

  echo "[Ubuntu 4/5] Tạo môi trường Python venv & cài thư viện AI (OpenCV / MediaPipe / ONVIF)..."
  if [ ! -d "$APP_DIR/.venv" ]; then
    python3 -m venv "$APP_DIR/.venv"
  fi
  # shellcheck disable=SC1091
  source "$APP_DIR/.venv/bin/activate"
  pip install --upgrade pip setuptools wheel
  pip install --no-cache-dir -r "$APP_DIR/requirements.txt"
  # Cài đặt MediaPipe Pose trên Ubuntu (nếu wheel hỗ trợ kiến trúc máy)
  pip install --no-cache-dir mediapipe || echo "[Luu y] MediaPipe khong kha dung cho kien truc nay — tu dong dung OpenCV Contour Fallback."

  echo "[Ubuntu 5/5] Hoàn tất script khởi động trong Ubuntu..."
  chmod +x "$APP_DIR/start_camhub.sh" "$APP_DIR/stop_camhub.sh" 2>/dev/null || true
  echo ">>> Cài đặt lõi Ubuntu hoàn tất tại: $APP_DIR"
}

install_on_termux_android() {
  print_banner
  echo "Phát hiện môi trường Android Termux — Tiến hành cài máy chủ Ubuntu (proot-distro)..."

  # 1. Giữ CPU hoạt động nền & cấp quyền bộ nhớ
  termux-wake-lock || true
  termux-setup-storage || true
  sleep 2

  mkdir -p "$HOME/.termux" "$HOME/.shortcuts" "$HOME/.termux/boot"
  grep -q "allow-external-apps" "$HOME/.termux/termux.properties" 2>/dev/null || echo "allow-external-apps = true" >> "$HOME/.termux/termux.properties"
  termux-reload-settings || true

  echo "[Termux 1/4] Cập nhật Termux (tự động giữ cấu hình cũ, không dừng hỏi dpkg) & cài proot-distro..."
  export DEBIAN_FRONTEND=noninteractive
  dpkg --configure -a --force-confold 2>/dev/null || true
  pkg update -y
  apt-get upgrade -y \
    -o Dpkg::Options::="--force-confdef" \
    -o Dpkg::Options::="--force-confold"
  pkg install -y \
    -o Dpkg::Options::="--force-confdef" \
    -o Dpkg::Options::="--force-confold" \
    proot-distro git curl wget termux-api

  echo "[Termux 2/4] Cài đặt máy chủ Ubuntu trong Termux..."
  if ! proot-distro list 2>/dev/null | grep -qE "ubuntu.*installed"; then
    proot-distro install ubuntu
  else
    echo "Ubuntu đã được cài đặt sẵn trên proot-distro."
  fi

  echo "[Termux 3/4] Cài đặt CamHub AI Pro bên trong máy chủ Ubuntu..."
  mkdir -p /sdcard/CamHub_Videos || true
  # Tải sẵn script vào $PREFIX/tmp (chia sẻ thẳng vào /tmp của Ubuntu qua --shared-tmp, không cần Ubuntu có sẵn curl)
  curl -fsSL "$SCRIPT_RAW_URL" -o "$PREFIX/tmp/setup_camhub_ubuntu.sh"
  proot-distro login ubuntu --shared-tmp --bind /sdcard:/sdcard -- bash -c "
    export CAMHUB_REPO_URL='$REPO_URL'
    bash /tmp/setup_camhub_ubuntu.sh --inside-ubuntu
  "

  echo "[Termux 4/4] Tạo lệnh điều khiển nhanh 'camhub' & tự khởi động (Termux:Boot)..."
  cat << 'EOF' > "$PREFIX/bin/camhub"
#!/data/data/com.termux/files/usr/bin/bash
ACTION="${1:-start}"
case "$ACTION" in
  start)
    termux-wake-lock || true
    proot-distro login ubuntu --shared-tmp --bind /sdcard:/sdcard -- bash -c "cd ~/camhub_server && bash ./start_camhub.sh"
    sleep 2
    termux-open-url "http://localhost:8000" 2>/dev/null || true
    ;;
  stop)
    proot-distro login ubuntu --shared-tmp -- bash -c "cd ~/camhub_server && bash ./stop_camhub.sh"
    ;;
  status)
    proot-distro login ubuntu --shared-tmp -- bash -c "pgrep -af 'uvicorn|mediamtx|cloudflared' || echo 'CamHub AI dang dung'"
    ;;
  logs)
    proot-distro login ubuntu --shared-tmp -- bash -c "tail -n 60 ~/camhub_server/server.log"
    ;;
  update)
    proot-distro login ubuntu --shared-tmp --bind /sdcard:/sdcard -- bash -c "cd ~/camhub_server && git pull origin main && source .venv/bin/activate && pip install -r requirements.txt"
    ;;
  *)
    echo "Cach dung: camhub {start|stop|status|logs|update}"
    ;;
esac
EOF
  chmod +x "$PREFIX/bin/camhub"

  cat << 'EOF' > "$HOME/.shortcuts/CamHub AI Pro.sh"
#!/data/data/com.termux/files/usr/bin/bash
camhub start
EOF
  chmod +x "$HOME/.shortcuts/CamHub AI Pro.sh"
  cp "$HOME/.shortcuts/CamHub AI Pro.sh" "$HOME/.termux/boot/start_camhub.sh"

  echo "=================================================================="
  echo "  CÀI ĐẶT HOÀN TẤT! ĐANG KHỞI ĐỘNG CAMHUB AI PRO TRÊN UBUNTU...   "
  echo "  • Mở trình duyệt: http://localhost:8000                         "
  echo "  • Lệnh quản lý:   camhub start | stop | status | logs | update  "
  echo "=================================================================="
  camhub start
}

# --- ENTRYPOINT ---
if [ "${1:-}" = "--inside-ubuntu" ]; then
  install_inside_ubuntu
elif [ -n "${TERMUX_VERSION:-}" ] || [ -d "/data/data/com.termux" ]; then
  install_on_termux_android
else
  print_banner
  echo "Phát hiện hệ thống Linux / Ubuntu trực tiếp — Tiến hành cài đặt..."
  install_inside_ubuntu
  echo "Khởi động CamHub AI Pro: cd ~/camhub_server && bash ./start_camhub.sh"
fi