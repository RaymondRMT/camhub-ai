# CamHub AI Pro — Hệ Thống Giám Sát Camera AI Phát Hiện Té Ngã Cho Gia Đình

Biến điện thoại Android (chạy máy chủ **Ubuntu qua Termux `proot-distro`**) hoặc máy tính Linux thành **Trạm Chủ Camera AI tại nhà** — phát hiện té ngã theo thời gian thực, gửi ảnh cảnh báo ngay tới **Bot & Nhóm Telegram Gia đình**, truy cập từ xa an toàn qua **Cloudflare Tunnel (Link tạm miễn phí hoặc Link cố định vĩnh viễn)** và điều khiển xoay/thu phóng camera ngay trên trình duyệt.

---

## Tính năng chính

* **AI Phát hiện Té ngã & Chuyển động trên Ubuntu ARM64:** Phân tích khung xương người bằng **MediaPipe Pose** (tự động chuyển sang **OpenCV Contour Fallback** nếu phần cứng không hỗ trợ wheel), tích hợp **Thermal Guard** tự giảm tải khi máy nóng > 43°C.
* **Cảnh báo tức thì qua Bot & Nhóm Telegram:** Gửi đồng thời hình ảnh khoảnh khắc té ngã + đường link xem trực tiếp tới nhiều Chat ID cá nhân và Group ID nhóm gia đình.
* **Truy cập từ xa Cloudflare Tunnel (Quick & Named Fixed Domain):**
  * Tự khởi tạo link HTTPS miễn phí (`*.trycloudflare.com`) kèm mã **QR Code** để chia sẻ nhanh.
  * Hỗ trợ gắn **Tên miền cố định (Cloudflare Named Tunnel Token)** để gia đình dùng liên tục một đường link duy nhất dù mất điện hay khởi động lại máy.
* **Phân quyền Quản trị viên (Admin) & Thành viên Gia đình:**
  * Màn hình khởi tạo tài khoản **Admin lần đầu** an toàn (PBKDF2-SHA256 + JWT + Nhật ký đăng nhập IP/Thiết bị).
  * Admin tạo, đặt lại mật khẩu, sao chép lời mời và thu hồi tài khoản thành viên gia đình.
* **Kết nối Camera Tự động & Thủ công:**
  * Tự động quét dải mạng LAN tìm camera mở cổng RTSP (`554`, `8554`) & ONVIF (`80`, `8000`, `8899`, `2020`).
  * Hỗ trợ sẵn mẫu RTSP & gợi ý mật khẩu cho **Imou, Ezviz, Hikvision, Dahua, KBVision, Yoosee, TP-Link Tapo** và chế độ nhập **Link RTSP thủ công** + kiểm tra kết nối trước khi lưu.
  * Phát trực tuyến mượt mà qua **MediaMTX HLS** kèm **MJPEG Fallback** tự động.
* **Phóng to Camera chỉ định kèm Điều khiển Xoay & Thu phóng (PTZ + Zoom):**
  * Khi bấm **Phóng to & PTZ** trên bất kỳ camera nào, giao diện mở chế độ tập trung tích hợp **D-Pad xoay 4 hướng / Dừng / Góc gốc**, **Thu phóng quang học ONVIF (`Zoom +` / `Zoom -`)**, **Thu phóng kỹ thuật số (`1.0x – 4.0x` + kéo di chuyển vùng phóng to)**, chọn tốc độ xoay, đàm thoại 2 chiều và chụp ảnh nhanh.

---

## Cài đặt nhanh trên Android (Termux + Máy chủ Ubuntu)

> Xem hướng dẫn chi tiết từng bước (kèm cấu hình Cloudflare tên miền cố định & Nhóm Telegram) tại tệp [INSTALL.md](./INSTALL.md).

### Bước 1: Cài ứng dụng Termux
1. Tải và cài đặt **Termux** bản mới nhất từ **F-Droid** hoặc GitHub (không dùng bản cũ trên Google Play).
2. *(Khuyên dùng)* Cài thêm **Termux:API** (đọc nhiệt độ pin, quét Wi-Fi) và **Termux:Boot** (tự khởi chạy trạm chủ khi điện thoại bật lại).

### Bước 2: Chạy 1 dòng lệnh cài máy chủ Ubuntu & CamHub AI Pro
Mở Termux, dán lệnh sau và nhấn Enter:

```bash
curl -fsSL https://raw.githubusercontent.com/RaymondRMT/camhub-ai/main/setup_camhub.sh | bash
```

Trình cài đặt sẽ tự động:
1. Cài `proot-distro` và triển khai máy chủ **Ubuntu** trong Termux.
2. Cài **Python venv**, **OpenCV**, **MediaPipe**, **ONVIF**, **MediaMTX** và **Cloudflared** đúng kiến trúc CPU (`arm64` / `armv7` / `amd64`).
3. Liên kết thư mục lưu video sang `/sdcard/CamHub_Videos` và tạo lệnh điều khiển nhanh `camhub`.

### Bước 3: Lệnh quản lý nhanh trong Termux
```bash
camhub start    # Khởi động trạm chủ Ubuntu & mở giao diện web
camhub stop     # Dừng toàn bộ dịch vụ
camhub status   # Kiểm tra trạng thái Uvicorn / MediaMTX / Cloudflared
camhub logs     # Xem nhật ký hoạt động
camhub update   # Cập nhật mã nguồn mới nhất từ GitHub
```

---

## Kiểm thử dự án (Dành cho nhà phát triển)

```bash
python3 -m unittest discover -s tests -v
```

## Cấu trúc thư mục

```text
camhub-ai/
├── main.py              # FastAPI backend (Auth, Family, Camera, PTZ/Zoom, Tunnel, Telegram)
├── ai_vision.py         # AI phát hiện té ngã (MediaPipe Pose + OpenCV Fallback) & MJPEG buffer
├── static/index.html    # Giao diện Web SPA Sáng/Tối, Focus PTZ/Zoom & Hướng dẫn cài đặt
├── setup_camhub.sh      # Trình cài đặt tự động Android Termux + Ubuntu proot-distro / Linux
├── start_camhub.sh      # Script khởi động MediaMTX + FastAPI
├── stop_camhub.sh       # Script dừng dịch vụ
├── requirements.txt     # Danh sách thư viện Python
├── tests/test_camhub.py # Bộ kiểm thử tự động API & luồng nghiệp vụ
├── android-launcher/    # Mã nguồn ứng dụng Android Launcher (Kotlin)
├── INSTALL.md           # Hướng dẫn cài đặt chi tiết từng bước
└── LICENSE              # Giấy phép mã nguồn mở MIT
```

## Giấy phép

Phát hành theo giấy phép **MIT License** — xem chi tiết tại [LICENSE](./LICENSE).

## Cài đặt nhanh trên máy tính Windows
1. Tải hoặc `git clone https://github.com/RaymondRMT/camhub-ai.git`.
2. Nhấp đúp **`setup_windows.bat`** (tự động cài Python, FFmpeg, Tesseract OCR, `cloudflared.exe`, `mediamtx.exe` và thư viện Python/MediaPipe).
3. Nhấp đúp **`start_windows.bat`** để chạy trạm chủ tại `http://127.0.0.1:8000` (dùng **`stop_windows.bat`** để tắt).
