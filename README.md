CamHub AI Pro — Hệ Thống Giám Sát Camera AI Phát Hiện Té Ngã Cho Gia Đình

Biến điện thoại Android (chạy máy chủ Ubuntu qua Termux proot-distro), máy tính Windows, hoặc Linux thành Trạm Chủ Camera AI tại nhà — phát hiện té ngã & chuyển động theo thời gian thực, gửi album ảnh cảnh báo kèm biển số OCR tới Bot & Nhóm Telegram Gia đình, truy cập từ xa an toàn qua Cloudflare Tunnel (Link tạm miễn phí hoặc Link cố định vĩnh viễn) và điều khiển xoay/thu phóng camera ngay trên trình duyệt.

Tính năng chính

AI Phát hiện Té ngã, Người & Xe theo thời gian thực:

Phân tích khung xương bằng MediaPipe Pose (tự động chuyển sang OpenCV Person + Fall AI nếu phần cứng không hỗ trợ wheel), tích hợp Thermal Guard tự giảm tải khi máy nóng > 43°C.

Khung AI & Nhãn tiếng Việt trực tiếp (Khung AI: BẬT/TẮT): Vẽ khung bao, khung xương và gắn nhãn Người di chuyển, Người đi xe, Xe, Người té ngã trực tiếp trên luồng xem và ảnh toàn cảnh.

Bộ lọc chống loá đèn pha ban đêm: Tự nhận biết chớp sáng toàn khung hình và loại bỏ vệt sáng mịn trên sân/tường để tránh báo giả vào ban đêm.

AI Tự học theo phản hồi (👍 Chính xác, 🔢 Sai biển số, 👎 Báo nhầm):

Tự động điều chỉnh ngưỡng chuyển động, lọc vùng lá cây/báo giả theo từng mắt camera, nhưng vẫn bắt bình thường nếu có người hoặc xe thật đi vào vùng từng báo nhầm.

Album Telegram thông minh & Gợi ý biển số OCR:

Thu thập chuỗi khung hình trong 3.5 giây để chọn ảnh sắc nét nhất, gửi đồng thời ảnh toàn cảnh + ảnh cắt vuông 1:1 phóng to (Người/Người lái từ vai lên, Phương tiện, Biển số xe kèm gợi ý ký tự Tesseract OCR) và đường link xem trực tiếp tới nhiều Chat ID cá nhân & Group ID gia đình.

Tách đa mắt thông minh & Xem lại tối ưu (H.264 + AAC):

Tự động tách camera đa ống kính (2–3 mắt dọc như Yoosee 3 mắt) thành từng khung Mắt 1 / Mắt 2 / Mắt 3 độc lập.

Ghi hình xem lại chia màu rõ ràng: Video dài (60p) chuẩn H.264 + âm thanh AAC (tạo keyframe mỗi 3 giây để xem được ngay cả khi đang ghi) và các clip sự kiện Chuyển động / Té ngã (~60s). Luồng xem trực tiếp tối giản không phát âm thanh nền để đảm bảo mượt và ổn định qua Tunnel.

Truy cập từ xa Cloudflare Tunnel (Quick & Named Fixed Domain):

Tự khởi tạo link HTTPS miễn phí (*.trycloudflare.com) kèm mã QR Code chia sẻ nhanh, hoặc gắn Tên miền cố định (Cloudflare Named Tunnel Token) dùng một đường link duy nhất vĩnh viễn.

Tự động chuyển sang chế độ Adaptive Snapshot chống đứng hình khi xem qua Cloudflare Tunnel.

Phóng to Camera & Điều khiển Xoay / Thu phóng (PTZ + Zoom):

Mở chế độ tập trung Phóng to & PTZ tích hợp D-Pad xoay 4 hướng / Dừng / Góc gốc, Thu phóng quang học ONVIF (Zoom + / Zoom -), Thu phóng kỹ thuật số (1.0x – 4.0x + kéo di chuyển), xoay khung 0°/90°/180°/270°, đàm thoại 2 chiều và chụp ảnh nhanh.

Phân quyền Quản trị viên (Admin) & Thành viên Gia đình:

Khởi tạo tài khoản Admin lần đầu (PBKDF2-SHA256 + JWT + Nhật ký đăng nhập IP/Thiết bị); Admin tạo, đặt lại mật khẩu, sao chép lời mời và thu hồi tài khoản thành viên.

Kết nối Camera Tự động & Thủ công:

Tự động quét LAN tìm camera mở cổng RTSP (554, 8554) & ONVIF (80, 8000, 8899, 2020); hỗ trợ sẵn mẫu cho Imou, Ezviz, Hikvision, Dahua, KBVision, Yoosee, TP-Link Tapo và nhập Link RTSP thủ công kèm kiểm tra kết nối trước khi lưu.

Cài đặt nhanh trên Android (Termux + Máy chủ Ubuntu)

Xem hướng dẫn chi tiết từng bước (kèm cấu hình Cloudflare tên miền cố định & Nhóm Telegram) tại tệp INSTALL.md.

Bước 1: Cài ứng dụng Termux

Tải và cài đặt Termux bản mới nhất từ F-Droid hoặc GitHub (không dùng bản cũ trên Google Play).

(Khuyên dùng) Cài thêm Termux:API (đọc nhiệt độ pin, quét Wi-Fi) và Termux:Boot (tự khởi chạy trạm chủ khi điện thoại bật lại).

Bước 2: Chạy 1 dòng lệnh cài máy chủ Ubuntu & CamHub AI Pro

Mở Termux, dán lệnh sau và nhấn Enter:

curl -fsSL https://raw.githubusercontent.com/RaymondRMT/camhub-ai/main/setup_camhub.sh | bash


Trình cài đặt sẽ tự động cài proot-distro + Ubuntu, Python venv, OpenCV, FFmpeg, Tesseract OCR, ONVIF, MediaMTX, Cloudflared, liên kết thư mục video sang /sdcard/CamHub_Videos và tạo lệnh camhub.

Bước 3: Lệnh quản lý nhanh trong Termux

camhub start    # Khởi động trạm chủ Ubuntu & mở giao diện web
camhub stop     # Dừng toàn bộ dịch vụ
camhub status   # Kiểm tra trạng thái Uvicorn / MediaMTX / Cloudflared
camhub logs     # Xem nhật ký hoạt động
camhub update   # Cập nhật mã nguồn mới nhất từ GitHub


Cài đặt nhanh trên máy tính Windows (1-click)

Tải hoặc git clone https://github.com/RaymondRMT/camhub-ai.git và giải nén ra thư mục (ví dụ C:\camhub-ai).

Nhấp đúp setup_windows.bat (tự động cài Python 3.12, FFmpeg, Tesseract OCR, cloudflared.exe, mediamtx.exe, tạo .venv và cài đầy đủ thư viện Python + MediaPipe Pose nếu dùng Python 3.10–3.12).

Nhấp đúp start_windows.bat để khởi chạy trạm chủ tại http://127.0.0.1:8000 (nhấp đúp stop_windows.bat khi muốn tắt).

Kiểm thử dự án (Dành cho nhà phát triển)

python3 -m unittest discover -s tests -v


Cấu trúc thư mục

camhub-ai/
├── main.py              # FastAPI backend (Auth, Family, Camera, PTZ/Zoom, Tunnel, Telegram)
├── ai_vision.py         # AI phát hiện té ngã, nhận diện khung/nhãn, lọc loá đêm & OCR biển số
├── static/index.html    # Giao diện Web SPA Sáng/Tối, Focus PTZ/Zoom & Hướng dẫn cài đặt
├── setup_camhub.sh      # Trình cài đặt tự động Android Termux + Ubuntu proot-distro / Linux
├── start_camhub.sh      # Script khởi động MediaMTX + FastAPI (Linux/Termux)
├── stop_camhub.sh       # Script dừng dịch vụ (Linux/Termux)
├── setup_windows.bat    # Trình cài đặt tự động 1-click cho Windows
├── setup_windows.ps1    # Script PowerShell cài Python, FFmpeg, Tesseract, Cloudflared, MediaMTX
├── start_windows.bat    # Script khởi động MediaMTX + FastAPI trên Windows
├── stop_windows.bat     # Script dừng toàn bộ tiến trình trên Windows
├── requirements.txt     # Danh sách thư viện Python
├── tests/test_camhub.py # Bộ kiểm thử tự động API & luồng nghiệp vụ
├── android-launcher/    # Mã nguồn ứng dụng Android Launcher (Kotlin)
├── INSTALL.md           # Hướng dẫn cài đặt chi tiết từng bước
└── LICENSE              # Giấy phép mã nguồn mở MIT


Giấy phép

Phát hành theo giấy phép MIT License — xem chi tiết tại LICENSE.
