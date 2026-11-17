# CamHub AI Pro — Hệ Thống Quản Lý Camera Gia Đình & Cảnh Báo Té Ngã Thông Minh

Biến chiếc điện thoại Android cũ của bạn thành **Trạm Chủ Camera AI (Edge Server)** mạnh mẽ, bảo mật tuyệt đối mà không tốn phí lưu trữ đám mây hàng tháng.

## Tính năng nổi bật
* **Gom mọi hãng Camera về 1 nơi:** Hỗ trợ Hikvision, Dahua, Imou, Ezviz, KBVision, Yoosee và chuẩn ONVIF. Tự động quét IP camera trong mạng Wi-Fi.
* **Điều khiển Xoay Camera (PTZ ONVIF) & Đàm thoại 2 chiều:** Xoay trái/phải/lên/xuống và nhấn giữ để nói chuyện trực tiếp qua loa camera ngay trên trình duyệt.
* **AI Phát hiện Chuyển động & Cảnh báo Té ngã:** Phân tích dáng người theo thời gian thực trên chip ARM điện thoại, tự động giảm tải khi máy nóng (Thermal Guard) và gửi ngay ảnh chụp khoảnh khắc khẩn cấp qua **Telegram Bot**.
* **Xem từ xa miễn phí qua 4G/5G + Chia sẻ QR Code:** Tích hợp sẵn đường hầm mã hóa Cloudflare Tunnel. Quản trị viên tạo tài khoản riêng cho từng thành viên gia đình và chia sẻ mã QR để người thân quét là xem ngay.
* **Lưu trữ thông minh 30 ngày:** Video tự động lưu vào thư mục `/CamHub_Videos` trên điện thoại và tự động dọn dẹp xoay vòng theo dung lượng thực tế.
* **Giao diện iOS Glassmorphism Sáng/Tối:** Hỗ trợ xem lưới **1x1 / 2x2 / 3x3**, chế độ toàn màn hình (Fullscreen) và trình xem lại (Playback) tua nhanh lên tới **16x**.

---

## Hướng dẫn cài đặt chỉ trong 2 phút

### Bước 1: Cài đặt nền tảng Termux
1. Tải và cài đặt ứng dụng **Termux** (bản mới nhất từ F-Droid hoặc GitHub).
2. *(Khuyên dùng)* Cài thêm **Termux:API** (để quét Wi-Fi & giám sát nhiệt độ pin) và **Termux:Boot** (để hệ thống tự chạy lại nếu điện thoại sập nguồn khởi động lại).

### Bước 2: Chạy trình cài đặt tự động 1-chạm
* **Cách A (Dùng ứng dụng Launcher):** Cài đặt tệp `CamHubInstaller.apk`, mở ứng dụng và nhấn nút **"CÀI ĐẶT & KHỞI CHẠY NGAY"**.
* **Cách B (Dán 1 dòng lệnh vào Termux):** Mở Termux, dán dòng lệnh dưới đây và nhấn Enter:
  ```bash
  curl -fsSL https://raw.githubusercontent.com/RaymondRMT/camhub-ai/main/setup_camhub.sh | bash
  ```
