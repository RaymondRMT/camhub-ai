# Hướng Dẫn Cài Đặt & Vận Hành Chi Tiết — CamHub AI Pro

Tài liệu này hướng dẫn từng bước từ cài đặt trạm chủ Ubuntu trên Android đến cấu hình Camera, Nhóm Telegram Gia đình và Link cố định Cloudflare Tunnel.

---

## Phần 1: Cài đặt Trạm chủ Ubuntu trên điện thoại Android

1. **Tải Termux từ F-Droid:** Mở trình duyệt trên điện thoại Android, cài đặt `Termux`, `Termux:API` và `Termux:Boot` từ F-Droid.
2. **Tắt tối ưu hóa pin cho Termux:** Vào *Cài đặt Android → Ứng dụng → Termux → Pin → Không hạn chế (Unrestricted)* để hệ thống chạy nền 24/7 không bị Android tắt.
3. **Chạy trình cài đặt tự động:** Mở Termux và chạy:
   ```bash
   curl -fsSL https://raw.githubusercontent.com/RaymondRMT/camhub-ai/main/setup_camhub.sh | bash
   ```
4. Quá trình cài đặt Ubuntu và thư viện AI mất khoảng **3–6 phút** tùy tốc độ mạng. Khi hoàn tất, truy cập giao diện tại `http://localhost:8000` (hoặc `http://<IP-dien-thoai>:8000` từ máy tính cùng Wi-Fi).

---

## Phần 2: Thiết lập lần đầu cho Quản trị viên (Admin)

1. Khi mở giao diện lần đầu, hệ thống tự hiển thị màn hình **Khởi tạo Quản trị viên lần đầu**.
2. Nhập **Tên hiển thị**, **Tên đăng nhập** và **Mật khẩu** (tối thiểu 4 ký tự) → bấm **Khởi tạo Quản trị viên**.
3. Sau khi đăng nhập, vào mục **Cài đặt** để theo dõi thanh tiến độ **5 bước thiết lập**.

---

## Phần 3: Kết nối Camera trong mạng gia đình

### Cách 1: Tự động quét mạng LAN
1. Đảm bảo điện thoại trạm chủ và các camera kết nối cùng một mạng Wi-Fi gia đình.
2. Trong mục **Cài đặt → 1. Kết nối Camera**, chọn tab **Tự động (Quét LAN)** → bấm **Quét Camera LAN**.
3. Chọn IP camera tìm được → chọn hãng camera (Imou, Ezviz, Hikvision, Dahua, KBVision, Yoosee, Tapo...) → nhập mật khẩu / Safety Code / Verification Code.
4. Bấm **Kiểm tra kết nối** để xác nhận cổng RTSP hoạt động, sau đó bấm **Lưu & Bắt đầu giám sát**.

### Cách 2: Nhập Link RTSP thủ công
1. Chọn tab **Thủ công (Link RTSP)**.
2. Dán đầy đủ URL RTSP (ví dụ `rtsp://admin:123456@192.168.1.50:554/stream1`) → bấm **Kiểm tra kết nối** → **Lưu & Bắt đầu giám sát**.

---

## Phần 4: Thiết lập Bot & Nhóm Telegram Gia đình

1. Mở Telegram, tìm **@BotFather**, nhắn `/newbot` và làm theo hướng dẫn để lấy **Bot Token**.
2. Tạo một **Nhóm Telegram Gia đình**, mời các thành viên và thêm Bot vừa tạo vào nhóm.
3. Nhắn 1 tin bất kỳ trong nhóm, sau đó mở link sau trên trình duyệt (thay `<TOKEN>` bằng Bot Token của bạn):
   ```text
   https://api.telegram.org/bot<TOKEN>/getUpdates
   ```
   Tìm trường `"chat":{"id":-100xxxxxxxxxx}` — đó là **Group ID** của nhóm gia đình.
4. Vào mục **Cài đặt → 2. Cảnh báo qua Bot & Nhóm Telegram**, dán **Bot Token** và **Chat ID / Group ID** (có thể nhập nhiều ID cách nhau bởi dấu phẩy, ví dụ: `123456789, -1009876543210`).
5. Bấm **Lưu cấu hình Telegram** rồi bấm **Gửi tin thử nghiệm** để kiểm tra.

---

## Phần 5: Gắn Link cố định vĩnh viễn qua Cloudflare Named Tunnel

Mặc định hệ thống tự tạo link miễn phí `*.trycloudflare.com` (đổi mỗi lần khởi động lại). Để có **link cố định vĩnh viễn** cho cả gia đình dùng liên tục:

1. Đăng ký tài khoản **Cloudflare** miễn phí và trỏ 1 tên miền về Cloudflare DNS.
2. Truy cập **Cloudflare Zero Trust → Networks → Tunnels → Create a tunnel → Cloudflared**.
3. Đặt tên tunnel (ví dụ `camhub-home`) và sao chép đoạn **Token** sau tham số `--token`.
4. Chuyển sang tab **Public Hostname → Add a public hostname**:
   * **Subdomain + Domain:** ví dụ `cam.nhacuatui.com`
   * **Service Type:** `HTTP`
   * **URL:** `127.0.0.1:8000`
5. Quay lại giao diện CamHub → mục **Gia đình & Tunnel → Gắn Link cố định vĩnh viễn**, nhập `https://cam.nhacuatui.com` và dán **Token** → bấm **Kích hoạt Link cố định**.

---

## Phần 6: Cấp tài khoản Gia đình & Điều khiển PTZ / Thu phóng

1. Trong mục **Gia đình & Tunnel**, tạo tài khoản riêng cho từng thành viên (Mẹ, Vợ, Con...) rồi bấm **Chép lời mời** hoặc cho người thân quét **Mã QR**.
2. Tại màn hình **Trực tiếp**, bấm **Phóng to & PTZ** trên camera bất kỳ để mở bảng điều khiển xoay 4 hướng, về góc gốc, **Zoom quang học (`Zoom +`/`Zoom -`)**, **Phóng to kỹ thuật số (`1.0x – 4.0x`)** và đàm thoại 2 chiều.
