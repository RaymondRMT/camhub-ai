import os
import sys
import tempfile
import unittest
from pathlib import Path

# Thiết lập môi trường test tách biệt trước khi import main
TEST_DIR = tempfile.TemporaryDirectory(prefix="camhub_test_")
os.environ["CAMHUB_DB_PATH"] = os.path.join(TEST_DIR.name, "test_camhub.db")
os.environ["CAMHUB_SECRET_FILE"] = os.path.join(TEST_DIR.name, ".jwt_secret")
os.environ["CAMHUB_DISABLE_TUNNEL"] = "1"

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from fastapi.testclient import TestClient  # noqa: E402
import main  # noqa: E402


class CamHubEndToEndTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(main.app)
        cls.admin_token = None
        cls.member_token = None
        cls.created_cam_id = None

    @classmethod
    def tearDownClass(cls):
        if cls.created_cam_id:
            main.ai_engine.stop_camera_monitor(cls.created_cam_id)
        TEST_DIR.cleanup()

    def _auth_header(self, token: str) -> dict:
        return {"Authorization": f"Bearer {token}"}

    def test_01_ui_and_first_time_admin_setup(self):
        # 1. Trang chủ trả về HTML hợp lệ
        r = self.client.get("/")
        self.assertEqual(r.status_code, 200)
        self.assertIn("CamHub AI Pro", r.text)
        self.assertIn("camFocusModal", r.text)

        # 2. Chưa có Admin ban đầu
        r = self.client.get("/api/auth/status")
        self.assertEqual(r.status_code, 200)
        self.assertFalse(r.json()["has_admin"])

        # 3. Khởi tạo Admin lần đầu
        r = self.client.post("/api/auth/setup-admin", json={
            "username": "admin_home",
            "display_name": "Quản Trị Gia Đình",
            "password": "StrongPassword123",
        })
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertEqual(data["role"], "admin")
        self.assertTrue(data["token"])
        self.__class__.admin_token = data["token"]

        # 4. Không cho phép khởi tạo Admin lần 2
        r2 = self.client.post("/api/auth/setup-admin", json={
            "username": "hacker",
            "display_name": "Hacker",
            "password": "Password123",
        })
        self.assertEqual(r2.status_code, 400)

        # 5. Kiểm tra status đã có Admin
        r = self.client.get("/api/auth/status")
        self.assertTrue(r.json()["has_admin"])

    def test_02_family_member_lifecycle_and_rbac(self):
        hdr = self._auth_header(self.admin_token)
        # Tạo thành viên gia đình
        r = self.client.post("/api/family", headers=hdr, json={
            "username": "mom_cam",
            "display_name": "Mẹ",
            "password": "mompassword",
        })
        self.assertEqual(r.status_code, 200)

        # Danh sách thành viên phải có Admin + Mẹ
        r = self.client.get("/api/family", headers=hdr)
        self.assertEqual(r.status_code, 200)
        members = r.json()["members"]
        self.assertEqual(len(members), 2)
        mom = [m for m in members if m["username"] == "mom_cam"][0]

        # Admin đặt lại mật khẩu thành viên
        r = self.client.put(f"/api/family/{mom['id']}/password", headers=hdr, json={"new_password": "newmompass"})
        self.assertEqual(r.status_code, 200)

        # Thành viên đăng nhập bằng mật khẩu mới
        r = self.client.post("/api/auth/login", json={
            "username": "mom_cam",
            "password": "newmompass",
            "remember_me": True,
        })
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["role"], "member")
        self.__class__.member_token = r.json()["token"]

        # Kiểm tra phân quyền: Thành viên KHÔNG được vào API quản trị
        r_forbidden = self.client.get("/api/family", headers=self._auth_header(self.member_token))
        self.assertEqual(r_forbidden.status_code, 403)

    def test_03_camera_auto_manual_snapshot_and_ptz_validation(self):
        hdr = self._auth_header(self.admin_token)

        # 1. Thêm camera theo mẫu hãng (tự động)
        r = self.client.post("/api/cameras", headers=hdr, json={
            "name": "Phòng Khách",
            "ip": "192.168.1.88",
            "brand": "imou",
            "cam_user": "admin",
            "cam_pass": "SAFE1234",
        })
        self.assertEqual(r.status_code, 200)
        cam_id = r.json()["id"]
        self.__class__.created_cam_id = cam_id
        self.assertIn("****", r.json()["rtsp_masked"])

        # 2. Thêm camera theo link RTSP thủ công
        r_manual = self.client.post("/api/cameras", headers=hdr, json={
            "name": "Sân Trước",
            "brand": "custom",
            "custom_rtsp_url": "rtsp://admin:secret@192.168.1.99:554/live/ch0",
        })
        self.assertEqual(r_manual.status_code, 200)
        manual_id = r_manual.json()["id"]

        # 3. Thành viên gia đình xem được danh sách camera & lấy snapshot
        r_list = self.client.get("/api/cameras", headers=self._auth_header(self.member_token))
        self.assertEqual(r_list.status_code, 200)
        self.assertEqual(len(r_list.json()["cameras"]), 2)

        r_snap = self.client.get(f"/api/cameras/{cam_id}/snapshot?token={self.member_token}")
        self.assertEqual(r_snap.status_code, 200)
        self.assertEqual(r_snap.headers["content-type"], "image/jpeg")
        self.assertTrue(len(r_snap.content) > 500)

        # 4. Kiểm tra validation lệnh PTZ hợp lệ (zoom_in/zoom_out/home/left/right/up/down/stop) vs lệnh sai
        r_bad_ptz = self.client.post(f"/api/cameras/{cam_id}/ptz", headers=hdr, json={"direction": "invalid_cmd"})
        self.assertEqual(r_bad_ptz.status_code, 400)

        # Lệnh zoom_in hợp lệ (trả 503 nếu chưa cài onvif-zeep hoặc 400 nếu không có camera thật trên LAN)
        r_zoom = self.client.post(f"/api/cameras/{cam_id}/ptz", headers=hdr, json={"direction": "zoom_in", "speed": 0.8})
        self.assertIn(r_zoom.status_code, (200, 400, 503))

        # Xóa camera thủ công
        r_del = self.client.delete(f"/api/cameras/{manual_id}", headers=hdr)
        self.assertEqual(r_del.status_code, 200)

    def test_04_cloudflare_fixed_domain_and_telegram_multi_target(self):
        hdr = self._auth_header(self.admin_token)

        # 1. Cấu hình Link cố định Cloudflare Named Tunnel (tự chuẩn hóa https://)
        r = self.client.post("/api/tunnel/config", headers=hdr, json={
            "cf_token": "eyJhIjoiZGVtb190b2tlbiJ9",
            "custom_domain": "cam.giadinh.vn",
        })
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["custom_domain"], "https://cam.giadinh.vn")

        r_info = self.client.get("/api/tunnel", headers=hdr)
        self.assertEqual(r_info.status_code, 200)
        self.assertEqual(r_info.json()["mode"], "named")
        self.assertEqual(r_info.json()["url"], "https://cam.giadinh.vn")
        self.assertTrue(r_info.json()["has_token"])

        # 2. Cấu hình Telegram với nhiều Chat ID + Group ID gia đình
        r_tg = self.client.post("/api/telegram/config", headers=hdr, json={
            "bot_token": "123456:DEMO_TOKEN",
            "chat_id": "99887766, -1001122334455",
            "alert_fall": True,
            "alert_motion": False,
        })
        self.assertEqual(r_tg.status_code, 200)
        self.assertEqual(r_tg.json()["targets"], ["99887766", "-1001122334455"])

        # 3. Kích hoạt cảnh báo té ngã thử nghiệm & kiểm tra log cảnh báo
        r_fall = self.client.post("/api/alerts/test-fall", headers=hdr, json={"cam_id": self.created_cam_id})
        self.assertEqual(r_fall.status_code, 200)
        self.assertEqual(r_fall.json()["alert"]["type"], "fall")

        r_alerts = self.client.get("/api/alerts", headers=self._auth_header(self.member_token))
        self.assertEqual(r_alerts.status_code, 200)
        self.assertGreaterEqual(len(r_alerts.json()["alerts"]), 1)

        # 4. Kiểm tra tiến độ Setup Checklist (5/5 bước đều đã hoàn tất)
        r_chk = self.client.get("/api/setup/checklist", headers=hdr)
        self.assertEqual(r_chk.status_code, 200)
        chk = r_chk.json()
        self.assertTrue(chk["has_admin"])
        self.assertTrue(chk["has_cameras"])
        self.assertTrue(chk["has_telegram"])
        self.assertTrue(chk["has_fixed_domain"])
        self.assertTrue(chk["has_family_members"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
