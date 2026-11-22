import os
import re
import time
import json
import uuid
import base64
import socket
import shutil
import secrets
import hashlib
import sqlite3
import tempfile
import ipaddress
import subprocess
import threading
import urllib.parse
import concurrent.futures
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Dict, Tuple
from fastapi import FastAPI, HTTPException, Depends, Header, Request, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import BaseModel
import httpx
import jwt

try:
    from onvif import ONVIFCamera
    HAS_ONVIF = True
except Exception:
    ONVIFCamera = None
    HAS_ONVIF = False

from ai_vision import CameraAIProcessor

os.makedirs("static", exist_ok=True)
os.makedirs("recordings", exist_ok=True)
os.makedirs("snapshots", exist_ok=True)

DB_PATH = "camhub.db"
SECRET_KEY_FILE = ".jwt_secret"
ALERTS_LOG = []
ALERTS_LOCK = threading.Lock()
ONVIF_CLIENTS = {}
TUNNEL_STATE = {"url": "", "status": "starting", "mode": "quick"}
tunnel_process = None
tunnel_lock = threading.Lock()
http_async_client: Optional[httpx.AsyncClient] = None

# Hỗ trợ luồng tối ưu cho từng hãng (sub = mượt nhẹ CPU cho AI & nhiều mắt cam, main = HD/FullHD)
# Hỗ trợ tham số {ch} cho thiết bị nhiều mắt cam (Multi-lens / Dual-lens / NVR)
BRAND_STREAM_TEMPLATES = {
    "hikvision": {
        "sub": "rtsp://{user}:{pwd}@{ip}:554/Streaming/Channels/{ch}02",
        "main": "rtsp://{user}:{pwd}@{ip}:554/Streaming/Channels/{ch}01",
    },
    "dahua": {
        "sub": "rtsp://{user}:{pwd}@{ip}:554/cam/realmonitor?channel={ch}&subtype=1",
        "main": "rtsp://{user}:{pwd}@{ip}:554/cam/realmonitor?channel={ch}&subtype=0",
    },
    "imou": {
        "sub": "rtsp://{user}:{pwd}@{ip}:554/cam/realmonitor?channel={ch}&subtype=1",
        "main": "rtsp://{user}:{pwd}@{ip}:554/cam/realmonitor?channel={ch}&subtype=0",
    },
    "ezviz": {
        "sub": "rtsp://{user}:{pwd}@{ip}:554/h264_stream",
        "main": "rtsp://{user}:{pwd}@{ip}:554/Streaming/Channels/{ch}01",
    },
    "kbvision": {
        "sub": "rtsp://{user}:{pwd}@{ip}:554/cam/realmonitor?channel={ch}&subtype=1",
        "main": "rtsp://{user}:{pwd}@{ip}:554/cam/realmonitor?channel={ch}&subtype=0",
    },
    "yoosee": {
        "sub": "rtsp://{user}:{pwd}@{ip}:554/onvif2",
        "main": "rtsp://{user}:{pwd}@{ip}:554/onvif1",
    },
    "tapo": {
        "sub": "rtsp://{user}:{pwd}@{ip}:554/stream2",
        "main": "rtsp://{user}:{pwd}@{ip}:554/stream1",
    },
    "custom": {
        "sub": "rtsp://{user}:{pwd}@{ip}:554/stream{ch}",
        "main": "rtsp://{user}:{pwd}@{ip}:554/stream{ch}",
    },
}

BRAND_RTSP_PATHS = {
    "imou": "/cam/realmonitor?channel=1&subtype=1",
    "dahua": "/cam/realmonitor?channel=1&subtype=1",
    "kbvision": "/cam/realmonitor?channel=1&subtype=1",
    "hikvision": "/Streaming/Channels/102",
    "ezviz": "/h264_stream",
    "yoosee": "/onvif2",
    "tapo": "/stream2",
    "custom": "/stream1",
}

BRAND_DEFAULT_HINTS = {
    "hikvision": {
        "default_user": "admin",
        "common_passwords": ["12345", "12345678a", "admin", "123456"],
        "guide": "Camera đời cũ dùng '12345'; đời mới dùng mật khẩu kích hoạt hoặc mã Verification Code (6 ký tự in hoa) dưới đáy camera.",
    },
    "dahua": {
        "default_user": "admin",
        "common_passwords": ["admin", "123456", "888888"],
        "guide": "Nếu chưa đổi mật khẩu, thử 'admin' hoặc '123456', hoặc nhập mã Safety Code in trên tem dán thân camera.",
    },
    "imou": {
        "default_user": "admin",
        "common_passwords": ["admin", "123456"],
        "guide": "Mật khẩu mặc định là mã Safety Code (8 ký tự gồm chữ & số) in trên tem dán ở đáy camera (hoặc xem trong App Imou Life -> Cài đặt -> Nhãn thiết bị).",
    },
    "ezviz": {
        "default_user": "admin",
        "common_passwords": ["123456", "admin", "888888"],
        "guide": "Mật khẩu mặc định bắt buộc là mã Verification Code (6 chữ cái IN HOA, VD: ABCDEF) in trên tem dán dưới chân đế camera.",
    },
    "kbvision": {
        "default_user": "admin",
        "common_passwords": ["admin", "123456", "888888"],
        "guide": "Mật khẩu mặc định thường là 'admin', '123456' hoặc mã Safety Code trên tem dán thiết bị.",
    },
    "yoosee": {
        "default_user": "admin",
        "common_passwords": ["123", "123456", "888888", "admin"],
        "guide": "Mật khẩu RTSP mặc định là '123' hoặc '888888' (cần bật mục 'Cài đặt RTSP / NVR' trong ứng dụng Yoosee). Camera 2 mắt dùng kênh 1 & 2.",
    },
    "tapo": {
        "default_user": "admin",
        "common_passwords": ["123456", "admin"],
        "guide": "Vào ứng dụng Tapo -> Cài đặt Camera -> Nâng cao -> Tài khoản Camera để xem hoặc tạo tài khoản RTSP/ONVIF.",
    },
    "custom": {
        "default_user": "admin",
        "common_passwords": ["admin", "123456", "12345", "888888", "123"],
        "guide": "Thử các mật khẩu mặc định phổ biến của chuẩn ONVIF ('admin', '123456', '888888' hoặc '123').",
    },
}


def get_db_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=15.0, check_same_thread=False)
    return conn


def get_jwt_secret() -> str:
    if os.path.exists(SECRET_KEY_FILE):
        with open(SECRET_KEY_FILE, "r", encoding="utf-8") as f:
            secret = f.read().strip()
            if secret:
                return secret
    secret = secrets.token_hex(32)
    with open(SECRET_KEY_FILE, "w", encoding="utf-8") as f:
        f.write(secret)
    try:
        os.chmod(SECRET_KEY_FILE, 0o600)
    except Exception:
        pass
    return secret


JWT_SECRET = get_jwt_secret()


def hash_password(password: str, salt: Optional[str] = None) -> str:
    if not salt:
        salt = secrets.token_hex(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 100_000)
    return f"{salt}${dk.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        salt, _ = stored.split("$", 1)
        return secrets.compare_digest(hash_password(password, salt), stored)
    except Exception:
        return False


def init_db():
    with get_db_connection() as conn:
        c = conn.cursor()
        c.execute("PRAGMA journal_mode=WAL;")
        c.execute("""CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            display_name TEXT,
            password_hash TEXT,
            role TEXT DEFAULT 'member',
            created_at TEXT
        )""")
        c.execute("""CREATE TABLE IF NOT EXISTS cameras (
            id TEXT PRIMARY KEY,
            name TEXT,
            ip TEXT,
            brand TEXT,
            cam_user TEXT,
            cam_pass TEXT,
            rtsp_url TEXT,
            channel INTEGER DEFAULT 1,
            stream_profile TEXT DEFAULT 'sub',
            device_group TEXT DEFAULT ''
        )""")
        # Tự động nâng cấp bảng cũ nếu thiếu cột channel / stream_profile / device_group
        existing_cols = {row[1] for row in c.execute("PRAGMA table_info(cameras)").fetchall()}
        if "channel" not in existing_cols:
            c.execute("ALTER TABLE cameras ADD COLUMN channel INTEGER DEFAULT 1")
        if "stream_profile" not in existing_cols:
            c.execute("ALTER TABLE cameras ADD COLUMN stream_profile TEXT DEFAULT 'sub'")
        if "device_group" not in existing_cols:
            c.execute("ALTER TABLE cameras ADD COLUMN device_group TEXT DEFAULT ''")

        c.execute("""CREATE TABLE IF NOT EXISTS login_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            status TEXT,
            ip_address TEXT,
            user_agent TEXT,
            timestamp TEXT
        )""")
        c.execute("""CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT
        )""")
        defaults = [
            ("retention_days", "30"),
            ("cf_token", ""),
            ("cf_custom_domain", ""),
            ("tg_bot_token", ""),
            ("tg_chat_id", ""),
            ("tg_alert_motion", "0"),
            ("tg_alert_fall", "1"),
        ]
        for k, v in defaults:
            c.execute("INSERT OR IGNORE INTO settings (key, value) VALUES (?, ?)", (k, v))
        conn.commit()


init_db()


def get_setting(key: str) -> str:
    with get_db_connection() as conn:
        row = conn.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
        return row[0] if row else ""


def set_setting(key: str, value: str):
    with get_db_connection() as conn:
        conn.execute("INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)", (key, value))
        conn.commit()


def log_login_attempt(username: str, status: str, request: Request):
    ip = (
        request.headers.get("CF-Connecting-IP")
        or request.headers.get("X-Forwarded-For")
        or (request.client.host if request.client else "Unknown")
    )
    ua = request.headers.get("User-Agent", "Unknown")[:120]
    with get_db_connection() as conn:
        conn.execute(
            "INSERT INTO login_logs (username, status, ip_address, user_agent, timestamp) VALUES (?, ?, ?, ?, ?)",
            (username, status, ip, ua, datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        )
        conn.execute("DELETE FROM login_logs WHERE id NOT IN (SELECT id FROM login_logs ORDER BY id DESC LIMIT 100)")
        conn.commit()


# --- TELEGRAM BOT ALERT ---
def send_telegram_alert(alert_data: dict):
    bot_token = get_setting("tg_bot_token").strip()
    chat_id = get_setting("tg_chat_id").strip()
    if not bot_token or not chat_id:
        return

    alert_type = alert_data.get("type")
    if alert_type == "motion" and get_setting("tg_alert_motion") != "1":
        return
    if alert_type == "fall" and get_setting("tg_alert_fall") != "1":
        return

    snap_rel = alert_data.get("snapshot", "").lstrip("/")
    prefix = "🚨 [KHẨN CẤP - TÉ NGÃ]" if alert_type == "fall" else "👁️ [CHUYỂN ĐỘNG]"
    caption = f"{prefix}\n{alert_data['message']}\n🕒 Thời gian: {alert_data['timestamp']}"

    def _worker():
        try:
            url = f"https://api.telegram.org/bot{bot_token}/sendPhoto"
            if snap_rel and os.path.exists(snap_rel):
                with open(snap_rel, "rb") as photo_file:
                    httpx.post(
                        url,
                        data={"chat_id": chat_id, "caption": caption},
                        files={"photo": photo_file},
                        timeout=12.0,
                    )
            else:
                httpx.post(
                    f"https://api.telegram.org/bot{bot_token}/sendMessage",
                    data={"chat_id": chat_id, "text": caption},
                    timeout=10.0,
                )
        except Exception:
            pass

    threading.Thread(target=_worker, daemon=True).start()


def handle_ai_alert(alert_data: dict):
    if "id" not in alert_data:
        alert_data["id"] = f"alert_{int(time.time() * 1000)}_{uuid.uuid4().hex[:6]}"
    with ALERTS_LOCK:
        ALERTS_LOG.insert(0, alert_data)
        if len(ALERTS_LOG) > 100:
            ALERTS_LOG.pop()
    send_telegram_alert(alert_data)

    def _notify_termux():
        try:
            subprocess.run(
                [
                    "termux-notification",
                    "--title", "CamHub AI Cảnh Báo",
                    "--content", alert_data["message"],
                    "--priority", "high",
                ],
                timeout=5,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception:
            pass

    threading.Thread(target=_notify_termux, daemon=True).start()


ai_engine = CameraAIProcessor(on_alert_callback=handle_ai_alert)


# --- REAL NETWORK & WIFI HELPERS ---
def _find_termux_cmd(cmd_name: str) -> Optional[str]:
    found = shutil.which(cmd_name)
    if found:
        return found
    termux_bin = f"/data/data/com.termux/files/usr/bin/{cmd_name}"
    if os.path.exists(termux_bin):
        return termux_bin
    return None


def _get_local_network_info() -> Dict:
    ip = ""
    iface = "wlan0"
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
    except Exception:
        ip = ""

    subnet = ".".join(ip.split(".")[:3]) + ".0/24" if ip and "." in ip else ""
    try:
        route_out = subprocess.check_output(["ip", "route", "get", "8.8.8.8"], timeout=2, stderr=subprocess.DEVNULL).decode("utf-8", errors="ignore")
        m_dev = re.search(r"dev\s+(\S+)", route_out)
        if m_dev:
            iface = m_dev.group(1)
    except Exception:
        pass

    ssid = ""
    rssi = -55
    freq = 0
    conn_cmd = _find_termux_cmd("termux-wifi-connectioninfo")
    if conn_cmd:
        try:
            raw = subprocess.check_output([conn_cmd], timeout=4, stderr=subprocess.DEVNULL).decode("utf-8", errors="ignore")
            info = json.loads(raw)
            ssid = str(info.get("ssid", "")).strip().strip('"')
            if ssid == "<unknown ssid>":
                ssid = ""
            rssi = int(info.get("rssi", -55))
            freq = int(info.get("frequency_mhz", 0))
            if info.get("ip"):
                ip = str(info["ip"])
                subnet = ".".join(ip.split(".")[:3]) + ".0/24"
        except Exception:
            pass

    if not ssid:
        iw_cmd = _find_termux_cmd("iwgetid")
        if iw_cmd:
            try:
                ssid = subprocess.check_output([iw_cmd, "-r"], timeout=2, stderr=subprocess.DEVNULL).decode("utf-8", errors="ignore").strip()
            except Exception:
                pass

    return {
        "ssid": ssid,
        "ip": ip or "192.168.1.100",
        "subnet": subnet or "192.168.1.0/24",
        "interface": iface,
        "rssi": rssi,
        "frequency_mhz": freq,
    }


# --- RTSP AUTO-PROBE & MULTI-LENS STREAM ENGINE ---
def build_rtsp_url(brand: str, ip: str, user: str, pwd: str, channel: int = 1, stream_profile: str = "sub") -> str:
    brand_key = (brand or "custom").lower().strip()
    profile_key = "main" if stream_profile == "main" else "sub"
    templates = BRAND_STREAM_TEMPLATES.get(brand_key, BRAND_STREAM_TEMPLATES["custom"])
    tmpl = templates.get(profile_key, templates["sub"])

    # Riêng Yoosee đa mắt: kênh 1 dùng onvif2/onvif1, kênh >= 2 dùng /live/ch0{ch-1}_1 hoặc onvif{ch+1}
    if brand_key == "yoosee" and channel > 1:
        ch_idx = channel - 1
        tmpl = "rtsp://{user}:{pwd}@{ip}:554/live/ch0" + str(ch_idx) + ("_0" if profile_key == "main" else "_1")
    elif brand_key == "ezviz" and channel > 1:
        tmpl = "rtsp://{user}:{pwd}@{ip}:554/Streaming/Channels/" + str(channel) + ("01" if profile_key == "main" else "02")

    safe_u = urllib.parse.quote((user or "admin").strip() or "admin", safe="")
    safe_p = urllib.parse.quote(pwd or "", safe="")
    return tmpl.format(user=safe_u, pwd=safe_p, ip=ip, ch=channel)


def _build_digest_auth_header(
    username: str,
    password: str,
    realm: str,
    nonce: str,
    method: str,
    uri: str,
) -> str:
    ha1 = hashlib.md5(f"{username}:{realm}:{password}".encode("utf-8")).hexdigest()
    ha2 = hashlib.md5(f"{method}:{uri}".encode("utf-8")).hexdigest()
    response = hashlib.md5(f"{ha1}:{nonce}:{ha2}".encode("utf-8")).hexdigest()
    return (
        f'Digest username="{username}", realm="{realm}", '
        f'nonce="{nonce}", uri="{uri}", response="{response}"'
    )


def _rtsp_describe_once(
    ip: str,
    port: int,
    path: str,
    username: str,
    password: str,
    timeout: float = 1.0,
) -> Tuple[int, str]:
    uri = f"rtsp://{ip}:{port}{path}"
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    try:
        sock.connect((ip, port))
        req1 = (
            f"DESCRIBE {uri} RTSP/1.0\r\n"
            f"CSeq: 1\r\n"
            f"User-Agent: CamHubAI/1.3\r\n"
            f"Accept: application/sdp\r\n\r\n"
        )
        sock.sendall(req1.encode("utf-8"))
        resp1 = sock.recv(4096).decode("utf-8", errors="ignore")
        if not resp1.startswith("RTSP/1.0"):
            return 0, ""

        parts = resp1.splitlines()[0].split()
        code1 = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
        server_hdr = ""
        for line in resp1.splitlines():
            if line.lower().startswith("server:"):
                server_hdr = line.split(":", 1)[1].strip()

        if code1 == 200 or code1 != 401:
            return code1, server_hdr

        realm_match = re.search(r'realm="([^"]+)"', resp1, re.IGNORECASE)
        nonce_match = re.search(r'nonce="([^"]+)"', resp1, re.IGNORECASE)
        if realm_match and nonce_match:
            auth_val = _build_digest_auth_header(
                username, password, realm_match.group(1), nonce_match.group(1), "DESCRIBE", uri
            )
            auth_header = f"Authorization: {auth_val}\r\n"
        else:
            b64 = base64.b64encode(f"{username}:{password}".encode("utf-8")).decode("ascii")
            auth_header = f"Authorization: Basic {b64}\r\n"

        req2 = (
            f"DESCRIBE {uri} RTSP/1.0\r\n"
            f"CSeq: 2\r\n"
            f"User-Agent: CamHubAI/1.3\r\n"
            f"Accept: application/sdp\r\n"
            f"{auth_header}\r\n"
        )
        try:
            sock.sendall(req2.encode("utf-8"))
            resp2 = sock.recv(4096).decode("utf-8", errors="ignore")
        except Exception:
            resp2 = ""

        if not resp2.startswith("RTSP/1.0"):
            sock.close()
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            sock.connect((ip, port))
            sock.sendall(req2.encode("utf-8"))
            resp2 = sock.recv(4096).decode("utf-8", errors="ignore")

        if not resp2.startswith("RTSP/1.0"):
            return 401, server_hdr
        parts2 = resp2.splitlines()[0].split()
        code2 = int(parts2[1]) if len(parts2) > 1 and parts2[1].isdigit() else 401
        return code2, server_hdr
    except Exception:
        return 0, ""
    finally:
        try:
            sock.close()
        except Exception:
            pass


def probe_camera_rtsp(
    ip: str,
    brand: str = "custom",
    cam_user: str = "admin",
    cam_pass: str = "",
    auto_try_defaults: bool = True,
    port: int = 554,
    stream_profile: str = "sub",
) -> Dict:
    brand_key = (brand or "custom").lower().strip()
    primary_path = BRAND_RTSP_PATHS.get(brand_key, "/stream1")

    passwords_to_try: List[str] = []
    if cam_pass:
        passwords_to_try.append(cam_pass)
    if auto_try_defaults:
        brand_pwds = BRAND_DEFAULT_HINTS.get(brand_key, BRAND_DEFAULT_HINTS["custom"])["common_passwords"]
        for pw in brand_pwds + ["admin", "123456", "12345", "888888", "123"]:
            if pw not in passwords_to_try:
                passwords_to_try.append(pw)
    if not passwords_to_try:
        passwords_to_try = ["admin"]

    user_clean = (cam_user or "admin").strip() or "admin"
    first_code, server_hdr = _rtsp_describe_once(ip, port, primary_path, user_clean, passwords_to_try[0])
    if first_code == 0:
        return {
            "reachable": False,
            "authenticated": False,
            "matched_user": user_clean,
            "matched_pass": cam_pass,
            "matched_brand": brand_key,
            "detected_channels": 1,
            "rtsp_url": "",
            "message": f"Không thể kết nối tới cổng RTSP {port} tại IP {ip}. Hãy kiểm tra IP hoặc bật ONVIF/RTSP trên camera.",
        }

    detected_brand = brand_key
    srv_lower = server_hdr.lower()
    if "hikvision" in srv_lower or "ezviz" in srv_lower:
        detected_brand = "hikvision" if brand_key == "custom" else brand_key
    elif "dahua" in srv_lower or "imou" in srv_lower:
        detected_brand = "imou" if brand_key == "custom" else brand_key

    for idx, pw in enumerate(passwords_to_try):
        code, _ = (first_code, server_hdr) if idx == 0 else _rtsp_describe_once(ip, port, primary_path, user_clean, pw)
        final_brand = detected_brand
        if code == 404:
            for alt_brand, alt_path in BRAND_RTSP_PATHS.items():
                if alt_path == primary_path:
                    continue
                alt_code, _ = _rtsp_describe_once(ip, port, alt_path, user_clean, pw)
                if alt_code == 200:
                    code = 200
                    final_brand = alt_brand
                    break

        if code == 200:
            # Kiểm tra thêm xem thiết bị có nhiều mắt cam / nhiều kênh (channel 2, 3) hay không
            detected_channels = 1
            for ch in [2, 3]:
                ch_url = build_rtsp_url(final_brand, ip, user_clean, pw, channel=ch, stream_profile=stream_profile)
                ch_path = "/" + ch_url.split(f":{port}/", 1)[1]
                ch_code, _ = _rtsp_describe_once(ip, port, ch_path, user_clean, pw, timeout=0.6)
                if ch_code == 200:
                    detected_channels = ch
                else:
                    break

            rtsp_url = build_rtsp_url(final_brand, ip, user_clean, pw, channel=1, stream_profile=stream_profile)
            return {
                "reachable": True,
                "authenticated": True,
                "matched_user": user_clean,
                "matched_pass": pw,
                "matched_brand": final_brand,
                "detected_channels": detected_channels,
                "rtsp_url": rtsp_url,
                "message": f"Kết nối RTSP thành công ({final_brand.upper()}, {detected_channels} mắt cam)! Mật khẩu: '{pw}'",
            }

    return {
        "reachable": True,
        "authenticated": False,
        "matched_user": user_clean,
        "matched_pass": cam_pass,
        "matched_brand": detected_brand,
        "detected_channels": 1,
        "rtsp_url": "",
        "message": "Cổng RTSP 554 đang mở nhưng không khớp mật khẩu mặc định. Vui lòng nhập mã Safety Code / Verification Code dưới đáy camera.",
    }


# --- MEDIAMTX & CLOUDFLARE TUNNEL ---
def register_mediamtx_path(cam_id: str, rtsp_url: str):
    payload = {"source": rtsp_url, "record": True}
    try:
        resp = httpx.post(
            f"http://127.0.0.1:9997/v3/config/paths/add/{cam_id}",
            json=payload,
            timeout=2.0,
        )
        if resp.status_code == 400:
            httpx.patch(
                f"http://127.0.0.1:9997/v3/config/paths/patch/{cam_id}",
                json=payload,
                timeout=2.0,
            )
    except Exception:
        pass


def sync_cameras_on_boot():
    with get_db_connection() as conn:
        rows = conn.execute("SELECT id, name, rtsp_url FROM cameras").fetchall()
    for cam_id, name, rtsp_url in rows:
        register_mediamtx_path(cam_id, rtsp_url)
        ai_engine.start_camera_monitor(cam_id, name, rtsp_url)


def start_cloudflare_tunnel():
    global tunnel_process, TUNNEL_STATE
    with tunnel_lock:
        if tunnel_process and tunnel_process.poll() is None:
            try:
                tunnel_process.terminate()
                tunnel_process.wait(timeout=3.0)
            except Exception:
                try:
                    tunnel_process.kill()
                except Exception:
                    pass

        cf_token = get_setting("cf_token").strip()
        custom_domain = get_setting("cf_custom_domain").strip()

        try:
            if cf_token:
                TUNNEL_STATE = {"url": custom_domain or "Named Tunnel Active", "status": "running", "mode": "named"}
                cmd = ["cloudflared", "tunnel", "--no-autoupdate", "run", "--token", cf_token]
                tunnel_process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                TUNNEL_STATE = {"url": "", "status": "starting", "mode": "quick"}
                cmd = ["cloudflared", "tunnel", "--no-autoupdate", "--url", "http://127.0.0.1:8000"]
                tunnel_process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

                def _watch_and_drain_url(proc):
                    if not proc.stdout:
                        return
                    for line in iter(proc.stdout.readline, ""):
                        if not TUNNEL_STATE.get("url"):
                            match = re.search(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com", line)
                            if match:
                                TUNNEL_STATE["url"] = match.group(0)
                                TUNNEL_STATE["status"] = "running"

                threading.Thread(target=_watch_and_drain_url, args=(tunnel_process,), daemon=True).start()
        except FileNotFoundError:
            TUNNEL_STATE = {"url": "", "status": "unavailable", "mode": "none"}
        except Exception:
            TUNNEL_STATE = {"url": "", "status": "error", "mode": "none"}


@asynccontextmanager
async def lifespan(app_instance: FastAPI):
    global http_async_client
    http_async_client = httpx.AsyncClient(timeout=10.0)
    start_cloudflare_tunnel()
    boot_timer = threading.Timer(1.5, sync_cameras_on_boot)
    boot_timer.daemon = True
    boot_timer.start()
    yield
    boot_timer.cancel()
    if http_async_client:
        await http_async_client.aclose()
    with tunnel_lock:
        if tunnel_process and tunnel_process.poll() is None:
            try:
                tunnel_process.terminate()
            except Exception:
                pass


app = FastAPI(title="CamHub AI Pro Server", lifespan=lifespan)

app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/recordings", StaticFiles(directory="recordings"), name="recordings")
app.mount("/snapshots", StaticFiles(directory="snapshots"), name="snapshots")


# --- AUTH DEPENDENCIES ---
def _extract_user_from_token(token: str) -> dict:
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
    except Exception:
        raise HTTPException(status_code=401, detail="Phiên đăng nhập hết hạn")


def get_current_user(authorization: Optional[str] = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Chưa đăng nhập")
    return _extract_user_from_token(authorization.split(" ", 1)[1])


def get_user_from_header_or_query(authorization: Optional[str] = Header(None), token: Optional[str] = None):
    if authorization and authorization.startswith("Bearer "):
        return _extract_user_from_token(authorization.split(" ", 1)[1])
    if token:
        return _extract_user_from_token(token)
    raise HTTPException(status_code=401, detail="Chưa đăng nhập")


def require_admin(user: dict = Depends(get_current_user)):
    if user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Yêu cầu quyền Quản trị viên")
    return user


def create_jwt_token(username: str, display_name: str, role: str, expire_days: int = 30) -> str:
    payload = {
        "sub": username,
        "name": display_name,
        "role": role,
        "exp": datetime.now(timezone.utc) + timedelta(days=expire_days),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm="HS256")


# --- MODELS ---
class LoginRequest(BaseModel):
    username: str
    password: str
    remember_me: bool = True


class RegisterAdminRequest(BaseModel):
    username: str
    display_name: str = "Quản trị viên"
    password: str


class ChangePasswordRequest(BaseModel):
    old_password: str
    new_password: str


class TelegramConfigRequest(BaseModel):
    bot_token: str
    chat_id: str
    alert_motion: bool = False
    alert_fall: bool = True


class FamilyMemberRequest(BaseModel):
    username: str
    display_name: str
    password: str


class TunnelConfigRequest(BaseModel):
    cf_token: str = ""
    custom_domain: str = ""


class WifiConnectRequest(BaseModel):
    ssid: str
    password: str = ""


class CameraCreateRequest(BaseModel):
    name: str
    ip: str
    brand: str
    cam_user: str = "admin"
    cam_pass: str = ""
    lens_count: int = 1
    stream_profile: str = "sub"


class CameraProbeRequest(BaseModel):
    ip: str
    brand: str = "imou"
    cam_user: str = "admin"
    cam_pass: str = ""
    auto_try_defaults: bool = True
    stream_profile: str = "sub"


class PTZCommand(BaseModel):
    direction: str
    speed: float = 0.6


class RetentionUpdate(BaseModel):
    days: int


# --- LIVE VIDEO: HLS PROXY + BUILT-IN REALTIME MJPEG STREAM FALLBACK ---
@app.get("/hls/{path:path}")
async def proxy_hls(path: str):
    if ".." in path or path.startswith("/"):
        raise HTTPException(status_code=400, detail="Đường dẫn HLS không hợp lệ")
    client = http_async_client
    try:
        if client is not None:
            resp = await client.get(f"http://127.0.0.1:8888/{path}")
        else:
            async with httpx.AsyncClient(timeout=4.0) as temp_client:
                resp = await temp_client.get(f"http://127.0.0.1:8888/{path}")
        return Response(
            content=resp.content,
            status_code=resp.status_code,
            media_type=resp.headers.get("content-type", "application/vnd.apple.mpegurl"),
        )
    except httpx.RequestError:
        raise HTTPException(status_code=502, detail="Luồng HLS MediaMTX chưa chạy, tự động chuyển sang MJPEG Realtime")


@app.get("/api/cameras/{cam_id}/mjpeg")
def stream_camera_mjpeg(cam_id: str, user: dict = Depends(get_user_from_header_or_query)):
    def _gen():
        while True:
            jpg = ai_engine.get_latest_jpeg(cam_id)
            if jpg:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Cache-Control: no-cache\r\n\r\n" + jpg + b"\r\n"
                )
            time.sleep(0.12)

    return StreamingResponse(
        _gen(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )


@app.get("/api/cameras/{cam_id}/snapshot")
def get_camera_snapshot(cam_id: str, user: dict = Depends(get_user_from_header_or_query)):
    jpg = ai_engine.get_latest_jpeg(cam_id)
    if not jpg:
        raise HTTPException(status_code=404, detail="Chưa nhận được khung hình từ camera")
    return Response(content=jpg, media_type="image/jpeg", headers={"Cache-Control": "no-cache"})


@app.get("/")
def serve_ui():
    return FileResponse("static/index.html")


# --- AUTH & SECURITY ROUTES ---
@app.get("/api/auth/status")
def auth_status():
    with get_db_connection() as conn:
        count = conn.execute("SELECT COUNT(*) FROM users WHERE role='admin'").fetchone()[0]
    return {"has_admin": count > 0}


@app.post("/api/auth/setup-admin")
def setup_admin(req: RegisterAdminRequest, request: Request):
    username = req.username.strip()
    display_name = (req.display_name or "Quản trị viên").strip()
    if not username or len(req.password) < 4:
        raise HTTPException(status_code=400, detail="Tài khoản không hợp lệ hoặc mật khẩu dưới 4 ký tự")

    with get_db_connection() as conn:
        if conn.execute("SELECT COUNT(*) FROM users WHERE role='admin'").fetchone()[0] > 0:
            raise HTTPException(status_code=400, detail="Đã tồn tại Quản trị viên")
        pwd_hash = hash_password(req.password)
        try:
            conn.execute(
                "INSERT INTO users (username, display_name, password_hash, role, created_at) VALUES (?, ?, ?, 'admin', ?)",
                (username, display_name, pwd_hash, datetime.now().strftime("%Y-%m-%d %H:%M")),
            )
            conn.commit()
        except sqlite3.IntegrityError:
            raise HTTPException(status_code=400, detail="Tên đăng nhập đã tồn tại")

    log_login_attempt(username, "Khởi tạo Admin", request)
    token = create_jwt_token(username, display_name, "admin", expire_days=30)
    return {"status": "ok", "token": token, "role": "admin", "display_name": display_name}


@app.post("/api/auth/login")
def login(req: LoginRequest, request: Request):
    username = req.username.strip()
    with get_db_connection() as conn:
        row = conn.execute(
            "SELECT username, display_name, password_hash, role FROM users WHERE username=?",
            (username,),
        ).fetchone()
    if not row or not verify_password(req.password, row[2]):
        log_login_attempt(username, "Thất bại (Sai mật khẩu)", request)
        raise HTTPException(status_code=401, detail="Sai tên đăng nhập hoặc mật khẩu")
    log_login_attempt(username, "Thành công", request)
    expire_days = 30 if req.remember_me else 1
    token = create_jwt_token(row[0], row[1], row[3], expire_days=expire_days)
    return {"status": "ok", "token": token, "role": row[3], "display_name": row[1]}


@app.post("/api/auth/change-password")
def change_password(req: ChangePasswordRequest, request: Request, user: dict = Depends(get_current_user)):
    if len(req.new_password) < 4:
        raise HTTPException(status_code=400, detail="Mật khẩu mới phải có ít nhất 4 ký tự")
    with get_db_connection() as conn:
        row = conn.execute("SELECT password_hash FROM users WHERE username=?", (user["sub"],)).fetchone()
        if not row or not verify_password(req.old_password, row[0]):
            raise HTTPException(status_code=400, detail="Mật khẩu hiện tại không chính xác")
        new_hash = hash_password(req.new_password)
        conn.execute("UPDATE users SET password_hash=? WHERE username=?", (new_hash, user["sub"]))
        conn.commit()
    log_login_attempt(user["sub"], "Đổi mật khẩu thành công", request)
    return {"status": "ok"}


@app.get("/api/security/login-logs")
def get_login_logs(admin: dict = Depends(require_admin)):
    with get_db_connection() as conn:
        rows = conn.execute(
            "SELECT username, status, ip_address, user_agent, timestamp FROM login_logs ORDER BY id DESC LIMIT 40"
        ).fetchall()
    return {
        "logs": [
            {"username": r[0], "status": r[1], "ip": r[2], "device": r[3], "time": r[4]}
            for r in rows
        ]
    }


# --- TELEGRAM CONFIG ---
@app.get("/api/telegram/config")
def get_telegram_config(admin: dict = Depends(require_admin)):
    return {
        "bot_token": get_setting("tg_bot_token"),
        "chat_id": get_setting("tg_chat_id"),
        "alert_motion": get_setting("tg_alert_motion") == "1",
        "alert_fall": get_setting("tg_alert_fall") == "1",
    }


@app.post("/api/telegram/config")
def save_telegram_config(req: TelegramConfigRequest, admin: dict = Depends(require_admin)):
    set_setting("tg_bot_token", req.bot_token.strip())
    set_setting("tg_chat_id", req.chat_id.strip())
    set_setting("tg_alert_motion", "1" if req.alert_motion else "0")
    set_setting("tg_alert_fall", "1" if req.alert_fall else "0")
    return {"status": "ok"}


@app.post("/api/telegram/test")
def test_telegram_alert(admin: dict = Depends(require_admin)):
    bot_token = get_setting("tg_bot_token").strip()
    chat_id = get_setting("tg_chat_id").strip()
    if not bot_token or not chat_id:
        raise HTTPException(status_code=400, detail="Chưa nhập Bot Token hoặc Chat ID")
    try:
        resp = httpx.post(
            f"https://api.telegram.org/bot{bot_token}/sendMessage",
            data={
                "chat_id": chat_id,
                "text": "✅ CamHub AI Pro: Kết nối Telegram Bot thành công! Hệ thống sẵn sàng gửi ảnh cảnh báo té ngã & chuyển động.",
            },
            timeout=10.0,
        )
    except httpx.RequestError:
        raise HTTPException(status_code=502, detail="Không thể kết nối tới máy chủ Telegram")
    if resp.status_code != 200:
        raise HTTPException(status_code=400, detail="Lỗi gửi tin nhắn Telegram, kiểm tra lại Token/Chat ID")
    return {"status": "ok"}


# --- FAMILY & TUNNEL ---
@app.get("/api/family")
def list_family(admin: dict = Depends(require_admin)):
    with get_db_connection() as conn:
        rows = conn.execute("SELECT id, username, display_name, role, created_at FROM users").fetchall()
    return {
        "members": [
            {"id": r[0], "username": r[1], "display_name": r[2], "role": r[3], "created_at": r[4]}
            for r in rows
        ]
    }


@app.post("/api/family")
def create_family_member(req: FamilyMemberRequest, admin: dict = Depends(require_admin)):
    username = req.username.strip()
    display_name = req.display_name.strip()
    if not username or not display_name or len(req.password) < 4:
        raise HTTPException(status_code=400, detail="Thông tin không hợp lệ hoặc mật khẩu dưới 4 ký tự")
    with get_db_connection() as conn:
        try:
            conn.execute(
                "INSERT INTO users (username, display_name, password_hash, role, created_at) VALUES (?, ?, ?, 'member', ?)",
                (username, display_name, hash_password(req.password), datetime.now().strftime("%Y-%m-%d %H:%M")),
            )
            conn.commit()
            return {"status": "ok"}
        except sqlite3.IntegrityError:
            raise HTTPException(status_code=400, detail="Tên đăng nhập đã tồn tại")


@app.delete("/api/family/{member_id}")
def delete_family_member(member_id: int, admin: dict = Depends(require_admin)):
    with get_db_connection() as conn:
        conn.execute("DELETE FROM users WHERE id=? AND role='member'", (member_id,))
        conn.commit()
    return {"status": "ok"}


@app.get("/api/tunnel")
def get_tunnel_info(admin: dict = Depends(require_admin)):
    return {
        "url": TUNNEL_STATE["url"],
        "status": TUNNEL_STATE["status"],
        "custom_domain": get_setting("cf_custom_domain"),
    }


@app.post("/api/tunnel/config")
def update_tunnel_config(req: TunnelConfigRequest, admin: dict = Depends(require_admin)):
    set_setting("cf_token", req.cf_token.strip())
    set_setting("cf_custom_domain", req.custom_domain.strip())
    start_cloudflare_tunnel()
    return {"status": "ok"}


@app.post("/api/tunnel/restart")
def restart_tunnel(admin: dict = Depends(require_admin)):
    start_cloudflare_tunnel()
    return {"status": "ok"}


# --- WIFI (REAL DEVICE NETWORK & SIGNAL), CAMERA, PTZ ONVIF & TWO-WAY AUDIO ---
@app.get("/api/wifi/scan")
def scan_wifi(admin: dict = Depends(require_admin)):
    current = _get_local_network_info()
    networks = []
    scan_source = "network_interface"

    scan_cmd = _find_termux_cmd("termux-wifi-scaninfo")
    if scan_cmd:
        try:
            output = subprocess.check_output([scan_cmd], timeout=8, stderr=subprocess.DEVNULL).decode("utf-8", errors="ignore")
            raw_list = json.loads(output)
            seen = {}
            if isinstance(raw_list, list):
                for net in raw_list:
                    ssid = str(net.get("ssid", "")).strip()
                    rssi = int(net.get("rssi", -100))
                    if ssid and (ssid not in seen or rssi > seen[ssid]["rssi"]):
                        seen[ssid] = {
                            "ssid": ssid,
                            "rssi": rssi,
                            "connected": ssid == current["ssid"],
                            "frequency_mhz": int(net.get("frequency_mhz", 0)),
                        }
            if seen:
                networks = sorted(seen.values(), key=lambda x: (not x["connected"], -x["rssi"]))
                scan_source = "termux_wifi_api"
        except Exception:
            pass

    if not networks and current["ssid"]:
        networks = [{
            "ssid": current["ssid"],
            "rssi": current["rssi"],
            "connected": True,
            "frequency_mhz": current["frequency_mhz"],
        }]

    return {
        "current": current,
        "networks": networks,
        "scan_source": scan_source,
        "note": (
            "" if scan_source == "termux_wifi_api"
            else "Đang lấy thông tin mạng LAN/Wi-Fi thực tế từ giao diện mạng thiết bị. Để quét danh sách Wi-Fi xung quanh, hãy cài ứng dụng Termux:API và cấp quyền Vị trí."
        ),
    }


@app.post("/api/wifi/connect")
def connect_wifi(req: WifiConnectRequest, admin: dict = Depends(require_admin)):
    ssid = req.ssid.strip()
    if not ssid:
        raise HTTPException(status_code=400, detail="Vui lòng chọn mạng Wi-Fi")
    wifi_enable_cmd = _find_termux_cmd("termux-wifi-enable")
    if wifi_enable_cmd:
        try:
            subprocess.run([wifi_enable_cmd, "true"], timeout=5, check=False)
            return {"status": "ok", "message": f"Đã kích hoạt Wi-Fi trên thiết bị cho mạng {ssid}"}
        except Exception as e:
            return {"status": "warning", "message": str(e)}
    return {
        "status": "info",
        "message": f"Android hạn chế đổi Wi-Fi tự động từ môi trường Linux. Vui lòng kết nối trực tiếp vào Wi-Fi '{ssid}' trong Cài đặt Android.",
    }


def _validate_camera_host(host: str) -> str:
    cleaned = host.strip()
    if not cleaned:
        raise HTTPException(status_code=400, detail="Địa chỉ IP Camera không được để trống")
    try:
        ipaddress.ip_address(cleaned)
        return cleaned
    except ValueError:
        if re.match(r"^[a-zA-Z0-9.-]+$", cleaned):
            return cleaned
        raise HTTPException(status_code=400, detail="Địa chỉ IP hoặc Hostname Camera không hợp lệ")


@app.get("/api/cameras/brand-hints")
def get_camera_brand_hints(user: dict = Depends(get_current_user)):
    return {"hints": BRAND_DEFAULT_HINTS}


@app.post("/api/cameras/probe")
def probe_camera_endpoint(req: CameraProbeRequest, admin: dict = Depends(require_admin)):
    ip_clean = _validate_camera_host(req.ip)
    return probe_camera_rtsp(
        ip=ip_clean,
        brand=req.brand,
        cam_user=req.cam_user,
        cam_pass=req.cam_pass,
        auto_try_defaults=req.auto_try_defaults,
        stream_profile=req.stream_profile,
    )


@app.get("/api/cameras/discover")
def discover_cameras(admin: dict = Depends(require_admin)):
    net_info = _get_local_network_info()
    ip = net_info["ip"]
    subnet = ".".join(ip.split(".")[:3])
    ips = [f"{subnet}.{i}" for i in range(1, 255)]
    found = []

    def _scan(target):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(0.35)
        try:
            ok = sock.connect_ex((target, 554)) == 0
        finally:
            sock.close()
        if not ok:
            return None
        probe = probe_camera_rtsp(target, brand="custom", cam_user="admin", cam_pass="", auto_try_defaults=True)
        return {
            "ip": target,
            "authenticated": probe["authenticated"],
            "matched_user": probe["matched_user"],
            "matched_pass": probe["matched_pass"] if probe["authenticated"] else "",
            "matched_brand": probe["matched_brand"],
            "detected_channels": probe.get("detected_channels", 1),
        }

    with concurrent.futures.ThreadPoolExecutor(max_workers=50) as ex:
        for res in ex.map(_scan, ips):
            if res:
                found.append(res)
    return {"subnet": net_info["subnet"], "host_ip": ip, "devices": found}


@app.get("/api/cameras")
def list_cameras(user: dict = Depends(get_current_user)):
    with get_db_connection() as conn:
        rows = conn.execute(
            "SELECT id, name, ip, brand, channel, stream_profile, device_group FROM cameras ORDER BY device_group, channel, id"
        ).fetchall()
    cams = []
    for r in rows:
        st = ai_engine.get_camera_status(r[0])
        cams.append({
            "id": r[0],
            "name": r[1],
            "ip": r[2],
            "brand": r[3],
            "channel": r[4] or 1,
            "stream_profile": r[5] or "sub",
            "device_group": r[6] or r[2],
            "online": st["online"],
            "fps": st["fps"],
        })
    return {"cameras": cams}


@app.post("/api/cameras")
def add_camera(req: CameraCreateRequest, admin: dict = Depends(require_admin)):
    ip_clean = _validate_camera_host(req.ip)
    brand_key = req.brand.lower().strip()
    user_clean = req.cam_user.strip() or "admin"
    pass_clean = req.cam_pass
    stream_profile = "main" if req.stream_profile == "main" else "sub"
    lens_count = max(1, min(8, int(req.lens_count or 1)))

    probe = probe_camera_rtsp(
        ip=ip_clean,
        brand=brand_key,
        cam_user=user_clean,
        cam_pass=pass_clean,
        auto_try_defaults=not bool(pass_clean),
        stream_profile=stream_profile,
    )
    if probe["authenticated"]:
        user_clean = probe["matched_user"]
        pass_clean = probe["matched_pass"]
        brand_key = probe["matched_brand"] or brand_key
        if lens_count == 1 and probe.get("detected_channels", 1) > 1:
            lens_count = probe["detected_channels"]

    base_name = req.name.strip() or f"Camera {ip_clean}"
    device_group = f"dev_{ip_clean}_{int(time.time())}"
    created_ids = []

    with get_db_connection() as conn:
        for ch in range(1, lens_count + 1):
            cam_id = f"cam_{int(time.time() * 1000)}_{ch}_{secrets.token_hex(2)}"
            lens_name = f"{base_name} (Mắt {ch})" if lens_count > 1 else base_name
            rtsp_url = build_rtsp_url(brand_key, ip_clean, user_clean, pass_clean, channel=ch, stream_profile=stream_profile)
            conn.execute(
                "INSERT INTO cameras (id, name, ip, brand, cam_user, cam_pass, rtsp_url, channel, stream_profile, device_group) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (cam_id, lens_name, ip_clean, brand_key, user_clean, pass_clean, rtsp_url, ch, stream_profile, device_group),
            )
            register_mediamtx_path(cam_id, rtsp_url)
            ai_engine.start_camera_monitor(cam_id, lens_name, rtsp_url)
            created_ids.append(cam_id)
        conn.commit()

    return {
        "status": "ok",
        "id": created_ids[0],
        "ids": created_ids,
        "lens_count": lens_count,
        "authenticated": probe["authenticated"],
        "matched_pass": pass_clean if probe["authenticated"] else "",
        "probe_message": probe["message"],
    }


@app.delete("/api/cameras/{cam_id}")
def delete_camera(cam_id: str, admin: dict = Depends(require_admin)):
    with get_db_connection() as conn:
        conn.execute("DELETE FROM cameras WHERE id=?", (cam_id,))
        conn.commit()
    ONVIF_CLIENTS.pop(cam_id, None)
    ai_engine.stop_camera_monitor(cam_id)
    try:
        httpx.delete(f"http://127.0.0.1:9997/v3/config/paths/delete/{cam_id}", timeout=2.0)
    except Exception:
        pass
    return {"status": "ok"}


def get_onvif_ptz(cam_id: str):
    if not HAS_ONVIF or ONVIFCamera is None:
        raise HTTPException(status_code=503, detail="Thư viện ONVIF (onvif-zeep) chưa được cài đặt trên thiết bị")
    if cam_id in ONVIF_CLIENTS:
        return ONVIF_CLIENTS[cam_id]
    with get_db_connection() as conn:
        row = conn.execute("SELECT ip, cam_user, cam_pass FROM cameras WHERE id=?", (cam_id,)).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Không tìm thấy camera")
    ip, user, pwd = row
    for port in [80, 8000, 8899, 2020]:
        try:
            cam = ONVIFCamera(ip, port, user, pwd)
            media = cam.create_media_service()
            ptz = cam.create_ptz_service()
            profiles = media.GetProfiles()
            if not profiles:
                continue
            token = profiles[0].token
            ONVIF_CLIENTS[cam_id] = (ptz, token)
            return ptz, token
        except Exception:
            continue
    raise HTTPException(status_code=400, detail="Camera chưa bật ONVIF hoặc không hỗ trợ xoay PTZ")


@app.post("/api/cameras/{cam_id}/ptz")
def control_ptz(cam_id: str, cmd: PTZCommand, user: dict = Depends(get_current_user)):
    ptz, token = get_onvif_ptz(cam_id)
    try:
        if cmd.direction == "stop":
            ptz.Stop({"ProfileToken": token})
            return {"status": "stopped"}

        req = ptz.create_type("ContinuousMove")
        req.ProfileToken = token
        req.Velocity = {"PanTilt": {"x": 0.0, "y": 0.0}, "Zoom": {"x": 0.0}}

        s = max(0.1, min(1.0, cmd.speed))
        if cmd.direction == "left":
            req.Velocity["PanTilt"]["x"] = -s
        elif cmd.direction == "right":
            req.Velocity["PanTilt"]["x"] = s
        elif cmd.direction == "up":
            req.Velocity["PanTilt"]["y"] = s
        elif cmd.direction == "down":
            req.Velocity["PanTilt"]["y"] = -s
        else:
            raise HTTPException(status_code=400, detail="Hướng điều khiển PTZ không hợp lệ")

        ptz.ContinuousMove(req)

        def _safe_stop():
            try:
                ptz.Stop({"ProfileToken": token})
            except Exception:
                pass

        t = threading.Timer(0.45, _safe_stop)
        t.daemon = True
        t.start()
        return {"status": "ok", "direction": cmd.direction}
    except HTTPException:
        raise
    except Exception:
        ONVIF_CLIENTS.pop(cam_id, None)
        raise HTTPException(status_code=500, detail="Mất kết nối điều khiển ONVIF PTZ")


@app.post("/api/cameras/{cam_id}/talk")
async def push_two_way_audio(cam_id: str, audio: UploadFile = File(...), user: dict = Depends(get_current_user)):
    with get_db_connection() as conn:
        row = conn.execute("SELECT rtsp_url FROM cameras WHERE id=?", (cam_id,)).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Không tìm thấy camera")

    tmp_dir = tempfile.gettempdir()
    os.makedirs(tmp_dir, exist_ok=True)
    temp_webm = os.path.join(tmp_dir, f"talk_{cam_id}_{int(time.time() * 1000)}.webm")

    content = await audio.read()
    with open(temp_webm, "wb") as f:
        f.write(content)

    def _stream_to_speaker():
        try:
            subprocess.run(
                [
                    "ffmpeg", "-y", "-i", temp_webm,
                    "-ar", "8000", "-ac", "1", "-c:a", "pcm_mulaw",
                    "-f", "rtsp", row[0],
                ],
                timeout=12,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception:
            pass
        finally:
            try:
                if os.path.exists(temp_webm):
                    os.remove(temp_webm)
            except OSError:
                pass

    threading.Thread(target=_stream_to_speaker, daemon=True).start()
    return {"status": "ok"}


# --- DEVICE HEALTH, STORAGE, RECORDINGS & ALERTS ---
@app.get("/api/system/health")
def get_device_health(user: dict = Depends(get_current_user)):
    bat_cmd = _find_termux_cmd("termux-battery-status")
    if bat_cmd:
        try:
            raw = subprocess.check_output([bat_cmd], timeout=3, stderr=subprocess.DEVNULL).decode("utf-8")
            info = json.loads(raw)
            return {
                "battery_pct": info.get("percentage", 100),
                "temperature": info.get("temperature", 35.0),
                "status": info.get("status", "OPTIMAL"),
            }
        except Exception:
            pass
    return {"battery_pct": 100, "temperature": 35.0, "status": "OPTIMAL"}


def cleanup_old_recordings(retention_days: int, free_bytes: int):
    cutoff = time.time() - (retention_days * 86400)
    all_files = []

    for folder, ext in [("recordings", ".mp4"), ("snapshots", ".jpg")]:
        for root, _, files in os.walk(folder):
            for f in files:
                if f.endswith(ext):
                    full_path = os.path.join(root, f)
                    try:
                        mtime = os.path.getmtime(full_path)
                        if mtime < cutoff:
                            os.remove(full_path)
                        elif ext == ".mp4":
                            all_files.append((mtime, full_path))
                    except OSError:
                        continue

    if free_bytes < 2 * (1024**3) and all_files:
        all_files.sort()
        for _, old_path in all_files[: max(1, len(all_files) // 5)]:
            try:
                if os.path.exists(old_path):
                    os.remove(old_path)
            except OSError:
                continue


@app.get("/api/storage")
def get_storage_info(user: dict = Depends(get_current_user)):
    days = int(get_setting("retention_days") or 30)
    total, used, free = shutil.disk_usage(".")
    cleanup_old_recordings(days, free)

    total, used, free = shutil.disk_usage(".")
    cam_bytes = 0
    for r, _, fs in os.walk("recordings"):
        for f in fs:
            if f.endswith(".mp4"):
                try:
                    cam_bytes += os.path.getsize(os.path.join(r, f))
                except OSError:
                    continue

    return {
        "total_gb": round(total / (1024**3), 2),
        "used_gb": round(used / (1024**3), 2),
        "free_gb": round(free / (1024**3), 2),
        "camera_used_mb": round(cam_bytes / (1024**2), 2),
        "retention_days": days,
    }


@app.post("/api/storage/retention")
def update_retention(req: RetentionUpdate, admin: dict = Depends(require_admin)):
    days = max(1, min(365, req.days))
    set_setting("retention_days", str(days))
    return {"status": "ok", "retention_days": days}


@app.get("/api/recordings")
def list_recordings(user: dict = Depends(get_current_user)):
    items = []
    for root, _, files in os.walk("recordings"):
        for f in files:
            if f.endswith(".mp4"):
                full_path = os.path.join(root, f)
                try:
                    mtime = os.path.getmtime(full_path)
                    size_mb = round(os.path.getsize(full_path) / (1024**2), 2)
                except OSError:
                    continue
                items.append({
                    "_mtime": mtime,
                    "filename": f,
                    "url": "/" + full_path.replace("\\", "/"),
                    "size_mb": size_mb,
                    "created_at": datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S"),
                })
    items.sort(key=lambda x: x["_mtime"], reverse=True)
    for item in items:
        item.pop("_mtime", None)
    return {"recordings": items}


@app.get("/api/alerts")
def get_alerts(user: dict = Depends(get_current_user)):
    with ALERTS_LOCK:
        return {"alerts": list(ALERTS_LOG)}
