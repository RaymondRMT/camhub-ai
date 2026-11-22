import os
import re
import hmac
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
from typing import Optional, List
from fastapi import FastAPI, HTTPException, Depends, Header, Query, Request, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import BaseModel
import httpx

try:
    import jwt as pyjwt
    HAS_PYJWT = True
except Exception:
    pyjwt = None
    HAS_PYJWT = False

try:
    from onvif import ONVIFCamera
    HAS_ONVIF = True
except Exception:
    ONVIFCamera = None
    HAS_ONVIF = False

from ai_vision import CameraAIProcessor, HAS_MEDIAPIPE, now_vn

os.makedirs("static", exist_ok=True)
os.makedirs("recordings", exist_ok=True)
os.makedirs("snapshots", exist_ok=True)

DB_PATH = os.environ.get("CAMHUB_DB_PATH", "camhub.db")
SECRET_KEY_FILE = os.environ.get("CAMHUB_SECRET_FILE", ".jwt_secret")
ALERTS_LOG = []
ALERTS_LOCK = threading.Lock()
ONVIF_CLIENTS = {}
TUNNEL_STATE = {"url": "", "status": "starting", "mode": "quick", "last_error": ""}
tunnel_process = None
tunnel_lock = threading.Lock()
http_async_client: Optional[httpx.AsyncClient] = None

BRAND_TEMPLATES = {
    "hikvision": "rtsp://{user}:{pwd}@{ip}:554/Streaming/Channels/102",
    "dahua": "rtsp://{user}:{pwd}@{ip}:554/cam/realmonitor?channel=1&subtype=1",
    "imou": "rtsp://{user}:{pwd}@{ip}:554/cam/realmonitor?channel=1&subtype=1",
    "ezviz": "rtsp://admin:{pwd}@{ip}:554/h264_stream",
    "kbvision": "rtsp://{user}:{pwd}@{ip}:554/cam/realmonitor?channel=1&subtype=1",
    "yoosee": "rtsp://{user}:{pwd}@{ip}:554/onvif2",
    "tapo": "rtsp://{user}:{pwd}@{ip}:554/stream2",
    "custom": "rtsp://{user}:{pwd}@{ip}:554/stream1",
}

BRAND_DEFAULT_HINTS = {
    "hikvision": {
        "default_user": "admin",
        "common_passwords": ["12345", "12345678a", "admin"],
        "guide": "Camera đời mới dùng mật khẩu kích hoạt lúc cài đặt hoặc mã Verification Code (6 ký tự in hoa) dưới đáy camera.",
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
        "common_passwords": ["123456", "admin"],
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
        "guide": "Mật khẩu RTSP mặc định là '123' hoặc '888888' (cần bật mục 'Cài đặt RTSP / NVR' trong ứng dụng Yoosee).",
    },
    "tapo": {
        "default_user": "admin",
        "common_passwords": ["123456", "admin"],
        "guide": "Vào ứng dụng Tapo -> Cài đặt Camera -> Nâng cao -> Tài khoản Camera để xem hoặc tạo tài khoản RTSP/ONVIF.",
    },
    "custom": {
        "default_user": "admin",
        "common_passwords": ["admin", "123456", "12345", "888888"],
        "guide": "Thử các mật khẩu mặc định phổ biến của chuẩn ONVIF hoặc chuyển sang chế độ nhập link RTSP thủ công.",
    },
}


# --- JWT HELPERS (PyJWT + RFC 7519 HS256 Fallback) ---
def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64url_decode(data: str) -> bytes:
    padding = "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode(data + padding)


def _jwt_encode_hs256(payload: dict, secret: str) -> str:
    if HAS_PYJWT and pyjwt is not None:
        return pyjwt.encode(payload, secret, algorithm="HS256")
    header_b64 = _b64url_encode(json.dumps({"alg": "HS256", "typ": "JWT"}, separators=(",", ":")).encode("utf-8"))
    norm = dict(payload)
    if isinstance(norm.get("exp"), datetime):
        norm["exp"] = int(norm["exp"].timestamp())
    payload_b64 = _b64url_encode(json.dumps(norm, separators=(",", ":")).encode("utf-8"))
    signing_input = f"{header_b64}.{payload_b64}".encode("ascii")
    sig = hmac.new(secret.encode("utf-8"), signing_input, hashlib.sha256).digest()
    return f"{header_b64}.{payload_b64}.{_b64url_encode(sig)}"


def _jwt_decode_hs256(token: str, secret: str) -> dict:
    if HAS_PYJWT and pyjwt is not None:
        return pyjwt.decode(token, secret, algorithms=["HS256"])
    parts = token.split(".")
    if len(parts) != 3:
        raise ValueError("Invalid token")
    signing_input = f"{parts[0]}.{parts[1]}".encode("ascii")
    expected_sig = hmac.new(secret.encode("utf-8"), signing_input, hashlib.sha256).digest()
    actual_sig = _b64url_decode(parts[2])
    if not hmac.compare_digest(expected_sig, actual_sig):
        raise ValueError("Invalid signature")
    payload = json.loads(_b64url_decode(parts[1]).decode("utf-8"))
    exp = payload.get("exp")
    if exp is not None and time.time() > float(exp):
        raise ValueError("Token expired")
    return payload


def get_db_connection() -> sqlite3.Connection:
    return sqlite3.connect(DB_PATH, timeout=15.0, check_same_thread=False)


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
            rtsp_url TEXT
        )""")
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


def normalize_domain_url(raw: str) -> str:
    cleaned = raw.strip().rstrip("/")
    if not cleaned:
        return ""
    if not re.match(r"^https?://", cleaned, flags=re.IGNORECASE):
        cleaned = "https://" + cleaned
    return cleaned


def parse_telegram_chat_ids(raw: str) -> List[str]:
    parts = re.split(r"[,;\n\s]+", raw.strip())
    seen = []
    for p in parts:
        p = p.strip()
        if p and p not in seen:
            seen.append(p)
    return seen


def mask_rtsp_url(rtsp_url: str) -> str:
    return re.sub(r"(rtsp://[^:@/]+:)([^@/]+)(@)", r"\1****\3", rtsp_url)


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
            (username, status, ip, ua, now_vn().strftime("%Y-%m-%d %H:%M:%S")),
        )
        conn.execute("DELETE FROM login_logs WHERE id NOT IN (SELECT id FROM login_logs ORDER BY id DESC LIMIT 100)")
        conn.commit()


# --- TELEGRAM BOT & FAMILY GROUP ALERT ---
def send_telegram_alert(alert_data: dict):
    bot_token = get_setting("tg_bot_token").strip()
    chat_ids = parse_telegram_chat_ids(get_setting("tg_chat_id"))
    if not bot_token or not chat_ids:
        return

    alert_type = alert_data.get("type")
    if alert_type == "motion" and get_setting("tg_alert_motion") != "1":
        return
    if alert_type == "fall" and get_setting("tg_alert_fall") != "1":
        return

    snap_rel = alert_data.get("snapshot", "").lstrip("/")
    prefix = "🚨 [KHẨN CẤP - PHÁT HIỆN TÉ NGÃ]" if alert_type == "fall" else "👁️ [CHUYỂN ĐỘNG]"
    remote_url = TUNNEL_STATE.get("url", "")
    link_line = f"\n🔗 Xem trực tiếp: {remote_url}" if remote_url.startswith("http") else ""
    caption = f"{prefix}\n{alert_data['message']}\n🕒 Thời gian: {alert_data['timestamp']}{link_line}"

    details = alert_data.get("detail_snapshots") or []

    def _worker():
        media_items = []
        if snap_rel and os.path.exists(snap_rel):
            try:
                with open(snap_rel, "rb") as f:
                    media_items.append(("overview.jpg", f.read(), caption))
            except OSError:
                pass

        for idx, d in enumerate(details[:4], start=1):
            d_rel = str(d.get("url", "")).lstrip("/")
            if d_rel and os.path.exists(d_rel):
                try:
                    with open(d_rel, "rb") as f:
                        media_items.append((f"detail_{idx}.jpg", f.read(), d.get("label", "Chi tiết phóng to")))
                except OSError:
                    continue

        for cid in chat_ids:
            try:
                if len(media_items) >= 2:
                    tg_media = []
                    files_payload = {}
                    for i, (fname, raw_b, item_cap) in enumerate(media_items):
                        attach_key = f"photo_{i}"
                        entry = {"type": "photo", "media": f"attach://{attach_key}"}
                        if i == 0:
                            entry["caption"] = item_cap
                        else:
                            entry["caption"] = item_cap
                        tg_media.append(entry)
                        files_payload[attach_key] = (fname, raw_b, "image/jpeg")
                    resp = httpx.post(
                        f"https://api.telegram.org/bot{bot_token}/sendMediaGroup",
                        data={"chat_id": cid, "media": json.dumps(tg_media, ensure_ascii=False)},
                        files=files_payload,
                        timeout=18.0,
                    )
                    if resp.status_code == 200:
                        continue
                if len(media_items) >= 1:
                    fname, raw_b, _ = media_items[0]
                    httpx.post(
                        f"https://api.telegram.org/bot{bot_token}/sendPhoto",
                        data={"chat_id": cid, "caption": caption},
                        files={"photo": (fname, raw_b, "image/jpeg")},
                        timeout=12.0,
                    )
                else:
                    httpx.post(
                        f"https://api.telegram.org/bot{bot_token}/sendMessage",
                        data={"chat_id": cid, "text": caption},
                        timeout=10.0,
                    )
            except Exception:
                continue

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


# --- MEDIAMTX & CLOUDFLARE TUNNEL ---
def check_mediamtx_alive() -> bool:
    try:
        r = httpx.get("http://127.0.0.1:9997/v3/paths/list", timeout=1.5)
        return r.status_code == 200
    except Exception:
        return False


def register_mediamtx_path(cam_id: str, rtsp_url: str):
    payload = {"source": rtsp_url, "record": True}
    try:
        resp = httpx.post(
            f"http://127.0.0.1:9997/v3/config/paths/add/{cam_id}",
            json=payload,
            timeout=3.0,
        )
        if resp.status_code == 400:
            httpx.patch(
                f"http://127.0.0.1:9997/v3/config/paths/patch/{cam_id}",
                json=payload,
                timeout=3.0,
            )
    except Exception:
        pass


def sync_cameras_on_boot():
    with get_db_connection() as conn:
        rows = conn.execute("SELECT id, name, brand, rtsp_url FROM cameras").fetchall()
    for cam_id, name, brand, rtsp_url in rows:
        if (brand or "").lower() != "yoosee" and "/onvif" not in (rtsp_url or "").lower():
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
        custom_domain = normalize_domain_url(get_setting("cf_custom_domain"))

        try:
            if cf_token:
                TUNNEL_STATE = {
                    "url": custom_domain or "Named Tunnel đang chạy",
                    "status": "running",
                    "mode": "named",
                    "last_error": "",
                }
                cmd = ["cloudflared", "tunnel", "--no-autoupdate", "run", "--token", cf_token]
                tunnel_process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                )

                def _watch_named(proc):
                    if not proc.stdout:
                        return
                    for line in iter(proc.stdout.readline, ""):
                        lower = line.lower()
                        if "err" in lower and ("failed" in lower or "invalid" in lower or "unauthorized" in lower):
                            TUNNEL_STATE["last_error"] = line.strip()[:180]
                    if proc.poll() not in (None, 0):
                        TUNNEL_STATE["status"] = "error"

                threading.Thread(target=_watch_named, args=(tunnel_process,), daemon=True).start()
            else:
                TUNNEL_STATE = {"url": "", "status": "starting", "mode": "quick", "last_error": ""}
                cmd = ["cloudflared", "tunnel", "--no-autoupdate", "--url", "http://127.0.0.1:8000"]
                tunnel_process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                )

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
            TUNNEL_STATE = {
                "url": custom_domain if cf_token else "",
                "status": "unavailable",
                "mode": "named" if cf_token else "none",
                "last_error": "Chưa cài đặt cloudflared trên thiết bị",
            }
        except Exception as e:
            TUNNEL_STATE = {"url": "", "status": "error", "mode": "none", "last_error": str(e)}


@asynccontextmanager
async def lifespan(app_instance: FastAPI):
    global http_async_client
    http_async_client = httpx.AsyncClient(timeout=10.0)
    if os.environ.get("CAMHUB_DISABLE_TUNNEL") != "1":
        start_cloudflare_tunnel()
    boot_timer = threading.Timer(2.0, sync_cameras_on_boot)
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
def decode_bearer_or_query_token(authorization: Optional[str], token: Optional[str]) -> dict:
    raw = None
    if authorization and authorization.startswith("Bearer "):
        raw = authorization.split(" ", 1)[1].strip()
    elif token:
        raw = token.strip()
    if not raw:
        raise HTTPException(status_code=401, detail="Chưa đăng nhập")
    try:
        return _jwt_decode_hs256(raw, JWT_SECRET)
    except Exception:
        raise HTTPException(status_code=401, detail="Phiên đăng nhập hết hạn")


def get_current_user(authorization: Optional[str] = Header(None)):
    return decode_bearer_or_query_token(authorization, None)


def get_stream_user(authorization: Optional[str] = Header(None), token: Optional[str] = Query(None)):
    return decode_bearer_or_query_token(authorization, token)


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
    return _jwt_encode_hs256(payload, JWT_SECRET)


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


class ResetMemberPasswordRequest(BaseModel):
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
    ip: str = ""
    brand: str = "imou"
    cam_user: str = "admin"
    cam_pass: str = ""
    custom_rtsp_url: str = ""


class CameraProbeRequest(BaseModel):
    ip: str = ""
    brand: str = "imou"
    cam_user: str = "admin"
    cam_pass: str = ""
    custom_rtsp_url: str = ""


class PTZCommand(BaseModel):
    direction: str
    speed: float = 0.6
    duration: float = 0.45


class RetentionUpdate(BaseModel):
    days: int


class TestFallRequest(BaseModel):
    cam_id: Optional[str] = None


# --- HLS REVERSE PROXY & UI ---
@app.get("/hls/{path:path}")
async def proxy_hls(path: str):
    if ".." in path or path.startswith("/"):
        raise HTTPException(status_code=400, detail="Đường dẫn HLS không hợp lệ")
    client = http_async_client
    try:
        if client is not None:
            resp = await client.get(f"http://127.0.0.1:8888/{path}")
        else:
            async with httpx.AsyncClient(timeout=10.0) as temp_client:
                resp = await temp_client.get(f"http://127.0.0.1:8888/{path}")
        return Response(
            content=resp.content,
            status_code=resp.status_code,
            media_type=resp.headers.get("content-type", "application/vnd.apple.mpegurl"),
        )
    except httpx.RequestError:
        raise HTTPException(status_code=502, detail="Luồng phát HLS (MediaMTX) chưa sẵn sàng")


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
                (username, display_name, pwd_hash, now_vn().strftime("%Y-%m-%d %H:%M")),
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


@app.get("/api/setup/checklist")
def get_setup_checklist(admin: dict = Depends(require_admin)):
    with get_db_connection() as conn:
        has_admin = conn.execute("SELECT COUNT(*) FROM users WHERE role='admin'").fetchone()[0] > 0
        cam_count = conn.execute("SELECT COUNT(*) FROM cameras").fetchone()[0]
        member_count = conn.execute("SELECT COUNT(*) FROM users WHERE role='member'").fetchone()[0]
    has_tg = bool(get_setting("tg_bot_token").strip() and parse_telegram_chat_ids(get_setting("tg_chat_id")))
    has_fixed = bool(get_setting("cf_token").strip() and get_setting("cf_custom_domain").strip())
    has_tunnel = bool(TUNNEL_STATE.get("url", "").startswith("http") or has_fixed)
    return {
        "has_admin": has_admin,
        "camera_count": cam_count,
        "has_cameras": cam_count > 0,
        "has_telegram": has_tg,
        "has_tunnel": has_tunnel,
        "has_fixed_domain": has_fixed,
        "member_count": member_count,
        "has_family_members": member_count > 0,
        "ai_engine": "MediaPipe Pose" if HAS_MEDIAPIPE else "OpenCV Contour Fallback",
        "mediamtx_ok": check_mediamtx_alive(),
    }


# --- TELEGRAM CONFIG ---
@app.get("/api/telegram/config")
def get_telegram_config(admin: dict = Depends(require_admin)):
    return {
        "bot_token": get_setting("tg_bot_token"),
        "chat_id": get_setting("tg_chat_id"),
        "chat_id_count": len(parse_telegram_chat_ids(get_setting("tg_chat_id"))),
        "alert_motion": get_setting("tg_alert_motion") == "1",
        "alert_fall": get_setting("tg_alert_fall") == "1",
    }


@app.post("/api/telegram/config")
def save_telegram_config(req: TelegramConfigRequest, admin: dict = Depends(require_admin)):
    set_setting("tg_bot_token", req.bot_token.strip())
    set_setting("tg_chat_id", req.chat_id.strip())
    set_setting("tg_alert_motion", "1" if req.alert_motion else "0")
    set_setting("tg_alert_fall", "1" if req.alert_fall else "0")
    return {"status": "ok", "targets": parse_telegram_chat_ids(req.chat_id)}


@app.post("/api/telegram/test")
def test_telegram_alert(admin: dict = Depends(require_admin)):
    bot_token = get_setting("tg_bot_token").strip()
    chat_ids = parse_telegram_chat_ids(get_setting("tg_chat_id"))
    if not bot_token or not chat_ids:
        raise HTTPException(status_code=400, detail="Chưa nhập Bot Token hoặc Chat ID / Group ID")

    remote_url = TUNNEL_STATE.get("url", "")
    link_part = f"\n🔗 Truy cập từ xa: {remote_url}" if remote_url.startswith("http") else ""
    msg = (
        "✅ CamHub AI Pro: Kết nối Telegram Bot & Nhóm Gia đình thành công!\n"
        "Hệ thống đã sẵn sàng gửi ảnh cảnh báo té ngã & chuyển động." + link_part
    )

    sent = []
    failed = []
    for cid in chat_ids:
        try:
            resp = httpx.post(
                f"https://api.telegram.org/bot{bot_token}/sendMessage",
                data={"chat_id": cid, "text": msg},
                timeout=10.0,
            )
            if resp.status_code == 200:
                sent.append(cid)
            else:
                failed.append(cid)
        except httpx.RequestError:
            failed.append(cid)

    if not sent:
        raise HTTPException(status_code=400, detail="Không gửi được tới Telegram. Kiểm tra lại Bot Token, Chat ID/Group ID và kết nối mạng.")
    return {"status": "ok", "sent": sent, "failed": failed}


# --- FAMILY & TUNNEL ---
@app.get("/api/family")
def list_family(admin: dict = Depends(require_admin)):
    with get_db_connection() as conn:
        rows = conn.execute("SELECT id, username, display_name, role, created_at FROM users ORDER BY id ASC").fetchall()
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
                (username, display_name, hash_password(req.password), now_vn().strftime("%Y-%m-%d %H:%M")),
            )
            conn.commit()
            return {"status": "ok"}
        except sqlite3.IntegrityError:
            raise HTTPException(status_code=400, detail="Tên đăng nhập đã tồn tại")


@app.put("/api/family/{member_id}/password")
def reset_family_password(member_id: int, req: ResetMemberPasswordRequest, admin: dict = Depends(require_admin)):
    if len(req.new_password) < 4:
        raise HTTPException(status_code=400, detail="Mật khẩu mới phải có ít nhất 4 ký tự")
    with get_db_connection() as conn:
        cur = conn.execute(
            "UPDATE users SET password_hash=? WHERE id=? AND role='member'",
            (hash_password(req.new_password), member_id),
        )
        conn.commit()
        if cur.rowcount == 0:
            raise HTTPException(status_code=404, detail="Không tìm thấy thành viên gia đình")
    return {"status": "ok"}


@app.delete("/api/family/{member_id}")
def delete_family_member(member_id: int, admin: dict = Depends(require_admin)):
    with get_db_connection() as conn:
        conn.execute("DELETE FROM users WHERE id=? AND role='member'", (member_id,))
        conn.commit()
    return {"status": "ok"}


@app.get("/api/tunnel")
def get_tunnel_info(admin: dict = Depends(require_admin)):
    return {
        "url": TUNNEL_STATE.get("url", ""),
        "status": TUNNEL_STATE.get("status", "starting"),
        "mode": TUNNEL_STATE.get("mode", "quick"),
        "custom_domain": get_setting("cf_custom_domain"),
        "has_token": bool(get_setting("cf_token").strip()),
        "last_error": TUNNEL_STATE.get("last_error", ""),
    }


@app.post("/api/tunnel/config")
def update_tunnel_config(req: TunnelConfigRequest, admin: dict = Depends(require_admin)):
    token_clean = req.cf_token.strip()
    domain_clean = normalize_domain_url(req.custom_domain)
    set_setting("cf_token", token_clean)
    set_setting("cf_custom_domain", domain_clean)
    if os.environ.get("CAMHUB_DISABLE_TUNNEL") != "1":
        start_cloudflare_tunnel()
    else:
        TUNNEL_STATE["mode"] = "named" if token_clean else "quick"
        TUNNEL_STATE["url"] = domain_clean if token_clean else ""
        TUNNEL_STATE["status"] = "running" if token_clean else "starting"
    return {"status": "ok", "mode": TUNNEL_STATE.get("mode"), "custom_domain": domain_clean}


@app.delete("/api/tunnel/config")
def clear_tunnel_config(admin: dict = Depends(require_admin)):
    set_setting("cf_token", "")
    set_setting("cf_custom_domain", "")
    if os.environ.get("CAMHUB_DISABLE_TUNNEL") != "1":
        start_cloudflare_tunnel()
    else:
        TUNNEL_STATE["mode"] = "quick"
        TUNNEL_STATE["url"] = ""
        TUNNEL_STATE["status"] = "starting"
    return {"status": "ok", "mode": "quick"}


@app.post("/api/tunnel/restart")
def restart_tunnel(admin: dict = Depends(require_admin)):
    if os.environ.get("CAMHUB_DISABLE_TUNNEL") != "1":
        start_cloudflare_tunnel()
    return {"status": "ok"}


# --- WIFI, CAMERA, PTZ ONVIF & TWO-WAY AUDIO ---
@app.get("/api/wifi/scan")
def scan_wifi(admin: dict = Depends(require_admin)):
    try:
        output = subprocess.check_output(
            ["termux-wifi-scaninfo"],
            timeout=8,
            stderr=subprocess.DEVNULL,
        ).decode("utf-8")
        networks = json.loads(output)
        seen = {}
        if isinstance(networks, list):
            for net in networks:
                ssid = str(net.get("ssid", "")).strip()
                rssi = int(net.get("rssi", -100))
                if ssid and (ssid not in seen or rssi > seen[ssid]["rssi"]):
                    seen[ssid] = {"ssid": ssid, "rssi": rssi}
        if seen:
            return {"networks": sorted(seen.values(), key=lambda x: x["rssi"], reverse=True), "source": "termux-api"}
    except Exception:
        pass
    return {"networks": [], "source": "unavailable", "message": "Cần cài ứng dụng Termux:API trên Android để quét Wi-Fi trực tiếp."}


@app.post("/api/wifi/connect")
def connect_wifi(req: WifiConnectRequest, admin: dict = Depends(require_admin)):
    ssid = req.ssid.strip()
    if not ssid:
        raise HTTPException(status_code=400, detail="Vui lòng chọn mạng Wi-Fi")
    try:
        subprocess.run(["termux-wifi-enable", "true"], timeout=5, check=False)
        return {"status": "ok", "message": f"Đã bật Wi-Fi cho mạng {ssid}. Nếu Android chặn tự kết nối, hãy chọn mạng này trong Cài đặt Wi-Fi của máy."}
    except Exception as e:
        return {"status": "warning", "message": str(e)}


@app.get("/api/cameras/brand-hints")
def get_camera_brand_hints(user: dict = Depends(get_current_user)):
    return {"hints": BRAND_DEFAULT_HINTS}


def _detect_local_subnet() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = "192.168.1.100"
    finally:
        s.close()
    return ".".join(ip.split(".")[:3])


@app.get("/api/cameras/discover")
def discover_cameras(admin: dict = Depends(require_admin)):
    subnet = _detect_local_subnet()
    ips = [f"{subnet}.{i}" for i in range(1, 255)]
    found = []

    def _scan(target):
        rtsp_open = []
        for port in (554, 8554):
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(0.28)
            try:
                if sock.connect_ex((target, port)) == 0:
                    rtsp_open.append(port)
            finally:
                sock.close()
        if not rtsp_open:
            return None
        onvif_open = []
        for port in (80, 8000, 8899, 2020):
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(0.22)
            try:
                if sock.connect_ex((target, port)) == 0:
                    onvif_open.append(port)
            finally:
                sock.close()
        return {"ip": target, "rtsp_ports": rtsp_open, "onvif_ports": onvif_open}

    with concurrent.futures.ThreadPoolExecutor(max_workers=60) as ex:
        for res in ex.map(_scan, ips):
            if res:
                found.append(res)
    return {"subnet": f"{subnet}.0/24", "devices": found}


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


def _build_camera_rtsp(ip: str, brand: str, cam_user: str, cam_pass: str, custom_rtsp_url: str) -> tuple[str, str, str]:
    custom = custom_rtsp_url.strip()
    brand_key = (brand or "custom").lower().strip()
    if custom:
        if not re.match(r"^rtsps?://", custom, flags=re.IGNORECASE):
            raise HTTPException(status_code=400, detail="Link RTSP thủ công phải bắt đầu bằng rtsp:// hoặc rtsps://")
        parsed = urllib.parse.urlparse(custom)
        host = parsed.hostname or ip.strip() or "custom-rtsp"
        return host, brand_key or "custom", custom

    ip_clean = _validate_camera_host(ip)
    template = BRAND_TEMPLATES.get(brand_key, BRAND_TEMPLATES["custom"])
    safe_user = urllib.parse.quote((cam_user or "admin").strip(), safe="")
    safe_pass = urllib.parse.quote(cam_pass or "", safe="")
    rtsp_url = template.format(user=safe_user, pwd=safe_pass, ip=ip_clean)
    return ip_clean, brand_key, rtsp_url


@app.post("/api/cameras/probe")
def probe_camera(req: CameraProbeRequest, admin: dict = Depends(require_admin)):
    host, _, rtsp_url = _build_camera_rtsp(req.ip, req.brand, req.cam_user, req.cam_pass, req.custom_rtsp_url)
    parsed = urllib.parse.urlparse(rtsp_url)
    target_host = parsed.hostname or host
    target_port = parsed.port or 554

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(2.0)
    try:
        tcp_ok = sock.connect_ex((target_host, target_port)) == 0
    except Exception:
        tcp_ok = False
    finally:
        sock.close()

    if not tcp_ok:
        return {
            "reachable": False,
            "rtsp_masked": mask_rtsp_url(rtsp_url),
            "message": f"Không thể kết nối tới {target_host}:{target_port}. Kiểm tra IP camera, nguồn điện và cùng mạng Wi-Fi.",
        }

    return {
        "reachable": True,
        "rtsp_masked": mask_rtsp_url(rtsp_url),
        "message": f"Cổng RTSP {target_host}:{target_port} đang mở và sẵn sàng kết nối!",
    }


@app.get("/api/cameras")
def list_cameras(user: dict = Depends(get_current_user)):
    with get_db_connection() as conn:
        rows = conn.execute("SELECT id, name, ip, brand, cam_user, rtsp_url FROM cameras").fetchall()
    result = []
    for r in rows:
        st = ai_engine.get_camera_status(r[0])
        result.append({
            "id": r[0],
            "name": r[1],
            "ip": r[2],
            "brand": r[3],
            "cam_user": r[4],
            "rtsp_masked": mask_rtsp_url(r[5] or ""),
            "online": st.get("online", False),
            "ai_engine": st.get("engine", "contour"),
            "lens_count": st.get("lens_count", 1),
        })
    return {"cameras": result}


@app.post("/api/cameras")
def add_camera(req: CameraCreateRequest, admin: dict = Depends(require_admin)):
    ip_clean, brand_key, rtsp_url = _build_camera_rtsp(
        req.ip, req.brand, req.cam_user, req.cam_pass, req.custom_rtsp_url
    )
    cam_id = f"cam_{int(time.time() * 1000)}_{secrets.token_hex(2)}"
    cam_name = req.name.strip() or f"Camera {ip_clean}"
    with get_db_connection() as conn:
        conn.execute(
            "INSERT INTO cameras VALUES (?, ?, ?, ?, ?, ?, ?)",
            (cam_id, cam_name, ip_clean, brand_key, req.cam_user.strip() or "admin", req.cam_pass, rtsp_url),
        )
        conn.commit()
    if brand_key != "yoosee" and "/onvif" not in rtsp_url.lower():
        register_mediamtx_path(cam_id, rtsp_url)
    ai_engine.start_camera_monitor(cam_id, cam_name, rtsp_url)
    return {"status": "ok", "id": cam_id, "rtsp_masked": mask_rtsp_url(rtsp_url)}


@app.put("/api/cameras/{cam_id}")
def update_camera(cam_id: str, req: CameraCreateRequest, admin: dict = Depends(require_admin)):
    with get_db_connection() as conn:
        existing = conn.execute("SELECT cam_pass FROM cameras WHERE id=?", (cam_id,)).fetchone()
        if not existing:
            raise HTTPException(status_code=404, detail="Không tìm thấy camera")
        effective_pass = req.cam_pass if req.cam_pass != "" else existing[0]
        ip_clean, brand_key, rtsp_url = _build_camera_rtsp(
            req.ip, req.brand, req.cam_user, effective_pass, req.custom_rtsp_url
        )
        cam_name = req.name.strip() or f"Camera {ip_clean}"
        conn.execute(
            "UPDATE cameras SET name=?, ip=?, brand=?, cam_user=?, cam_pass=?, rtsp_url=? WHERE id=?",
            (cam_name, ip_clean, brand_key, req.cam_user.strip() or "admin", effective_pass, rtsp_url, cam_id),
        )
        conn.commit()
    ONVIF_CLIENTS.pop(cam_id, None)
    if brand_key != "yoosee" and "/onvif" not in rtsp_url.lower():
        register_mediamtx_path(cam_id, rtsp_url)
    ai_engine.start_camera_monitor(cam_id, cam_name, rtsp_url)
    return {"status": "ok", "id": cam_id}


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


@app.get("/api/cameras/{cam_id}/snapshot")
def camera_snapshot(cam_id: str, lens: int = Query(0, ge=0, le=4), user: dict = Depends(get_stream_user)):
    with get_db_connection() as conn:
        row = conn.execute("SELECT name FROM cameras WHERE id=?", (cam_id,)).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Không tìm thấy camera")
    frame = ai_engine.get_latest_frame(cam_id, lens=lens)
    if not frame:
        frame = ai_engine.build_placeholder_jpeg(row[0], "Dang cho tin hieu RTSP tu camera...")
    return Response(
        content=frame,
        media_type="image/jpeg",
        headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0", "Pragma": "no-cache"},
    )


@app.get("/api/cameras/{cam_id}/mjpeg")
def camera_mjpeg_stream(cam_id: str, lens: int = Query(0, ge=0, le=4), user: dict = Depends(get_stream_user)):
    with get_db_connection() as conn:
        row = conn.execute("SELECT name FROM cameras WHERE id=?", (cam_id,)).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Không tìm thấy camera")
    cam_name = row[0]

    def _gen():
        placeholder = ai_engine.build_placeholder_jpeg(cam_name, "Dang ket noi luong RTSP...")
        while True:
            frame = ai_engine.get_latest_frame(cam_id, lens=lens) or placeholder
            header = (
                f"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: {len(frame)}\r\n\r\n"
            ).encode("ascii")
            yield header + frame + b"\r\n"
            time.sleep(0.25)

    return StreamingResponse(
        _gen(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "X-Accel-Buffering": "no",
        },
    )


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
    raise HTTPException(status_code=400, detail="Camera chưa bật ONVIF hoặc không hỗ trợ xoay/zoom quang học PTZ")


VALID_PTZ_DIRECTIONS = {"left", "right", "up", "down", "zoom_in", "zoom_out", "home", "stop"}


@app.post("/api/cameras/{cam_id}/ptz")
def control_ptz(cam_id: str, cmd: PTZCommand, user: dict = Depends(get_current_user)):
    direction = cmd.direction.strip().lower()
    if direction not in VALID_PTZ_DIRECTIONS:
        raise HTTPException(status_code=400, detail="Lệnh điều khiển PTZ không hợp lệ")

    ptz, token = get_onvif_ptz(cam_id)
    try:
        if direction == "stop":
            ptz.Stop({"ProfileToken": token, "PanTilt": True, "Zoom": True})
            return {"status": "stopped"}

        if direction == "home":
            try:
                ptz.GotoHomePosition({"ProfileToken": token})
                return {"status": "ok", "direction": "home"}
            except Exception:
                raise HTTPException(status_code=400, detail="Camera không hỗ trợ vị trí gốc (Home Position)")

        req = ptz.create_type("ContinuousMove")
        req.ProfileToken = token
        req.Velocity = {"PanTilt": {"x": 0.0, "y": 0.0}, "Zoom": {"x": 0.0}}

        s = max(0.1, min(1.0, cmd.speed))
        if direction == "left":
            req.Velocity["PanTilt"]["x"] = -s
        elif direction == "right":
            req.Velocity["PanTilt"]["x"] = s
        elif direction == "up":
            req.Velocity["PanTilt"]["y"] = s
        elif direction == "down":
            req.Velocity["PanTilt"]["y"] = -s
        elif direction == "zoom_in":
            req.Velocity["Zoom"]["x"] = s
        elif direction == "zoom_out":
            req.Velocity["Zoom"]["x"] = -s

        ptz.ContinuousMove(req)

        dur = max(0.15, min(2.0, cmd.duration))

        def _safe_stop():
            try:
                ptz.Stop({"ProfileToken": token, "PanTilt": True, "Zoom": True})
            except Exception:
                pass

        t = threading.Timer(dur, _safe_stop)
        t.daemon = True
        t.start()
        return {"status": "ok", "direction": direction, "speed": s}
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

_NET_PREV = {"ts": 0.0, "rx": 0, "tx": 0}
_NET_LOCK = threading.Lock()


def _read_net_speed_and_ping() -> dict:
    rx_bytes, tx_bytes = 0, 0
    got_counters = False
    try:
        import psutil
        io = psutil.net_io_counters()
        rx_bytes, tx_bytes = int(io.bytes_recv), int(io.bytes_sent)
        got_counters = True
    except Exception:
        try:
            with open("/proc/net/dev", "r", encoding="utf-8") as f:
                for line in f.readlines()[2:]:
                    if ":" not in line:
                        continue
                    iface, rest = line.split(":", 1)
                    if iface.strip() == "lo":
                        continue
                    cols = rest.split()
                    if len(cols) >= 9:
                        rx_bytes += int(cols[0])
                        tx_bytes += int(cols[8])
                        got_counters = True
        except Exception:
            pass

    down_kbps = 0.0
    up_kbps = 0.0
    if got_counters:
        now = time.time()
        with _NET_LOCK:
            prev_ts = _NET_PREV["ts"]
            if prev_ts > 0 and now > prev_ts:
                dt = now - prev_ts
                down_kbps = max(0.0, ((rx_bytes - _NET_PREV["rx"]) * 8.0) / (dt * 1000.0))
                up_kbps = max(0.0, ((tx_bytes - _NET_PREV["tx"]) * 8.0) / (dt * 1000.0))
            _NET_PREV["ts"] = now
            _NET_PREV["rx"] = rx_bytes
            _NET_PREV["tx"] = tx_bytes

    ping_ms = None
    if os.environ.get("CAMHUB_DISABLE_TUNNEL") != "1":
        t0 = time.perf_counter()
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(1.5)
        try:
            s.connect(("1.1.1.1", 443))
            ping_ms = round((time.perf_counter() - t0) * 1000.0, 1)
        except Exception:
            ping_ms = None
        finally:
            s.close()

    return {
        "net_down_kbps": round(down_kbps, 1),
        "net_up_kbps": round(up_kbps, 1),
        "ping_ms": ping_ms,
    }


def _read_cpu_ram_gpu() -> dict:
    cpu_pct = 0.0
    ram_pct = 0.0
    cores = os.cpu_count() or 1
    try:
        import psutil
        cpu_pct = round(psutil.cpu_percent(interval=0.1), 1)
        ram_pct = round(psutil.virtual_memory().percent, 1)
    except Exception:
        try:
            load1 = os.getloadavg()[0]
            cpu_pct = round(min(100.0, (load1 / max(1, cores)) * 100.0), 1)
        except Exception:
            pass
        try:
            mem = {}
            with open("/proc/meminfo", "r", encoding="utf-8") as f:
                for line in f:
                    k, v = line.split(":", 1)
                    mem[k.strip()] = int(v.strip().split()[0])
            if mem.get("MemTotal"):
                used_kb = mem["MemTotal"] - mem.get("MemAvailable", mem.get("MemFree", 0))
                ram_pct = round((used_kb / mem["MemTotal"]) * 100.0, 1)
        except Exception:
            pass

    gpu_name = "ARM SoC GPU"
    gpu_pct = None
    # Đọc thông số GPU Adreno / Mali trên Android nếu kernel cho phép
    for path in (
        "/sys/class/kgsl/kgsl-3d0/gpu_model",
        "/sys/devices/platform/soc/3d00000.qcom,kgsl-3d0/kgsl/kgsl-3d0/gpu_model",
    ):
        try:
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    gpu_name = f.read().strip()[:28] or gpu_name
                    break
        except Exception:
            pass
    for path in (
        "/sys/class/kgsl/kgsl-3d0/gpu_busy_percentage",
        "/sys/class/misc/mali0/device/utilization",
    ):
        try:
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    val = re.findall(r"\d+", f.read())
                    if val:
                        gpu_pct = int(val[0])
                        break
        except Exception:
            pass
    return {
        "cpu_pct": cpu_pct,
        "cpu_cores": cores,
        "ram_pct": ram_pct,
        "gpu_name": gpu_name,
        "gpu_pct": gpu_pct,
    }


@app.get("/api/system/health")
def get_device_health(user: dict = Depends(get_current_user)):
    sys_info = {**_read_cpu_ram_gpu(), **_read_net_speed_and_ping()}
    try:
        raw = subprocess.check_output(
            ["termux-battery-status"],
            timeout=2,
            stderr=subprocess.DEVNULL,
        ).decode("utf-8")
        info = json.loads(raw)
        return {
            "battery_pct": info.get("percentage", 100),
            "temperature": info.get("temperature", 35.0),
            "status": info.get("status", "OPTIMAL"),
            "ai_engine": ai_engine.active_engine_label,
            **sys_info,
        }
    except Exception:
        temp = ai_engine._read_device_temp() or 35.0
        return {
            "battery_pct": 100,
            "temperature": round(temp, 1),
            "status": "OPTIMAL",
            "ai_engine": ai_engine.active_engine_label,
            **sys_info,
        }


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
    from ai_vision import VN_TZ
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
                lower = f.lower()
                if lower.startswith("fall_"):
                    category, label = "fall", "Té ngã (~60s)"
                elif lower.startswith("motion_"):
                    category, label = "motion", "Chuyển động (~60s)"
                else:
                    category, label = "continuous", "Video dài (30p)"
                items.append({
                    "_mtime": mtime,
                    "filename": f,
                    "category": category,
                    "label": label,
                    "url": "/" + full_path.replace("\\", "/"),
                    "size_mb": size_mb,
                    "created_at": datetime.fromtimestamp(mtime, VN_TZ).strftime("%Y-%m-%d %H:%M:%S"),
                })
    items.sort(key=lambda x: x["_mtime"], reverse=True)
    for item in items:
        item.pop("_mtime", None)
    return {"recordings": items}


@app.get("/api/alerts")
def get_alerts(user: dict = Depends(get_current_user)):
    with ALERTS_LOCK:
        return {"alerts": list(ALERTS_LOG)}


@app.post("/api/alerts/test-fall")
def trigger_test_fall(req: TestFallRequest, admin: dict = Depends(require_admin)):
    cam_id = req.cam_id or "demo_cam"
    cam_name = "Camera Thử Nghiệm"
    with get_db_connection() as conn:
        if req.cam_id:
            row = conn.execute("SELECT id, name FROM cameras WHERE id=?", (req.cam_id,)).fetchone()
        else:
            row = conn.execute("SELECT id, name FROM cameras LIMIT 1").fetchone()
        if row:
            cam_id, cam_name = row[0], row[1]
    alert = ai_engine.trigger_test_fall_alert(cam_id, cam_name)
    return {"status": "ok", "alert": alert}
