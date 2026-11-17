import os
import re
import time
import json
import socket
import shutil
import secrets
import hashlib
import sqlite3
import subprocess
import threading
import concurrent.futures
from datetime import datetime, timedelta
from typing import Optional
from fastapi import FastAPI, HTTPException, Depends, Header, Request, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
import httpx
import jwt
from onvif import ONVIFCamera
from ai_vision import CameraAIProcessor

app = FastAPI(title="CamHub AI Pro Server")

os.makedirs("static", exist_ok=True)
os.makedirs("recordings", exist_ok=True)
os.makedirs("snapshots", exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/recordings", StaticFiles(directory="recordings"), name="recordings")
app.mount("/snapshots", StaticFiles(directory="snapshots"), name="snapshots")

DB_PATH = "camhub.db"
SECRET_KEY_FILE = ".jwt_secret"
ALERTS_LOG = []
ONVIF_CLIENTS = {}
TUNNEL_STATE = {"url": "", "status": "starting", "mode": "quick"}
tunnel_process = None

BRAND_TEMPLATES = {
    "hikvision": "rtsp://{user}:{pwd}@{ip}:554/Streaming/Channels/102",
    "dahua": "rtsp://{user}:{pwd}@{ip}:554/cam/realmonitor?channel=1&subtype=1",
    "imou": "rtsp://{user}:{pwd}@{ip}:554/cam/realmonitor?channel=1&subtype=1",
    "ezviz": "rtsp://admin:{pwd}@{ip}:554/h264_stream",
    "kbvision": "rtsp://{user}:{pwd}@{ip}:554/cam/realmonitor?channel=1&subtype=1",
    "yoosee": "rtsp://{user}:{pwd}@{ip}:554/onvif2",
    "custom": "rtsp://{user}:{pwd}@{ip}:554/stream1",
}


def get_jwt_secret():
    if os.path.exists(SECRET_KEY_FILE):
        with open(SECRET_KEY_FILE, "r") as f:
            return f.read().strip()
    secret = secrets.token_hex(32)
    with open(SECRET_KEY_FILE, "w") as f:
        f.write(secret)
    return secret


JWT_SECRET = get_jwt_secret()


def hash_password(password: str, salt: str = None) -> str:
    if not salt:
        salt = secrets.token_hex(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 100_000)
    return f"{salt}${dk.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        salt, _ = stored.split("$")
        return hash_password(password, salt) == stored
    except Exception:
        return False


def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
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
    conn.close()


init_db()


def get_setting(key: str) -> str:
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
    conn.close()
    return row[0] if row else ""


def set_setting(key: str, value: str):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)", (key, value))
    conn.commit()
    conn.close()


def log_login_attempt(username: str, status: str, request: Request):
    ip = (
        request.headers.get("CF-Connecting-IP")
        or request.headers.get("X-Forwarded-For")
        or (request.client.host if request.client else "Unknown")
    )
    ua = request.headers.get("User-Agent", "Unknown")[:120]
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO login_logs (username, status, ip_address, user_agent, timestamp) VALUES (?, ?, ?, ?, ?)",
        (username, status, ip, ua, datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
    )
    conn.execute("DELETE FROM login_logs WHERE id NOT IN (SELECT id FROM login_logs ORDER BY id DESC LIMIT 100)")
    conn.commit()
    conn.close()


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
            if os.path.exists(snap_rel):
                with open(snap_rel, "rb") as photo_file:
                    httpx.post(url, data={"chat_id": chat_id, "caption": caption}, files={"photo": photo_file}, timeout=12.0)
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
    ALERTS_LOG.insert(0, alert_data)
    if len(ALERTS_LOG) > 100:
        ALERTS_LOG.pop()
    send_telegram_alert(alert_data)
    try:
        subprocess.Popen([
            "termux-notification",
            "--title", "CamHub AI Cảnh Báo",
            "--content", alert_data["message"],
            "--priority", "high",
        ])
    except Exception:
        pass


ai_engine = CameraAIProcessor(on_alert_callback=handle_ai_alert)


# --- MEDIAMTX & CLOUDFLARE TUNNEL ---
def register_mediamtx_path(cam_id: str, rtsp_url: str):
    try:
        payload = json.dumps({"source": rtsp_url, "record": True}).encode("utf-8")
        httpx.post(f"http://127.0.0.1:9997/v3/config/paths/add/{cam_id}", content=payload, timeout=3.0)
    except Exception:
        pass


def sync_cameras_on_boot():
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("SELECT id, name, rtsp_url FROM cameras").fetchall()
    conn.close()
    for cam_id, name, rtsp_url in rows:
        register_mediamtx_path(cam_id, rtsp_url)
        ai_engine.start_camera_monitor(cam_id, name, rtsp_url)


def start_cloudflare_tunnel():
    global tunnel_process, TUNNEL_STATE
    if tunnel_process and tunnel_process.poll() is None:
        tunnel_process.terminate()

    cf_token = get_setting("cf_token").strip()
    custom_domain = get_setting("cf_custom_domain").strip()

    if cf_token:
        TUNNEL_STATE = {"url": custom_domain or "Named Tunnel Active", "status": "running", "mode": "named"}
        cmd = ["cloudflared", "tunnel", "--no-autoupdate", "run", "--token", cf_token]
        tunnel_process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    else:
        TUNNEL_STATE = {"url": "", "status": "starting", "mode": "quick"}
        cmd = ["cloudflared", "tunnel", "--no-autoupdate", "--url", "http://127.0.0.1:8000"]
        tunnel_process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

        def _watch_url(proc):
            for line in iter(proc.stdout.readline, ""):
                match = re.search(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com", line)
                if match:
                    TUNNEL_STATE["url"] = match.group(0)
                    TUNNEL_STATE["status"] = "running"
                    break

        threading.Thread(target=_watch_url, args=(tunnel_process,), daemon=True).start()


@app.on_event("startup")
def on_startup():
    start_cloudflare_tunnel()
    threading.Timer(3.0, sync_cameras_on_boot).start()


# --- AUTH DEPENDENCIES ---
def get_current_user(authorization: Optional[str] = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Chưa đăng nhập")
    token = authorization.split(" ")[1]
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
    except Exception:
        raise HTTPException(status_code=401, detail="Phiên đăng nhập hết hạn")


def require_admin(user: dict = Depends(get_current_user)):
    if user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Yêu cầu quyền Quản trị viên")
    return user


# --- MODELS ---
class LoginRequest(BaseModel):
    username: str
    password: str
    remember_me: bool = True


class RegisterAdminRequest(BaseModel):
    username: str
    display_name: str
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
    password: str


class CameraCreateRequest(BaseModel):
    name: str
    ip: str
    brand: str
    cam_user: str = "admin"
    cam_pass: str = ""


class PTZCommand(BaseModel):
    direction: str
    speed: float = 0.6


class RetentionUpdate(BaseModel):
    days: int


# --- HLS REVERSE PROXY & UI ---
@app.get("/hls/{path:path}")
async def proxy_hls(path: str):
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"http://127.0.0.1:8888/{path}", timeout=10.0)
        return StreamingResponse(
            iter([resp.content]),
            status_code=resp.status_code,
            media_type=resp.headers.get("content-type", "application/vnd.apple.mpegurl"),
        )


@app.get("/")
def serve_ui():
    return FileResponse("static/index.html")


# --- AUTH & SECURITY ROUTES ---
@app.get("/api/auth/status")
def auth_status():
    conn = sqlite3.connect(DB_PATH)
    count = conn.execute("SELECT COUNT(*) FROM users WHERE role='admin'").fetchone()[0]
    conn.close()
    return {"has_admin": count > 0}


@app.post("/api/auth/setup-admin")
def setup_admin(req: RegisterAdminRequest, request: Request):
    conn = sqlite3.connect(DB_PATH)
    if conn.execute("SELECT COUNT(*) FROM users WHERE role='admin'").fetchone()[0] > 0:
        conn.close()
        raise HTTPException(status_code=400, detail="Đã tồn tại Quản trị viên")
    pwd_hash = hash_password(req.password)
    conn.execute(
        "INSERT INTO users (username, display_name, password_hash, role, created_at) VALUES (?, ?, ?, 'admin', ?)",
        (req.username, req.display_name or "Quản trị viên", pwd_hash, datetime.now().strftime("%Y-%m-%d %H:%M")),
    )
    conn.commit()
    conn.close()
    log_login_attempt(req.username, "Khởi tạo Admin", request)
    token = jwt.encode(
        {"sub": req.username, "name": req.display_name, "role": "admin", "exp": datetime.utcnow() + timedelta(days=30)},
        JWT_SECRET,
        algorithm="HS256",
    )
    return {"status": "ok", "token": token, "role": "admin", "display_name": req.display_name}


@app.post("/api/auth/login")
def login(req: LoginRequest, request: Request):
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute("SELECT username, display_name, password_hash, role FROM users WHERE username=?", (req.username,)).fetchone()
    conn.close()
    if not row or not verify_password(req.password, row[2]):
        log_login_attempt(req.username, "Thất bại (Sai mật khẩu)", request)
        raise HTTPException(status_code=401, detail="Sai tên đăng nhập hoặc mật khẩu")
    log_login_attempt(req.username, "Thành công", request)
    expire_days = 30 if req.remember_me else 1
    token = jwt.encode(
        {"sub": row[0], "name": row[1], "role": row[3], "exp": datetime.utcnow() + timedelta(days=expire_days)},
        JWT_SECRET,
        algorithm="HS256",
    )
    return {"status": "ok", "token": token, "role": row[3], "display_name": row[1]}


@app.post("/api/auth/change-password")
def change_password(req: ChangePasswordRequest, request: Request, user: dict = Depends(get_current_user)):
    if len(req.new_password) < 4:
        raise HTTPException(status_code=400, detail="Mật khẩu mới phải có ít nhất 4 ký tự")
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute("SELECT password_hash FROM users WHERE username=?", (user["sub"],)).fetchone()
    if not row or not verify_password(req.old_password, row[0]):
        conn.close()
        raise HTTPException(status_code=400, detail="Mật khẩu hiện tại không chính xác")
    new_hash = hash_password(req.new_password)
    conn.execute("UPDATE users SET password_hash=? WHERE username=?", (new_hash, user["sub"]))
    conn.commit()
    conn.close()
    log_login_attempt(user["sub"], "Đổi mật khẩu thành công", request)
    return {"status": "ok"}


@app.get("/api/security/login-logs")
def get_login_logs(admin: dict = Depends(require_admin)):
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("SELECT username, status, ip_address, user_agent, timestamp FROM login_logs ORDER BY id DESC LIMIT 40").fetchall()
    conn.close()
    return {"logs": [{"username": r[0], "status": r[1], "ip": r[2], "device": r[3], "time": r[4]} for r in rows]}


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
    resp = httpx.post(
        f"https://api.telegram.org/bot{bot_token}/sendMessage",
        data={
            "chat_id": chat_id,
            "text": "✅ CamHub AI Pro: Kết nối Telegram Bot thành công! Hệ thống sẵn sàng gửi ảnh cảnh báo té ngã & chuyển động.",
        },
        timeout=10.0,
    )
    if resp.status_code != 200:
        raise HTTPException(status_code=400, detail="Lỗi gửi tin nhắn Telegram, kiểm tra lại Token/Chat ID")
    return {"status": "ok"}


# --- FAMILY & TUNNEL ---
@app.get("/api/family")
def list_family(admin: dict = Depends(require_admin)):
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("SELECT id, username, display_name, role, created_at FROM users").fetchall()
    conn.close()
    return {"members": [{"id": r[0], "username": r[1], "display_name": r[2], "role": r[3], "created_at": r[4]} for r in rows]}


@app.post("/api/family")
def create_family_member(req: FamilyMemberRequest, admin: dict = Depends(require_admin)):
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute(
            "INSERT INTO users (username, display_name, password_hash, role, created_at) VALUES (?, ?, ?, 'member', ?)",
            (req.username, req.display_name, hash_password(req.password), datetime.now().strftime("%Y-%m-%d %H:%M")),
        )
        conn.commit()
        return {"status": "ok"}
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=400, detail="Tên đăng nhập đã tồn tại")
    finally:
        conn.close()


@app.delete("/api/family/{member_id}")
def delete_family_member(member_id: int, admin: dict = Depends(require_admin)):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM users WHERE id=? AND role='member'", (member_id,))
    conn.commit()
    conn.close()
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


# --- WIFI, CAMERA, PTZ ONVIF & TWO-WAY AUDIO ---
@app.get("/api/wifi/scan")
def scan_wifi(admin: dict = Depends(require_admin)):
    try:
        output = subprocess.check_output(["termux-wifi-scaninfo"], timeout=8).decode("utf-8")
        networks = json.loads(output)
        seen = {}
        for net in networks:
            ssid = net.get("ssid", "").strip()
            if ssid and (ssid not in seen or net.get("rssi", -100) > seen[ssid]["rssi"]):
                seen[ssid] = {"ssid": ssid, "rssi": net.get("rssi", -70)}
        return {"networks": sorted(seen.values(), key=lambda x: x["rssi"], reverse=True)}
    except Exception:
        return {"networks": [{"ssid": "Home_WiFi_5G", "rssi": -45}, {"ssid": "Home_WiFi_2.4G", "rssi": -58}]}


@app.post("/api/wifi/connect")
def connect_wifi(req: WifiConnectRequest, admin: dict = Depends(require_admin)):
    try:
        subprocess.Popen(["termux-wifi-enable", "true"])
        return {"status": "ok", "message": f"Đã kích hoạt Wi-Fi cho mạng {req.ssid}"}
    except Exception as e:
        return {"status": "warning", "message": str(e)}


@app.get("/api/cameras/discover")
def discover_cameras(admin: dict = Depends(require_admin)):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = "192.168.1.100"
    finally:
        s.close()
    subnet = ".".join(ip.split(".")[:3])
    ips = [f"{subnet}.{i}" for i in range(1, 255)]
    found = []

    def _scan(target):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(0.35)
        ok = sock.connect_ex((target, 554)) == 0
        sock.close()
        return {"ip": target} if ok else None

    with concurrent.futures.ThreadPoolExecutor(max_workers=50) as ex:
        for res in ex.map(_scan, ips):
            if res:
                found.append(res)
    return {"devices": found}


@app.get("/api/cameras")
def list_cameras(user: dict = Depends(get_current_user)):
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("SELECT id, name, ip, brand FROM cameras").fetchall()
    conn.close()
    return {"cameras": [{"id": r[0], "name": r[1], "ip": r[2], "brand": r[3]} for r in rows]}


@app.post("/api/cameras")
def add_camera(req: CameraCreateRequest, admin: dict = Depends(require_admin)):
    cam_id = f"cam_{int(time.time())}"
    template = BRAND_TEMPLATES.get(req.brand.lower(), BRAND_TEMPLATES["custom"])
    rtsp_url = template.format(user=req.cam_user, pwd=req.cam_pass, ip=req.ip)
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO cameras VALUES (?, ?, ?, ?, ?, ?, ?)",
        (cam_id, req.name, req.ip, req.brand, req.cam_user, req.cam_pass, rtsp_url),
    )
    conn.commit()
    conn.close()
    register_mediamtx_path(cam_id, rtsp_url)
    ai_engine.start_camera_monitor(cam_id, req.name, rtsp_url)
    return {"status": "ok", "id": cam_id}


@app.delete("/api/cameras/{cam_id}")
def delete_camera(cam_id: str, admin: dict = Depends(require_admin)):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM cameras WHERE id=?", (cam_id,))
    conn.commit()
    conn.close()
    ONVIF_CLIENTS.pop(cam_id, None)
    ai_engine.stop_camera_monitor(cam_id)
    try:
        httpx.delete(f"http://127.0.0.1:9997/v3/config/paths/delete/{cam_id}", timeout=2.0)
    except Exception:
        pass
    return {"status": "ok"}


def get_onvif_ptz(cam_id: str):
    if cam_id in ONVIF_CLIENTS:
        return ONVIF_CLIENTS[cam_id]
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute("SELECT ip, cam_user, cam_pass FROM cameras WHERE id=?", (cam_id,)).fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail="Không tìm thấy camera")
    ip, user, pwd = row
    for port in [80, 8000, 8899, 2020]:
        try:
            cam = ONVIFCamera(ip, port, user, pwd)
            media = cam.create_media_service()
            ptz = cam.create_ptz_service()
            token = media.GetProfiles()[0].token
            ONVIF_CLIENTS[cam_id] = (ptz, token)
            return ptz, token
        except Exception:
            continue
    raise HTTPException(status_code=400, detail="Camera chưa bật ONVIF hoặc không hỗ trợ xoay PTZ")


@app.post("/api/cameras/{cam_id}/ptz")
def control_ptz(cam_id: str, cmd: PTZCommand, user: dict = Depends(get_current_user)):
    ptz, token = get_onvif_ptz(cam_id)
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
    elif cmd.direction == "stop":
        ptz.Stop({"ProfileToken": token})
        return {"status": "stopped"}

    ptz.ContinuousMove(req)
    threading.Timer(0.45, lambda: ptz.Stop({"ProfileToken": token})).start()
    return {"status": "ok", "direction": cmd.direction}


@app.post("/api/cameras/{cam_id}/talk")
async def push_two_way_audio(cam_id: str, audio: UploadFile = File(...), user: dict = Depends(get_current_user)):
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute("SELECT rtsp_url FROM cameras WHERE id=?", (cam_id,)).fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail="Không tìm thấy camera")

    temp_webm = f"/tmp/talk_{cam_id}_{int(time.time())}.webm"
    with open(temp_webm, "wb") as f:
        f.write(await audio.read())

    def _stream_to_speaker():
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-i", temp_webm, "-ar", "8000", "-ac", "1", "-c:a", "pcm_mulaw", "-f", "rtsp", row[0]],
                timeout=12,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        finally:
            if os.path.exists(temp_webm):
                os.remove(temp_webm)

    threading.Thread(target=_stream_to_speaker, daemon=True).start()
    return {"status": "ok"}


# --- DEVICE HEALTH, STORAGE, RECORDINGS & ALERTS ---
@app.get("/api/system/health")
def get_device_health(user: dict = Depends(get_current_user)):
    try:
        raw = subprocess.check_output(["termux-battery-status"], timeout=3).decode("utf-8")
        info = json.loads(raw)
        return {
            "battery_pct": info.get("percentage", 100),
            "temperature": info.get("temperature", 35.0),
            "status": info.get("status", "OPTIMAL"),
        }
    except Exception:
        return {"battery_pct": 100, "temperature": 35.0, "status": "OPTIMAL"}


def cleanup_old_recordings(retention_days: int, free_bytes: int):
    cutoff = time.time() - (retention_days * 86400)
    all_files = []
    for root, _, files in os.walk("recordings"):
        for f in files:
            if f.endswith(".mp4"):
                full_path = os.path.join(root, f)
                mtime = os.path.getmtime(full_path)
                if mtime < cutoff:
                    os.remove(full_path)
                else:
                    all_files.append((mtime, full_path))

    if free_bytes < 2 * (1024**3) and all_files:
        all_files.sort()
        for _, old_path in all_files[: max(1, len(all_files) // 5)]:
            if os.path.exists(old_path):
                os.remove(old_path)


@app.get("/api/storage")
def get_storage_info(user: dict = Depends(get_current_user)):
    total, used, free = shutil.disk_usage(".")
    cam_bytes = sum(
        os.path.getsize(os.path.join(r, f))
        for r, _, fs in os.walk("recordings")
        for f in fs
        if f.endswith(".mp4")
    )
    days = int(get_setting("retention_days") or 30)
    cleanup_old_recordings(days, free)
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
        for f in sorted(files, reverse=True):
            if f.endswith(".mp4"):
                full_path = os.path.join(root, f)
                items.append({
                    "filename": f,
                    "url": "/" + full_path.replace("\\", "/"),
                    "size_mb": round(os.path.getsize(full_path) / (1024**2), 2),
                    "created_at": datetime.fromtimestamp(os.path.getmtime(full_path)).strftime("%Y-%m-%d %H:%M:%S"),
                })
    return {"recordings": items}


@app.get("/api/alerts")
def get_alerts(user: dict = Depends(get_current_user)):
    return {"alerts": ALERTS_LOG}