import os
import time
import math
import json
import uuid
import shutil
import urllib.request
import subprocess
import threading
from datetime import datetime, timedelta, timezone
from collections import deque
from typing import Optional
import cv2
import numpy as np

# Múi giờ Việt Nam (UTC+7) mặc định để khớp thời gian thực tế dù chạy trong Ubuntu proot (UTC)
VN_TZ = timezone(timedelta(hours=int(os.environ.get("CAMHUB_TZ_OFFSET", "7"))))


def now_vn() -> datetime:
    return datetime.now(VN_TZ)


try:
    import mediapipe as mp
    HAS_MP_LEGACY = hasattr(mp, "solutions") and hasattr(mp.solutions, "pose")
    HAS_MP_TASKS = hasattr(mp, "tasks") and hasattr(mp.tasks, "vision")
except Exception:
    mp = None
    HAS_MP_LEGACY = False
    HAS_MP_TASKS = False

HAS_MEDIAPIPE = HAS_MP_LEGACY or HAS_MP_TASKS

POSE_TASK_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)
POSE_TASK_PATH = os.path.join("models", "pose_landmarker_lite.task")

os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp|stimeout;5000000")




class FaceAndPlateEnhancer:
    """Nhận diện Gương mặt/Đầu (kể cả đeo khẩu trang, đội nón) & Biển số xe, phóng to + làm nét AI."""

    def __init__(self):
        data_dir = getattr(getattr(cv2, "data", None), "haarcascades", "")
        self.face_frontal = self._load_cascade(data_dir, "haarcascade_frontalface_alt2.xml")
        self.face_profile = self._load_cascade(data_dir, "haarcascade_profileface.xml")
        self.upper_body = self._load_cascade(data_dir, "haarcascade_upperbody.xml")
        self.plate_cascade = self._load_cascade(data_dir, "haarcascade_russian_plate_number.xml")

    @staticmethod
    def _load_cascade(data_dir: str, filename: str):
        if not data_dir:
            return None
        path = os.path.join(data_dir, filename)
        if os.path.exists(path):
            c = cv2.CascadeClassifier(path)
            return c if not c.empty() else None
        return None

    @staticmethod
    def enhance_zoom_crop(crop: np.ndarray, min_width: int = 420) -> np.ndarray:
        """Phóng to thông minh LANCZOS4 + khử nhiễu giữ biên + cân bằng tương phản CLAHE + làm nét Unsharp."""
        if crop is None or crop.size == 0:
            return crop
        h, w = crop.shape[:2]
        if w < min_width:
            scale = min_width / float(max(1, w))
            new_w = max(2, int(round(w * scale)))
            new_h = max(2, int(round(h * scale)))
            crop = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

        # Khử nhiễu nén JPEG nhưng giữ nguyên cạnh chữ/mắt
        denoised = cv2.bilateralFilter(crop, d=5, sigmaColor=42, sigmaSpace=42)

        # Cân bằng sáng cục bộ (hữu ích khi đội nón bị tối mặt hoặc biển số chói nắng)
        lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(4, 4))
        l_eq = clahe.apply(l_ch)
        balanced = cv2.cvtColor(cv2.merge((l_eq, a_ch, b_ch)), cv2.COLOR_LAB2BGR)

        # Unsharp Mask làm nét chi tiết nhỏ
        blur = cv2.GaussianBlur(balanced, (0, 0), sigmaX=1.35)
        sharpened = cv2.addWeighted(balanced, 1.55, blur, -0.55, 0)
        return sharpened

    @staticmethod
    def _expand_box(x: int, y: int, w: int, h: int, img_w: int, img_h: int, pad: float = 0.28):
        dx = int(w * pad)
        dy = int(h * pad)
        x1 = max(0, x - dx)
        y1 = max(0, y - dy)
        x2 = min(img_w, x + w + dx)
        y2 = min(img_h, y + h + dy)
        return x1, y1, x2, y2

    @staticmethod
    def _iou(box_a, box_b) -> float:
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        inter = (ix2 - ix1) * (iy2 - iy1)
        area = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
        return inter / float(max(1, area))

    def _dedup_boxes(self, boxes: list, limit: int = 2) -> list:
        out = []
        for b in boxes:
            if all(self._iou(b, prev) < 0.35 for prev in out):
                out.append(b)
                if len(out) >= limit:
                    break
        return out

    def detect_faces_or_heads(self, img: np.ndarray, person_boxes: Optional[list] = None) -> list:
        """Tìm gương mặt (thẳng/nghiêng) và vùng đầu-mặt khi đeo khẩu trang/nón."""
        h, w = img.shape[:2]
        gray = cv2.equalizeHist(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        raw_boxes = []

        if self.face_frontal is not None:
            for (fx, fy, fw, fh) in self.face_frontal.detectMultiScale(gray, 1.12, 4, minSize=(26, 26)):
                raw_boxes.append(self._expand_box(fx, fy, fw, fh, w, h, pad=0.32))

        if self.face_profile is not None:
            for (fx, fy, fw, fh) in self.face_profile.detectMultiScale(gray, 1.15, 4, minSize=(26, 26)):
                raw_boxes.append(self._expand_box(fx, fy, fw, fh, w, h, pad=0.32))

        # Nếu đeo khẩu trang/nón che mất mặt chuẩn, lấy vùng đầu từ upper-body hoặc khung người chuyển động
        if not raw_boxes and self.upper_body is not None:
            for (ux, uy, uw, uh) in self.upper_body.detectMultiScale(gray, 1.12, 3, minSize=(55, 55)):
                head_h = int(uh * 0.48)
                head_w = int(uw * 0.52)
                head_x = ux + (uw - head_w) // 2
                raw_boxes.append(self._expand_box(head_x, uy, head_w, head_h, w, h, pad=0.22))

        if not raw_boxes and person_boxes:
            for (px, py, pw, ph) in person_boxes:
                if ph > pw * 0.9 and ph >= 50:
                    head_h = max(28, int(ph * 0.30))
                    head_w = max(28, int(pw * 0.65))
                    head_x = px + (pw - head_w) // 2
                    raw_boxes.append(self._expand_box(head_x, py, head_w, head_h, w, h, pad=0.25))

        return self._dedup_boxes(raw_boxes, limit=2)

    def detect_license_plates(self, img: np.ndarray) -> list:
        """Tìm biển số xe (1 dòng & 2 dòng) bằng Haar + hình học mật độ cạnh."""
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        candidates = []

        if self.plate_cascade is not None:
            for (px, py, pw, ph) in self.plate_cascade.detectMultiScale(gray, 1.12, 4, minSize=(42, 16)):
                if 1.1 <= (pw / float(max(1, ph))) <= 5.5:
                    candidates.append(self._expand_box(px, py, pw, ph, w, h, pad=0.22))

        # Bổ sung phát hiện hình chữ nhật biển số có mật độ nét chữ cao (biển 1 dòng & 2 dòng VN)
        if not candidates:
            rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 5))
            blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rect_kernel)
            grad_x = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=3)
            grad_x = np.absolute(grad_x)
            max_v = float(np.max(grad_x)) or 1.0
            grad_x = np.uint8(255.0 * (grad_x / max_v))
            grad_x = cv2.GaussianBlur(grad_x, (5, 5), 0)
            grad_x = cv2.morphologyEx(grad_x, cv2.MORPH_CLOSE, rect_kernel)
            _, thresh = cv2.threshold(grad_x, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            thresh = cv2.erode(thresh, None, iterations=1)
            thresh = cv2.dilate(thresh, None, iterations=2)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)[:12]
            for c in contours:
                cx, cy, cw, ch = cv2.boundingRect(c)
                ratio = cw / float(max(1, ch))
                area = cw * ch
                # Bỏ qua vùng quá nhỏ/quá lớn hoặc góc trên cùng (nơi chứa dòng đồng hồ OSD camera)
                if cy < int(h * 0.14):
                    continue
                if 1200 <= area <= int(w * h * 0.12) and (1.15 <= ratio <= 5.2):
                    roi = gray[cy : cy + ch, cx : cx + cw]
                    if roi.size > 0 and float(np.std(roi)) > 36.0:
                        candidates.append(self._expand_box(cx, cy, cw, ch, w, h, pad=0.18))

        return self._dedup_boxes(candidates, limit=2)

    def extract_detail_crops(
        self,
        img: np.ndarray,
        prefix: str,
        person_boxes: Optional[list] = None,
    ) -> list:
        """Trích xuất & làm nét ảnh gương mặt/đầu và biển số xe, trả về danh sách dict chi tiết."""
        if img is None or img.size == 0:
            return []
        details = []

        for idx, (x1, y1, x2, y2) in enumerate(self.detect_faces_or_heads(img, person_boxes), start=1):
            crop = img[y1:y2, x1:x2]
            enhanced = self.enhance_zoom_crop(crop, min_width=420)
            out_file = f"snapshots/{prefix}_face{idx}.jpg"
            cv2.imwrite(out_file, enhanced, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            details.append({
                "kind": "face",
                "label": "👤 Gương mặt / Đầu (Đã phóng to & làm nét)",
                "url": f"/{out_file}",
            })

        for idx, (x1, y1, x2, y2) in enumerate(self.detect_license_plates(img), start=1):
            crop = img[y1:y2, x1:x2]
            enhanced = self.enhance_zoom_crop(crop, min_width=440)
            out_file = f"snapshots/{prefix}_plate{idx}.jpg"
            cv2.imwrite(out_file, enhanced, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            details.append({
                "kind": "plate",
                "label": "🚗 Biển số xe (Đã phóng to & làm nét)",
                "url": f"/{out_file}",
            })

        return details


class RollingH264Writer:
    """Ghi video dài (30-60 phút/tệp) chuẩn H.264 fMP4 xem được ngay cả khi đang ghi."""

    def __init__(self, cam_id: str, segment_seconds: int = 1800):
        self.cam_id = cam_id
        self.segment_seconds = max(1800, min(3600, segment_seconds))
        self.proc = None
        self.seg_start = 0.0
        self.lock = threading.Lock()
        self.out_dir = os.path.join("recordings", cam_id)
        os.makedirs(self.out_dir, exist_ok=True)

    def _open_new_segment(self):
        self.close()
        ts_name = now_vn().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(self.out_dir, f"continuous_{ts_name}.mp4")
        cmd = [
            "ffmpeg", "-nostdin",
            "-noautorotate", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "image2pipe", "-vcodec", "mjpeg", "-framerate", "4", "-i", "-",
            "-vf", "scale='min(720,iw)':-2",
            "-c:v", "libx264", "-preset", "superfast", "-crf", "33",
            "-maxrate", "600k", "-bufsize", "1200k",
            "-pix_fmt", "yuv420p",
            "-movflags", "+frag_keyframe+empty_moov+default_base_moof",
            out_path,
        ]
        try:
            self.proc = subprocess.Popen(
                cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            self.seg_start = time.time()
        except Exception:
            self.proc = None

    def write_frame(self, jpg_bytes: bytes):
        with self.lock:
            now = time.time()
            if self.proc is None or self.proc.poll() is not None or (now - self.seg_start >= self.segment_seconds):
                self._open_new_segment()
            if self.proc and self.proc.stdin:
                try:
                    self.proc.stdin.write(jpg_bytes)
                    self.proc.stdin.flush()
                except Exception:
                    self.close()

    def close(self):
        if self.proc is not None:
            try:
                if self.proc.stdin:
                    self.proc.stdin.close()
                self.proc.wait(timeout=3)
            except Exception:
                try:
                    self.proc.kill()
                except Exception:
                    pass
            self.proc = None


class EventClipManager:
    """Quản lý bộ đệm vòng 15s và tự động xuất clip ngắn ~60s cho Chuyển động / Té ngã."""

    def __init__(self, cam_id: str, pre_seconds: int = 15, post_seconds: int = 45):
        self.cam_id = cam_id
        self.ring = deque(maxlen=max(20, pre_seconds * 4))
        self.active_events = {}  # event_type -> {"frames": list, "end_ts": float, "out_path": str}
        self.lock = threading.Lock()
        self.post_seconds = post_seconds
        self.out_dir = os.path.join("recordings", cam_id)
        os.makedirs(self.out_dir, exist_ok=True)

    def push_frame(self, jpg_bytes: bytes):
        now = time.time()
        to_finalize = []
        with self.lock:
            self.ring.append(jpg_bytes)
            for ev_type, job in list(self.active_events.items()):
                job["frames"].append(jpg_bytes)
                if now >= job["end_ts"] or len(job["frames"]) >= 280:
                    to_finalize.append((ev_type, job["frames"], job["out_path"]))
                    del self.active_events[ev_type]
        for _, frames, out_path in to_finalize:
            threading.Thread(target=self._encode_clip, args=(frames, out_path), daemon=True).start()

    def trigger_clip(self, event_type: str):
        now = time.time()
        with self.lock:
            if event_type in self.active_events:
                return
            ts_name = now_vn().strftime("%Y%m%d_%H%M%S")
            out_path = os.path.join(self.out_dir, f"{event_type}_{ts_name}.mp4")
            self.active_events[event_type] = {
                "frames": list(self.ring),
                "end_ts": now + self.post_seconds,
                "out_path": out_path,
            }

    def flush_all(self):
        with self.lock:
            jobs = list(self.active_events.values())
            self.active_events.clear()
        for job in jobs:
            if len(job["frames"]) >= 8:
                threading.Thread(target=self._encode_clip, args=(job["frames"], job["out_path"]), daemon=True).start()

    def _encode_clip(self, frames: list, out_path: str):
        if not frames:
            return
        cmd = [
            "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "image2pipe", "-vcodec", "mjpeg", "-framerate", "4", "-i", "-",
            "-vf", "scale='min(720,iw)':-2",
            "-c:v", "libx264", "-preset", "superfast", "-crf", "31",
            "-maxrate", "750k", "-bufsize", "1500k",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            out_path,
        ]
        try:
            proc = subprocess.Popen(
                cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            if proc.stdin:
                for jpg in frames:
                    proc.stdin.write(jpg)
                proc.stdin.close()
            proc.wait(timeout=30)
        except Exception:
            pass


class CameraAIProcessor:
    def __init__(self, on_alert_callback):
        self.on_alert = on_alert_callback
        self.workers = {}
        self.workers_lock = threading.Lock()
        self.frames = {}  # key: (cam_id, lens_idx) -> jpeg bytes (0 = full, 1..3 = split lens)
        self.frames_lock = threading.Lock()
        self.cam_status = {}
        self.thermal_delay = 0.20  # Mặc định ~5 FPS tiết kiệm pin & giảm nhiệt
        os.makedirs("snapshots", exist_ok=True)
        os.makedirs("models", exist_ok=True)

        self.mp_pose = mp.solutions.pose if HAS_MP_LEGACY else None
        self.has_ffmpeg_cli = shutil.which("ffmpeg") is not None

        # Bộ nhận diện dáng người đứng OpenCV HOG (dùng khi không có MediaPipe model)
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        self.enhancer = FaceAndPlateEnhancer()

        self.recorders = {}
        self.clip_managers = {}
        self.long_clip_sec = int(os.environ.get("CAMHUB_LONG_CLIP_SEC", "1800"))
        self.active_engine_label = self._resolve_engine_label()
        threading.Thread(target=self._thermal_guard_loop, daemon=True).start()
        if HAS_MP_TASKS and not HAS_MP_LEGACY and not os.path.exists(POSE_TASK_PATH):
            threading.Thread(target=self._ensure_pose_task_model, daemon=True).start()

    def _resolve_engine_label(self) -> str:
        if HAS_MP_LEGACY:
            return "MediaPipe Pose"
        if HAS_MP_TASKS and os.path.exists(POSE_TASK_PATH):
            return "MediaPipe Pose Tasks"
        return "OpenCV Person + Fall AI"

    def _ensure_pose_task_model(self):
        try:
            os.makedirs("models", exist_ok=True)
            if not os.path.exists(POSE_TASK_PATH):
                urllib.request.urlretrieve(POSE_TASK_URL, POSE_TASK_PATH)
            self.active_engine_label = self._resolve_engine_label()
        except Exception:
            pass

    def _create_pose_landmarker(self):
        if not HAS_MP_TASKS or not os.path.exists(POSE_TASK_PATH):
            return None
        try:
            BaseOptions = mp.tasks.BaseOptions
            PoseLandmarker = mp.tasks.vision.PoseLandmarker
            PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
            VisionRunningMode = mp.tasks.vision.RunningMode
            options = PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=POSE_TASK_PATH),
                running_mode=VisionRunningMode.IMAGE,
                min_pose_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            return PoseLandmarker.create_from_options(options)
        except Exception:
            return None

    def _read_device_temp(self) -> Optional[float]:
        try:
            raw = subprocess.check_output(
                ["termux-battery-status"],
                timeout=2,
                stderr=subprocess.DEVNULL,
            ).decode("utf-8")
            return float(json.loads(raw).get("temperature", 35.0))
        except Exception:
            pass
        for path in (
            "/sys/class/thermal/thermal_zone0/temp",
            "/sys/class/power_supply/battery/temp",
        ):
            try:
                if os.path.exists(path):
                    with open(path, "r", encoding="utf-8") as f:
                        val = float(f.read().strip())
                        return val / 1000.0 if val > 200 else (val / 10.0 if val > 80 else val)
            except Exception:
                continue
        return None

    def _thermal_guard_loop(self):
        """Tự động giãn nhịp AI khi nhiệt độ thiết bị vượt 43°C để bảo vệ máy chạy 24/7."""
        while True:
            temp = self._read_device_temp()
            if temp is not None:
                if temp >= 43.0:
                    self.thermal_delay = 0.55
                elif temp <= 38.5:
                    self.thermal_delay = 0.20
            time.sleep(20)

    def start_camera_monitor(self, cam_id: str, cam_name: str, rtsp_url: str):
        self.stop_camera_monitor(cam_id)
        stop_event = threading.Event()
        with self.frames_lock:
            self.cam_status[cam_id] = {
                "online": False,
                "engine": self.active_engine_label,
                "last_frame_ts": 0.0,
                "lens_count": 1,
            }
        t = threading.Thread(
            target=self._process_loop,
            args=(cam_id, cam_name, rtsp_url, stop_event),
            daemon=True,
        )
        with self.workers_lock:
            self.recorders[cam_id] = RollingH264Writer(cam_id, segment_seconds=self.long_clip_sec)
            self.clip_managers[cam_id] = EventClipManager(cam_id, pre_seconds=15, post_seconds=45)
            self.workers[cam_id] = (stop_event, t)
        t.start()

    def stop_camera_monitor(self, cam_id: str):
        with self.workers_lock:
            worker = self.workers.pop(cam_id, None)
            rec = self.recorders.pop(cam_id, None)
            cm = self.clip_managers.pop(cam_id, None)
        if rec:
            rec.close()
        if cm:
            cm.flush_all()
        if worker:
            stop_event, _ = worker
            stop_event.set()
        with self.frames_lock:
            keys_to_del = [k for k in self.frames if k[0] == cam_id]
            for k in keys_to_del:
                self.frames.pop(k, None)
            self.cam_status.pop(cam_id, None)

    def get_latest_frame(self, cam_id: str, lens: int = 0) -> Optional[bytes]:
        with self.frames_lock:
            return self.frames.get((cam_id, lens)) or self.frames.get((cam_id, 0))

    def get_camera_status(self, cam_id: str) -> dict:
        with self.frames_lock:
            st = dict(self.cam_status.get(cam_id, {}))
        if not st:
            return {"online": False, "engine": self.active_engine_label, "lens_count": 1}
        is_recent = (time.time() - st.get("last_frame_ts", 0.0)) < 12.0
        st["online"] = bool(st.get("online") and is_recent)
        st.setdefault("lens_count", 1)
        return st

    def build_placeholder_jpeg(self, title: str, subtitle: str = "Dang ket noi luong camera...") -> bytes:
        canvas = np.zeros((360, 640, 3), dtype=np.uint8)
        canvas[:] = (22, 16, 12)
        cv2.putText(canvas, title[:36], (36, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (248, 250, 252), 2, cv2.LINE_AA)
        cv2.putText(canvas, subtitle[:50], (36, 208), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (148, 163, 184), 1, cv2.LINE_AA)
        ok, buf = cv2.imencode(".jpg", canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        return buf.tobytes() if ok else b""

    def trigger_test_fall_alert(self, cam_id: str, cam_name: str) -> dict:
        ts_ms = int(time.time() * 1000)
        snap_file = f"snapshots/fall_test_{cam_id}_{ts_ms}.jpg"
        latest = self.get_latest_frame(cam_id, 0)
        if latest:
            with open(snap_file, "wb") as f:
                f.write(latest)
        else:
            img = self.build_placeholder_jpeg(
                f"TEST FALL ALERT - {cam_name}",
                now_vn().strftime("%Y-%m-%d %H:%M:%S"),
            )
            with open(snap_file, "wb") as f:
                f.write(img)

        alert_data = {
            "id": f"fall_test_{cam_id}_{ts_ms}_{uuid.uuid4().hex[:6]}",
            "type": "fall",
            "cam_id": cam_id,
            "cam_name": cam_name,
            "message": f"[THỬ NGHIỆM] KHẨN CẤP: Phát hiện người bị té ngã tại {cam_name}!",
            "snapshot": f"/{snap_file}",
            "detail_snapshots": [],
            "timestamp": now_vn().strftime("%Y-%m-%d %H:%M:%S"),
        }
        self.on_alert(alert_data)
        return alert_data

    def _calculate_body_angle_landmarks(self, landmarks) -> float:
        # Indices MediaPipe Pose chuẩn: 11=L_SHOULDER, 12=R_SHOULDER, 23=L_HIP, 24=R_HIP
        l_shoulder = landmarks[11]
        r_shoulder = landmarks[12]
        l_hip = landmarks[23]
        r_hip = landmarks[24]

        vis_vals = [getattr(pt, "visibility", 1.0) for pt in (l_shoulder, r_shoulder, l_hip, r_hip)]
        if min(vis_vals) < 0.35:
            return 0.0

        mid_shoulder_x = (l_shoulder.x + r_shoulder.x) / 2.0
        mid_shoulder_y = (l_shoulder.y + r_shoulder.y) / 2.0
        mid_hip_x = (l_hip.x + r_hip.x) / 2.0
        mid_hip_y = (l_hip.y + r_hip.y) / 2.0

        dx = abs(mid_shoulder_x - mid_hip_x)
        dy = abs(mid_shoulder_y - mid_hip_y) + 1e-6
        return math.degrees(math.atan2(dx, dy))

    def _detect_fall_opencv_smart(self, small_bgr: np.ndarray, fg_clean: np.ndarray, st: dict, now: float) -> bool:
        """
        Phát hiện té ngã thông minh bằng OpenCV khi chưa có MediaPipe:
        Chỉ báo té ngã khi từng ghi nhận dáng đứng/người (hoặc sụp độ cao đột ngột) rồi chuyển sang nằm ngang bất động.
        Tránh báo giả khi xe máy / vật thể dài chạy ngang qua sân.
        """
        contours, _ = cv2.findContours(fg_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return False
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        if area < 2400:
            return False
        _, y, w, h = cv2.boundingRect(largest)
        aspect = w / float(h + 1e-6)

        # Ghi nhận trạng thái dáng đứng trước đó (h > w * 1.15 hoặc phát hiện người bằng HOG)
        if h > w * 1.15:
            st["last_upright_ts"] = now
            return False

        # Chỉ xem là nghi ngờ té ngã nếu nằm ngang (w/h > 1.38) VÀ trong vòng 8 giây trước có dáng đứng hoặc thay đổi chiều cao đột ngột
        had_upright = (now - st.get("last_upright_ts", 0.0)) <= 8.0
        prev_h = st.get("prev_bbox_h", h)
        st["prev_bbox_h"] = h
        sudden_drop = (prev_h > 45) and (h < prev_h * 0.65)
        return (aspect > 1.38) and (had_upright or sudden_drop)

    def _detect_lens_count(self, frame: np.ndarray) -> int:
        h, w = frame.shape[:2]
        if w <= 0:
            return 1
        ratio = h / float(w)
        if ratio >= 1.55:
            return 3
        if ratio >= 1.05:
            return 2
        return 1

    def _split_lenses(self, frame: np.ndarray, lens_count: int) -> list:
        if lens_count <= 1:
            return [frame]
        h = frame.shape[0]
        step = h // lens_count
        return [frame[i * step : (i + 1) * step if i < lens_count - 1 else h] for i in range(lens_count)]

    def _spawn_ffmpeg_mjpeg(self, rtsp_url: str, transport: str):
        cmd = [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel", "error",
            "-fflags", "+nobuffer+discardcorrupt",
            "-flags", "low_delay",
            "-analyzeduration", "2000000",
            "-probesize", "2000000",
            "-rtsp_transport", transport,
            "-timeout", "8000000",
            "-i", rtsp_url,
            "-an",
            "-vf", "scale='min(1280,iw)':-2,format=yuvj420p",
            "-fps_mode", "drop",
            "-strict", "unofficial",
            "-f", "image2pipe",
            "-vcodec", "mjpeg",
            "-q:v", "3",
            "-",
        ]
        return subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=0,
        )

    def _iter_ffmpeg_jpegs(self, rtsp_url: str, transport: str, stop_event: threading.Event):
        proc = self._spawn_ffmpeg_mjpeg(rtsp_url, transport)
        buf = bytearray()
        last_data_ts = [time.time()]

        # Watchdog: tự ngắt nếu ffmpeg bị treo không trả frame quá 12 giây
        def _watchdog():
            while not stop_event.is_set() and proc.poll() is None:
                if time.time() - last_data_ts[0] > 12.0:
                    try:
                        proc.kill()
                    except Exception:
                        pass
                    break
                if stop_event.wait(2.0):
                    break

        threading.Thread(target=_watchdog, daemon=True).start()

        try:
            while not stop_event.is_set():
                if not proc.stdout:
                    break
                chunk = proc.stdout.read(8192)
                if not chunk:
                    break
                last_data_ts[0] = time.time()
                buf.extend(chunk)
                while True:
                    start = buf.find(b"\xff\xd8")
                    if start == -1:
                        if len(buf) > 131072:
                            buf.clear()
                        break
                    end = buf.find(b"\xff\xd9", start + 2)
                    if end == -1:
                        if start > 0:
                            del buf[:start]
                        break
                    jpg = bytes(buf[start : end + 2])
                    del buf[: end + 2]
                    yield jpg
        finally:
            try:
                proc.terminate()
                proc.wait(timeout=2)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass

    def _run_ai_on_lens(
        self,
        cam_id: str,
        label_name: str,
        lens_img: np.ndarray,
        lens_jpg: bytes,
        lens_idx: int,
        lens_states: dict,
        local_pose,
        task_landmarker,
        now: float,
    ):
        if lens_idx not in lens_states:
            lens_states[lens_idx] = {
                "bg": cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=25, detectShadows=False),
                "frames_seen": 0,
                "last_motion_alert": 0.0,
                "last_fall_alert": 0.0,
                "horizontal_start_time": None,
                "last_upright_ts": 0.0,
                "prev_bbox_h": 0,
            }
        st = lens_states[lens_idx]
        st["frames_seen"] += 1

        small = cv2.resize(lens_img, (480, 270))
        h_start = st["horizontal_start_time"]
        lr = 0.0 if (h_start is not None and (now - h_start) < 10.0) else -1.0
        fg_mask = st["bg"].apply(small, learningRate=lr)
        if st["frames_seen"] <= 6:
            return
        fg_clean = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        has_motion = cv2.countNonZero(fg_clean) > 900

        person_boxes = []
        if has_motion:
            cnts, _ = cv2.findContours(fg_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            ih, iw = lens_img.shape[:2]
            sx, sy = iw / 480.0, ih / 270.0
            for c in cnts:
                if cv2.contourArea(c) >= 1200:
                    bx, by, bw, bh = cv2.boundingRect(c)
                    person_boxes.append((int(bx * sx), int(by * sy), int(bw * sx), int(bh * sy)))

        if has_motion and (now - st["last_motion_alert"] > 30.0):
            ts_ms = int(now * 1000)
            snap_file = f"snapshots/motion_{cam_id}_L{lens_idx}_{ts_ms}.jpg"
            with open(snap_file, "wb") as f:
                f.write(lens_jpg)
            details = self.enhancer.extract_detail_crops(
                lens_img, f"motion_{cam_id}_L{lens_idx}_{ts_ms}", person_boxes
            )
            cm = self.clip_managers.get(cam_id)
            if cm is not None:
                cm.trigger_clip("motion")
            self.on_alert({
                "id": f"motion_{cam_id}_{lens_idx}_{ts_ms}_{uuid.uuid4().hex[:6]}",
                "type": "motion",
                "cam_id": cam_id,
                "cam_name": label_name,
                "message": f"Phát hiện chuyển động tại {label_name}",
                "snapshot": f"/{snap_file}",
                "detail_snapshots": details,
                "timestamp": now_vn().strftime("%Y-%m-%d %H:%M:%S"),
            })
            st["last_motion_alert"] = now

        if has_motion or (st["horizontal_start_time"] is not None):
            is_horizontal = False
            if local_pose is not None:
                rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                results = local_pose.process(rgb)
                if results and results.pose_landmarks:
                    angle = self._calculate_body_angle_landmarks(results.pose_landmarks.landmark)
                    is_horizontal = angle > 58.0
            elif task_landmarker is not None:
                try:
                    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                    res = task_landmarker.detect(mp_image)
                    if res and res.pose_landmarks and len(res.pose_landmarks) > 0:
                        angle = self._calculate_body_angle_landmarks(res.pose_landmarks[0])
                        is_horizontal = angle > 58.0
                except Exception:
                    is_horizontal = self._detect_fall_opencv_smart(small, fg_clean, st, now)
            else:
                is_horizontal = self._detect_fall_opencv_smart(small, fg_clean, st, now)

            if is_horizontal:
                if st["horizontal_start_time"] is None:
                    st["horizontal_start_time"] = now
                elif (now - st["horizontal_start_time"]) >= 2.5 and (now - st["last_fall_alert"] > 20.0):
                    ts_ms = int(now * 1000)
                    snap_file = f"snapshots/fall_{cam_id}_L{lens_idx}_{ts_ms}.jpg"
                    with open(snap_file, "wb") as f:
                        f.write(lens_jpg)
                    details = self.enhancer.extract_detail_crops(
                        lens_img, f"fall_{cam_id}_L{lens_idx}_{ts_ms}", person_boxes
                    )
                    cm = self.clip_managers.get(cam_id)
                    if cm is not None:
                        cm.trigger_clip("fall")
                    self.on_alert({
                        "id": f"fall_{cam_id}_{lens_idx}_{ts_ms}_{uuid.uuid4().hex[:6]}",
                        "type": "fall",
                        "cam_id": cam_id,
                        "cam_name": label_name,
                        "message": f"KHẨN CẤP: Phát hiện người bị té ngã tại {label_name}!",
                        "snapshot": f"/{snap_file}",
                        "detail_snapshots": details,
                        "timestamp": now_vn().strftime("%Y-%m-%d %H:%M:%S"),
                    })
                    st["last_fall_alert"] = now
                    st["horizontal_start_time"] = None
            else:
                st["horizontal_start_time"] = None

    def _analyze_frame(
        self,
        cam_id: str,
        cam_name: str,
        jpg_bytes: bytes,
        lens_states: dict,
        local_pose,
        task_landmarker,
        state: dict,
    ):
        now = time.time()
        # Giữ nhịp tối đa ~5 FPS ở Python thay vì để ffmpeg lặp khung hình cũ
        if now - state.get("last_decode_ts", 0.0) < 0.25:
            return
        state["last_decode_ts"] = now

        arr = np.frombuffer(jpg_bytes, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            return

        # Kiểm tra khung hình có bị đóng băng (trùng lặp y hệt > 12s) không
        tiny = cv2.resize(frame, (64, 36))
        prev_tiny = state.get("prev_tiny")
        if prev_tiny is None or float(np.mean(cv2.absdiff(prev_tiny, tiny))) > 0.35:
            state["prev_tiny"] = tiny
            state["last_unique_ts"] = now
        elif now - state.get("last_unique_ts", now) > 12.0:
            state["force_reconnect"] = True
            return

        lens_count = self._detect_lens_count(frame)
        lenses = self._split_lenses(frame, lens_count)
        lens_jpegs = []

        if lens_count == 1:
            lens_jpegs.append((1, jpg_bytes))
        else:
            for idx, crop in enumerate(lenses, start=1):
                ok_c, buf_c = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
                if ok_c:
                    lens_jpegs.append((idx, buf_c.tobytes()))

        engine_name = (
            "MediaPipe Pose"
            if local_pose is not None
            else ("MediaPipe Pose Tasks" if task_landmarker is not None else "OpenCV Person + Fall AI")
        )
        self.active_engine_label = engine_name

        rec = self.recorders.get(cam_id)
        if rec is not None and self.has_ffmpeg_cli:
            rec.write_frame(jpg_bytes)
        cm = self.clip_managers.get(cam_id)
        if cm is not None and self.has_ffmpeg_cli:
            cm.push_frame(jpg_bytes)

        with self.frames_lock:
            self.frames[(cam_id, 0)] = jpg_bytes
            for idx, l_jpg in lens_jpegs:
                self.frames[(cam_id, idx)] = l_jpg
            self.cam_status[cam_id] = {
                "online": True,
                "engine": engine_name,
                "last_frame_ts": now,
                "lens_count": lens_count,
            }

        if now - state["last_ai_ts"] < self.thermal_delay:
            return
        state["last_ai_ts"] = now

        for idx, crop in enumerate(lenses, start=1):
            l_jpg = dict(lens_jpegs).get(idx, jpg_bytes)
            label = f"{cam_name} (Mắt {idx})" if lens_count > 1 else cam_name
            self._run_ai_on_lens(
                cam_id, label, crop, l_jpg, idx, lens_states, local_pose, task_landmarker, now
            )

    def _process_loop(self, cam_id: str, cam_name: str, rtsp_url: str, stop_event: threading.Event):
        local_pose = None
        task_landmarker = None
        if HAS_MP_LEGACY and self.mp_pose is not None:
            try:
                local_pose = self.mp_pose.Pose(
                    static_image_mode=False,
                    model_complexity=0,
                    min_detection_confidence=0.55,
                    min_tracking_confidence=0.55,
                )
            except Exception:
                local_pose = None
        elif HAS_MP_TASKS:
            task_landmarker = self._create_pose_landmarker()

        lens_states = {}
        state = {"last_ai_ts": 0.0}
        local_mtx_url = f"rtsp://127.0.0.1:8554/{cam_id}"
        is_yoosee = "/onvif" in rtsp_url.lower()
        sources = [
            (rtsp_url, "udp"),
            (rtsp_url, "tcp"),
            (local_mtx_url, "tcp"),
        ] if is_yoosee else [
            (local_mtx_url, "tcp"),
            (rtsp_url, "udp"),
            (rtsp_url, "tcp"),
        ]
        src_idx = 0
        retry_delay = 3.0

        try:
            while not stop_event.is_set():
                # Nếu model PoseLandmarker vừa tải xong ở nền, nạp ngay vào luồng
                if local_pose is None and task_landmarker is None and HAS_MP_TASKS and os.path.exists(POSE_TASK_PATH):
                    task_landmarker = self._create_pose_landmarker()

                got_frame = False
                if self.has_ffmpeg_cli:
                    src_url, transport = sources[src_idx % len(sources)]
                    state["force_reconnect"] = False
                    state["last_unique_ts"] = time.time()
                    for jpg in self._iter_ffmpeg_jpegs(src_url, transport, stop_event):
                        got_frame = True
                        retry_delay = 3.0
                        self._analyze_frame(
                            cam_id, cam_name, jpg, lens_states, local_pose, task_landmarker, state
                        )
                        if state.get("force_reconnect"):
                            break
                    if not got_frame:
                        src_idx += 1
                else:
                    cap = cv2.VideoCapture(rtsp_url)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    try:
                        while not stop_event.is_set():
                            ret, frame = cap.read()
                            if not ret or frame is None:
                                break
                            got_frame = True
                            retry_delay = 3.0
                            h, w = frame.shape[:2]
                            target_h = max(2, int(round(640 * (h / float(max(1, w))))) & ~1)
                            preview = cv2.resize(frame, (640, target_h))
                            ok, buf = cv2.imencode(".jpg", preview, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
                            if ok:
                                self._analyze_frame(
                                    cam_id, cam_name, buf.tobytes(), lens_states, local_pose, task_landmarker, state
                                )
                            if stop_event.wait(self.thermal_delay):
                                break
                    finally:
                        cap.release()

                with self.frames_lock:
                    if cam_id in self.cam_status:
                        self.cam_status[cam_id]["online"] = False

                if stop_event.wait(retry_delay):
                    break
                retry_delay = min(10.0, retry_delay + 2.0)
        finally:
            if local_pose is not None:
                try:
                    local_pose.close()
                except Exception:
                    pass
            if task_landmarker is not None:
                try:
                    task_landmarker.close()
                except Exception:
                    pass
