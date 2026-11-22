import os
import re
import time
import queue
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





class AdaptiveAILearner:
    """Tự học nhẹ từ phản hồi Đúng/Sai của người dùng: giảm báo giả vùng cây cối & ưu tiên vùng đường đi."""

    def __init__(self, state_file: str = "ai_learning.json"):
        self.state_file = os.environ.get("CAMHUB_AI_LEARN_FILE", state_file)
        self.lock = threading.Lock()
        self.data = self._load()

    def _load(self) -> dict:
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, "r", encoding="utf-8") as f:
                    d = json.load(f)
                    if isinstance(d, dict):
                        return d
            except Exception:
                pass
        return {}

    def _save(self):
        try:
            with open(self.state_file, "w", encoding="utf-8") as f:
                json.dump(self.data, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    @staticmethod
    def _key(cam_id: str, lens_idx: int) -> str:
        return f"{cam_id}::L{lens_idx}"

    def _get_entry(self, key: str) -> dict:
        if key not in self.data:
            self.data[key] = {
                "motion_threshold": 900,
                "true_count": 0,
                "false_count": 0,
                "false_zones": [],
                "true_zones": [],
            }
        return self.data[key]

    def get_motion_threshold(self, cam_id: str, lens_idx: int) -> int:
        with self.lock:
            entry = self.data.get(self._key(cam_id, lens_idx))
            return int(entry.get("motion_threshold", 900)) if entry else 900

    def is_in_false_zone(self, cam_id: str, lens_idx: int, norm_box: tuple) -> bool:
        nx, ny, nw, nh = norm_box
        cx, cy = nx + nw / 2.0, ny + nh / 2.0
        with self.lock:
            entry = self.data.get(self._key(cam_id, lens_idx))
            if not entry:
                return False
            for z in entry.get("false_zones", []):
                if z["hits"] >= 2 and z["x1"] <= cx <= z["x2"] and z["y1"] <= cy <= z["y2"]:
                    return True
        return False

    def zone_bonus(self, cam_id: str, lens_idx: int, norm_box: tuple) -> float:
        nx, ny, nw, nh = norm_box
        cx, cy = nx + nw / 2.0, ny + nh / 2.0
        with self.lock:
            entry = self.data.get(self._key(cam_id, lens_idx))
            if not entry:
                return 1.0
            for z in entry.get("true_zones", []):
                if z["x1"] <= cx <= z["x2"] and z["y1"] <= cy <= z["y2"]:
                    return 1.35
        return 1.0

    def record_feedback(self, cam_id: str, lens_idx: int, verdict: str, norm_boxes: Optional[list] = None) -> dict:
        with self.lock:
            entry = self._get_entry(self._key(cam_id, lens_idx))
            if verdict == "false":
                entry["false_count"] += 1
                entry["motion_threshold"] = min(2600, int(entry["motion_threshold"]) + 150)
                for b in (norm_boxes or [])[:2]:
                    nx, ny, nw, nh = b
                    cx, cy = nx + nw / 2.0, ny + nh / 2.0
                    matched = False
                    for z in entry["false_zones"]:
                        if z["x1"] <= cx <= z["x2"] and z["y1"] <= cy <= z["y2"]:
                            z["hits"] += 1
                            matched = True
                            break
                    if not matched:
                        entry["false_zones"].append({
                            "x1": round(max(0.0, nx - 0.04), 3),
                            "y1": round(max(0.0, ny - 0.04), 3),
                            "x2": round(min(1.0, nx + nw + 0.04), 3),
                            "y2": round(min(1.0, ny + nh + 0.04), 3),
                            "hits": 1,
                        })
                        entry["false_zones"] = entry["false_zones"][-15:]
            else:
                entry["true_count"] += 1
                entry["motion_threshold"] = max(700, int(entry["motion_threshold"]) - 60)
                for b in (norm_boxes or [])[:2]:
                    nx, ny, nw, nh = b
                    entry["true_zones"].append({
                        "x1": round(max(0.0, nx - 0.05), 3),
                        "y1": round(max(0.0, ny - 0.05), 3),
                        "x2": round(min(1.0, nx + nw + 0.05), 3),
                        "y2": round(min(1.0, ny + nh + 0.05), 3),
                    })
                    entry["true_zones"] = entry["true_zones"][-20:]
            self._save()
            return dict(entry)

    def get_summary(self) -> dict:
        with self.lock:
            total_true = sum(int(v.get("true_count", 0)) for v in self.data.values())
            total_false = sum(int(v.get("false_count", 0)) for v in self.data.values())
            zones_blocked = sum(
                1 for v in self.data.values() for z in v.get("false_zones", []) if z.get("hits", 0) >= 2
            )
            return {
                "total_true": total_true,
                "total_false": total_false,
                "zones_blocked": zones_blocked,
                "lenses": dict(self.data),
            }

    def reset(self):
        with self.lock:
            self.data = {}
            self._save()


class FaceAndPlateEnhancer:
    """Nhận diện & phóng to có chọn lọc trên đối tượng ĐANG CHUYỂN ĐỘNG: Người (từ vai lên) & Xe + Biển số."""

    def __init__(self):
        data_dir = getattr(getattr(cv2, "data", None), "haarcascades", "")
        self.face_frontal = self._load_cascade(data_dir, "haarcascade_frontalface_alt2.xml")
        self.face_profile = self._load_cascade(data_dir, "haarcascade_profileface.xml")
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
    def read_plate_text_hint(plate_crop: np.ndarray) -> str:
        """Đọc gợi ý ký tự biển số bằng Tesseract OCR (nếu đã cài tesseract-ocr trên Ubuntu)."""
        if plate_crop is None or plate_crop.size == 0 or shutil.which("tesseract") is None:
            return ""
        try:
            gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (0, 0), fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
            _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            proc = subprocess.run(
                [
                    "tesseract", "stdin", "stdout",
                    "--psm", "6",
                    "-c", "tessedit_char_whitelist=0123456789ABCDEFGHKLMNPSTUVXYZ-.",
                ],
                input=cv2.imencode(".png", bw)[1].tobytes(),
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                timeout=2.0,
            )
            raw = proc.stdout.decode("utf-8", errors="ignore").strip()
            cleaned = re.sub(r"[^0-9A-Z.-]+", " ", raw.upper()).strip()
            alnum_len = len(re.sub(r"[^0-9A-Z]", "", cleaned))
            return cleaned[:22] if alnum_len >= 5 else ""
        except Exception:
            return ""

    @staticmethod
    def _make_square_box(x1: int, y1: int, x2: int, y2: int, img_w: int, img_h: int):
        """Mở rộng khung cắt thành hình vuông 1:1 quanh tâm để hiển thị gọn đẹp trên Telegram."""
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)
        side = min(max(bw, bh), img_w, img_h)
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        sx1 = max(0, cx - side // 2)
        sy1 = max(0, cy - side // 2)
        sx2 = sx1 + side
        sy2 = sy1 + side
        if sx2 > img_w:
            sx2 = img_w
            sx1 = max(0, img_w - side)
        if sy2 > img_h:
            sy2 = img_h
            sy1 = max(0, img_h - side)
        return sx1, sy1, sx2, sy2

    @staticmethod
    def enhance_zoom_crop(crop: np.ndarray, min_width: int = 440, is_plate: bool = False) -> np.ndarray:
        """Phóng to khung vuông LANCZOS4 + tăng nét chữ biển số (không làm nhòe nét mảnh)."""
        if crop is None or crop.size == 0:
            return crop
        work = crop if is_plate else cv2.bilateralFilter(crop, d=3, sigmaColor=22, sigmaSpace=22)
        h, w = work.shape[:2]
        if w < min_width or h < min_width:
            scale = min_width / float(max(1, min(w, h)))
            work = cv2.resize(
                work,
                (max(2, int(round(w * scale))), max(2, int(round(h * scale)))),
                interpolation=cv2.INTER_LANCZOS4,
            )

        lab = cv2.cvtColor(work, cv2.COLOR_BGR2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.4 if is_plate else 1.35, tileGridSize=(4, 4))
        l_eq = clahe.apply(l_ch)
        balanced = cv2.cvtColor(cv2.merge((l_eq, a_ch, b_ch)), cv2.COLOR_LAB2BGR)

        blur = cv2.GaussianBlur(balanced, (0, 0), sigmaX=1.0 if is_plate else 1.1)
        alpha = 1.45 if is_plate else 1.18
        return cv2.addWeighted(balanced, alpha, blur, -(alpha - 1.0), 0)

    @staticmethod
    def _clamp_box(x1: int, y1: int, x2: int, y2: int, img_w: int, img_h: int):
        return max(0, x1), max(0, y1), min(img_w, x2), min(img_h, y2)

    @staticmethod
    def _is_green_foliage(crop: np.ndarray) -> bool:
        if crop is None or crop.size == 0:
            return True
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        green_mask = cv2.inRange(hsv, (32, 35, 30), (92, 255, 255))
        return (cv2.countNonZero(green_mask) / float(crop.shape[0] * crop.shape[1])) > 0.35

    @staticmethod
    def _is_red_object(crop: np.ndarray) -> bool:
        if crop is None or crop.size == 0:
            return False
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        r1 = cv2.inRange(hsv, (0, 80, 70), (14, 255, 255))
        r2 = cv2.inRange(hsv, (165, 80, 70), (180, 255, 255))
        red_pixels = cv2.countNonZero(r1) + cv2.countNonZero(r2)
        return (red_pixels / float(crop.shape[0] * crop.shape[1])) > 0.22

    def _find_plate_in_vehicle(self, veh_crop: np.ndarray):
        vh, vw = veh_crop.shape[:2]
        if vw < 40 or vh < 30:
            return None
        gray = cv2.cvtColor(veh_crop, cv2.COLOR_BGR2GRAY)
        best_box = None
        best_score = 0.0

        # 1) Tìm các vùng ứng viên bằng BlackHat + SobelX (lọc bỏ đèn hậu đỏ & lá xanh)
        rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 5))
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rect_kernel)
        grad_x = np.absolute(cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=3))
        max_v = float(np.max(grad_x)) or 1.0
        grad_x = np.uint8(255.0 * (grad_x / max_v))
        grad_x = cv2.morphologyEx(cv2.GaussianBlur(grad_x, (3, 3), 0), cv2.MORPH_CLOSE, rect_kernel)
        _, thresh = cv2.threshold(grad_x, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cand_rects = [cv2.boundingRect(c) for c in sorted(cnts, key=cv2.contourArea, reverse=True)[:10]]
        if self.plate_cascade is not None:
            for rect in self.plate_cascade.detectMultiScale(gray, 1.1, 3, minSize=(24, 12)):
                cand_rects.append(tuple(int(v) for v in rect))

        for (cx, cy, cw, ch) in cand_rects:
            ratio = cw / float(max(1, ch))
            area = cw * ch
            if cy < int(vh * 0.28) or area < 260 or area > int(vw * vh * 0.28):
                continue
            if not (1.0 <= ratio <= 4.8):
                continue
            roi = veh_crop[cy : cy + ch, cx : cx + cw]
            if roi.size == 0 or self._is_green_foliage(roi) or self._is_red_object(roi):
                continue
            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            # Biển số trắng/bạc hoặc vàng có độ sáng cao & tương phản chữ đen rõ
            bright_neutral = cv2.inRange(hsv_roi, (0, 0, 125), (180, 75, 255))
            bright_yellow = cv2.inRange(hsv_roi, (15, 70, 115), (38, 255, 255))
            plate_bg_ratio = (cv2.countNonZero(bright_neutral) + cv2.countNonZero(bright_yellow)) / float(area)
            contrast = float(np.std(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)))
            if plate_bg_ratio < 0.22 or contrast < 28.0:
                continue
            score = plate_bg_ratio * contrast * min(2.0, area / 600.0)
            if score > best_score:
                best_score = score
                best_box = self._clamp_box(cx - 14, cy - 10, cx + cw + 14, cy + ch + 10, vw, vh)

        return best_box

    def extract_detail_crops(
        self,
        img: np.ndarray,
        prefix: str,
        moving_boxes: Optional[list] = None,
    ) -> list:
        """Chỉ cắt từ vùng ĐANG CHUYỂN ĐỘNG: Người -> từ vai lên; Xe -> toàn xe & biển số."""
        if img is None or img.size == 0 or not moving_boxes:
            return []
        ih, iw = img.shape[:2]
        details = []

        # Sắp xếp các vùng chuyển động lớn nhất trước (bỏ qua vùng quá nhỏ)
        valid_boxes = [
            b for b in sorted(moving_boxes, key=lambda x: x[2] * x[3], reverse=True)
            if b[2] >= 28 and b[3] >= 28 and (b[2] * b[3]) >= 1400
        ][:3]

        person_count = 0
        vehicle_count = 0

        for (bx, by, bw, bh) in valid_boxes:
            # Bỏ qua vùng góc trên cùng nơi có chữ đồng hồ OSD hoặc lá cây
            if by < int(ih * 0.08) and bh < int(ih * 0.16):
                continue
            roi_check = img[max(0, by) : min(ih, by + bh), max(0, bx) : min(iw, bx + bw)]
            if self._is_green_foliage(roi_check):
                continue

            ratio = bw / float(max(1, bh))
            area = bw * bh

            # Chuẩn bị khung mở rộng để kiểm tra xem có phải người đi xe / phương tiện không
            pad_vx = int(bw * 0.18)
            pad_vy = int(bh * 0.18)
            vx1, vy1, vx2, vy2 = self._clamp_box(bx - pad_vx, by - pad_vy, bx + bw + pad_vx, by + bh + pad_vy, iw, ih)
            veh_crop = img[vy1:vy2, vx1:vx2]
            plate_box = self._find_plate_in_vehicle(veh_crop) if (veh_crop.size > 0 and area >= 3600) else None

            # Phân loại kết hợp:
            # - Người đi bộ đứng thẳng: ratio < 0.72 và không có biển số -> chỉ cắt người (từ vai lên)
            # - Người đang chạy xe (hoặc xe máy/ô tô): ratio >= 0.72 HOẶC phát hiện biển số -> cắt CẢ người (từ vai lên) LẪN xe + biển số!
            is_rider_or_vehicle = (ratio >= 0.72) or (plate_box is not None)
            has_person_top = (ratio <= 1.55) or (plate_box is not None and bh >= 65)

            if has_person_top and person_count < 2:
                pad_x = int(bw * 0.32)
                top_y = max(0, by - int(bh * 0.35))
                shoulder_y = min(ih, by + max(42, int(bh * 0.45)))
                x1, y1, x2, y2 = self._clamp_box(bx - pad_x, top_y, bx + bw + pad_x, shoulder_y, iw, ih)
                x1, y1, x2, y2 = self._make_square_box(x1, y1, x2, y2, iw, ih)
                if (x2 - x1) >= 24 and (y2 - y1) >= 24:
                    person_count += 1
                    enhanced = self.enhance_zoom_crop(img[y1:y2, x1:x2], min_width=440, is_plate=False)
                    out_file = f"snapshots/{prefix}_person{person_count}.jpg"
                    cv2.imwrite(out_file, enhanced, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
                    details.append({
                        "kind": "face",
                        "label": "👤 Người / Người lái (Từ vai lên)",
                        "url": f"/{out_file}",
                    })

            if is_rider_or_vehicle and vehicle_count < 1 and veh_crop.size > 0 and (vx2 - vx1) >= 48:
                svx1, svy1, svx2, svy2 = self._make_square_box(vx1, vy1, vx2, vy2, iw, ih)
                veh_sq = img[svy1:svy2, svx1:svx2]
                vehicle_count += 1
                enhanced_veh = self.enhance_zoom_crop(veh_sq, min_width=480, is_plate=False)
                out_veh = f"snapshots/{prefix}_vehicle{vehicle_count}.jpg"
                cv2.imwrite(out_veh, enhanced_veh, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
                details.append({
                    "kind": "vehicle",
                    "label": "🏍️/🚗 Phương tiện di chuyển",
                    "url": f"/{out_veh}",
                })
                if plate_box is not None:
                    px1, py1, px2, py2 = plate_box
                    gpx1, gpy1, gpx2, gpy2 = self._make_square_box(
                        vx1 + px1, vy1 + py1, vx1 + px2, vy1 + py2, iw, ih
                    )
                    enhanced_plate = self.enhance_zoom_crop(img[gpy1:gpy2, gpx1:gpx2], min_width=440, is_plate=True)
                    out_plate = f"snapshots/{prefix}_plate{vehicle_count}.jpg"
                    cv2.imwrite(out_plate, enhanced_plate, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
                    plate_hint = self.read_plate_text_hint(enhanced_plate)
                    lbl = f"🔢 Biển số xe (Gợi ý: {plate_hint})" if plate_hint else "🔢 Biển số xe (Phóng to)"
                    details.append({
                        "kind": "plate",
                        "label": lbl,
                        "text_hint": plate_hint,
                        "url": f"/{out_plate}",
                    })

        return details


class RollingH264Writer:
    """Ghi video dài (30-60 phút/tệp) bất đồng bộ bằng hàng đợi riêng để không bao giờ làm trễ luồng trực tiếp."""

    def __init__(self, cam_id: str, segment_seconds: int = 1800):
        self.cam_id = cam_id
        self.segment_seconds = max(1800, min(3600, segment_seconds))
        self.proc = None
        self.seg_start = 0.0
        self.out_dir = os.path.join("recordings", cam_id)
        os.makedirs(self.out_dir, exist_ok=True)
        self._q = queue.Queue(maxsize=2)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._writer_loop, daemon=True)
        self._thread.start()

    def _open_new_segment(self):
        self._close_proc()
        ts_name = now_vn().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(self.out_dir, f"continuous_{ts_name}.mp4")
        cmd = [
            "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
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

    def _writer_loop(self):
        while not self._stop.is_set():
            try:
                jpg_bytes = self._q.get(timeout=0.5)
            except queue.Empty:
                continue
            if jpg_bytes is None:
                break
            now = time.time()
            if self.proc is None or self.proc.poll() is not None or (now - self.seg_start >= self.segment_seconds):
                self._open_new_segment()
            if self.proc and self.proc.stdin:
                try:
                    self.proc.stdin.write(jpg_bytes)
                    self.proc.stdin.flush()
                except Exception:
                    self._close_proc()

    def write_frame(self, jpg_bytes: bytes):
        if self._stop.is_set():
            return
        if self._q.full():
            try:
                self._q.get_nowait()
            except queue.Empty:
                pass
        try:
            self._q.put_nowait(jpg_bytes)
        except queue.Full:
            pass

    def _close_proc(self):
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

    def close(self):
        self._stop.set()
        try:
            self._q.put_nowait(None)
        except Exception:
            pass
        self._close_proc()


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
        self.learner = AdaptiveAILearner()

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
            "-vf", "scale='min(1920,iw)':-2,format=yuvj420p",
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
                latest_jpg = None
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
                    latest_jpg = bytes(buf[start : end + 2])
                    del buf[: end + 2]
                if latest_jpg is not None:
                    yield latest_jpg
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
                "burst_active": False,
                "burst_start": 0.0,
                "burst_best": None,
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
        learned_thresh = self.learner.get_motion_threshold(cam_id, lens_idx)
        has_motion = cv2.countNonZero(fg_clean) > learned_thresh

        person_boxes = []
        norm_boxes = []
        if has_motion:
            cnts, _ = cv2.findContours(fg_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            ih, iw = lens_img.shape[:2]
            sx, sy = iw / 480.0, ih / 270.0
            for c in cnts:
                if cv2.contourArea(c) >= 1200:
                    bx, by, bw, bh = cv2.boundingRect(c)
                    rx, ry, rw, rh = int(bx * sx), int(by * sy), int(bw * sx), int(bh * sy)
                    nb = (rx / float(max(1, iw)), ry / float(max(1, ih)), rw / float(max(1, iw)), rh / float(max(1, ih)))
                    if not self.learner.is_in_false_zone(cam_id, lens_idx, nb):
                        person_boxes.append((rx, ry, rw, rh))
                        norm_boxes.append(nb)
            if not person_boxes:
                has_motion = False

        # Thu thập chuỗi khung hình trong 3.5 giây khi có chuyển động để chọn ảnh sắc nét & đầy đủ nhất trước khi gửi Telegram
        if has_motion and not st["burst_active"] and (now - st["last_motion_alert"] > 30.0):
            st["burst_active"] = True
            st["burst_start"] = now
            st["burst_best"] = None
            cm = self.clip_managers.get(cam_id)
            if cm is not None:
                cm.trigger_clip("motion")

        if st["burst_active"]:
            if person_boxes:
                ih, iw = lens_img.shape[:2]
                bx, by, bw, bh = max(person_boxes, key=lambda b: b[2] * b[3])
                roi = lens_img[max(0, by) : min(ih, by + bh), max(0, bx) : min(iw, bx + bw)]
                sharp = float(cv2.Laplacian(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()) if roi.size > 0 else 0.0
                # Ưu tiên khung hình đối tượng nằm gọn trong ảnh (không bị cắt nửa người/xe ở sát mép) + độ nét cao
                margin_ok = 1.0 if (bx > 8 and by > 8 and (bx + bw) < (iw - 8) and (by + bh) < (ih - 8)) else 0.45
                nb_best = (bx / float(max(1, iw)), by / float(max(1, ih)), bw / float(max(1, iw)), bh / float(max(1, ih)))
                z_bonus = self.learner.zone_bonus(cam_id, lens_idx, nb_best)
                score = (bw * bh) * margin_ok * z_bonus * (1.0 + min(3.0, sharp / 120.0))
                if st["burst_best"] is None or score > st["burst_best"]["score"]:
                    st["burst_best"] = {
                        "score": score,
                        "img": lens_img.copy(),
                        "jpg": lens_jpg,
                        "boxes": list(person_boxes),
                        "norm_boxes": list(norm_boxes),
                    }

            if (now - st["burst_start"]) >= 3.5:
                best = st["burst_best"] or {
                    "img": lens_img,
                    "jpg": lens_jpg,
                    "boxes": person_boxes,
                }
                st["burst_active"] = False
                st["burst_best"] = None
                st["last_motion_alert"] = now

                ts_ms = int(now * 1000)
                snap_file = f"snapshots/motion_{cam_id}_L{lens_idx}_{ts_ms}.jpg"
                with open(snap_file, "wb") as f:
                    f.write(best["jpg"])
                details = self.enhancer.extract_detail_crops(
                    best["img"], f"motion_{cam_id}_L{lens_idx}_{ts_ms}", best["boxes"]
                )
                self.on_alert({
                    "id": f"motion_{cam_id}_{lens_idx}_{ts_ms}_{uuid.uuid4().hex[:6]}",
                    "type": "motion",
                    "cam_id": cam_id,
                    "lens_idx": lens_idx,
                    "cam_name": label_name,
                    "message": f"Phát hiện chuyển động tại {label_name}",
                    "snapshot": f"/{snap_file}",
                    "detail_snapshots": details,
                    "norm_boxes": best.get("norm_boxes", []),
                    "feedback": None,
                    "timestamp": now_vn().strftime("%Y-%m-%d %H:%M:%S"),
                })

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
                        "lens_idx": lens_idx,
                        "cam_name": label_name,
                        "message": f"KHẨN CẤP: Phát hiện người bị té ngã tại {label_name}!",
                        "snapshot": f"/{snap_file}",
                        "detail_snapshots": details,
                        "norm_boxes": norm_boxes,
                        "feedback": None,
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
                ok_c, buf_c = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
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
