import os
import re
import time
import queue
import math
import json
import uuid
import shutil
import socket
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

YOLO_ONNX_URL = (
    "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n.onnx"
)
YOLO_ONNX_PATH = os.path.join("models", "yolov5n.onnx")


class SmartObjectDNN:
    """Bộ nhận diện Deep Learning nhẹ (YOLOv5n ONNX qua cv2.dnn): Người, Xe máy, Ô tô, Xe tải."""

    # COCO class IDs quan tâm: 0=person, 1=bicycle, 2=car, 3=motorcycle, 5=bus, 7=truck
    TARGET_CLASSES = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

    def __init__(self, model_path: str = YOLO_ONNX_PATH):
        self.model_path = model_path
        self.net = None
        self.lock = threading.Lock()
        self.input_size = 416
        self._try_load()

    def _try_load(self) -> bool:
        if self.net is not None:
            return True
        if not os.path.exists(self.model_path) or os.path.getsize(self.model_path) < 100000:
            return False
        try:
            net = cv2.dnn.readNetFromONNX(self.model_path)
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            with self.lock:
                self.net = net
            return True
        except Exception:
            return False

    @property
    def is_ready(self) -> bool:
        return self.net is not None or self._try_load()

    def detect(self, bgr_img: np.ndarray, conf_thresh: float = 0.32, nms_thresh: float = 0.45) -> list:
        if not self.is_ready or bgr_img is None or bgr_img.size == 0:
            return []
        ih, iw = bgr_img.shape[:2]
        sz = self.input_size
        # Letterbox vuông để không méo tỷ lệ người/xe
        scale = min(sz / float(max(1, iw)), sz / float(max(1, ih)))
        nw, nh = max(1, int(round(iw * scale))), max(1, int(round(ih * scale)))
        resized = cv2.resize(bgr_img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((sz, sz, 3), 114, dtype=np.uint8)
        pad_x = (sz - nw) // 2
        pad_y = (sz - nh) // 2
        canvas[pad_y : pad_y + nh, pad_x : pad_x + nw] = resized

        blob = cv2.dnn.blobFromImage(canvas, 1.0 / 255.0, (sz, sz), swapRB=True, crop=False)
        try:
            with self.lock:
                self.net.setInput(blob)
                preds = self.net.forward()
        except Exception:
            return []

        if preds is None or preds.size == 0:
            return []
        arr = preds[0] if preds.ndim == 3 else preds
        if arr.ndim != 2 or arr.shape[1] < 13:
            return []

        obj_conf = arr[:, 4]
        mask = obj_conf >= conf_thresh
        arr = arr[mask]
        if arr.shape[0] == 0:
            return []

        cls_scores = arr[:, 5:]
        cls_ids = np.argmax(cls_scores, axis=1)
        cls_confs = cls_scores[np.arange(arr.shape[0]), cls_ids]
        scores = arr[:, 4] * cls_confs

        boxes = []
        confs = []
        labels = []
        for i in range(arr.shape[0]):
            cid = int(cls_ids[i])
            sc = float(scores[i])
            if cid not in self.TARGET_CLASSES or sc < conf_thresh:
                continue
            cx, cy, bw, bh = float(arr[i, 0]), float(arr[i, 1]), float(arr[i, 2]), float(arr[i, 3])
            x1 = int(round(((cx - bw / 2.0) - pad_x) / scale))
            y1 = int(round(((cy - bh / 2.0) - pad_y) / scale))
            x2 = int(round(((cx + bw / 2.0) - pad_x) / scale))
            y2 = int(round(((cy + bh / 2.0) - pad_y) / scale))
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(iw, x2), min(ih, y2)
            if x2 - x1 >= 18 and y2 - y1 >= 18:
                boxes.append([x1, y1, x2 - x1, y2 - y1])
                confs.append(sc)
                labels.append(self.TARGET_CLASSES[cid])

        if not boxes:
            return []
        idxs = cv2.dnn.NMSBoxes(boxes, confs, conf_thresh, nms_thresh)
        results = []
        if len(idxs) > 0:
            for idx in np.array(idxs).flatten():
                results.append({
                    "cls": labels[idx],
                    "conf": round(confs[idx], 3),
                    "box": tuple(boxes[idx]),
                })
        return results


os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp|stimeout;5000000")






try:
    from PIL import Image, ImageDraw, ImageFont
    HAS_PIL = True
except Exception:
    Image = ImageDraw = ImageFont = None
    HAS_PIL = False


class AIOverlayRenderer:
    """Vẽ khung nhận diện AI, khung xương (nếu có Pose) & nhãn tiếng Việt lên khung hình."""

    POSE_CONNECTIONS = (
        (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
        (11, 23), (12, 24), (23, 24), (23, 25), (25, 27),
        (24, 26), (26, 28), (0, 11), (0, 12),
    )

    COLORS_BGR = {
        "fall": (58, 58, 239),       # Đỏ nổi bật: Người té ngã
        "rider": (11, 158, 245),     # Cam: Người đi xe
        "vehicle": (246, 130, 59),   # Xanh dương: Xe
        "person": (129, 185, 16),    # Xanh ngọc: Người di chuyển
    }

    ASCII_LABELS = {
        "Người té ngã": "Nguoi te nga",
        "Người đi xe": "Nguoi di xe",
        "Xe": "Xe",
        "Ô tô": "O to",
        "Xe tải": "Xe tai",
        "Người di chuyển": "Nguoi di chuyen",
        "Người đi xe (Đầu xe)": "Nguoi di xe (Dau xe)",
        "Người đi xe (Đuôi xe)": "Nguoi di xe (Duoi xe)",
        "Ô tô (Đầu xe)": "O to (Dau xe)",
        "Ô tô (Đuôi xe)": "O to (Duoi xe)",
        "Biển số": "Bien so",
    }

    def __init__(self):
        self._fonts = {}
        self._font_path = self._find_unicode_font()

    @staticmethod
    def _find_unicode_font() -> Optional[str]:
        candidates = (
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
            "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf",
            "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
            "/system/fonts/Roboto-Bold.ttf",
            "C:/Windows/Fonts/arialbd.ttf",
            "C:/Windows/Fonts/segoeui.ttf",
            "C:/Windows/Fonts/tahomabd.ttf",
        )
        for p in candidates:
            if os.path.exists(p):
                return p
        return None

    def _get_font(self, size: int):
        if not HAS_PIL or not self._font_path:
            return None
        if size not in self._fonts:
            try:
                self._fonts[size] = ImageFont.truetype(self._font_path, size)
            except Exception:
                self._fonts[size] = None
        return self._fonts[size]

    @staticmethod
    def _draw_corner_box(img: np.ndarray, x1: int, y1: int, x2: int, y2: int, color: tuple, thickness: int = 3):
        # Khung phân vùng chính dày hơn +1px (2px thân khung + 4-5px góc) tập trung rõ vật thể
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        cl = max(10, min(26, (x2 - x1) // 4, (y2 - y1) // 4))
        for (px, py, dx, dy) in (
            (x1, y1, 1, 1), (x2, y1, -1, 1),
            (x1, y2, 1, -1), (x2, y2, -1, -1),
        ):
            cv2.line(img, (px, py), (px + dx * cl, py), color, thickness + 1, cv2.LINE_AA)
            cv2.line(img, (px, py), (px, py + dy * cl), color, thickness + 1, cv2.LINE_AA)

    @staticmethod
    def _draw_fallback_posture(img: np.ndarray, x1: int, y1: int, x2: int, y2: int, kind: str, color: tuple):
        """Vẽ khung xương định hướng trực quan khi chạy OpenCV (chưa có MediaPipe)."""
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)
        cx = (x1 + x2) // 2
        if kind == "person":
            head_y = y1 + max(6, int(bh * 0.14))
            sh_y = y1 + int(bh * 0.28)
            hip_y = y1 + int(bh * 0.56)
            foot_y = y2 - max(4, int(bh * 0.06))
            r = max(4, min(12, int(bw * 0.16)))
            cv2.circle(img, (cx, head_y), r, color, 2, cv2.LINE_AA)
            cv2.line(img, (cx - int(bw * 0.24), sh_y), (cx + int(bw * 0.24), sh_y), color, 2, cv2.LINE_AA)
            cv2.line(img, (cx, head_y + r), (cx, hip_y), color, 2, cv2.LINE_AA)
            cv2.line(img, (cx, hip_y), (cx - int(bw * 0.18), foot_y), color, 2, cv2.LINE_AA)
            cv2.line(img, (cx, hip_y), (cx + int(bw * 0.18), foot_y), color, 2, cv2.LINE_AA)
        elif kind == "fall":
            cy = (y1 + y2) // 2
            head_x = x1 + max(8, int(bw * 0.15))
            sh_x = x1 + int(bw * 0.30)
            hip_x = x1 + int(bw * 0.60)
            foot_x = x2 - max(6, int(bw * 0.08))
            r = max(4, min(12, int(bh * 0.18)))
            cv2.circle(img, (head_x, cy), r, color, 2, cv2.LINE_AA)
            cv2.line(img, (head_x + r, cy), (hip_x, cy), color, 2, cv2.LINE_AA)
            cv2.line(img, (sh_x, cy - int(bh * 0.22)), (sh_x, cy + int(bh * 0.22)), color, 2, cv2.LINE_AA)
            cv2.line(img, (hip_x, cy), (foot_x, cy - int(bh * 0.18)), color, 2, cv2.LINE_AA)
            cv2.line(img, (hip_x, cy), (foot_x, cy + int(bh * 0.18)), color, 2, cv2.LINE_AA)

    def render(self, frame: np.ndarray, detections: list, skeleton: Optional[list] = None, y_offset: int = 0, region_h: Optional[int] = None) -> np.ndarray:
        if frame is None or frame.size == 0 or (not detections and not skeleton):
            return frame
        out = frame.copy()
        ih, iw = out.shape[:2]
        rh = region_h or ih

        # 1. Vẽ khung xương MediaPipe Pose nếu có
        if skeleton:
            pts = {}
            for idx, (nx, ny, vis) in enumerate(skeleton):
                if vis >= 0.35:
                    px = int(max(0, min(iw - 1, nx * iw)))
                    py = int(max(0, min(ih - 1, y_offset + ny * rh)))
                    pts[idx] = (px, py)
            skel_color = (58, 58, 239) if any(d.get("kind") == "fall" for d in detections) else (0, 235, 160)
            for a, b in self.POSE_CONNECTIONS:
                if a in pts and b in pts:
                    cv2.line(out, pts[a], pts[b], skel_color, 2, cv2.LINE_AA)
            for pt in pts.values():
                cv2.circle(out, pt, 3, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(out, pt, 3, skel_color, 1, cv2.LINE_AA)

        # 2. Vẽ khung bao & khung xương dẫn hướng
        label_jobs = []
        for det in detections:
            bx, by, bw, bh = det["box"]
            x1 = max(0, bx)
            y1 = max(0, y_offset + by)
            x2 = min(iw - 1, bx + bw)
            y2 = min(ih - 1, y_offset + by + bh)
            if x2 <= x1 + 4 or y2 <= y1 + 4:
                continue
            kind = det.get("kind", "person")
            color = self.COLORS_BGR.get(kind, (129, 185, 16))
            self._draw_corner_box(out, x1, y1, x2, y2, color, thickness=4 if kind == "fall" else 3)
            if not skeleton and kind in ("person", "fall"):
                self._draw_fallback_posture(out, x1, y1, x2, y2, kind, color)
            # Vẽ phân vùng phụ chi tiết: Khuôn mặt (từ vai lên), Vùng đầu/đuôi xe, Vùng biển số xe
            hb = det.get("head_box")
            if hb:
                hx, hy, hw, hh = hb
                hx1, hy1 = max(0, hx), max(0, y_offset + hy)
                hx2, hy2 = min(iw - 1, hx + hw), min(ih - 1, y_offset + hy + hh)
                if hx2 > hx1 + 6 and hy2 > hy1 + 6:
                    cv2.rectangle(out, (hx1, hy1), (hx2, hy2), (248, 189, 56), 2, cv2.LINE_AA)
            pb = det.get("plate_box")
            if pb:
                px, py, pw, ph = pb
                px1, py1 = max(0, px), max(0, y_offset + py)
                px2, py2 = min(iw - 1, px + pw), min(ih - 1, y_offset + py + ph)
                if px2 > px1 + 6 and py2 > py1 + 4:
                    cv2.rectangle(out, (px1, py1), (px2, py2), (21, 204, 250), 2, cv2.LINE_AA)
                    label_jobs.append((px1, py1, "Biển số", (21, 175, 235)))
            label_jobs.append((x1, y1, det.get("label", "Người di chuyển"), color))

        # 3. Vẽ nhãn chữ tiếng Việt có dấu (PIL) hoặc fallback ASCII (OpenCV)
        font_size = 15 if iw >= 640 else 13
        font = self._get_font(font_size)
        if font is not None and label_jobs:
            pil_img = Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_img)
            for (x1, y1, lbl, color_bgr) in label_jobs:
                color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])
                bbox = draw.textbbox((0, 0), lbl, font=font)
                tw = (bbox[2] - bbox[0]) + 12
                th = (bbox[3] - bbox[1]) + 8
                ly1 = max(0, y1 - th - 2) if y1 >= th + 4 else min(ih - th - 1, y1 + 2)
                lx2 = min(iw - 1, x1 + tw)
                draw.rectangle([x1, ly1, lx2, ly1 + th], fill=color_rgb)
                draw.text((x1 + 6, ly1 + 3), lbl, font=font, fill=(255, 255, 255))
            out = cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2BGR)
        else:
            for (x1, y1, lbl, color_bgr) in label_jobs:
                txt = self.ASCII_LABELS.get(lbl, lbl)
                (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                ly1 = max(0, y1 - th - 10) if y1 >= th + 12 else min(ih - th - 10, y1 + 2)
                cv2.rectangle(out, (x1, ly1), (min(iw - 1, x1 + tw + 10), ly1 + th + 8), color_bgr, -1)
                cv2.putText(out, txt, (x1 + 5, ly1 + th + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        return out


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
                "plate_min_bright": 0.34,
                "true_count": 0,
                "false_count": 0,
                "plate_wrong_count": 0,
                "false_zones": [],
                "true_zones": [],
                "plate_bad_zones": [],
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

    def get_plate_min_bright(self, cam_id: str, lens_idx: int) -> float:
        with self.lock:
            entry = self.data.get(self._key(cam_id, lens_idx))
            return float(entry.get("plate_min_bright", 0.34)) if entry else 0.34

    def is_plate_bad_zone(self, cam_id: str, lens_idx: int, rel_box: tuple) -> bool:
        rx, ry, rw, rh = rel_box
        cx, cy = rx + rw / 2.0, ry + rh / 2.0
        with self.lock:
            entry = self.data.get(self._key(cam_id, lens_idx))
            if not entry:
                return False
            for z in entry.get("plate_bad_zones", []):
                if z["x1"] <= cx <= z["x2"] and z["y1"] <= cy <= z["y2"]:
                    return True
        return False

    def record_feedback(self, cam_id: str, lens_idx: int, verdict: str, norm_boxes: Optional[list] = None, plate_rel_boxes: Optional[list] = None) -> dict:
        with self.lock:
            entry = self._get_entry(self._key(cam_id, lens_idx))
            if verdict == "plate_wrong":
                entry["plate_wrong_count"] = int(entry.get("plate_wrong_count", 0)) + 1
                entry["plate_min_bright"] = round(min(0.55, float(entry.get("plate_min_bright", 0.34)) + 0.05), 2)
                for pb in (plate_rel_boxes or [])[:2]:
                    rx, ry, rw, rh = pb
                    entry.setdefault("plate_bad_zones", []).append({
                        "x1": round(max(0.0, rx - 0.05), 3),
                        "y1": round(max(0.0, ry - 0.05), 3),
                        "x2": round(min(1.0, rx + rw + 0.05), 3),
                        "y2": round(min(1.0, ry + rh + 0.05), 3),
                    })
                    entry["plate_bad_zones"] = entry["plate_bad_zones"][-12:]
                self._save()
                return dict(entry)
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

    def _find_plate_in_vehicle(self, veh_crop: np.ndarray, min_bright: float = 0.34, cam_id: str = "", lens_idx: int = 1, learner=None):
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
            # Biển số trắng/bạc hoặc vàng có độ sáng cao & tương phản chữ đen rõ (loại trừ chóa đèn pha trắng loá)
            bright_neutral = cv2.inRange(hsv_roi, (0, 0, 125), (180, 75, 255))
            bright_yellow = cv2.inRange(hsv_roi, (15, 70, 115), (38, 255, 255))
            plate_bg_ratio = (cv2.countNonZero(bright_neutral) + cv2.countNonZero(bright_yellow)) / float(area)
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            contrast = float(np.std(roi_gray))
            overexposed = cv2.countNonZero(cv2.inRange(roi_gray, 242, 255)) / float(area)
            if overexposed > 0.55 and contrast < 48.0:
                continue
            rel_box = (cx / float(vw), cy / float(vh), cw / float(vw), ch / float(vh))
            if learner is not None and learner.is_plate_bad_zone(cam_id, lens_idx, rel_box):
                continue
            # Yêu cầu nền trắng/vàng đặc (tránh nan hoa bánh xe)
            if plate_bg_ratio < min_bright or contrast < 30.0:
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
        cam_id: str = "",
        lens_idx: int = 1,
        learner=None,
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
            min_b = learner.get_plate_min_bright(cam_id, lens_idx) if learner is not None else 0.34
            plate_box = self._find_plate_in_vehicle(veh_crop, min_bright=min_b, cam_id=cam_id, lens_idx=lens_idx, learner=learner) if (veh_crop.size > 0 and area >= 3600) else None

            # Phân loại kết hợp chặt chẽ hơn: chỉ cắt Xe + Biển số khi thực sự là phương tiện (tránh người cúi bị cắt nhầm biển số)
            is_rider_or_vehicle = (ratio >= 1.15) or (plate_box is not None)
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
                    vh_c, vw_c = veh_crop.shape[:2]
                    details.append({
                        "kind": "plate",
                        "label": lbl,
                        "text_hint": plate_hint,
                        "rel_box": [round(px1 / float(max(1, vw_c)), 3), round(py1 / float(max(1, vh_c)), 3), round((px2 - px1) / float(max(1, vw_c)), 3), round((py2 - py1) / float(max(1, vh_c)), 3)],
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
            "-c:v", "libx264", "-preset", "ultrafast", "-tune", "zerolatency", "-g", "16", "-crf", "33",
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
        self.object_dnn = SmartObjectDNN()
        self.overlay_renderer = AIOverlayRenderer()
        self.overlay_cache = {}  # (cam_id, lens_idx) -> {"ts": float, "detections": list, "skeleton": list}
        self.ai_overlay_enabled = os.environ.get("CAMHUB_AI_OVERLAY", "1") != "0"

        self.recorders = {}
        self.clip_managers = {}
        self.audio_subs = {}  # cam_id -> set of queue.Queue
        self.audio_lock = threading.Lock()
        self.long_clip_sec = int(os.environ.get("CAMHUB_LONG_CLIP_SEC", "3600"))
        self.active_engine_label = self._resolve_engine_label()
        threading.Thread(target=self._thermal_guard_loop, daemon=True).start()
        if HAS_MP_TASKS and not HAS_MP_LEGACY and not os.path.exists(POSE_TASK_PATH):
            threading.Thread(target=self._ensure_pose_task_model, daemon=True).start()
        if os.environ.get("CAMHUB_ENABLE_YOLO_DL", "1") != "0" and not self.object_dnn.is_ready:
            threading.Thread(target=self._ensure_yolo_model, daemon=True).start()

    def _resolve_engine_label(self) -> str:
        dnn_tag = " + YOLOv5n" if getattr(self, "object_dnn", None) and self.object_dnn.is_ready else ""
        if HAS_MP_LEGACY:
            return f"MediaPipe Pose{dnn_tag}"
        if HAS_MP_TASKS and os.path.exists(POSE_TASK_PATH):
            return f"MediaPipe Pose Tasks{dnn_tag}"
        return f"OpenCV Person + Fall AI{dnn_tag}"

    def _ensure_yolo_model(self):
        try:
            os.makedirs("models", exist_ok=True)
            if not os.path.exists(YOLO_ONNX_PATH) or os.path.getsize(YOLO_ONNX_PATH) < 100000:
                tmp_path = YOLO_ONNX_PATH + ".part"
                urllib.request.urlretrieve(YOLO_ONNX_URL, tmp_path)
                if os.path.exists(tmp_path) and os.path.getsize(tmp_path) > 100000:
                    os.replace(tmp_path, YOLO_ONNX_PATH)
            self.object_dnn._try_load()
            self.active_engine_label = self._resolve_engine_label()
        except Exception:
            pass

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
            if not self.has_ffmpeg_cli:
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
                self.overlay_cache.pop(k, None)
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

    @staticmethod
    def _is_night_or_ir(small_bgr: np.ndarray) -> bool:
        """Nhận biết chế độ ban đêm hoặc hồng ngoại trắng đen để bật lọc vệt đèn rọi."""
        hsv = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2HSV)
        mean_sat = float(np.mean(hsv[:, :, 1]))
        mean_val = float(np.mean(hsv[:, :, 2]))
        return (mean_sat < 16.0) or (mean_val < 92.0)

    @staticmethod
    def _has_new_foreground_edges(small_bgr: np.ndarray, ref_gray: Optional[np.ndarray], box: tuple) -> bool:
        """Ban đêm: kiểm tra có đường viền MỚI của người/xe hay chỉ là bậc cửa/nền sân cũ bị đèn rọi sáng lên."""
        if ref_gray is None:
            return True
        x, y, w, h = box
        ih, iw = small_bgr.shape[:2]
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(iw, x + w), min(ih, y + h)
        area = (x2 - x1) * (y2 - y1)
        if area < 100:
            return False
        cur_roi = cv2.cvtColor(small_bgr[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
        ref_roi = ref_gray[y1:y2, x1:x2]
        # Cân bằng độ sáng cục bộ để không tính nhầm vùng nền cũ chỉ tăng độ sáng do đèn pha
        cur_eq = cv2.normalize(cur_roi, None, 0, 255, cv2.NORM_MINMAX)
        ref_eq = cv2.normalize(ref_roi, None, 0, 255, cv2.NORM_MINMAX)
        cur_edges = cv2.Canny(cur_eq, 50, 140)
        ref_edges = cv2.Canny(ref_eq, 50, 140)
        ref_dilated = cv2.dilate(ref_edges, np.ones((5, 5), np.uint8))
        new_edges = cv2.bitwise_and(cur_edges, cv2.bitwise_not(ref_dilated))
        new_ratio = cv2.countNonZero(new_edges) / float(max(1, area))
        return new_ratio >= 0.013

    @staticmethod
    def _has_object_structure(small_bgr: np.ndarray, box: tuple) -> bool:
        """Phân biệt vật thể thật (người/xe có đường nét) với vùng loá đèn pha trên nền xi măng/tường."""
        x, y, w, h = box
        ih, iw = small_bgr.shape[:2]
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(iw, x + w), min(ih, y + h)
        if x2 - x1 < 10 or y2 - y1 < 10:
            return False
        area = (x2 - x1) * (y2 - y1)
        # Bỏ qua vùng quá lớn (> 38% khung hình) thường do đèn pha quét toàn sân hoặc đổi ngày/đêm
        if area > int(iw * ih * 0.38):
            return False
        roi = small_bgr[y1:y2, x1:x2]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 40, 120)
        edge_ratio = cv2.countNonZero(edges) / float(max(1, area))
        mean_luma = float(np.mean(gray))
        # Vùng loá sáng trắng bệt hoặc vệt sáng mịn trên nền sân có rất ít đường nét cạnh
        if mean_luma > 225.0 and edge_ratio < 0.035:
            return False
        return edge_ratio >= 0.018

    def _is_clear_person_or_vehicle(self, small_bgr: np.ndarray, small_box: tuple, full_img: np.ndarray, full_box: tuple) -> bool:
        """Kiểm tra chặt chẽ hơn khi vật thể đi vào vùng báo giả: chỉ cho qua nếu thực sự có dáng/đường nét rõ của Người hoặc Xe."""
        bx, by, bw, bh = small_box
        rx, ry, rw, rh = full_box
        ih, iw = full_img.shape[:2]
        roi_full = full_img[max(0, ry) : min(ih, ry + rh), max(0, rx) : min(iw, rx + rw)]
        if roi_full.size == 0 or self.enhancer._is_green_foliage(roi_full):
            return False

        # 1. Nếu nhận diện được dáng người bằng HOG trên vùng ảnh gốc
        if rw >= 52 and rh >= 96:
            try:
                pad_x, pad_y = int(rw * 0.15), int(rh * 0.15)
                px1, py1 = max(0, rx - pad_x), max(0, ry - pad_y)
                px2, py2 = min(iw, rx + rw + pad_x), min(ih, ry + rh + pad_y)
                hog_roi = full_img[py1:py2, px1:px2]
                if hog_roi.shape[1] >= 64 and hog_roi.shape[0] >= 128:
                    rects, _ = self.hog.detectMultiScale(hog_roi, winStride=(8, 8), padding=(4, 4), scale=1.08)
                    if len(rects) > 0:
                        return True
            except Exception:
                pass

        # 2. Hoặc vật thể đủ lớn, tỷ lệ hợp lý (người/xe) và đường nét cạnh + tương phản rõ ràng
        if (bw * bh) < 3200:
            return False
        ratio = bw / float(max(1, bh))
        if not (0.25 <= ratio <= 2.6):
            return False
        s_ih, s_iw = small_bgr.shape[:2]
        roi_s = small_bgr[max(0, by) : min(s_ih, by + bh), max(0, bx) : min(s_iw, bx + bw)]
        if roi_s.size == 0:
            return False
        gray = cv2.cvtColor(roi_s, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 45, 130)
        edge_ratio = cv2.countNonZero(edges) / float(max(1, roi_s.shape[0] * roi_s.shape[1]))
        contrast = float(np.std(gray))
        return edge_ratio >= 0.045 and contrast >= 32.0

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
        fx, y, w, h = cv2.boundingRect(largest)
        if not self._has_object_structure(small_bgr, (fx, y, w, h)):
            return False
        if st.get("is_night") and not self._has_new_foreground_edges(small_bgr, st.get("ref_gray"), (fx, y, w, h)):
            return False
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

    def subscribe_audio(self, cam_id: str) -> queue.Queue:
        q = queue.Queue(maxsize=32)
        with self.audio_lock:
            self.audio_subs.setdefault(cam_id, set()).add(q)
        return q

    def unsubscribe_audio(self, cam_id: str, q: queue.Queue):
        with self.audio_lock:
            subs = self.audio_subs.get(cam_id)
            if subs:
                subs.discard(q)

    def _broadcast_audio(self, cam_id: str, chunk: bytes):
        with self.audio_lock:
            subs = list(self.audio_subs.get(cam_id, ()))
        for q in subs:
            if q.full():
                try:
                    q.get_nowait()
                except queue.Empty:
                    pass
            try:
                q.put_nowait(chunk)
            except queue.Full:
                pass

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

    def _spawn_ffmpeg_mjpeg(self, rtsp_url: str, transport: str, cam_id: str, audio_port: int = 0):
        rec_dir = os.path.join("recordings", cam_id)
        os.makedirs(rec_dir, exist_ok=True)
        seg_pattern = os.path.join(rec_dir, "continuous_%Y%m%d_%H%M%S.mp4")

        cmd = [
            "ffmpeg",
            "-nostdin",
            "-noautorotate",
            "-hide_banner",
            "-loglevel", "error",
            "-fflags", "+nobuffer+discardcorrupt+genpts",
            "-flags", "low_delay",
            "-analyzeduration", "2000000",
            "-probesize", "2000000",
            "-rtsp_transport", transport,
            "-timeout", "8000000",
            "-i", rtsp_url,
            # Đầu ra 1: MJPEG cho xem trực tuyến & AI
            "-map", "0:v:0",
            "-an",
            "-vf", "select='not(mod(n,4))',scale='min(1920,iw)':-2,format=yuvj420p",
            "-fps_mode", "drop",
            "-strict", "unofficial",
            "-f", "image2pipe",
            "-vcodec", "mjpeg",
            "-q:v", "4",
            "-",
            # Đầu ra 2: Lưu video dài 60 phút/tệp chuẩn H.264 + âm thanh AAC (ép keyframe mỗi 3s để luôn xem được ngay)
            "-map", "0:v:0",
            "-map", "0:a?",
            "-vf", "select='not(mod(n,4))',scale='min(720,iw)':-2",
            "-fps_mode", "cfr", "-r", "4",
            "-c:v", "libx264", "-preset", "ultrafast", "-tune", "zerolatency",
            "-g", "12", "-keyint_min", "12", "-sc_threshold", "0", "-crf", "33",
            "-maxrate", "600k", "-bufsize", "1200k",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "48k", "-ar", "16000", "-ac", "1",
            "-f", "segment", "-segment_time", str(max(1800, min(3600, self.long_clip_sec))),
            "-strftime", "1", "-reset_timestamps", "1",
            "-segment_format_options", "movflags=+frag_keyframe+empty_moov+default_base_moof",
            seg_pattern,
        ]
        if audio_port > 0:
            cmd.extend([
                # Đầu ra 3: Âm thanh trực tuyến MP3 gửi qua UDP nội bộ
                "-map", "0:a",
                "-vn",
                "-c:a", "libmp3lame", "-b:a", "32k", "-ar", "16000", "-ac", "1",
                "-f", "mp3",
                f"udp://127.0.0.1:{audio_port}?pkt_size=1024",
            ])

        env = os.environ.copy()
        env["TZ"] = "Asia/Ho_Chi_Minh"
        return subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=0,
            env=env,
        )

    def _iter_ffmpeg_jpegs(self, rtsp_url: str, transport: str, stop_event: threading.Event, cam_id: str, with_audio: bool = True):
        audio_sock = None
        audio_port = 0
        if with_audio:
            try:
                audio_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                audio_sock.bind(("127.0.0.1", 0))
                audio_sock.settimeout(0.5)
                audio_port = audio_sock.getsockname()[1]
            except Exception:
                audio_port = 0
                if audio_sock:
                    audio_sock.close()
                    audio_sock = None

        proc = self._spawn_ffmpeg_mjpeg(rtsp_url, transport, cam_id, audio_port=audio_port)
        buf = bytearray()
        last_data_ts = [time.time()]

        if audio_sock is not None:
            def _audio_rx():
                while not stop_event.is_set() and proc.poll() is None:
                    try:
                        pkt, _ = audio_sock.recvfrom(4096)
                        if pkt:
                            self._broadcast_audio(cam_id, pkt)
                    except socket.timeout:
                        continue
                    except Exception:
                        break
            threading.Thread(target=_audio_rx, daemon=True).start()

        def _watchdog():
            while not stop_event.is_set() and proc.poll() is None:
                if time.time() - last_data_ts[0] > 20.0:
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
                chunk = os.read(proc.stdout.fileno(), 262144)
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
            if audio_sock is not None:
                try:
                    audio_sock.close()
                except Exception:
                    pass

    def _classify_moving_object(self, img: np.ndarray, box: tuple, is_fall: bool = False) -> dict:
        """Phân loại chặt chẽ: Người di chuyển (kể cả cúi/vung tay), Người đi xe (trước/sau/ngang), Ô tô, Xe tải."""
        bx, by, bw, bh = box
        ih, iw = img.shape[:2]
        if is_fall:
            return {"box": box, "kind": "fall", "label": "Người té ngã", "direction": ""}

        ratio = bw / float(max(1, bh))
        area_ratio = (bw * bh) / float(max(1, iw * ih))

        # 1) Xe tải / Ô tô: yêu cầu khối đặc lớn & bề ngang rõ rệt (tránh xe máy + quầng đèn bị nhầm thành Ô tô)
        if area_ratio >= 0.34 and bw >= int(iw * 0.45) and bh >= int(ih * 0.48):
            return {"box": box, "kind": "vehicle", "label": "Xe tải", "direction": ""}
        if area_ratio >= 0.21 and bw >= int(iw * 0.34) and ratio >= 1.12:
            return {"box": box, "kind": "vehicle", "label": "Ô tô", "direction": ""}

        # 2) Kiểm tra dấu hiệu phương tiện thực sự (đèn pha, đèn hậu đỏ, biển số, khối bánh/yếm xe)
        roi = img[max(0, by) : min(ih, by + bh), max(0, bx) : min(iw, bx + bw)]
        has_plate = False
        has_headlight = False
        has_taillight = False
        has_bike_body = False
        if roi.size > 0 and (bw * bh) >= 3200:
            rh, rw = roi.shape[:2]
            lower = roi[int(rh * 0.42) : rh, :]
            if lower.size > 0:
                has_taillight = self.enhancer._is_red_object(lower)
                gray_low = cv2.cvtColor(lower, cv2.COLOR_BGR2GRAY)
                bright_ratio = cv2.countNonZero(cv2.inRange(gray_low, 220, 255)) / float(max(1, lower.shape[0] * lower.shape[1]))
                has_headlight = (bright_ratio >= 0.04) and (not has_taillight)
            mid_lower = roi[int(rh * 0.50) : int(rh * 0.92), int(rw * 0.15) : int(rw * 0.85)]
            if mid_lower.size > 0:
                g_ml = cv2.cvtColor(mid_lower, cv2.COLOR_BGR2GRAY)
                dark_mass = cv2.countNonZero(cv2.inRange(g_ml, 0, 80)) / float(max(1, g_ml.shape[0] * g_ml.shape[1]))
                has_bike_body = (dark_mass >= 0.40) and (bw >= 56) and (bh >= 84)

            pad_x, pad_y = int(bw * 0.12), int(bh * 0.12)
            vx1, vy1 = max(0, bx - pad_x), max(0, by - pad_y)
            vx2, vy2 = min(iw, bx + bw + pad_x), min(ih, by + bh + pad_y)
            has_plate = self.enhancer._find_plate_in_vehicle(img[vy1:vy2, vx1:vx2]) is not None

        has_vehicle_cue = has_plate or has_taillight or has_headlight or has_bike_body
        if has_vehicle_cue or ratio >= 1.25:
            if ratio > 1.65 and not has_headlight:
                return {"box": box, "kind": "vehicle", "label": "Xe", "direction": "ngang"}
            direction = "sau" if (has_plate or has_taillight) else ("truoc" if has_headlight else "")
            return {"box": box, "kind": "rider", "label": "Người đi xe", "direction": direction}

        return {"box": box, "kind": "person", "label": "Người di chuyển", "direction": ""}

    @staticmethod
    def _box_iou(a: tuple, b: tuple) -> float:
        ax1, ay1, aw, ah = a
        bx1, by1, bw, bh = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax1 + aw, bx1 + bw), min(ay1 + ah, by1 + bh)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        if inter <= 0:
            return 0.0
        union = (aw * ah) + (bw * bh) - inter
        return inter / float(max(1, union))

    def _enrich_sub_rois(self, img: np.ndarray, det: dict) -> dict:
        """Gắn thêm khung phân vùng con: Khuôn mặt (từ vai lên), Đầu xe/Đuôi xe và Vùng biển số xe."""
        bx, by, bw, bh = det["box"]
        ih, iw = img.shape[:2]
        kind = det.get("kind", "person")

        # 1) Khung khuôn mặt từ vai lên cho Người di chuyển & Người đi xe
        if kind in ("person", "rider"):
            hx = max(0, bx + int(bw * 0.12))
            hy = max(0, by - int(bh * 0.04))
            hw = min(iw - hx, max(20, int(bw * 0.76)))
            hh = min(ih - hy, max(24, int(bh * (0.42 if kind == "rider" else 0.36))))
            det["head_box"] = (hx, hy, hw, hh)

        # 2) Xác định Đầu xe / Đuôi xe và Vùng biển số xe cho phương tiện
        if kind in ("rider", "vehicle"):
            pad_x, pad_y = int(bw * 0.08), int(bh * 0.08)
            vx1, vy1 = max(0, bx - pad_x), max(0, by - pad_y)
            vx2, vy2 = min(iw, bx + bw + pad_x), min(ih, by + bh + pad_y)
            veh_crop = img[vy1:vy2, vx1:vx2]
            if veh_crop.size > 0:
                pbox = self.enhancer._find_plate_in_vehicle(veh_crop)
                if pbox is not None:
                    px1, py1, px2, py2 = pbox
                    det["plate_box"] = (vx1 + px1, vy1 + py1, max(12, px2 - px1), max(10, py2 - py1))
                # Phân biệt Đầu xe (đèn pha sáng) vs Đuôi xe (đèn hậu đỏ / biển số phía sau)
                rh = veh_crop.shape[0]
                lower = veh_crop[int(rh * 0.42) : rh, :]
                is_rear = (pbox is not None) or (lower.size > 0 and self.enhancer._is_red_object(lower))
                is_front = False
                if lower.size > 0 and not is_rear:
                    gl = cv2.cvtColor(lower, cv2.COLOR_BGR2GRAY)
                    br = cv2.countNonZero(cv2.inRange(gl, 220, 255)) / float(max(1, lower.shape[0] * lower.shape[1]))
                    is_front = br >= 0.035
                base_lbl = det.get("label", "Xe")
                if base_lbl in ("Người đi xe", "Ô tô"):
                    if is_rear:
                        det["label"] = f"{base_lbl} (Đuôi xe)"
                        det["direction"] = "sau"
                    elif is_front:
                        det["label"] = f"{base_lbl} (Đầu xe)"
                        det["direction"] = "truoc"
        return det

    def _classify_boxes_for_overlay(self, img: np.ndarray, boxes: list, is_horizontal: bool = False) -> list:
        if img is None or img.size == 0 or not boxes:
            return []
        ih, iw = img.shape[:2]

        # Ưu tiên Deep Learning YOLOv5n ONNX nếu đã tải sẵn trên máy (chỉ giữ vật thể giao với vùng đang chuyển động)
        if getattr(self, "object_dnn", None) and self.object_dnn.is_ready:
            dnn_raw = self.object_dnn.detect(img, conf_thresh=0.32)
            active_dnn = [
                d for d in dnn_raw
                if any(self._box_iou(d["box"], mb) >= 0.08 for mb in boxes)
            ]
            if active_dnn:
                persons = [d for d in active_dnn if d["cls"] == "person"]
                motos = [d for d in active_dnn if d["cls"] in ("motorcycle", "bicycle")]
                cars = [d for d in active_dnn if d["cls"] == "car"]
                trucks = [d for d in active_dnn if d["cls"] in ("truck", "bus")]

                used_persons = set()
                detections = []
                for m in motos:
                    mx, my, mw, mh = m["box"]
                    best_p_idx = None
                    best_ov = 0.0
                    for pi, p in enumerate(persons):
                        if pi in used_persons:
                            continue
                        ov = self._box_iou(m["box"], p["box"])
                        if ov > best_ov:
                            best_ov = ov
                            best_p_idx = pi
                    if best_p_idx is not None and best_ov >= 0.05:
                        used_persons.add(best_p_idx)
                        px, py, pw, ph = persons[best_p_idx]["box"]
                        ux1, uy1 = min(mx, px), min(my, py)
                        ux2, uy2 = max(mx + mw, px + pw), max(my + mh, py + ph)
                        detections.append(self._enrich_sub_rois(img, {
                            "box": (ux1, uy1, ux2 - ux1, uy2 - uy1),
                            "kind": "rider",
                            "label": "Người đi xe",
                            "direction": "",
                        }))
                    else:
                        detections.append(self._enrich_sub_rois(img, {
                            "box": m["box"],
                            "kind": "rider",
                            "label": "Người đi xe",
                            "direction": "",
                        }))
                for c in cars:
                    detections.append(self._enrich_sub_rois(img, {
                        "box": c["box"], "kind": "vehicle", "label": "Ô tô", "direction": "",
                    }))
                for t in trucks:
                    detections.append(self._enrich_sub_rois(img, {
                        "box": t["box"], "kind": "vehicle", "label": "Xe tải", "direction": "",
                    }))
                for pi, p in enumerate(persons):
                    if pi in used_persons:
                        continue
                    px, py, pw, ph = p["box"]
                    is_p_fall = is_horizontal and (pw / float(max(1, ph)) > 1.25)
                    detections.append(self._enrich_sub_rois(img, {
                        "box": p["box"],
                        "kind": "fall" if is_p_fall else "person",
                        "label": "Người té ngã" if is_p_fall else "Người di chuyển",
                        "direction": "",
                    }))
                if detections:
                    return detections[:5]

        # Fallback: thuật toán phân loại hình thái + ánh sáng khi chưa có model YOLOv5n
        detections = []
        sorted_boxes = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)[:5]
        for idx_b, (bx, by, bw, bh) in enumerate(sorted_boxes):
            if bw < 26 or bh < 26 or (bw * bh) < 1200:
                continue
            if by < int(ih * 0.08) and bh < int(ih * 0.16):
                continue
            roi_chk = img[max(0, by) : min(ih, by + bh), max(0, bx) : min(iw, bx + bw)]
            if self.enhancer._is_green_foliage(roi_chk):
                continue
            det = self._classify_moving_object(img, (bx, by, bw, bh), is_fall=(is_horizontal and idx_b == 0))
            detections.append(self._enrich_sub_rois(img, det))
        return detections

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
        cur_gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        cur_luma = float(np.mean(cur_gray))
        prev_luma = st.get("prev_luma")
        st["prev_luma"] = cur_luma
        luma_jump = (prev_luma is not None) and (abs(cur_luma - prev_luma) > 18.0)
        is_night = self._is_night_or_ir(small)
        st["is_night"] = is_night

        h_start = st["horizontal_start_time"]
        lr = 0.25 if luma_jump else (0.0 if (h_start is not None and (now - h_start) < 10.0) else -1.0)
        fg_mask = st["bg"].apply(small, learningRate=lr)
        if st["frames_seen"] <= 6:
            st["ref_gray"] = cur_gray.copy()
            return
        fg_clean = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        fg_pixels = cv2.countNonZero(fg_clean)
        # Bỏ qua khi ánh sáng đèn pha quét ngập >45% khung hình hoặc độ sáng toàn cảnh nhảy vọt
        if luma_jump or (fg_pixels > int(480 * 270 * 0.45)):
            st["burst_active"] = False
            st["burst_best"] = None
            st["ref_gray"] = cv2.addWeighted(st.get("ref_gray", cur_gray), 0.7, cur_gray, 0.3, 0)
            return

        learned_thresh = self.learner.get_motion_threshold(cam_id, lens_idx)
        has_motion = fg_pixels > learned_thresh
        if not has_motion:
            st["ref_gray"] = cv2.addWeighted(st.get("ref_gray", cur_gray), 0.85, cur_gray, 0.15, 0)

        person_boxes = []
        norm_boxes = []
        if has_motion:
            cnts, _ = cv2.findContours(fg_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            ih, iw = lens_img.shape[:2]
            sx, sy = iw / 480.0, ih / 270.0
            for c in cnts:
                if cv2.contourArea(c) >= 1200:
                    bx, by, bw, bh = cv2.boundingRect(c)
                    if not self._has_object_structure(small, (bx, by, bw, bh)):
                        continue
                    if is_night and not self._has_new_foreground_edges(small, st.get("ref_gray"), (bx, by, bw, bh)):
                        continue
                    rx, ry, rw, rh = int(bx * sx), int(by * sy), int(bw * sx), int(bh * sy)
                    nb = (rx / float(max(1, iw)), ry / float(max(1, ih)), rw / float(max(1, iw)), rh / float(max(1, ih)))
                    in_false_zone = self.learner.is_in_false_zone(cam_id, lens_idx, nb)
                    if (not in_false_zone) or self._is_clear_person_or_vehicle(small, (bx, by, bw, bh), lens_img, (rx, ry, rw, rh)):
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
                details = self.enhancer.extract_detail_crops(
                    best["img"], f"motion_{cam_id}_L{lens_idx}_{ts_ms}", best["boxes"], cam_id=cam_id, lens_idx=lens_idx, learner=self.learner
                )
                overview_jpg = best["jpg"]
                if self.ai_overlay_enabled:
                    best_dets = self._classify_boxes_for_overlay(best["img"], best["boxes"], is_horizontal=False)
                    if best_dets:
                        ann = self.overlay_renderer.render(best["img"], best_dets, None)
                        ok_ov, buf_ov = cv2.imencode(".jpg", ann, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                        if ok_ov:
                            overview_jpg = buf_ov.tobytes()
                with open(snap_file, "wb") as f:
                    f.write(overview_jpg)
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

        is_horizontal = False
        skeleton_pts = []
        if has_motion or (st["horizontal_start_time"] is not None):
            if local_pose is not None:
                rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                results = local_pose.process(rgb)
                if results and results.pose_landmarks:
                    lms = results.pose_landmarks.landmark
                    angle = self._calculate_body_angle_landmarks(lms)
                    is_horizontal = angle > 58.0
                    skeleton_pts = [(float(pt.x), float(pt.y), float(getattr(pt, "visibility", 1.0))) for pt in lms]
            elif task_landmarker is not None:
                try:
                    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                    res = task_landmarker.detect(mp_image)
                    if res and res.pose_landmarks and len(res.pose_landmarks) > 0:
                        lms = res.pose_landmarks[0]
                        angle = self._calculate_body_angle_landmarks(lms)
                        is_horizontal = angle > 58.0
                        skeleton_pts = [(float(pt.x), float(pt.y), float(getattr(pt, "visibility", 1.0))) for pt in lms]
                except Exception:
                    is_horizontal = self._detect_fall_opencv_smart(small, fg_clean, st, now)
            else:
                is_horizontal = self._detect_fall_opencv_smart(small, fg_clean, st, now)

        # Phân loại các vật thể đang di chuyển để hiển thị khung AI & nhãn trực tiếp
        ih, iw = lens_img.shape[:2]
        detections = []
        if person_boxes:
            sorted_boxes = sorted(person_boxes, key=lambda b: b[2] * b[3], reverse=True)[:5]
            for idx_b, (bx, by, bw, bh) in enumerate(sorted_boxes):
                if bw < 26 or bh < 26 or (bw * bh) < 1200:
                    continue
                if by < int(ih * 0.08) and bh < int(ih * 0.16):
                    continue
                roi_chk = lens_img[max(0, by) : min(ih, by + bh), max(0, bx) : min(iw, bx + bw)]
                if self.enhancer._is_green_foliage(roi_chk):
                    continue
                ratio = bw / float(max(1, bh))
                if is_horizontal and idx_b == 0:
                    detections.append({"box": (bx, by, bw, bh), "kind": "fall", "label": "Người té ngã"})
                elif ratio >= 0.72:
                    if ratio <= 1.55:
                        detections.append({"box": (bx, by, bw, bh), "kind": "rider", "label": "Người đi xe"})
                    else:
                        detections.append({"box": (bx, by, bw, bh), "kind": "vehicle", "label": "Xe"})
                else:
                    detections.append({"box": (bx, by, bw, bh), "kind": "person", "label": "Người di chuyển"})

        if detections or skeleton_pts:
            with self.frames_lock:
                self.overlay_cache[(cam_id, lens_idx)] = {
                    "ts": now,
                    "detections": detections,
                    "skeleton": skeleton_pts,
                }

            if is_horizontal:
                if st["horizontal_start_time"] is None:
                    st["horizontal_start_time"] = now
                elif (now - st["horizontal_start_time"]) >= 2.5 and (now - st["last_fall_alert"] > 20.0):
                    ts_ms = int(now * 1000)
                    snap_file = f"snapshots/fall_{cam_id}_L{lens_idx}_{ts_ms}.jpg"
                    details = self.enhancer.extract_detail_crops(
                        lens_img, f"fall_{cam_id}_L{lens_idx}_{ts_ms}", person_boxes, cam_id=cam_id, lens_idx=lens_idx, learner=self.learner
                    )
                    fall_jpg = lens_jpg
                    if self.ai_overlay_enabled:
                        ann_f = self.overlay_renderer.render(lens_img, detections, skeleton_pts)
                        ok_f, buf_f = cv2.imencode(".jpg", ann_f, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                        if ok_f:
                            fall_jpg = buf_f.tobytes()
                    with open(snap_file, "wb") as f:
                        f.write(fall_jpg)
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
        state["last_decode_ts"] = now

        arr = np.frombuffer(jpg_bytes, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            return

        # Kiểm tra khung hình có bị đóng băng (trùng lặp gần như tuyệt đối > 35s) không
        tiny = cv2.resize(frame, (64, 36))
        prev_tiny = state.get("prev_tiny")
        if prev_tiny is None:
            state["prev_tiny"] = tiny
            state["last_unique_ts"] = now
        else:
            diff = cv2.absdiff(prev_tiny, tiny)
            if float(np.mean(diff)) > 0.08 or int(np.max(diff)) >= 5:
                state["prev_tiny"] = tiny
                state["last_unique_ts"] = now
            elif now - state.get("last_unique_ts", now) > 35.0:
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
        if rec is not None and not self.has_ffmpeg_cli:
            rec.write_frame(jpg_bytes)
        cm = self.clip_managers.get(cam_id)
        if cm is not None and self.has_ffmpeg_cli:
            cm.push_frame(jpg_bytes)

        live_full_jpg = jpg_bytes
        live_lens_jpegs = lens_jpegs
        if self.ai_overlay_enabled:
            with self.frames_lock:
                ov_map = {
                    idx: dict(self.overlay_cache[(cam_id, idx)])
                    for idx in range(1, lens_count + 1)
                    if (cam_id, idx) in self.overlay_cache and (now - self.overlay_cache[(cam_id, idx)]["ts"]) <= 1.5
                }
            if ov_map:
                if lens_count == 1 and 1 in ov_map:
                    ann_single = self.overlay_renderer.render(lenses[0], ov_map[1]["detections"], ov_map[1]["skeleton"])
                    ok_s, buf_s = cv2.imencode(".jpg", ann_single, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
                    if ok_s:
                        live_full_jpg = buf_s.tobytes()
                        live_lens_jpegs = [(1, live_full_jpg)]
                else:
                    new_lens_jpegs = []
                    full_ann = frame.copy()
                    step_h = frame.shape[0] // lens_count
                    for idx, crop in enumerate(lenses, start=1):
                        ov = ov_map.get(idx)
                        if ov:
                            ann_l = self.overlay_renderer.render(crop, ov["detections"], ov["skeleton"])
                            ok_l, buf_l = cv2.imencode(".jpg", ann_l, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
                            if ok_l:
                                new_lens_jpegs.append((idx, buf_l.tobytes()))
                            else:
                                new_lens_jpegs.append((idx, dict(lens_jpegs).get(idx, jpg_bytes)))
                            y_off = (idx - 1) * step_h
                            full_ann = self.overlay_renderer.render(
                                full_ann, ov["detections"], ov["skeleton"], y_offset=y_off, region_h=crop.shape[0]
                            )
                        else:
                            new_lens_jpegs.append((idx, dict(lens_jpegs).get(idx, jpg_bytes)))
                    live_lens_jpegs = new_lens_jpegs
                    ok_f, buf_f = cv2.imencode(".jpg", full_ann, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
                    if ok_f:
                        live_full_jpg = buf_f.tobytes()

        with self.frames_lock:
            self.frames[(cam_id, 0)] = live_full_jpg
            for idx, l_jpg in live_lens_jpegs:
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
                    with_audio = False  # Chỉ ghi âm thanh vào video Xem lại; không stream MP3 trực tiếp để tối ưu độ ổn định
                    for jpg in self._iter_ffmpeg_jpegs(src_url, transport, stop_event, cam_id, with_audio=with_audio):
                        got_frame = True
                        retry_delay = 3.0
                        self._analyze_frame(
                            cam_id, cam_name, jpg, lens_states, local_pose, task_landmarker, state
                        )
                        if state.get("force_reconnect"):
                            break
                    if not got_frame:
                        if with_audio:
                            state["with_audio"] = False
                        else:
                            state["with_audio"] = True
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
