import os
import time
import math
import json
import uuid
import shutil
import subprocess
import threading
from datetime import datetime
from typing import Optional
import cv2
import numpy as np

try:
    import mediapipe as mp
    HAS_MEDIAPIPE = hasattr(mp, "solutions") and hasattr(mp.solutions, "pose")
except Exception:
    mp = None
    HAS_MEDIAPIPE = False

os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp|stimeout;5000000")


class CameraAIProcessor:
    def __init__(self, on_alert_callback):
        self.on_alert = on_alert_callback
        self.workers = {}
        self.workers_lock = threading.Lock()
        self.frames = {}
        self.frames_lock = threading.Lock()
        self.cam_status = {}
        self.thermal_delay = 0.20  # Mặc định ~5 FPS tiết kiệm pin & giảm nhiệt
        os.makedirs("snapshots", exist_ok=True)

        self.mp_pose = mp.solutions.pose if HAS_MEDIAPIPE else None
        self.has_ffmpeg_cli = shutil.which("ffmpeg") is not None
        threading.Thread(target=self._thermal_guard_loop, daemon=True).start()

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
                "engine": "mediapipe" if HAS_MEDIAPIPE else "contour",
                "last_frame_ts": 0.0,
            }
        t = threading.Thread(
            target=self._process_loop,
            args=(cam_id, cam_name, rtsp_url, stop_event),
            daemon=True,
        )
        with self.workers_lock:
            self.workers[cam_id] = (stop_event, t)
        t.start()

    def stop_camera_monitor(self, cam_id: str):
        with self.workers_lock:
            worker = self.workers.pop(cam_id, None)
        if worker:
            stop_event, _ = worker
            stop_event.set()
        with self.frames_lock:
            self.frames.pop(cam_id, None)
            self.cam_status.pop(cam_id, None)

    def get_latest_frame(self, cam_id: str) -> Optional[bytes]:
        with self.frames_lock:
            return self.frames.get(cam_id)

    def get_camera_status(self, cam_id: str) -> dict:
        with self.frames_lock:
            st = dict(self.cam_status.get(cam_id, {}))
        if not st:
            return {"online": False, "engine": "mediapipe" if HAS_MEDIAPIPE else "contour"}
        is_recent = (time.time() - st.get("last_frame_ts", 0.0)) < 15.0
        st["online"] = bool(st.get("online") and is_recent)
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
        latest = self.get_latest_frame(cam_id)
        if latest:
            with open(snap_file, "wb") as f:
                f.write(latest)
        else:
            img = self.build_placeholder_jpeg(
                f"TEST FALL ALERT - {cam_name}",
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
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
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        self.on_alert(alert_data)
        return alert_data

    def _calculate_body_angle(self, landmarks):
        l_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        r_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        l_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
        r_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]

        if min(l_shoulder.visibility, r_shoulder.visibility, l_hip.visibility, r_hip.visibility) < 0.35:
            return 0.0

        mid_shoulder_x = (l_shoulder.x + r_shoulder.x) / 2.0
        mid_shoulder_y = (l_shoulder.y + r_shoulder.y) / 2.0
        mid_hip_x = (l_hip.x + r_hip.x) / 2.0
        mid_hip_y = (l_hip.y + r_hip.y) / 2.0

        dx = abs(mid_shoulder_x - mid_hip_x)
        dy = abs(mid_shoulder_y - mid_hip_y) + 1e-6
        return math.degrees(math.atan2(dx, dy))

    def _detect_horizontal_posture_fallback(self, fg_mask) -> bool:
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return False
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < 3200:
            return False
        _, _, w, h = cv2.boundingRect(largest)
        return (w / float(h + 1e-6)) > 1.35

    def _spawn_ffmpeg_mjpeg(self, rtsp_url: str, transport: str):
        """Giải mã RTSP trực tiếp bằng FFmpeg hệ thống (tương thích 100% ARM64 Ubuntu, tránh lỗi OpenCV VIDEOIO)."""
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel", "error",
            "-rtsp_transport", transport,
            "-rw_timeout", "7000000",
            "-i", rtsp_url,
            "-vf", "scale=640:360",
            "-r", "5",
            "-f", "image2pipe",
            "-vcodec", "mjpeg",
            "-q:v", "6",
            "-",
        ]
        return subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=0,
        )

    def _iter_ffmpeg_jpegs(self, rtsp_url: str, transport: str, stop_event: threading.Event):
        proc = self._spawn_ffmpeg_mjpeg(rtsp_url, transport)
        buf = bytearray()
        try:
            while not stop_event.is_set():
                if not proc.stdout:
                    break
                chunk = proc.stdout.read(8192)
                if not chunk:
                    break
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

    def _analyze_frame(
        self,
        cam_id: str,
        cam_name: str,
        jpg_bytes: bytes,
        bg_subtractor,
        local_pose,
        state: dict,
    ):
        now = time.time()
        with self.frames_lock:
            self.frames[cam_id] = jpg_bytes
            self.cam_status[cam_id] = {
                "online": True,
                "engine": "mediapipe" if local_pose is not None else "contour",
                "last_frame_ts": now,
            }

        # Giãn nhịp phân tích AI theo nhiệt độ máy để tiết kiệm CPU
        if now - state["last_ai_ts"] < self.thermal_delay:
            return
        state["last_ai_ts"] = now

        arr = np.frombuffer(jpg_bytes, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            return

        small = cv2.resize(frame, (480, 270))
        h_start = state["horizontal_start_time"]
        lr = 0.0 if (h_start is not None and (now - h_start) < 10.0) else -1.0
        fg_mask = bg_subtractor.apply(small, learningRate=lr)
        has_motion = cv2.countNonZero(fg_mask) > 2800

        if has_motion and (now - state["last_motion_alert"] > 30.0):
            ts_ms = int(now * 1000)
            snap_file = f"snapshots/motion_{cam_id}_{ts_ms}.jpg"
            with open(snap_file, "wb") as f:
                f.write(jpg_bytes)
            self.on_alert({
                "id": f"motion_{cam_id}_{ts_ms}_{uuid.uuid4().hex[:6]}",
                "type": "motion",
                "cam_id": cam_id,
                "cam_name": cam_name,
                "message": f"Phát hiện chuyển động tại {cam_name}",
                "snapshot": f"/{snap_file}",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            })
            state["last_motion_alert"] = now

        if has_motion or (state["horizontal_start_time"] is not None):
            is_horizontal = False
            if local_pose is not None:
                rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                results = local_pose.process(rgb)
                if results and results.pose_landmarks:
                    angle = self._calculate_body_angle(results.pose_landmarks.landmark)
                    is_horizontal = angle > 58.0
            else:
                is_horizontal = self._detect_horizontal_posture_fallback(fg_mask)

            if is_horizontal:
                if state["horizontal_start_time"] is None:
                    state["horizontal_start_time"] = now
                elif (now - state["horizontal_start_time"]) >= 2.5 and (now - state["last_fall_alert"] > 20.0):
                    ts_ms = int(now * 1000)
                    snap_file = f"snapshots/fall_{cam_id}_{ts_ms}.jpg"
                    with open(snap_file, "wb") as f:
                        f.write(jpg_bytes)
                    self.on_alert({
                        "id": f"fall_{cam_id}_{ts_ms}_{uuid.uuid4().hex[:6]}",
                        "type": "fall",
                        "cam_id": cam_id,
                        "cam_name": cam_name,
                        "message": f"KHẨN CẤP: Phát hiện người bị té ngã tại {cam_name}!",
                        "snapshot": f"/{snap_file}",
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    })
                    state["last_fall_alert"] = now
                    state["horizontal_start_time"] = None
            else:
                state["horizontal_start_time"] = None

    def _process_loop(self, cam_id: str, cam_name: str, rtsp_url: str, stop_event: threading.Event):
        local_pose = None
        if HAS_MEDIAPIPE and self.mp_pose is not None:
            try:
                local_pose = self.mp_pose.Pose(
                    static_image_mode=False,
                    model_complexity=0,
                    min_detection_confidence=0.55,
                    min_tracking_confidence=0.55,
                )
            except Exception:
                local_pose = None

        bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=300, varThreshold=40, detectShadows=False
        )
        state = {
            "last_ai_ts": 0.0,
            "last_motion_alert": 0.0,
            "last_fall_alert": 0.0,
            "horizontal_start_time": None,
        }

        transports = ["tcp", "udp"]
        t_idx = 0
        retry_delay = 5.0

        try:
            while not stop_event.is_set():
                got_frame = False
                if self.has_ffmpeg_cli:
                    transport = transports[t_idx % len(transports)]
                    for jpg in self._iter_ffmpeg_jpegs(rtsp_url, transport, stop_event):
                        got_frame = True
                        retry_delay = 5.0
                        self._analyze_frame(cam_id, cam_name, jpg, bg_subtractor, local_pose, state)
                    if not got_frame:
                        t_idx += 1  # Tự động chuyển thử giữa TCP <-> UDP (hữu ích cho Yoosee/ONVIF)
                else:
                    cap = cv2.VideoCapture(rtsp_url)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    try:
                        while not stop_event.is_set():
                            ret, frame = cap.read()
                            if not ret or frame is None:
                                break
                            got_frame = True
                            retry_delay = 5.0
                            preview = cv2.resize(frame, (640, 360))
                            ok, buf = cv2.imencode(".jpg", preview, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
                            if ok:
                                self._analyze_frame(cam_id, cam_name, buf.tobytes(), bg_subtractor, local_pose, state)
                            if stop_event.wait(self.thermal_delay):
                                break
                    finally:
                        cap.release()

                with self.frames_lock:
                    if cam_id in self.cam_status:
                        self.cam_status[cam_id]["online"] = False

                if stop_event.wait(retry_delay):
                    break
                # Giãn thời gian thử lại (5s -> 15s) khi camera mất kết nối để không làm nóng máy
                retry_delay = min(15.0, retry_delay + 3.0)
        finally:
            if local_pose is not None:
                try:
                    local_pose.close()
                except Exception:
                    pass
