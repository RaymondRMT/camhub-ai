import os
import time
import math
import json
import uuid
import subprocess
import threading
from datetime import datetime
from typing import Optional
import cv2

try:
    import mediapipe as mp
    HAS_MEDIAPIPE = hasattr(mp, "solutions") and hasattr(mp.solutions, "pose")
except Exception:
    mp = None
    HAS_MEDIAPIPE = False

# Tối ưu tùy chọn FFmpeg cho luồng RTSP: dùng TCP ổn định và đặt timeout 5 giây tránh treo cap.read()
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp|stimeout;5000000")


class CameraAIProcessor:
    def __init__(self, on_alert_callback):
        self.on_alert = on_alert_callback
        self.workers = {}
        self.workers_lock = threading.Lock()
        self.latest_frames = {}
        self.frames_lock = threading.Lock()
        self.camera_status = {}
        self.thermal_delay = 0.12  # Mặc định ~8 FPS
        os.makedirs("snapshots", exist_ok=True)

        self.mp_pose = mp.solutions.pose if HAS_MEDIAPIPE else None
        threading.Thread(target=self._thermal_guard_loop, daemon=True).start()

    def _thermal_guard_loop(self):
        """Tự động giãn nhịp AI khi nhiệt độ pin vượt 43°C để bảo vệ máy 24/7."""
        while True:
            try:
                raw = subprocess.check_output(
                    ["termux-battery-status"],
                    timeout=3,
                    stderr=subprocess.DEVNULL,
                ).decode("utf-8")
                temp = float(json.loads(raw).get("temperature", 35.0))
                if temp >= 43.0:
                    self.thermal_delay = 0.45
                elif temp <= 38.5:
                    self.thermal_delay = 0.12
            except Exception:
                pass
            time.sleep(20)

    def start_camera_monitor(self, cam_id: str, cam_name: str, rtsp_url: str):
        self.stop_camera_monitor(cam_id)
        stop_event = threading.Event()
        with self.frames_lock:
            self.camera_status[cam_id] = {"online": False, "fps": 0.0, "last_frame_ts": 0.0}
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
            self.latest_frames.pop(cam_id, None)
            self.camera_status.pop(cam_id, None)

    def get_latest_jpeg(self, cam_id: str) -> Optional[bytes]:
        with self.frames_lock:
            return self.latest_frames.get(cam_id)

    def get_camera_status(self, cam_id: str) -> dict:
        with self.frames_lock:
            return dict(self.camera_status.get(cam_id, {"online": False, "fps": 0.0, "last_frame_ts": 0.0}))

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
        """Fallback phát hiện tư thế nằm ngang bằng hình học Contour khi thiết bị không dùng MediaPipe."""
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return False
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < 3200:
            return False
        _, _, w, h = cv2.boundingRect(largest)
        return (w / float(h + 1e-6)) > 1.35

    def _open_capture(self, rtsp_url: str):
        cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

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

        cap = self._open_capture(rtsp_url)
        bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=300, varThreshold=40, detectShadows=False
        )

        last_motion_alert = 0.0
        last_fall_alert = 0.0
        horizontal_start_time = None
        prev_ts = time.time()

        try:
            while not stop_event.is_set():
                ret, frame = cap.read()
                if not ret or frame is None:
                    with self.frames_lock:
                        self.camera_status[cam_id] = {"online": False, "fps": 0.0, "last_frame_ts": 0.0}
                    cap.release()
                    if stop_event.wait(3.0):
                        break
                    cap = self._open_capture(rtsp_url)
                    continue

                now = time.time()
                dt = max(0.01, now - prev_ts)
                prev_ts = now

                # Mã hóa khung hình preview/MJPEG (640x360) phục vụ truyền trực tuyến realtime không cần MediaMTX
                preview = cv2.resize(frame, (640, 360))
                ok_jpg, jpg_buf = cv2.imencode(".jpg", preview, [int(cv2.IMWRITE_JPEG_QUALITY), 72])
                if ok_jpg:
                    with self.frames_lock:
                        self.latest_frames[cam_id] = jpg_buf.tobytes()
                        self.camera_status[cam_id] = {
                            "online": True,
                            "fps": round(1.0 / dt, 1),
                            "last_frame_ts": now,
                        }

                small = cv2.resize(frame, (480, 270))
                lr = 0.0 if (horizontal_start_time is not None and (now - horizontal_start_time) < 10.0) else -1.0
                fg_mask = bg_subtractor.apply(small, learningRate=lr)
                motion_pixels = cv2.countNonZero(fg_mask)
                has_motion = motion_pixels > 2800

                if has_motion and (now - last_motion_alert > 30.0):
                    ts_ms = int(now * 1000)
                    snap_file = f"snapshots/motion_{cam_id}_{ts_ms}.jpg"
                    cv2.imwrite(snap_file, frame)
                    self.on_alert({
                        "id": f"motion_{cam_id}_{ts_ms}_{uuid.uuid4().hex[:6]}",
                        "type": "motion",
                        "cam_id": cam_id,
                        "cam_name": cam_name,
                        "message": f"Phát hiện chuyển động tại {cam_name}",
                        "snapshot": f"/{snap_file}",
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    })
                    last_motion_alert = now

                if has_motion or (horizontal_start_time is not None):
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
                        if horizontal_start_time is None:
                            horizontal_start_time = now
                        elif (now - horizontal_start_time) >= 2.5 and (now - last_fall_alert > 20.0):
                            ts_ms = int(now * 1000)
                            snap_file = f"snapshots/fall_{cam_id}_{ts_ms}.jpg"
                            cv2.imwrite(snap_file, frame)
                            self.on_alert({
                                "id": f"fall_{cam_id}_{ts_ms}_{uuid.uuid4().hex[:6]}",
                                "type": "fall",
                                "cam_id": cam_id,
                                "cam_name": cam_name,
                                "message": f"KHẨN CẤP: Phát hiện người bị té ngã tại {cam_name}!",
                                "snapshot": f"/{snap_file}",
                                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            })
                            last_fall_alert = now
                            horizontal_start_time = None
                    else:
                        horizontal_start_time = None

                if stop_event.wait(self.thermal_delay):
                    break
        finally:
            cap.release()
            if local_pose is not None:
                try:
                    local_pose.close()
                except Exception:
                    pass
