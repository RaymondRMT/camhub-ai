import cv2
import time
import math
import json
import os
import subprocess
import threading
import mediapipe as mp
from datetime import datetime


class CameraAIProcessor:
    def __init__(self, on_alert_callback):
        self.on_alert = on_alert_callback
        self.workers = {}
        self.thermal_delay = 0.15  # Mặc định ~6 FPS
        os.makedirs("snapshots", exist_ok=True)

        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=0,
            min_detection_confidence=0.55,
            min_tracking_confidence=0.55,
        )
        threading.Thread(target=self._thermal_guard_loop, daemon=True).start()

    def _thermal_guard_loop(self):
        """Tự động giãn nhịp AI khi nhiệt độ pin vượt 43°C để bảo vệ máy 24/7."""
        while True:
            try:
                raw = subprocess.check_output(["termux-battery-status"], timeout=3).decode("utf-8")
                temp = float(json.loads(raw).get("temperature", 35.0))
                if temp >= 43.0:
                    self.thermal_delay = 0.50
                elif temp <= 38.5:
                    self.thermal_delay = 0.15
            except Exception:
                pass
            time.sleep(20)

    def start_camera_monitor(self, cam_id: str, cam_name: str, rtsp_url: str):
        if cam_id in self.workers:
            self.stop_camera_monitor(cam_id)
        stop_event = threading.Event()
        t = threading.Thread(
            target=self._process_loop,
            args=(cam_id, cam_name, rtsp_url, stop_event),
            daemon=True,
        )
        self.workers[cam_id] = stop_event
        t.start()

    def stop_camera_monitor(self, cam_id: str):
        if cam_id in self.workers:
            self.workers[cam_id].set()
            del self.workers[cam_id]

    def _calculate_body_angle(self, landmarks):
        l_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        r_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        l_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
        r_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]

        mid_shoulder_x = (l_shoulder.x + r_shoulder.x) / 2
        mid_shoulder_y = (l_shoulder.y + r_shoulder.y) / 2
        mid_hip_x = (l_hip.x + r_hip.x) / 2
        mid_hip_y = (l_hip.y + r_hip.y) / 2

        dx = abs(mid_shoulder_x - mid_hip_x)
        dy = abs(mid_shoulder_y - mid_hip_y) + 1e-6
        return math.degrees(math.atan2(dx, dy))

    def _process_loop(self, cam_id: str, cam_name: str, rtsp_url: str, stop_event: threading.Event):
        cap = cv2.VideoCapture(rtsp_url)
        bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=300, varThreshold=40, detectShadows=False
        )

        last_motion_alert = 0
        last_fall_alert = 0
        horizontal_start_time = None

        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                time.sleep(3)
                cap.release()
                cap = cv2.VideoCapture(rtsp_url)
                continue

            small = cv2.resize(frame, (480, 270))
            now = time.time()

            fg_mask = bg_subtractor.apply(small)
            if cv2.countNonZero(fg_mask) > 2800:
                if now - last_motion_alert > 30:
                    snap_file = f"snapshots/motion_{cam_id}_{int(now)}.jpg"
                    cv2.imwrite(snap_file, frame)
                    self.on_alert({
                        "type": "motion",
                        "cam_id": cam_id,
                        "cam_name": cam_name,
                        "message": f"Phát hiện chuyển động tại {cam_name}",
                        "snapshot": f"/{snap_file}",
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    })
                    last_motion_alert = now

                rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                results = self.pose.process(rgb)

                if results.pose_landmarks:
                    angle = self._calculate_body_angle(results.pose_landmarks.landmark)
                    if angle > 58.0:
                        if horizontal_start_time is None:
                            horizontal_start_time = now
                        elif (now - horizontal_start_time) >= 2.5 and (now - last_fall_alert > 20):
                            snap_file = f"snapshots/fall_{cam_id}_{int(now)}.jpg"
                            cv2.imwrite(snap_file, frame)
                            self.on_alert({
                                "type": "fall",
                                "cam_id": cam_id,
                                "cam_name": cam_name,
                                "message": f"KHẨN CẤP: Phát hiện người bị té ngã tại {cam_name}!",
                                "snapshot": f"/{snap_file}",
                                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            })
                            last_fall_alert = now
                    else:
                        horizontal_start_time = None

            time.sleep(self.thermal_delay)

        cap.release()