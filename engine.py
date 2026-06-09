import cv2
import mediapipe as mp
import numpy as np
import time
import threading
import os
import csv
from datetime import datetime
from pygame import mixer
import config
import metrics

class DriverSafetySimulator:
    def __init__(self):
        mixer.init()
        self.audio_lock = threading.Lock()
        self.current_alert_level = 0
        self.is_playing = False
        
        self.sound_warning = mixer.Sound(config.AUDIO_WARNING) if os.path.exists(config.AUDIO_WARNING) else None
        self.sound_critical = mixer.Sound(config.AUDIO_CRITICAL) if os.path.exists(config.AUDIO_CRITICAL) else None
        self.active_channel = None
        
        self._init_log_file()
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.6, min_tracking_confidence=0.6
        )
        
        self.eye_counter = 0
        self.yawn_counter = 0
        self.distraction_counter = 0
        
        # --- NEW: ML LOGGER PARAMETERS ---
        self.ml_dataset_file = "driver_fatigue_dataset.csv"

    def _init_log_file(self):
        if not os.path.exists(config.LOG_FILE):
            with open(config.LOG_FILE, mode='w', newline='') as f:
                csv.writer(f).writerow(["Timestamp", "Alert_Type", "Alert_Level", "EAR", "MAR", "Head_Deviation", "Latency_MS"])

    def _log_telemetry_async(self, alert_text, level, ear, mar, head_dev, latency):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        try:
            with open(config.LOG_FILE, mode='a', newline='') as f:
                csv.writer(f).writerow([timestamp, alert_text, level, round(ear, 3), round(mar, 3), round(head_dev, 3), round(latency, 1)])
        except IOError:
            pass

    # --- NEW: BACKEND FEATURE LOGGING FUNCTION ---
    def _log_ml_features_async(self, ear, mar, head_deviation, label):
        """Asynchronously writes pure features to the training dataset file."""
        file_exists = os.path.isfile(self.ml_dataset_file)
        try:
            with open(self.ml_dataset_file, mode="a", newline="") as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(["timestamp", "ear", "mar", "head_deviation", "label"])
                writer.writerow([time.time(), round(ear, 4), round(mar, 4), round(head_deviation, 4), label])
        except IOError:
            pass

    def _play_audio_async(self, alert_level):
        if alert_level == 2 and self.sound_critical:
            with self.audio_lock:
                self.active_channel = self.sound_critical.play(loops=-1)
            while self.current_alert_level == 2:
                time.sleep(0.05)
            with self.audio_lock:
                if self.active_channel: self.active_channel.stop(); self.active_channel = None
        elif alert_level == 1 and self.sound_warning:
            with self.audio_lock:
                self.active_channel = self.sound_warning.play()
            start_play = time.time()
            while time.time() - start_play < 1.5:
                if self.current_alert_level == 0:
                    with self.audio_lock:
                        if self.active_channel: self.active_channel.stop()
                    break
                time.sleep(0.05)
        with self.audio_lock:
            self.is_playing = False

    def trigger_alert(self, level):
        self.current_alert_level = level
        if not self.is_playing:
            with self.audio_lock: self.is_playing = True
            threading.Thread(target=self._play_audio_async, args=(level,), daemon=True).start()

    def stop_alerts(self):
        self.current_alert_level = 0
        with self.audio_lock:
            if self.active_channel: self.active_channel.stop(); self.active_channel = None
            self.is_playing = False

    # --- MODIFIED: Added ml_mode parameter ---
    def process_single_frame(self, frame, ml_mode="🔴 Do Not Log Data"):
        """Processes a single camera frame synchronously to prevent threading lockups."""
        start_time = time.perf_counter()

         # Mirror the frame horizontally to align camera coordinates with metric geometry
        frame = cv2.flip(frame, 1)

        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        status_text = "SYSTEM ACTIVE"
        status_color = config.COLOR_GREEN
        frame_alert_level = 0
        avg_ear, mar, head_dev = 0.0, 0.0, 0.0

        if results.multi_face_landmarks:
            mesh_points = np.array([[int(lm.x * w), int(lm.y * h)] for lm in results.multi_face_landmarks[0].landmark])
            avg_ear = (metrics.calculate_ear(mesh_points, config.LEFT_EYE) + metrics.calculate_ear(mesh_points, config.RIGHT_EYE)) / 2.0
            mar = metrics.calculate_mar(mesh_points, config.MOUTH)
            head_dev = metrics.calculate_head_deviation(mesh_points)
            
            # --- NEW: MACHINE LEARNING COLLECTION HOOK ---
            if "Label 0" in ml_mode:
                threading.Thread(target=self._log_ml_features_async, args=(avg_ear, mar, head_dev, 0), daemon=True).start()
                status_text = "ML DATA LOGGING: ALERT STATE (0)"
                status_color = config.COLOR_GREEN
            elif "Label 1" in ml_mode:
                threading.Thread(target=self._log_ml_features_async, args=(avg_ear, mar, head_dev, 1), daemon=True).start()
                status_text = "ML DATA LOGGING: FATIGUED STATE (1)"
                status_color = config.COLOR_YELLOW

            if mar > config.MAR_THRESHOLD: self.yawn_counter += 1
            else: self.yawn_counter = max(0, self.yawn_counter - 1)

            if avg_ear < config.EAR_THRESHOLD: self.eye_counter += 1
            else: self.eye_counter = max(0, self.eye_counter - 3)

            if head_dev > config.HEAD_TILT_THRESHOLD: self.distraction_counter += 1
            else: self.distraction_counter = max(0, self.distraction_counter - 2)

            # Keep existing logic only if we aren't overriding status text via ML logs
            if "🔴 Do Not Log" in ml_mode:
                if self.yawn_counter >= config.YAWN_FRAMES:
                    status_text = "WARNING: FREQUENT YAWNING"
                    status_color = config.COLOR_YELLOW
                    frame_alert_level = 1
                    self.eye_counter = max(0, self.eye_counter - 2)
                elif mar > (config.MAR_THRESHOLD * 0.7) and self.yawn_counter > 5:
                    status_text = "ANALYZING MOUTH TRACKING..."
                    status_color = config.COLOR_YELLOW
                    frame_alert_level = 0
                elif self.eye_counter >= config.EYE_CLOSED_FRAMES:
                    status_text = "CRITICAL ALERT: DROWSINESS"
                    status_color = config.COLOR_RED
                    frame_alert_level = 2
                elif self.distraction_counter >= config.DISTRACTED_FRAMES:
                    status_text = "CRITICAL ALERT: DISTRACTED"
                    status_color = config.COLOR_RED
                    frame_alert_level = 2

            if frame_alert_level > 0: self.trigger_alert(frame_alert_level)
            else: self.stop_alerts()

            for idx in config.LEFT_EYE + config.RIGHT_EYE: cv2.circle(frame, tuple(mesh_points[idx]), 1, config.COLOR_GREEN, -1)
            for idx in config.MOUTH: cv2.circle(frame, tuple(mesh_points[idx]), 1, config.COLOR_YELLOW, -1)
            cv2.putText(frame, f"EAR: {avg_ear:.2f}", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, config.COLOR_GREEN, 2)
            cv2.putText(frame, f"MAR: {mar:.2f}", (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, config.COLOR_GREEN, 2)
            cv2.putText(frame, f"Head Dev: {head_dev:.2f}", (30, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, config.COLOR_GREEN, 2)
        else:
            if self.eye_counter >= config.EYE_CLOSED_FRAMES or self.distraction_counter >= config.DISTRACTED_FRAMES:
                status_text = "CRITICAL ALERT: FACE LOST"; status_color = config.COLOR_RED; frame_alert_level = 2
                self.trigger_alert(frame_alert_level)
            else:
                self.stop_alerts()
                status_text = "TRACKING LOST - SCANNING..."
                status_color = config.COLOR_YELLOW
                self.eye_counter = max(0, self.eye_counter - 1)
                self.yawn_counter = max(0, self.yawn_counter - 1)
                self.distraction_counter = max(0, self.distraction_counter - 1)
  
        latency_ms = (time.perf_counter() - start_time) * 1000
        if frame_alert_level > 0:
            threading.Thread(target=self._log_telemetry_async, args=(status_text, frame_alert_level, avg_ear, mar, head_dev, latency_ms), daemon=True).start()
        
        cv2.rectangle(frame, (0, 0), (w, 45), status_color, -1)
        cv2.putText(frame, status_text, (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20, 20, 20), 3)
        cv2.putText(frame, f"Latency: {latency_ms:.1f}ms", (w - 180, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 2)
        
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
