import datetime
import os
import time

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage

import closed_loop_control
import vision
import world_frame


class CameraWorker(QThread):
    image_ready = Signal(QImage)
    detection_ready = Signal(object)
    fps_updated = Signal(float)
    timing_updated = Signal(float, float)
    error = Signal(str)
    status_changed = Signal(str)
    background_capture_progress = Signal(int, int)
    background_capture_done = Signal(str)
    background_capture_failed = Signal(str)
    recording_started = Signal(str)
    recording_stopped = Signal(str)
    recording_failed = Signal(str)

    DEFAULT_BACKENDS = [
        ("CAP_MSMF", cv2.CAP_MSMF),
        ("CAP_DSHOW", cv2.CAP_DSHOW),
    ]

    def __init__(self, camera_index, camera_name="top", backends=None,
                 frame_width=1920, frame_height=1080, waypoints=None, parent=None):
        super().__init__(parent)
        self.camera_index = camera_index
        self.camera_name = camera_name
        self.backends = backends or self.DEFAULT_BACKENDS
        self.frame_width = frame_width
        self.frame_height = frame_height

        self.waypoints = waypoints if waypoints is not None else []

        self.detection_enabled = True
        self._running = False
        self.tracker = vision.MicrobotTracker(camera=self.camera_name)

        self._capturing_background = False
        self._bg_sum = None
        self._bg_sumsq = None
        self._bg_count = 0
        self._bg_target = 0

        self._recording = False
        self._video_writer = None
        self._recording_path = None

    def set_detection_enabled(self, enabled):
        self.detection_enabled = bool(enabled)

    def start_background_capture(self, n_frames=60):
        config = vision.get_config(self.camera_name)
        if not config.get("background_image_path"):
            self.background_capture_failed.emit(
                f"No background_image_path configured for camera {self.camera_name!r} in vision.py."
            )
            return
        self._bg_sum = None
        self._bg_sumsq = None
        self._bg_count = 0
        self._bg_target = n_frames
        self._capturing_background = True

    def start_recording(self, directory="recordings", path=None):
        if self._recording:
            return
        if path is None:
            os.makedirs(directory, exist_ok=True)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.join(directory, f"{self.camera_name}_{timestamp}.avi")
        self._recording_path = path
        self._recording = True

    def stop_recording(self):
        self._recording = False

    def stop(self):
        self._running = False
        self.wait(2000)

    def _open_camera(self):
        for name, backend in self.backends:
            self.status_changed.emit(f"Trying camera {self.camera_index} via {name}...")
            cap = cv2.VideoCapture(self.camera_index, backend)
            if not cap.isOpened():
                cap.release()
                continue

            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
            time.sleep(0.3)

            ok, frame = cap.read()
            if ok and frame is not None:
                actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                self.status_changed.emit(
                    f"Camera {self.camera_index} connected via {name} "
                    f"({actual_w}x{actual_h})"
                )
                return cap, name

            cap.release()

        return None, None

    def _draw_world_overlay(self, display):
        try:
            ox, oy = world_frame.world_to_pixel((0, 0, 0), self.camera_name)
        except ValueError:
            return

        ox, oy = int(round(ox)), int(round(oy))
        cv2.drawMarker(display, (ox, oy), (255, 255, 0), cv2.MARKER_CROSS, 24, 2)
        cv2.putText(display, "ORIGIN", (ox + 12, oy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        for i, wp in enumerate(list(self.waypoints)):
            try:
                px, py = world_frame.world_to_pixel(wp.position, self.camera_name)
            except ValueError:
                return
            px, py = int(round(px)), int(round(py))
            cv2.drawMarker(display, (px, py), (255, 0, 255), cv2.MARKER_DIAMOND, 18, 2)
            cv2.putText(display, wp.label, (px + 10, py + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

        path = closed_loop_control.LIVE_OVERLAY["path"]
        if path:
            pts = np.array([world_frame.world_to_pixel(p, self.camera_name) for p in path], dtype=np.int32)
            cv2.polylines(display, [pts.reshape(-1, 1, 2)], False, (0, 200, 255), 1, cv2.LINE_AA)
        ref = closed_loop_control.LIVE_OVERLAY["ref"]
        if ref is not None:
            rx, ry = world_frame.world_to_pixel(ref, self.camera_name)
            cv2.circle(display, (int(round(rx)), int(round(ry))), 7, (0, 200, 255), 2)

    def _accumulate_background_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        if self._bg_sum is None:
            self._bg_sum = np.zeros_like(gray)
            self._bg_sumsq = np.zeros_like(gray)
        self._bg_sum += gray
        self._bg_sumsq += gray * gray
        self._bg_count += 1
        self.background_capture_progress.emit(self._bg_count, self._bg_target)

        display = frame.copy()
        cv2.putText(display, f"CAPTURING BACKGROUND {self._bg_count}/{self._bg_target}",
                    (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 2)
        cv2.putText(display, "KEEP TANK EMPTY - no microbot, no rod, no hand",
                    (30, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 1)

        if self._bg_count >= self._bg_target:
            self._finalize_background_capture()

        return display, None

    def _finalize_background_capture(self):
        mean = self._bg_sum / self._bg_count
        var = self._bg_sumsq / self._bg_count - mean * mean
        std = np.sqrt(np.clip(var, 0, None))
        bg_image = mean.astype(np.uint8)
        noise_sigma = float(std.mean())

        config = vision.get_config(self.camera_name)
        path = config["background_image_path"]
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            ok = cv2.imwrite(path, bg_image)
            if not ok:
                raise IOError(f"cv2.imwrite returned False for {path}")
        except (IOError, OSError) as exc:
            self.background_capture_failed.emit(f"Could not save background: {exc}")
        else:
            self.status_changed.emit(
                f"Background captured for {self.camera_name} "
                f"(noise sigma={noise_sigma:.2f}) -> {path}"
            )
            self.background_capture_done.emit(path)

        self._capturing_background = False
        self._bg_sum = None
        self._bg_sumsq = None
        self._bg_count = 0

    def run(self):
        self._running = True
        self.status_changed.emit(f"Opening camera {self.camera_index}...")

        cap, backend_name = self._open_camera()

        if cap is None:
            tried = ", ".join(name for name, _ in self.backends)
            self.error.emit(
                f"Could not open camera index {self.camera_index} "
                f"(tried: {tried}). Check the index and that no other "
                f"app has it open."
            )
            self._running = False
            return

        last_fps_t = time.time()
        frame_count = 0
        capture_time_total = 0.0
        detect_time_total = 0.0

        while self._running:
            capture_start = time.perf_counter()
            ok, frame = cap.read()
            t_grab = time.perf_counter()
            capture_time_total += t_grab - capture_start

            if not ok:
                self.error.emit(f"Frame grab failed on camera {self.camera_index}")
                break

            if self._capturing_background:
                display, result = self._accumulate_background_frame(frame)
            else:
                raw = frame.copy() if self._recording else None
                if self.detection_enabled:
                    detect_start = time.perf_counter()
                    result = self.tracker.update(frame)
                    display = vision.draw_overlay(frame, result)
                    detect_time_total += time.perf_counter() - detect_start
                else:
                    result = None
                    display = frame
                self._draw_world_overlay(display)

                if self._recording:
                    if self._video_writer is None:
                        h, w = display.shape[:2]
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        if not fps or fps <= 1:
                            fps = 30.0
                        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                        writer = cv2.VideoWriter(self._recording_path, fourcc, fps, (w, h))
                        if not writer.isOpened():
                            self.recording_failed.emit(f"Could not open {self._recording_path} for writing")
                            self._recording = False
                        else:
                            self._video_writer = writer
                            self.recording_started.emit(self._recording_path)
                    if self._video_writer is not None and raw is not None:
                        self._video_writer.write(raw)
                elif self._video_writer is not None:
                    self._video_writer.release()
                    self._video_writer = None
                    self.recording_stopped.emit(self._recording_path)

            self.detection_ready.emit(result)

            qimg = self._to_qimage(display)
            self.image_ready.emit(qimg)

            frame_count += 1
            now = time.time()
            if now - last_fps_t >= 1.0:
                elapsed = now - last_fps_t
                self.fps_updated.emit(frame_count / elapsed)
                if frame_count > 0:
                    self.timing_updated.emit(
                        (capture_time_total / frame_count) * 1000.0,
                        (detect_time_total / frame_count) * 1000.0,
                    )
                frame_count = 0
                capture_time_total = 0.0
                detect_time_total = 0.0
                last_fps_t = now

        if self._video_writer is not None:
            self._video_writer.release()
            self._video_writer = None
            self.recording_stopped.emit(self._recording_path)
            self._recording = False

        cap.release()
        self.status_changed.emit(f"Camera {self.camera_index} disconnected")

    @staticmethod
    def _to_qimage(bgr_frame):
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        return QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()
