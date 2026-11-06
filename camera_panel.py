from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QSpinBox, QCheckBox, QGroupBox, QSizePolicy, QMessageBox
)

from camera_worker import CameraWorker


class CameraPanel(QWidget):
    detection_ready = Signal(object)

    def __init__(self, title, default_index, camera_name="top", waypoints=None, parent=None):
        super().__init__(parent)
        self.title = title
        self.camera_name = camera_name
        self.waypoints = waypoints if waypoints is not None else []
        self.worker = None

        self._build_ui(title, default_index)

    def _build_ui(self, title, default_index):
        layout = QVBoxLayout(self)

        header = QGroupBox(title)
        header_layout = QHBoxLayout(header)

        header_layout.addWidget(QLabel("Camera index:"))
        self.index_spin = QSpinBox()
        self.index_spin.setRange(0, 10)
        self.index_spin.setValue(default_index)
        header_layout.addWidget(self.index_spin)

        self.connect_btn = QPushButton("Connect")
        self.connect_btn.clicked.connect(self.toggle_connection)
        header_layout.addWidget(self.connect_btn)

        self.detection_check = QCheckBox("Detection overlay")
        self.detection_check.setChecked(True)
        self.detection_check.toggled.connect(self._on_detection_toggled)
        header_layout.addWidget(self.detection_check)

        self.recapture_btn = QPushButton("Recapture Background")
        self.recapture_btn.setEnabled(False)
        self.recapture_btn.clicked.connect(self._on_recapture_clicked)
        header_layout.addWidget(self.recapture_btn)

        self.record_btn = QPushButton("Start Recording")
        self.record_btn.setEnabled(False)
        self.record_btn.clicked.connect(self._on_record_clicked)
        header_layout.addWidget(self.record_btn)

        header_layout.addStretch()
        layout.addWidget(header)

        self.video_label = QLabel("Not connected")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(320, 180)
        self.video_label.setStyleSheet(
            "background-color: #111; color: #888; border: 1px solid #333;"
        )
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.video_label, stretch=1)

        status_row = QHBoxLayout()
        self.status_label = QLabel("Idle")
        self.fps_label = QLabel("-- fps")
        self.timing_label = QLabel("")
        status_row.addWidget(self.status_label)
        status_row.addStretch()
        status_row.addWidget(self.timing_label)
        status_row.addWidget(self.fps_label)
        layout.addLayout(status_row)

    def toggle_connection(self):
        if self.worker is not None and self.worker.isRunning():
            self.disconnect_camera()
        else:
            self.connect_camera()

    def connect_camera(self):
        index = self.index_spin.value()
        self.worker = CameraWorker(index, camera_name=self.camera_name, waypoints=self.waypoints)
        self.worker.set_detection_enabled(self.detection_check.isChecked())
        self.worker.image_ready.connect(self._on_image)
        self.worker.detection_ready.connect(self._on_detection)
        self.worker.fps_updated.connect(self._on_fps)
        self.worker.timing_updated.connect(self._on_timing)
        self.worker.status_changed.connect(self.status_label.setText)
        self.worker.error.connect(self._on_error)
        self.worker.background_capture_progress.connect(self._on_capture_progress)
        self.worker.background_capture_done.connect(self._on_capture_done)
        self.worker.background_capture_failed.connect(self._on_capture_failed)
        self.worker.recording_started.connect(self._on_recording_started)
        self.worker.recording_stopped.connect(self._on_recording_stopped)
        self.worker.recording_failed.connect(self._on_recording_failed)
        self.worker.start()

        self.connect_btn.setText("Disconnect")
        self.index_spin.setEnabled(False)
        self.recapture_btn.setEnabled(True)
        self.record_btn.setEnabled(True)

    def disconnect_camera(self):
        if self.worker is not None:
            self.worker.stop()
            self.worker = None
        self.connect_btn.setText("Connect")
        self.index_spin.setEnabled(True)
        self.recapture_btn.setEnabled(False)
        self.record_btn.setEnabled(False)
        self.record_btn.setText("Start Recording")
        self.video_label.setText("Not connected")
        self.fps_label.setText("-- fps")

    def _on_image(self, qimg):
        pixmap = QPixmap.fromImage(qimg).scaled(
            self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.video_label.setPixmap(pixmap)

    def _on_detection(self, result):
        self.detection_ready.emit(result)

    def _on_fps(self, fps):
        self.fps_label.setText(f"{fps:.1f} fps")

    def _on_timing(self, capture_ms, detect_ms):
        self.timing_label.setText(f"capture {capture_ms:.0f}ms | detect {detect_ms:.0f}ms  ")

    def _on_error(self, message):
        self.status_label.setText(f"Error: {message}")
        self.disconnect_camera()

    def _on_detection_toggled(self, checked):
        if self.worker is not None:
            self.worker.set_detection_enabled(checked)

    def _on_recapture_clicked(self):
        if self.worker is None or not self.worker.isRunning():
            return
        reply = QMessageBox.question(
            self, "Recapture background", "Empty the tank, then click Yes.",
            QMessageBox.Yes | QMessageBox.Cancel, QMessageBox.Cancel,
        )
        if reply != QMessageBox.Yes:
            return
        self.recapture_btn.setEnabled(False)
        self.worker.start_background_capture()

    def _on_capture_progress(self, count, target):
        self.status_label.setText(f"Capturing background: {count}/{target}")

    def _on_capture_done(self, path):
        self.status_label.setText("Background saved")
        self.recapture_btn.setEnabled(True)

    def _on_capture_failed(self, message):
        self.status_label.setText(f"Background capture failed: {message}")
        self.recapture_btn.setEnabled(True)

    def _on_record_clicked(self):
        if self.worker is None or not self.worker.isRunning():
            return
        if self.worker._recording:
            self.worker.stop_recording()
        else:
            self.worker.start_recording()

    def _on_recording_started(self, path):
        self.record_btn.setText("Stop Recording")
        self.status_label.setText(f"Recording -> {path}")

    def _on_recording_stopped(self, path):
        self.record_btn.setText("Start Recording")
        self.status_label.setText(f"Recording saved -> {path}")

    def _on_recording_failed(self, message):
        self.record_btn.setText("Start Recording")
        self.status_label.setText(f"Recording failed: {message}")

    def closeEvent(self, event):
        self.disconnect_camera()
        super().closeEvent(event)
