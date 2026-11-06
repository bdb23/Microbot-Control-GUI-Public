import math
import time

import numpy as np
import serial
from PySide6.QtCore import QThread, Signal, QMutexLocker, QTimer
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox,
    QLineEdit, QPlainTextEdit, QGroupBox, QGridLayout, QSpinBox
)
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg


class ReadWorker(QThread):
    reading_ready = Signal(float, float, float)
    error = Signal(str)

    def __init__(self, coil, coil_lock, parent=None):
        super().__init__(parent)
        self.coil = coil
        self.coil_lock = coil_lock

    def run(self):
        try:
            with QMutexLocker(self.coil_lock):
                bx, by, bz = self.coil.sense_read()
        except (TimeoutError, ValueError, RuntimeError, serial.SerialException) as exc:
            self.error.emit(str(exc))
            return
        self.reading_ready.emit(bx, by, bz)


class ZeroWorker(QThread):
    offset_ready = Signal(float, float, float)
    error = Signal(str)

    def __init__(self, coil, coil_lock, num_samples=10, parent=None):
        super().__init__(parent)
        self.coil = coil
        self.coil_lock = coil_lock
        self.num_samples = num_samples

    def run(self):
        sum_x = sum_y = sum_z = 0.0
        try:
            with QMutexLocker(self.coil_lock):
                for _ in range(self.num_samples):
                    bx, by, bz = self.coil.sense_read()
                    sum_x += bx
                    sum_y += by
                    sum_z += bz
        except (TimeoutError, ValueError, RuntimeError, serial.SerialException) as exc:
            self.error.emit(str(exc))
            return
        n = self.num_samples
        self.offset_ready.emit(sum_x / n, sum_y / n, sum_z / n)


class MonitorWorker(QThread):
    reading_ready = Signal(float, float, float)
    error = Signal(str)

    def __init__(self, coil, coil_lock, interval_ms, parent=None):
        super().__init__(parent)
        self.coil = coil
        self.coil_lock = coil_lock
        self.interval_ms = interval_ms
        self._running = False

    def stop(self):
        self._running = False
        self.wait(2000)

    def run(self):
        self._running = True
        while self._running:
            try:
                with QMutexLocker(self.coil_lock):
                    bx, by, bz = self.coil.sense_read()
            except (TimeoutError, ValueError, RuntimeError, serial.SerialException) as exc:
                self.error.emit(str(exc))
                break
            self.reading_ready.emit(bx, by, bz)
            self.msleep(self.interval_ms)


class CaptureWorker(QThread):
    capture_ready = Signal(list)
    error = Signal(str)

    def __init__(self, coil, coil_lock, sample_rate_hz, num_samples, parent=None):
        super().__init__(parent)
        self.coil = coil
        self.coil_lock = coil_lock
        self.sample_rate_hz = sample_rate_hz
        self.num_samples = num_samples

    def run(self):
        try:
            with QMutexLocker(self.coil_lock):
                samples = self.coil.capture_rotating_field(
                    self.sample_rate_hz, self.num_samples
                )
        except (TimeoutError, ValueError, RuntimeError, serial.SerialException) as exc:
            self.error.emit(str(exc))
            return
        self.capture_ready.emit(samples)


def hall_to_coil_frame(bx, by, bz):
    coil_x = bx
    coil_y = -bz
    coil_z = by
    return coil_x, coil_y, coil_z


def firmware_orthonormal_basis(nx, ny, nz):
    norm = math.sqrt(nx * nx + ny * ny + nz * nz)
    if norm < 1e-6:
        nx, ny, nz, norm = 0.0, 0.0, 1.0, 1.0
    nx, ny, nz = nx / norm, ny / norm, nz / norm

    rx, ry, rz = 0.0, 0.0, 1.0
    if abs(nz) > 0.9:
        rx, ry, rz = 1.0, 0.0, 0.0

    ux = ry * nz - rz * ny
    uy = rz * nx - rx * nz
    uz = rx * ny - ry * nx
    unorm = math.sqrt(ux * ux + uy * uy + uz * uz)
    ux, uy, uz = ux / unorm, uy / unorm, uz / unorm

    vx = ny * uz - nz * uy
    vy = nz * ux - nx * uz
    vz = nx * uy - ny * ux

    return (nx, ny, nz), (ux, uy, uz), (vx, vy, vz)


def _project(coil_x, coil_y, coil_z, axis_vec):
    n_hat, u_hat, v_hat = firmware_orthonormal_basis(*axis_vec)
    u = coil_x * u_hat[0] + coil_y * u_hat[1] + coil_z * u_hat[2]
    v = coil_x * v_hat[0] + coil_y * v_hat[1] + coil_z * v_hat[2]
    n = coil_x * n_hat[0] + coil_y * n_hat[1] + coil_z * n_hat[2]
    return u, v, n


def analyze_rotation(samples, sample_rate_hz, axis_vec=(0.0, 0.0, 1.0)):
    hall_bx = np.array([s[0] for s in samples])
    hall_by = np.array([s[1] for s in samples])
    hall_bz = np.array([s[2] for s in samples])
    mags = np.sqrt(hall_bx ** 2 + hall_by ** 2 + hall_bz ** 2)

    coil_x, coil_y, coil_z = hall_to_coil_frame(hall_bx, hall_by, hall_bz)
    u, v, n_comp = _project(coil_x, coil_y, coil_z, axis_vec)

    inplane_radius = np.sqrt(u ** 2 + v ** 2)

    n = len(samples)
    t = np.arange(n) / sample_rate_hz
    angles = np.unwrap(np.arctan2(v, u))

    if n >= 2:
        slope, _intercept = np.polyfit(t, angles, 1)
    else:
        slope = 0.0
    achieved_freq_hz = slope / (2.0 * np.pi)

    return {
        "mean_magnitude": float(np.mean(mags)),
        "std_magnitude": float(np.std(mags)),
        "achieved_freq_hz": float(achieved_freq_hz),
        "inplane_radius_mean": float(np.mean(inplane_radius)),
        "inplane_radius_std": float(np.std(inplane_radius)),
        "out_of_plane_std": float(np.std(n_comp)),
        "out_of_plane_mean": float(np.mean(n_comp)),
    }


class CaptureCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None):
        fig = Figure(figsize=(5, 2.5), tight_layout=True)
        super().__init__(fig)
        self.setParent(parent)
        self.ax_circle = fig.add_subplot(1, 2, 1)
        self.ax_mag = fig.add_subplot(1, 2, 2)
        self._draw_empty()

    def _draw_empty(self):
        self.ax_circle.clear()
        self.ax_mag.clear()
        self.ax_circle.set_title("In-plane trace (u, v)")
        self.ax_mag.set_title("|B| vs time")
        self.ax_mag.set_xlabel("time (s)")
        self.ax_mag.set_ylabel("|B| (mT)")
        self.draw()

    def plot_capture(self, samples, sample_rate_hz, axis_vec=(0.0, 0.0, 1.0)):
        hall_bx = np.array([s[0] for s in samples])
        hall_by = np.array([s[1] for s in samples])
        hall_bz = np.array([s[2] for s in samples])
        mags = np.sqrt(hall_bx ** 2 + hall_by ** 2 + hall_bz ** 2)
        t = [i / sample_rate_hz for i in range(len(samples))]

        coil_x, coil_y, coil_z = hall_to_coil_frame(hall_bx, hall_by, hall_bz)
        u, v, _n_comp = _project(coil_x, coil_y, coil_z, axis_vec)

        self.ax_circle.clear()
        self.ax_circle.plot(u, v, "-o", markersize=3)
        self.ax_circle.set_aspect("equal", adjustable="box")
        nx, ny, nz = axis_vec
        self.ax_circle.set_title(f"In-plane trace, axis=({nx:.2f},{ny:.2f},{nz:.2f})")
        self.ax_circle.set_xlabel("u (mT)")
        self.ax_circle.set_ylabel("v (mT)")

        self.ax_mag.clear()
        self.ax_mag.plot(t, mags, "-o", markersize=3)
        self.ax_mag.set_title("|B| vs time")
        self.ax_mag.set_xlabel("time (s)")
        self.ax_mag.set_ylabel("|B| (mT)")

        self.draw()


class HallSensorPanel(QWidget):
    def __init__(self, control_panel, parent=None):
        super().__init__(parent)
        self.control_panel = control_panel
        self.read_worker = None
        self.monitor_worker = None
        self.capture_worker = None
        self.zero_worker = None
        self.zero_offset = (0.0, 0.0, 0.0)
        self._build_ui()

        self.status_timer = QTimer(self)
        self.status_timer.setInterval(500)
        self.status_timer.timeout.connect(self._update_status)
        self.status_timer.start()
        self._update_status()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        self.conn_status_label = QLabel("Checking connection...")
        layout.addWidget(self.conn_status_label)

        reading_box = QGroupBox("Live field reading")
        grid = QGridLayout(reading_box)
        big_font = "font-size: 18pt; font-weight: bold;"

        grid.addWidget(QLabel("Bx:"), 0, 0)
        self.bx_label = QLabel("--")
        self.bx_label.setStyleSheet(big_font)
        grid.addWidget(self.bx_label, 0, 1)

        grid.addWidget(QLabel("By:"), 1, 0)
        self.by_label = QLabel("--")
        self.by_label.setStyleSheet(big_font)
        grid.addWidget(self.by_label, 1, 1)

        grid.addWidget(QLabel("Bz:"), 2, 0)
        self.bz_label = QLabel("--")
        self.bz_label.setStyleSheet(big_font)
        grid.addWidget(self.bz_label, 2, 1)

        grid.addWidget(QLabel("|B| (magnitude):"), 3, 0)
        self.mag_label = QLabel("--")
        self.mag_label.setStyleSheet(big_font + "color: #2a7;")
        grid.addWidget(self.mag_label, 3, 1)

        layout.addWidget(reading_box)

        zero_box = QGroupBox("Zero offset")
        zero_layout = QHBoxLayout(zero_box)
        self.zero_btn = QPushButton("Zero Now")
        self.zero_btn.clicked.connect(self.zero_sensor)
        zero_layout.addWidget(self.zero_btn)

        self.clear_zero_btn = QPushButton("Clear")
        self.clear_zero_btn.clicked.connect(self.clear_zero)
        zero_layout.addWidget(self.clear_zero_btn)

        self.zero_status_label = QLabel("No offset applied")
        zero_layout.addWidget(self.zero_status_label)
        zero_layout.addStretch()
        layout.addWidget(zero_box)

        compare_box = QGroupBox("Compare with expected value")
        compare_layout = QHBoxLayout(compare_box)
        compare_layout.addWidget(QLabel("Expected |B|:"))
        self.expected_line = QLineEdit()
        self.expected_line.textChanged.connect(self._update_comparison)
        compare_layout.addWidget(self.expected_line)
        self.diff_label = QLabel("--")
        compare_layout.addWidget(self.diff_label)
        compare_layout.addStretch()
        layout.addWidget(compare_box)

        controls_row = QHBoxLayout()
        self.read_once_btn = QPushButton("Read Once")
        self.read_once_btn.clicked.connect(self.read_once)
        controls_row.addWidget(self.read_once_btn)

        controls_row.addWidget(QLabel("Interval:"))
        self.interval_combo = QComboBox()
        self.interval_combo.addItems(["100 ms", "250 ms", "500 ms", "1000 ms"])
        self.interval_combo.setCurrentText("250 ms")
        controls_row.addWidget(self.interval_combo)

        self.monitor_btn = QPushButton("Start Monitoring")
        self.monitor_btn.clicked.connect(self.toggle_monitoring)
        controls_row.addWidget(self.monitor_btn)

        controls_row.addStretch()
        layout.addLayout(controls_row)

        capture_box = QGroupBox("Capture rotating field")
        capture_layout = QVBoxLayout(capture_box)

        capture_controls = QHBoxLayout()
        capture_controls.addWidget(QLabel("Rotation axis (nx,ny,nz):"))
        self.capture_axis_line = QLineEdit()
        self.capture_axis_line.setText("0,0,1")
        self.capture_axis_line.setMaximumWidth(120)
        capture_controls.addWidget(self.capture_axis_line)

        for label, vec in [("X", "1,0,0"), ("Y", "0,1,0"), ("Z", "0,0,1")]:
            btn = QPushButton(label)
            btn.setMaximumWidth(28)
            btn.clicked.connect(lambda checked=False, v=vec: self.capture_axis_line.setText(v))
            capture_controls.addWidget(btn)

        capture_controls.addWidget(QLabel("Sample rate (Hz):"))
        self.capture_rate_spin = QSpinBox()
        self.capture_rate_spin.setRange(1, 1000)
        self.capture_rate_spin.setValue(200)
        capture_controls.addWidget(self.capture_rate_spin)

        capture_controls.addWidget(QLabel("Samples:"))
        self.capture_count_spin = QSpinBox()
        self.capture_count_spin.setRange(2, 2048)
        self.capture_count_spin.setSingleStep(2)
        self.capture_count_spin.setValue(400)
        capture_controls.addWidget(self.capture_count_spin)

        capture_controls.addWidget(QLabel("Commanded freq (Hz):"))
        self.commanded_freq_line = QLineEdit()
        self.commanded_freq_line.setMaximumWidth(120)
        capture_controls.addWidget(self.commanded_freq_line)

        self.capture_btn = QPushButton("Start Capture")
        self.capture_btn.clicked.connect(self.start_capture)
        capture_controls.addWidget(self.capture_btn)
        capture_controls.addStretch()
        capture_layout.addLayout(capture_controls)

        self.capture_canvas = CaptureCanvas(self)
        self.capture_canvas.setMinimumHeight(220)
        capture_layout.addWidget(self.capture_canvas)

        self.capture_result_label = QLabel("No capture yet.")
        capture_layout.addWidget(self.capture_result_label)

        layout.addWidget(capture_box)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(5000)
        self.log.setStyleSheet("font-family: Consolas, monospace;")
        layout.addWidget(self.log, stretch=1)

    def _update_status(self):
        connected = self.control_panel.coil is not None
        if connected:
            self.conn_status_label.setText("Connected (via Control tab)")
        else:
            self.conn_status_label.setText(
                "Not connected - connect on the Control tab first"
            )
            if self.monitor_worker is not None:
                self._stop_monitoring()

        self.read_once_btn.setEnabled(connected)
        self.monitor_btn.setEnabled(connected)
        self.zero_btn.setEnabled(connected and self.zero_worker is None)
        if self.capture_worker is None:
            self.capture_btn.setEnabled(connected)

    def zero_sensor(self):
        coil = self.control_panel.coil
        if coil is None:
            self._log_line("Not connected.")
            return

        self.zero_btn.setEnabled(False)
        self.zero_status_label.setText("Zeroing (averaging 10 readings)...")
        self.zero_worker = ZeroWorker(coil, self.control_panel.coil_lock, num_samples=10)
        self.zero_worker.offset_ready.connect(self._on_zero_ready)
        self.zero_worker.error.connect(self._on_zero_error)
        self.zero_worker.start()

    def _on_zero_ready(self, bx, by, bz):
        self.zero_worker = None
        self.zero_offset = (bx, by, bz)
        self.zero_btn.setEnabled(self.control_panel.coil is not None)
        self.zero_status_label.setText(
            f"Offset: Bx={bx:.4f} By={by:.4f} Bz={bz:.4f} mT (subtracted from readings below)"
        )
        self._log_line(
            f"-- zeroed: offset Bx={bx:.4f} By={by:.4f} Bz={bz:.4f} mT --"
        )

    def _on_zero_error(self, message):
        self.zero_worker = None
        self.zero_btn.setEnabled(self.control_panel.coil is not None)
        self.zero_status_label.setText(f"Zero failed: {message}")
        self._log_line(f"Zero error: {message}")

    def clear_zero(self):
        self.zero_offset = (0.0, 0.0, 0.0)
        self.zero_status_label.setText("No offset applied")
        self._log_line("-- zero offset cleared --")

    def read_once(self):
        coil = self.control_panel.coil
        if coil is None:
            self._log_line("Not connected.")
            return

        self.read_once_btn.setEnabled(False)
        self.read_worker = ReadWorker(coil, self.control_panel.coil_lock)
        self.read_worker.reading_ready.connect(self._on_reading)
        self.read_worker.error.connect(self._on_error)
        self.read_worker.finished.connect(lambda: self.read_once_btn.setEnabled(True))
        self.read_worker.start()

    def toggle_monitoring(self):
        if self.monitor_worker is not None:
            self._stop_monitoring()
        else:
            self._start_monitoring()

    def _start_monitoring(self):
        coil = self.control_panel.coil
        if coil is None:
            self._log_line("Not connected.")
            return

        interval_ms = int(self.interval_combo.currentText().split()[0])
        self.monitor_worker = MonitorWorker(coil, self.control_panel.coil_lock, interval_ms)
        self.monitor_worker.reading_ready.connect(self._on_reading)
        self.monitor_worker.error.connect(self._on_monitor_error)
        self.monitor_worker.start()

        self.monitor_btn.setText("Stop Monitoring")
        self.read_once_btn.setEnabled(False)
        self.interval_combo.setEnabled(False)
        self._log_line(f"-- monitoring started ({interval_ms} ms interval) --")

    def _stop_monitoring(self):
        if self.monitor_worker is not None:
            self.monitor_worker.stop()
            self.monitor_worker = None
        self.monitor_btn.setText("Start Monitoring")
        self.read_once_btn.setEnabled(self.control_panel.coil is not None)
        self.interval_combo.setEnabled(True)
        self._log_line("-- monitoring stopped --")

    def _on_reading(self, bx, by, bz):
        ox, oy, oz = self.zero_offset
        bx, by, bz = bx - ox, by - oy, bz - oz
        magnitude = math.sqrt(bx ** 2 + by ** 2 + bz ** 2)

        self.bx_label.setText(f"{bx:.4f}")
        self.by_label.setText(f"{by:.4f}")
        self.bz_label.setText(f"{bz:.4f}")
        self.mag_label.setText(f"{magnitude:.4f}")

        self._last_magnitude = magnitude
        self._update_comparison()

        timestamp = time.strftime("%H:%M:%S")
        self._log_line(
            f"[{timestamp}] Bx={bx:.4f} By={by:.4f} Bz={bz:.4f} |B|={magnitude:.4f}"
        )

    def _update_comparison(self):
        magnitude = getattr(self, "_last_magnitude", None)
        text = self.expected_line.text().strip()

        if magnitude is None or not text:
            self.diff_label.setText("--")
            return

        try:
            expected = float(text)
        except ValueError:
            self.diff_label.setText("(expected value must be a number)")
            return

        diff = magnitude - expected
        if expected != 0:
            pct = (diff / expected) * 100
            self.diff_label.setText(f"diff: {diff:+.4f}  ({pct:+.1f}%)")
        else:
            self.diff_label.setText(f"diff: {diff:+.4f}")

    def _on_error(self, message):
        self._log_line(f"Read error: {message}")

    def _on_monitor_error(self, message):
        self._log_line(f"Monitor error: {message}")
        self._stop_monitoring()

    def start_capture(self):
        coil = self.control_panel.coil
        if coil is None:
            self._log_line("Not connected.")
            return

        axis_text = self.capture_axis_line.text().strip()
        try:
            parts = [float(v) for v in axis_text.replace(",", " ").split()]
            if len(parts) != 3:
                raise ValueError("need exactly 3 components")
            if math.sqrt(sum(p * p for p in parts)) < 1e-6:
                raise ValueError("axis vector cannot be zero-length")
        except ValueError as exc:
            self._log_line(f"Invalid rotation axis {axis_text!r}: {exc}")
            return
        axis_vec = tuple(parts)

        sample_rate_hz = float(self.capture_rate_spin.value())
        num_samples = int(self.capture_count_spin.value())

        self.capture_btn.setEnabled(False)
        self.capture_btn.setText("Capturing...")
        self.capture_result_label.setText(
            f"Capturing {num_samples} samples @ {sample_rate_hz:.0f}Hz "
            f"({num_samples / sample_rate_hz:.2f}s)..."
        )
        self._log_line(
            f"-- capture started: {num_samples} samples @ {sample_rate_hz:.0f}Hz, "
            f"axis {axis_vec} --"
        )

        self.capture_worker = CaptureWorker(
            coil, self.control_panel.coil_lock, sample_rate_hz, num_samples
        )
        self.capture_worker.capture_ready.connect(
            lambda samples: self._on_capture_ready(samples, sample_rate_hz, axis_vec)
        )
        self.capture_worker.error.connect(self._on_capture_error)
        self.capture_worker.start()

    def _on_capture_ready(self, samples, sample_rate_hz, axis_vec):
        self.capture_worker = None
        self.capture_btn.setEnabled(self.control_panel.coil is not None)
        self.capture_btn.setText("Start Capture")

        ox, oy, oz = self.zero_offset
        samples = [(bx - ox, by - oy, bz - oz) for (bx, by, bz) in samples]

        self.capture_canvas.plot_capture(samples, sample_rate_hz, axis_vec)
        stats = analyze_rotation(samples, sample_rate_hz, axis_vec)

        achieved = stats["achieved_freq_hz"]
        direction = "CCW (+)" if achieved >= 0 else "CW (-)"

        r_mean = stats["inplane_radius_mean"]
        circularity_pct = (stats["inplane_radius_std"] / r_mean * 100) if r_mean > 1e-9 else 0.0
        planarity_pct = (stats["out_of_plane_std"] / r_mean * 100) if r_mean > 1e-9 else 0.0

        result_text = (
            f"axis=({axis_vec[0]:.2f},{axis_vec[1]:.2f},{axis_vec[2]:.2f})  "
            f"Achieved: {abs(achieved):.3f} Hz, {direction}  |  "
            f"|B| = {stats['mean_magnitude']:.4f} \u00b1 {stats['std_magnitude']:.4f} mT  |  "
            f"circularity: {circularity_pct:.1f}% radius variation  |  "
            f"planarity: {planarity_pct:.1f}% out-of-plane"
        )

        commanded_text = self.commanded_freq_line.text().strip()
        if commanded_text:
            try:
                commanded = float(commanded_text)
                if commanded != 0:
                    err_pct = (abs(achieved) - abs(commanded)) / abs(commanded) * 100
                    result_text += f"  |  vs commanded {commanded:.3f} Hz: {err_pct:+.1f}%"
            except ValueError:
                pass

        self.capture_result_label.setText(result_text)
        self._log_line(f"-- capture complete: {result_text} --")

    def _on_capture_error(self, message):
        self.capture_worker = None
        self.capture_btn.setEnabled(self.control_panel.coil is not None)
        self.capture_btn.setText("Start Capture")
        self.capture_result_label.setText(f"Capture failed: {message}")
        self._log_line(f"Capture error: {message}")

    def _log_line(self, text):
        self.log.appendPlainText(text)

    def closeEvent(self, event):
        self._stop_monitoring()
        super().closeEvent(event)
