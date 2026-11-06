import threading
import time

import numpy as np
import serial
from PySide6.QtCore import QThread, Signal, QMutexLocker
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel, QPushButton, QGroupBox, QComboBox, QCheckBox,
    QPlainTextEdit,
)

import closed_loop_control
import keyboard_drive

CONTROLLERS = [("Line following", "line"), ("Pursuit", "pursuit")]

PHASE_NAMES = {
    "layflat": "laying flat", "start": "starting", "start_hold": "starting", "turning": "turning to face the waypoint",
    "swimming": "swimming", "flipping": "reversing spin",
}


class NavigationWorker(QThread):
    status = Signal(dict)
    finished_run = Signal(str)
    log_line = Signal(str)

    def __init__(self, coil, coil_lock, navigator, parent=None):
        super().__init__(parent)
        self.coil = coil
        self.coil_lock = coil_lock
        self.navigator = navigator
        self.config = navigator.config
        self._lock = threading.Lock()
        self._measurement = None
        self._abort = False

    def set_measurement(self, xyz, top_result=None, side_result=None):
        with self._lock:
            self._measurement = (np.asarray(xyz, dtype=np.float64), time.monotonic(), top_result, side_result)

    def request_stop(self):
        self._abort = True

    def run(self):
        cfg = self.config
        period = 1.0 / cfg["control_rate_hz"]
        reason = "stopped"
        last_sent = None
        t0 = time.monotonic()
        try:
            with QMutexLocker(self.coil_lock):
                self.coil.set_slew(cfg["axis_rate_dps"], cfg["freq_rate_hz_per_s"], 0.0)
            while not self._abort:
                now = time.monotonic()
                with self._lock:
                    meas = self._measurement
                if meas is None or now - meas[1] > cfg["lost_stop_s"]:
                    reason = "lost the bot's position - stopped"
                    break
                pos, _, top, side = meas
                cmd = self.navigator.update(pos, now - t0, top, side)
                for event in self.navigator.events:
                    self.log_line.emit(f"t={now - t0:5.1f}s  {event}")
                if cmd is None:
                    reason = self.navigator.stop_reason or "route finished"
                    break
                if cmd["static"] is not None:
                    b = np.asarray(cmd["static"]) * cmd["magnitude"]
                    if last_sent is None or last_sent[0] != "static" or not np.allclose(b, last_sent[1], atol=1e-4):
                        with QMutexLocker(self.coil_lock):
                            self.coil.set_static(float(b[0]), float(b[1]), float(b[2]))
                        last_sent = ("static", b)
                else:
                    axis = np.asarray(cmd["coil_axis"])
                    if (last_sent is None or last_sent[0] != "rotate"
                            or last_sent[2:] != (cmd["freq_hz"], cmd["magnitude"], cmd["direction"])
                            or float(np.dot(axis, last_sent[1])) < np.cos(np.radians(0.5))):
                        with QMutexLocker(self.coil_lock):
                            self.coil.set_rotate(magnitude=cmd["magnitude"], axis=cmd["coil_axis"],
                                                 freq_hz=cmd["freq_hz"], direction=cmd["direction"])
                        last_sent = ("rotate", axis, cmd["freq_hz"], cmd["magnitude"], cmd["direction"])
                closed_loop_control.LIVE_OVERLAY["path"] = [cmd["leg_start"], cmd["target"]]
                closed_loop_control.LIVE_OVERLAY["ref"] = cmd["target"]
                self.status.emit(dict(cmd, position=tuple(pos)))
                sleep_s = period - (time.monotonic() - now)
                if sleep_s > 0:
                    self.msleep(int(sleep_s * 1000))
        except (ValueError, TimeoutError, RuntimeError, OSError, serial.SerialException) as exc:
            reason = f"coil error: {exc}"
        finally:
            try:
                with QMutexLocker(self.coil_lock):
                    self.coil.stop()
            except (ValueError, TimeoutError, RuntimeError, OSError, serial.SerialException) as exc:
                reason += f" (and the stop command failed: {exc} - check the coils!)"
            closed_loop_control.LIVE_OVERLAY["path"] = []
            closed_loop_control.LIVE_OVERLAY["ref"] = None
        self.finished_run.emit(reason)


class ClosedLoopPanel(QWidget):
    def __init__(self, control_panel, waypoints, parent=None):
        super().__init__(parent)
        self.control_panel = control_panel
        self.waypoints = waypoints
        self.worker = None
        self.waypoints_changed = None
        self._latest_xyz = None
        self._latest_views = (None, None)
        control_panel.disconnect_hooks.append(self.shutdown)
        self._build_ui()

    def _build_ui(self):
        outer = QVBoxLayout(self)

        settings_box = QGroupBox("Navigation")
        form = QFormLayout(settings_box)
        self.controller_combo = QComboBox()
        self.controller_combo.addItems([name for name, _ in CONTROLLERS])
        form.addRow("Controller", self.controller_combo)
        self.route_combo = QComboBox()
        self.route_combo.addItem("Choose a route")
        self.route_combo.addItems(list(closed_loop_control.ROUTES))
        self.route_combo.activated.connect(self._load_route)
        form.addRow("Route", self.route_combo)
        self.loop_check = QCheckBox("Loop the route until STOP")
        form.addRow("", self.loop_check)
        outer.addWidget(settings_box)

        drive_box = QGroupBox("Keyboard drive")
        drive_layout = QVBoxLayout(drive_box)
        self.drive_check = QCheckBox("Arrow keys = X / Y,  W / S = up / down")
        self.drive_check.toggled.connect(self._toggle_drive)
        drive_layout.addWidget(self.drive_check)
        self.drive_status = QLabel("")
        self.drive_status.setWordWrap(True)
        drive_layout.addWidget(self.drive_status)
        outer.addWidget(drive_box)
        self.drive = keyboard_drive.KeyboardDrive(self.control_panel, self)
        self.drive.status_changed.connect(self.drive_status.setText)

        button_row = QHBoxLayout()
        self.start_btn = QPushButton("Start")
        self.start_btn.clicked.connect(self.start_run)
        button_row.addWidget(self.start_btn)
        self.stop_btn = QPushButton("STOP")
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet("font-weight: bold; color: #c00;")
        self.stop_btn.clicked.connect(self.stop_run)
        button_row.addWidget(self.stop_btn)
        outer.addLayout(button_row)

        status_box = QGroupBox("Status")
        status_layout = QVBoxLayout(status_box)
        self.status_label = QLabel("Idle")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("font-family: Consolas, monospace;")
        status_layout.addWidget(self.status_label)
        outer.addWidget(status_box)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(2000)
        outer.addWidget(self.log, stretch=1)

    def _load_route(self, index):
        if index == 0 or self._running():
            return
        name = self.route_combo.itemText(index)
        self.waypoints.clear()
        self.waypoints.extend(closed_loop_control.route_waypoints(name))
        if self.waypoints_changed:
            self.waypoints_changed()
        self._log(f"{name} loaded")

    def _toggle_drive(self, on):
        if on and self._running():
            self.drive_status.setText("Stop the navigation run first.")
            self.drive_check.setChecked(False)
            return
        if self.drive.set_enabled(on) is False:
            self.drive_check.setChecked(False)
            return
        self.start_btn.setEnabled(not on)

    def on_measurement(self, fused, top_result=None, side_result=None):
        if fused is None or any(fused[k] is None for k in ("x", "y", "z")):
            return
        self._latest_xyz = (fused["x"], fused["y"], fused["z"])
        self._latest_views = (top_result, side_result)
        if self._running():
            self.worker.set_measurement(self._latest_xyz, top_result, side_result)

    def _running(self):
        return self.worker is not None and self.worker.isRunning()

    def start_run(self):
        cp = self.control_panel
        if cp.coil is None:
            self._log("Coils not connected (Control tab).")
            return
        if self.drive.enabled:
            self._log("Keyboard drive is on - switch it off first.")
            return
        if not self.waypoints:
            self._log("No waypoints - choose a route or add some first.")
            return
        if self._latest_xyz is None:
            self._log("No position yet - both cameras must be detecting the bot.")
            return
        name, path_follow = CONTROLLERS[self.controller_combo.currentIndex()]
        navigator = closed_loop_control.WaypointNavigator(list(self.waypoints), self._latest_xyz,
                                                          config={"path_follow": path_follow},
                                                          loop=self.loop_check.isChecked())
        self.worker = NavigationWorker(cp.coil, cp.coil_lock, navigator)
        self.worker.set_measurement(self._latest_xyz, *self._latest_views)
        self.worker.status.connect(self._on_status)
        self.worker.finished_run.connect(self._on_finished)
        self.worker.log_line.connect(self._log)
        self.worker.start()
        self._set_running(True)
        looping = ", looping" if navigator.loop else ""
        self._log(f"-- started: {name}, {len(self.waypoints)} waypoints{looping} --")

    def stop_run(self):
        if self._running():
            self.worker.request_stop()
            self.status_label.setText("Stopping...")

    def shutdown(self):
        if self.drive.enabled:
            self.drive_check.setChecked(False)
        if self._running():
            self.worker.request_stop()
            self.worker.wait(3000)

    def _set_running(self, running):
        self.start_btn.setEnabled(not running)
        self.stop_btn.setEnabled(running)
        for widget in (self.controller_combo, self.route_combo, self.loop_check, self.drive_check):
            widget.setEnabled(not running)
        self.control_panel.send_btn.setEnabled(not running)

    def _on_status(self, s):
        x, y, z = s["position"]
        lines = [
            f"Lap {s['lap']}   waypoint {s['leg']}/{s['legs']}   distance {s['distance_mm']:5.1f} mm",
            f"Position ({x:5.1f}, {y:5.1f}, {z:5.1f}) mm   {s['freq_hz']:4.1f} Hz",
            f"Phase: {PHASE_NAMES.get(s['phase'], s['phase'])}",
        ]
        if s["head_err_deg"] is not None:
            lines.append(f"Heading error {s['head_err_deg']:4.1f} deg   correction {s['corr_deg']:4.1f} deg")
        self.status_label.setText("\n".join(lines))

    def _on_finished(self, reason):
        self._set_running(False)
        self.status_label.setText(f"Finished: {reason}. Coils stopped.")
        self._log(f"-- finished: {reason} --")

    def _log(self, text):
        self.log.appendPlainText(text)
