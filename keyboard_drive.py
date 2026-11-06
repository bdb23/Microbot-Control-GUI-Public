import numpy as np
from PySide6.QtCore import QEvent, QObject, Qt, Signal, QMutexLocker
from PySide6.QtWidgets import QApplication, QLineEdit, QTextEdit, QPlainTextEdit

import closed_loop_control

KEY_DIRS = {
    Qt.Key_Right: (1.0, 0.0, 0.0), Qt.Key_Left: (-1.0, 0.0, 0.0),
    Qt.Key_Up: (0.0, 1.0, 0.0), Qt.Key_Down: (0.0, -1.0, 0.0),
    Qt.Key_W: (0.0, 0.0, 1.0), Qt.Key_S: (0.0, 0.0, -1.0),
}
KEY_NAMES = {Qt.Key_Right: "+X", Qt.Key_Left: "-X", Qt.Key_Up: "+Y", Qt.Key_Down: "-Y", Qt.Key_W: "+Z", Qt.Key_S: "-Z"}

DRIVE_FREQ_HZ = 12.0


class KeyboardDrive(QObject):
    status_changed = Signal(str)

    def __init__(self, control_panel, parent=None):
        super().__init__(parent)
        self.control_panel = control_panel
        self.enabled = False
        self._held = set()
        self._last = None

    def set_enabled(self, on):
        if on == self.enabled:
            return True
        if on and self.control_panel.coil is None:
            self.status_changed.emit("Connect the coils (Control tab) first.")
            return False
        self.enabled = on
        self._held.clear()
        self._last = None
        app = QApplication.instance()
        if on:
            app.installEventFilter(self)
            cfg = closed_loop_control.CONTROL_CONFIG
            with QMutexLocker(self.control_panel.coil_lock):
                self.control_panel.coil.set_slew(cfg["axis_rate_dps"], cfg["freq_rate_hz_per_s"], 0.0)
            self._send(("hold",))
            self.status_changed.emit("Keyboard drive on")
        else:
            app.removeEventFilter(self)
            self._send(("off",))
            self.status_changed.emit("Keyboard drive off")
        return True

    def eventFilter(self, obj, event):
        if not self.enabled or event.type() not in (QEvent.KeyPress, QEvent.KeyRelease):
            return False
        key = event.key()
        if key not in KEY_DIRS:
            return False
        if isinstance(QApplication.focusWidget(), (QLineEdit, QTextEdit, QPlainTextEdit)):
            return False
        if event.isAutoRepeat():
            return True
        if event.type() == QEvent.KeyPress:
            self._held.add(key)
        else:
            self._held.discard(key)
        self._update()
        return True

    def _update(self):
        v = np.zeros(3)
        for k in self._held:
            v += KEY_DIRS[k]
        if np.linalg.norm(v) < 1e-6:
            cmd = ("hold",)
        else:
            cmd = ("rotate", tuple(np.round(v / np.linalg.norm(v), 4)))
        if cmd != self._last:
            self._send(cmd)

    def _send(self, cmd):
        coil = self.control_panel.coil
        if coil is None:
            self.enabled = False
            QApplication.instance().removeEventFilter(self)
            self.status_changed.emit("Keyboard drive stopped - coils disconnected")
            return
        cfg = closed_loop_control.CONTROL_CONFIG
        try:
            with QMutexLocker(self.control_panel.coil_lock):
                if cmd[0] == "rotate":
                    coil.set_rotate(magnitude=cfg["field_magnitude"], axis=closed_loop_control.world_direction_to_coil(cmd[1]),
                                    freq_hz=DRIVE_FREQ_HZ, direction=int(cfg["rotation_dir"]))
                elif cmd[0] == "hold":
                    b = np.asarray(closed_loop_control.world_direction_to_coil((0.0, 0.0, 1.0))) * cfg["field_magnitude"]
                    coil.set_static(float(b[0]), float(b[1]), float(b[2]))
                else:
                    coil.stop()
        except Exception as exc:
            self.status_changed.emit(f"Keyboard drive: command failed: {exc}")
            return
        self._last = cmd
        if cmd[0] == "rotate":
            self.status_changed.emit("Swimming " + " ".join(KEY_NAMES[k] for k in sorted(self._held)))
        elif self.enabled:
            self.status_changed.emit("No key - holding flat")
