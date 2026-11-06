import serial
import serial.tools.list_ports
from PySide6.QtCore import QThread, Signal, QMutex, QMutexLocker
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox,
    QLineEdit, QPlainTextEdit, QGroupBox
)

from coil_controller import CoilController


TEMPLATES = {
    "Stop": "stop",
    "Sense Start": "sense_start",
    "Sense Stop": "sense_stop",
    "Set Static": "set_static bx=0.0 by=0.0 bz=1.0",
    "Set Rotate": "set_rotate mag=1.0 axis=0,0,1 freq=15.0 dir=1",
    "Set Slew": "set_slew axis=90 freq=20 mag=2",
}


def parse_command(text):
    tokens = text.strip().split()
    if not tokens:
        raise ValueError("Empty command")

    cmd_name = tokens[0].lower()
    kwargs = {}
    for tok in tokens[1:]:
        if "=" not in tok:
            raise ValueError(f"Expected key=value, got {tok!r}")
        key, val = tok.split("=", 1)
        kwargs[key.strip().lower()] = val.strip()

    return cmd_name, kwargs


def dispatch(coil: CoilController, cmd_name: str, kwargs: dict):
    if cmd_name == "stop":
        return coil.stop()

    if cmd_name == "sense_start":
        return coil.sense_start()

    if cmd_name == "sense_stop":
        return coil.sense_stop()

    if cmd_name == "set_static":
        try:
            bx, by, bz = float(kwargs["bx"]), float(kwargs["by"]), float(kwargs["bz"])
        except KeyError as exc:
            raise ValueError(f"set_static missing parameter: {exc}")
        except ValueError:
            raise ValueError("set_static: bx/by/bz must be numbers")
        return coil.set_static(bx, by, bz)

    if cmd_name == "set_rotate":
        try:
            mag = float(kwargs["mag"])
            axis = tuple(float(v) for v in kwargs["axis"].split(","))
            freq = float(kwargs["freq"])
            direction = int(kwargs.get("dir", 1))
        except KeyError as exc:
            raise ValueError(f"set_rotate missing parameter: {exc}")
        except ValueError:
            raise ValueError("set_rotate: mag/axis/freq/dir must be numbers, axis as x,y,z")
        if len(axis) != 3:
            raise ValueError("set_rotate: axis must have exactly 3 components (x,y,z)")
        return coil.set_rotate(magnitude=mag, axis=axis, freq_hz=freq, direction=direction)

    if cmd_name == "set_slew":
        try:
            axis_dps = float(kwargs["axis"])
            freq_hzps = float(kwargs["freq"])
            mag_ps = float(kwargs["mag"])
        except KeyError as exc:
            raise ValueError(f"set_slew missing parameter: {exc}")
        except ValueError:
            raise ValueError("set_slew: axis/freq/mag must be numbers "
                              "(deg/s, Hz/s, 1/s respectively - <=0 leaves that limit unchanged)")
        return coil.set_slew(axis_deg_per_s=axis_dps, freq_hz_per_s=freq_hzps, mag_per_s=mag_ps)

    raise ValueError(
        f"Unknown command {cmd_name!r}. Expected one of: "
        f"stop, sense_start, sense_stop, set_static, set_rotate, set_slew"
    )


class ConnectWorker(QThread):
    connected = Signal(object)
    error = Signal(str)

    def __init__(self, port, baud, parent=None):
        super().__init__(parent)
        self.port = port
        self.baud = baud

    def run(self):
        try:
            coil = CoilController(self.port, baudrate=self.baud)
        except (serial.SerialException, OSError) as exc:
            self.error.emit(str(exc))
            return
        self.connected.emit(coil)


class CommandWorker(QThread):
    succeeded = Signal(str, object)
    failed = Signal(str, str)

    def __init__(self, coil, coil_lock, cmd_name, kwargs, parent=None):
        super().__init__(parent)
        self.coil = coil
        self.coil_lock = coil_lock
        self.cmd_name = cmd_name
        self.kwargs = kwargs

    def run(self):
        try:
            with QMutexLocker(self.coil_lock):
                result = dispatch(self.coil, self.cmd_name, self.kwargs)
        except (ValueError, TimeoutError, RuntimeError, serial.SerialException) as exc:
            self.failed.emit(self.cmd_name, str(exc))
            return
        self.succeeded.emit(self.cmd_name, result)


class ControlPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.coil = None
        self.coil_lock = QMutex()
        self.connect_worker = None
        self.command_worker = None
        self.disconnect_hooks = []
        self._build_ui()
        self.refresh_ports()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        conn_box = QGroupBox("Connection")
        conn_layout = QHBoxLayout(conn_box)

        conn_layout.addWidget(QLabel("Port:"))
        self.port_combo = QComboBox()
        self.port_combo.setMinimumWidth(140)
        conn_layout.addWidget(self.port_combo)

        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self.refresh_ports)
        conn_layout.addWidget(refresh_btn)

        conn_layout.addWidget(QLabel("Baud:"))
        self.baud_combo = QComboBox()
        self.baud_combo.setEditable(True)
        self.baud_combo.addItems(["9600", "19200", "38400", "57600", "115200"])
        self.baud_combo.setCurrentText("115200")
        conn_layout.addWidget(self.baud_combo)

        self.connect_btn = QPushButton("Connect")
        self.connect_btn.clicked.connect(self.toggle_connection)
        conn_layout.addWidget(self.connect_btn)

        conn_layout.addStretch()
        layout.addWidget(conn_box)

        quick_box = QGroupBox("Quick commands")
        quick_layout = QHBoxLayout(quick_box)
        for label, template in TEMPLATES.items():
            btn = QPushButton(label)
            btn.clicked.connect(lambda checked=False, t=template: self._load_template(t))
            quick_layout.addWidget(btn)
        layout.addWidget(quick_box)

        input_row = QHBoxLayout()
        self.input_line = QLineEdit()
        self.input_line.returnPressed.connect(self.send_command)
        input_row.addWidget(self.input_line)

        self.send_btn = QPushButton("Send")
        self.send_btn.clicked.connect(self.send_command)
        input_row.addWidget(self.send_btn)
        layout.addLayout(input_row)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(5000)
        self.log.setStyleSheet("font-family: Consolas, monospace;")
        layout.addWidget(self.log, stretch=1)

        self.status_label = QLabel("Disconnected")
        layout.addWidget(self.status_label)

    def refresh_ports(self):
        current = self.port_combo.currentText()
        self.port_combo.clear()
        ports = [p.device for p in serial.tools.list_ports.comports()]
        self.port_combo.addItems(ports)
        if current in ports:
            self.port_combo.setCurrentText(current)

    def toggle_connection(self):
        if self.coil is not None:
            self.disconnect_coil()
        else:
            self.connect_coil()

    def connect_coil(self):
        port = self.port_combo.currentText()
        if not port:
            self._log_line("No port selected.")
            return
        try:
            baud = int(self.baud_combo.currentText())
        except ValueError:
            self._log_line(f"Invalid baud rate: {self.baud_combo.currentText()}")
            return

        self.status_label.setText(f"Connecting to {port}...")
        self.connect_btn.setEnabled(False)

        self.connect_worker = ConnectWorker(port, baud)
        self.connect_worker.connected.connect(self._on_connected)
        self.connect_worker.error.connect(self._on_connect_error)
        self.connect_worker.start()

    def _on_connected(self, coil):
        self.coil = coil
        self.connect_btn.setText("Disconnect")
        self.connect_btn.setEnabled(True)
        self.status_label.setText(f"Connected: {self.port_combo.currentText()}")
        self._log_line("-- connected --")

    def _on_connect_error(self, message):
        self.connect_btn.setEnabled(True)
        self.status_label.setText("Disconnected")
        self._log_line(f"Connect failed: {message}")

    def disconnect_coil(self):
        for hook in self.disconnect_hooks:
            hook()
        if self.coil is not None:
            try:
                self.coil.close()
            except serial.SerialException:
                pass
            self.coil = None
        self.connect_btn.setText("Connect")
        self.status_label.setText("Disconnected")
        self._log_line("-- disconnected --")

    def _load_template(self, template):
        self.input_line.setText(template)
        self.input_line.setFocus()

    def send_command(self):
        if self.coil is None:
            self._log_line("Not connected.")
            return

        text = self.input_line.text().strip()
        if not text:
            return

        try:
            cmd_name, kwargs = parse_command(text)
        except ValueError as exc:
            self._log_line(f"Parse error: {exc}")
            return

        self._log_line(f"TX: {text}")
        self.send_btn.setEnabled(False)

        self.command_worker = CommandWorker(self.coil, self.coil_lock, cmd_name, kwargs)
        self.command_worker.succeeded.connect(self._on_command_success)
        self.command_worker.failed.connect(self._on_command_failed)
        self.command_worker.start()

    def _on_command_success(self, cmd_name, result):
        self.send_btn.setEnabled(True)
        if result is None:
            self._log_line(f"OK: {cmd_name} acknowledged (no data)")
        else:
            self._log_line(f"OK: {cmd_name} ACK: {bytes(result).hex(' ').upper()}")

    def _on_command_failed(self, cmd_name, message):
        self.send_btn.setEnabled(True)
        self._log_line(f"FAILED: {cmd_name}: {message}")

    def _log_line(self, text):
        self.log.appendPlainText(text)

    def closeEvent(self, event):
        self.disconnect_coil()
        super().closeEvent(event)
