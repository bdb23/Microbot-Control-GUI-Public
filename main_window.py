import time

from PySide6.QtCore import Qt
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QSplitter, QTabWidget, QPushButton, QLabel, QGroupBox,
    QDoubleSpinBox, QListWidget, QScrollArea, QFrame,
)

import closed_loop_control
import world_frame
from camera_panel import CameraPanel
from closed_loop_panel import ClosedLoopPanel
from control_panel import ControlPanel
from hall_sensor_panel import HallSensorPanel


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Microbot Control")
        screen = QGuiApplication.primaryScreen()
        if screen is not None:
            avail = screen.availableGeometry()
            self.resize(min(1500, int(avail.width() * 0.95)), min(850, int(avail.height() * 0.92)))
        else:
            self.resize(1500, 850)

        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QHBoxLayout(central)

        self.waypoints = []

        self.cam_a = CameraPanel("Top Camera", default_index=2, camera_name="top", waypoints=self.waypoints)
        self.cam_b = CameraPanel("Side Camera", default_index=0, camera_name="side", waypoints=self.waypoints)
        cam_splitter = QSplitter(Qt.Vertical)
        cam_splitter.addWidget(self.cam_a)
        cam_splitter.addWidget(self.cam_b)

        world_box = QGroupBox("Microbot Position")
        world_layout = QHBoxLayout(world_box)
        self.world_position_label = QLabel("Waiting for both cameras...")
        self.world_position_label.setWordWrap(True)
        world_layout.addWidget(self.world_position_label)

        self._latest_top_detection = None
        self._latest_side_detection = None
        self._top_detection_t = 0.0
        self._side_detection_t = 0.0
        self.cam_a.detection_ready.connect(self._on_top_detection)
        self.cam_b.detection_ready.connect(self._on_side_detection)

        waypoint_box = QGroupBox("Waypoints")
        waypoint_layout = QVBoxLayout(waypoint_box)
        entry_row = QHBoxLayout()
        self.waypoint_spins = []
        for axis in "xyz":
            low, high = closed_loop_control.TANK_BOUNDS_MM[axis]
            spin = QDoubleSpinBox()
            spin.setRange(low, high)
            spin.setDecimals(1)
            spin.setSuffix(" mm")
            spin.setValue((low + high) / 2)
            spin.setMinimumWidth(80)
            entry_row.addWidget(QLabel(f"{axis.upper()}:"))
            entry_row.addWidget(spin, stretch=1)
            self.waypoint_spins.append(spin)
        add_btn = QPushButton("Add Waypoint")
        add_btn.clicked.connect(self._add_waypoint)
        entry_row.addWidget(add_btn)
        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(self._clear_waypoints)
        entry_row.addWidget(clear_btn)
        waypoint_layout.addLayout(entry_row)
        self.waypoint_list_widget = QListWidget()
        self.waypoint_list_widget.setMaximumHeight(80)
        self.waypoint_list_widget.setMinimumHeight(40)
        waypoint_layout.addWidget(self.waypoint_list_widget)

        self.tabs = QTabWidget()
        self.control_panel = ControlPanel()
        self.add_panel("Control", self.control_panel)
        self.hall_panel = HallSensorPanel(self.control_panel)
        self.add_panel("Hall Sensor", self.hall_panel)
        self.closed_loop_panel = ClosedLoopPanel(self.control_panel, self.waypoints)
        self.closed_loop_panel.waypoints_changed = self._refresh_waypoint_list
        self.add_panel("Closed Loop", self.closed_loop_panel)

        right_column = QWidget()
        right_layout = QVBoxLayout(right_column)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.addWidget(world_box)
        right_layout.addWidget(waypoint_box)
        right_layout.addWidget(self.tabs, stretch=1)

        main_splitter = QSplitter(Qt.Horizontal)
        main_splitter.addWidget(cam_splitter)
        main_splitter.addWidget(right_column)
        main_splitter.setStretchFactor(0, 11)
        main_splitter.setStretchFactor(1, 9)
        root_layout.addWidget(main_splitter)

    def add_panel(self, title, widget):
        page = QScrollArea()
        page.setWidgetResizable(True)
        page.setFrameShape(QFrame.NoFrame)
        page.setWidget(widget)
        self.tabs.addTab(page, title)

    def _on_top_detection(self, result):
        self._latest_top_detection = result
        self._top_detection_t = time.monotonic()
        self._update_world_position()

    def _on_side_detection(self, result):
        self._latest_side_detection = result
        self._side_detection_t = time.monotonic()
        self._update_world_position()

    def _update_world_position(self):
        fused = world_frame.fuse_position(self._latest_top_detection, self._latest_side_detection)
        if fused["x"] is None or fused["y"] is None or fused["z"] is None:
            self.world_position_label.setText("Waiting for both cameras...")
        else:
            self.world_position_label.setText(f"X = {fused['x']:.1f} mm   Y = {fused['y']:.1f} mm   Z = {fused['z']:.1f} mm")
        max_age = closed_loop_control.CONTROL_CONFIG["position_timeout_s"]
        now = time.monotonic()
        fresh = now - self._top_detection_t < max_age and now - self._side_detection_t < max_age
        self.closed_loop_panel.on_measurement(fused if fresh else None,
                                              self._latest_top_detection, self._latest_side_detection)

    def _add_waypoint(self):
        x, y, z = (spin.value() for spin in self.waypoint_spins)
        self.waypoints.append(closed_loop_control.Waypoint(x, y, z, label=f"#{len(self.waypoints) + 1}"))
        self._refresh_waypoint_list()

    def _clear_waypoints(self):
        self.waypoints.clear()
        self._refresh_waypoint_list()

    def _refresh_waypoint_list(self):
        self.waypoint_list_widget.clear()
        for wp in self.waypoints:
            x, y, z = wp.position
            self.waypoint_list_widget.addItem(f"{wp.label}   ({x:.1f}, {y:.1f}, {z:.1f}) mm")

    def closeEvent(self, event):
        self.closed_loop_panel.shutdown()
        self.cam_a.disconnect_camera()
        self.cam_b.disconnect_camera()
        self.hall_panel._stop_monitoring()
        self.control_panel.disconnect_coil()
        super().closeEvent(event)
