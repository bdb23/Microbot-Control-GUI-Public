import json
import math
import os

import numpy as np


TANK_BOUNDS_MM = {"x": (0.0, 120.0), "y": (0.0, 50.0), "z": (0.0, 50.0)}

COIL_AXIS_MAP = {"x": (1, "x"), "y": (-1, "y"), "z": (-1, "z")}

AXIS_MAP_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibration", "coil_axis_map.json")

if os.path.exists(AXIS_MAP_FILE):
    with open(AXIS_MAP_FILE) as f:
        COIL_AXIS_MAP.update({k: (int(v[0]), v[1]) for k, v in json.load(f).items()})

ROUTES = {
    "Route A - rectangle": [
        ("A1", 25, 15, 18), ("A2", 95, 15, 18), ("A3", 95, 35, 22), ("A4", 25, 35, 22),
    ],
    "Route B - spiral climb": [
        ("B1", 25, 16, 16), ("B2", 60, 16, 18), ("B3", 95, 16, 20), ("B4", 95, 31, 22),
        ("B5", 60, 31, 24), ("B6", 25, 31, 26), ("B7", 60, 24, 28),
    ],
    "Route C - slalom": [
        ("C1", 20, 16, 18), ("C2", 40, 30, 24), ("C3", 60, 16, 28), ("C4", 80, 30, 24),
        ("C5", 100, 16, 18), ("C6", 100, 30, 17), ("C7", 60, 30, 17),
    ],
}

CONTROL_CONFIG = {
    "path_follow": "line",
    "freq_hz": 18.0,
    "freq_min_hz": 12.0,
    "speed_hz_per_mm": 2.0,
    "field_magnitude": 1.0,
    "rotation_dir": 1,
    "head_kp": 0.2,
    "head_ki": 0.5,
    "head_kd": 0.0,
    "head_d_filter_s": 0.3,
    "head_i_limit_deg": 30.0,
    "head_max_corr_deg": 60.0,
    "line_lookahead_mm": 5.0,
    "line_ki": 0.7,
    "line_int_max_mm": 6.0,
    "arrival_tolerance_mm": 2.0,
    "depth_tolerance_mm": 3.0,
    "capture_radius_mm": 6.0,
    "closest_approach_s": 2.0,
    "orbit_stop_laps": 2.0,
    "turn_freq_hz": 10.0,
    "align_deg": 25.0,
    "turn_timeout_s": 8.0,
    "lay_flat_s": 2.0,
    "start_hold_s": 0.6,
    "tail_first_s": 1.5,
    "axis_rate_dps": 720.0,
    "freq_rate_hz_per_s": 40.0,
    "control_rate_hz": 10.0,
    "leg_timeout_s": 60.0,
    "wrong_way_mm": 4.0,
    "wrong_way_after_s": 3.0,
    "stuck_s": 3.0,
    "stuck_mm": 1.0,
    "position_timeout_s": 0.5,
    "lost_stop_s": 1.5,
}

LIVE_OVERLAY = {"path": [], "ref": None}

SIDE_END_ON = 0.35


class Waypoint:
    def __init__(self, x, y, z, label=None):
        self.position = tuple(min(max(float(v), TANK_BOUNDS_MM[a][0]), TANK_BOUNDS_MM[a][1])
                              for v, a in zip((x, y, z), "xyz"))
        self.label = label or f"({x:.0f},{y:.0f},{z:.0f})"


def route_waypoints(name):
    points = ROUTES[name]
    return [Waypoint(x, y, z, label) for label, x, y, z in points + points[:1]]


def world_direction_to_coil(direction_xyz):
    coil = {"x": 0.0, "y": 0.0, "z": 0.0}
    for world_axis, value in zip("xyz", direction_xyz):
        sign, coil_axis = COIL_AXIS_MAP[world_axis]
        coil[coil_axis] += sign * value
    return (coil["x"], coil["y"], coil["z"])


def _unit(v):
    v = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


class LineLeg:
    def __init__(self, start_xyz, waypoint, config):
        self.start = np.asarray(start_xyz, dtype=np.float64)
        self.target = np.asarray(waypoint.position, dtype=np.float64)
        self.waypoint = waypoint
        self.config = config
        self.length = float(np.linalg.norm(self.target - self.start))
        self.u = _unit(self.target - self.start) if self.length > 1e-6 else np.array([1.0, 0.0, 0.0])
        self.int_off = np.zeros(3)

    def measure(self, pos):
        rel = np.asarray(pos, dtype=np.float64) - self.start
        along = float(np.dot(rel, self.u))
        return along, rel - along * self.u

    def integrate(self, pos, dt):
        ki = self.config["line_ki"]
        if ki <= 0 or dt <= 0:
            return
        _, off = self.measure(pos)
        self.int_off = self.int_off + off * dt
        limit = self.config["line_int_max_mm"] / ki
        n = float(np.linalg.norm(self.int_off))
        if n > limit:
            self.int_off *= limit / n

    def within_tolerance(self, pos):
        rel = self.target - np.asarray(pos, dtype=np.float64)
        return (math.hypot(rel[0], rel[1]) <= self.config["arrival_tolerance_mm"]
                and abs(rel[2]) <= self.config["depth_tolerance_mm"])

    def arrived(self, pos):
        if self.within_tolerance(pos):
            return True
        if self.config["path_follow"] != "line":
            return False
        along, _ = self.measure(pos)
        near = float(np.linalg.norm(self.target - np.asarray(pos, dtype=np.float64))) <= self.config["capture_radius_mm"]
        return along >= self.length and near

    def direction(self, pos):
        pos = np.asarray(pos, dtype=np.float64)
        along, _ = self.measure(pos)
        if self.config["path_follow"] == "line":
            aim = self.start + (along + self.config["line_lookahead_mm"]) * self.u - pos
            aim = aim - self.config["line_ki"] * self.int_off
        else:
            aim = self.target - pos
        if float(np.linalg.norm(aim)) < 1e-6:
            aim = self.u
        return _unit(aim), along


def view_headings(top_result=None, side_result=None):
    out = []
    for r in (top_result, side_result):
        if r and r.get("angle_deg") is not None:
            th = math.radians(r["angle_deg"])
            out.append(np.array([math.cos(th), -math.sin(th)]))
        else:
            out.append(None)
    top, side = out
    if (top is not None and side is not None and abs(top[1]) > 0.3 and abs(side[0]) > 0.3
            and np.sign(top[1]) != np.sign(side[0])):
        out[1] = -side
    if out[0] is not None and out[1] is not None and abs(out[0][1]) < SIDE_END_ON:
        out[1] = None
    return out


def head_direction(top_result=None, side_result=None):
    top, side = view_headings(top_result, side_result)
    if top is not None and side is not None:
        tx, ty = top
        sy, sz = side
        if abs(ty) > 0.2 and abs(sy) > 0.2 and np.sign(ty) == np.sign(sy):
            return _unit([tx, ty, sz * ty / sy])
        return _unit([tx, 0.5 * (ty + sy), sz])
    if top is not None:
        return _unit([top[0], top[1], 0.0])
    if side is not None:
        return _unit([0.0, side[0], side[1]])
    return None


def facing_error_deg(direction, top_result=None, side_result=None):
    worst = None
    views = view_headings(top_result, side_result)
    projections = (np.array([direction[0], direction[1]]), np.array([direction[1], direction[2]]))
    for front, p in zip(views, projections):
        if front is None or np.linalg.norm(p) < 0.5:
            continue
        angle = math.degrees(math.acos(float(np.clip(np.dot(front, _unit(p)), -1.0, 1.0))))
        worst = angle if worst is None else max(worst, angle)
    return worst


def rotate_by(v, rotvec):
    v = np.asarray(v, dtype=np.float64)
    rotvec = np.asarray(rotvec, dtype=np.float64)
    angle = float(np.linalg.norm(rotvec))
    if angle < 1e-12:
        return v.copy()
    k = rotvec / angle
    return v * math.cos(angle) + np.cross(k, v) * math.sin(angle) + k * float(np.dot(k, v)) * (1 - math.cos(angle))


def _signed_angle_2d(a, b):
    return math.atan2(a[0] * b[1] - a[1] * b[0], a[0] * b[0] + a[1] * b[1])


def heading_error(direction, top_result=None, side_result=None):
    d = _unit(direction)
    top, side = view_headings(top_result, side_result)
    if top is not None and side is not None:
        h = head_direction(top_result, side_result)
        axis = np.cross(h, d)
        if np.linalg.norm(axis) < 1e-9:
            return np.zeros(3)
        return _unit(axis) * math.atan2(float(np.linalg.norm(axis)), float(np.dot(h, d)))
    if top is not None and math.hypot(d[0], d[1]) > 0.3:
        return np.array([0.0, 0.0, _signed_angle_2d(top, d[:2])])
    if side is not None and math.hypot(d[1], d[2]) > 0.3:
        return np.array([_signed_angle_2d(side, d[1:]), 0.0, 0.0])
    return None


class HeadingPID:
    def __init__(self, config):
        self.config = config
        self.reset()

    def reset(self):
        self.integral = np.zeros(3)
        self.prev_error = None
        self.d_state = np.zeros(3)

    def update(self, error, dt, freeze=False):
        cfg = self.config
        if error is None or dt <= 0:
            self.prev_error = None
            out = cfg["head_ki"] * self.integral
        elif freeze:
            self.prev_error = None
            out = cfg["head_kp"] * error + cfg["head_ki"] * self.integral
        else:
            self.integral += error * dt
            limit = math.radians(cfg["head_i_limit_deg"]) / max(cfg["head_ki"], 1e-9)
            n = float(np.linalg.norm(self.integral))
            if n > limit:
                self.integral *= limit / n
            if self.prev_error is not None:
                raw = (error - self.prev_error) / dt
                alpha = dt / (cfg["head_d_filter_s"] + dt)
                self.d_state += alpha * (raw - self.d_state)
            self.prev_error = error.copy()
            out = cfg["head_kp"] * error + cfg["head_ki"] * self.integral + cfg["head_kd"] * self.d_state
        cap = math.radians(cfg["head_max_corr_deg"])
        n = float(np.linalg.norm(out))
        return out * (cap / n) if n > cap else out


class WaypointNavigator:
    def __init__(self, waypoints, start_xyz, config=None, loop=False):
        self.config = dict(CONTROL_CONFIG, **(config or {}))
        self.waypoints = list(waypoints)
        if not self.waypoints:
            raise ValueError("need at least one waypoint")
        self.loop = loop and len({wp.position for wp in self.waypoints}) > 1
        self.lap = 1
        self.index = 0
        self.leg = LineLeg(start_xyz, self.waypoints[0], self.config)
        self.leg_start_t = None
        self.phase = "layflat"
        self.phase_t = None
        self.spin = 1
        self.flips = 0
        self.history = []
        self.heading_pid = HeadingPID(self.config)
        self.done = False
        self.stop_reason = None
        self.events = []
        self._last_t = None
        self._tail_s = 0.0
        self._capture = None
        self._orbit_reason = None
        self._last_dir = None
        self._turn_from = None
        self._start_dir = None
        self._flip_axis = None
        self._swim_along0 = 0.0

    def _set_phase(self, phase, t, pos=None):
        self.phase = phase
        self.phase_t = t
        if phase == "turning":
            self._turn_from = self._last_dir
        if phase == "swimming" and pos is not None:
            self._swim_along0 = self.leg.measure(pos)[0]

    def _closest_approach(self, pos, t):
        cfg = self.config
        if self.phase != "swimming":
            self._capture = None
            return False
        recent = [p for tt, p in self.history if tt >= t - 0.5] or [pos]
        rel = self.leg.target - np.mean(recent, axis=0)
        dist = float(np.linalg.norm(rel))
        if dist >= cfg["capture_radius_mm"]:
            self._capture = None
            return False
        bearing = math.atan2(rel[1], rel[0])
        c = self._capture
        if c is None:
            self._capture = {"best": dist, "best_t": t, "bearing": bearing, "swept": 0.0}
            return False
        if math.hypot(rel[0], rel[1]) > 1.5:
            c["swept"] += abs((bearing - c["bearing"] + math.pi) % (2 * math.pi) - math.pi)
        c["bearing"] = bearing
        if c["swept"] >= 2 * math.pi * cfg["orbit_stop_laps"]:
            self._orbit_reason = f"circling {self.leg.waypoint.label} - stopped"
            return False
        if dist < c["best"] - 0.15:
            c["best"], c["best_t"] = dist, t
        return t - c["best_t"] >= cfg["closest_approach_s"]

    def _velocity(self, window_s=0.8):
        if len(self.history) < 2:
            return None
        t1, p1 = self.history[-1]
        t0, p0 = next(((t, p) for t, p in self.history if t >= t1 - window_s), self.history[0])
        return (p1 - p0) / (t1 - t0) if t1 - t0 > 0.4 else None

    def _check_tail_first(self, dt, top_result, side_result, events):
        v = self._velocity()
        if v is None or float(np.linalg.norm(v)) < 1.0 or self.phase not in ("turning", "swimming"):
            return
        if float(np.linalg.norm(self.leg.target - self.history[-1][1])) < self.config["capture_radius_mm"]:
            self._tail_s = 0.0
            return
        head = head_direction(top_result, side_result)
        if head is None:
            return
        err = math.degrees(math.acos(float(np.clip(np.dot(_unit(v), head), -1.0, 1.0))))
        if err > 135:
            self._tail_s += dt
        elif err < 60:
            self._tail_s = 0.0
        if self._tail_s < self.config["tail_first_s"]:
            return
        self._tail_s = 0.0
        if self.flips >= 3:
            self.stop_reason = "swimming tail-first after 3 flips - stopped"
            return
        self.flips += 1
        self._flip_axis = self.spin * self._last_dir
        self.spin = -self.spin
        self.heading_pid.reset()
        events.append("swimming tail-first - reversing the spin")
        self._set_phase("flipping", self._last_t)

    def _next_leg(self, pos, t, events):
        self.index += 1
        if self.index >= len(self.waypoints):
            if not self.loop:
                self.done = True
                return False
            self.index = 1 if self.waypoints[-1].position == self.waypoints[0].position else 0
            self.lap += 1
            events.append(f"-- lap {self.lap} --")
        self.leg = LineLeg(pos, self.waypoints[self.index], self.config)
        self.leg_start_t = t
        self._set_phase("turning", t)
        events.append(f"leg {self.index + 1}/{len(self.waypoints)} -> {self.leg.waypoint.label}")
        return True

    def update(self, pos, t, top_result=None, side_result=None):
        cfg = self.config
        pos = np.asarray(pos, dtype=np.float64)
        events = []
        self.events = events
        dt = 0.0 if self._last_t is None else t - self._last_t
        self._last_t = t
        self.history.append((t, pos.copy()))
        while self.history and self.history[0][0] < t - max(2.0, cfg["stuck_s"] + 0.5):
            self.history.pop(0)
        if self.leg_start_t is None:
            self.leg_start_t = t
            self.phase_t = t
            events.append(f"leg 1/{len(self.waypoints)} -> {self.leg.waypoint.label}")

        for _ in range(len(self.waypoints)):
            if not (self.leg.arrived(pos) or self._closest_approach(pos, t)):
                break
            dist = float(np.linalg.norm(self.leg.target - pos))
            events.append(f"reached {self.leg.waypoint.label}: {dist:.1f} mm from it")
            self._capture = None
            if not self._next_leg(pos, t, events):
                return None

        if self._orbit_reason:
            self.stop_reason = self._orbit_reason
            return None
        self._check_tail_first(dt, top_result, side_result, events)
        if self.stop_reason:
            return None

        if self.phase == "swimming" and cfg["path_follow"] == "line":
            self.leg.integrate(pos, dt)
        direction, along = self.leg.direction(pos)
        facing = facing_error_deg(direction, top_result, side_result)
        freq = cfg["freq_hz"]
        static = None

        if self.phase == "layflat":
            static = (0.0, 0.0, 1.0)
            freq = 0.0
            if t - self.phase_t >= cfg["lay_flat_s"]:
                self._set_phase("start", t)
        if self.phase == "start":
            head = head_direction(top_result, side_result)
            if head is not None:
                direction = head
            freq = cfg["turn_freq_hz"]
            self._start_dir = direction
            self._set_phase("start_hold", t)
        elif self.phase == "start_hold":
            direction, freq = self._start_dir, cfg["turn_freq_hz"]
            if t - self.phase_t >= cfg["start_hold_s"]:
                self._set_phase("turning", t)
        elif self.phase == "flipping":
            freq = cfg["turn_freq_hz"]
            if t - self.phase_t >= (cfg["freq_hz"] + cfg["turn_freq_hz"]) / cfg["freq_rate_hz_per_s"] + 0.4:
                self._set_phase("turning", t)

        if self.phase == "turning":
            freq = cfg["turn_freq_hz"]
            turned = t - self.phase_t
            start_dir = self._turn_from if self._turn_from is not None else direction
            need = math.degrees(math.acos(float(np.clip(np.dot(start_dir, direction), -1, 1))))
            slewed = turned >= need / cfg["axis_rate_dps"] + 0.5
            if (slewed and (facing is None or facing <= cfg["align_deg"])) or turned >= cfg["turn_timeout_s"]:
                events.append(f"facing {self.leg.waypoint.label} - swimming")
                self._set_phase("swimming", t, pos)
                freq = cfg["freq_hz"]

        head_err = None
        correction = np.zeros(3)
        if self.phase == "swimming":
            head_err = heading_error(direction, top_result, side_result)
            dist = float(np.linalg.norm(self.leg.target - pos))
            correction = self.heading_pid.update(head_err, dt, freeze=dist < cfg["capture_radius_mm"])
            direction = _unit(rotate_by(direction, correction))
            freq = min(cfg["freq_hz"], max(cfg["freq_min_hz"], cfg["speed_hz_per_mm"] * dist))
            if head_err is not None:
                freq = max(cfg["freq_min_hz"], freq * math.cos(min(float(np.linalg.norm(head_err)), math.pi / 2)))
            swum = t - self.phase_t
            if swum > cfg["wrong_way_after_s"] and along - self._swim_along0 < -cfg["wrong_way_mm"]:
                self.stop_reason = f"moving away from {self.leg.waypoint.label} - check the coil axis map"
                return None
            if swum > cfg["stuck_s"] and t - self.history[0][0] >= cfg["stuck_s"]:
                window = [p for tt, p in self.history if tt >= t - cfg["stuck_s"]]
                moved = float(np.max(np.linalg.norm(np.asarray(window) - window[0], axis=1)))
                if moved < cfg["stuck_mm"]:
                    self.stop_reason = "stuck against a wall, the floor or the surface - nudge it free"
                    return None
        if t - self.leg_start_t > cfg["leg_timeout_s"]:
            self.stop_reason = f"leg to {self.leg.waypoint.label} took too long - stopped"
            return None

        if static is not None:
            axis = static
        elif self.phase == "flipping":
            axis = tuple(self._flip_axis)
            direction = tuple(self.spin * np.asarray(self._flip_axis))
        else:
            axis = tuple(self.spin * np.asarray(direction))
        self._last_dir = np.asarray(direction, dtype=np.float64)
        return {
            "coil_axis": world_direction_to_coil(axis),
            "static": None if static is None else world_direction_to_coil(static),
            "freq_hz": freq,
            "direction": int(self.spin * cfg["rotation_dir"]),
            "magnitude": cfg["field_magnitude"],
            "phase": self.phase,
            "leg": self.index + 1,
            "lap": self.lap,
            "legs": len(self.waypoints),
            "target": tuple(self.leg.target),
            "leg_start": tuple(self.leg.start),
            "distance_mm": float(np.linalg.norm(self.leg.target - pos)),
            "head_err_deg": None if head_err is None else math.degrees(float(np.linalg.norm(head_err))),
            "corr_deg": math.degrees(float(np.linalg.norm(correction))),
            "events": events,
        }
