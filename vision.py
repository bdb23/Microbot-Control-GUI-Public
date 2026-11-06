import os

import cv2
import numpy as np

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

TOP_CAMERA = {
    "roi": {"x": 0, "y": 0, "w": 1920, "h": 1080},
    "background_image_path": os.path.join(_MODULE_DIR, "calibration", "background_top.png"),
    "track_margin_px": 60,
    "max_lost_frames": 5,
    "edge_retry_expand_px": 80,
    "front_flip_frames": 15,
    "front_memory_frames": 15,
    "front_confident_votes": 5,
    "diff_threshold": 12,
    "diff_threshold_low": 7,
    "edge_gain": 0.35,
    "core_darkness_min": 35,
    "close_kernel": 21,
    "close_kernel_low": 31,
    "min_area": 800,
    "max_area": 60000,
    "track_distance_scale_px": 150,
    "heading_min_elongation": 1.6,
    "tip_is_front": False,
    "use_tip_cue": True,
    "part_darkness_frac": 0.45,
    "tip_min_area": 30,
    "tip_max_area": 600,
    "tip_max_elongation": 2.0,
}

SIDE_CAMERA = dict(
    TOP_CAMERA,
    roi={"x": 0, "y": 300, "w": 1920, "h": 780},
    background_image_path=os.path.join(_MODULE_DIR, "calibration", "background_side.png"),
    diff_threshold=14,
    edge_gain=0.5,
    use_tip_cue=False,
)

CAMERA_CONFIGS = {"top": TOP_CAMERA, "side": SIDE_CAMERA}

_background_cache = {}
_prepared_cache = {}


def get_config(camera="top"):
    return CAMERA_CONFIGS[camera]


def _load_background(config):
    path = config["background_image_path"]
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        mtime = None
    cached = _background_cache.get(path)
    if cached is None or cached[0] != mtime:
        cached = (mtime, cv2.imread(path, cv2.IMREAD_GRAYSCALE))
        _background_cache[path] = cached
    return cached[1]


def _prepared_background(config):
    background = _load_background(config)
    if background is None:
        return None
    path = config["background_image_path"]
    cached = _prepared_cache.get(path)
    if cached is None or cached[0] is not background:
        bg = cv2.GaussianBlur(background, (5, 5), 0).astype(np.float32)
        gx = cv2.Sobel(bg, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(bg, cv2.CV_32F, 0, 1, ksize=3)
        grad = cv2.dilate(cv2.magnitude(gx, gy), np.ones((5, 5), np.uint8))
        cached = (background, (bg, grad))
        _prepared_cache[path] = cached
    return cached[1]


def _exposure_offset(frame, background_blurred):
    small = cv2.cvtColor(np.ascontiguousarray(frame[::16, ::16]), cv2.COLOR_BGR2GRAY).astype(np.float32)
    return float(np.median(background_blurred[::16, ::16] - small))


def _clean_mask(mask, close_k):
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    k = max(1, close_k | 1)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))


def _tip_end(body, darkness, major, mx, my, proj_body, config):
    core = float(np.percentile(darkness[body], 99))
    if core <= 0:
        return 0, 0.0
    dark = (body & (darkness > config["part_darkness_frac"] * core)).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(dark)
    if n <= 2:
        return 0, 0.0
    darkest = np.unravel_index(np.argmax(np.where(body, darkness, -np.inf)), darkness.shape)
    magnet_id = labels[darkest]
    half_pos, half_neg = max(proj_body.max(), 1.0), max(-proj_body.min(), 1.0)
    votes = []
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < config["tip_min_area"] or i == magnet_id or area > config["tip_max_area"]:
            continue
        ys, xs = np.nonzero(labels == i)
        p = (xs - mx) * major[0] + (ys - my) * major[1]
        p = np.where(p > 0, p / half_pos, p / half_neg)
        pos = float(p.mean())
        if len(xs) >= 3:
            ev = np.linalg.eigvalsh(np.cov(np.vstack([xs, ys])))
            elong = float(np.sqrt(ev[1] / max(ev[0], 1e-6)))
        else:
            elong = 1.0
        if elong < config["tip_max_elongation"] and abs(pos) >= 0.45:
            votes.append(1 if pos > 0 else -1)
    if votes and abs(sum(votes)) == len(votes):
        return int(np.sign(sum(votes))), 0.6
    return 0, 0.0


def _heading(body, weights, darkness, core_mask, config):
    ys, xs = np.nonzero(body)
    w = weights[ys, xs]
    total = w.sum()
    if total <= 0 or len(xs) < 20:
        return None, 0.0, None
    mx, my = (w * xs).sum() / total, (w * ys).sum() / total
    dx, dy = xs - mx, ys - my
    cov = np.array([[(w * dx * dx).sum(), (w * dx * dy).sum()],
                    [(w * dx * dy).sum(), (w * dy * dy).sum()]]) / total
    evals, evecs = np.linalg.eigh(cov)
    major = evecs[:, 1]
    elongation = float(np.sqrt(evals[1] / max(evals[0], 1e-6)))

    cys, cxs = np.nonzero(core_mask)
    magnet = (float(cxs.mean()), float(cys.mean())) if len(cxs) else None
    if elongation < config["heading_min_elongation"] or len(cxs) == 0:
        return None, 0.0, magnet

    proj_body = dx * major[0] + dy * major[1]
    tip_end, evidence = 0, 0.0
    if config["use_tip_cue"]:
        tip_end, evidence = _tip_end(body, darkness, major, mx, my, proj_body, config)
    if tip_end == 0:
        proj_dark = (cxs - mx) * major[0] + (cys - my) * major[1]
        reach_pos = proj_dark.max() / max(proj_body.max(), 1.0)
        reach_neg = -proj_dark.min() / max(-proj_body.min(), 1.0)
        tip_end = -1 if reach_neg > reach_pos else 1
        evidence = float(np.clip(abs(reach_pos - reach_neg), 0.0, 1.0))
    if tip_end < 0:
        major = -major
    if not config["tip_is_front"]:
        major = -major
    angle = float(np.degrees(np.arctan2(major[1], major[0])) % 360)
    return angle, evidence, magnet


def _detect(frame, config, x0, y0, x1, y1, prev_center=None):
    prepared = _prepared_background(config)
    if prepared is None:
        return None
    bg_full, grad_full = prepared
    if bg_full.shape[:2] != frame.shape[:2]:
        return None
    window = frame[y0:y1, x0:x1]
    if window.size == 0:
        return None

    gray = cv2.GaussianBlur(cv2.cvtColor(window, cv2.COLOR_BGR2GRAY), (5, 5), 0).astype(np.float32)
    darkness = bg_full[y0:y1, x0:x1] - gray - _exposure_offset(frame, bg_full)
    absdiff = np.abs(darkness)
    grad = grad_full[y0:y1, x0:x1]

    strict = ((absdiff > config["diff_threshold"] + config["edge_gain"] * grad).astype(np.uint8)) * 255
    strict = _clean_mask(strict, config["close_kernel"])
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(strict)

    best = None
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < config["min_area"] or area > config["max_area"]:
            continue
        bx, by, bw, bh = stats[i, :4]
        sub = labels[by:by + bh, bx:bx + bw] == i
        core = float(np.percentile(darkness[by:by + bh, bx:bx + bw][sub], 99))
        if core < config["core_darkness_min"]:
            continue
        score = float(absdiff[by:by + bh, bx:bx + bw][sub].sum())
        if prev_center is not None:
            dist = np.hypot(centroids[i][0] + x0 - prev_center[0], centroids[i][1] + y0 - prev_center[1])
            score /= 1.0 + (dist / config["track_distance_scale_px"]) ** 2
        if best is None or score > best[0]:
            best = (score, i, core)
    if best is None:
        return None
    _, best_id, core = best
    seed = labels == best_id

    low = ((absdiff > config["diff_threshold_low"] + config["edge_gain"] * grad).astype(np.uint8)) * 255
    low = _clean_mask(low, config["close_kernel_low"])
    _, low_labels = cv2.connectedComponents(low)
    grown_ids = np.unique(low_labels[seed])
    grown_ids = grown_ids[grown_ids > 0]
    body = seed | np.isin(low_labels, grown_ids)
    if body.sum() > config["max_area"]:
        body = seed

    weights = np.where(body, absdiff, 0.0)
    total = weights.sum()
    ys, xs = np.nonzero(body)
    cx = float((weights[ys, xs] * xs).sum() / total)
    cy = float((weights[ys, xs] * ys).sum() / total)
    bx, by = int(xs.min()), int(ys.min())
    bw, bh = int(xs.max()) - bx + 1, int(ys.max()) - by + 1

    core_mask = seed & (darkness > 0.6 * core)
    angle, evidence, magnet = _heading(body, weights, darkness, core_mask, config)
    if angle is None and magnet is not None:
        cx, cy = magnet

    return {
        "cx": cx + x0, "cy": cy + y0,
        "bbox": (bx + x0, by + y0, bw, bh),
        "angle_deg": angle,
        "tip_evidence": evidence,
        "magnet": (magnet[0] + x0, magnet[1] + y0) if magnet else None,
        "touches_edge": bx <= 1 or by <= 1 or bx + bw >= (x1 - x0) - 1 or by + bh >= (y1 - y0) - 1,
    }


class MicrobotTracker:
    def __init__(self, camera="top"):
        self.camera = camera
        self.have_track = False
        self.cx = self.cy = 0.0
        self.bbox = None
        self.lost_frames = 0
        self.heading_deg = None
        self.frames_since_heading = 0
        self.flip_votes = 0

    def _search_window(self, config, expand_px=0):
        roi = config["roi"]
        x0, y0, x1, y1 = roi["x"], roi["y"], roi["x"] + roi["w"], roi["y"] + roi["h"]
        if self.have_track and self.lost_frames < config["max_lost_frames"] and self.bbox:
            bx, by, bw, bh = self.bbox
            margin = config["track_margin_px"] + expand_px
            return (max(x0, int(bx - margin)), max(y0, int(by - margin)),
                    min(x1, int(bx + bw + margin)), min(y1, int(by + bh + margin)))
        return x0, y0, x1, y1

    def update(self, frame):
        config = get_config(self.camera)
        tracked = self.have_track and self.lost_frames < config["max_lost_frames"] and self.bbox is not None
        prev_center = (self.cx, self.cy) if tracked else None
        found = _detect(frame, config, *self._search_window(config), prev_center)
        if found is not None and found["touches_edge"] and tracked:
            retry = _detect(frame, config, *self._search_window(config, config["edge_retry_expand_px"]), prev_center)
            if retry is not None:
                found = retry
        if found is None and tracked:
            found = _detect(frame, config, *self._search_window(config, 10 ** 6), prev_center)

        if found is None:
            self.lost_frames += 1
            if self.lost_frames >= config["max_lost_frames"]:
                self.have_track = False
                self.bbox = None
                self.heading_deg = None
            return None

        heading = found["angle_deg"]
        remembered = self.heading_deg is not None and self.frames_since_heading <= config["front_memory_frames"]
        if heading is not None and remembered:
            evidence = found["tip_evidence"]
            if abs((heading - self.heading_deg + 180) % 360 - 180) > 90:
                if evidence >= 0.99:
                    self.flip_votes += config["front_confident_votes"]
                elif evidence >= 0.5:
                    self.flip_votes += 2
                elif evidence >= 0.25:
                    self.flip_votes += 1
                if self.flip_votes < config["front_flip_frames"]:
                    heading = (heading + 180) % 360
                else:
                    self.flip_votes = 0
            elif evidence >= 0.25:
                self.flip_votes = 0
        if heading is not None:
            self.heading_deg = heading
            self.frames_since_heading = 0
        else:
            self.frames_since_heading += 1

        self.have_track = True
        self.cx, self.cy = found["cx"], found["cy"]
        self.bbox = found["bbox"]
        self.lost_frames = 0
        return {"cx": found["cx"], "cy": found["cy"], "angle_deg": heading, "bbox": found["bbox"],
                "magnet": found["magnet"]}


def draw_overlay(frame, result):
    overlay = frame.copy()
    if not result:
        cv2.putText(overlay, "NOT DETECTED", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        return overlay

    cx, cy = int(result["cx"]), int(result["cy"])
    bx, by, bw, bh = result["bbox"]
    cv2.rectangle(overlay, (bx, by), (bx + bw, by + bh), (0, 255, 0), 1)
    cv2.circle(overlay, (cx, cy), 4, (0, 255, 0), -1)
    if result["magnet"]:
        ax, ay = int(result["magnet"][0]), int(result["magnet"][1])
        cv2.circle(overlay, (ax, ay), 3, (0, 165, 255), -1)
    else:
        ax, ay = cx, cy

    if result["angle_deg"] is None:
        cv2.putText(overlay, "end-on", (bx, max(12, by - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        return overlay

    angle = np.deg2rad(result["angle_deg"])
    dx, dy = np.cos(angle), np.sin(angle)
    half = max(bw, bh) / 2
    cv2.line(overlay, (int(ax - dx * half), int(ay - dy * half)), (int(ax + dx * half), int(ay + dy * half)),
             (255, 255, 0), 1)
    fx, fy = int(ax + dx * half), int(ay + dy * half)
    cv2.arrowedLine(overlay, (ax, ay), (fx, fy), (0, 165, 255), 2, tipLength=0.3)
    cv2.putText(overlay, "front", (fx + 5, fy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
    return overlay
