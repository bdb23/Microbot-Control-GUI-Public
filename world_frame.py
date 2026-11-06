CALIBRATION = {
    "top": {
        "f_px": 2195.5751,
        "mm_per_px_offset": 0.024548,
        "camera_height_above_world_z_zero_mm": 147,
        "origin_px_x": 287,
        "origin_px_x_k": -45633.6026,
        "origin_px_y": 738,
        "origin_px_y_k": 41967.2361,
    },
    "side": {
        "f_px": 2177.1067,
        "mm_per_px_offset": 0.010016,
        "camera_standoff_from_world_x_zero_mm": 169,
        "origin_px_x": 716,
        "origin_px_x_k": -35194.7712,
        "origin_px_y": 802,
        "origin_px_y_k": 31774.8452,
    },
}

FUSION_PASSES = 8


def _reference_pixel(origin_px, k, depth_mm, depth_at_origin_mm):
    return origin_px + k * (1.0/depth_mm - 1.0/depth_at_origin_mm)


def _mm_per_px(depth_mm, cal_camera):
    return depth_mm / cal_camera["f_px"] + cal_camera.get("mm_per_px_offset", 0.0)


def _require_calibrated(cal):
    for camera in ("top", "side"):
        for key, value in cal[camera].items():
            if value is None:
                raise ValueError(
                    f"CALIBRATION[{camera!r}][{key!r}] is not filled in yet - "
                    f"see the calibration guide before calling fuse_position()"
                )


def _top_depth_mm(world_z_mm, cal):
    return cal["top"]["camera_height_above_world_z_zero_mm"] - world_z_mm


def _side_depth_mm(world_x_mm, cal):
    return cal["side"]["camera_standoff_from_world_x_zero_mm"] - world_x_mm


def world_to_pixel(world_xyz, camera, calibration=None):
    cal = calibration or CALIBRATION
    _require_calibrated(cal)
    x, y, z = world_xyz

    if camera == "top":
        depth = _top_depth_mm(z, cal)
        mm_per_px = _mm_per_px(depth, cal["top"])
        d0 = cal["top"]["camera_height_above_world_z_zero_mm"]
        ref_x = _reference_pixel(cal["top"]["origin_px_x"], cal["top"].get("origin_px_x_k", 0.0), depth, d0)
        ref_y = _reference_pixel(cal["top"]["origin_px_y"], cal["top"].get("origin_px_y_k", 0.0), depth, d0)
        px_x = ref_x + x / mm_per_px
        px_y = ref_y - y / mm_per_px
        return px_x, px_y

    if camera == "side":
        depth = _side_depth_mm(x, cal)
        mm_per_px = _mm_per_px(depth, cal["side"])
        d0 = cal["side"]["camera_standoff_from_world_x_zero_mm"]
        ref_x = _reference_pixel(cal["side"]["origin_px_x"], cal["side"].get("origin_px_x_k", 0.0), depth, d0)
        ref_y = _reference_pixel(cal["side"]["origin_px_y"], cal["side"].get("origin_px_y_k", 0.0), depth, d0)
        px_x = ref_x + y / mm_per_px
        px_y = ref_y - z / mm_per_px
        return px_x, px_y

    raise ValueError(f"Unknown camera {camera!r} - expected 'top' or 'side'")


def fuse_position(top_result, side_result, calibration=None, n_passes=FUSION_PASSES):
    cal = calibration or CALIBRATION
    _require_calibrated(cal)

    have_top = top_result is not None
    have_side = side_result is not None
    depth_corrected = have_top and have_side

    est_world_z = cal["top"]["camera_height_above_world_z_zero_mm"] / 2.0
    est_world_x = cal["side"]["camera_standoff_from_world_x_zero_mm"] / 2.0

    world_x = world_z = None
    y_from_top = y_from_side = None

    for _ in range(n_passes):
        if have_top:
            depth_top = _top_depth_mm(est_world_z, cal)
            mm_per_px_top = _mm_per_px(depth_top, cal["top"])
            d0_top = cal["top"]["camera_height_above_world_z_zero_mm"]
            ref_x_top = _reference_pixel(cal["top"]["origin_px_x"], cal["top"].get("origin_px_x_k", 0.0), depth_top, d0_top)
            ref_y_top = _reference_pixel(cal["top"]["origin_px_y"], cal["top"].get("origin_px_y_k", 0.0), depth_top, d0_top)
            world_x = (top_result["cx"] - ref_x_top) * mm_per_px_top
            y_from_top = (ref_y_top - top_result["cy"]) * mm_per_px_top
            est_world_x = world_x

        if have_side:
            depth_side = _side_depth_mm(est_world_x, cal)
            mm_per_px_side = _mm_per_px(depth_side, cal["side"])
            d0_side = cal["side"]["camera_standoff_from_world_x_zero_mm"]
            ref_x_side = _reference_pixel(cal["side"]["origin_px_x"], cal["side"].get("origin_px_x_k", 0.0), depth_side, d0_side)
            ref_y_side = _reference_pixel(cal["side"]["origin_px_y"], cal["side"].get("origin_px_y_k", 0.0), depth_side, d0_side)
            y_from_side = (side_result["cx"] - ref_x_side) * mm_per_px_side
            world_z = (ref_y_side - side_result["cy"]) * mm_per_px_side
            est_world_z = world_z

    if y_from_top is not None and y_from_side is not None:
        world_y = (y_from_top + y_from_side) / 2.0
        y_disagreement_mm = abs(y_from_top - y_from_side)
    elif y_from_top is not None:
        world_y = y_from_top
        y_disagreement_mm = None
    elif y_from_side is not None:
        world_y = y_from_side
        y_disagreement_mm = None
    else:
        world_y = None
        y_disagreement_mm = None

    return {
        "x": world_x, "y": world_y, "z": world_z,
        "y_from_top": y_from_top, "y_from_side": y_from_side,
        "y_disagreement_mm": y_disagreement_mm,
        "x_depth_corrected": depth_corrected if have_top else None,
        "z_depth_corrected": depth_corrected if have_side else None,
    }
