"""Stateless spatial evidence receipts and candidate visual overlays.

Shared by the front-end CLI/MCP adapters and online evaluation harnesses.
Never holds session state, live-ID aliases, transfer history, or task-scoring logic.
All projections and overlays explicitly label results as CANDIDATE ONLY and never
certify collision-free paths, grasp stability, or task completion.
"""
import io
import math
import numpy as np

from .scene_geometry import _validate_transform


OVERLAY_BANNER_LINE1 = "CANDIDATE OVERLAY ONLY - NOT collision-checked, NOT executed, NOT grasp/task success"


def normalize_intrinsics(intrinsics, *, width=640, height=480):
    """Normalize dict or 3x3 matrix intrinsics to {fx, fy, ppx, ppy, width, height}."""
    if isinstance(intrinsics, dict):
        fx = float(intrinsics["fx"])
        fy = float(intrinsics["fy"])
        ppx = float(intrinsics.get("ppx", intrinsics.get("cx")))
        ppy = float(intrinsics.get("ppy", intrinsics.get("cy")))
        w = int(intrinsics.get("width", width))
        h = int(intrinsics.get("height", height))
    else:
        k = np.asarray(intrinsics, dtype=np.float64)
        if k.shape != (3, 3):
            raise ValueError("intrinsics must be a dict or 3x3 matrix")
        fx = float(k[0, 0])
        fy = float(k[1, 1])
        ppx = float(k[0, 2])
        ppy = float(k[1, 2])
        w = int(width)
        h = int(height)
    if not all(math.isfinite(v) for v in (fx, fy, ppx, ppy)) or fx <= 0 or fy <= 0 or w <= 0 or h <= 0:
        raise ValueError("Invalid camera intrinsics")
    return {"fx": fx, "fy": fy, "ppx": ppx, "ppy": ppy, "width": w, "height": h}


def project_base_to_pixel(points_base_m, intrinsics, base_from_camera):
    """Project 3D base-frame points into 2D camera pixel coordinates (exact inverse of _deproject)."""
    intr = normalize_intrinsics(intrinsics)
    transform = _validate_transform(base_from_camera, "base_from_camera")
    arr = np.asarray(points_base_m, dtype=np.float64)
    single = arr.ndim == 1
    if single:
        arr = arr.reshape(1, 3)
    if arr.ndim != 2 or arr.shape[1] != 3 or not np.all(np.isfinite(arr)):
        raise ValueError("points_base_m must be finite 3D coordinates")
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    points_cam = (arr - translation) @ rotation
    results = []
    for x_c, y_c, z_c in points_cam:
        z_val = float(z_c)
        if z_val <= 1e-6:
            results.append({
                "pixel_xy": None,
                "pixel_int": None,
                "depth_cam_m": z_val,
                "in_front": False,
                "in_image": False,
            })
            continue
        u = float(intr["fx"] * x_c / z_val + intr["ppx"])
        v = float(intr["fy"] * y_c / z_val + intr["ppy"])
        u_int = int(round(u))
        v_int = int(round(v))
        in_image = bool(0 <= u_int < intr["width"] and 0 <= v_int < intr["height"])
        results.append({
            "pixel_xy": [u, v],
            "pixel_int": [u_int, v_int],
            "depth_cam_m": z_val,
            "in_front": True,
            "in_image": in_image,
        })
    return results[0] if single else results


def relative_tcp_delta(target_xyz_m, measured_tcp_pose, *, observation_id=None, sim_time_s=None, captured_at=None):
    """Compute base-frame vector from capture-aligned measured TCP to target_xyz_m."""
    if not isinstance(measured_tcp_pose, dict) or "xyz_m" not in measured_tcp_pose:
        raise ValueError("measured_tcp_pose must contain xyz_m")
    tcp_xyz = [float(v) for v in measured_tcp_pose["xyz_m"]]
    tgt_xyz = [float(v) for v in target_xyz_m]
    if len(tcp_xyz) != 3 or len(tgt_xyz) != 3 or not all(math.isfinite(v) for v in tcp_xyz + tgt_xyz):
        raise ValueError("TCP and target coordinates must be finite 3-vectors")
    delta = [float(tgt_xyz[i] - tcp_xyz[i]) for i in range(3)]
    dist = float(math.sqrt(sum(d * d for d in delta)))
    rpy = measured_tcp_pose.get("rpy_deg")
    return {
        "tcp_to_target_delta_m": delta,
        "tcp_to_target_distance_m": dist,
        "relative_reference": {
            "frame": "base",
            "units": "m",
            "definition": "target_xyz_m minus capture_aligned_measured_tcp_xyz_m",
            "observation_id": observation_id,
            "captured_sim_time_s": float(sim_time_s) if isinstance(sim_time_s, (int, float)) and math.isfinite(sim_time_s) else None,
            "captured_at": captured_at if isinstance(captured_at, str) and captured_at else None,
            "measured_tcp_xyz_m": tcp_xyz,
            "measured_tcp_rpy_deg": [float(v) for v in rpy] if isinstance(rpy, (list, tuple)) and len(rpy) == 3 else None,
            "scope": "Relative to stationary capture-aligned TCP at observation timestamp only; invalid after any subsequent motion.",
        },
    }


def _surface_spans(summary, height=None):
    lower = summary["observed_bounds_m"]["lower"]
    upper = summary["observed_bounds_m"]["upper"]
    extent = summary.get("observed_extent_m") or [float(upper[i] - lower[i]) for i in range(3)]
    minor_span = float(summary["observed_minor_width_m"]) if "observed_minor_width_m" in summary else None
    major_span = float(summary["observed_major_span_m"]) if "observed_major_span_m" in summary else None
    out = {
        "visible_minor_span_m": minor_span,
        "visible_major_span_m": major_span,
        "visible_xyz_extent_m": [float(v) for v in extent],
        "scope": "Observed visible surface spans only (2%..98% quantiles of foreground ROI points); NOT full 3D object bounding box or hidden back-surface thickness.",
    }
    if height is not None:
        out["table_to_visible_top_height_m"] = float(height)
    return out


def grasp_estimate(kin, seed, summary, xy, top, height, view, target_hash, *,
                   measured_tcp_pose=None, sim_time_s=None, captured_at=None):
    """Compute candidate overhead grasp poses and explicit spatial evidence receipt."""
    ambiguous = bool(summary["orientation_ambiguous"])
    minor_yaw = float(summary["minor_axis_yaw_deg"])
    major_yaw = float(summary.get("major_axis_yaw_deg", (minor_yaw + 180.0) % 180.0 - 90.0))
    yaw = 0.0 if ambiguous else minor_yaw
    width = summary["observed_minor_width_m"] + 0.010
    if width > 0.07:
        raise ValueError("Observed closing width plus margin exceeds gripper capability")
    if measured_tcp_pose is None and hasattr(kin, "pose"):
        try:
            measured_tcp_pose = kin.pose(seed)
        except Exception:
            measured_tcp_pose = None
    x, y = xy
    z = top - height / 2 - 0.0125
    poses = None
    selected_angle = yaw
    for angle in (yaw, yaw + 180, yaw - 180):
        try:
            candidate = {}
            for label, dz in [("approach", 0.10), ("grasp", 0.0), ("lift", 0.10)]:
                target_pt = [float(x), float(y), float(z + dz)]
                solved = kin.solve(target_pt, [180, 0, angle], seed)
                entry = {"xyz_m": target_pt, "rpy_deg": [180, 0, angle], **solved}
                if isinstance(measured_tcp_pose, dict) and "xyz_m" in measured_tcp_pose:
                    rel = relative_tcp_delta(
                        target_pt, measured_tcp_pose,
                        observation_id=view, sim_time_s=sim_time_s, captured_at=captured_at,
                    )
                    entry["tcp_to_target_delta_m"] = rel["tcp_to_target_delta_m"]
                    entry["tcp_to_target_distance_m"] = rel["tcp_to_target_distance_m"]
                candidate[label] = entry
            poses = candidate
            selected_angle = float(angle)
            break
        except Exception:
            poses = None
    if poses is None:
        raise ValueError("No legal grasp/approach IK; reobserve or select a different region")
    result = {
        "status": "ok",
        "observation_id": view,
        "source_measurement_id": "grasp-" + target_hash,
        "poses": poses,
        "opening_width_m": 0.07,
        "observed_height_m": height,
        "observed_width_m": summary["observed_minor_width_m"],
        "observed_bounds_m": summary["observed_bounds_m"],
        "observed_surface_spans_m": _surface_spans(summary, height=height),
        "closing_axis_yaw_deg": selected_angle,
        "object_minor_axis_yaw_deg": minor_yaw,
        "object_major_axis_yaw_deg": major_yaw,
        "orientation_ambiguous": ambiguous,
        "orientation_semantics": (
            "closing_axis_yaw_deg is the 2-finger pinch direction (TCP +X at rpy=[180,0,yaw]), "
            "aligned with the visible PCA minor axis (or 0 deg when orientation_ambiguous=True); "
            "object_major_axis_yaw_deg is the perpendicular visible elongation axis."
        ),
        "tcp_above_table_m": height / 2 - 0.0125,
        "geometry_source": "calibrated current RGB-D; ROI above measured support plane",
        "assumption": "Selected upright object is resting on this tabletop; occluded geometry is not measured",
        "robot_moved": False,
        "collision_checked": False,
    }
    if isinstance(measured_tcp_pose, dict) and "xyz_m" in measured_tcp_pose:
        rel = relative_tcp_delta(
            poses["grasp"]["xyz_m"], measured_tcp_pose,
            observation_id=view, sim_time_s=sim_time_s, captured_at=captured_at,
        )
        result["tcp_to_target_delta_m"] = rel["tcp_to_target_delta_m"]
        result["tcp_to_target_distance_m"] = rel["tcp_to_target_distance_m"]
        result["relative_reference"] = rel["relative_reference"]
    return result


def placement_geometry(source, summary, plane, camera, relation, yaw_deg, clearance=None, gripper_width_m=None, *,
                       measured_tcp_pose=None, observation_id=None, sim_time_s=None, captured_at=None):
    """Observed envelopes and calibrated image direction; no simulator oracle."""
    if relation not in ("left", "right", "front", "behind", "top"):
        raise ValueError("Unknown image relation")
    if clearance is not None and (not math.isfinite(clearance) or not 0 <= clearance <= 0.10):
        raise ValueError("Invalid placement clearance")
    lower = np.asarray(summary["observed_bounds_m"]["lower"])
    upper = np.asarray(summary["observed_bounds_m"]["upper"])
    centre = (lower + upper) / 2
    normal = np.asarray(plane["normal"])
    origin = np.asarray(plane["point_on_plane_m"])
    if normal[2] < 0.9:
        raise ValueError("Placement requires an upward tabletop plane")
    floor = origin[2] - (normal[0] * (centre[0] - origin[0]) + normal[1] * (centre[1] - origin[1])) / normal[2]
    if not 0.01 < upper[2] - floor < 0.25:
        raise ValueError("Reference region contains unsupported or occluding geometry; reobserve")
    gap = 0.02 if clearance is None else clearance
    bounds = source["observed_bounds_m"]
    sl = np.asarray(bounds["lower"])
    su = np.asarray(bounds["upper"])
    angle = math.radians(yaw_deg - source["poses"]["grasp"]["rpy_deg"][2])
    rotation = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
    offset = rotation @ ((sl + su)[:2] / 2 - np.asarray(source["poses"]["grasp"]["xyz_m"][:2]))
    xy = centre[:2].copy()
    direction = None
    sr = None
    rr = None
    fr = None
    if relation != "top":
        transform = np.asarray(camera["base_from_camera"])
        axis = 0 if relation in ("left", "right") else 1
        sign = -1 if relation in ("left", "behind") else 1
        ray = centre - transform[:3, 3]
        delta = transform[:3, axis] * sign
        denominator = float(normal @ ray)
        if abs(denominator) < 1e-6:
            raise ValueError("Camera ray is parallel to the measured support plane")
        tangent = delta - ray * float(normal @ delta) / denominator
        if np.linalg.norm(tangent[:2]) < 1e-6:
            raise ValueError("Image relation has no resolved tabletop direction")
        direction = tangent[:2] / np.linalg.norm(tangent[:2])
        half = (su - sl)[:2] / 2
        sr = float(np.abs(direction @ rotation) @ half)
        rr = float(np.abs(direction) @ ((upper - lower)[:2] / 2))
        envelope = sr
        if gripper_width_m is not None:
            if not math.isfinite(gripper_width_m) or not 0 <= gripper_width_m <= 0.0701:
                raise ValueError("Invalid measured gripper width")
            theta = math.radians(yaw_deg)
            tool_rotation = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
            fr = float(np.abs(direction @ tool_rotation) @ np.array([gripper_width_m / 2 + 0.026, 0.026]))
            envelope = max(sr, fr + float(offset @ direction))
        xy += direction * (envelope + rr + gap)
    xy -= offset
    floor = origin[2] - (normal[0] * (xy[0] - origin[0]) + normal[1] * (xy[1] - origin[1])) / normal[2]
    support = float(upper[2]) if relation == "top" else float(floor)
    z = support + source["tcp_above_table_m"] + (gap if relation == "top" else 0.0)
    target_xyz = [float(xy[0]), float(xy[1]), float(z)]
    result = {
        "xyz_m": target_xyz,
        "relation": relation,
        "clearance_m": gap,
        "image_direction_xy_base": None if direction is None else direction.tolist(),
        "source_half_extent_along_direction_m": sr,
        "reference_half_extent_along_direction_m": rr,
        "finger_half_extent_along_direction_m": fr,
        "assumed_source_offset_xy_from_tcp_m": offset.tolist(),
        "assumed_source_yaw_change_deg": math.degrees(angle),
        "target_semantics": "base-frame TCP",
        "support_height_m": support,
        "reference_observed_surface_spans_m": _surface_spans(summary),
        "scope": "Observed envelopes, assumed upright held source and its recorded TCP offset. Slip, hidden geometry and collision clearance are not certified.",
    }
    if isinstance(measured_tcp_pose, dict) and "xyz_m" in measured_tcp_pose:
        rel = relative_tcp_delta(
            target_xyz, measured_tcp_pose,
            observation_id=observation_id, sim_time_s=sim_time_s, captured_at=captured_at,
        )
        result["tcp_to_target_delta_m"] = rel["tcp_to_target_delta_m"]
        result["tcp_to_target_distance_m"] = rel["tcp_to_target_distance_m"]
        result["relative_reference"] = rel["relative_reference"]
    return result


def preview_pose_receipt(kin, seed, xyz_m, rpy_deg=None, *,
                         measured_tcp_pose=None, observation_id=None, sim_time_s=None, captured_at=None):
    """Compute endpoint IK preview and capture-aligned relative TCP delta."""
    pose_now = measured_tcp_pose if isinstance(measured_tcp_pose, dict) and "xyz_m" in measured_tcp_pose else kin.pose(seed)
    target_rpy = rpy_deg if rpy_deg is not None else pose_now["rpy_deg"]
    rel = relative_tcp_delta(
        xyz_m, pose_now,
        observation_id=observation_id, sim_time_s=sim_time_s, captured_at=captured_at,
    )
    try:
        solved = kin.solve(xyz_m, target_rpy, seed)
        return {
            "status": "ok",
            **solved,
            "observation_id": observation_id,
            "tcp_to_target_delta_m": rel["tcp_to_target_delta_m"],
            "tcp_to_target_distance_m": rel["tcp_to_target_distance_m"],
            "relative_reference": rel["relative_reference"],
            "collision_checked": False,
            "robot_moved": False,
        }
    except Exception as exc:
        return {
            "status": "unavailable",
            "error": {"code": "ik_no_solution", "message": str(exc)[:350]},
            "observation_id": observation_id,
            "tcp_to_target_delta_m": rel["tcp_to_target_delta_m"],
            "tcp_to_target_distance_m": rel["tcp_to_target_distance_m"],
            "relative_reference": rel["relative_reference"],
            "collision_checked": False,
            "robot_moved": False,
        }


def _bounds_corners(bounds):
    lower = bounds["lower"]
    upper = bounds["upper"]
    return [
        [lower[0], lower[1], lower[2]],
        [upper[0], lower[1], lower[2]],
        [upper[0], upper[1], lower[2]],
        [lower[0], upper[1], lower[2]],
        [lower[0], lower[1], upper[2]],
        [upper[0], lower[1], upper[2]],
        [upper[0], upper[1], upper[2]],
        [lower[0], upper[1], upper[2]],
    ]


def render_candidate_overlay(
    rgb_bytes,
    *,
    intrinsics,
    base_from_camera,
    calibration_id,
    observation_id,
    sim_time_s=None,
    kind="grasp_candidate",
    target_xyz_m=None,
    approach_xyz_m=None,
    closing_yaw_deg=None,
    major_yaw_deg=None,
    opening_width_m=0.07,
    measured_tcp_xyz_m=None,
    surface_points_base_m=None,
    median_point_base_m=None,
    observed_bounds_m=None,
    object_roi=None,
    table_roi=None,
    object_polygon=None,
    table_polygon=None,
):
    """Render a candidate-only visual overlay onto a copy of rgb_bytes.

    Never modifies the original image bytes or files.
    Returns (jpeg_bytes, metadata_dict).
    """
    from PIL import Image, ImageDraw

    image = Image.open(io.BytesIO(rgb_bytes)).convert("RGB")
    intr = normalize_intrinsics(intrinsics, width=image.width, height=image.height)
    transform = _validate_transform(base_from_camera, "base_from_camera")
    draw = ImageDraw.Draw(image, "RGBA")

    projected_meta = {
        "target_pixel_xy": None,
        "approach_pixel_xy": None,
        "measured_tcp_pixel_xy": None,
        "median_surface_pixel_xy": None,
        "closing_finger_pixels_xy": None,
        "major_axis_pixels_xy": None,
        "visible_surface_point_count": 0,
    }

    if isinstance(table_roi, (list, tuple)) and len(table_roi) == 4:
        x1, y1, x2, y2 = [int(v) for v in table_roi]
        draw.rectangle([x1, y1, x2, y2], outline=(0, 188, 212, 200), width=2)
        draw.text((x1 + 3, max(28, y1 + 2)), "table_roi", fill=(0, 229, 255, 255))

    if isinstance(object_roi, (list, tuple)) and len(object_roi) == 4:
        x1, y1, x2, y2 = [int(v) for v in object_roi]
        draw.rectangle([x1, y1, x2, y2], outline=(255, 193, 7, 230), width=2)
        draw.text((x1 + 3, max(28, y1 - 12)), "target_roi", fill=(255, 213, 79, 255))

    for label, vertices, color in (
        ('target_polygon', object_polygon, (255, 193, 7, 230)),
        ('table_polygon', table_polygon, (0, 188, 212, 200)),
    ):
        if vertices is not None:
            points = [tuple(p) for p in vertices]
            draw.line(points + [points[0]], fill=color, width=2)
            draw.text((points[0][0]+3, max(28, points[0][1]-12)), label, fill=color)

    if surface_points_base_m is not None:
        pts = np.asarray(surface_points_base_m, dtype=np.float64)
        if pts.ndim == 2 and pts.shape[1] == 3 and len(pts) > 0:
            step = max(1, len(pts) // 250)
            sampled = pts[::step]
            proj_pts = project_base_to_pixel(sampled, intr, transform)
            valid_count = 0
            for item in proj_pts:
                if item["in_front"] and item["in_image"]:
                    u, v = item["pixel_int"]
                    draw.ellipse([u - 1, v - 1, u + 1, v + 1], fill=(0, 230, 118, 190))
                    valid_count += 1
            projected_meta["visible_surface_point_count"] = valid_count
    elif isinstance(observed_bounds_m, dict) and "lower" in observed_bounds_m and "upper" in observed_bounds_m:
        corners = _bounds_corners(observed_bounds_m)
        proj_corners = project_base_to_pixel(corners, intr, transform)
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7),
        ]
        for a, b in edges:
            if proj_corners[a]["in_front"] and proj_corners[b]["in_front"]:
                p1 = proj_corners[a]["pixel_int"]
                p2 = proj_corners[b]["pixel_int"]
                draw.line([p1[0], p1[1], p2[0], p2[1]], fill=(0, 230, 118, 170), width=1)
        projected_meta["visible_surface_point_count"] = sum(1 for c in proj_corners if c["in_front"])

    if median_point_base_m is not None:
        med_proj = project_base_to_pixel(median_point_base_m, intr, transform)
        if med_proj["in_front"]:
            projected_meta["median_surface_pixel_xy"] = med_proj["pixel_xy"]
            u, v = med_proj["pixel_int"]
            draw.ellipse([u - 3, v - 3, u + 3, v + 3], fill=(0, 230, 118, 255), outline=(0, 77, 64, 255))

    tcp_px = None
    if measured_tcp_xyz_m is not None:
        tcp_proj = project_base_to_pixel(measured_tcp_xyz_m, intr, transform)
        if tcp_proj["in_front"]:
            projected_meta["measured_tcp_pixel_xy"] = tcp_proj["pixel_xy"]
            tcp_px = tcp_proj["pixel_int"]
            u, v = tcp_px
            draw.rectangle([u - 5, v - 5, u + 5, v + 5], outline=(41, 121, 255, 255), fill=(41, 121, 255, 160), width=2)
            draw.text((u + 7, v - 6), "TCP(t)", fill=(130, 177, 255, 255))

    if approach_xyz_m is not None:
        app_proj = project_base_to_pixel(approach_xyz_m, intr, transform)
        if app_proj["in_front"]:
            projected_meta["approach_pixel_xy"] = app_proj["pixel_xy"]
            u_a, v_a = app_proj["pixel_int"]
            draw.ellipse([u_a - 4, v_a - 4, u_a + 4, v_a + 4], outline=(255, 235, 59, 230), width=2)

    if target_xyz_m is not None:
        tgt_proj = project_base_to_pixel(target_xyz_m, intr, transform)
        if tgt_proj["in_front"]:
            projected_meta["target_pixel_xy"] = tgt_proj["pixel_xy"]
            u_t, v_t = tgt_proj["pixel_int"]
            if tcp_px is not None:
                draw.line([tcp_px[0], tcp_px[1], u_t, v_t], fill=(68, 138, 255, 210), width=2)
            if major_yaw_deg is not None and math.isfinite(major_yaw_deg):
                rad_maj = math.radians(float(major_yaw_deg))
                half_maj = 0.03
                p_m1 = [target_xyz_m[0] - half_maj * math.cos(rad_maj), target_xyz_m[1] - half_maj * math.sin(rad_maj), target_xyz_m[2]]
                p_m2 = [target_xyz_m[0] + half_maj * math.cos(rad_maj), target_xyz_m[1] + half_maj * math.sin(rad_maj), target_xyz_m[2]]
                pm1, pm2 = project_base_to_pixel([p_m1, p_m2], intr, transform)
                if pm1["in_front"] and pm2["in_front"]:
                    projected_meta["major_axis_pixels_xy"] = [pm1["pixel_xy"], pm2["pixel_xy"]]
                    draw.line([pm1["pixel_int"][0], pm1["pixel_int"][1], pm2["pixel_int"][0], pm2["pixel_int"][1]],
                              fill=(0, 229, 255, 210), width=2)
            if closing_yaw_deg is not None and math.isfinite(closing_yaw_deg):
                rad_close = math.radians(float(closing_yaw_deg))
                half_open = max(0.015, float(opening_width_m or 0.07) / 2.0)
                p_f1 = [target_xyz_m[0] - half_open * math.cos(rad_close), target_xyz_m[1] - half_open * math.sin(rad_close), target_xyz_m[2]]
                p_f2 = [target_xyz_m[0] + half_open * math.cos(rad_close), target_xyz_m[1] + half_open * math.sin(rad_close), target_xyz_m[2]]
                pf1, pf2 = project_base_to_pixel([p_f1, p_f2], intr, transform)
                if pf1["in_front"] and pf2["in_front"]:
                    projected_meta["closing_finger_pixels_xy"] = [pf1["pixel_xy"], pf2["pixel_xy"]]
                    u1, v1 = pf1["pixel_int"]
                    u2, v2 = pf2["pixel_int"]
                    draw.line([u1, v1, u2, v2], fill=(255, 64, 129, 240), width=3)
                    draw.ellipse([u1 - 3, v1 - 3, u1 + 3, v1 + 3], fill=(255, 64, 129, 255))
                    draw.ellipse([u2 - 3, v2 - 3, u2 + 3, v2 + 3], fill=(255, 64, 129, 255))
            draw.line([u_t - 7, v_t, u_t + 7, v_t], fill=(255, 23, 68, 255), width=2)
            draw.line([u_t, v_t - 7, u_t, v_t + 7], fill=(255, 23, 68, 255), width=2)
            draw.ellipse([u_t - 4, v_t - 4, u_t + 4, v_t + 4], outline=(255, 235, 59, 255), width=2)
            draw.text((u_t + 8, v_t + 4), f"CANDIDATE ({kind})", fill=(255, 235, 59, 255))

    cal_short = str(calibration_id)[:12] if calibration_id else "unknown"
    sim_label = f"{float(sim_time_s):.3f}s" if isinstance(sim_time_s, (int, float)) and math.isfinite(sim_time_s) else "n/a"
    line2 = f"obs={observation_id} | cal={cal_short} | sim_t={sim_label} | kind={kind} | raw RGB preserved"
    draw.rectangle([0, 0, image.width, 28], fill=(18, 18, 18, 215))
    draw.text((6, 2), OVERLAY_BANNER_LINE1, fill=(255, 213, 79, 255))
    draw.text((6, 14), line2, fill=(224, 224, 224, 255))

    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=90)
    jpeg_bytes = buf.getvalue()
    meta = {
        "status": "ok",
        "kind": kind,
        "observation_id": observation_id,
        "calibration_id": calibration_id,
        "captured_sim_time_s": float(sim_time_s) if isinstance(sim_time_s, (int, float)) and math.isfinite(sim_time_s) else None,
        "raw_image_preserved": True,
        "banner": OVERLAY_BANNER_LINE1,
        "certification": "candidate_only_not_collision_or_grasp_certified",
        "projected_pixels": projected_meta,
        "selected_regions": {
            "target": {'polygon':object_polygon} if object_polygon is not None else {'roi':object_roi},
            "table": {'polygon':table_polygon} if table_polygon is not None else {'roi':table_roi},
        },
    }
    return jpeg_bytes, meta
