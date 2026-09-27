"""RGB-D pixel grounding with explicit calibration and source validity."""
from __future__ import annotations
from dataclasses import dataclass
import math
import time
import numpy as np


class GroundingError(ValueError):
    pass


@dataclass(frozen=True)
class GroundedPoint:
    xyz_m: tuple[float, float, float]
    pixel_xy: tuple[float, float]
    calibration_id: str
    source_stamp_s: float
    epoch: int
    valid_until: float


def ground_pixel(pixel_xy, depth, camera_info, *, source_stamp_s, epoch,
                 max_age_s=.5, now=None):
    now = time.monotonic() if now is None else now
    if not math.isfinite(source_stamp_s) or not 0 <= now-source_stamp_s <= max_age_s:
        raise GroundingError("stale_or_invalid_observation")
    k = np.asarray(camera_info.get("intrinsics"), dtype=float)
    transform = np.asarray(camera_info.get("base_from_camera"), dtype=float)
    calibration_id = camera_info.get("calibration_id")
    if not calibration_id or k.shape != (3, 3) or transform.shape != (4, 4):
        raise GroundingError("no_calibration")
    if not np.isfinite(k).all() or not np.isfinite(transform).all() or min(k[0,0], k[1,1]) <= 0:
        raise GroundingError("invalid_calibration")
    r = transform[:3,:3]
    if not np.allclose(r.T@r, np.eye(3), atol=1e-4) or not np.isclose(np.linalg.det(r),1,atol=1e-4) or not np.allclose(transform[3], [0,0,0,1]):
        raise GroundingError("invalid_extrinsics")
    depth = np.asarray(depth)
    if depth.ndim != 2 or len(pixel_xy) != 2 or not np.isfinite(pixel_xy).all():
        raise GroundingError("invalid_pixel_or_depth")
    u, v = map(float, pixel_xy)
    if not 0 <= u < depth.shape[1] or not 0 <= v < depth.shape[0]:
        raise GroundingError("pixel_out_of_bounds")
    x,y = int(round(u)), int(round(v))
    x,y = min(x,depth.shape[1]-1),min(y,depth.shape[0]-1)
    patch = depth[max(0,y-1):y+2,max(0,x-1):x+2]
    values = patch[np.isfinite(patch) & (patch > .01) & (patch < 5)]
    if len(values) < max(1, patch.size//2) or np.ptp(values) > .08:
        raise GroundingError("missing_or_discontinuous_depth")
    z = float(np.median(values))
    camera_point = np.linalg.solve(k, np.array([u,v,1.])) * z
    xyz = transform @ np.r_[camera_point,1]
    return GroundedPoint(tuple(float(a) for a in xyz[:3]), (u,v), str(calibration_id),
                         source_stamp_s, int(epoch), source_stamp_s+max_age_s)


def require_workspace(point, lower=(-.05,-.45,-.02), upper=(.65,.45,.65)):
    if any(not lo <= v <= hi for v,lo,hi in zip(point.xyz_m,lower,upper)):
        raise GroundingError("outside_workspace")


def scene_unchanged(before, after, *, max_mean_difference=5.0):
    """Conservative image consistency check; not an object identity proof."""
    a,b = np.asarray(before), np.asarray(after)
    if a.shape != b.shape or a.ndim != 3:
        return False
    # Avoid treating a moved small object as unchanged due to whole-image averaging.
    difference = np.abs(a.astype(float)-b.astype(float)).mean(axis=2)
    return float(difference.mean()) <= max_mean_difference and float((difference>30).mean()) < .002
