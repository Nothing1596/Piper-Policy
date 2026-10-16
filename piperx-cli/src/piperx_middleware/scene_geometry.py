"""Local CPU scene-geometry worker.

``compute(root, request) -> dict`` reads persisted observation JSON files,
produces geometry evidence, stores it under the task directory and answers
queries from stored result ids. It never imports the robot SDK, opens CAN or
touches the network. All lengths are metres; ``base_from_camera`` is a 4x4
row-major rigid transform mapping camera-frame points into the base frame. No
grasp, insertion, collision-free or free-space claim is ever made.

Errors
------
* invalid request / unsafe path / missing or conflicting calibration ->
  ``ValueError``
* unsupported, degenerate or data-less request -> ``{"status": "unavailable",
  "operation": ..., "task_id": ..., "reason": ...}``

Common fields
-------------
``operation``  one of ``surface_point``, ``accumulate_region``, ``fit_plane``,
               ``point_distance``, ``point_plane_distance``.
``task_id``    safe id (``[A-Za-z0-9][A-Za-z0-9._-]{0,63}``);
               ``<root>/tasks/<task_id>/plan.json`` must already exist.

Observations
------------
``observation_path`` must resolve under ``<root>/observations`` or
``<root>/tasks`` (traversal and symlinks are rejected); ``depth_path`` resolves
the same way. The document holds ``camera_serial``, ``captured_at``, ``frame``,
``depth_path``, ``depth_scale_m``, ``intrinsics`` (``width``, ``height``,
``fx``, ``fy``, ``ppx``, ``ppy``, optional ``coeffs``) and an optional
``calibration_id``. ``depth.npy`` is loaded with ``allow_pickle=False``; only
zero distortion is supported. ``base_from_camera`` + ``calibration_id`` are
injected only by trusted integration: with ``base_from_camera`` the
``calibration_id`` is required, must match any ``calibration_id`` recorded in
the observation, and one camera serial must not appear under two calibrations
in one request. Observation identity is the resolved observation path: equal
depth bytes from another path are a different view, never a duplicate.

surface_point
-------------
``observation_path`` + ``pixel`` [u, v] + optional ``patch_half_size``
(default 2). Every pixel of the (2h+1)^2 patch must be valid (0 < z <= 10 m)
and locally consistent, else the request is ``unavailable``. The stored point
is the component-wise median of the deprojected patch and carries ``frame``,
``point_camera_m``/``point_m``, ``source`` and ``calibration_provenance``.

accumulate_region
-----------------
``target_id`` + ``batch_id`` + ``observations`` (each
``{observation_path, roi | polygon, base_from_camera, calibration_id}``), with
optional ``voxel_size_m`` (0.002) and ``stride`` (1). Identical entries are
deduplicated; one observation path with two transforms/calibrations, or one
camera serial with two calibrations, is rejected. At most 100000 voxel
centroids are written as
``tasks/<task_id>/geometry/targets/<target_id>/<batch_id>.npz`` plus ``.json``
metadata. The id pair is immutable: the same recipe is idempotent, a different
recipe or a torn/tampered pair is rejected. Metadata records the NPZ sha256,
the request digest and each source's serial and calibration; batch loads verify
the digest before use.

fit_plane
---------
Either one fresh region (fields as an accumulate entry) or ``target_id`` +
``batch_id``. Deterministic seeded RANSAC followed by SVD; collinear, too small
in extent, too thick or under-supported support is ``unavailable``. The normal
follows ``reference_normal`` when supplied, else the largest component is made
positive. A batch-derived plane keeps the batch's per-serial calibration
provenance and source observations; ``calibration_id`` is set only when every
source declares the same one.

point_distance / point_plane_distance
-------------------------------------
Stored ``point_id`` / ``plane_id`` ids only; raw coordinates are rejected.
Pairs must share a frame, and camera-frame pairs must come from one camera
serial. Base-frame pairs from different cameras keep their own calibrations and
are permitted; differing calibrations for one declared serial are rejected.
Results echo ``calibration_provenance`` (role, result id, serial, calibration)
so a trusted parent can revalidate them against current operator
registrations. ``point_plane_distance`` reports the signed distance, the side
and ``clearance_verified: False``.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:  # optional camera extra; only used to rasterise polygons
    import cv2
except Exception:  # pragma: no cover - exercised when the camera extra is absent
    cv2 = None

OPERATIONS = ('surface_point', 'accumulate_region', 'fit_plane',
              'point_distance', 'point_plane_distance')
SAFE_ID = re.compile(r'^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$')
RAW_XYZ_KEYS = ('xyz', 'xyz_m', 'point', 'point_m', 'coordinates', 'point_xyz',
                'xyz_camera_m', 'xyz_base_m')
MAX_DEPTH_M = 10.0
MAX_POINTS = 100000
DEFAULT_VOXEL_M = 0.002
DEFAULT_DISCONTINUITY_M = 0.05
MIN_NEIGHBOURS = 8
SIGNED_DISTANCE_CONVENTION = (
    'signed_distance(p) = dot(normal, p - point_on_plane_m); '
    'positive on the side the stored normal points toward')
SAFETY_NOTE = ('Geometry evidence only: no grasp, insertion or collision-free claim is made '
               'and no unobserved free space is implied.')
DEFAULT_RANSAC = {'seed': 0, 'max_iterations': 500, 'distance_threshold_m': 0.005,
                  'min_inliers': 50, 'min_inlier_fraction': 0.3, 'min_extent_m': 0.02,
                  'min_width_m': 0.005, 'max_thickness_m': 0.05}


class _Unavailable(Exception):
    """Supported operation that cannot honestly produce a result."""


@dataclass(frozen=True)
class _Obs:
    path: Path
    rel: str
    depth_digest: str
    depth_m: np.ndarray
    intr: dict
    scale: float
    frame: str
    captured_at: str
    serial: str
    recorded_calibration_id: object
    provenance: str = ''


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def _number(value, field):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f'{field} must be a finite number')
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f'{field} must be a finite number')
    return value


def _safe_id(value, field):
    if not isinstance(value, str) or not SAFE_ID.match(value):
        raise ValueError(f'{field} must be a safe id ([A-Za-z0-9][A-Za-z0-9._-]{{0,63}})')
    return value


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     default=_json_default).encode('utf-8')).hexdigest()


def _json_default(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f'not JSON serialisable: {type(value).__name__}')


def _atomic_write_bytes(path, data):
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError('Geometry output must not follow symlinks')
    path.parent.mkdir(parents=True, exist_ok=True)
    handle_fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + '.', suffix='.tmp')
    try:
        with os.fdopen(handle_fd, 'wb') as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return path


def _atomic_write_json(path, payload):
    return _atomic_write_bytes(
        path, (json.dumps(payload, indent=2, sort_keys=True, default=_json_default)
               + '\n').encode('utf-8'))


def _npz_bytes(arrays):
    """Serialise arrays with ``np.savez`` so they can be digested and committed atomically."""
    buffer = io.BytesIO()
    np.savez(buffer, **arrays)
    return buffer.getvalue()


def _resolve_under(root, raw, *, field, suffixes=()):
    """Resolve a request path inside <root>/observations or <root>/tasks."""
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f'{field} must be a non-empty path string')
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = root / candidate
    try:
        resolved = candidate.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ValueError(f'{field} does not exist: {raw}') from exc
    allowed = [root / 'observations', root / 'tasks']
    if not any(resolved == base or base in resolved.parents for base in allowed):
        raise ValueError(f'{field} must stay under root/observations or root/tasks: {raw}')
    if not resolved.is_file():
        raise ValueError(f'{field} is not a regular file: {raw}')
    lexical = Path(os.path.abspath(str(candidate)))
    for base, path in ((root, lexical), (root, resolved)):
        try:
            parts = path.relative_to(root).parts
        except ValueError:
            continue
        cursor = root
        for part in parts:
            cursor = cursor / part
            if cursor.is_symlink():
                raise ValueError(f'{field} must not be a symlink: {raw}')
    if suffixes and resolved.suffix.lower() not in suffixes:
        raise ValueError(f'{field} must end with {" or ".join(suffixes)}: {raw}')
    return resolved


def _task(root, request):
    task_id = _safe_id(request.get('task_id'), 'task_id')
    task = root / 'tasks' / task_id
    plan = task / 'plan.json'
    if not plan.is_file() or plan.is_symlink():
        raise ValueError(f'tasks/{task_id}/plan.json must already exist before geometry work')
    _resolve_under(root, str(plan), field='task_id', suffixes=('.json',))
    try:
        plan_doc = json.loads(plan.read_text(encoding='utf-8'))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f'tasks/{task_id}/plan.json is not valid JSON') from exc
    if not isinstance(plan_doc, dict):
        raise ValueError(f'tasks/{task_id}/plan.json must contain a JSON object')
    return task_id, task


def _validate_transform(value, field):
    try:
        transform = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{field} must be a 4x4 numeric matrix') from exc
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError(f'{field} must be a finite 4x4 matrix')
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6):
        raise ValueError(f'{field} last row must be [0, 0, 0, 1]')
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4):
        raise ValueError(f'{field} rotation must be orthonormal')
    if abs(float(np.linalg.det(rotation)) - 1.0) > 1e-4:
        raise ValueError(f'{field} rotation determinant must be +1')
    return transform


def _transform_points(transform, points):
    homogeneous = np.concatenate([points, np.ones((len(points), 1))], axis=1)
    return (homogeneous @ transform.T)[:, :3]


def _transform_for(entry, obs, field):
    transform = entry.get('base_from_camera')
    calibration_id = entry.get('calibration_id')
    if calibration_id is not None and (not isinstance(calibration_id, str) or not calibration_id.strip()):
        raise ValueError(f'{field}.calibration_id must be a non-empty string when supplied')
    if transform is None:
        return None, calibration_id
    transform = _validate_transform(transform, f'{field}.base_from_camera')
    if calibration_id is None:
        raise ValueError(f'{field}.calibration_id is required with base_from_camera; '
                         'base-frame geometry is only computed from trusted injected calibration')
    recorded = obs.recorded_calibration_id
    if recorded is not None and recorded != calibration_id:
        raise ValueError(f'{field}.calibration_id {calibration_id!r} conflicts with the '
                         f'calibration_id {recorded!r} recorded in the observation')
    return transform, calibration_id


def _load_observation(root, raw, *, field='observation_path'):
    path = _resolve_under(root, raw, field=field, suffixes=('.json',))
    if path.stat().st_size > 2*1024*1024:
        raise ValueError('Observation metadata exceeds 2 MiB')
    try:
        doc = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f'{field} is not a readable JSON observation: {raw}') from exc
    if not isinstance(doc, dict):
        raise ValueError(f'{field} must contain a JSON object')
    intr = doc.get('intrinsics')
    if not isinstance(intr, dict):
        raise ValueError(f'{field}: intrinsics object is required')
    for key in ('fx', 'fy', 'ppx', 'ppy'):
        _number(intr.get(key), f'{field}.intrinsics.{key}')
    for key in ('width', 'height'):
        value = intr.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f'{field}.intrinsics.{key} must be a positive integer')
    coeffs = intr.get('coeffs', [0.0] * 5)
    if not isinstance(coeffs, (list, tuple)) or not coeffs:
        raise ValueError(f'{field}.intrinsics.coeffs must be a non-empty list of numbers')
    for index, coeff in enumerate(coeffs):
        _number(coeff, f'{field}.intrinsics.coeffs[{index}]')
    if any(abs(float(coeff)) > 1e-12 for coeff in coeffs):
        raise _Unavailable('non-zero distortion coefficients are not supported in this version; '
                           'supply rectified depth or zero coeffs together with an explicit reason')
    scale = _number(doc.get('depth_scale_m'), f'{field}.depth_scale_m')
    if scale <= 0:
        raise ValueError(f'{field}.depth_scale_m must be positive')
    depth_ref = doc.get('depth_path')
    if not isinstance(depth_ref, str) or not depth_ref.strip():
        raise ValueError(f'{field}: observation has no depth_path')
    depth_ref = Path(depth_ref)
    if not depth_ref.is_absolute():
        depth_ref = path.parent / depth_ref
    depth_path = _resolve_under(root, str(depth_ref), field=f'{field}.depth_path',
                                suffixes=('.npy',))
    if depth_path.stat().st_size > 32*1024*1024 or intr['width']*intr['height'] > 2073600:
        raise _Unavailable('Depth exceeds 32 MiB or 1920x1080 pixel budget')
    try:
        raw_bytes = depth_path.read_bytes()
    except OSError as exc:
        raise ValueError(f'{field}.depth_path is not readable: {depth_ref}') from exc
    try:
        depth = np.load(io.BytesIO(raw_bytes), allow_pickle=False)
    except Exception as exc:
        raise _Unavailable(f'depth .npy could not be read with allow_pickle=False: {exc}') from exc
    if depth.ndim != 2:
        raise _Unavailable(f'depth array must be 2-D; got shape {depth.shape}')
    if depth.dtype.kind not in 'uif':
        raise _Unavailable(f'depth dtype {depth.dtype} is not numeric')
    expected = (int(intr['height']), int(intr['width']))
    if depth.shape != expected:
        raise _Unavailable(f'depth shape {depth.shape} does not match intrinsics height x width {expected}')
    frame = doc.get('frame') or 'camera_color_optical'
    if not isinstance(frame, str):
        raise ValueError(f'{field}.frame must be a string')
    captured_at = doc.get('captured_at')
    if captured_at is not None and not isinstance(captured_at, str):
        raise ValueError(f'{field}.captured_at must be a string')
    serial = doc.get('camera_serial')
    if serial is not None and not isinstance(serial, str):
        raise ValueError(f'{field}.camera_serial must be a string')
    return _Obs(path=path,
                rel=str(path.relative_to(root)),
                depth_digest=hashlib.sha256(raw_bytes).hexdigest(),
                depth_m=depth.astype(np.float64) * scale,
                intr={key: float(intr[key]) for key in ('fx', 'fy', 'ppx', 'ppy')}
                     | {'width': int(intr['width']), 'height': int(intr['height'])},
                scale=scale,
                frame=frame,
                captured_at=captured_at or '',
                serial=serial or '',
                recorded_calibration_id=doc.get('calibration_id'),
                provenance=doc.get('provenance', ''))


def _source(obs):
    return {'observation': obs.rel, 'captured_at': obs.captured_at,
            'camera_serial': obs.serial, 'frame': obs.frame,
            'depth_digest': obs.depth_digest, 'depth_scale_m': obs.scale,
            'provenance': obs.provenance, 'simulated': obs.provenance == 'mujoco_sensor'}


def _provenance_entry(serial, calibration_id, source=None):
    """One camera-serial -> calibration-id mapping carried by stored geometry."""
    entry = {'camera_serial': serial if isinstance(serial, str) else '',
             'calibration_id': calibration_id if isinstance(calibration_id, str) else None}
    if isinstance(source, dict):
        for key in ('observation', 'depth_digest'):
            if source.get(key) is not None:
                entry[key] = source[key]
    return entry


def _clean_calibration(value):
    return value if isinstance(value, str) and value.strip() else None


def _provenance(doc):
    """Per-source serial/calibration mapping declared by a stored result.

    New results carry ``calibration_provenance``; results stored before that
    field existed fall back to the single top-level calibration and source.
    """
    raw = doc.get('calibration_provenance')
    entries = []
    if isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict):
                continue
            entries.append(_provenance_entry(item.get('camera_serial'),
                                             _clean_calibration(item.get('calibration_id')),
                                             item))
    if entries:
        return entries
    source = doc.get('source') if isinstance(doc.get('source'), dict) else {}
    return [_provenance_entry(source.get('camera_serial'),
                              _clean_calibration(doc.get('calibration_id')), source)]


def _pair_provenance(first_role, first, second_role, second):
    """Flatten two stored results into role-tagged provenance entries."""
    entries = []
    for role, doc in ((first_role, first), (second_role, second)):
        for entry in _provenance(doc):
            entries.append({'role': role, 'result_id': doc.get('result_id'), **entry})
    return entries


def _clean_depth(depth_m):
    # Keep native SciPy loading in the geometry worker, not the stdio frontend
    # which also imports this module for transform validation.
    from scipy import ndimage
    valid = np.isfinite(depth_m) & (depth_m > 0.0) & (depth_m <= MAX_DEPTH_M)
    filled = valid.astype(np.float64)
    neighbours = ndimage.uniform_filter(filled, size=3, mode='constant', cval=0.0) * 9.0
    dmax = ndimage.maximum_filter(np.where(valid, depth_m, -np.inf), size=3,
                                  mode='constant', cval=-np.inf)
    dmin = ndimage.minimum_filter(np.where(valid, depth_m, np.inf), size=3,
                                  mode='constant', cval=np.inf)
    consistent = valid & (neighbours >= MIN_NEIGHBOURS) & ((dmax - dmin) <= DEFAULT_DISCONTINUITY_M)
    return valid, consistent


def _deproject(us, vs, z, intr):
    x = (us - intr['ppx']) * z / intr['fx']
    y = (vs - intr['ppy']) * z / intr['fy']
    return np.stack([x, y, z], axis=-1)


def _region_mask(shape, entry, field):
    roi = entry.get('roi')
    polygon = entry.get('polygon')
    if (roi is None) == (polygon is None):
        raise ValueError(f'{field}: exactly one of roi or polygon is required')
    height, width = shape
    if roi is not None:
        if (not isinstance(roi, (list, tuple)) or len(roi) != 4
                or any(isinstance(v, bool) or not isinstance(v, int) for v in roi)):
            raise ValueError(f'{field}.roi must be [x1, y1, x2, y2] in integer pixels')
        x1, y1, x2, y2 = (int(v) for v in roi)
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError(f'{field}.roi is outside the depth image')
        mask = np.zeros(shape, dtype=bool)
        mask[y1:y2, x1:x2] = True
        return mask, {'roi': [x1, y1, x2, y2]}
    try:
        points = np.asarray(polygon, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{field}.polygon must be a list of [x, y] points') from exc
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3 or not np.all(np.isfinite(points)):
        raise ValueError(f'{field}.polygon must contain at least 3 finite [x, y] points')
    if (points[:, 0].min() < 0 or points[:, 1].min() < 0
            or points[:, 0].max() > width or points[:, 1].max() > height):
        raise ValueError(f'{field}.polygon is outside the depth image')
    return _polygon_mask(shape, points), {'polygon': points.tolist()}


def _polygon_mask(shape, points):
    if cv2 is not None:
        mask = np.zeros(shape, dtype=np.uint8)
        cv2.fillPoly(mask, [np.round(points).astype(np.int32)], 1)
        return mask.astype(bool)
    height, width = shape  # pragma: no cover - fallback when the camera extra is absent
    xs = np.arange(width, dtype=np.float64) + 0.5
    ys = np.arange(height, dtype=np.float64) + 0.5
    inside = np.zeros(shape, dtype=bool)
    for index in range(len(points)):
        x1, y1 = points[index]
        x2, y2 = points[(index + 1) % len(points)]
        if y1 == y2:
            continue
        band = (ys[:, None] >= min(y1, y2)) & (ys[:, None] < max(y1, y2))
        crossing = x1 + (ys[:, None] - y1) * (x2 - x1) / (y2 - y1)
        inside ^= band & (xs[None, :] < crossing)
    return inside


def _voxel_downsample(points, voxel_size_m):
    """Deterministic voxel centroids; also returns the source index of each centroid."""
    keys = np.floor(points / voxel_size_m).astype(np.int64)
    unique, first, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    inverse = inverse.reshape(-1)
    counts = np.bincount(inverse, minlength=len(unique)).astype(np.float64)
    sums = np.zeros((len(unique), 3), dtype=np.float64)
    np.add.at(sums, inverse, points)
    return sums / counts[:, None], first.reshape(-1)


def _result_path(task, result_id):
    return task / 'geometry' / 'results' / f'{result_id}.json'


def _store_result(task, result_id, payload):
    return _atomic_write_json(_result_path(task, result_id), payload)


def _load_result(root, task, result_id, kind, field):
    _safe_id(result_id, field)
    path = _resolve_under(root, str(_result_path(task, result_id)), field=field,
                          suffixes=('.json',))
    try:
        doc = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f'{field} is not a readable stored result: {result_id}') from exc
    if not isinstance(doc, dict) or doc.get('kind') != kind:
        raise ValueError(f'{field} does not reference a stored {kind} result: {result_id}')
    return doc


# --------------------------------------------------------------------------- #
# surface_point
# --------------------------------------------------------------------------- #
def _surface_point(root, task_id, task, request):
    pixel = request.get('pixel')
    if (not isinstance(pixel, (list, tuple)) or len(pixel) != 2
            or any(isinstance(v, bool) or not isinstance(v, int) for v in pixel)):
        raise ValueError('pixel must be [u, v] in integer pixels')
    u, v = (int(v) for v in pixel)
    half = request.get('patch_half_size', 2)
    if isinstance(half, bool) or not isinstance(half, int) or not 1 <= half <= 8:
        raise ValueError('patch_half_size must be an integer between 1 and 8')
    obs = _load_observation(root, request.get('observation_path'))
    height, width = obs.depth_m.shape
    if not (0 <= u < width and 0 <= v < height):
        raise ValueError('pixel is outside the depth image')
    if u < half or v < half or u >= width - half or v >= height - half:
        raise _Unavailable('patch_half_size does not fit inside the depth image at this pixel')
    transform, calibration_id = _transform_for(request, obs, 'request')
    _, consistent = _clean_depth(obs.depth_m)
    window = np.s_[v - half:v + half + 1, u - half:u + half + 1]
    patch_ok = consistent[window]
    total = int(patch_ok.size)
    valid_count = int(patch_ok.sum())
    if valid_count != total:
        raise _Unavailable(
            f'surface patch rejected: {total - valid_count}/{total} pixels are missing or '
            f'discontinuous (zero depth, outside 0..{MAX_DEPTH_M} m, or a depth jump above '
            f'{DEFAULT_DISCONTINUITY_M} m)')
    patch_depth = obs.depth_m[window]
    rows, cols = np.mgrid[v - half:v + half + 1, u - half:u + half + 1]
    patch_points = _deproject(cols.ravel(), rows.ravel(), patch_depth.ravel(), obs.intr)
    point_camera = np.median(patch_points, axis=0)
    result = {
        'status': 'ok', 'operation': 'surface_point', 'kind': 'point',
        'task_id': task_id, 'frame': 'base' if transform is not None else obs.frame,
        'camera_frame': obs.frame, 'units': 'm', 'calibration_id': calibration_id,
        'transform_applied': transform is not None,
        'point_camera_m': [float(value) for value in point_camera],
        'method': 'consistent_patch_median_deprojection',
        'quality': {
            'valid_fraction': valid_count / total,
            'patch_pixels': total,
            'patch_half_size': half,
            'depth_std_m': float(np.std(patch_depth)),
            'depth_range_m': float(np.ptp(patch_depth)),
            'discontinuity_limit_m': DEFAULT_DISCONTINUITY_M,
        },
        'pixel': [u, v],
        'source': _source(obs),
        'calibration_provenance': [_provenance_entry(obs.serial, calibration_id,
                                                      {'observation': obs.rel,
                                                       'depth_digest': obs.depth_digest})],
        'safety_note': SAFETY_NOTE,
    }
    if transform is not None:
        point_base = _transform_points(transform, point_camera.reshape(1, 3))[0]
        result['point_base_m'] = [float(value) for value in point_base]
        result['point_m'] = result['point_base_m']
    else:
        result['point_m'] = result['point_camera_m']
    recipe = {'operation': 'surface_point', 'observation': obs.rel,
              'depth_digest': obs.depth_digest, 'pixel': [u, v], 'patch_half_size': half,
              'transform': None if transform is None else transform.tolist(),
              'calibration_id': calibration_id}
    result['result_id'] = 'pt_' + _digest(recipe)[:16]
    stored = _store_result(task, result['result_id'], result)
    result['result_path'] = str(stored.relative_to(root))
    return result


# --------------------------------------------------------------------------- #
# accumulate_region (shared with fit_plane mode A)
# --------------------------------------------------------------------------- #
def _collect_region_points(root, entries, *, voxel_size_m, stride, field):
    if not isinstance(entries, list) or not entries:
        raise ValueError(f'{field} must be a non-empty list')
    if len(entries) > 256:
        raise ValueError(f'{field} supports at most 256 entries')
    observations = {}
    view_state = {}
    serial_calibration = {}
    seen = set()
    chunks = []
    chunk_entries = []
    meta = []
    duplicates = 0
    sampled_total = 0
    for index, entry in enumerate(entries):
        where = f'{field}[{index}]'
        if not isinstance(entry, dict):
            raise ValueError(f'{where} must be an object')
        loaded = _load_observation(root, entry.get('observation_path'))
        cached = observations.get(loaded.rel)
        if cached is None:
            cached = (loaded, _clean_depth(loaded.depth_m)[1])
            observations[loaded.rel] = cached
        obs, consistent = cached
        transform, calibration_id = _transform_for(entry, obs, where)
        if transform is None or calibration_id is None:
            raise ValueError(f'{where}: base_from_camera and calibration_id are required to '
                             'accumulate or fit base-frame geometry')
        transform_key = _digest(transform.tolist())
        mask, region = _region_mask(obs.depth_m.shape, entry, where)
        if obs.rel in view_state and view_state[obs.rel] != (transform_key, calibration_id):
            raise ValueError(f'conflicting geometry for the same observation {obs.rel}: '
                             'base_from_camera/calibration_id differ from an earlier entry; '
                             'equal depth bytes from another observation are a different view')
        view_state[obs.rel] = (transform_key, calibration_id)
        if obs.serial:
            if obs.serial in serial_calibration and serial_calibration[obs.serial] != calibration_id:
                raise ValueError(f'mixed calibration for camera serial {obs.serial}: '
                                 f'{serial_calibration[obs.serial]!r} vs {calibration_id!r}')
            serial_calibration[obs.serial] = calibration_id
        entry_key = (obs.rel, _digest(region), stride, transform_key, calibration_id,
                     float(voxel_size_m))
        if entry_key in seen:
            duplicates += 1
            continue
        seen.add(entry_key)
        selected = mask & consistent
        if stride > 1:
            thinned = np.zeros_like(selected)
            thinned[::stride, ::stride] = True
            selected &= thinned
        rows, cols = np.nonzero(selected)
        record = {'entry_index': index, 'source': _source(obs), 'region': region,
                  'calibration_id': calibration_id, 'transform_sha256': transform_key,
                  'stride': stride, 'consistent_pixels': int((mask & consistent).sum()),
                  'sampled_points': int(len(rows)), 'voxel_centroids': 0}
        if len(rows) == 0:
            record['note'] = 'no consistent depth pixel in this region'
            meta.append(record)
            continue
        depths = obs.depth_m[rows, cols]
        camera_points = _deproject(cols.astype(np.float64), rows.astype(np.float64),
                                   depths, obs.intr)
        chunks.append(_transform_points(transform, camera_points))
        chunk_entries.append(np.full(len(camera_points), len(meta), dtype=np.int64))
        sampled_total += len(rows)
        if sampled_total > 2000000:
            raise _Unavailable('Query exceeds two million samples; use a smaller ROI or larger stride')
        meta.append(record)
    if not chunks:
        raise _Unavailable('no consistent depth pixels in the requested region(s); '
                           'nothing was accumulated')
    points = np.concatenate(chunks, axis=0)
    entry_ids = np.concatenate(chunk_entries)
    voxels, first = _voxel_downsample(points, voxel_size_m)
    if len(voxels) > MAX_POINTS:
        raise _Unavailable(f'region produced {len(voxels)} voxel centroids, above the '
                           f'{MAX_POINTS}-point cap; increase voxel_size_m')
    counts = np.bincount(entry_ids[first], minlength=len(meta))
    for index, record in enumerate(meta):
        record['voxel_centroids'] = int(counts[index])
    return {'points': voxels, 'duplicates': duplicates, 'observations': meta,
            'entry_index': entry_ids[first], 'sampled_points': sampled_total,
            'consistent_pixels': sum(record['consistent_pixels'] for record in meta)}


def _check_batch_immutable(target_id, batch_id, npz_path, meta_path, request_digest, npz_bytes):
    """Refuse conflicting or unverifiable rewrites of an existing target/batch id."""
    existing = None
    if meta_path.exists():
        try:
            existing = json.loads(meta_path.read_text(encoding='utf-8'))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f'batch {target_id}/{batch_id} already has unreadable metadata; '
                             'target_id/batch_id are immutable, so the operator must repair or '
                             'remove the stored pair') from exc
    if isinstance(existing, dict):
        if existing.get('request_digest') != request_digest:
            raise ValueError(f'batch {target_id}/{batch_id} already exists with a different '
                             'recipe; target_id/batch_id are immutable, choose a new batch_id')
        return
    if npz_path.exists():
        # A committed NPZ always has metadata; an orphan NPZ is a torn pair. Completing it is
        # only safe when this recipe reproduces the exact bytes already on disk.
        try:
            same_bytes = npz_path.read_bytes() == npz_bytes
        except OSError:
            same_bytes = False
        if not same_bytes:
            raise ValueError(f'batch {target_id}/{batch_id} exists as a torn pair without '
                             'metadata and the stored NPZ does not match this recipe; '
                             'target_id/batch_id are immutable, so the operator must repair or '
                             'remove the stored pair')


def _accumulate_region(root, task_id, task, request):
    target_id = _safe_id(request.get('target_id'), 'target_id')
    batch_id = _safe_id(request.get('batch_id'), 'batch_id')
    voxel_size_m = _number(request.get('voxel_size_m', DEFAULT_VOXEL_M), 'voxel_size_m')
    if not 1e-4 <= voxel_size_m <= 0.1:
        raise ValueError('voxel_size_m must be between 0.0001 and 0.1 metres')
    stride = request.get('stride', 1)
    if isinstance(stride, bool) or not isinstance(stride, int) or not 1 <= stride <= 32:
        raise ValueError('stride must be an integer between 1 and 32')
    info = _collect_region_points(root, request.get('observations'),
                                  voxel_size_m=voxel_size_m, stride=stride,
                                  field='observations')
    points = info['points']
    plane_id=request.get('plane_id')
    if plane_id:
        plane=_load_result(root,task,plane_id,'plane','plane_id')
        if plane.get('frame')!='base':raise ValueError('Region support plane must be in base frame')
        normal=np.asarray(plane['normal']);origin=np.asarray(plane['point_on_plane_m'])
        if normal[2]<.9:raise ValueError('Region support plane must face upward')
        points=points[(points-origin)@normal>.008]
        if not len(points):raise _Unavailable('No observed points above the selected support plane')
    batch_dir = task / 'geometry' / 'targets' / target_id
    npz_path = batch_dir / f'{batch_id}.npz'
    meta_path = batch_dir / f'{batch_id}.json'
    result_id = f'batch.{target_id}.{batch_id}'
    npz_rel = str(npz_path.relative_to(root))
    meta_rel = str(meta_path.relative_to(root))
    request_digest = _digest({'target_id': target_id, 'batch_id': batch_id,
                              'voxel_size_m': voxel_size_m, 'stride': stride,
                              **({'support_plane_id':plane_id,'support_plane':plane,'min_height_m':.008} if plane_id else {}),
                              'observations': info['observations']})
    npz_bytes = _npz_bytes({'points_base_m': points.astype(np.float64),
                            'entry_index': (info['entry_index'][(info['points']-origin)@normal>.008]
                                            if plane_id else info['entry_index']).astype(np.int64)})
    metadata = {
        'kind': 'batch', 'result_id': result_id, 'task_id': task_id,
        'target_id': target_id, 'batch_id': batch_id, 'frame': 'base', 'units': 'm',
        'voxel_size_m': voxel_size_m, 'stride': stride, 'voxel_cap': MAX_POINTS,
        'point_count': int(len(points)), 'points_npz': npz_rel, 'metadata_json': meta_rel,
        'npz_sha256': hashlib.sha256(npz_bytes).hexdigest(),
        'duplicates_skipped': info['duplicates'], 'observations': info['observations'],
        'sampled_points': info['sampled_points'],
        'method': 'consistent_depth_region_deprojection_voxel_centroids',
        'request_digest': request_digest,
        'notes': ['Only measured surface voxels are stored; no free-space, occupancy or '
                  'clearance volume is produced.',
                  'Voxel centroids merge repeated observations of the same region.',
                  'target_id/batch_id are immutable: the same recipe rewrites the same pair, '
                  'a different recipe or a torn pair is rejected.'],
        'safety_note': SAFETY_NOTE,
    }
    _check_batch_immutable(target_id, batch_id, npz_path, meta_path, request_digest, npz_bytes)
    _atomic_write_bytes(npz_path, npz_bytes)
    _atomic_write_json(meta_path, metadata)
    return {
        'status': 'ok', 'operation': 'accumulate_region', 'kind': 'batch',
        'task_id': task_id, 'result_id': result_id, 'target_id': target_id,
        'batch_id': batch_id, 'points': int(len(points)), 'points_npz': npz_rel,
        'metadata_json': meta_rel, 'frame': 'base', 'units': 'm',
        'voxel_size_m': voxel_size_m, 'stride': stride, 'voxel_cap': MAX_POINTS,
        'duplicates_skipped': info['duplicates'], 'observations': info['observations'],
        'surface_summary': _surface_summary(points),
        **({'support_plane_id':plane_id,'background_filter_height_m':.008,
            'calibration_provenance':plane.get('calibration_provenance',[])+[
                _provenance_entry(o['source']['camera_serial'],o['calibration_id'],o['source']) for o in info['observations']]} if plane_id else {}),
        'method': metadata['method'],
        'quality': {'voxels': int(len(points)), 'sampled_points': info['sampled_points'],
                    'consistent_pixels': info['consistent_pixels'],
                    'source_observation_count': len({record['source']['observation']
                                                     for record in info['observations']})},
        'notes': metadata['notes'], 'safety_note': SAFETY_NOTE,
    }


def _cluster_diagnostics(points, cell_size_m=0.012):
    """Deterministic 26-neighbour 3D voxel-grid connected components without SciPy."""
    if len(points) == 0:
        return 0, 0.0, False, np.empty((0,), dtype=np.int64)
    cells = np.floor(points / cell_size_m).astype(np.int64)
    cell_map = {}
    for idx, cell in enumerate(map(tuple, cells)):
        cell_map.setdefault(cell, []).append(idx)
    visited = set()
    components = []
    offsets = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)
               if (dx, dy, dz) != (0, 0, 0)]
    for start in cell_map:
        if start in visited:
            continue
        visited.add(start)
        queue = [start]
        member_indices = []
        while queue:
            cur = queue.pop()
            member_indices.extend(cell_map[cur])
            cx, cy, cz = cur
            for dx, dy, dz in offsets:
                nb = (cx + dx, cy + dy, cz + dz)
                if nb in cell_map and nb not in visited:
                    visited.add(nb)
                    queue.append(nb)
        components.append(np.asarray(sorted(member_indices), dtype=np.int64))
    components.sort(key=len, reverse=True)
    min_sig = max(3, int(math.ceil(0.02 * len(points))))
    sig = [c for c in components if len(c) >= min_sig] or components[:1]
    dom = components[0]
    return len(sig), float(len(dom) / len(points)), bool(len(sig) > 1), dom


def _xy_pca(points):
    xy = points[:, :2] - np.mean(points[:, :2], axis=0)
    covariance = xy.T @ xy / max(1, len(points))
    eigenvalues, axes = np.linalg.eigh(covariance)
    minor = axes[:, 0]
    major = axes[:, 1]
    projected_minor = xy @ minor
    projected_major = xy @ major
    minor_yaw = float((math.degrees(math.atan2(minor[1], minor[0])) + 90) % 180 - 90)
    major_yaw = float((math.degrees(math.atan2(major[1], major[0])) + 90) % 180 - 90)
    return {
        'minor_axis_xy': minor.tolist(),
        'minor_axis_yaw_deg': minor_yaw,
        'observed_minor_width_m': float(np.quantile(projected_minor, .98) - np.quantile(projected_minor, .02)),
        'major_axis_xy': major.tolist(),
        'major_axis_yaw_deg': major_yaw,
        'observed_major_span_m': float(np.quantile(projected_major, .98) - np.quantile(projected_major, .02)),
        'orientation_ambiguous': bool(eigenvalues[1] <= 1.2 * eigenvalues[0] + 1e-12),
    }


def _surface_summary(points):
    """Compact measured ROI geometry, never a full hidden-object model."""
    lower, upper = np.quantile(points, [.02, .98], axis=0)
    center = np.median(points, axis=0)
    pca = _xy_pca(points)
    cluster_count, dom_fraction, multi_clusters, dom_idx = _cluster_diagnostics(points)
    dom_pts = points[dom_idx] if len(dom_idx) else points
    dom_lower, dom_upper = np.quantile(dom_pts, [.02, .98], axis=0)

    top_band_m = 0.012
    top_mask = (points[:, 2] >= upper[2] - top_band_m) & (points[:, 2] <= upper[2] + 0.006)
    top_pts = points[top_mask]
    if len(top_pts) >= 5:
        _, _, _, top_dom_idx = _cluster_diagnostics(top_pts)
        if len(top_dom_idx) >= 5:
            top_pts = top_pts[top_dom_idx]
    top_summary = None
    if len(top_pts) >= 5:
        top_lo, top_hi = np.quantile(top_pts, [.02, .98], axis=0)
        top_med = np.median(top_pts, axis=0)
        top_pca = _xy_pca(top_pts)
        top_summary = {
            'band_thickness_m': top_band_m,
            'point_count': int(len(top_pts)),
            'point_fraction': float(len(top_pts) / len(points)),
            'median_point_m': top_med.tolist(),
            'observed_bounds_m': {'lower': top_lo.tolist(), 'upper': top_hi.tolist()},
            'observed_extent_m': (top_hi - top_lo).tolist(),
            **top_pca,
        }
        # When significant vertical relief is present (oblique side wall on a tall
        # cylinder/block or a lower adjacent object), an isotropic top face must
        # not inherit a false elongation axis from camera-facing side-wall pixels.
        if (upper[2] - lower[2]) > 0.015 and len(top_pts) >= max(8, int(0.08 * len(points))):
            if top_pca['orientation_ambiguous']:
                pca['orientation_ambiguous'] = True

    step = max(1, len(points) // 200)
    sampled = np.round(points[::step][:200], 5).tolist()
    return {
        'frame': 'base',
        'units': 'm',
        'median_point_m': center.tolist(),
        'observed_bounds_m': {'lower': lower.tolist(), 'upper': upper.tolist()},
        'observed_extent_m': (upper - lower).tolist(),
        **pca,
        'cluster_count': cluster_count,
        'dominant_cluster_fraction': dom_fraction,
        'multiple_clusters_detected': multi_clusters,
        'dominant_cluster_bounds_m': {'lower': dom_lower.tolist(), 'upper': dom_upper.tolist()},
        'dominant_cluster_median_point_m': np.median(dom_pts, axis=0).tolist(),
        'top_surface_summary': top_summary,
        'sampled_points_base_m': sampled,
        'scope': 'Visible ROI surface points only. Background and occlusion affect this summary; not full object dimensions or grasp feasibility.',
    }


# --------------------------------------------------------------------------- #
# fit_plane
# --------------------------------------------------------------------------- #
def _ransac_params(request):
    raw = request.get('ransac', {})
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError('ransac must be an object')
    unknown = set(raw) - set(DEFAULT_RANSAC)
    if unknown:
        raise ValueError(f'ransac has unknown keys: {sorted(unknown)}')
    params = dict(DEFAULT_RANSAC)
    params.update(raw)
    if isinstance(params['seed'], bool) or not isinstance(params['seed'], int) or params['seed'] < 0:
        raise ValueError('ransac.seed must be a non-negative integer')
    if (isinstance(params['max_iterations'], bool) or not isinstance(params['max_iterations'], int)
            or not 1 <= params['max_iterations'] <= 10000):
        raise ValueError('ransac.max_iterations must be an integer between 1 and 10000')
    threshold = _number(params['distance_threshold_m'], 'ransac.distance_threshold_m')
    if not 0 < threshold <= 0.1:
        raise ValueError('ransac.distance_threshold_m must be in (0, 0.1]')
    if (isinstance(params['min_inliers'], bool) or not isinstance(params['min_inliers'], int)
            or params['min_inliers'] < 3):
        raise ValueError('ransac.min_inliers must be an integer >= 3')
    fraction = _number(params['min_inlier_fraction'], 'ransac.min_inlier_fraction')
    if not 0 < fraction <= 1:
        raise ValueError('ransac.min_inlier_fraction must be in (0, 1]')
    for key in ('min_extent_m', 'min_width_m', 'max_thickness_m'):
        value = _number(params[key], f'ransac.{key}')
        if value < 0:
            raise ValueError(f'ransac.{key} must be >= 0')
        params[key] = value
    params['distance_threshold_m'] = threshold
    params['min_inlier_fraction'] = fraction
    return params


def _svd_plane(points):
    centroid = points.mean(axis=0)
    try:
        _, singular, vt = np.linalg.svd(points - centroid, full_matrices=False)
    except np.linalg.LinAlgError as exc:
        raise _Unavailable(f'plane SVD failed: {exc}') from exc
    if singular[0] <= 1e-12 or singular[1] <= 1e-12:
        raise _Unavailable('degenerate support: points are coincident or collinear')
    normal = vt[-1]
    normal = normal / float(np.linalg.norm(normal))
    return {'normal': normal, 'offset': -float(normal @ centroid), 'centroid': centroid,
            'singular': singular}


def _ransac_plane(points, params):
    count = len(points)
    if count < 3:
        raise _Unavailable(f'at least 3 points are required to fit a plane; got {count}')
    rng = np.random.default_rng(params['seed'])
    threshold = params['distance_threshold_m']
    best = None
    for iteration in range(params['max_iterations']):
        p0, p1, p2 = points[rng.choice(count, size=3, replace=False)]
        candidate = np.cross(p1 - p0, p2 - p0)
        norm = float(np.linalg.norm(candidate))
        if not np.isfinite(norm) or norm < 1e-12:
            continue
        normal = candidate / norm
        offset = -float(normal @ p0)
        distance = np.abs(points @ normal + offset)
        inlier_mask = distance <= threshold
        inliers = int(inlier_mask.sum())
        if inliers < 3:
            continue
        score = (inliers, -float(distance[inlier_mask].mean()), -iteration)
        if best is None or score > best[0]:
            best = (score, normal, offset)
    if best is None:
        raise _Unavailable('deterministic RANSAC found no non-degenerate plane hypothesis')
    normal, offset = best[1], best[2]
    for _ in range(2):
        distance = np.abs(points @ normal + offset)
        inlier_mask = distance <= threshold
        if int(inlier_mask.sum()) < 3:
            raise _Unavailable('inadequate inlier support after SVD refinement')
        refined = _svd_plane(points[inlier_mask])
        normal, offset = refined['normal'], refined['offset']
    signed = points @ normal + offset
    inlier_mask = np.abs(signed) <= threshold
    final = _svd_plane(points[inlier_mask])
    normal, offset = final['normal'], final['offset']
    inliers = int(inlier_mask.sum())
    spread = final['singular'] / math.sqrt(inliers)
    if spread[0] < params['min_extent_m']:
        raise _Unavailable(f'inadequate spatial extent: principal spread {spread[0]:.4f} m '
                           f'< min_extent_m {params["min_extent_m"]}')
    if spread[1] < params['min_width_m']:
        raise _Unavailable(f'degenerate support: collinear points, second spread '
                           f'{spread[1]:.4f} m < min_width_m {params["min_width_m"]}')
    if spread[2] > params['max_thickness_m']:
        raise _Unavailable(f'inlier band too thick ({spread[2]:.4f} m > max_thickness_m '
                           f'{params["max_thickness_m"]}); not treated as a plane')
    fraction = inliers / count
    if inliers < params['min_inliers']:
        raise _Unavailable(f'inadequate support: {inliers} inliers < min_inliers '
                           f'{params["min_inliers"]}')
    if fraction < params['min_inlier_fraction']:
        raise _Unavailable(f'inadequate support fraction: {fraction:.3f} < min_inlier_fraction '
                           f'{params["min_inlier_fraction"]}')
    residual = np.abs(signed[inlier_mask])
    return {
        'normal': normal, 'offset': offset, 'centroid': final['centroid'],
        'inlier_count': inliers, 'input_count': count, 'inlier_fraction': fraction,
        'residual_rms': float(np.sqrt(np.mean(residual ** 2))),
        'residual_max': float(residual.max()),
        'residual_p95': float(np.percentile(residual, 95)),
        'spread': spread, 'seed': params['seed'], 'max_iterations': params['max_iterations'],
        'distance_threshold_m': threshold,
    }


def _fit_plane(root, task_id, task, request):
    if request.get('batch_id') is not None:
        if request.get('observation_path') is not None:
            raise ValueError('fit_plane takes either a batch (target_id + batch_id) or one '
                             'observation region, not both')
        points, source, provenance = _load_batch(root, task, request)
        declared = {entry['calibration_id'] for entry in provenance}
        calibration_id = declared.pop() if len(declared) == 1 else None
    else:
        voxel_size_m = _number(request.get('voxel_size_m', DEFAULT_VOXEL_M), 'voxel_size_m')
        if not 1e-4 <= voxel_size_m <= 0.1:
            raise ValueError('voxel_size_m must be between 0.0001 and 0.1 metres')
        stride = request.get('stride', 1)
        if isinstance(stride, bool) or not isinstance(stride, int) or not 1 <= stride <= 32:
            raise ValueError('stride must be an integer between 1 and 32')
        entry = {'observation_path': request.get('observation_path'), 'roi': request.get('roi'),
                 'polygon': request.get('polygon'),
                 'base_from_camera': request.get('base_from_camera'),
                 'calibration_id': request.get('calibration_id')}
        info = _collect_region_points(root, [entry], voxel_size_m=voxel_size_m, stride=stride,
                                      field='fit_plane')
        record = info['observations'][0]
        points, source = info['points'], record['source']
        calibration_id = record['calibration_id']
        provenance = [_provenance_entry(record['source']['camera_serial'], calibration_id,
                                        record['source'])]
    params = _ransac_params(request)
    fit = _ransac_plane(points, params)
    normal, offset = fit['normal'], fit['offset']
    reference = request.get('reference_normal')
    if reference is not None:
        try:
            reference = np.asarray(reference, dtype=np.float64).reshape(-1)
        except (TypeError, ValueError) as exc:
            raise ValueError('reference_normal must be [x, y, z]') from exc
        if reference.shape != (3,) or not np.all(np.isfinite(reference)):
            raise ValueError('reference_normal must be a finite [x, y, z] vector')
        norm = float(np.linalg.norm(reference))
        if norm < 1e-9:
            raise ValueError('reference_normal must be a non-zero vector')
        reference = reference / norm
        alignment = float(normal @ reference)
        if abs(alignment) < 1e-9:
            raise _Unavailable('reference_normal is parallel to the fitted plane; the plane '
                               'normal sign is ambiguous')
        if alignment < 0:
            normal, offset = -normal, -offset
        orientation = {'convention': 'aligned_to_reference_normal',
                       'reference_normal': [float(value) for value in reference],
                       'dot': abs(alignment)}
    else:
        component = int(np.argmax(np.abs(normal)))
        if normal[component] < 0:
            normal, offset = -normal, -offset
        orientation = {'convention': 'canonical_largest_component_positive',
                       'reference_normal': None, 'component': 'xyz'[component]}
    centroid = fit['centroid']
    recipe = {'operation': 'fit_plane', 'points_digest':
              hashlib.sha256(np.ascontiguousarray(points, dtype=np.float64).tobytes()).hexdigest(),
              'params': params, 'orientation': orientation['convention'],
              'reference_normal': orientation['reference_normal'],
              'provenance_digest': _digest(provenance)}
    result_id = 'pl_' + _digest(recipe)[:16]
    result = {
        'status': 'ok', 'operation': 'fit_plane', 'kind': 'plane', 'task_id': task_id,
        'result_id': result_id, 'plane_id': result_id, 'frame': 'base', 'units': 'm',
        'calibration_id': calibration_id,
        'calibration_provenance': provenance,
        'normal': [float(value) for value in normal],
        'point_on_plane_m': [float(value) for value in centroid],
        'signed_distance_offset_m': float(offset),
        'signed_distance_convention': SIGNED_DISTANCE_CONVENTION,
        'orientation': orientation,
        'support': {'inlier_count': fit['inlier_count'], 'input_count': fit['input_count'],
                    'inlier_fraction': fit['inlier_fraction']},
        'residual_m': {'rms': fit['residual_rms'], 'max_abs': fit['residual_max'],
                       'p95_abs': fit['residual_p95']},
        'quality': {'spread_m': [float(value) for value in fit['spread']],
                    'thickness_rms_m': float(fit['spread'][2]),
                    'distance_threshold_m': fit['distance_threshold_m'],
                    'seed': fit['seed'], 'max_iterations': fit['max_iterations'],
                    'input_points': fit['input_count']},
        'method': 'deterministic_seeded_ransac_then_svd',
        'source': {'points': source, 'points_digest': recipe['points_digest']},
        'accuracy_note': ('RANSAC inlier residual and support only; this is not a total '
                          'accuracy, flatness or clearance claim.'),
        'safety_note': SAFETY_NOTE,
    }
    result['result_path'] = str(_store_result(task, result_id, result).relative_to(root))
    return result


def _load_batch(root, task, request):
    """Load a batch only when its committed metadata matches the stored NPZ byte for byte."""
    target_id = _safe_id(request.get('target_id'), 'target_id')
    batch_id = _safe_id(request.get('batch_id'), 'batch_id')
    batch_dir = task / 'geometry' / 'targets' / target_id
    npz_path = batch_dir / f'{batch_id}.npz'
    meta_path = batch_dir / f'{batch_id}.json'
    resolved_npz = _resolve_under(root, str(npz_path), field='batch_id', suffixes=('.npz',))
    resolved_meta = _resolve_under(root, str(meta_path), field='batch_id', suffixes=('.json',))
    try:
        metadata = json.loads(resolved_meta.read_text(encoding='utf-8'))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f'batch {target_id}/{batch_id} metadata is not readable JSON') from exc
    if not isinstance(metadata, dict) or metadata.get('kind') != 'batch':
        raise ValueError(f'batch {target_id}/{batch_id} metadata is not a batch record')
    if metadata.get('target_id') != target_id or metadata.get('batch_id') != batch_id:
        raise ValueError(f'batch {target_id}/{batch_id} metadata describes another target or batch')
    npz_rel = str(resolved_npz.relative_to(root))
    meta_rel = str(resolved_meta.relative_to(root))
    if metadata.get('points_npz') != npz_rel:
        raise ValueError(f'batch {target_id}/{batch_id} metadata does not reference its own NPZ')
    try:
        raw = resolved_npz.read_bytes()
    except OSError as exc:
        raise ValueError(f'batch {target_id}/{batch_id} NPZ is not readable: {exc}') from exc
    digest = hashlib.sha256(raw).hexdigest()
    declared_digest = metadata.get('npz_sha256')
    if not isinstance(declared_digest, str):
        raise ValueError(f'batch {target_id}/{batch_id} metadata has no NPZ digest; the pair '
                         'cannot be verified as committed, re-run accumulate_region for it')
    if declared_digest != digest:
        raise ValueError(f'batch {target_id}/{batch_id} metadata/NPZ digest mismatch: the pair is '
                         'torn or was modified; re-run accumulate_region with the same recipe')
    try:
        with np.load(io.BytesIO(raw), allow_pickle=False) as data:
            points = np.asarray(data['points_base_m'], dtype=np.float64)
    except Exception as exc:
        raise ValueError(f'batch {target_id}/{batch_id} is not a readable points NPZ: '
                         f'{exc}') from exc
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        raise ValueError(f'batch {target_id}/{batch_id} does not hold an (N, 3) point cloud')
    if metadata.get('point_count') != len(points):
        raise ValueError(f'batch {target_id}/{batch_id} metadata point_count does not match '
                         'its NPZ')
    provenance = _batch_provenance(metadata, target_id, batch_id)
    source = {'target_id': target_id, 'batch_id': batch_id, 'batch_npz': npz_rel,
              'batch_metadata': meta_rel, 'npz_sha256': digest,
              'request_digest': metadata.get('request_digest'),
              'observations': metadata.get('observations')}
    return points, source, provenance


def _batch_provenance(metadata, target_id, batch_id):
    """Validate batch metadata sources and return their per-serial calibration mapping."""
    records = metadata.get('observations')
    if not isinstance(records, list) or not records:
        raise ValueError(f'batch {target_id}/{batch_id} metadata has no observation sources')
    entries = []
    serial_calibration = {}
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f'batch {target_id}/{batch_id} metadata observations[{index}] '
                             'is not an object')
        calibration = _clean_calibration(record.get('calibration_id'))
        if calibration is None:
            raise ValueError(f'batch {target_id}/{batch_id} metadata observations[{index}] '
                             'has no calibration_id')
        source = record.get('source') if isinstance(record.get('source'), dict) else {}
        serial = source.get('camera_serial') if isinstance(source.get('camera_serial'), str) else ''
        if serial:
            previous = serial_calibration.get(serial)
            if previous is not None and previous != calibration:
                raise ValueError(f'batch {target_id}/{batch_id} metadata mixes calibrations for '
                                 f'camera serial {serial}: {previous!r} vs {calibration!r}')
            serial_calibration[serial] = calibration
        entries.append(_provenance_entry(serial, calibration, source))
    return entries


# --------------------------------------------------------------------------- #
# queries
# --------------------------------------------------------------------------- #
def _reject_raw_xyz(request, operation):
    present = [key for key in RAW_XYZ_KEYS if key in request]
    if present:
        raise ValueError(f'{operation} accepts stored result ids only; raw coordinates are '
                         f'not accepted ({", ".join(present)})')


def _query_point(root, task, request, field):
    doc = _load_result(root, task, request.get(field), 'point', field)
    point = doc.get('point_m')
    if (not isinstance(point, list) or len(point) != 3
            or not all(isinstance(value, (int, float)) and not isinstance(value, bool)
                       and math.isfinite(float(value)) for value in point)):
        raise ValueError(f'{field} references a point result without a finite point_m')
    return doc, np.asarray(point, dtype=np.float64)


def _calibration_map(doc):
    """Declared camera serial -> set of calibration ids recorded in a stored result."""
    mapping = {}
    for entry in _provenance(doc):
        mapping.setdefault(entry['camera_serial'], set()).add(entry['calibration_id'])
    return mapping


def _frame_compatibility(first, second, what):
    """Validate a stored result pair and report its calibration match.

    Returns ``True`` when the declared calibrations agree, ``False`` when a
    shared base-frame pair comes from different declared cameras with their own
    calibrations (permitted by design), and ``None`` when at least one side
    declares no calibration. A calibration that changed for one declared camera
    serial is rejected for every frame.
    """
    if first.get('frame') != second.get('frame'):
        raise ValueError(f'{what} frames differ ({first.get("frame")} vs {second.get("frame")}); '
                         'resample both results in one frame before measuring')
    first_map = _calibration_map(first)
    second_map = _calibration_map(second)
    if first.get('frame') != 'base':
        if first.get('source',{}).get('observation') != second.get('source',{}).get('observation'):
            raise ValueError('Optical-frame distances require points from the same observation; otherwise use calibrated base coordinates')
        first_serials = {serial for serial in first_map if serial}
        second_serials = {serial for serial in second_map if serial}
        if first_serials != second_serials:
            raise ValueError(f'{what} are camera-frame results from different cameras '
                             f'({sorted(first_serials)} vs {sorted(second_serials)}); '
                             'resample both in base')
    for serial in sorted(set(first_map) & set(second_map)):
        if not serial:
            continue
        declared_first = {value for value in first_map[serial] if value}
        declared_second = {value for value in second_map[serial] if value}
        if declared_first and declared_second and declared_first != declared_second:
            raise ValueError(f'{what} declare conflicting calibrations for camera serial {serial} '
                             f'({sorted(declared_first)} vs {sorted(declared_second)}); a camera '
                             'must not mix results from two calibrations')
    first_calibrations = {value for values in first_map.values() for value in values if value}
    second_calibrations = {value for values in second_map.values() for value in values if value}
    if not first_calibrations or not second_calibrations:
        return None
    shared = [serial for serial in first_map if serial and serial in second_map]
    if shared or first_calibrations == second_calibrations:
        return True
    return False


def _point_distance(root, task_id, task, request):
    _reject_raw_xyz(request, 'point_distance')
    first, first_point = _query_point(root, task, request, 'point_id')
    second, second_point = _query_point(root, task, request, 'other_point_id')
    calibration_match = _frame_compatibility(first, second, 'point results')
    delta = second_point - first_point
    notes = ['Distance between two stored measured points; it is not a clearance, '
             'collision or reachability check.']
    if calibration_match is False:
        notes.append('The stored points declare different calibrations with no camera serial in '
                     'common; the base-frame pair is permitted. Revalidate '
                     'calibration_provenance against current operator registrations before reuse.')
    return {
        'status': 'ok', 'operation': 'point_distance', 'kind': 'distance',
        'task_id': task_id, 'point_id': first['result_id'],
        'other_point_id': second['result_id'], 'frame': first['frame'], 'units': 'm',
        'distance_m': float(np.linalg.norm(delta)),
        'delta_m': [float(value) for value in delta],
        'calibration_id': first.get('calibration_id'),
        'calibration_match': calibration_match,
        'calibration_provenance': _pair_provenance('point_id', first,
                                                   'other_point_id', second),
        'method': 'stored_point_euclidean_distance',
        'quality': {'first_valid_fraction': first.get('quality', {}).get('valid_fraction'),
                    'second_valid_fraction': second.get('quality', {}).get('valid_fraction')},
        'sources': [first.get('source'), second.get('source')],
        'clearance_verified': False,
        'notes': notes,
        'safety_note': SAFETY_NOTE,
    }


def _point_plane_distance(root, task_id, task, request):
    _reject_raw_xyz(request, 'point_plane_distance')
    point_doc, point = _query_point(root, task, request, 'point_id')
    plane = _load_result(root, task, request.get('plane_id'), 'plane', 'plane_id')
    calibration_match = _frame_compatibility(point_doc, plane, 'point and plane')
    normal = np.asarray(plane.get('normal'), dtype=np.float64)
    origin = np.asarray(plane.get('point_on_plane_m'), dtype=np.float64)
    if normal.shape != (3,) or origin.shape != (3,) or not np.all(np.isfinite(normal)) \
            or not np.all(np.isfinite(origin)):
        raise ValueError('plane_id references a plane result without a finite normal/origin')
    reference = request.get('reference_normal')
    orientation = {'convention': plane.get('orientation', {}).get(
        'convention', 'stored_plane_normal'), 'reference_normal': None,
        'normal_flipped_for_query': False}
    if reference is not None:
        try:
            reference = np.asarray(reference, dtype=np.float64).reshape(-1)
        except (TypeError, ValueError) as exc:
            raise ValueError('reference_normal must be [x, y, z]') from exc
        if reference.shape != (3,) or not np.all(np.isfinite(reference)):
            raise ValueError('reference_normal must be a finite [x, y, z] vector')
        norm = float(np.linalg.norm(reference))
        if norm < 1e-9:
            raise ValueError('reference_normal must be a non-zero vector')
        reference = reference / norm
        alignment = float(normal @ reference)
        if abs(alignment) < 1e-9:
            raise _Unavailable('reference_normal is parallel to the stored plane; the signed '
                               'distance sign is ambiguous')
        if alignment < 0:
            normal = -normal
            orientation['normal_flipped_for_query'] = True
        orientation['convention'] = 'request_reference_normal'
        orientation['reference_normal'] = [float(value) for value in reference]
    signed = float(normal @ (point - origin))
    band = plane.get('residual_m', {}).get('rms')
    band = float(band) if isinstance(band, (int, float)) else None
    if band is not None and abs(signed) <= band:
        side = 'within_plane_residual_band'
    elif signed > 0:
        side = 'normal_side'
    elif signed < 0:
        side = 'opposite_side'
    else:
        side = 'on_plane'
    notes = ['Signed distance uses the reported normal convention; the plane is a fitted '
             'inlier model, not a certified surface or clearance boundary.']
    if calibration_match is False:
        notes.append('The stored point and plane declare different calibrations with no camera '
                     'serial in common; the base-frame pair is permitted. Revalidate '
                     'calibration_provenance against current operator registrations before reuse.')
    return {
        'status': 'ok', 'operation': 'point_plane_distance', 'kind': 'distance',
        'task_id': task_id, 'point_id': point_doc['result_id'], 'plane_id': plane['result_id'],
        'frame': plane['frame'], 'units': 'm',
        'signed_distance_m': signed, 'absolute_distance_m': abs(signed), 'side': side,
        'calibration_id': plane.get('calibration_id'),
        'calibration_match': calibration_match,
        'calibration_provenance': _pair_provenance('point_id', point_doc, 'plane_id', plane),
        'normal_used': [float(value) for value in normal],
        'orientation': orientation,
        'signed_distance_convention': plane.get('signed_distance_convention',
                                                SIGNED_DISTANCE_CONVENTION),
        'plane_quality': plane.get('quality'), 'plane_support': plane.get('support'),
        'plane_residual_m': plane.get('residual_m'),
        'method': 'stored_point_to_stored_plane_signed_distance',
        'sources': [point_doc.get('source'), plane.get('source')],
        'clearance_verified': False,
        'notes': notes,
        'safety_note': SAFETY_NOTE,
    }


_OPERATIONS = {
    'surface_point': _surface_point,
    'accumulate_region': _accumulate_region,
    'fit_plane': _fit_plane,
    'point_distance': _point_distance,
    'point_plane_distance': _point_plane_distance,
}


def compute(root: Path, request: dict) -> dict:
    """Run one geometry request against a task directory under ``root``.

    See the module docstring for the exact request shapes. Invalid requests raise
    ``ValueError``; requests that cannot honestly be answered return
    ``{'status': 'unavailable', 'reason': ...}``.
    """
    root_path = Path(root)
    if not root_path.is_dir():
        raise ValueError(f'root is not a directory: {root}')
    root_path = root_path.resolve()
    if not isinstance(request, dict):
        raise ValueError('request must be a JSON object')
    operation = request.get('operation')
    if operation not in OPERATIONS:
        raise ValueError(f'operation must be one of {", ".join(OPERATIONS)}')
    task_id, task = _task(root_path, request)
    try:
        result = _OPERATIONS[operation](root_path, task_id, task, request)
        if result.get('status') == 'ok' and operation in ('point_distance','point_plane_distance'):
            ident = 'distance_' + _digest(result)[:16]
            result['result_id'] = ident
            result['result_path'] = str(_result_path(task,ident).relative_to(root_path))
            _store_result(task,ident,result)
        return result
    except _Unavailable as exc:
        return {'status': 'unavailable', 'operation': operation, 'task_id': task_id,
                'reason': str(exc)}
    except ValueError:
        raise
    except Exception as exc:  # defensive backstop for a subprocess worker
        return {'status': 'unavailable', 'operation': operation, 'task_id': task_id,
                'reason': f'internal {type(exc).__name__}: {exc}'}
