"""Deterministic PiperX MDH FK and finite multistart numerical IK.

Coordinates are base-to-TCP metres; angles are degrees. RPY is extrinsic xyz,
Rz(yaw) @ Ry(pitch) @ Rx(roll). Solver tolerances express numerical convergence,
not physical accuracy or collision checking.
"""
from __future__ import annotations

import math
import warnings

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .models import DomainError, JOINT_LIMITS_DEG

# SDK ROBOT_MDH_PRESET['piper_x']: (d, a, alpha, theta_offset), metres/radians.
# agilexrobotics/pyAgxArm commit 841a625f5f4920e776f20b934eb13048b747e6d0, api/constants.py.
PIPERX_MDH = (
    (0.123, 0.0, 0.0, math.pi),
    (0.0, 0.0, math.pi / 2, 0.13578661580515886),
    (0.0, 0.28502999999999995, 0.0, 2.8380798966679794),
    (0.0, 0.27364, 0.0, 0.08063421144213803),
    (0.0, 0.07465999999999999, -math.pi / 2, math.pi / 2),
    (0.03526, 0.0, math.pi / 2, 0.0),
)
POSITION_TOLERANCE_M = 1e-5
ORIENTATION_TOLERANCE_DEG = 1e-3
LIMIT_ROUNDOFF_DEG = 1e-8


def _vector(value, size, name, code):
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise DomainError(code, f'{name} must contain {size} finite numbers.', 400) from exc
    if result.shape != (size,) or not np.all(np.isfinite(result)):
        raise DomainError(code, f'{name} must have exact shape ({size},) and finite values.', 400)
    return result


def _mdh_link_matrix(d, a, alpha, theta):
    """Rx(alpha) Tx(a) Rz(theta) Tz(d)."""
    ca, sa = math.cos(alpha), math.sin(alpha)
    ct, st = math.cos(theta), math.sin(theta)
    return np.array([[ct, -st, 0, a], [ca*st, ca*ct, -sa, -sa*d],
                     [sa*st, sa*ct, ca, ca*d], [0, 0, 0, 1]], dtype=float)


def _legal_equivalent(joints, seed):
    """Select equivalent legal angles; snap only tiny numerical excursions."""
    answer = []
    for value, reference, (low, high) in zip(joints, seed, JOINT_LIMITS_DEG):
        principal = math.remainder(float(value), 360.0)
        options = []
        first = math.ceil((low - LIMIT_ROUNDOFF_DEG - principal) / 360.0)
        last = math.floor((high + LIMIT_ROUNDOFF_DEG - principal) / 360.0)
        for turns in range(first, last + 1):
            angle = principal + turns * 360.0
            if low - LIMIT_ROUNDOFF_DEG <= angle < low:
                angle = float(low)
            elif high < angle <= high + LIMIT_ROUNDOFF_DEG:
                angle = float(high)
            if low <= angle <= high:
                options.append(angle)
        if not options:
            return None
        answer.append(min(options, key=lambda x: (abs(x - reference), x)))
    return np.asarray(answer)


class PiperKinematics:
    def __init__(self, tcp_offset_m=None, tcp_offset_rpy_deg=None):
        self._tcp_offset = np.eye(4)
        if tcp_offset_m is not None:
            self._tcp_offset[:3, 3] = _vector(tcp_offset_m, 3, 'tcp_offset_m', 'invalid_tcp')
        if tcp_offset_rpy_deg is not None:
            offset = _vector(tcp_offset_rpy_deg, 3, 'tcp_offset_rpy_deg', 'invalid_tcp')
            self._tcp_offset[:3, :3] = Rotation.from_euler('xyz', offset, degrees=True).as_matrix()

    def matrix(self, joints_deg):
        joints = np.deg2rad(_vector(joints_deg, 6, 'joints_deg', 'invalid_joints'))
        transform = np.eye(4)
        for q, (d, a, alpha, theta_offset) in zip(joints, PIPERX_MDH):
            transform = transform @ _mdh_link_matrix(d, a, alpha, q + theta_offset)
        return transform @ self._tcp_offset

    def pose(self, joints_deg):
        transform = self.matrix(joints_deg)
        with warnings.catch_warnings():
            # Euler coordinates are nonunique at gimbal lock; the matrix is valid.
            warnings.filterwarnings('ignore', message='Gimbal lock detected', category=UserWarning)
            rpy = Rotation.from_matrix(transform[:3, :3]).as_euler('xyz', degrees=True)
        return {'xyz_m': transform[:3, 3].tolist(), 'rpy_deg': rpy.tolist(), 'frame': 'base'}

    def solve(self, xyz_m, rpy_deg, seed_deg, *, nearby=False):
        target_xyz = _vector(xyz_m, 3, 'xyz_m', 'invalid_target')
        target_rpy = _vector(rpy_deg, 3, 'rpy_deg', 'invalid_target')
        seed = _vector(seed_deg, 6, 'seed_deg', 'invalid_joints')
        inverse_target = Rotation.from_euler('xyz', target_rpy, degrees=True).inv()

        def residual(q):
            transform = self.matrix(q)
            rotation_error = (inverse_target * Rotation.from_matrix(transform[:3, :3])).as_rotvec()
            return np.concatenate([transform[:3, 3] - target_xyz, rotation_error])

        seeds = [seed, np.array([0, 90, -90, 0, 0, 0]), np.array([0, 45, -45, 0, 0, 0]),
                 np.array([45, 90, -90, 0, 0, 0]), np.array([-45, 90, -90, 0, 0, 0])]
        candidates = []
        numerical_failures = 0
        for start in seeds:
            try:
                result = least_squares(residual, start, method='lm', max_nfev=1000,
                                       ftol=1e-9, xtol=1e-9)
            except (ValueError, FloatingPointError, OverflowError):
                numerical_failures += 1
                continue
            if result.x.shape != (6,) or not np.all(np.isfinite(result.x)):
                continue
            solution = _legal_equivalent(result.x, seed)
            if solution is None:
                continue
            # Recompute after periodic normalization and boundary-roundoff snapping.
            errors = residual(solution)
            position_error = float(np.linalg.norm(errors[:3]))
            orientation_error = float(np.rad2deg(np.linalg.norm(errors[3:])))
            if not (position_error <= POSITION_TOLERANCE_M and
                    orientation_error <= ORIENTATION_TOLERANCE_DEG):
                continue
            candidates.append((float(np.linalg.norm(solution - seed)), solution,
                               position_error, orientation_error))
            if nearby and np.max(np.abs(solution-seed))<=5:
                # A locally continuous, independently checked solution is sufficient
                # for one small linear waypoint; avoid four unrelated global seeds.
                break
        if not candidates:
            raise DomainError('ik_no_solution',
                'Finite multistart solver found no joint-limit-valid solution meeting '
                f'position <= {POSITION_TOLERANCE_M:g} m and orientation <= '
                f'{ORIENTATION_TOLERANCE_DEG:g} deg; numerical failures: {numerical_failures}. '
                'This does not prove the target globally unreachable.', 422)
        _, solution, position_error, orientation_error = min(candidates, key=lambda x: x[0])
        return {'joints_deg': solution.tolist(), 'position_error_m': position_error,
                'orientation_error_deg': orientation_error, 'solver': 'least_squares_lm'}
