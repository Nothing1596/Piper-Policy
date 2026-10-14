"""Compact analytic task-space primitives resolved against measured feedback.

Inspired by the Harness VLA primitive set (arXiv:2607.08448, Appendix B): four
primitives covering absolute position, relative translation, absolute
orientation, and gripper width. Targets are base-frame metres; orientations are
absolute extrinsic xyz RPY in degrees (R = Rz @ Ry @ Rx), not the RPent
simulator's pitch convention.

Every TCP primitive is solved by the hard-coded PiperKinematics solver seeded
with measured joints and executed as a JointMove: the arm reaches the target
TCP pose in joint space, without guaranteeing a Cartesian straight line. There is no
collision checking, no step limit, and no grasp-success claim for the
continuous gripper width.
"""
from __future__ import annotations

import numpy as np
from pydantic import ValidationError

from .models import (DomainError, GripperMove, JointMove, MoveBy, MoveTo,
                     MoveLinear, Rotate, SetGripper, Settings)


def _kinematics(settings: Settings):
    # Keep model/transport imports independent of the numerical solver.
    from .kinematics import PiperKinematics
    return PiperKinematics(tcp_offset_m=list(settings.tcp_offset_m),
                           tcp_offset_rpy_deg=list(settings.tcp_offset_rpy_deg))


def resolve_primitive(command: MoveTo | MoveBy | Rotate | SetGripper, q_deg: list[float],
                      settings: Settings) -> tuple[JointMove | GripperMove, dict]:
    """Resolve one primitive against measured joints into an executable move plus metadata."""
    if isinstance(command, SetGripper):
        move = GripperMove(width_m=command.width_m, effort_protocol=command.effort_protocol,
                           timeout_s=command.timeout_s,completion=command.completion)
        return move, {"primitive": command.model_dump(),
                      "target_gripper_width_m": command.width_m,
                      "path_semantics": "gripper_width"}
    kin = _kinematics(settings)
    measured = kin.pose(q_deg)
    if isinstance(command, MoveLinear):
        if command.native_controller and settings.control_profile == "calibration":
            raise DomainError("unsupported_profile", "Native MOVE_L requires the direct profile; calibration uses bounded samples.", 422)
        distance = float(np.linalg.norm(np.asarray(command.xyz_m) - measured["xyz_m"]))
        count = max(1, int(np.ceil(distance / command.step_m)))
        if count > 100:
            raise DomainError("path_budget", "Line exceeds 100 samples; split the task explicitly.", 422)
        points, seed = [], list(q_deg)
        for i in range(1, count+1):
            xyz = [(b-a)*i/count+a for a,b in zip(measured["xyz_m"], command.xyz_m)]
            solved = kin.solve(xyz, measured["rpy_deg"], seed,nearby=True)
            point = [float(v) for v in solved["joints_deg"]]
            if max(abs(a-b) for a,b in zip(point, seed)) > 5:
                raise DomainError("path_discontinuity", "IK branch jumps more than 5 degrees between samples.", 422)
            points.append(point)
            seed = point
        from .kinematics import PiperKinematics
        return JointMove(joints_deg=seed, speed_percent=command.speed_percent, timeout_s=command.timeout_s), {
            "primitive": command.model_dump(), "waypoints_deg": points,
            "measured_tcp_pose": measured, "target_tcp_pose": {"xyz_m": command.xyz_m, "rpy_deg": measured["rpy_deg"]},
            "path_semantics": "controller_move_l" if command.native_controller and settings.backend == "agx" else "sampled_cartesian_line_joint_interpolation",
            "native_controller": command.native_controller and settings.backend == "agx",
            "target_flange_pose": PiperKinematics().pose(seed), "step_m": command.step_m,
            "collision_checking": False, "continuous_line_accuracy_verified": False}
    if isinstance(command, MoveTo):
        xyz_m = list(command.xyz_m)
        rpy_deg = list(command.rpy_deg) if command.rpy_deg is not None else list(measured["rpy_deg"])
    elif isinstance(command, MoveBy):
        delta = list(command.delta_m)
        if command.frame == "tcp":
            rotation = np.asarray(kin.matrix(q_deg), dtype=float)[:3, :3]
            delta = (rotation @ np.asarray(delta, dtype=float)).tolist()
        xyz_m = [m + d for m, d in zip(measured["xyz_m"], delta)]
        rpy_deg = list(measured["rpy_deg"])
    elif isinstance(command, Rotate):
        xyz_m = list(measured["xyz_m"])
        rpy_deg = list(command.rpy_deg)
    else:
        raise DomainError("invalid_request", "Unknown primitive kind.", 422)
    result = kin.solve(xyz_m, rpy_deg, list(q_deg))
    try:
        move = JointMove(joints_deg=[float(v) for v in result["joints_deg"]],
                         speed_percent=command.speed_percent, timeout_s=command.timeout_s)
    except (KeyError, TypeError, ValueError, ValidationError) as exc:
        raise DomainError("solver_failed", "The analytic solver did not return six finite joint angles.", 422) from exc
    return move, {"primitive": command.model_dump(), "measured_tcp_pose": measured,
                  "target_tcp_pose": {"xyz_m": xyz_m, "rpy_deg": rpy_deg, "frame": "base"},
                  "solver": result, "path_semantics": "joint_space_endpoint"}
