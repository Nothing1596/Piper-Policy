from __future__ import annotations

import math
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Protocol

from .models import DomainError


@dataclass
class State:
    connected: bool
    q_deg: list[float] | None = None
    velocity_deg_s: list[float] | None = None
    enabled: list[bool] | None = None
    ctrl_mode: int | None = None
    motion_mode: int | None = None
    teach_status: int | None = None
    arm_status: int | None = None
    error_code: int | None = None
    feedback_age_s: float | None = None
    gripper_width_m: float | None = None
    gripper_enabled: bool = False
    gripper_error: bool = False
    gripper_age_s: float | None = None
    gripper_mode: str | None = None
    received_frames: int = 0
    tx_frames: int = 0
    diagnostics: dict = field(default_factory=dict)
    feedback_stamp_s: float | None = None
    gripper_stamp_s: float | None = None
    collision_status: list[bool] | None = None
    motor_telemetry: list[dict] | None = None

    def public(self):
        return asdict(self)


class Backend(Protocol):
    name: str
    def connect(self) -> None: ...
    def close(self) -> None: ...
    def snapshot(self) -> State: ...
    def begin_joint(self, current_deg: list[float], speed_percent: int) -> None: ...
    def set_control_mode(self, current_deg: list[float], speed_percent: int, checkpoint) -> None: ...
    def joint_target(self, joints_deg: list[float]) -> None: ...
    def gripper_target(self, width_m: float, effort_protocol: float) -> None: ...


class SimBackend:
    """Deterministic kinematic stand-in; no CAN, no collision or dynamics claim."""
    name = "sim"

    def __init__(self):
        self.lock = threading.RLock()
        self.connected = False
        self.q = [0.0] * 6
        self.width = 0.02
        self.commands = []
        self.fault = None
        self.teach = False
        self.ctrl_mode, self.motion_mode = 1, 1
        self.stale = False
        self.follow = True
        self.estopped = False
        self.collision = False
        self.runtime_parameters = {}

    def connect(self):
        self.connected = True

    def close(self):
        self.connected = False

    def snapshot(self):
        with self.lock:
            if not self.connected:
                return State(False)
            state = State(True, self.q.copy(), [0.] * 6, [True] * 6,
                         2 if self.teach else self.ctrl_mode, self.motion_mode, 1 if self.teach else 0,
                         1 if self.fault else 0, 1 if self.fault else 0,
                         1.0 if self.stale else 0.001, self.width, True, False,
                         1.0 if self.stale else 0.001, "width", 100, len(self.commands),
                         {"simulation": True, "physical_validation": False})
            state.feedback_stamp_s = state.gripper_stamp_s = time.monotonic() - state.feedback_age_s
            state.collision_status = [self.collision] * 6
            state.motor_telemetry = [{"joint": i, "motor_temp_c": 25.0, "foc_temp_c": 25.0,
                                      "bus_current_a": 0.0, "source": "synthetic"} for i in range(1, 7)]
            state.diagnostics["software_estop_latched"] = self.estopped
            return state

    def emergency_stop(self):
        with self.lock:
            self.estopped = True
            self.commands.append(("estop",))

    def _write_guard(self):
        if self.estopped:
            raise DomainError("estop_latched", "Operator must clear the software stop latch.")

    def begin_joint(self, current_deg, speed_percent):
        with self.lock:
            self._write_guard()
            self.commands.append(("begin", list(current_deg), speed_percent))

    def set_control_mode(self, current_deg, speed_percent, checkpoint):
        with self.lock:
            checkpoint()
            self._write_guard()
            self.commands.append(("control_mode", list(current_deg), speed_percent))
            if self.follow:
                self.ctrl_mode, self.motion_mode = 1, 1

    def joint_target(self, joints_deg):
        with self.lock:
            self._write_guard()
            self.commands.append(("joint", list(joints_deg)))
            if self.follow:
                self.q = list(joints_deg)

    def gripper_target(self, width_m, effort_protocol):
        with self.lock:
            self._write_guard()
            self.commands.append(("gripper", width_m, effort_protocol))
            if self.follow:
                self.width = width_m


def check_state(s: State, timeout: float, *, gripper: bool = False, moving: bool = False,
                max_velocity: float | None = 3, require_position_mode: bool = True):
    if not s.connected:
        raise DomainError("not_connected", "Connect the robot first.")
    if s.diagnostics.get("software_estop_latched"):
        raise DomainError("estop_latched", "Software emergency stop is latched.")
    if s.collision_status and any(s.collision_status):
        raise DomainError("collision_detected", "Controller collision feedback is active.")
    if s.feedback_age_s is None or not math.isfinite(s.feedback_age_s) or not 0 <= s.feedback_age_s <= timeout:
        raise DomainError("stale_feedback", "Required CAN feedback is missing or stale; no motion is admitted.")
    if s.feedback_stamp_s is None or not math.isfinite(s.feedback_stamp_s):
        raise DomainError("invalid_feedback", "Feedback source timestamp is missing or nonfinite.")
    if s.q_deg is None or len(s.q_deg) != 6 or not all(math.isfinite(x) for x in s.q_deg):
        raise DomainError("invalid_feedback", "Six finite joint angles are required.")
    # Vendor enums 2/6 are STOP_RECORDING/TERMINATE_EXECUTION, not active teaching.
    # They are admissible only alongside CAN/MOVE_J and all other readiness checks.
    if s.teach_status not in (0, 2, 6) or (require_position_mode and (s.ctrl_mode != 1 or s.motion_mode != 1)):
        raise DomainError("control_mode", "Require CAN position control / MOVE_J. Use robot_set_control_mode for an explicit transition; active teaching requires operator handling.")
    if s.arm_status != 0 or s.error_code != 0 or s.enabled != [True] * 6 or s.diagnostics.get("communication_error") or s.diagnostics.get("driver_fault"):
        raise DomainError("robot_not_ready", "All joints must already be enabled with no reported faults.")
    velocities = s.velocity_deg_s
    bound = max_velocity if moving else 0.5
    if velocities is None or len(velocities) != 6 or any(not math.isfinite(x) or (max_velocity is not None and abs(x) > bound) for x in velocities):
        raise DomainError("velocity_limit", "Measured joint velocity is missing or exceeds the admission limit.")
    if gripper and (s.gripper_mode != "width" or not s.gripper_enabled or s.gripper_error or
                    s.gripper_width_m is None or not math.isfinite(s.gripper_width_m) or
                    s.gripper_age_s is None or not math.isfinite(s.gripper_age_s) or not 0 <= s.gripper_age_s <= timeout or
                    s.gripper_stamp_s is None or not math.isfinite(s.gripper_stamp_s)):
        raise DomainError("gripper_not_ready", "Fresh, enabled, fault-free width-mode gripper feedback is required.")
