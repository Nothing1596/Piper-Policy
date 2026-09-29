from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat
from .interaction_types import InteractionPolicy

Number = Annotated[float, Field(strict=True, allow_inf_nan=False)]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class Settings(StrictModel):
    backend: Literal["sim", "agx", "mujoco"] = "sim"
    managed_control: bool = Field(default=False, strict=True)
    managed_profile_id: str | None = None
    interaction_policy: InteractionPolicy | None = None
    simulation_seed: int = 0
    simulation_asset: Path | None = None
    host: str = "127.0.0.1"
    port: int = Field(default=8765, ge=0, le=65535)
    data_dir: Path = Path(".runtime")
    sdk_root: Path | None = None
    cando_source: Path | None = None
    can_interface: Literal["agx_cando", "socketcan"] = "agx_cando"
    can_channel: str = "0"
    cando_device_index: int | None = Field(default=None, ge=0, le=254, strict=True)
    firmware_profile: Literal["v189"] = "v189"
    control_profile: Literal["direct", "calibration"] = "direct"
    allow_motion: bool = Field(default=True, strict=True)
    connect_timeout_s: FiniteFloat = Field(default=2.0, gt=0, le=30)
    max_speed_percent: int = Field(default=1, ge=0, le=100)
    max_move_deg: FiniteFloat = Field(default=3.0, gt=0, le=30)
    max_velocity_deg_s: FiniteFloat = Field(default=3.0, gt=0, le=10)
    reference_deg_s: FiniteFloat = Field(default=0.5, gt=0, le=1)
    feedback_timeout_s: FiniteFloat = Field(default=0.15, gt=0, le=0.2)
    plan_ttl_s: FiniteFloat = Field(default=60, ge=1, le=120)
    gripper_max_m: FiniteFloat = Field(default=0.07, gt=0, le=0.09)
    gripper_effort_limit: FiniteFloat = Field(default=1.0, gt=0, le=1)
    tcp_offset_m: list[Number] = Field(default_factory=lambda: [0.0, 0.0, 0.0], min_length=3, max_length=3)
    tcp_offset_rpy_deg: list[Number] = Field(default_factory=lambda: [0.0, 0.0, 0.0], min_length=3, max_length=3)


class JointMove(StrictModel):
    kind: Literal["joint"] = "joint"
    joints_deg: list[Number] = Field(min_length=6, max_length=6)
    speed_percent: int = Field(default=5, strict=True, ge=0, le=100)
    timeout_s: Number = Field(default=30, gt=0)


class GripperMove(StrictModel):
    kind: Literal["gripper"] = "gripper"
    width_m: Number = Field(ge=0, le=0.09)
    effort_protocol: Number = Field(default=0.5, ge=0, le=32.767)
    timeout_s: Number = Field(default=10, gt=0)
    completion: Literal['width','bilateral_contact'] = 'width'


class ControlMode(StrictModel):
    """Fixed CAN/MOVE_J transition; no arbitrary mode or enable/reset fields."""
    kind: Literal["control_mode"] = "control_mode"
    speed_percent: int = Field(default=5, strict=True, ge=0, le=100)
    timeout_s: Number = Field(default=3, gt=0)


class ControlModeRequest(ControlMode):
    request_id: str = Field(min_length=8, max_length=128, pattern=r"^[A-Za-z0-9_.:-]+$")


Move = Annotated[JointMove | GripperMove | ControlMode, Field(discriminator="kind")]


class PreviewRequest(StrictModel):
    command: Move


class ExecuteRequest(StrictModel):
    plan_id: str = Field(min_length=1, max_length=80)
    request_id: str = Field(min_length=8, max_length=128, pattern=r"^[A-Za-z0-9_.:-]+$")


class MoveRequest(PreviewRequest):
    request_id: str = Field(min_length=8, max_length=128, pattern=r"^[A-Za-z0-9_.:-]+$")


class MoveTo(StrictModel):
    kind: Literal["move_to"] = "move_to"
    xyz_m: list[Number] = Field(min_length=3, max_length=3)
    rpy_deg: list[Number] | None = Field(default=None, min_length=3, max_length=3)
    speed_percent: int = Field(default=5, strict=True, ge=0, le=100)
    timeout_s: Number = Field(default=30, gt=0)


class MoveBy(StrictModel):
    kind: Literal["move_by"] = "move_by"
    delta_m: list[Number] = Field(min_length=3, max_length=3)
    frame: Literal["base", "tcp"] = "base"
    speed_percent: int = Field(default=5, strict=True, ge=0, le=100)
    timeout_s: Number = Field(default=30, gt=0)


class MoveLinear(StrictModel):
    """Fixed-orientation TCP line, preflighted at bounded sample spacing."""
    kind: Literal["move_linear"] = "move_linear"
    xyz_m: list[Number] = Field(min_length=3, max_length=3)
    step_m: Number = Field(default=0.002, ge=0.0005, le=0.005)
    speed_percent: int = Field(default=5, strict=True, ge=1, le=100)
    native_controller: bool = Field(default=False, strict=True)
    timeout_s: Number = Field(default=30, gt=0, le=120)


class Rotate(StrictModel):
    kind: Literal["rotate"] = "rotate"
    rpy_deg: list[Number] = Field(min_length=3, max_length=3)
    speed_percent: int = Field(default=5, strict=True, ge=0, le=100)
    timeout_s: Number = Field(default=30, gt=0)


class SetGripper(StrictModel):
    kind: Literal["set_gripper"] = "set_gripper"
    width_m: Number = Field(ge=0, le=0.09)
    effort_protocol: Number = Field(default=0.5, ge=0, le=32.767)
    timeout_s: Number = Field(default=10, gt=0)
    completion: Literal['width','bilateral_contact'] = 'width'


Primitive = Annotated[MoveTo | MoveBy | Rotate | SetGripper | MoveLinear, Field(discriminator="kind")]


class PrimitivePreviewRequest(StrictModel):
    command: Primitive


class RuntimeParameters(StrictModel):
    tcp_offset_m: list[Number] | None = Field(default=None, min_length=3, max_length=3)
    tcp_offset_rpy_deg: list[Number] | None = Field(default=None, min_length=3, max_length=3)
    payload: Literal["empty", "half", "full"] | None = None
    collision_rating: int | None = Field(default=None, strict=True, ge=0, le=8)
    joint_acc_rad_s2: Number | None = Field(default=None, ge=0.01, le=10)


class SimFault(StrictModel):
    fault: Literal["none", "stale", "collision", "tracking", "driver", "teaching"]


class PrimitiveRequest(StrictModel):
    command: Primitive
    request_id: str = Field(min_length=8, max_length=128, pattern=r"^[A-Za-z0-9_.:-]+$")


class LeaseRequest(StrictModel):
    duration_s: int = Field(default=120, strict=True, ge=5, le=600)
    joint_radius_deg: Number = Field(default=3.0, gt=0, le=30)
    allow_gripper: bool = Field(default=False, strict=True)


class ShutdownRequest(StrictModel):
    expected_instance_id: str = Field(strict=True, min_length=1, max_length=80)


class DomainError(Exception):
    def __init__(self, code: str, message: str, status: int = 409):
        super().__init__(message)
        self.code, self.message, self.status = code, message, status


JOINT_LIMITS_DEG = [(-150, 150), (0, 180), (-170, 0), (-89, 89), (-89, 89), (-180, 180)]
