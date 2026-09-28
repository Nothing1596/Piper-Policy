"""Shared single-terminal contracts; operator policy is separate from model input."""
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field

class InteractionModel(BaseModel):
    model_config = ConfigDict(extra='forbid', validate_assignment=True)

class ExecutionLimits(InteractionModel):
    joint_lower_deg: list[float] | None = Field(default=None, min_length=6, max_length=6, allow_inf_nan=False)
    joint_upper_deg: list[float] | None = Field(default=None, min_length=6, max_length=6, allow_inf_nan=False)
    max_speed_percent: int = Field(default=100, strict=True, ge=0, le=100)
    gripper_min_m: float = Field(default=0., ge=0, le=.09, allow_inf_nan=False)
    gripper_max_m: float = Field(default=.07, ge=0, le=.09, allow_inf_nan=False)
    max_effort_protocol: float = Field(default=32.767, ge=0, le=32.767, allow_inf_nan=False)

class AutoApproval(InteractionModel):
    max_joint_step_deg: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    max_tcp_step_m: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    max_speed_percent: int | None = Field(default=None, strict=True, ge=0, le=100)
    max_effort_protocol: float | None = Field(default=None, ge=0, le=32.767, allow_inf_nan=False)

class InteractionPolicy(InteractionModel):
    mode: Literal['always','risk','auto'] = 'risk'
    limits: ExecutionLimits = Field(default_factory=ExecutionLimits)
    automatic: AutoApproval = Field(default_factory=AutoApproval)
    version: int = Field(default=1, ge=1, strict=True)

@dataclass
class RuntimeConnection:
    url: str
    model_token_file: Path
    operator_token_file: Path
    instance_id: str
    owned: bool
    profile_root: Path
    profile_id: str
    mode: str
    target: str
