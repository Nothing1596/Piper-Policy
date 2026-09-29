"""Shared single-terminal contracts; operator policy is separate from model input."""
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator

FiniteNumber = Annotated[float, Field(strict=True, allow_inf_nan=False)]

class InteractionModel(BaseModel):
    model_config = ConfigDict(extra='forbid', validate_assignment=True)

class ExecutionLimits(InteractionModel):
    joint_lower_deg: list[FiniteNumber] | None = Field(default=None, min_length=6, max_length=6)
    joint_upper_deg: list[FiniteNumber] | None = Field(default=None, min_length=6, max_length=6)
    max_speed_percent: int = Field(default=100, strict=True, ge=0, le=100)
    gripper_min_m: FiniteNumber = Field(default=0., ge=0, le=.09, allow_inf_nan=False)
    gripper_max_m: FiniteNumber = Field(default=.07, ge=0, le=.09, allow_inf_nan=False)
    max_effort_protocol: FiniteNumber = Field(default=32.767, ge=0, le=32.767, allow_inf_nan=False)

    @model_validator(mode='after')
    def ordered_limits(self):
        if self.gripper_min_m > self.gripper_max_m:
            raise ValueError('Gripper minimum must not exceed maximum')
        if self.joint_lower_deg is not None and self.joint_upper_deg is not None:
            if any(lo >= hi for lo, hi in zip(self.joint_lower_deg, self.joint_upper_deg)):
                raise ValueError('Each joint lower bound must be below its upper bound')
        return self

class AutoApproval(InteractionModel):
    max_joint_step_deg: FiniteNumber | None = Field(default=None, gt=0, allow_inf_nan=False)
    max_tcp_step_m: FiniteNumber | None = Field(default=None, gt=0, allow_inf_nan=False)
    max_speed_percent: int | None = Field(default=None, strict=True, ge=0, le=100)
    max_effort_protocol: FiniteNumber | None = Field(default=None, ge=0, le=32.767, allow_inf_nan=False)

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

class SessionAcquire(InteractionModel):
    shutdown_on_loss: bool = Field(default=False, strict=True)
    owner: str = Field(min_length=1, max_length=128)

class SessionReference(InteractionModel):
    session_id: str = Field(min_length=1, max_length=128)

class ApprovalDecision(InteractionModel):
    approved: bool = Field(strict=True)

class PolicyUpdate(InteractionModel):
    policy: InteractionPolicy

class ProfileSettingsUpdate(InteractionModel):
    changes: dict
