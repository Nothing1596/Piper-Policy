"""Bridge the upstream action layout into the guarded preview API.

Upstream clients.py and features.py are unmodified MIT-licensed copies.
No synthetic second arm or camera is invented for the single-arm installation.
"""
import math
from typing import Literal

from pydantic import Field, model_validator

from .models import GripperMove, JointMove, Number, StrictModel
from .vendor.piper_aio.features import FEATURES

UPSTREAM = "https://github.com/innovator-zero/piper-aio"
REVISION = "49f83278181f82b36fa4724d329f8a91e76ac605"


class AioAction(StrictModel):
    action: list[Number] = Field(min_length=7, max_length=14)
    layout: Literal["single", "dual"]
    arm_side: Literal["single", "left", "right"]
    component: Literal["joints", "gripper"]
    speed_percent: int = Field(default=1, strict=True, ge=1, le=5)
    effort_protocol: Number = Field(default=0.5, gt=0, le=1)

    @model_validator(mode="after")
    def explicit_layout(self):
        if self.layout == "single" and (len(self.action) != 7 or self.arm_side != "single"):
            raise ValueError("single layout requires exactly 7 values and arm_side=single")
        if self.layout == "dual" and (len(self.action) != 14 or self.arm_side == "single"):
            raise ValueError("dual layout requires 14 values and an explicit left/right selection")
        return self

    def to_command(self):
        start = 7 if self.arm_side == "right" else 0
        row = self.action[start:start+7]
        if self.component == "joints":
            return JointMove(joints_deg=[math.degrees(v) for v in row[:6]], speed_percent=self.speed_percent)
        return GripperMove(width_m=row[6], effort_protocol=self.effort_protocol)


def describe():
    return {"repository": UPSTREAM, "revision": REVISION,
            "reused_files": ["inference/clients.py", "convert_data/features.py"],
            "upstream_features": FEATURES,
            "joint_units_in_action": "radians", "gripper_units_in_action": "metres",
            "single_arm_layout": "j1..j6 radians, gripper width metres",
            "dual_arm_layout": "left seven values, then right seven values; select arm explicitly",
            "installed_hardware": "one arm; no fabricated second arm or camera observation",
            "execution": "one component preview; not a synchronized dual-arm or real-time chunk executor",
            "openpi_client": "piperx_middleware.vendor.piper_aio.clients.OpenpiClient (requires upstream openpi_client and numpy)",
            "camera_capture_integrated": False, "ros_node_deployed": False}
