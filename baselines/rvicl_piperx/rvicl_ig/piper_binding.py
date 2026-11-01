"""Bind RPent's actual RoboTwin primitives to the single-arm six-joint PiperX.

The inherited move_to/rotate_wrist/set_gripper/release implementations are used
unchanged. Only single-arm validation, six-joint radian planning/dispatch, and
gripper normalization against scene.gripper_max_m are adapted.
"""

from collections.abc import Mapping
from copy import deepcopy
import numpy as np

from .rpent_backend import primitive_specs as _base_primitive_specs, upstream


class _InvalidArmError(ValueError, KeyError):
    """Raised before I/O when a non-left arm is requested on single-arm PiperX."""


class _SingleArmRobotState(Mapping):
    """Lazily materialized single-arm robot_state mapping without second-arm keys."""

    def __init__(self, scene):
        self._scene = scene
        self._values = None

    def _load(self):
        if self._values is not None:
            return
        self._scene.check_cancelled()
        max_m = float(self._scene.gripper_max_m)
        if not np.isfinite(max_m) or max_m <= 0.0:
            raise ValueError("scene.gripper_max_m must be a positive finite float")
        state = self._scene.state()
        arms = state.get("arms") if isinstance(state, dict) else None
        if not isinstance(arms, dict) or set(arms) != {"left"}:
            raise ValueError("Single-arm PiperX scene.state() must expose only arms['left']")
        left = arms["left"]
        pose = np.asarray(left["tcp_pose_world"], dtype=np.float64)
        if pose.shape != (7,) or not np.isfinite(pose).all() or np.linalg.norm(pose[3:]) < 1e-10:
            raise ValueError("left tcp_pose_world must be 7 finite floats with nonzero quaternion")
        if "joints_rad" in left:
            joints = np.asarray(left["joints_rad"], dtype=np.float64)
            if joints.shape != (6,) or not np.isfinite(joints).all():
                raise ValueError("left joints_rad must be 6 finite radian floats")
        opening_m = float(left["gripper_opening_m"])
        if not np.isfinite(opening_m):
            raise ValueError("left gripper_opening_m must be finite")
        self._values = {"left_eef_pose": pose.tolist(),
                        "left_gripper": float(np.clip(opening_m / max_m, 0.0, 1.0))}

    def __getitem__(self, key):
        if key not in ("left_eef_pose", "left_gripper"):
            if isinstance(key, str) and key.endswith(("_eef_pose", "_gripper")):
                raise _InvalidArmError("Only arm='left' exists on the single PiperX adapter")
            raise KeyError(key)
        self._load()
        return self._values[key]

    def __iter__(self):
        return iter(("left_eef_pose", "left_gripper"))

    def __len__(self):
        return 2


def primitive_specs(root):
    specs = deepcopy(_base_primitive_specs(root))
    for spec in specs:
        spec["function"]["parameters"]["properties"]["arm"]["enum"] = ["left"]
        spec["function"]["description"] += " Single PiperX: arm='left' only; world equals robot base."
    return specs


class PiperEnvironment:
    """RPent environment boundary for a single six-joint PiperX in simulation."""

    def __init__(self, scene):
        self.scene = scene

    @property
    def terminated(self):
        return False

    @property
    def truncated(self):
        return bool(self.scene.truncated)

    def _episode_status(self):
        return {
            "take_action_cnt": int(self.scene.steps),
            "step_lim": int(self.scene.job["max_episode_steps"]),
            "actual_seed": int(self.scene.job["seed"]),
        }

    @property
    def last_info(self):
        return {
            "robot_state": _SingleArmRobotState(self.scene),
            "episode_status": self._episode_status(),
        }

    def plan_arm_path(self, arm, target):
        if arm != "left":
            raise _InvalidArmError("Only arm='left' exists on the single PiperX adapter")
        raw_target = np.asarray(target)
        if raw_target.dtype == bool:
            raise ValueError("Target must be seven finite xyz/wxyz values with a nonzero quaternion")
        target = np.asarray(target, dtype=np.float64)
        if target.shape != (7,) or not np.isfinite(target).all() or np.linalg.norm(target[3:]) < 1e-10:
            raise ValueError("Target must be seven finite xyz/wxyz values with a nonzero quaternion")
        planned = self.scene.plan_arm_path(target)
        if not isinstance(planned, dict):
            raise ValueError("scene.plan_arm_path must return a status mapping")
        if planned.get("status") != "Success" or planned.get("position") is None:
            return {"status": str(planned.get("status", "plan_failed")), "position": None}
        position = np.asarray(planned["position"], dtype=np.float64)
        if position.ndim != 2 or position.shape[0] < 1 or position.shape[1] != 6 or not np.isfinite(position).all():
            raise ValueError("PiperX planned position must be a finite array of shape (n, 6)")
        return {"status": "Success", "position": position}


def create_primitives(scene, root):
    base, _ = upstream(root)

    class PiperPrimitives(base):
        """Six-joint single-arm PiperX binding; inherit the four RPent primitive bodies."""

        @staticmethod
        def _validate_qpos_updates_request(updates):
            if not isinstance(updates, list) or not updates:
                raise ValueError("qpos updates must be a nonempty list")
            normalized = []
            for update in updates:
                if not isinstance(update, dict):
                    raise ValueError("Each qpos update must be a dict")
                if set(update) - {"arm", "arm_qpos", "gripper"}:
                    raise ValueError("Unexpected keys in qpos update")
                if update.get("arm") != "left":
                    raise _InvalidArmError("Only arm='left' exists on the single PiperX adapter")
                item = {"arm": "left"}
                if update.get("arm_qpos") is not None:
                    raw_qpos = np.asarray(update["arm_qpos"])
                    if raw_qpos.dtype == bool:
                        raise ValueError("PiperX arm_qpos must contain six finite radian joints")
                    qpos = np.asarray(update["arm_qpos"], dtype=np.float64)
                    if qpos.shape != (6,) or not np.isfinite(qpos).all():
                        raise ValueError("PiperX arm_qpos must contain six finite radian joints")
                    item["arm_qpos"] = qpos
                if update.get("gripper") is not None:
                    raw_gripper = update["gripper"]
                    if isinstance(raw_gripper, (bool, np.bool_)) or np.ndim(raw_gripper) != 0:
                        raise ValueError("PiperX normalized gripper must be finite and within 0..1")
                    value = float(raw_gripper)
                    # Upstream interpolation of a legal zero can produce -1 ULP.
                    # Snap machine roundoff only, never a physical target error.
                    epsilon = 8 * np.finfo(np.float64).eps
                    if -epsilon <= value < 0.:
                        value = 0.
                    elif 1. < value <= 1. + epsilon:
                        value = 1.
                    if not np.isfinite(value) or not (0.0 <= value <= 1.0):
                        raise ValueError("PiperX normalized gripper must be finite and within 0..1")
                    item["gripper"] = value
                if len(item) == 1:
                    raise ValueError("Update must set arm_qpos and/or gripper")
                normalized.append(item)
            return normalized

        def apply_qpos_updates(self, updates):
            normalized = self._validate_qpos_updates_request(updates)
            self._check_cancelled()
            result = scene.apply_updates(normalized)
            status = result.get("episode_status") if isinstance(result, dict) else None
            episode_status = (
                {
                    "take_action_cnt": int(status["take_action_cnt"]),
                    "step_lim": int(status["step_lim"]),
                    "actual_seed": int(status["actual_seed"]),
                }
                if isinstance(status, dict) and {"take_action_cnt", "step_lim", "actual_seed"} <= set(status)
                else self.env._episode_status()
            )
            out = {
                "action_type": "qpos",
                "requested_actions": int(result.get("requested_actions", len(normalized))),
                "executed_actions": int(result.get("executed_actions", 0)),
                "episode_status": episode_status,
            }
            if isinstance(result, dict) and result.get("error") is not None:
                out["error"] = result["error"]
            return out

    return PiperPrimitives(
        env=PiperEnvironment(scene),
        model=None,
        seed=scene.job["seed"],
        check_cancelled=scene.check_cancelled,
    )
