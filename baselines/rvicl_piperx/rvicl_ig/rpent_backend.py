"""Bind RPent's actual RoboTwin dual-arm primitives to seven-joint IG Pandas.

The inherited move_to/rotate_wrist/set_gripper/release implementations are used
unchanged. Only native planning, joint count, and action dispatch are adapted.
"""

import sys
from functools import lru_cache
from pathlib import Path

import numpy as np


@lru_cache(maxsize=2)
def upstream(root):
    root = Path(root).resolve()
    if not (root / "robots/robotwin/primitives.py").is_file():
        raise FileNotFoundError(f"Missing RPent dual-arm primitives in {root}")
    sys.path.insert(0, str(root))
    from robots.robotwin.primitives import RoboTwinPrimitives
    from robots.robotwin.tools import TOOLS_SPEC
    if Path(sys.modules[RoboTwinPrimitives.__module__].__file__).resolve() != root / "robots/robotwin/primitives.py":
        raise RuntimeError("A different RPent checkout was already imported")
    return RoboTwinPrimitives, TOOLS_SPEC


def primitive_specs(root):
    _, specs = upstream(root)
    names = {"move_to", "rotate_wrist", "set_gripper", "release"}
    return [{"type": "function", "function": {"name": s["name"], "description": s["description"],
             "parameters": s["input_schema"]}} for s in specs if s["name"] in names]


class PandaEnvironment:
    """RPent's native environment boundary, with private benchmark scores omitted."""

    def __init__(self, scene):
        self.scene = scene

    @property
    def terminated(self):
        return False  # Official IG runs ignore goal termination and reach the horizon.

    @property
    def truncated(self):
        return self.scene.truncated

    @property
    def last_info(self):
        state = self.scene.state()
        robot = {}
        for side, arm in state["arms"].items():
            robot[f"{side}_eef_pose"] = arm["tcp_pose_world"]
            robot[f"{side}_gripper"] = float(np.clip(arm["gripper_opening_m"]/.04, 0, 1))
        return {"robot_state": robot, "episode_status": {
            "take_action_cnt": self.scene.steps, "step_lim": self.scene.job["max_episode_steps"],
            "actual_seed": self.scene.job["seed"]}}

    def plan_arm_path(self, arm, target):
        if arm not in {"left", "right"}:
            raise ValueError("arm must be left or right")
        try:
            paths = self.scene._plan([{arm: list(target)}])[0]
            return {"status": "Success", "position": paths[self.scene.mapping[arm]]}
        except ValueError as error:
            return {"status": str(error), "position": None}


def create_primitives(scene, root):
    base, _ = upstream(root)

    class PandaPrimitives(base):
        """Seven-joint native binding; inherit the four RPent primitive bodies."""

        @staticmethod
        def _validate_qpos_updates_request(updates):
            if not isinstance(updates, list) or not updates:
                raise ValueError("qpos updates must be a nonempty list")
            normalized = []
            for update in updates:
                if not isinstance(update, dict) or update.get("arm") not in {"left", "right"}:
                    raise ValueError("Each update requires a valid arm")
                item = {"arm": update["arm"]}
                if update.get("arm_qpos") is not None:
                    qpos = np.asarray(update["arm_qpos"], dtype=float)
                    if qpos.shape != (7,) or not np.isfinite(qpos).all():
                        raise ValueError("Panda arm_qpos must contain seven finite joints")
                    item["arm_qpos"] = qpos
                if update.get("gripper") is not None:
                    value = float(update["gripper"])
                    if not np.isfinite(value) or not 0 <= value <= 1:
                        raise ValueError("Panda normalized gripper must be 0..1")
                    item["gripper"] = value
                if len(item) == 1:
                    raise ValueError("Update must set joints and/or gripper")
                normalized.append(item)
            return normalized

        def apply_qpos_updates(self, updates):
            updates = self._validate_qpos_updates_request(updates)
            executed = 0
            for update in updates:
                if scene.truncated:
                    break
                targets = scene.joints()
                i = scene.mapping[update["arm"]]
                if "arm_qpos" in update:
                    targets[i] = update["arm_qpos"]
                if "gripper" in update:
                    scene.grippers[i] = 2*update["gripper"] - 1
                scene.step(targets)
                executed += 1
            return {"action_type": "qpos", "requested_actions": len(updates),
                    "executed_actions": executed, "episode_status": self.env.last_info["episode_status"]}

        def move_both(self, left, right, substeps=0):
            """Synchronized composition using the same RPent environment planner."""
            if type(substeps) is not int or substeps < 0:
                raise ValueError("substeps must be nonnegative")
            starts, paths = scene.joints(), []
            for i in (0, 1):
                side = next(side for side,j in scene.mapping.items() if j == i)
                target = left if side == "left" else right
                if target is None:
                    paths.append(starts[i][None, :])
                    continue
                plan = self.env.plan_arm_path(side, target)
                if plan["status"] != "Success":
                    return {"completed": False, "stop_reason": "plan_failed", "plan_status": plan["status"]}
                paths.append(np.asarray(plan["position"]))
            count = max(map(len, paths))
            if substeps:
                count = min(count, substeps)
            executed = 0
            for t in range(count):
                if scene.truncated:
                    break
                targets = []
                for path in paths:
                    index = (t+1)*(len(path)-1)/count
                    low, high = int(index), min(int(index)+1, len(path)-1)
                    targets.append(path[low]*(1-index+low) + path[high]*(index-low))
                scene.step(targets)
                executed += 1
            self.native_actions += executed
            return {"completed": count == executed, "executed_steps": executed,
                    "episode_status": self.env.last_info["episode_status"]}

    return PandaPrimitives(env=PandaEnvironment(scene), model=None, seed=scene.job["seed"],
                           check_cancelled=lambda: None)
