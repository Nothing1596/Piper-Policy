from pathlib import Path
import os

import numpy as np
import pytest

from rvicl_ig.agent import tool_specs
from rvicl_ig.piper_binding import create_primitives, primitive_specs, upstream

RPENT = Path(os.environ.get("RPENT_ROOT", Path(__file__).resolve().parents[1] / "deps/RPent"))
pytestmark = pytest.mark.skipif(not RPENT.is_dir(), reason="RPent checkout unavailable")


def _qmult(left, right):
    w1, x1, y1, z1 = left
    w2, x2, y2, z2 = right
    return np.asarray(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


class FakePiperScene:
    robot_profile = "piperx"

    def __init__(
        self,
        *,
        gripper_max_m=0.07,
        gripper_opening_m=0.0175,
        tcp_pose_world=None,
        joints_rad=None,
        planned_path=None,
        plan_status="Success",
        max_episode_steps=50,
        seed=40194941,
    ):
        self.job = {"seed": seed, "max_episode_steps": max_episode_steps}
        self.steps = 0
        self.gripper_max_m = float(gripper_max_m)
        self.gripper_opening_m = float(gripper_opening_m)
        self.tcp_pose_world = list(
            tcp_pose_world if tcp_pose_world is not None else [0.25, -0.05, 0.18, 1.0, 0.0, 0.0, 0.0]
        )
        self.joints_rad = np.asarray(
            joints_rad if joints_rad is not None else [0.1, -0.2, 0.3, -0.1, 0.2, -0.3],
            dtype=np.float64,
        )
        self.planned_path = (
            np.asarray(planned_path, dtype=np.float64)
            if planned_path is not None
            else np.stack(
                [self.joints_rad + 0.01 * i for i in range(1, 7)],
                axis=0,
            )
        )
        self.plan_status = plan_status
        self.cancel_error = None
        self.cancelled_checks = 0
        self.state_calls = 0
        self.plan_calls = []
        self.apply_calls = []

    @property
    def truncated(self):
        return self.steps >= self.job["max_episode_steps"]

    def check_cancelled(self):
        self.cancelled_checks += 1
        if self.cancel_error is not None:
            raise self.cancel_error

    def state(self):
        self.state_calls += 1
        return {
            "arms": {
                "left": {
                    "tcp_pose_world": list(self.tcp_pose_world),
                    "joints_rad": self.joints_rad.tolist(),
                    "gripper_opening_m": self.gripper_opening_m,
                }
            }
        }

    def plan_arm_path(self, target_wxyz7):
        target = np.asarray(target_wxyz7, dtype=np.float64).copy()
        self.plan_calls.append(target)
        if self.plan_status != "Success":
            return {"status": self.plan_status, "position": None}
        return {"status": "Success", "position": self.planned_path.copy()}

    def apply_updates(self, normalized_updates):
        batch = []
        for item in normalized_updates:
            copied = {"arm": item["arm"]}
            if "arm_qpos" in item:
                copied["arm_qpos"] = np.asarray(item["arm_qpos"], dtype=np.float64).copy()
            if "gripper" in item:
                copied["gripper"] = float(item["gripper"])
            batch.append(copied)
        self.apply_calls.append(batch)
        executed = 0
        for item in batch:
            if self.truncated:
                break
            if "arm_qpos" in item:
                self.joints_rad = item["arm_qpos"].copy()
            if "gripper" in item:
                self.gripper_opening_m = item["gripper"] * self.gripper_max_m
            self.steps += 1
            executed += 1
        if self.plan_calls and executed > 0 and any("arm_qpos" in u for u in batch):
            self.tcp_pose_world = self.plan_calls[-1].tolist()
        return {
            "requested_actions": len(batch),
            "executed_actions": executed,
            "episode_status": {
                "take_action_cnt": self.steps,
                "step_lim": self.job["max_episode_steps"],
                "actual_seed": self.job["seed"],
                "eval_success": True,
                "private_reward": 1.0,
            },
        }


def test_inherits_actual_rpent_primitive_methods_and_single_arm_specs():
    scene = FakePiperScene()
    primitives = create_primitives(scene, str(RPENT))
    base, _ = upstream(str(RPENT))
    for name in ("move_to", "rotate_wrist", "set_gripper", "release"):
        assert getattr(primitives, name).__func__ is getattr(base, name)
    assert isinstance(primitives, base)
    assert primitives.model is None
    assert primitives.seed == scene.job["seed"]
    assert primitives._check_cancelled == scene.check_cancelled
    assert not hasattr(primitives, "move_both")

    specs = primitive_specs(str(RPENT))
    assert {s["function"]["name"] for s in specs} == {
        "move_to",
        "rotate_wrist",
        "set_gripper",
        "release",
    }
    for spec in specs:
        props = spec["function"]["parameters"]["properties"]
        assert "arm" in props
        assert props["arm"]["enum"] == ["left"]
        assert "arm" in spec["function"]["parameters"]["required"]

    all_specs = tool_specs(str(RPENT), sam_enabled=False, robot_profile="piperx")
    names = {s["function"]["name"] for s in all_specs}
    assert "move_both" not in names


def test_last_info_exposes_only_single_arm_and_normalizes_gripper_without_gt_leak():
    scene = FakePiperScene(gripper_max_m=0.07, gripper_opening_m=0.0175)
    primitives = create_primitives(scene, str(RPENT))
    info = primitives.env.last_info

    assert set(info) == {"robot_state", "episode_status"}
    assert set(info["robot_state"].keys()) == {"left_eef_pose", "left_gripper"}
    assert "right_eef_pose" not in info["robot_state"]
    assert "right_gripper" not in info["robot_state"]
    assert info["robot_state"]["left_eef_pose"] == scene.tcp_pose_world
    assert info["robot_state"]["left_gripper"] == pytest.approx(0.25)

    assert set(info["episode_status"].keys()) == {
        "take_action_cnt",
        "step_lim",
        "actual_seed",
    }
    assert "eval_success" not in info["episode_status"]

    exec_result = primitives.set_gripper(arm="left", val=0.8, steps=4)
    assert exec_result["completed"] is True
    assert exec_result["gripper_val"] == pytest.approx(0.8)
    assert scene.gripper_opening_m == pytest.approx(0.8 * 0.07)
    assert set(exec_result["episode_status"].keys()) == {
        "take_action_cnt",
        "step_lim",
        "actual_seed",
    }
    assert "eval_success" not in exec_result["episode_status"]
    assert "private_reward" not in exec_result["episode_status"]


def test_rejects_wrong_arm_before_any_io():
    scene = FakePiperScene()
    primitives = create_primitives(scene, str(RPENT))

    with pytest.raises(ValueError, match="left"):
        primitives.move_to(arm="right", xyz=[0.3, 0.0, 0.2], quat=[1.0, 0.0, 0.0, 0.0])
    assert scene.state_calls == 0
    assert len(scene.plan_calls) == 0
    assert len(scene.apply_calls) == 0

    with pytest.raises(ValueError, match="left"):
        primitives.move_to(arm="right", xyz=[0.3, 0.0, 0.2])
    assert scene.state_calls == 0
    assert len(scene.plan_calls) == 0
    assert len(scene.apply_calls) == 0

    with pytest.raises(ValueError, match="left"):
        primitives.rotate_wrist(arm="right", delta_yaw_deg=20.0)
    assert scene.state_calls == 0
    assert len(scene.plan_calls) == 0
    assert len(scene.apply_calls) == 0

    with pytest.raises(ValueError, match="left"):
        primitives.set_gripper(arm="right", val=0.5, steps=3)
    assert scene.state_calls == 0
    assert len(scene.apply_calls) == 0

    with pytest.raises(ValueError, match="left"):
        primitives.release(arm="right", steps=3)
    assert scene.state_calls == 0
    assert len(scene.apply_calls) == 0

    with pytest.raises(ValueError, match="left"):
        primitives.env.plan_arm_path("right", [0.3, 0.0, 0.2, 1.0, 0.0, 0.0, 0.0])
    assert scene.state_calls == 0
    assert len(scene.plan_calls) == 0


@pytest.mark.parametrize(
    "bad_batch",
    [
        [],
        [{"arm": "left"}],
        [{"arm": "right", "gripper": 0.5}],
        [{"arm": "left", "arm_qpos": np.zeros(7)}],
        [{"arm": "left", "arm_qpos": np.zeros(5)}],
        [{"arm": "left", "arm_qpos": [0.0, 0.0, np.nan, 0.0, 0.0, 0.0]}],
        [{"arm": "left", "arm_qpos": [0.0, 0.0, np.inf, 0.0, 0.0, 0.0]}],
        [{"arm": "left", "gripper": -0.01}],
        [{"arm": "left", "gripper": 1.01}],
        [{"arm": "left", "gripper": float("nan")}],
        [{"arm": "left", "gripper": float("inf")}],
        [{"arm": "left", "gripper": True}],
        [
            {"arm": "left", "arm_qpos": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6], "gripper": 0.5},
            {"arm": "left", "arm_qpos": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]},
            {"arm": "right", "gripper": 0.5},
        ],
        [
            {"arm": "left", "arm_qpos": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]},
            {"arm": "left", "arm_qpos": np.zeros(7)},
        ],
        [
            {"arm": "left", "gripper": 0.2},
            {"arm": "left", "gripper": 1.2},
        ],
    ],
)
def test_batch_validation_precedes_io(bad_batch):
    scene = FakePiperScene()
    primitives = create_primitives(scene, str(RPENT))
    with pytest.raises(ValueError):
        primitives.apply_qpos_updates(bad_batch)
    assert scene.state_calls == 0
    assert len(scene.plan_calls) == 0
    assert len(scene.apply_calls) == 0
    assert scene.steps == 0


def test_set_gripper_and_move_to_reject_invalid_values_before_apply_io():
    scene = FakePiperScene(gripper_max_m=0.08, gripper_opening_m=0.04)
    primitives = create_primitives(scene, str(RPENT))

    with pytest.raises(ValueError):
        primitives.set_gripper(arm="left", val=1.2, steps=4)
    assert len(scene.apply_calls) == 0
    assert scene.steps == 0

    with pytest.raises(ValueError):
        primitives.set_gripper(arm="left", val=-0.1, steps=4)
    assert len(scene.apply_calls) == 0
    assert scene.steps == 0

    with pytest.raises(ValueError):
        primitives.move_to(
            arm="left",
            xyz=[0.2, float("nan"), 0.2],
            quat=[1.0, 0.0, 0.0, 0.0],
        )
    assert len(scene.plan_calls) == 0
    assert len(scene.apply_calls) == 0

    with pytest.raises(ValueError):
        primitives.move_to(
            arm="left",
            xyz=[0.2, 0.0, 0.2],
            quat=[1.0, 0.0, 0.0, 0.0],
            gripper=1.5,
        )
    assert len(scene.apply_calls) == 0
    assert scene.steps == 0


def test_rotate_wrist_uses_world_z_yaw_not_j6():
    # Tilted 90 deg about world X so world-Z yaw differs from local-Z / J6 rotation.
    s = float(np.sqrt(0.5))
    initial_quat = np.asarray([s, s, 0.0, 0.0], dtype=np.float64)
    initial_pose = [0.28, -0.04, 0.22, *initial_quat.tolist()]
    scene = FakePiperScene(tcp_pose_world=initial_pose, gripper_max_m=0.06, gripper_opening_m=0.03)
    primitives = create_primitives(scene, str(RPENT))

    delta_deg = 60.0
    yaw = np.deg2rad(delta_deg)
    world_z = np.asarray([np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)], dtype=np.float64)
    expected_world_yaw_quat = _qmult(world_z, initial_quat)
    intrinsic_local_z_quat = _qmult(initial_quat, world_z)
    assert not np.allclose(expected_world_yaw_quat, intrinsic_local_z_quat)

    result = primitives.rotate_wrist(arm="left", delta_yaw_deg=delta_deg, gripper=0.75, substeps=1)
    assert result["completed"] is True
    assert result["requested_delta_yaw_deg"] == pytest.approx(delta_deg)
    assert len(scene.plan_calls) == 1
    planned_target = scene.plan_calls[0]
    np.testing.assert_allclose(planned_target[:3], initial_pose[:3])
    np.testing.assert_allclose(planned_target[3:], expected_world_yaw_quat)
    assert not np.allclose(planned_target[3:], intrinsic_local_z_quat)
    assert scene.gripper_opening_m == pytest.approx(0.75 * 0.06)


def test_legal_zero_gripper_target_is_not_rejected_by_upstream_interpolation_roundoff():
    scene = FakePiperScene(gripper_opening_m=0.0010500000000000002)
    primitives = create_primitives(scene, str(RPENT))
    result = primitives.set_gripper(arm="left", val=0., steps=10)
    assert result["completed"]
    assert scene.gripper_opening_m == 0.
    with pytest.raises(ValueError, match="0..1"):
        primitives.apply_qpos_updates([{"arm": "left", "gripper": -1e-10}])


def test_move_to_preserves_full_path_and_substeps_sampling():
    path6 = np.asarray(
        [[0.01 * (i + 1) * (j + 1) for j in range(6)] for i in range(6)],
        dtype=np.float64,
    )
    scene = FakePiperScene(planned_path=path6)
    primitives = create_primitives(scene, str(RPENT))

    # substeps=0 preserves the full 6-waypoint trajectory.
    res_full = primitives.move_to(
        arm="left",
        xyz=[0.3, 0.0, 0.2],
        quat=[1.0, 0.0, 0.0, 0.0],
        substeps=0,
    )
    assert res_full["completed"] is True
    assert res_full["waypoints"] == 6
    assert res_full["executed_actions"] == 6
    assert len(scene.apply_calls[-1]) == 6
    for idx, update in enumerate(scene.apply_calls[-1]):
        assert update["arm"] == "left"
        assert "gripper" not in update
        assert update["arm_qpos"].shape == (6,)
        np.testing.assert_allclose(update["arm_qpos"], path6[idx])

    # substeps=1 executes only the final waypoint.
    res_one = primitives.move_to(
        arm="left",
        xyz=[0.31, 0.0, 0.2],
        quat=[1.0, 0.0, 0.0, 0.0],
        substeps=1,
    )
    assert res_one["completed"] is True
    assert res_one["waypoints"] == 1
    assert len(scene.apply_calls[-1]) == 1
    np.testing.assert_allclose(scene.apply_calls[-1][0]["arm_qpos"], path6[-1])

    # substeps=3 downsamples with np.linspace(0, 5, 3).astype(int) == [0, 2, 5].
    res_three = primitives.move_to(
        arm="left",
        xyz=[0.32, 0.0, 0.2],
        quat=[1.0, 0.0, 0.0, 0.0],
        substeps=3,
    )
    assert res_three["completed"] is True
    assert res_three["waypoints"] == 3
    expected_indices = np.linspace(0, len(path6) - 1, 3).astype(int)
    for update, idx in zip(scene.apply_calls[-1], expected_indices):
        np.testing.assert_allclose(update["arm_qpos"], path6[idx])


def test_plan_failure_returns_uncompleted_without_applying_updates():
    scene = FakePiperScene(plan_status="IK_UNREACHABLE: target out of workspace")
    primitives = create_primitives(scene, str(RPENT))
    result = primitives.move_to(
        arm="left",
        xyz=[0.9, 0.0, 0.9],
        quat=[1.0, 0.0, 0.0, 0.0],
    )
    assert result["completed"] is False
    assert result["stop_reason"] == "plan_failed"
    assert result["plan_status"] == "IK_UNREACHABLE: target out of workspace"
    assert len(scene.apply_calls) == 0


def test_check_cancelled_propagates_unswallowed():
    scene = FakePiperScene()
    primitives = create_primitives(scene, str(RPENT))
    scene.cancel_error = RuntimeError("Control-session heartbeat lost")

    with pytest.raises(RuntimeError, match="heartbeat lost"):
        primitives.move_to(arm="left", xyz=[0.3, 0.0, 0.2], quat=[1.0, 0.0, 0.0, 0.0])
    assert scene.state_calls == 0
    assert len(scene.plan_calls) == 0
    assert len(scene.apply_calls) == 0

    with pytest.raises(RuntimeError, match="heartbeat lost"):
        primitives.rotate_wrist(arm="left", delta_yaw_deg=15.0)
    assert scene.state_calls == 0
    assert len(scene.plan_calls) == 0
    assert len(scene.apply_calls) == 0

    with pytest.raises(RuntimeError, match="heartbeat lost"):
        primitives.set_gripper(arm="left", val=0.5, steps=2)
    assert scene.state_calls == 0
    assert len(scene.apply_calls) == 0

    with pytest.raises(RuntimeError, match="heartbeat lost"):
        primitives.apply_qpos_updates([{"arm": "left", "gripper": 0.5}])
    assert len(scene.apply_calls) == 0
