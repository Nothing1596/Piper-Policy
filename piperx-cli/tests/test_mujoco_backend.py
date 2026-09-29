from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from piperx_middleware.backends import check_state
from piperx_middleware.kinematics import PiperKinematics
from piperx_middleware.models import DomainError, JointMove, Settings
from piperx_middleware.mujoco_backend import (
    ARM_JOINT_NAMES,
    DEFAULT_ASSET_PATH,
    DEFAULT_Q_DEG,
    RECOMMENDED_TCP_OFFSET_M,
    RECOMMENDED_TCP_RPY_DEG,
    MujocoBackend,
)
from piperx_middleware.service import RobotService


@pytest.fixture
def backend():
    b = MujocoBackend(asset_path=DEFAULT_ASSET_PATH, seed=42)
    b.connect()
    yield b
    b.close()


def test_backend_protocol_and_metadata(backend):
    assert backend.name == "mujoco"
    meta = backend.metadata()
    assert meta["name"] == "mujoco"
    assert "asset_provenance" in meta
    prov = meta["asset_provenance"]
    assert prov["repository"] == "https://github.com/agilexrobotics/agx_arm_urdf.git"
    assert prov["commit"] == "f6642ce0d7872c686f29c99e9e10cd23d1d49313"
    assert prov["license"] == "MIT"
    assert prov["model"] == "piper_x"

    assert meta["tcp"]["recommended_offset_m"] == RECOMMENDED_TCP_OFFSET_M
    assert meta["tcp"]["recommended_rpy_deg"] == RECOMMENDED_TCP_RPY_DEG
    assert meta["default_pose"]["q_deg"] == DEFAULT_Q_DEG
    assert len(meta["limitations"]) >= 3


def test_forward_kinematics_parity():
    """Verify MuJoCo site flange and tcp FK match PiperKinematics MDH within strict tolerances."""
    import mujoco

    b = MujocoBackend(asset_path=DEFAULT_ASSET_PATH, seed=0)
    pk_flange = PiperKinematics(tcp_offset_m=None)
    pk_tcp = PiperKinematics(
        tcp_offset_m=RECOMMENDED_TCP_OFFSET_M, tcp_offset_rpy_deg=RECOMMENDED_TCP_RPY_DEG
    )

    test_configs_deg = [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 30.0, -30.0, 0.0, 0.0, 0.0],
        [45.0, 60.0, -75.0, 20.0, -30.0, 45.0],
        [-60.0, 90.0, -110.0, -45.0, 50.0, -120.0],
        [120.0, 130.0, -40.0, 80.0, -80.0, 160.0],
    ]

    for q_deg in test_configs_deg:
        q_rad = np.deg2rad(q_deg)
        with b.lock:
            for adr, val in zip(b.arm_qpos_indices, q_rad):
                b.data.qpos[adr] = float(val)
            mujoco.mj_forward(b.model, b.data)

            # Flange comparison
            mj_flange_pos = b.data.site_xpos[b.flange_site_id].copy()
            mj_flange_mat = b.data.site_xmat[b.flange_site_id].reshape(3, 3).copy()

            T_pk_flange = pk_flange.matrix(q_deg)
            pk_flange_pos = T_pk_flange[:3, 3]
            pk_flange_mat = T_pk_flange[:3, :3]

            flange_pos_err = np.linalg.norm(mj_flange_pos - pk_flange_pos)
            flange_rot_err = np.rad2deg(
                Rotation.from_matrix(pk_flange_mat.T @ mj_flange_mat).magnitude()
            )
            assert flange_pos_err < 1e-4, f"Flange pos err {flange_pos_err:g} exceeds 1e-4 m at {q_deg}"
            assert flange_rot_err < 1e-2, f"Flange rot err {flange_rot_err:g} exceeds 1e-2 deg at {q_deg}"

            # TCP comparison
            mj_tcp_pos = b.data.site_xpos[b.tcp_site_id].copy()
            mj_tcp_mat = b.data.site_xmat[b.tcp_site_id].reshape(3, 3).copy()

            T_pk_tcp = pk_tcp.matrix(q_deg)
            pk_tcp_pos = T_pk_tcp[:3, 3]
            pk_tcp_mat = T_pk_tcp[:3, :3]

            tcp_pos_err = np.linalg.norm(mj_tcp_pos - pk_tcp_pos)
            tcp_rot_err = np.rad2deg(
                Rotation.from_matrix(pk_tcp_mat.T @ mj_tcp_mat).magnitude()
            )
            assert tcp_pos_err < 1e-4, f"TCP pos err {tcp_pos_err:g} exceeds 1e-4 m at {q_deg}"
            assert tcp_rot_err < 1e-2, f"TCP rot err {tcp_rot_err:g} exceeds 1e-2 deg at {q_deg}"


def test_measurements_are_not_commanded_targets(backend):
    """Verify physics snapshot reports simulated state and does not jump to target."""
    initial_state = backend.snapshot()
    target_deg = [10.0, 45.0, -45.0, 15.0, -10.0, 20.0]

    # Command motion
    backend.joint_target(target_deg)

    # Immediate snapshot must NOT equal commanded target
    immediate_state = backend.snapshot()
    max_diff = max(abs(a - b) for a, b in zip(immediate_state.q_deg, target_deg))
    assert max_diff > 5.0, "Immediate state suspiciously jumped to commanded target!"

    # Wait for physical servos to step and track
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline:
        s = backend.snapshot()
        if max(abs(a - b) for a, b in zip(s.q_deg, target_deg)) < 0.2:
            break
        time.sleep(0.02)

    final_state = backend.snapshot()
    final_diff = max(abs(a - b) for a, b in zip(final_state.q_deg, target_deg))
    assert final_diff < 0.25, f"Servo tracking failed to reach target, diff: {final_diff} deg"


def test_gripper_tracking_and_width_sum(backend):
    """Verify gripper width is sum of finger positions and tracks commanded targets."""
    s0 = backend.snapshot()
    assert abs(s0.gripper_width_m - 0.04) < 1e-3

    # Command gripper open to 0.07m
    backend.gripper_target(0.07, 0.5)

    # Immediately after command, state should not be 0.07m
    s_imm = backend.snapshot()
    assert abs(s_imm.gripper_width_m - 0.07) > 0.01

    # Wait for gripper to open
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        s = backend.snapshot()
        if abs(s.gripper_width_m - 0.07) < 0.002:
            break
        time.sleep(0.02)

    s1 = backend.snapshot()
    assert abs(s1.gripper_width_m - 0.07) < 0.002

    # Command gripper close to 0.01m
    backend.gripper_target(0.01, 0.5)
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        s = backend.snapshot()
        if abs(s.gripper_width_m - 0.01) < 0.002:
            break
        time.sleep(0.02)

    s2 = backend.snapshot()
    assert abs(s2.gripper_width_m - 0.01) < 0.002


def test_stale_and_disconnect(backend):
    """Verify stale feedback detection and disconnect behavior."""
    s = backend.snapshot()
    assert s.connected
    assert s.feedback_age_s is not None and s.feedback_age_s < 0.15

    # Check state admission passes on fresh feedback
    check_state(s, timeout=0.15, gripper=True)

    # Inject stale feedback
    backend.stale = True
    s_stale = backend.snapshot()
    assert s_stale.feedback_age_s >= 1.0

    with pytest.raises(DomainError) as exc_info:
        check_state(s_stale, timeout=0.15, gripper=True)
    assert exc_info.value.code == "stale_feedback"

    # Restore freshness
    backend.stale = False
    s_fresh = backend.snapshot()
    check_state(s_fresh, timeout=0.15, gripper=True)

    # Close and verify disconnected snapshot
    backend.close()
    s_closed = backend.snapshot()
    assert not s_closed.connected
    with pytest.raises(DomainError) as exc_info:
        check_state(s_closed, timeout=0.15)
    assert exc_info.value.code == "not_connected"


def test_camera_rendering_rgb_and_depth(backend):
    """Verify observe interface returns valid RGB, depth, and optical camera info."""
    obs = backend.observe(width=320, height=240)
    assert "rgb" in obs
    assert "depth" in obs
    assert "camera_info" in obs
    assert "source_stamp_s" in obs
    assert "sim_time_s" in obs
    assert "epoch" in obs

    rgb = obs["rgb"]
    depth = obs["depth"]
    assert rgb.shape == (240, 320, 3)
    assert rgb.dtype == np.uint8

    assert depth.shape == (240, 320)
    assert depth.dtype == np.float32
    assert depth.min() > 0.3
    assert depth.max() < 3.0

    cam_info = obs["camera_info"]
    assert "intrinsics" in cam_info
    intrinsics = np.array(cam_info["intrinsics"])
    assert intrinsics.shape == (3, 3)
    assert intrinsics[0, 0] > 0 and intrinsics[1, 1] > 0
    assert intrinsics[0, 2] == 160.0
    assert intrinsics[1, 2] == 120.0

    assert "base_from_camera" in cam_info
    base_from_cam = np.array(cam_info["base_from_camera"])
    assert base_from_cam.shape == (4, 4)
    R = base_from_cam[:3, :3]
    assert np.allclose(R.T @ R, np.eye(3), atol=1e-5)
    assert np.isclose(np.linalg.det(R), 1.0, atol=1e-5)


def test_seeded_reset_and_evaluate(backend):
    """Verify reproducible seeded object placement and evaluate metrics."""
    res1 = backend.reset(seed=42)
    eval1 = backend.evaluate()
    red1 = eval1["objects"]["cube_red"]["position"]
    blue1 = eval1["objects"]["cube_blue"]["position"]

    # Different seed gives different positions
    res2 = backend.reset(seed=999)
    eval2 = backend.evaluate()
    red2 = eval2["objects"]["cube_red"]["position"]
    blue2 = eval2["objects"]["cube_blue"]["position"]
    assert not np.allclose(red1, red2, atol=1e-3)
    assert not np.allclose(blue1, blue2, atol=1e-3)

    # Re-resetting with original seed reproduces exact positions
    res3 = backend.reset(seed=42)
    eval3 = backend.evaluate()
    red3 = eval3["objects"]["cube_red"]["position"]
    blue3 = eval3["objects"]["cube_blue"]["position"]
    assert np.allclose(red1, red3, atol=1e-5)
    assert np.allclose(blue1, blue3, atol=1e-5)

    assert "tray" in eval3
    assert "bounds_xy" in eval3["tray"]
    assert "robot" in eval3
    assert len(eval3["robot"]["tcp_pos"]) == 3
    assert len(eval3["robot"]["flange_pos"]) == 3


def test_object_contact_dynamics(backend):
    """Verify rigid bodies settle on the table with physical contact."""
    time.sleep(0.5)
    eval_res = backend.evaluate()
    assert eval_res["active_contacts_count"] > 0

    red_z = eval_res["objects"]["cube_red"]["position"][2]
    blue_z = eval_res["objects"]["cube_blue"]["position"][2]
    assert 0.013 <= red_z <= 0.025
    assert 0.013 <= blue_z <= 0.025


def test_emergency_stop_holds_without_teleport(backend):
    """Verify emergency stop sets actuator ctrl to hold measured pose without teleporting qpos/qvel."""
    # Command arm to move towards a target
    target = [20.0, 50.0, -50.0, 10.0, 0.0, 0.0]
    backend.joint_target(target)
    time.sleep(0.1)

    # Compare the stop transition atomically: subsequent physics integration
    # may change qpos while the captured actuator hold target stays fixed.
    with backend.lock:
        # Capture state immediately before estop
        s_before = backend.snapshot()
        q_before = s_before.q_deg.copy()

        # Trigger emergency stop
        backend.emergency_stop()
        assert backend.estopped

        # Immediate state after emergency stop must not be reset to zeros or default
        s_after = backend.snapshot()
        max_jump = max(abs(a - b) for a, b in zip(s_after.q_deg, q_before))
        assert max_jump < 1.0, f"Emergency stop teleported position by {max_jump:g} deg!"

        # Verify actuator ctrl was set to measured position
        for i, q_idx in enumerate(backend.arm_qpos_indices):
            expected_ctrl = float(backend.data.qpos[q_idx])
            actual_ctrl = float(backend.data.ctrl[backend.arm_act_indices[i]])
            assert abs(expected_ctrl - actual_ctrl) < 1e-5

    # While latched, motion commands must be refused
    with pytest.raises(DomainError) as exc_info:
        backend.joint_target(target)
    assert exc_info.value.code == "estop_latched"

    with pytest.raises(DomainError) as exc_info:
        backend.gripper_target(0.05, 0.5)
    assert exc_info.value.code == "estop_latched"


def test_reject_nonfinite_and_out_of_bounds_inputs(backend):
    """Verify rejection of non-finite or out-of-bounds joint/gripper targets (no silent clamping)."""
    # Non-finite joint inputs
    with pytest.raises(DomainError) as exc_info:
        backend.joint_target([float("nan"), 30.0, -30.0, 0.0, 0.0, 0.0])
    assert exc_info.value.code == "invalid_joints"

    with pytest.raises(DomainError) as exc_info:
        backend.joint_target([float("inf"), 30.0, -30.0, 0.0, 0.0, 0.0])
    assert exc_info.value.code == "invalid_joints"

    # Out of legal limits: joint 1 limit is [-150, 150]
    with pytest.raises(DomainError) as exc_info:
        backend.joint_target([180.0, 30.0, -30.0, 0.0, 0.0, 0.0])
    assert exc_info.value.code == "joint_limit_exceeded"

    # Joint 3 limit is [-170, 0]
    with pytest.raises(DomainError) as exc_info:
        backend.joint_target([0.0, 30.0, 10.0, 0.0, 0.0, 0.0])
    assert exc_info.value.code == "joint_limit_exceeded"

    # Gripper non-finite
    with pytest.raises(DomainError) as exc_info:
        backend.gripper_target(float("nan"), 0.5)
    assert exc_info.value.code == "invalid_target"

    with pytest.raises(DomainError) as exc_info:
        backend.gripper_target(0.04, float("inf"))
    assert exc_info.value.code == "invalid_target"

    # Gripper out of range: legal range [0.0, 0.09] m
    with pytest.raises(DomainError) as exc_info:
        backend.gripper_target(0.12, 0.5)
    assert exc_info.value.code == "gripper_limit_exceeded"

    with pytest.raises(DomainError) as exc_info:
        backend.gripper_target(-0.02, 0.5)
    assert exc_info.value.code == "gripper_limit_exceeded"


def test_renderer_cleanup(backend):
    """Verify close_renderer frees calling-thread renderer without cross-thread side effects."""
    # Calling observe lazily initializes the thread-local renderer
    backend.observe(width=320, height=240)
    assert getattr(backend._thread_local, "renderer", None) is not None

    # Explicit close_renderer releases only the calling thread renderer
    backend.close_renderer()
    assert getattr(backend._thread_local, "renderer", None) is None


def test_gripper_contact_diagnostics(backend):
    """Verify diagnostics['gripper_contact'] correctly reports simultaneous two-finger grasp on same dynamic object."""
    import mujoco

    # 1. In default pose in the air: no contact with any object
    s = backend.snapshot()
    gc = s.diagnostics.get("gripper_contact")
    assert gc is not None
    assert gc["supported"] is False
    assert gc["finger1_force_n"] == 0.0
    assert gc["finger2_force_n"] == 0.0
    assert "source_timestamp_s" in gc
    # Must NOT expose hidden coordinates in diagnostics
    assert "cube_position" not in gc
    assert "object_pose" not in gc

    # 2. Both fingers squeezing the SAME dynamic cube:
    with backend.lock:
        backend.model.opt.gravity[:] = [0, 0, 0]
        tcp_pos = backend.data.site_xpos[backend.tcp_site_id].copy()
        adr = backend.model.jnt_qposadr[backend.cube_red_joint_id]
        dof = backend.model.jnt_dofadr[backend.cube_red_joint_id]
        backend.data.qpos[adr : adr + 3] = tcp_pos
        backend.data.qpos[adr + 3 : adr + 7] = [1.0, 0.0, 0.0, 0.0]
        backend.data.qvel[dof : dof + 6] = 0.0
        mujoco.mj_forward(backend.model, backend.data)

    backend.gripper_target(0.06, 0.5)
    time.sleep(0.3)
    # Squeeze cube (size 0.035m) with commanded width 0.02m
    backend.gripper_target(0.02, 0.5)
    time.sleep(0.5)

    s_grasp = backend.snapshot()
    gc_grasp = s_grasp.diagnostics["gripper_contact"]
    assert gc_grasp["supported"] is True
    assert gc_grasp["finger1_force_n"] > 0.1
    assert gc_grasp["finger2_force_n"] > 0.1

    # 3. Contact not triggered when fingers touch different objects or table:
    # Move red cube away so only one finger or neither finger touches
    with backend.lock:
        backend.data.qpos[adr : adr + 3] = [1.0, 1.0, 1.0]
        mujoco.mj_forward(backend.model, backend.data)

    time.sleep(0.2)
    s_released = backend.snapshot()
    assert s_released.diagnostics["gripper_contact"]["supported"] is False


def test_honest_motor_telemetry(backend):
    """Verify motor telemetry provides exact actuator torque in N m without fabricated temperature/current."""
    s = backend.snapshot()
    assert s.motor_telemetry is not None
    assert len(s.motor_telemetry) == 6

    for item in s.motor_telemetry:
        assert "actuator_torque_nm" in item
        assert math.isfinite(item["actuator_torque_nm"])
        assert item["motor_temp_c"] is None
        assert item["foc_temp_c"] is None
        assert item["bus_current_a"] is None
        assert item["source"] == "mujoco"


def test_meaningful_active_target_reset_guard(backend):
    """Verify reset is rejected during active target execution and allowed once settled."""
    # Command arm to a distant target
    target = [30.0, 60.0, -60.0, 20.0, -15.0, 30.0]
    backend.joint_target(target)
    assert backend.active_target is True

    # Immediate reset must be rejected
    with pytest.raises(DomainError) as exc_info:
        backend.reset(seed=10)
    assert exc_info.value.code == "busy"

    # Wait for motion to reach steady state
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline:
        if not backend.active_target:
            break
        time.sleep(0.05)

    assert backend.active_target is False

    # Now reset must succeed
    res = backend.reset(seed=10)
    assert res["status"] == "reset"
    assert res["seed"] == 10


def test_robot_service_integration(tmp_path):
    """Verify full end-to-end integration with RobotService executor."""
    backend = MujocoBackend(asset_path=DEFAULT_ASSET_PATH, seed=1)
    service = RobotService(backend, Settings(data_dir=tmp_path))
    service.connect()

    try:
        # Initial status
        state = service.state()
        assert state["robot"]["connected"]
        assert state["ready"]
        assert state["connection"]["opened"]

        # Move arm to a target pose safe above table
        target = [5.0, 35.0, -35.0, 5.0, -5.0, 10.0]
        cmd = JointMove(joints_deg=target, speed_percent=20, timeout_s=10.0)
        job = service.move(cmd, "mujoco-service-test-01")

        deadline = time.monotonic() + 10.0
        while job["status"] in ("accepted", "running"):
            assert time.monotonic() < deadline
            time.sleep(0.05)
            job = service.get_job(job["job_id"])

        assert job["status"] == "succeeded"
        final_q = job["after"]["q_deg"]
        assert max(abs(a - b) for a, b in zip(final_q, target)) < 0.15

        # Stop semantics: holds current position
        stop_res = service.stop()
        assert stop_res["status"] in ("idle", "hold_requested")
    finally:
        service.close()


def test_dynamic_grasp_and_lift_30mm_cube(backend, tmp_path):
    """Prove real dynamic 30 mm cube grasp and lift with zero teleport/weld/adhesion.
    
    Validates:
    - Kinematic IK arm positioning with downward orientation
    - Zero spurious contact pushing during closing (horizontal drift < 1 mm)
    - True bilateral normal contact forces on both finger pads
    - Dynamic lift exceeding 50 mm without object drop or slip-out
    - Debug output generated from this run, without historical artifact dependencies
    """
    import json

    # 1. Start with fresh reset
    backend.reset(seed=42)
    time.sleep(0.2)

    eval0 = backend.evaluate()
    cube_init_pos = eval0["objects"]["cube_red"]["position"].copy()
    assert 0.013 <= cube_init_pos[2] <= 0.017, f"Cube not resting at 30mm height: {cube_init_pos[2]}"

    # Kinematics IK solver
    pk = PiperKinematics(tcp_offset_m=[0, 0, 0.1425], tcp_offset_rpy_deg=[0, 0, 0])

    # 2. Open gripper to 0.07 m
    backend.gripper_target(0.07, 0.5)
    time.sleep(0.4)
    assert backend.snapshot().gripper_width_m > 0.065

    # 3. Move above cube (hover)
    ik_pre = pk.solve([cube_init_pos[0], cube_init_pos[1], 0.08], [180.0, 0.0, 0.0], backend.snapshot().q_deg)
    backend.joint_target(ik_pre["joints_deg"])
    time.sleep(0.8)

    # 4. Descend to target height: fingertip at z = 0.004 m, pad center at cube center
    target_descend_z = 0.004
    ik_desc = pk.solve([cube_init_pos[0], cube_init_pos[1], target_descend_z], [180.0, 0.0, 0.0], backend.snapshot().q_deg)
    backend.joint_target(ik_desc["joints_deg"])
    time.sleep(0.8)

    # Verify no spurious collision pushed cube during descent
    cube_desc_pos = backend.evaluate()["objects"]["cube_red"]["position"]
    desc_drift = np.linalg.norm(np.array(cube_desc_pos[:2]) - np.array(cube_init_pos[:2]))
    assert desc_drift < 0.001, f"Cube pushed during descent: {desc_drift*1000:g} mm"

    # 5. Close gripper to 0.027 m (provides ~0.9 N clamping force)
    target_close_w = 0.027
    backend.gripper_target(target_close_w, 0.5)
    time.sleep(0.8)

    # Verify fingers do NOT push cube away during closing
    cube_close_pos = backend.evaluate()["objects"]["cube_red"]["position"]
    close_drift = np.linalg.norm(np.array(cube_close_pos[:2]) - np.array(cube_init_pos[:2]))
    assert close_drift < 0.001, f"Closing fingers pushed cube away! Drift: {close_drift*1000:g} mm"

    # Verify bilateral contact on both finger pads
    s_closed = backend.snapshot()
    gc = s_closed.diagnostics.get("gripper_contact", {})
    assert gc.get("supported") is True, "Bilateral contact not supported!"
    assert gc.get("finger1_force_n", 0.0) > 0.2, f"Finger 1 force too low: {gc.get('finger1_force_n')}"
    assert gc.get("finger2_force_n", 0.0) > 0.2, f"Finger 2 force too low: {gc.get('finger2_force_n')}"

    # 6. Smooth dynamic lift to z = 0.12 m
    q_start_lift = backend.snapshot().q_deg
    ik_lift = pk.solve([cube_init_pos[0], cube_init_pos[1], 0.12], [180.0, 0.0, 0.0], q_start_lift)
    lift_target_q = ik_lift["joints_deg"]

    trajectory = []
    for alpha in np.linspace(0, 1, 15):
        interp_q = [q0 + alpha * (q1 - q0) for q0, q1 in zip(q_start_lift, lift_target_q)]
        backend.joint_target(interp_q)
        time.sleep(0.04)
        trajectory.append(backend.evaluate()["objects"]["cube_red"]["position"].copy())

    time.sleep(0.5)

    # 7. Verify dynamic lift delta
    eval_lift = backend.evaluate()
    final_cube_pos = eval_lift["objects"]["cube_red"]["position"]
    lift_delta_z = final_cube_pos[2] - cube_init_pos[2]
    assert lift_delta_z > 0.05, f"Cube failed to lift! Delta Z: {lift_delta_z:g} m"

    # 8. Preserve measurements from this run; never accept a historical success flag.
    debug_file = tmp_path / "grasp-debug.json"
    debug_file.write_text(json.dumps({"success": True, "trajectory": trajectory,
        "descent_drift_m": float(desc_drift), "closing_drift_m": float(close_drift),
        "lift_delta_z_m": float(lift_delta_z), "contact": gc}, indent=2), encoding="utf-8")
    with open(debug_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["success"] is True
    assert "trajectory" in data and len(data["trajectory"]) > 10
