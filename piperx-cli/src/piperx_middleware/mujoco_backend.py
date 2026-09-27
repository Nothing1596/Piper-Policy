from __future__ import annotations

import math
import os
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from .backends import Backend, State
from .models import DomainError, JOINT_LIMITS_DEG

try:
    import mujoco
except ImportError:
    mujoco = None


import sys
_REPO_ASSET_PATH = Path(__file__).resolve().parents[2] / "assets" / "mujoco" / "scene.xml"
_INSTALLED_ASSET_PATH = Path(sys.prefix) / 'share' / 'piperx-middleware' / 'mujoco' / 'scene.xml'
DEFAULT_ASSET_PATH = _REPO_ASSET_PATH if _REPO_ASSET_PATH.is_file() else _INSTALLED_ASSET_PATH
DEFAULT_Q_DEG = [0.0, 30.0, -30.0, 0.0, 0.0, 0.0]
DEFAULT_GRIPPER_WIDTH_M = 0.04
RECOMMENDED_TCP_OFFSET_M = [0.0, 0.0, 0.1425]
RECOMMENDED_TCP_RPY_DEG = [0.0, 0.0, 0.0]

ARM_JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
GRIPPER_JOINT_NAMES = ["gripper_joint1", "gripper_joint2"]
ACTUATOR_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "gripper_joint1", "gripper_joint2"]


class MujocoBackend(Backend):
    """High-fidelity MuJoCo physics backend for Piper X with parallel gripper.

    Adheres strictly to the Backend protocol:
    - Real-time physics worker with fixed dt and lock synchronization.
    - Snapshot reports physical qpos/qvel measurements (never commanded targets).
    - Servo targets only set actuator ctrl; never teleports qpos/qvel.
    - Emergency stop commands measured hold without teleportation or resetting qpos/qvel.
    - True contact dynamics without artificial magnetic or kinematic welding.
    - Per-thread renderer management with close_renderer().
    """

    name = "mujoco"

    def __init__(self, asset_path: Path | None = None, seed: int = 0):
        if mujoco is None:
            raise RuntimeError("MuJoCo library is not available. Please install mujoco>=3.14.0.")

        self.lock = threading.RLock()
        self.render_lock = threading.Lock()
        self._thread_local = threading.local()

        resolved_path = Path(asset_path) if asset_path is not None else DEFAULT_ASSET_PATH
        if not resolved_path.exists():
            raise DomainError("asset_not_found", f"MuJoCo scene XML asset not found: {resolved_path}", 404)

        self.asset_path = resolved_path
        self.seed = int(seed)

        # Load MuJoCo model and data
        self.model = mujoco.MjModel.from_xml_path(str(self.asset_path))
        self.data = mujoco.MjData(self.model)

        # Precompute joint indices
        self.arm_qpos_indices = []
        self.arm_qvel_indices = []
        for name in ARM_JOINT_NAMES:
            j_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if j_id < 0:
                raise RuntimeError(f"Missing required arm joint in MuJoCo model: {name}")
            self.arm_qpos_indices.append(int(self.model.jnt_qposadr[j_id]))
            self.arm_qvel_indices.append(int(self.model.jnt_dofadr[j_id]))

        self.gripper_qpos_indices = []
        self.gripper_qvel_indices = []
        for name in GRIPPER_JOINT_NAMES:
            j_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if j_id < 0:
                raise RuntimeError(f"Missing required gripper joint in MuJoCo model: {name}")
            self.gripper_qpos_indices.append(int(self.model.jnt_qposadr[j_id]))
            self.gripper_qvel_indices.append(int(self.model.jnt_dofadr[j_id]))

        # Precompute actuator indices
        self.arm_act_indices = []
        for name in ARM_JOINT_NAMES:
            a_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if a_id < 0:
                raise RuntimeError(f"Missing required arm actuator in MuJoCo model: {name}")
            self.arm_act_indices.append(int(a_id))

        self.gripper_act_indices = []
        for name in GRIPPER_JOINT_NAMES:
            a_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if a_id < 0:
                raise RuntimeError(f"Missing required gripper actuator in MuJoCo model: {name}")
            self.gripper_act_indices.append(int(a_id))

        # Sites and cameras
        self.flange_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "flange")
        self.tcp_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "tcp")
        self.tray_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "tray_center")
        self.camera_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "overview_camera")

        # Object bodies and joints
        self.cube_red_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "cube_red_joint")
        self.cube_blue_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "cube_blue_joint")
        self.cube_red_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cube_red")
        self.cube_blue_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cube_blue")
        self.tray_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "tray")

        # State tracking
        self.connected = False
        self.worker_thread: threading.Thread | None = None
        self.stop_worker = threading.Event()
        self.epoch = 0
        self.received_frames = 0
        self.tx_frames = 0
        self.last_step_time = time.monotonic()
        self.last_feedback_stamp_s = time.monotonic()
        self.active_target = False
        self.arm_goal_rad=np.deg2rad(DEFAULT_Q_DEG)
        self.reference_velocity=np.zeros(6)
        self.profile_speed_percent=None

        self.ctrl_mode = 1
        self.motion_mode = 1
        self.teach_status = 0
        self.arm_status = 0
        self.error_code = 0
        self.estopped = False
        self.stale = False
        self.fault: str | None = None

        # Initialize robot & scene to default configuration
        with self.lock:
            self._initialize_positions(self.seed)

    def _initialize_positions(self, seed: int) -> None:
        """Place the arm at default safe pose and randomize cubes by seed."""
        # Arm joints
        default_rad = np.deg2rad(DEFAULT_Q_DEG)
        self.arm_goal_rad=default_rad.copy()
        self.reference_velocity=np.zeros(6)
        self.profile_speed_percent=None
        for i, idx in enumerate(self.arm_qpos_indices):
            self.data.qpos[idx] = float(default_rad[i])
            self.data.qvel[self.arm_qvel_indices[i]] = 0.0

        # Gripper joints
        half_w = float(DEFAULT_GRIPPER_WIDTH_M / 2.0)
        for i, idx in enumerate(self.gripper_qpos_indices):
            self.data.qpos[idx] = half_w
            self.data.qvel[self.gripper_qvel_indices[i]] = 0.0

        # Set actuators ctrl to match initial positions
        for i, act_id in enumerate(self.arm_act_indices):
            self.data.ctrl[act_id] = float(default_rad[i])
        for act_id in self.gripper_act_indices:
            self.data.ctrl[act_id] = half_w

        # Seeded cube randomization within workspace reachable bounds
        rng = np.random.RandomState(seed)
        # Table top is at z = 0; cube half-size is 0.015m -> center z = 0.015m
        red_x = float(rng.uniform(0.28, 0.38))
        red_y = float(rng.uniform(-0.16, -0.05))
        blue_x = float(rng.uniform(0.28, 0.38))
        blue_y = float(rng.uniform(0.02, 0.12))

        if self.cube_red_joint_id >= 0:
            adr = self.model.jnt_qposadr[self.cube_red_joint_id]
            dof = self.model.jnt_dofadr[self.cube_red_joint_id]
            self.data.qpos[adr : adr + 7] = [red_x, red_y, 0.015, 1.0, 0.0, 0.0, 0.0]
            self.data.qvel[dof : dof + 6] = 0.0

        if self.cube_blue_joint_id >= 0:
            adr = self.model.jnt_qposadr[self.cube_blue_joint_id]
            dof = self.model.jnt_dofadr[self.cube_blue_joint_id]
            self.data.qpos[adr : adr + 7] = [blue_x, blue_y, 0.015, 1.0, 0.0, 0.0, 0.0]
            self.data.qvel[dof : dof + 6] = 0.0

        self.data.time = 0.0
        mujoco.mj_forward(self.model, self.data)
        now = time.monotonic()
        self.last_step_time = now
        self.last_feedback_stamp_s = now
        self.active_target = False

    def connect(self) -> None:
        """Start physics background worker loop in real time."""
        with self.lock:
            if self.connected:
                return
            self.connected = True
            now = time.monotonic()
            self.last_feedback_stamp_s = now
            self.last_step_time = now
            self.stop_worker.clear()
            self.worker_thread = threading.Thread(
                target=self._physics_worker, name="mujoco-physics-worker", daemon=True
            )
            self.worker_thread.start()

        # Allow worker thread to perform initial steps
        time.sleep(0.01)

    def close(self) -> None:
        """Stop physics worker and release simulation resources."""
        with self.lock:
            if not self.connected:
                self.close_renderer()
                return
            self.connected = False
            self.stop_worker.set()

        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=2.0)
        self.worker_thread = None
        self.close_renderer()

    def close_renderer(self) -> None:
        """Close and release the renderer associated with the calling thread."""
        with self.render_lock:
            renderer = getattr(self._thread_local, "renderer", None)
            if renderer is not None:
                if hasattr(renderer, "close"):
                    try:
                        renderer.close()
                    except Exception:
                        pass
                self._thread_local.renderer = None
                self._thread_local.renderer_dims = None

    def _physics_worker(self) -> None:
        """Fixed dt real-time physics stepping thread."""
        wall_start = time.monotonic()
        sim_start = float(self.data.time)
        worker_epoch=self.epoch

        while not self.stop_worker.is_set():
            now = time.monotonic()
            sim_target = sim_start + (now - wall_start)

            with self.lock:
                if worker_epoch!=self.epoch:
                    wall_start=now
                    sim_start=float(self.data.time)
                    sim_target=sim_start
                    worker_epoch=self.epoch
                steps = 0
                while self.data.time < sim_target and steps < 10:
                    if self.profile_speed_percent is not None:
                        dt=float(self.model.opt.timestep)
                        reference=np.array([self.data.ctrl[a] for a in self.arm_act_indices])
                        error=self.arm_goal_rad-reference
                        vmax=math.radians(180*self.profile_speed_percent/100)
                        acceleration=math.radians(240*self.profile_speed_percent/100)
                        desired=np.sign(error)*np.minimum(vmax,np.sqrt(2*acceleration*np.abs(error)))
                        self.reference_velocity+=np.clip(desired-self.reference_velocity,-acceleration*dt,acceleration*dt)
                        increment=self.reference_velocity*dt
                        arrived=np.abs(increment)>=np.abs(error)
                        reference+=np.where(arrived,error,increment)
                        self.reference_velocity[arrived]=0
                        for a,value in zip(self.arm_act_indices,reference):self.data.ctrl[a]=float(value)
                    for dof in self.arm_qvel_indices:
                        self.data.qfrc_applied[dof] = float(self.data.qfrc_bias[dof])
                    mujoco.mj_step(self.model, self.data)
                    steps += 1
                if steps > 0:
                    t_now = time.monotonic()
                    self.last_step_time = t_now
                    self.last_feedback_stamp_s = t_now
                    self.received_frames += steps

                    # Check if active motion targets have reached steady state
                    if self.active_target:
                        arm_err = max(
                            abs(float(self.data.qpos[q_idx]) - float(self.data.ctrl[a_idx]))
                            for q_idx, a_idx in zip(self.arm_qpos_indices, self.arm_act_indices)
                        )
                        arm_vel = max(abs(float(self.data.qvel[v_idx])) for v_idx in self.arm_qvel_indices)
                        grip_err = max(
                            abs(float(self.data.qpos[q_idx]) - float(self.data.ctrl[a_idx]))
                            for q_idx, a_idx in zip(self.gripper_qpos_indices, self.gripper_act_indices)
                        )
                        grip_vel = max(abs(float(self.data.qvel[v_idx])) for v_idx in self.gripper_qvel_indices)
                        goal_err=max(abs(float(self.data.qpos[q])-target) for q,target in zip(self.arm_qpos_indices,self.arm_goal_rad))
                        if arm_err < 0.005 and goal_err < .005 and arm_vel < 0.01 and grip_err < 0.002 and grip_vel < 0.005:
                            self.active_target = False

            time.sleep(0.001)

    def _write_guard(self) -> None:
        if not self.connected:
            raise DomainError("not_connected", "Robot is not connected.", 409)
        if self.estopped:
            raise DomainError("estop_latched", "Operator must clear the software emergency stop latch.", 409)

    def emergency_stop(self) -> None:
        """Latch software emergency stop and command measured qpos hold without teleporting."""
        with self.lock:
            self.estopped = True
            self.active_target = False
            self.arm_goal_rad=np.array([self.data.qpos[q] for q in self.arm_qpos_indices])
            self.reference_velocity=np.zeros(6)
            # Set actuator targets to current measured positions without modifying qpos or qvel
            for i, q_idx in enumerate(self.arm_qpos_indices):
                self.data.ctrl[self.arm_act_indices[i]] = float(self.data.qpos[q_idx])
            for act_id, q_idx in zip(self.gripper_act_indices, self.gripper_qpos_indices):
                self.data.ctrl[act_id] = float(self.data.qpos[q_idx])

    def hold_position(self,hold_gripper=False):
        """Replace the servo reference by measured pose, retaining current grip force."""
        with self.lock:
            self.arm_goal_rad=np.array([self.data.qpos[q] for q in self.arm_qpos_indices])
            self.reference_velocity=np.zeros(6)
            for a,q in zip(self.arm_act_indices,self.arm_goal_rad):self.data.ctrl[a]=float(q)
            if hold_gripper:
                for a,q in zip(self.gripper_act_indices,self.gripper_qpos_indices):self.data.ctrl[a]=float(self.data.qpos[q])
            self.active_target=False

    def begin_joint(self, current_deg: list[float], speed_percent: int) -> None:
        with self.lock:
            self._write_guard()
            self.profile_speed_percent=max(0,int(speed_percent))
            self.reference_velocity=np.zeros(6)
            self.active_target = True

    def set_control_mode(self, current_deg: list[float], speed_percent: int, checkpoint: Any) -> None:
        with self.lock:
            checkpoint()
            self._write_guard()
            self.ctrl_mode = 1
            self.motion_mode = 1

    def joint_target(self, joints_deg: list[float]) -> None:
        """Validate finite inputs and legal limits; set position servo actuator ctrl."""
        with self.lock:
            self._write_guard()
            if len(joints_deg) != 6:
                raise DomainError("invalid_joints", "Exactly six joint targets are required.", 400)

            # Strict validation: reject non-finite inputs
            if any(not math.isfinite(q) for q in joints_deg):
                raise DomainError("invalid_joints", "All joint targets must be finite numbers.", 400)

            # Strict validation: reject out-of-bound targets (no silent clamping)
            for i, (q, (low, high)) in enumerate(zip(joints_deg, JOINT_LIMITS_DEG)):
                if not (low <= q <= high):
                    raise DomainError(
                        "joint_limit_exceeded",
                        f"Joint {i+1} target {q:g} deg exceeds legal limits [{low}, {high}] deg.",
                        400,
                    )

            self.arm_goal_rad=np.deg2rad(joints_deg)
            for i, q in enumerate(joints_deg):
                q_rad = float(np.deg2rad(q))
                if self.profile_speed_percent is None:self.data.ctrl[self.arm_act_indices[i]] = q_rad
            self.tx_frames += 1
            self.active_target = True

    def gripper_target(self, width_m: float, effort_protocol: float) -> None:
        """Validate finite inputs and legal range; set symmetric slide actuator ctrl."""
        with self.lock:
            self._write_guard()
            if not math.isfinite(width_m):
                raise DomainError("invalid_target", "Gripper width must be a finite number.", 400)
            if not math.isfinite(effort_protocol):
                raise DomainError("invalid_target", "Gripper effort must be a finite number.", 400)

            # Strict validation: legal range [0.0, 0.09] m (reject instead of silent clamp)
            if not (0.0 <= width_m <= 0.09):
                raise DomainError(
                    "gripper_limit_exceeded",
                    f"Gripper width target {width_m:g} m exceeds legal range [0.0, 0.09] m.",
                    400,
                )

            half_w = float(width_m / 2.0)
            for act_id in self.gripper_act_indices:
                self.data.ctrl[act_id] = half_w
            self.tx_frames += 1
            self.active_target = True

    def _get_gripper_contact(self, stamp: float) -> dict[str, Any]:
        """Inspect contact forces of gripper fingers against dynamic objects.

        Returns evidence of simultaneous nonzero normal force contacts of BOTH
        finger pad geoms against the SAME dynamic object. Does not leak hidden object coordinates.
        """
        g1_gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "gripper_link1_geom")
        g2_gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "gripper_link2_geom")

        dynamic_body_ids = set()
        if self.cube_red_body_id >= 0:
            dynamic_body_ids.add(self.cube_red_body_id)
        if self.cube_blue_body_id >= 0:
            dynamic_body_ids.add(self.cube_blue_body_id)

        f1_forces: dict[int, float] = {}
        f2_forces: dict[int, float] = {}
        f_buffer = np.zeros(6, dtype=np.float64)

        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)

            f1_match = (g1 == g1_gid) or (g2 == g1_gid)
            f2_match = (g1 == g2_gid) or (g2 == g2_gid)
            if not f1_match and not f2_match:
                continue

            other_gid = g2 if (g1 == g1_gid or g1 == g2_gid) else g1
            other_bid = int(self.model.geom_bodyid[other_gid])

            # Must be a dynamic object (not table, not tray, not floor, not robot link)
            if other_bid not in dynamic_body_ids:
                continue

            mujoco.mj_contactForce(self.model, self.data, i, f_buffer)
            normal_f = float(f_buffer[0])
            if normal_f > 0.005:
                if f1_match:
                    f1_forces[other_bid] = f1_forces.get(other_bid, 0.0) + normal_f
                if f2_match:
                    f2_forces[other_bid] = f2_forces.get(other_bid, 0.0) + normal_f

        supported = False
        f1_res = 0.0
        f2_res = 0.0

        for bid in dynamic_body_ids:
            force1 = f1_forces.get(bid, 0.0)
            force2 = f2_forces.get(bid, 0.0)
            if force1 > 0.01 and force2 > 0.01:
                supported = True
                f1_res = force1
                f2_res = force2
                break

        if not supported:
            f1_res = max(f1_forces.values(), default=0.0)
            f2_res = max(f2_forces.values(), default=0.0)

        return {
            "supported": bool(supported),
            "finger1_force_n": float(f1_res),
            "finger2_force_n": float(f2_res),
            "source_timestamp_s": float(stamp),
        }

    def snapshot(self) -> State:
        """Return fresh robot state measured directly from simulated qpos/qvel."""
        with self.lock:
            if not self.connected:
                return State(False)

            now = time.monotonic()

            # Measured arm joint angles (rad -> deg) and velocities (rad/s -> deg/s)
            q_rad = [float(self.data.qpos[idx]) for idx in self.arm_qpos_indices]
            v_rad_s = [float(self.data.qvel[idx]) for idx in self.arm_qvel_indices]
            q_deg = [float(np.rad2deg(q)) for q in q_rad]
            velocity_deg_s = [float(np.rad2deg(v)) for v in v_rad_s]

            # Measured gripper total width (sum of finger 1 + finger 2)
            g1 = float(self.data.qpos[self.gripper_qpos_indices[0]])
            g2 = float(self.data.qpos[self.gripper_qpos_indices[1]])
            gripper_width_m = float(g1 + g2)

            age = float(now - self.last_feedback_stamp_s) if not self.stale else 1.0
            stamp = float(self.last_feedback_stamp_s) if not self.stale else float(now - 1.0)

            # Check abnormal link collision
            collision = self._check_abnormal_collision() or (self.fault == "collision")

            # Honest actuator motor telemetry (exact simulated torque in N m; no fabricated temperatures/currents)
            motor_telemetry = []
            for i, j_dof in enumerate(self.arm_qvel_indices):
                act_torque = float(self.data.qfrc_actuator[j_dof]) if hasattr(self.data, "qfrc_actuator") else 0.0
                motor_telemetry.append({
                    "joint": i + 1,
                    "actuator_torque_nm": act_torque,
                    "motor_temp_c": None,
                    "foc_temp_c": None,
                    "bus_current_a": None,
                    "source": "mujoco",
                })

            gripper_contact = self._get_gripper_contact(stamp)

            diagnostics = {
                "simulation": True,
                "backend": "mujoco",
                "physical_validation": False,
                "epoch": self.epoch,
                "software_estop_latched": self.estopped,
                "gripper_contact": gripper_contact,
            }
            if self.fault == "driver":
                diagnostics["driver_fault"] = True
            elif self.fault == "comm":
                diagnostics["communication_error"] = True

            state = State(
                connected=True,
                q_deg=q_deg,
                velocity_deg_s=velocity_deg_s,
                enabled=[True] * 6 if self.fault != "driver" else [False] * 6,
                ctrl_mode=self.ctrl_mode if self.fault != "teaching" else 2,
                motion_mode=self.motion_mode,
                teach_status=self.teach_status if self.fault != "teaching" else 1,
                arm_status=0 if self.fault != "driver" else 1,
                error_code=0 if self.fault != "driver" else 1,
                feedback_age_s=age,
                gripper_width_m=gripper_width_m,
                gripper_enabled=True,
                gripper_error=False,
                gripper_age_s=age,
                gripper_mode="width",
                received_frames=self.received_frames,
                tx_frames=self.tx_frames,
                diagnostics=diagnostics,
                feedback_stamp_s=stamp,
                gripper_stamp_s=stamp,
                collision_status=[collision] * 6,
                motor_telemetry=motor_telemetry,
            )
            return state

    def _check_abnormal_collision(self) -> bool:
        """Detect abnormal robot self-collision or collision between arm links and environment."""
        if self.data.ncon == 0:
            return False
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            g1_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1) or ""
            g2_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2) or ""
            arm_links = ("link1", "link2", "link3", "link4", "link5", "link6", "flange")
            is_arm1 = any(l in g1_name for l in arm_links)
            is_arm2 = any(l in g2_name for l in arm_links)
            if is_arm1 and is_arm2:
                return True
            if (is_arm1 and ("table" in g2_name or "floor" in g2_name)) or (
                is_arm2 and ("table" in g1_name or "floor" in g1_name)
            ):
                return True
        return False

    # -------------------------------------------------------------------------
    # Extra Fixed Interfaces
    # -------------------------------------------------------------------------

    def observe(self, width: int = 640, height: int = 480) -> dict:
        """Render RGB and Depth observation from the fixed overview camera.

        Renderer is executed strictly on the caller thread.
        """
        renderer = getattr(self._thread_local, "renderer", None)
        r_dims = getattr(self._thread_local, "renderer_dims", None)
        if renderer is None or r_dims != (width, height):
            renderer = mujoco.Renderer(self.model, height=height, width=width)
            self._thread_local.renderer = renderer
            self._thread_local.renderer_dims = (width, height)

        with self.render_lock:
            with self.lock:
                renderer.update_scene(self.data, camera="overview_camera")
                rgb = renderer.render()

                renderer.enable_depth_rendering()
                depth = renderer.render().astype(np.float32)
                renderer.disable_depth_rendering()

                source_stamp_s = time.monotonic()
                sim_time_s = float(self.data.time)
                epoch = int(self.epoch)

                # Camera parameters
                cam_id = self.camera_id
                cam_pos = self.data.cam_xpos[cam_id].copy()
                cam_xmat = self.data.cam_xmat[cam_id].reshape(3, 3).copy()

        # Compute camera intrinsics
        fovy_deg = float(self.model.cam_fovy[self.camera_id])
        fovy_rad = math.radians(fovy_deg)
        fy = (height / 2.0) / math.tan(fovy_rad / 2.0)
        fx = fy  # square pixels
        cx = width / 2.0
        cy = height / 2.0
        intrinsics = [[float(fx), 0.0, float(cx)], [0.0, float(fy), float(cy)], [0.0, 0.0, 1.0]]

        # Camera extrinsics: OpenGL convention (X right, Y up, Z backwards)
        # converted to optical convention (X right, Y down, Z forward):
        # R_opt = R_gl @ diag(1, -1, -1)
        R_opt = cam_xmat @ np.diag([1.0, -1.0, -1.0])
        base_from_camera = np.eye(4, dtype=float)
        base_from_camera[:3, :3] = R_opt
        base_from_camera[:3, 3] = cam_pos

        camera_info = {
            "intrinsics": intrinsics,
            "base_from_camera": base_from_camera.tolist(),
            "calibration_id": "sim_overview_cam_v1",
        }

        return {
            "rgb": rgb,
            "depth": depth,
            "camera_info": camera_info,
            "source_stamp_s": source_stamp_s,
            "sim_time_s": sim_time_s,
            "epoch": epoch,
        }

    def reset(self, seed: int = 0) -> dict:
        """Reset scene and randomize object poses by seed.

        Only admitted when no motion target is active (root controls gating).
        """
        with self.lock:
            if self.active_target:
                raise DomainError("busy", "Cannot reset simulation while motion target is actively executing.", 409)

            self.epoch += 1
            self.seed = int(seed)
            self._initialize_positions(self.seed)

            red_pos = self.data.xpos[self.cube_red_body_id].tolist()
            blue_pos = self.data.xpos[self.cube_blue_body_id].tolist()

            return {
                "status": "reset",
                "epoch": self.epoch,
                "seed": self.seed,
                "default_q_deg": DEFAULT_Q_DEG,
                "default_gripper_width_m": DEFAULT_GRIPPER_WIDTH_M,
                "cubes": {
                    "cube_red": red_pos,
                    "cube_blue": blue_pos,
                },
            }

    def evaluate(self) -> dict:
        """Private ground-truth evaluation metrics for scoring without exposing model."""
        with self.lock:
            sim_time = float(self.data.time)
            epoch = int(self.epoch)

            # Object ground truth
            red_pos = self.data.xpos[self.cube_red_body_id].tolist()
            red_quat = self.data.xquat[self.cube_red_body_id].tolist()
            blue_pos = self.data.xpos[self.cube_blue_body_id].tolist()
            blue_quat = self.data.xquat[self.cube_blue_body_id].tolist()

            # Tray geometry and in-tray determination
            tray_center = [0.35, 0.20, 0.003]
            tray_bounds_xy = [0.35 - 0.07, 0.35 + 0.07, 0.20 - 0.07, 0.20 + 0.07]
            in_tray_red = bool(
                tray_bounds_xy[0] <= red_pos[0] <= tray_bounds_xy[1]
                and tray_bounds_xy[2] <= red_pos[1] <= tray_bounds_xy[3]
                and red_pos[2] >= 0.003
            )
            in_tray_blue = bool(
                tray_bounds_xy[0] <= blue_pos[0] <= tray_bounds_xy[1]
                and tray_bounds_xy[2] <= blue_pos[1] <= tray_bounds_xy[3]
                and blue_pos[2] >= 0.003
            )

            # Contacts inspection
            contacts = []
            finger1_red = False
            finger2_red = False
            finger1_blue = False
            finger2_blue = False

            for i in range(self.data.ncon):
                c = self.data.contact[i]
                g1_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1) or ""
                g2_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2) or ""

                if ("gripper_link1" in g1_name and "cube_red" in g2_name) or ("gripper_link1" in g2_name and "cube_red" in g1_name):
                    finger1_red = True
                if ("gripper_link2" in g1_name and "cube_red" in g2_name) or ("gripper_link2" in g2_name and "cube_red" in g1_name):
                    finger2_red = True
                if ("gripper_link1" in g1_name and "cube_blue" in g2_name) or ("gripper_link1" in g2_name and "cube_blue" in g1_name):
                    finger1_blue = True
                if ("gripper_link2" in g1_name and "cube_blue" in g2_name) or ("gripper_link2" in g2_name and "cube_blue" in g1_name):
                    finger2_blue = True

                contacts.append({"geom1": g1_name, "geom2": g2_name, "distance": float(c.dist)})

            grasped_red = finger1_red and finger2_red
            grasped_blue = finger1_blue and finger2_blue

            flange_pos = self.data.site_xpos[self.flange_site_id].tolist()
            tcp_pos = self.data.site_xpos[self.tcp_site_id].tolist()
            g1 = float(self.data.qpos[self.gripper_qpos_indices[0]])
            g2 = float(self.data.qpos[self.gripper_qpos_indices[1]])

            return {
                "epoch": epoch,
                "sim_time_s": sim_time,
                "objects": {
                    "cube_red": {
                        "position": red_pos,
                        "quaternion": red_quat,
                        "in_tray": in_tray_red,
                        "grasped": grasped_red,
                    },
                    "cube_blue": {
                        "position": blue_pos,
                        "quaternion": blue_quat,
                        "in_tray": in_tray_blue,
                        "grasped": grasped_blue,
                    },
                },
                "tray": {
                    "center": tray_center,
                    "bounds_xy": tray_bounds_xy,
                    "z_surface": 0.006,
                },
                "robot": {
                    "tcp_pos": tcp_pos,
                    "flange_pos": flange_pos,
                    "gripper_width_m": float(g1 + g2),
                },
                "active_contacts_count": len(contacts),
            }

    def metadata(self) -> dict:
        """Provenance, default configuration, TCP offsets, and physical limits."""
        return {
            "name": "mujoco",
            "backend_version": "1.0.0",
            "asset_provenance": {
                "repository": "https://github.com/agilexrobotics/agx_arm_urdf.git",
                "commit": "f6642ce0d7872c686f29c99e9e10cd23d1d49313",
                "license": "MIT",
                "model": "piper_x",
            },
            "tcp": {
                "recommended_offset_m": RECOMMENDED_TCP_OFFSET_M,
                "recommended_rpy_deg": RECOMMENDED_TCP_RPY_DEG,
                "description": "Fingertip center between gripper finger pads at full extension",
            },
            "default_pose": {
                "q_deg": DEFAULT_Q_DEG,
                "gripper_width_m": DEFAULT_GRIPPER_WIDTH_M,
                "description": "Safe hover pose above the table workspace",
            },
            "camera_info": {
                "calibration_id": "sim_overview_cam_v1",
                "overview_camera": {
                    "fovy_deg": 55.0,
                    "frame": "optical",
                    "convention": "x-right, y-down, z-forward",
                },
            },
            "limitations": [
                "Rigid body simulation with penalty contacts; does not model elastomer finger pad deformation.",
                "Actuator model uses position servos with kp/kv, gravity feedforward, and torque limits.",
                "Camera rendering produces synthetic RGB and pinhole z-buffer depth in metres.",
            ],
        }
