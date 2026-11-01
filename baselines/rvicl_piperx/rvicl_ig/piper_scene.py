"""Single-arm simulation boundary over the existing Piper HTTP executor.

This module starts no policy, injects no skills, and never opens CAN. Physics,
feedback, limits, job identities and shutdown remain owned by RobotService.
"""
import base64
from contextlib import contextmanager
import io
import json
from pathlib import Path
import threading
import time
import uuid

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from piperx_middleware.client import RobotClient
from piperx_middleware.kinematics import PiperKinematics
from piperx_middleware.managed_runtime import RuntimeManager
from piperx_middleware.models import DomainError, MoveTo, Settings
from piperx_middleware.primitives import resolve_primitive
from piperx_middleware.mujoco_backend import RECOMMENDED_TCP_OFFSET_M, RECOMMENDED_TCP_RPY_DEG
from piperx_middleware.profiles import ensure_profile

from .io import save
from .perception import Perception
from .recording import ExecutionFrames
from .piper_prompt import hardware_parameters


class PiperScene:
    robot_profile = "piperx"
    cameras = {"head": "mujoco-overview", "left_wrist": "mujoco-wrist"}

    def __init__(self, job, directory):
        self.job, self.directory = dict(job), Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.steps, self.capture_count, self.records = 0, 0, []
        self.paused, self.last_error = None, None
        self._exit = threading.Event()
        self._heartbeat = None
        self.client = self.operator = self.connection = None
        self._last_stamps = {}
        self._execution = None
        self._process_metadata = {}
        self._operator_trace = []
        self.last_ik_feedback = None
        self.manager = RuntimeManager(self.directory / "executor", "simulation", simulation_backend="mujoco")
        try:
            profile = ensure_profile(self.manager.base, "simulation", simulation_backend="mujoco")
            config_path = profile.root / "config.json"
            config = json.loads(config_path.read_text())
            # This is a new, isolated simulation profile; no physical/user config is edited.
            if profile.runtime_path.exists():
                raise RuntimeError("Use a fresh run directory; refusing to attach a prior experiment")
            config.update(tcp_offset_m=RECOMMENDED_TCP_OFFSET_M,
                          tcp_offset_rpy_deg=RECOMMENDED_TCP_RPY_DEG,
                          simulation_seed=job["seed"])
            if job.get("simulation_asset"):
                config["simulation_asset"] = str(Path(job["simulation_asset"]).resolve(strict=True))
            settings = Settings.model_validate(config)
            self.settings = settings
            self.robot_parameters = hardware_parameters(settings)
            save(config_path, settings.model_dump(mode="json"))
            self.kinematics = PiperKinematics(settings.tcp_offset_m, settings.tcp_offset_rpy_deg)
            self.gripper_max_m = settings.gripper_max_m
            self.feedback_timeout_s = settings.feedback_timeout_s
            self.connection = self.manager.ensure_sync()
            self.operator = RobotClient(self.connection.url, self.connection.operator_token_file)
            session = self._required(self.operator.call("POST", "/operator/session",
                {"owner": "rvicl-piperx", "shutdown_on_loss": True}))
            self.session_id = session["session_id"]
            self.client = RobotClient(self.connection.url, self.connection.model_token_file,
                                      session_id=self.session_id)
            self._heartbeat = threading.Thread(target=self._heartbeat_loop, name="rvicl-session", daemon=True)
            self._heartbeat.start()
            caps = self._required(self.client.call("GET", "/v1/capabilities"))
            if caps.get("backend") != "mujoco" or caps.get("instance_id") != self.connection.instance_id:
                raise RuntimeError("Expected this run's identified MuJoCo executor")
            self._required(self.client.call("POST", "/v1/connect", {}))
            raw = self._state()
            self.identity = (raw["instance_id"], raw["connection_epoch"])
            self.perception = Perception(job["rpent_root"], self.directory / "perception",
                                        {**job, "camera_names": list(self.cameras)})
            save(self.directory / "executor_identity.json", {
                "instance_id": self.identity[0], "connection_epoch": self.identity[1],
                "backend": "mujoco", "executor_owned": self.connection.owned,
                "tcp_offset_m": settings.tcp_offset_m, "tcp_offset_rpy_deg": settings.tcp_offset_rpy_deg,
                "physics": self._required(self.client.call("GET", "/v1/simulation/metadata")),
                "step_semantics": "accepted waypoint/hold count; physics runs continuously in wall time",
                "cameras": self.cameras, "camera_synchronization": "sequential captures with individual timestamps"})
            if job.get("operator_evaluation_trace"):
                save(self.directory / "operator_initial.json", self._operator_measurement())
        except BaseException:
            self.close()
            raise

    @staticmethod
    def _required(result):
        if "error" in result:
            raise RuntimeError(str(result["error"]))
        return result

    @property
    def truncated(self):
        return self.steps >= self.job["max_episode_steps"]

    def _heartbeat_loop(self):
        while not self._exit.wait(1):
            result = self.operator.call("POST", "/operator/session/heartbeat", {"session_id": self.session_id})
            if "error" in result:
                self.paused = "Control-session heartbeat failed: " + str(result["error"])
                return

    def check_cancelled(self):
        if self.paused:
            raise RuntimeError(self.paused)
        if self._exit.is_set():
            raise RuntimeError("Adapter is closed")

    def _state(self):
        raw = self._required(self.client.call("GET", "/v1/state"))
        if hasattr(self, "identity") and (raw.get("instance_id"), raw.get("connection_epoch")) != self.identity:
            self.paused = "Executor identity changed; no reconnect or action replay is performed"
            raise RuntimeError(self.paused)
        if not raw.get("ready"):
            raise RuntimeError("Executor feedback unavailable: " + str(raw.get("not_ready_reason")))
        return raw

    def state(self):
        raw = self._state()
        q = raw["robot"]["q_deg"]
        transform = self.kinematics.matrix(q)
        quat = Rotation.from_matrix(transform[:3, :3]).as_quat(scalar_first=True).tolist()
        return {"arms": {"left": {"joints_rad": np.deg2rad(q).tolist(), "joints_deg": list(q),
            "tcp_pose_world": [*transform[:3, 3].tolist(), *quat],
            "tcp_rpy_extrinsic_xyz_deg": self.kinematics.pose(q)["rpy_deg"],
            "tcp_axes_world": {axis: transform[:3, index].tolist() for index, axis in enumerate(("x", "y", "z"))},
            "gripper_opening_m": raw["robot"]["gripper_width_m"]}},
            "robot_parameters": self.robot_parameters,
            "last_ik_feedback": self.last_ik_feedback,
            "steps": self.steps, "remaining_steps": self.job["max_episode_steps"] - self.steps,
            "pose_convention": "PiperX TCP, xyz metres + wxyz; world equals robot base",
            "step_semantics": "accepted actuator waypoint/hold count, not physics ticks",
            "observation_identity": raw["observation_identity"]}

    def plan_arm_path(self, target):
        """Bind RPent's path contract to Piper's EXISTING joint-endpoint resolver.

        Panda mplib is robot-specific and cannot be reused for six-joint Piper.
        The native primitive still selects/executes this joint path. No new
        Cartesian planner, collision check, or hidden object pose is introduced.
        """
        self.check_cancelled()
        target = np.asarray(target, dtype=float)
        if target.shape != (7,) or not np.isfinite(target).all() or np.linalg.norm(target[3:]) < 1e-10:
            raise ValueError("Target must be seven finite xyz/wxyz values with a nonzero quaternion")
        raw = self._state()
        q = np.array(raw["robot"]["q_deg"], dtype=float)
        rotation = Rotation.from_quat(target[3:], scalar_first=True)
        self.last_ik_feedback = {
            "requested_xyz_m": target[:3].tolist(),
            "requested_quaternion_wxyz": target[3:].tolist(),
            "requested_tcp_axes_world": {axis: rotation.as_matrix()[:, i].tolist()
                                         for i, axis in enumerate(("x", "y", "z"))},
            "seed_joints_deg": q.tolist(), "scope": "this full pose and measured seed",
            "collision_checked": False, "global_unreachability_proven": False,
        }
        try:
            rpy = rotation.as_euler("xyz", degrees=True).tolist()
            move, _ = resolve_primitive(MoveTo(xyz_m=target[:3].tolist(), rpy_deg=rpy), q.tolist(), self.settings)
        except DomainError as error:
            self.last_ik_feedback.update(status="no_solution_found" if error.code == "ik_no_solution" else "rejected",
                                         error_code=error.code, solver_message=error.message,
                                         executed_steps=0)
            return {"status": f"{error.code}: {error.message}", "position": None}
        self.last_ik_feedback.update(status="solution_found", error_code=None)
        goal = np.asarray(move.joints_deg)
        count = max(1, int(np.ceil(np.max(np.abs(goal-q)) / 2.)))
        return {"status": "Success", "position": np.deg2rad(np.linspace(q, goal, count+1)[1:])}

    def _action(self, command):
        self.check_cancelled()
        self._state()
        request_id = "rvicl-" + uuid.uuid4().hex
        save(self.directory / "requests" / f"{request_id}.json", {"request_id": request_id, "command": command})
        result = self.client.call("POST", "/v1/move", {"request_id": request_id, "command": command})
        if result.get("error", {}).get("code") in {"transport_unknown", "invalid_response"}:
            # Reconciliation is a read; never send the motion again.
            result = self.client.call("GET", "/v1/requests/" + request_id)
        if isinstance(result.get("error"), dict):
            if result["error"].get("code") == "contact_completion_requires_closing":
                # The controller explicitly rejected this request before admission.
                # Keep the rejection and partial primitive; never retry it here.
                save(self.directory / "receipts" / f"{request_id}.json", result)
                self.last_error = result["error"]
                return False
            self.paused = "Action outcome requires inspection; no replay: " + str(result["error"])
            raise RuntimeError(self.paused)
        deadline = time.monotonic() + command["timeout_s"] + 10
        width_samples, last_width_stamp = 0, None
        width_stop_requested = False
        while result.get("status") in {"accepted", "running"}:
            self.check_cancelled()
            if time.monotonic() >= deadline:
                self.paused = f"Unknown completion for {request_id}; query the original request, do not replay"
                raise RuntimeError(self.paused)
            if command.get("completion") == "bilateral_contact" and not width_stop_requested:
                # Native RPent gripper commands do not require grasp contact.
                # The existing executor can finish on contact OR on width, but
                # has no combined mode. For an empty closing, verify width from
                # three fresh samples, request a measured hold, and keep the
                # original cancelled receipt. Never label that job succeeded.
                raw = self._state()
                robot = raw["robot"]
                stamp = robot.get("gripper_stamp_s")
                at_width = (robot.get("gripper_age_s") is not None and
                    robot["gripper_age_s"] <= self.feedback_timeout_s and stamp is not None and
                    abs(robot["gripper_width_m"] - command["width_m"]) < .001)
                width_samples = width_samples + 1 if at_width and stamp != last_width_stamp else 0
                last_width_stamp = stamp
                if width_samples >= 3 and raw.get("active_job_id") == result.get("job_id"):
                    self._required(self.client.call("POST", "/v1/stop", {}))
                    width_stop_requested = True
            time.sleep(.05)
            result = self.client.call("GET", "/v1/requests/" + request_id)
            if isinstance(result.get("error"), dict):
                self.paused = "Lost action receipt: " + str(result["error"])
                raise RuntimeError(self.paused)
        if result.get("status") not in {"succeeded", "failed", "cancelled"}:
            save(self.directory / "receipts" / f"{request_id}.json", result)
            self.paused = f"Uncertain action {request_id}; operator reconciliation required"
            raise RuntimeError(self.paused)
        width_completed = False
        if width_stop_requested and result.get("status") == "cancelled":
            robot = self._state()["robot"]
            width_completed = (robot.get("gripper_age_s") is not None and
                robot["gripper_age_s"] <= self.feedback_timeout_s and
                abs(robot["gripper_width_m"] - command["width_m"]) < .001)
            result = {**result, "adapter_completion": {
                "measured_width_confirmed": width_completed,
                "criterion": "three fresh target-width samples, requested hold, post-hold width",
                "grasp_verified": False}}
        save(self.directory / "receipts" / f"{request_id}.json", result)
        self.steps += 1
        completed = result.get("status") == "succeeded" or width_completed
        self.last_error = None if completed else result
        self._capture_process()
        return completed

    def apply_updates(self, updates):
        # The primitive binding validates the entire update batch before entry.
        completed = 0
        for update in updates:
            self.check_cancelled()
            if self.truncated:
                break
            ok = True
            if "arm_qpos" in update:
                ok = self._action({"kind": "joint", "joints_deg": np.rad2deg(update["arm_qpos"]).tolist(),
                    "speed_percent": self.job.get("speed_percent", 20), "timeout_s": 30.})
            if ok and "gripper" in update and not self.truncated:
                target = float(update["gripper"]) * self.gripper_max_m
                # Match the existing 1 mm width-completion tolerance. Repeated
                # full-open/hold values can differ by servo noise or float ULPs;
                # those are width requests, not contact-seeking close requests.
                closing = target < self._state()["robot"]["gripper_width_m"] - .001
                ok = self._action({"kind": "gripper", "width_m": target, "effort_protocol": .5,
                    "timeout_s": 10., "completion": "bilateral_contact" if closing else "width"})
            if not ok:
                break
            completed += 1
        return {"requested_actions": len(updates), "executed_actions": completed,
                "error": self.last_error, "episode_status": {"take_action_cnt": self.steps,
                "step_lim": self.job["max_episode_steps"], "actual_seed": self.job["seed"]}}

    def observe(self):
        self._state()
        self.capture_count += 1
        views, images = {}, []
        folder = self.directory / "observations" / f"{self.capture_count:05d}"
        folder.mkdir(parents=True)
        for camera in self.cameras:
            data = self._sensor(camera)
            stamp = data["source_stamp_s"]
            rgb_bytes = base64.b64decode(data.pop("rgb_jpeg_b64"), validate=True)
            depth = np.load(io.BytesIO(base64.b64decode(data.pop("depth_npy_b64"), validate=True)), allow_pickle=False)
            rgb = np.array(Image.open(io.BytesIO(rgb_bytes)).convert("RGB"))
            path = folder / (camera + ".jpg")
            path.write_bytes(rgb_bytes)
            np.save(folder / (camera + "-depth.npy"), depth, allow_pickle=False)
            save(folder / (camera + ".json"), data)
            views[camera] = {"rgb": rgb, "depth_m": depth, "intrinsic_cv": data["camera_info"]["intrinsics"],
                "extrinsic_cv": np.linalg.inv(data["camera_info"]["base_from_camera"]),
                "source_stamp_s": stamp, "sim_time_s": data["sim_time_s"]}
            images.append({"path": str(path), "camera": camera, "image_kind": "current_observation",
                "camera_group": list(self.cameras), "simulation_step": self.steps,
                "source_stamp_s": stamp, "sim_time_s": data["sim_time_s"]})
        step = self.perception.record(views, self.steps, capture_id=self.capture_count)
        return {"state": self.state(), "perception_step": step,
                "cameras": {c: {"image_shape": list(v["depth_m"].shape), "source_stamp_s": v["source_stamp_s"],
                                "sim_time_s": v["sim_time_s"]} for c, v in views.items()}, "_images": images}

    def _sensor(self, camera):
        data = self._required(self.client.call("GET", "/v1/simulation/observation?serial=" + self.cameras[camera]))
        if (data.get("instance_id"), data.get("connection_epoch")) != self.identity:
            self.paused = "Observation identity changed"
            raise RuntimeError(self.paused)
        stamp = data["source_stamp_s"]
        if not np.isfinite(stamp) or stamp <= self._last_stamps.get(camera, -np.inf):
            raise RuntimeError("Camera returned stale or invalid capture time")
        self._last_stamps[camera] = stamp
        return data

    def wait(self, steps):
        self.check_cancelled()
        if type(steps) is not int or not 1 <= steps <= 50:
            raise ValueError("wait steps must be 1..50")
        actual = min(steps, self.job["max_episode_steps"] - self.steps)
        time.sleep(actual * .05)
        self.steps += actual
        self._capture_process()
        return {"completed": actual == steps, "executed_steps": actual, "hold_period_s": .05}

    def _capture_process(self):
        if self._execution is None:
            return
        views = {}
        # The process sampler needs bounded RGB, not a persisted world map
        # and depth dump for every internal actuator waypoint.
        for camera in self.cameras:
            data = self._sensor(camera)
            rgb = base64.b64decode(data["rgb_jpeg_b64"], validate=True)
            views[camera] = np.array(Image.open(io.BytesIO(rgb)).convert("RGB"))
            self._process_metadata[self.steps, camera] = {
                "source_stamp_s": data["source_stamp_s"], "sim_time_s": data["sim_time_s"],
                "camera_group": list(self.cameras)}
        self._execution.capture(views, self.steps)

    @contextmanager
    def capture_execution(self, name, arguments):
        images = []
        self._process_metadata = {}
        record = getattr(self, "_current_record", None) or {}
        self._execution = ExecutionFrames(self.directory / "execution_frames" / uuid.uuid4().hex,
                                           name, self.cameras, turn=record.get("turn"), call_id=record.get("call_id"))
        self._capture_process()
        try:
            yield images
        finally:
            for item in self._execution.finish():
                item.update(self._process_metadata[item["simulation_step"], item["camera"]])
                images.append(item)
            self._execution = None

    @contextmanager
    def record_tool(self, name, arguments, **metadata):
        record = {"name": name, "arguments": arguments, **metadata, "start_steps": self.steps}
        previous = getattr(self, "_current_record", None)
        self._current_record = record
        try:
            yield record
        except BaseException as error:
            record["error"] = {"type": type(error).__name__, "message": str(error)}
            raise
        finally:
            self._current_record = previous
            record["end_steps"] = self.steps
            self.records.append(record)
            save(self.directory / "tools.json", self.records)
            if self.job.get("operator_evaluation_trace"):
                # Post-tool evidence is operator-only and never added to the tool result.
                try:
                    self._operator_trace.append({**self._operator_measurement(),
                        "tool": name, "turn": metadata.get("turn"), "call_id": metadata.get("call_id")})
                    save(self.directory / "operator_trace.json", self._operator_trace)
                except Exception as error:
                    save(self.directory / "operator_capture_error.json", {"error": str(error)})

    def _operator_measurement(self):
        return self._required(self.operator.call("GET", "/operator/simulation/evaluate"))

    def finish(self):
        self.wait(min(10, max(1, self.job["max_episode_steps"] - self.steps)))
        # The operator-only result never enters model-visible observations.
        result = self._operator_measurement()
        if self.job.get("operator_evaluation_trace"):
            samples = [result]
            deadline = time.monotonic() + 10
            while len(samples) < 6 or samples[-1]["sim_time_s"] - samples[0]["sim_time_s"] < 1.0:
                if time.monotonic() > deadline:
                    raise RuntimeError("Terminal physics observation did not advance for one simulated second")
                time.sleep(.25)
                samples.append(self._operator_measurement())
            save(self.directory / "operator_terminal.json", samples)
            result = samples[-1]
        return result

    def close(self):
        self._exit.set()
        if self._heartbeat is not None:
            self._heartbeat.join(12)
        if self.operator is not None:
            if hasattr(self, "session_id"):
                self.operator.call("POST", "/operator/session/release", {"session_id": self.session_id})
            self.operator.close()
            self.operator = None
        if self.client is not None:
            self.client.close()
            self.client = None
        if self.connection is not None:
            result = self.manager.release_sync(self.connection)
            save(self.directory / "cleanup.json", result)
            if self.connection.owned and result.get("stopped") is not True:
                raise RuntimeError("Owned executor shutdown unconfirmed; inspect cleanup.json before continuing")
            self.connection = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
