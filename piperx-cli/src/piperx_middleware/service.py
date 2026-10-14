from __future__ import annotations

import copy
import math
import os
import threading
import time
import uuid

from .backends import Backend, check_state
from .models import ControlMode, DomainError, ExecuteRequest, GripperMove, JointMove, JOINT_LIMITS_DEG, LeaseRequest, Settings
from .store import Store
from .runtime_controls import RuntimeControls
from .interaction_service import InteractionControls


def _command_defaults(command):
    """Keep retries of pre-contact-completion journal records idempotent."""
    if isinstance(command,dict) and command.get('kind') in ('gripper','set_gripper'):
        return {'completion':'width',**command}
    return command


# --- Bounded recovery from a confirmed blocked motion (MuJoCo, direct profile) -
# A timed-out or stalled ordinary motion may be re-expressed as
# failed/motion_blocked/recoverable only after the servo reference has been
# replaced by the measured pose and fresh feedback has confirmed standstill on
# the same connection. These are fixed internal limits, not operator, model or
# configuration knobs. The whole confirmation stays inside the client's 5 s
# post-timeout polling grace.
_RECOVERY_HOLD_SAMPLE_INTERVAL_S = 0.05
_RECOVERY_HOLD_MAX_S = 2.5
_RECOVERY_HOLD_MIN_SAMPLES = 3
_RECOVERY_HOLD_MIN_SPAN_S = 0.25
_RECOVERY_HOLD_MAX_VELOCITY_DEG_S = 0.5
_RECOVERY_HOLD_MAX_JOINT_DELTA_DEG = 0.05
_RECOVERY_HOLD_MAX_WIDTH_DELTA_M = 0.0005

# Sustained measured-progress watchdog. A motion is declared stalled only when
# the measured joint vector (or gripper width) does not change by more than
# epsilon over a full window, so slow but genuinely progressing motion keeps
# running. The watchdog deliberately watches measured displacement instead of
# the scalar distance to the final commanded target: a legal curved
# Cartesian/IK path can hold one axis's final-target error flat (or make it
# grow) while other joints move.
_STALL_WARMUP_S = 0.4
_STALL_WINDOW_S = 1.2
_STALL_MIN_PROGRESS_DEG = 0.02
_STALL_MIN_PROGRESS_M = 0.0002

# Original failures that may become motion_blocked after a confirmed hold.
# Anything else (faults, stale feedback, limits, collisions, cancellation,
# control-mode changes) keeps its existing unknown/cancellation handling.
_BLOCKED_RECOVERY_CODES = ("execution_timeout", "motion_stalled")


class _ProgressWatch:
    """Worker-local watchdog over measured joint/width displacement.

    Samples the measured feedback vector, not the scalar error to the final
    commanded joints. A curved Cartesian/IK path (or a stationary axis whose
    final-target error stays largest) must not be called stalled while the
    measured joints keep changing; only a full window with no displacement
    beyond epsilon -- including a reference permanently paused in the existing
    .12-.3 tracking guard band -- is blocked.
    """

    __slots__ = ("started_at", "samples")

    def __init__(self, started_at: float):
        self.started_at = started_at
        self.samples: list[tuple[float, tuple[float, ...]]] = []

    def stalled(self, now: float, measured, epsilon: float) -> bool:
        self.samples.append((now, tuple(float(v) for v in measured)))
        cutoff = now - 2 * _STALL_WINDOW_S
        while len(self.samples) > 2 and self.samples[0][0] < cutoff:
            self.samples.pop(0)
        if now - self.started_at < _STALL_WARMUP_S:
            return False
        baseline = None
        for sampled_at, values in self.samples:
            if sampled_at <= now - _STALL_WINDOW_S:
                baseline = values
            else:
                break
        if baseline is None:
            return False
        return max(abs(value - base) for _, values in self.samples
                   for value, base in zip(values, baseline)) < epsilon


class RobotService(InteractionControls, RuntimeControls):
    """One serialized action owner, shared by HTTP and every MCP client."""
    def __init__(self, backend: Backend, settings: Settings):
        self.backend, self.settings = backend, settings
        self.store = Store(settings.data_dir)
        self.lock = threading.RLock()
        # Never held across hardware I/O: cancellation must not wait for a blocked DLL.
        self.control_lock = threading.RLock()
        self.plans = {}
        self.epoch = 0
        self.lease = None
        self.active = None
        self.thread = None
        self.native_linear_active = False
        self.cancel = threading.Event()
        self.cancel_lock = threading.Lock()
        self.stop_generation = 0
        self.closed = False
        self.closing = False
        self.connection_attempted = False
        self.instance_id = str(uuid.uuid4())
        self._init_interaction()
        self._init_controls()

    def _require_open(self):
        if self.closed or self.closing:
            raise DomainError("shutting_down", "Executor is closing; no new connection or action is accepted.", 409)

    def capabilities(self):
        return {
            "api_version": "1", "backend": self.backend.name,
            "process_id": os.getpid(), "instance_id": self.instance_id,
            "simulation": self.backend.name in ("sim", "mujoco"),
            "hardware_motion_configured": self.settings.allow_motion,
            "control_profile": self.settings.control_profile,
            "requires_control_window": self.settings.control_profile == "calibration" and not self.interaction_required,
            "requires_control_session": self.interaction_required,
            "approval_modes": ["always", "risk", "auto"],
            "mcp_tools": ["robot_status", "robot_connect", "robot_disconnect", "robot_move_joints", "robot_gripper", "robot_stop",
                          "move_to", "move_by", "rotate", "set_gripper", "robot_set_control_mode", "move_linear", "robot_diagnostics"],
            "http_operations": ["control-mode", "connect", "disconnect", "state", "move", "primitives", "preview", "execute", "job", "stop", "shutdown",
                                "parameters", "jobs", "calls", "devices"],
            "excluded": ["raw CAN", "reset", "disable", "enable", "teaching commands", "zero calibration", "MIT", "camera-frame targets", "collision planning"],
            "cartesian_primitives": {"frame": "base", "position_units": "metres", "orientation": "extrinsic xyz RPY degrees (Rz Ry Rx)",
                "path_semantics": "joint_space_endpoint", "tcp_offset_m": self.settings.tcp_offset_m,
                "tcp_offset_rpy_deg": self.settings.tcp_offset_rpy_deg,
                "default_tcp": "flange; configure the measured flange-to-tool transform for fingertip targets"},
            "joint_units": "degrees", "gripper_units": "metres",
            "gripper_effort": "protocol field; physical N versus Nm is unresolved, not calibrated finger force",
            "stop_semantics": "Request current measured joint position (and current width for an active gripper job). Requires fresh feedback and working CAN. No 0x150 damping command. Not a hardware emergency stop.",
            "limits": {"max_move_deg": self.settings.max_move_deg if self.settings.control_profile == "calibration" else None,
                       "speed_percent": self.settings.max_speed_percent if self.settings.control_profile == "calibration" else 100,
                       "observed_velocity_deg_s": self.settings.max_velocity_deg_s if self.settings.control_profile == "calibration" else None,
                       "gripper_width_m": [0, self.settings.gripper_max_m],
                       "gripper_effort_protocol": [0, self.settings.gripper_effort_limit if self.settings.control_profile == "calibration" else 32.767],
                       "feedback_timeout_s": self.settings.feedback_timeout_s,
                       "joint_limits_deg": JOINT_LIMITS_DEG},
            "collision_checking": False,
            "collision_feedback": True,
            "operator_operations": ["estop", "clear-estop", "runtime-parameters", "control-window"],
            "linear_motion": {"path_semantics": "sampled_cartesian_line_joint_interpolation",
                              "orientation": "fixed", "max_waypoints": 100, "collision_checking": False,
                              "native_move_l": self.backend.name == "agx", "native_requires_direct_profile": True},
            "velocity_semantics": "SDK motor telemetry converted from rad/s to deg/s; not independently validated joint speed. Direct-profile completion uses fresh position samples.",
        }

    def _check(self, state, *, gripper=False, moving=False, require_position_mode=True):
        if self.estop_latched:
            raise DomainError("estop_latched", "Operator must clear the software emergency stop latch.")
        check_state(state, self.settings.feedback_timeout_s, gripper=gripper, moving=moving,
                    require_position_mode=require_position_mode,
                    max_velocity=self.settings.max_velocity_deg_s if self.settings.control_profile == "calibration" else None)

    def devices(self):
        from .device_discovery import discover_devices
        with self.lock:
            self._require_open()
            return discover_devices(self)

    def connect(self, reconnect=False, device_id=None):
        with self.lock:
            self._require_open()
            if self.active:
                raise DomainError("busy", "An action is active.")
            from .device_discovery import cleanup_discovery
            cleanup_discovery(self)
            if reconnect:
                self._disconnect_locked()
            if device_id is not None:
                from .device_discovery import select_device
                select_device(self, device_id)
            if not self.backend.snapshot().connected:
                if self.connection_attempted:
                    raise DomainError("reconnect_required", "Previous connection failed; use robot_connect(reconnect=True) to release it before reopening.")
                self.store.event("connecting", backend=self.backend.name)
                self.connection_attempted = True
                try:
                    self.backend.connect()
                except DomainError:
                    # Preserve codes the backend already classified, such as
                    # hardware_owned or cleanup_incomplete.
                    raise
                except Exception as exc:
                    # The SDK raises its own types (for example
                    # can.exceptions.CanInitializationError when no adapter is
                    # present). Letting those escape produced an HTTP 500 with an
                    # empty body, so a client could not read error.code and could
                    # not tell "no CAN adapter" from "server bug".
                    raise DomainError(
                        "can_unavailable",
                        f"{self.backend.name} backend could not open its transport: "
                        f"{type(exc).__name__}: {exc}",
                        503,
                    ) from exc
                self.epoch += 1
                self.lease = None
                self.plans.clear()
                deadline = time.monotonic() + self.settings.connect_timeout_s
                # Wait for first feedback, never reopen an already-open device implicitly.
                while time.monotonic() < deadline:
                    sample = self.backend.snapshot()
                    if sample.feedback_age_s is not None:
                        break
                    time.sleep(.02)
            return self.state()

    def _disconnect_locked(self):
        self._cancel_pending("connection_changed")
        self.lease = None
        self.plans.clear()
        self.measured_limits = None
        self.epoch += 1
        from .device_discovery import cleanup_discovery
        cleanup_discovery(self)
        self.backend.close()  # A failed cleanup retains ownership and aborts reconnect.
        self.connection_attempted = False
        self.store.event("disconnected", connection_epoch=self.epoch)

    def disconnect(self):
        with self.lock:
            self._require_open()
            if self.active:
                raise DomainError("busy", "An action is active; disconnect does not stop a robot.")
            self._disconnect_locked()
            return self.state()

    def state(self):
        with self.lock:
            s = self.backend.snapshot()
            try:
                self._check(s)
                ready, reason = True, None
            except DomainError as exc:
                ready, reason = False, {"code": exc.code, "message": exc.message}
            lease = self._public_lease()
            live = (s.connected and s.feedback_age_s is not None and
                    math.isfinite(s.feedback_age_s) and s.feedback_age_s <= self.settings.feedback_timeout_s)
            failed_session = not s.connected and self.connection_attempted
            reconnect_needed = failed_session or (s.connected and not live)
            link_status = ("faulted" if failed_session else "disconnected" if not s.connected else "live" if live else
                           "no_feedback" if not s.received_frames else "stale")
            tcp = None
            if s.q_deg is not None and len(s.q_deg) == 6 and all(math.isfinite(x) for x in s.q_deg):
                from .kinematics import PiperKinematics
                tcp = PiperKinematics(self.settings.tcp_offset_m, self.settings.tcp_offset_rpy_deg).pose(s.q_deg)
                tcp.update(source="forward_kinematics", feedback_stamp_s=s.feedback_stamp_s,
                           feedback_age_s=s.feedback_age_s, feedback_live=live)
            return {"instance_id": self.instance_id, "robot": s.public(), "tcp": tcp, "ready": ready, "not_ready_reason": reason,
                    "interaction": {"policy_mode": self.policy.mode, "session": self.sessions.public(),
                                    "control_session_required": self.interaction_required, "scene_collision_checked": False},
                    "observation_identity": {"instance_id": self.instance_id, "connection_epoch": self.epoch,
                        "clock_domain": "executor_host_monotonic", "source_stamp_s": s.feedback_stamp_s,
                        "valid_until_s": s.feedback_stamp_s + self.settings.feedback_timeout_s if s.feedback_stamp_s is not None else None,
                        "cross_host_synchronized": False},
                    "connection_epoch": self.epoch, "active_job_id": self.active,
                    "control_window": lease, "backend": self.backend.name,
                    "connection": {"opened": s.connected, "feedback_live": live, "status": link_status,
                        "reconnect_recommended": reconnect_needed,
                        "reconnect_action": "robot_connect(reconnect=True)" if reconnect_needed else None}}

    def _public_lease(self):
        lease = self.lease
        if lease is None:
            return None
        return {k: v for k, v in lease.items() if k != "deadline"} | {
            "remaining_s": max(0., lease["deadline"] - time.monotonic())}

    def arm_window(self, req: LeaseRequest):
        with self.lock:
            self._require_open()
            if not self.settings.allow_motion:
                raise DomainError("read_only", "Server was started in read-only mode.", 403)
            if self.active:
                raise DomainError("busy", "Wait for the active action or stop it.")
            s = self.backend.snapshot()
            self._check(s, gripper=req.allow_gripper)
            with self.control_lock:
                lease = {"window_id": str(uuid.uuid4()), "anchor_deg": s.q_deg.copy(),
                              "radius_deg": req.joint_radius_deg, "allow_gripper": req.allow_gripper,
                              "deadline": time.monotonic() + req.duration_s,
                              "expires_at": time.time() + req.duration_s}
                self.store.event("operator_control_window", **{k:v for k,v in lease.items() if k != "deadline"})
                self.lease = lease
                self.cancel.clear()
                return self._public_lease()

    def _check_window(self, state, command):
        if not self.settings.allow_motion:
            raise DomainError("read_only", "Motion is disabled in the server configuration.", 403)
        if isinstance(command, ControlMode):
            # Mode preparation holds the measured pose and creates no motion window.
            if self.settings.control_profile == "calibration" and command.speed_percent > self.settings.max_speed_percent:
                raise DomainError("speed_limit", "Speed exceeds the configured controller limit.", 422)
            return
        if self.interaction_required or self.settings.control_profile == "direct":
            return
        lease = self.lease
        if self.cancel.is_set() or not lease or time.monotonic() >= lease["deadline"]:
            raise DomainError("no_control_window", "The operator must open a bounded control window.", 403)
        anchor, radius = lease["anchor_deg"], lease["radius_deg"]
        points = [state.q_deg]
        if isinstance(command, JointMove):
            points.append(command.joints_deg)
        elif not lease["allow_gripper"]:
            raise DomainError("gripper_not_authorized", "This control window excludes the gripper.", 403)
        if any(abs(v - a) > radius + 0.01 for point in points for v, a in zip(point, anchor)):
            raise DomainError("window_bounds", "Measured or target joint angles leave the operator's absolute envelope.")

    def preview(self, command: JointMove | GripperMove | ControlMode):
        with self.lock:
            self._require_open()
            if self.active:
                raise DomainError("busy", "A motion is already active.")
            with self.cancel_lock:
                generation = self.stop_generation
            state = self.backend.snapshot()
            self._check(state, gripper=isinstance(command, GripperMove), require_position_mode=not isinstance(command, ControlMode))
            if isinstance(command,GripperMove) and command.completion=='bilateral_contact':
                if self.backend.name!='mujoco' or self.settings.control_profile=='calibration':
                    raise DomainError('contact_completion_unavailable','Bilateral contact completion requires the MuJoCo force sensor adapter.',422)
                if command.width_m>=state.gripper_width_m:
                    raise DomainError('contact_completion_requires_closing','Contact completion is valid only for a closing command.',422)
            recovery = None
            if self.policy.force:
                recovery = self._evaluate_interaction(command, state).get("recovery_bounds_deg")
            if isinstance(command, JointMove):
                if recovery is None and any(not low <= q <= high for q, (low, high) in zip(command.joints_deg, JOINT_LIMITS_DEG)):
                    raise DomainError("joint_limits", "Target exceeds PiperX joint limits.", 422)
                if self.measured_limits and self.measured_limits.get("available") and self.measured_limits.get("connection_epoch") == self.epoch:
                    if any(not row["min_deg"] <= q <= row["max_deg"] for q, row in zip(command.joints_deg, self.measured_limits["joints"])):
                        raise DomainError("measured_joint_limits", "Target exceeds the controller's queried limits.", 422)
                delta = max(abs(a - b) for a, b in zip(command.joints_deg, state.q_deg))
                calibration = self.settings.control_profile == "calibration"
                if calibration and delta > self.settings.max_move_deg:
                    raise DomainError("step_limit", "Target exceeds the configured per-action displacement.", 422)
                if calibration and command.speed_percent > self.settings.max_speed_percent:
                    raise DomainError("speed_limit", "Speed exceeds the configured controller limit.", 422)
                if calibration and command.timeout_s < delta / self.settings.reference_deg_s + 0.5:
                    raise DomainError("timeout_too_short", "Timeout is shorter than the bounded reference trajectory.", 422)
            elif isinstance(command, ControlMode):
                self._check_window(state, command)
                if recovery is None and any(not low <= q <= high for q, (low, high) in zip(state.q_deg, JOINT_LIMITS_DEG)):
                    raise DomainError("joint_limits", "Measured pose cannot be preloaded outside joint limits.", 422)
            elif command.width_m > self.settings.gripper_max_m or (self.settings.control_profile == "calibration" and command.effort_protocol > self.settings.gripper_effort_limit):
                raise DomainError("gripper_limits", "Requested width or protocol effort exceeds configured bounds.", 422)
            now = time.monotonic()
            self.plans = {k: v for k, v in self.plans.items() if now < v["deadline"]}
            if len(self.plans) >= 128:
                raise DomainError("too_many_plans", "Wait for earlier plans to expire.", 429)
            ident = str(uuid.uuid4())
            plan = {"plan_id": ident, "command": command.model_dump(), "start_deg": state.q_deg.copy(),
                    "start_gripper_m": state.gripper_width_m, "epoch": self.epoch,
                    "expires_at": time.time() + self.settings.plan_ttl_s,
                    "deadline": now + self.settings.plan_ttl_s, "consumed": False,
                    "stop_generation": generation,
                    "checks": "Bounds and fresh state only; no scene collision or grasp feasibility check."}
            if recovery is not None:
                plan["recovery_bounds_deg"] = recovery
            self.plans[ident] = plan
            self.store.event("plan_created", plan_id=ident, command=plan["command"], start_deg=plan["start_deg"])
            return {k: copy.deepcopy(v) for k, v in plan.items() if k != "deadline"}

    def execute(self, req: ExecuteRequest):
        with self.lock:
            self._require_open()
            previous = self.store.get(request_id=req.request_id)
            if previous:
                if previous["plan_id"] != req.plan_id:
                    raise DomainError("idempotency_conflict", "This request_id already refers to another plan.")
                return previous
            self._require_session()
            with self.cancel_lock:
                acceptance_generation = self.stop_generation
            if self.active:
                raise DomainError("busy", "Exactly one action may execute at a time.")
            plan = self.plans.get(req.plan_id)
            if not plan or plan["epoch"] != self.epoch or time.monotonic() >= plan["deadline"]:
                raise DomainError("expired_plan", "Preview a new plan against current feedback.")
            if plan["consumed"]:
                raise DomainError("consumed_plan", "This plan was already submitted. Reuse its original request_id.")
            command = {"joint": JointMove, "gripper": GripperMove, "control_mode": ControlMode}[plan["command"]["kind"]](**plan["command"])
            state = self.backend.snapshot()
            self._check(state, gripper=isinstance(command, GripperMove), require_position_mode=not isinstance(command, ControlMode))
            self._check_window(state, command)
            if max(abs(a - b) for a, b in zip(state.q_deg, plan["start_deg"])) > 0.1:
                raise DomainError("state_changed", "Robot moved since preview; preview a new plan.")
            if isinstance(command, GripperMove) and abs(state.gripper_width_m - plan["start_gripper_m"]) > 0.001:
                raise DomainError("state_changed", "Gripper moved since preview.")
            job = {"job_id": str(uuid.uuid4()), "request_id": req.request_id, "plan_id": req.plan_id,
                   "status": "accepted", "command": plan["command"], "accepted_at": time.time(),
                   "simulation": self.backend.name in ("sim", "mujoco"), "physical_validation": False,
                   "command_attempted": False, "stop_result": None}
            if "primitive" in plan:
                job.update(primitive=copy.deepcopy(plan["primitive"]), resolution=copy.deepcopy(plan["resolution"]))
            if self._queue_approval(job, command, plan, state):
                return copy.deepcopy(job)
            return self._start_job(job, command, plan)

    def _start_job(self, job, command, plan, *, new=True):
        # Caller holds self.lock. Admission is durable before the worker can write.
        self._require_session()
        with self.control_lock:
            with self.cancel_lock:
                if plan["stop_generation"] != self.stop_generation:
                    raise DomainError("cancelled", "Stop requested while preparing the action; no command sent.")
                if self.interaction_required or self.settings.control_profile == "direct" or isinstance(command, ControlMode):
                    self.cancel.clear()
            self.store.put(job, new=new)
            plan["consumed"] = True
            self.active = job["job_id"]
        self.thread = threading.Thread(target=self._run, args=(job, command, copy.deepcopy(plan)),
                                       name="piperx-action", daemon=True)
        self.thread.start()
        return copy.deepcopy(job)

    def move(self, command: JointMove | GripperMove | ControlMode, request_id: str):
        """Direct model entry; internal preview and acceptance remain atomic."""
        with self.lock:
            previous = self.store.get(request_id=request_id)
            if previous:
                if "primitive" in previous or _command_defaults(previous["command"]) != command.model_dump():
                    raise DomainError("idempotency_conflict", "This request_id already refers to another command.")
                return previous
            plan = self.preview(command)
            return self.execute(ExecuteRequest(plan_id=plan["plan_id"], request_id=request_id))

    def preview_primitive(self, command):
        """Resolve model parameters once, then use the existing durable action owner."""
        from .primitives import resolve_primitive
        with self.lock:
            self._require_open()
            if self.policy.force and command.kind != "set_gripper":
                raise DomainError("recovery_joint_only", "Force recovery supports explicit joint targets and control-mode switching; turn force off for Cartesian primitives.", 422)
            original = command.model_dump()
            if self.active:
                raise DomainError("busy", "Exactly one action may execute at a time.")
            with self.cancel_lock:
                generation = self.stop_generation
            sample = self.backend.snapshot()
            self._check(sample, gripper=command.kind == "set_gripper")
            resolved, metadata = resolve_primitive(command, sample.q_deg.copy(), self.settings)
            for point in metadata.get("waypoints_deg", []):
                self._check_window(sample, JointMove(joints_deg=point, speed_percent=command.speed_percent,
                                                     timeout_s=command.timeout_s))
                if self.measured_limits and self.measured_limits.get("available") and self.measured_limits.get("connection_epoch") == self.epoch:
                    if any(not row["min_deg"] <= q <= row["max_deg"] for q, row in zip(point, self.measured_limits["joints"])):
                        raise DomainError("measured_joint_limits", "An intermediate path sample exceeds queried controller limits.", 422)
            metadata = metadata | {"seed_joints_deg": sample.q_deg.copy(),
                                   "tcp_offset_m": self.settings.tcp_offset_m.copy(),
                                   "tcp_offset_rpy_deg": self.settings.tcp_offset_rpy_deg.copy(),
                                   "robot_model": "piper_x"}
            plan = self.preview(resolved)
            internal = self.plans[plan["plan_id"]]
            internal.update(primitive=original, resolution=metadata, start_deg=sample.q_deg.copy(),
                            stop_generation=generation)
            return {k: copy.deepcopy(v) for k, v in internal.items() if k != "deadline"}

    def primitive(self, command, request_id: str):
        with self.lock:
            self._require_open()
            previous = self.store.get(request_id=request_id)
            if previous:
                if _command_defaults(previous.get("primitive")) != command.model_dump():
                    raise DomainError("idempotency_conflict", "This request_id already refers to another command.")
                return previous
            plan = self.preview_primitive(command)
            return self.execute(ExecuteRequest(plan_id=plan["plan_id"], request_id=request_id))

    def _check_cancel_deadline(self, deadline):
        if self.cancel.is_set():
            raise DomainError("cancelled", "Stop requested.")
        if time.monotonic() >= deadline:
            raise DomainError("execution_timeout", "Action deadline reached.")

    def _guard_running(self, command, deadline):
        self._check_cancel_deadline(deadline)
        state = self.backend.snapshot()
        try:
            self._check(state, gripper=isinstance(command, GripperMove), moving=True,
                        require_position_mode=not isinstance(command, ControlMode) and not self.native_linear_active)
            if self.native_linear_active and (state.ctrl_mode != 1 or state.motion_mode not in (1, 2)):
                raise DomainError("control_mode", "Native line requires CAN position control / MOVE_L.")
        except DomainError as exc:
            # Preserve the exact rejected sample, not a later stationary read.
            exc.rejected_state = state.public()
            raise
        self._check_window(state, command)
        from .recovery_limits import worker_bounds, check_bounds
        if worker_bounds() is not None:
            check_bounds(state.q_deg, worker_bounds(), "recovery_limit")
        # snapshot() may block while stop arrives or the deadline expires.
        self._check_cancel_deadline(deadline)
        return state

    def _run(self, job, command, plan):
        from .recovery_limits import recovery_scope
        with recovery_scope(plan.get("recovery_bounds_deg")):
            self._run_motion(job, command, plan)

    def _run_motion(self, job, command, plan):
        deadline = time.monotonic() + command.timeout_s
        last_write_at, last_settled_stamp = time.monotonic(), None
        last_settled_q = None
        last_settled_width = None
        last_observed_stamp = None
        calibration = self.settings.control_profile == "calibration"
        waypoints = plan.get("resolution", {}).get("waypoints_deg", [])
        native_linear = plan.get("resolution", {}).get("native_controller", False)
        waypoint_index = 0
        # Only the MuJoCo direct-profile motions this recovery covers get the
        # measured-progress watchdog; real CAN/calibration/control-mode paths
        # keep their existing deadline-only behaviour.
        track_progress = self._ordinary_recovery_command(command) and not native_linear
        progress_watch = _ProgressWatch(time.monotonic())
        motion_identity = None
        try:
            with self.lock:
                state = self._guard_running(command, deadline)
                motion_identity = self._connection_identity(state)
                job["status"] = "running"
                self.store.put(job)
                job["command_attempted"] = True
                self.store.put(job)
                self.store.event("motion_start", job_id=job["job_id"], command=command.model_dump())
                self._check_cancel_deadline(deadline)
                if isinstance(command, ControlMode):
                    def mode_checkpoint():
                        latest = self._guard_running(command, deadline)
                        if max(abs(a-b) for a,b in zip(latest.q_deg, state.q_deg)) > .1:
                            raise DomainError("state_changed", "Measured pose changed while preparing control mode.")
                    self.backend.set_control_mode(state.q_deg, command.speed_percent, mode_checkpoint)
                    self.plans.clear()
                elif isinstance(command, JointMove):
                    if native_linear:
                        self.native_linear_active = True
                        self.backend.linear_target(state.q_deg, plan["resolution"]["target_flange_pose"], command.speed_percent,
                                                   lambda: self._check_cancel_deadline(deadline))
                    else:
                        self.backend.begin_joint(state.q_deg, command.speed_percent)
                    if not calibration and not waypoints and not native_linear:
                        self._guard_running(command, deadline)
                        self.backend.joint_target(command.joints_deg)
                elif not calibration:
                    self.backend.gripper_target(command.width_m, command.effort_protocol)
                last_write_at = time.monotonic()
            current = state.q_deg.copy()
            width = state.gripper_width_m
            settled = 0
            while True:
                if self.cancel.wait(0.05):
                    raise DomainError("cancelled", "Stop requested.")
                with self.lock:
                    state = self._guard_running(command, deadline)
                    if isinstance(command, ControlMode):
                        reached = (state.ctrl_mode == 1 and state.motion_mode == 1 and state.teach_status in (0, 2, 6)
                                   and max(abs(a-b) for a,b in zip(state.q_deg, current)) < .12)
                    elif native_linear:
                        from .kinematics import PiperKinematics
                        from scipy.spatial.transform import Rotation
                        import numpy as np
                        pose = PiperKinematics(self.settings.tcp_offset_m, self.settings.tcp_offset_rpy_deg).pose(state.q_deg)
                        target = plan["resolution"]["target_tcp_pose"]
                        angular = (Rotation.from_euler("xyz", pose["rpy_deg"], degrees=True).inv() *
                                   Rotation.from_euler("xyz", target["rpy_deg"], degrees=True)).magnitude()
                        reached = state.motion_mode == 2 and np.linalg.norm(np.array(pose["xyz_m"])-target["xyz_m"]) < .0005 and angular < math.radians(.1)
                    elif waypoints and waypoint_index < len(waypoints):
                        target = waypoints[waypoint_index]
                        self._check_window(state, JointMove(joints_deg=target, speed_percent=command.speed_percent,
                                                            timeout_s=command.timeout_s))
                        tracking = max(abs(a-b) for a, b in zip(state.q_deg, current))
                        if tracking > 0.3:
                            raise DomainError("tracking_error", "Linear reference is not being tracked.")
                        step = self.settings.reference_deg_s * 0.05 if calibration else 0.1
                        # Wait for measured tracking before advancing the
                        # sampled reference. Repeatedly advancing while a
                        # speed/acceleration-limited servo catches up can
                        # manufacture a tracking fault on a clear path. Keep
                        # the existing 0.3 degree rejection and deadline.
                        next_q = ([q + max(-step, min(step, t-q)) for q, t in zip(current, target)]
                                  if tracking <= .12 else current)
                        self._check_cancel_deadline(deadline)
                        self.backend.joint_target(next_q)
                        last_write_at = time.monotonic()
                        current = next_q
                        if max(abs(a-b) for a,b in zip(current, target)) < 1e-8:
                            waypoint_index += 1
                        reached = False
                    elif waypoints:
                        reached = max(abs(a-b) for a,b in zip(state.q_deg, command.joints_deg)) < .12
                    elif not calibration:
                        reached = (max(abs(a - b) for a, b in zip(state.q_deg, command.joints_deg)) < .12
                                   if isinstance(command, JointMove) else abs(state.gripper_width_m - command.width_m) < .001)
                        if isinstance(command,GripperMove) and command.completion=='bilateral_contact':
                            contact=state.diagnostics.get('gripper_contact',{})
                            stamp_contact=contact.get('source_timestamp_s')
                            reached=(contact.get('supported') is True and stamp_contact is not None and
                                     0<=time.monotonic()-stamp_contact<=self.settings.feedback_timeout_s)
                    elif isinstance(command, JointMove):
                        if max(abs(a - b) for a, b in zip(state.q_deg, current)) > 0.3:
                            raise DomainError("tracking_error", "Measured joints failed to track the bounded reference.")
                        step = min(0.1, self.settings.reference_deg_s * 0.05)
                        next_q = [q + max(-step, min(step, target - q)) for q, target in zip(current, command.joints_deg)]
                        at_target = all(abs(a - b) < 1e-8 for a, b in zip(next_q, command.joints_deg))
                        if next_q != current:
                            self.store.event("joint_reference", job_id=job["job_id"], joints_deg=next_q)
                            self._check_cancel_deadline(deadline)
                            self.backend.joint_target(next_q)
                            last_write_at = time.monotonic()
                        current = next_q
                        reached = at_target and max(abs(a - b) for a, b in zip(state.q_deg, command.joints_deg)) < 0.12
                    else:
                        if abs(state.gripper_width_m - width) > 0.004:
                            raise DomainError("tracking_error", "Gripper cannot track requested width (possibly contact); goal not confirmed.")
                        next_width = width + max(-0.0005, min(0.0005, command.width_m - width))
                        if next_width != width:
                            self.store.event("gripper_reference", job_id=job["job_id"], width_m=next_width,
                                             effort_protocol=command.effort_protocol)
                            self._check_cancel_deadline(deadline)
                            self.backend.gripper_target(next_width, command.effort_protocol)
                            last_write_at = time.monotonic()
                        width = next_width
                        reached = abs(width - command.width_m) < 1e-8 and abs(state.gripper_width_m - command.width_m) < 0.001
                    if track_progress and not reached:
                        # Watch the measured feedback vector, not the scalar
                        # error to the final commanded joints: a curved path can
                        # keep one axis's final-target error flat while other
                        # joints move, and that is still real progress.
                        if isinstance(command, JointMove):
                            measured = tuple(state.q_deg)
                            epsilon = _STALL_MIN_PROGRESS_DEG
                        else:
                            measured = (state.gripper_width_m,)
                            epsilon = _STALL_MIN_PROGRESS_M
                        if progress_watch.stalled(time.monotonic(), measured, epsilon):
                            raise DomainError(
                                "motion_stalled",
                                "Measured motion stopped progressing toward the target while the target was "
                                "not reached; the command is blocked at the measured pose.", 409)
                    stamp = state.feedback_stamp_s
                    if isinstance(command, GripperMove):
                        stamp = min(stamp, state.gripper_stamp_s) if stamp is not None and state.gripper_stamp_s is not None else None
                    stable = (max(abs(v) for v in state.velocity_deg_s) < .5 if calibration else
                              last_settled_q is not None and max(abs(a-b) for a,b in zip(state.q_deg, last_settled_q)) < .02)
                    if not calibration and isinstance(command, GripperMove):
                        stable = stable and last_settled_width is not None and abs(state.gripper_width_m - last_settled_width) < .0001
                    verified = reached and stamp is not None and stamp > last_write_at and stable
                    if verified and stamp != last_settled_stamp:
                        settled += 1
                        last_settled_stamp = stamp
                    elif not verified:
                        settled = 0
                    if stamp is not None and stamp > last_write_at and stamp != last_observed_stamp:
                        last_settled_q = state.q_deg.copy()
                        last_settled_width = state.gripper_width_m
                        last_observed_stamp = stamp
                    if settled >= 3:
                        job.update(status="succeeded", after=state.public(), verification="Fresh feedback confirmed CAN/MOVE_J at the measured pose; no reset or enable command sent." if isinstance(command, ControlMode) else "Fresh robot feedback reached target; no visual/contact success claim.")
                        if isinstance(command,GripperMove) and command.completion=='bilateral_contact':
                            job['verification']='Three fresh stable samples confirmed bilateral forces on one dynamic object; lift/task success is not implied.'
                            job['contact_evidence']=state.diagnostics['gripper_contact']
                        break
        except Exception as exc:
            with self.lock:
                self.lease = None
                hold = self._hold(command) if job["command_attempted"] else {"status": "not_needed"}
                error_code = getattr(exc, "code", "execution_error")
                recovery = None
                if (job["command_attempted"] and motion_identity is not None
                        and error_code in _BLOCKED_RECOVERY_CODES
                        and self._ordinary_recovery_command(command)):
                    recovery = self._blocked_motion_recovery(job, command, motion_identity, hold, error_code, exc)
                if recovery is not None:
                    # Failed and recoverable, never a success claim.
                    job.update(status="failed", error_code="motion_blocked", recoverable=True,
                               original_error_code=error_code, error=str(exc),
                               stop_result=recovery["stop_result"], after=recovery["after"])
                else:
                    job.update(status="cancelled" if isinstance(exc, DomainError) and error_code == "cancelled" and hold["status"] in ("hold_requested", "not_needed") else
                               "outcome_unknown" if job["command_attempted"] else "failed",
                               error=str(exc), error_code=error_code, stop_result=hold)
                if hasattr(exc, "rejected_state"):
                    job["rejected_state"] = exc.rejected_state
        finally:
            with self.lock:
                job["finished_at"] = time.time()
                try:
                    self.store.put(job)
                    self.store.event("job_finished", job_id=job["job_id"], status=job["status"])
                finally:
                    self.active = None
                    self.native_linear_active = False

    def _hold(self, command=None):
        """Never substitute the vendor's damping/downward stop for a position hold."""
        try:
            if self.estop_latched:
                return {"status": "estop_latched", "confirmed_stopped": False}
            state = self.backend.snapshot()
            self._check(state, gripper=isinstance(command, GripperMove), moving=True, require_position_mode=not self.native_linear_active)
            if self.native_linear_active:
                if state.ctrl_mode != 1 or state.motion_mode not in (1, 2):
                    raise DomainError("control_mode", "Cannot hold outside CAN position modes.")
                self.backend.set_control_mode(state.q_deg, 1, lambda: None)
            audit_error = None
            try:
                self.store.event("hold_request", q_deg=state.q_deg)
            except Exception as exc:
                # A failed journal must not prevent a best-effort stopping write.
                audit_error = str(exc)
            if self.backend.name=='mujoco' and hasattr(self.backend,'hold_position'):
                self.backend.hold_position(hold_gripper=isinstance(command,GripperMove))
            else:
                self.backend.joint_target(state.q_deg)
                if isinstance(command, GripperMove):
                    self.backend.gripper_target(state.gripper_width_m, command.effort_protocol)
            # Monotonic instant immediately after the position-hold write. Only
            # feedback source stamps strictly later than this are accepted as
            # post-hold stop evidence, so cached pre-hold frames never count.
            hold_requested_at = time.monotonic()
            return {"status": "hold_requested", "confirmed_stopped": False, "audit_error": audit_error,
                    "hold_requested_monotonic_s": hold_requested_at,
                    "message": "Fresh measured position requested; verify physical standstill. This is not a hardware emergency stop."}
        except Exception as exc:
            return {"status": "outcome_unknown", "confirmed_stopped": False,
                    "message": "Cannot confirm stop; use the on-site power/emergency control.", "error": str(exc)}

    def _ordinary_recovery_command(self, command) -> bool:
        """True only for the MuJoCo direct-profile motions this recovery covers.

        Ordinary joint moves and ordinary/contact gripper completions. A contact
        completion that never reports contact (an empty/missed closing) or stops
        making progress is still only ever re-expressed as failed/recoverable
        after a confirmed post-hold stop; it is never a success. Calibration,
        control-mode transitions and native controller lines keep their existing
        outcome handling.
        """
        if self.backend.name != "mujoco" or self.settings.control_profile != "direct":
            return False
        if isinstance(command, JointMove):
            return True
        return isinstance(command, GripperMove) and command.completion in ("width", "bilateral_contact")

    def _connection_identity(self, state):
        diagnostics = getattr(state, "diagnostics", None)
        return (self.instance_id, self.epoch,
                diagnostics.get("epoch") if isinstance(diagnostics, dict) else None)

    def _identity_unchanged(self, identity, state) -> bool:
        return self._connection_identity(state) == identity

    def _hold_evidence_stamp(self, command, state):
        """Older of the source timestamps required for post-hold stop evidence.

        A gripper move additionally binds the gripper source timestamp, so a
        frozen or cached gripper frame cannot ride on fresh arm feedback.
        """
        stamp = state.feedback_stamp_s
        if isinstance(command, GripperMove):
            if stamp is None or state.gripper_stamp_s is None:
                return None
            stamp = min(stamp, state.gripper_stamp_s)
        return stamp

    def _contact_evidence_stale(self, command, state) -> bool:
        """True when a contact completion carries unusable required contact feedback.

        A missing/empty contact diagnostic means no contact was reported (an
        empty/missed closing): that never supports success, but by itself it
        does not forbid a recovery either. A present contact reading whose
        source stamp is missing, non-finite, in the future or older than the
        feedback timeout is stale required feedback and must keep the outcome
        unknown instead of underwriting a stop claim.
        """
        if not (isinstance(command, GripperMove) and command.completion == "bilateral_contact"):
            return False
        diagnostics = getattr(state, "diagnostics", None)
        contact = diagnostics.get("gripper_contact") if isinstance(diagnostics, dict) else None
        if not isinstance(contact, dict) or not contact:
            return False
        stamp = contact.get("source_timestamp_s")
        if stamp is None or not isinstance(stamp, (int, float)) or not math.isfinite(stamp):
            return True
        age = time.monotonic() - stamp
        return age < 0 or age > self.settings.feedback_timeout_s

    def _confirm_measured_hold(self, command, identity, hold):
        """Confirm standstill from fresh samples stamped after the hold write.

        Returns (evidence, None) only when the hold was journalled and at least
        three feedback samples whose source timestamps strictly advance after
        the position-hold write, over a settled interval, show low measured
        velocity and low measured joint/width change on the same connection.
        The sample that completes that window is never accepted on its own: one
        more read must arrive, be a strictly advancing post-hold required stamp
        and stay inside the same velocity/joint/width bounds, and that actual
        latest public sample is what gets returned. A last read that is still
        moving resets the settled window, and a stale/replayed last read is
        skipped, so neither can certify standstill; verification stays bounded
        by the hold deadline. Cached pre-hold frames and out-of-order frames are
        never counted as new stop evidence. Otherwise returns (None, reason): an
        unconfirmed hold is never recoverable.
        """
        if hold.get("status") != "hold_requested":
            return None, "measured position hold was not requested"
        if hold.get("audit_error"):
            return None, "hold request could not be journalled; not recoverable"
        hold_requested_at = hold.get("hold_requested_monotonic_s")
        if hold_requested_at is None or not math.isfinite(hold_requested_at):
            return None, "hold request time is unavailable; not recoverable"
        deadline = time.monotonic() + _RECOVERY_HOLD_MAX_S
        samples, reference_q, reference_width = [], None, None
        last_stamp = None
        # Set once the settled window is complete; the next fresh in-bounds read
        # is the one that certifies standstill and is returned as `after`.
        window_ready = False
        gripper = isinstance(command, GripperMove)
        reason = "no fresh advancing feedback after the hold write arrived while the hold was active"
        while time.monotonic() < deadline:
            if self.cancel.is_set():
                return None, "stop requested while confirming the hold; not recoverable"
            state = self.backend.snapshot()
            if not self._identity_unchanged(identity, state):
                return None, "connection identity changed while confirming the hold"
            try:
                self._check(state, gripper=gripper, moving=True, require_position_mode=True)
            except DomainError as exc:
                return None, f"cannot confirm standstill ({exc.code})"
            if self._contact_evidence_stale(command, state):
                # Required contact feedback is unusable: fail closed now instead
                # of waiting out the hold deadline on a broken channel.
                return None, "required gripper contact feedback is stale or invalid; not recoverable"
            stamp = self._hold_evidence_stamp(command, state)
            # Only strictly advancing timestamps produced after the hold write
            # are new stop evidence: equal, cached pre-hold and out-of-order
            # stamps are skipped without resetting the settled window.
            if (stamp is None or not math.isfinite(stamp) or stamp <= hold_requested_at
                    or (last_stamp is not None and stamp <= last_stamp)):
                time.sleep(_RECOVERY_HOLD_SAMPLE_INTERVAL_S)  # wait for a fresh frame
                continue
            last_stamp = stamp
            velocity = max((abs(v) for v in state.velocity_deg_s if math.isfinite(v)), default=math.inf)
            if reference_q is None:
                reference_q, reference_width = state.q_deg.copy(), state.gripper_width_m
            if (velocity > _RECOVERY_HOLD_MAX_VELOCITY_DEG_S
                    or max(abs(a - b) for a, b in zip(state.q_deg, reference_q)) > _RECOVERY_HOLD_MAX_JOINT_DELTA_DEG
                    or abs(state.gripper_width_m - reference_width) > _RECOVERY_HOLD_MAX_WIDTH_DELTA_M):
                # Still moving on a fresh post-hold read: this read cannot
                # certify a stop, so the settled window restarts at it.
                samples, reference_q, reference_width = [], state.q_deg.copy(), state.gripper_width_m
                window_ready = False
                reason = "measured motion did not settle during the hold"
                time.sleep(_RECOVERY_HOLD_SAMPLE_INTERVAL_S)
                continue
            samples.append({"stamp": stamp, "q_deg": state.q_deg.copy(),
                            "gripper_width_m": state.gripper_width_m, "max_velocity_deg_s": velocity})
            span = samples[-1]["stamp"] - samples[0]["stamp"]
            if not window_ready:
                window_ready = len(samples) >= _RECOVERY_HOLD_MIN_SAMPLES and span >= _RECOVERY_HOLD_MIN_SPAN_S
                time.sleep(_RECOVERY_HOLD_SAMPLE_INTERVAL_S)
                continue
            # Fresh, advancing and inside every bound: this read is the latest
            # public sample and the one the recovery is certified on.
            confirmation = {
                "method": "bounded measured position hold; fresh advancing measured samples",
                "samples": len(samples), "settled_span_s": round(span, 4),
                "max_velocity_deg_s": round(max(s["max_velocity_deg_s"] for s in samples), 4),
                "max_joint_delta_deg": round(max(max(abs(a - b) for a, b in zip(s["q_deg"], reference_q)) for s in samples), 4),
                "max_width_delta_m": round(max(abs(s["gripper_width_m"] - reference_width) for s in samples), 6)}
            return {"stop_result": dict(hold, confirmed_stopped=True, confirmation=confirmation),
                    "after": state.public()}, None
        return None, reason

    def _blocked_motion_recovery(self, job, command, identity, hold, original_code, exc):
        """Re-express one confirmed blocked MuJoCo motion as failed/recoverable."""
        try:
            evidence, reason = self._confirm_measured_hold(command, identity, hold)
        except Exception as err:
            # An unexpected confirmation failure is an unconfirmed stop, never
            # a recoverable claim; the job keeps its existing unknown outcome.
            hold["stop_confirmation"] = {"confirmed_stopped": False,
                                         "reason": f"hold confirmation failed ({type(err).__name__})"}
            return None
        if evidence is None:
            hold["stop_confirmation"] = {"confirmed_stopped": False, "reason": reason}
            return None
        confirmation = evidence["stop_result"]["confirmation"]
        try:
            self.store.event("motion_blocked_recovery", job_id=job["job_id"], error_code=original_code,
                             reason=str(exc), samples=confirmation["samples"],
                             settled_span_s=confirmation["settled_span_s"])
        except Exception as err:
            # Failure to persist the recovery claim is not recoverable.
            hold["stop_confirmation"] = {"confirmed_stopped": False,
                                         "reason": f"recovery audit write failed ({type(err).__name__})"}
            return None
        return evidence

    def stop(self):
        with self.cancel_lock:
            self.stop_generation += 1
            self.cancel.set()
        with self.control_lock:
            self.cancel.set()
            self.lease = None
            if self.lock.acquire(blocking=False):
                try:
                    self._cancel_pending("stop_requested")
                finally:
                    self.lock.release()
            if self.active:
                # Worker owns writes. An already blocked native call cannot be interrupted here.
                return {"status": "cancellation_requested", "job_id": self.active, "confirmed_stopped": False}
            return {"status": "idle", "confirmed_stopped": False, "message": "No middleware action active; control window closed. No CAN command sent."}

    def get_job(self, ident):
        job = self.store.get(job_id=ident)
        if job is None:
            raise DomainError("not_found", "Unknown job identifier.", 404)
        return job

    def get_request(self, request_id):
        job = self.store.get(request_id=request_id)
        if job is None:
            raise DomainError("not_found", "Unknown request identifier.", 404)
        return job

    def shutdown_idle(self, expected_instance_id: str):
        with self.lock:
            if expected_instance_id != self.instance_id:
                raise DomainError("instance_mismatch", "Executor instance changed; nothing was shut down.", 409)
            if self.active:
                raise DomainError("busy", "Stop and resolve the active action before shutting down the executor.", 409)
            self.closing = True
        self._supervisor_exit.set()
        if self._supervisor is not threading.current_thread():
            self._supervisor.join(1)
        self.close()
        return {"status": "resources_released", "process_id": os.getpid(),
                "instance_id": self.instance_id, "robot_stop_command_sent": False}

    def close(self):
        with self.lock:
            if self.closed:
                return
            self.closing = True
        self._supervisor_exit.set()
        if self._supervisor is not threading.current_thread():
            self._supervisor.join(1)
        self.stop()
        if self.thread:
            self.thread.join(3)
            if self.thread.is_alive():
                raise RuntimeError("Executor did not stop; refusing to close CAN while its worker is alive.")
        with self.lock:
            from .device_discovery import cleanup_discovery
            cleanup_discovery(self)
            self.backend.close()
            self.store.close()
            self.closed = True
