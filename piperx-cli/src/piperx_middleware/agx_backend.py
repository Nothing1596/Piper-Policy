"""PiperX SDK adapter. Importing this module never opens hardware.

Every outgoing frame must match the exact transaction opened by an SDK call.
No reset/teach/enable/zero command is exposed or permitted.
"""
from collections import Counter, deque
from contextlib import contextmanager
import copy
import math
from pathlib import Path
import struct
import sys
import tempfile
import threading
import time

import can

from .backends import State
from .models import DomainError, JOINT_LIMITS_DEG, Settings
from .process_lock import ProcessLock
from .sdk_controls import SDKControls

_owner = None
_owner_lock = threading.Lock()


def joint_frames(q_deg):
    q = [round(math.radians(v) * (180 / math.pi) * 1000) for v in q_deg]
    return [(0x155 + i, struct.pack(">ii", q[2*i], q[2*i+1])) for i in range(3)]


class GuardedBus(can.BusABC):
    _SHUTDOWN_LOCK_TIMEOUT_S = 1.0

    def __init__(self, channel, **kwargs):
        global _owner
        self.owner = _owner
        if self.owner is None:
            raise RuntimeError("No hardware owner")
        self.inner = self.owner._open_transport(channel, **kwargs)
        self._shutdown_lock = threading.RLock()
        super().__init__(channel=channel)
        self.owner.bus = self

    def _recv_internal(self, timeout):
        try:
            msg = self.inner.recv(timeout)
            self.owner._rx_receipt_stamp = getattr(self.inner, "last_dequeued_received_s", None)
            return msg, False
        except Exception as exc:
            self.owner.fault = f"CAN receive failed: {type(exc).__name__}: {exc}"
            raise

    def send(self, msg, timeout=None):
        owner = self.owner
        with owner.tx_lock:
            valid = (not self._is_shutdown and owner.settings.allow_motion and owner.tx_thread == threading.get_ident() and
                     (not owner.estopped or owner.tx_emergency) and
                     owner.expected and owner.expected[0] == (msg.arbitration_id, bytes(msg.data)) and
                     msg.dlc == 8 and not any((msg.is_extended_id, msg.is_remote_frame, msg.is_error_frame,
                                              msg.is_fd, msg.bitrate_switch)))
            if not valid:
                owner.fault = "Unexpected CAN transmission blocked"
                raise DomainError("tx_blocked", owner.fault)
            owner.expected.popleft()
            try:
                self.inner.send(msg, timeout=0.1)
                owner.tx_count += 1
            except Exception as exc:
                owner.fault = f"CAN send failed: {type(exc).__name__}"
                raise

    def shutdown(self):
        # SDK receive failures also close this bus, independently of RobotService.
        # Match the transaction's lock order; never free a handle during native TX.
        deadline = time.monotonic() + self._SHUTDOWN_LOCK_TIMEOUT_S
        if not self.owner.tx_lock.acquire(timeout=self._SHUTDOWN_LOCK_TIMEOUT_S):
            raise can.CanOperationError("CAN transmit is still active; cleanup deferred")
        try:
            if not self._shutdown_lock.acquire(timeout=max(0.0, deadline - time.monotonic())):
                raise can.CanOperationError("CAN shutdown is still active; cleanup deferred")
            try:
                if not self._is_shutdown:
                    self.inner.shutdown()
                    super().shutdown()
            finally:
                self._shutdown_lock.release()
        finally:
            self.owner.tx_lock.release()


class AgxBackend(SDKControls):
    name = "agx"

    def __init__(self, settings: Settings, transport_factory=None):
        self.settings = settings
        self.estopped = False
        self.tx_emergency = False
        self.transport_factory = transport_factory
        self.arm = self.gripper = self.bus = self.device_lock = None
        self.expected = deque()
        self.tx_lock, self.rx_lock = threading.RLock(), threading.RLock()
        self.tx_thread = None
        self.counts, self.last_seen = Counter(), {}
        self.tx_count, self.fault = 0, None
        self._closing_threads = []
        self._pending_transport = None
        self._rx_receipt_stamp = None

    def _open_transport(self, channel, **kwargs):
        if self.transport_factory:
            return self.transport_factory()
        if self.settings.can_interface == "agx_cando":
            if self.settings.cando_source:
                sys.path.insert(0, str(self.settings.cando_source))
            from .cando_transport import SessionCandoBus
            # Retain the object even if native initialization/cleanup raises.
            transport = SessionCandoBus.__new__(SessionCandoBus)
            self._pending_transport = transport
            transport.__init__(channel=channel, device_index=self.settings.cando_device_index, bitrate=1_000_000,
                               local_loopback=False, receive_own_messages=False)
            return transport
        return can.Bus(interface="socketcan", channel=channel, bitrate=1_000_000,
                       local_loopback=False, receive_own_messages=False)

    def connect(self):
        global _owner
        if self.arm is not None and self.arm.is_connected():
            return
        if self.arm is not None or self.device_lock is not None:
            self.close()
        with _owner_lock:
            if _owner is not None and _owner is not self:
                raise DomainError("hardware_owned", "Another executor already owns CAN.")
            _owner = self
        try:
            # A new connection must never inherit recent-looking samples from the old one.
            with self.rx_lock:
                self.counts.clear()
                self.last_seen.clear()
                self.tx_count = 0
                self.fault = None
                self._rx_receipt_stamp = None
            # Same OS user, independent of data-directory or listening-port choices.
            lock_name = "piperx-can-" + self.settings.can_interface + "-" + self.settings.can_channel.replace("/", "_")
            self.device_lock = ProcessLock(Path(tempfile.gettempdir()) / (lock_name + ".lock"))
            if self.settings.sdk_root:
                sys.path.insert(0, str(self.settings.sdk_root))
            from pyAgxArm import AgxArmFactory, ArmModel, create_agx_arm_config
            can.interfaces.BACKENDS["piperx_guarded"] = (__name__, "GuardedBus")
            interfaces = frozenset(can.interfaces.BACKENDS)
            can.interfaces.VALID_INTERFACES = can.util.VALID_INTERFACES = can.VALID_INTERFACES = interfaces
            config = create_agx_arm_config(robot=ArmModel.PIPER_X,
                firmeware_version=self.settings.firmware_profile, interface="piperx_guarded",
                channel=self.settings.can_channel, bitrate=1_000_000, local_loopback=False,
                receive_own_messages=False, timeout=0.02, auto_connect=False)
            self.arm = AgxArmFactory.create_arm(config)
            self.arm.set_auto_set_motion_mode_enabled(False)
            self.arm.set_joint_limits_enabled(True)
            self.gripper = self.arm.init_effector(self.arm.OPTIONS.EFFECTOR.AGX_GRIPPER)
            self.arm.get_context().register_parser_packet_fun(self._on_frame)
            self.arm.connect()
        except Exception:
            self.close()
            raise

    def _on_frame(self, msg):
        if msg.dlc != 8 or msg.is_extended_id or msg.is_error_frame or msg.is_remote_frame or msg.is_fd:
            return
        with self.rx_lock:
            self.last_seen[msg.arbitration_id] = self._rx_receipt_stamp if self._rx_receipt_stamp is not None else time.monotonic()
            self.counts[msg.arbitration_id] += 1

    def snapshot(self):
        s = State(bool(self.arm and self.arm.is_connected()), tx_frames=self.tx_count)
        with self.rx_lock:
            now = time.monotonic()
            seen, counts = self.last_seen.copy(), self.counts.copy()
        s.received_frames = sum(counts.values())
        required = [0x2A1, 0x2A5, 0x2A6, 0x2A7, *range(0x251, 0x257), *range(0x261, 0x267)]
        s.feedback_age_s = max(now - seen[i] for i in required) if all(i in seen for i in required) else None
        s.gripper_age_s = now - seen[0x2A8] if 0x2A8 in seen else None
        s.feedback_stamp_s = min(seen[i] for i in required) if all(i in seen for i in required) else None
        s.gripper_stamp_s = seen.get(0x2A8)
        s.diagnostics = {"firmware_profile": self.settings.firmware_profile,
            "profile_source": "operator configuration, not a new firmware query",
            "feedback_clock": "CANDO native receipt before Python queue (otherwise host callback); messages are not synchronized",
            "missing_ids": [hex(i) for i in required if i not in seen],
            "stale_ids": [hex(i) for i in required if i in seen and now - seen[i] > self.settings.feedback_timeout_s],
            "frames_by_id": {hex(i): v for i, v in sorted(counts.items())},
            "communication_error": self.fault, "physical_validation": False}
        s.diagnostics["software_estop_latched"] = self.estopped
        if self.bus and callable(getattr(self.bus.inner, "diagnostics", None)):
            s.diagnostics["transport"] = self.bus.inner.diagnostics()
        if not s.connected:
            return s
        try:
            def message(value):
                return copy.deepcopy(value.msg) if value is not None else None
            status = message(self.arm.get_arm_status())
            joints = message(self.arm.get_joint_angles())
            motors = [message(self.arm.get_motor_states(i)) for i in range(1, 7)]
            drivers = [message(self.arm.get_driver_states(i)) for i in range(1, 7)]
            grip = message(self.gripper.get_gripper_status())
            if 0x2A1 in seen and status:
                for source, target in (("ctrl_mode", "ctrl_mode"), ("mode_feedback", "motion_mode"),
                    ("teach_status", "teach_status"), ("arm_status", "arm_status"), ("err_code", "error_code")):
                    val = getattr(status, source)
                    setattr(s, target, int(getattr(val, "value", val)))
            if joints is not None and all(i in seen for i in (0x2A5, 0x2A6, 0x2A7)):
                s.q_deg = [math.degrees(x) for x in joints]
            if all(motors) and all(i in seen for i in range(0x251, 0x257)):
                s.velocity_deg_s = [math.degrees(m.velocity) for m in motors]
            if all(drivers) and all(i in seen for i in range(0x261, 0x267)):
                s.enabled = [bool(d.foc_status.driver_enable_status) for d in drivers]
                s.diagnostics["driver_fault"] = any(d.foc_status_code & 0xBF for d in drivers)
                s.collision_status = [bool(d.foc_status.collision_status) for d in drivers]
                s.motor_telemetry = [{"joint": i, "motor_temp_c": d.motor_temp,
                    "foc_temp_c": d.foc_temp, "bus_current_a": d.bus_current,
                    "foc_status_code": d.foc_status_code, "source": "sdk_feedback",
                    "feedback_stamp_s": seen[0x260+i], "feedback_age_s": now-seen[0x260+i]}
                    for i, d in enumerate(drivers, 1)]
            if grip and 0x2A8 in seen:
                s.gripper_width_m, s.gripper_mode = grip.value, grip.mode
                s.gripper_enabled = bool(grip.foc_status.driver_enable_status)
                s.gripper_error = bool(grip.status_code & 0x3F)
            error = self.arm.get_context().get_comm_error()
            if error:
                s.diagnostics["communication_error"] = str(error)
        except Exception as exc:
            s.diagnostics["communication_error"] = f"Feedback decode failed: {type(exc).__name__}: {exc}"
        return s

    @contextmanager
    def _transaction(self, expected, *, emergency=False):
        with self.tx_lock:
            if not self.settings.allow_motion or (self.fault and not emergency) or not self.arm or not self.arm.is_connected():
                raise DomainError("hardware_write_denied", "Hardware is read-only, disconnected, or faulted.")
            if self.estopped and not emergency:
                raise DomainError("estop_latched", "Software emergency stop is latched.")
            if self.expected:
                raise RuntimeError("Nested CAN transaction")
            self.expected = deque(expected)
            self.tx_thread = threading.get_ident()
            self.tx_emergency = emergency
            try:
                yield
                if self.expected:
                    self.fault = "SDK did not emit the complete expected transaction"
                    raise RuntimeError(self.fault)
            finally:
                self.expected.clear()
                self.tx_thread = None
                self.tx_emergency = False

    def emergency_stop(self):
        # Mark before waiting for the TX owner. New ordinary transactions are refused.
        # This still cannot interrupt a blocked native transport; never claim physical stop.
        self.estopped = True
        if not self.tx_lock.acquire(timeout=0.2):
            raise DomainError("outcome_unknown", "CAN TX is blocked; stop latch set, damping frame not sent.")
        try:
            with self._transaction([(0x150, bytes([1, 0, 0, 0, 0, 0, 0, 0]))], emergency=True):
                self.arm.electronic_emergency_stop()
        finally:
            self.tx_lock.release()

    @staticmethod
    def _validate_joints(q):
        from .recovery_limits import worker_bounds
        bounds = worker_bounds() or JOINT_LIMITS_DEG
        if len(q) != 6 or any(not math.isfinite(v) or not lo <= v <= hi for v, (lo, hi) in zip(q, bounds)):
            raise DomainError("joint_limits", "Invalid or out-of-range joint target.")

    def _move_j(self, joints_deg):
        from .recovery_limits import worker_bounds
        bounds = worker_bounds()
        if bounds is None:
            return self.arm.move_j([math.radians(v) for v in joints_deg])
        self._validate_joints(joints_deg)
        original = self.arm._config
        limits = original.get("joint_limits", {})
        if len(limits) != 6 or not self.arm.get_joint_limits_enabled():
            raise DomainError("recovery_sdk_limits", "Recovery requires the SDK's six-joint limit checker.")
        # The SDK normally silently clamps. Keep its checker enabled and replace
        # only this call's host bounds; never send controller limit-setting frames.
        self.arm._config = dict(original, joint_limits={
            name: [math.radians(lo), math.radians(hi)]
            for name, (lo, hi) in zip(limits, bounds)})
        try:
            return self.arm.move_j([math.radians(v) for v in joints_deg])
        finally:
            self.arm._config = original

    def begin_joint(self, current_deg, speed_percent):
        self._validate_joints(current_deg)
        limit = self.settings.max_speed_percent if self.settings.control_profile == "calibration" else 100
        if type(speed_percent) is not int or not 0 <= speed_percent <= limit:
            raise DomainError("speed_limit", "Invalid controller speed.")
        # Preload the measured pose before refreshing MOVE_J. Never reset/exit teaching.
        expected = joint_frames(current_deg) + [(0x151, bytes([1, 255, speed_percent, 0, 0, 0, 0, 0])),
                                                (0x151, bytes([1, 1, speed_percent, 0, 0, 0, 0, 0]))]
        with self._transaction(expected):
            self._move_j(current_deg)
            self.arm.set_speed_percent(speed_percent)
            self.arm.set_motion_mode("j")

    def set_control_mode(self, current_deg, speed_percent, checkpoint):
        self._validate_joints(current_deg)
        limit = self.settings.max_speed_percent if self.settings.control_profile == "calibration" else 100
        if type(speed_percent) is not int or not 0 <= speed_percent <= limit:
            raise DomainError("speed_limit", "Invalid controller speed.")
        # Pin the reviewed SDK message type. One combined mode frame avoids
        # set_speed_percent's intermediate CAN / move_mode=255 transition.
        mode = self.arm._MSG_ModeCtrl()
        mode.ctrl_mode = 1
        mode.move_spd_rate_ctrl = speed_percent
        expected = joint_frames(current_deg) + [(0x151, bytes([1, 1, speed_percent, 0, 0, 0, 0, 0]))]
        with self._transaction(expected):
            checkpoint()
            self._move_j(current_deg)
            checkpoint()
            self.arm._msg_mode = mode
            self.arm.set_motion_mode("j")

    def joint_target(self, joints_deg):
        self._validate_joints(joints_deg)
        with self._transaction(joint_frames(joints_deg)):
            self._move_j(joints_deg)

    def linear_target(self, current_deg, flange_pose, speed_percent, checkpoint):
        self._validate_joints(current_deg)
        xyz = flange_pose["xyz_m"]
        rpy = flange_pose["rpy_deg"]
        if not all(math.isfinite(v) for v in xyz+rpy) or type(speed_percent) is not int or not 1 <= speed_percent <= 100:
            raise DomainError("invalid_target", "Invalid native linear target.", 422)
        # SDK move_l takes FLANGE metres and radians, not TCP metres/degrees.
        units = [round(v*1e6) for v in xyz] + [round(v*1000) for v in rpy]
        frames = [(0x152+i, struct.pack(">ii", *units[2*i:2*i+2])) for i in range(3)]
        mode = self.arm._MSG_ModeCtrl()
        mode.ctrl_mode, mode.move_spd_rate_ctrl = 1, speed_percent
        with self._transaction(joint_frames(current_deg) + [(0x151, bytes([1, 2, speed_percent, 0, 0, 0, 0, 0]))] + frames):
            checkpoint()
            self._move_j(current_deg)
            checkpoint()
            self.arm._msg_mode = mode
            self.arm.set_motion_mode("l")
            checkpoint()
            self.arm.move_l(xyz + [math.radians(v) for v in rpy])

    def gripper_target(self, width_m, effort_protocol):
        effort_limit = self.settings.gripper_effort_limit if self.settings.control_profile == "calibration" else 32.767
        if not (math.isfinite(width_m) and math.isfinite(effort_protocol) and
                0 <= width_m <= self.settings.gripper_max_m and 0 <= effort_protocol <= effort_limit):
            raise DomainError("gripper_limits", "Invalid gripper target.")
        data = struct.pack(">iHBB", round(width_m * 1e6), round(effort_protocol * 1e3), 1, 0)
        with self._transaction([(0x159, data)]):
            self.gripper.move_gripper_m(value=width_m, force=effort_protocol)

    def close(self):
        global _owner
        try:
            if self.arm:
                ctx = self.arm.get_context()
                # The SDK clears these references even if its join times out.
                for thread in (getattr(ctx, "_read_th", None), getattr(ctx, "_monitor_th", None),
                               getattr(getattr(ctx, "fps", None), "thread", None)):
                    if thread is not None and thread not in self._closing_threads:
                        self._closing_threads.append(thread)
                self.arm.disconnect()
            # SDK disconnect suppresses comm.close errors. Verify our own bus
            # explicitly before declaring the adapter available to another owner.
            if self.bus:
                self.bus.shutdown()
            elif self._pending_transport is not None:
                self._pending_transport.shutdown()
            if any(thread.is_alive() for thread in self._closing_threads):
                raise RuntimeError("SDK reader/monitor thread did not exit")
        except Exception as exc:
            self.fault = f"CAN cleanup incomplete: {exc}"
            raise DomainError("cleanup_incomplete", self.fault) from exc
        else:
            self.arm = self.gripper = self.bus = None
            self._pending_transport = None
            self._closing_threads.clear()
            if self.device_lock:
                self.device_lock.close()
                self.device_lock = None
            with _owner_lock:
                if _owner is self:
                    _owner = None
