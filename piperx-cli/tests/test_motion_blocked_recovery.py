"""Runtime recovery from a confirmed blocked motion.

Every test drives the real ``RobotService._run_motion`` worker (through
``service.move``/``service.primitive``) against a deterministic MuJoCo-named
backend that only reports measured samples. No test calls the stall watchdog or
the hold-confirmation helper directly, and no test writes qpos/qvel, teleports
or replays a command.

Contract covered here:
* a stalled ordinary MuJoCo motion ends as status=failed,
  error_code=motion_blocked, recoverable=true only after a bounded measured
  position hold was confirmed by >=3 fresh advancing settled samples;
* moving waypoints whose largest final-target error stays flat (a curved
  Cartesian/IK path) are never called stalled while measured joints move, while
  a permanently immobile .12-.3 tracking guard still blocks;
* slow but genuinely progressing motion is never cut off early;
* cached pre-hold frames, out-of-order frames and a frozen gripper source stamp
  are never counted as post-hold stop evidence;
* the read that certifies the hold must itself be a strictly advancing
  post-hold stamp inside the velocity/joint/width bounds, and that actual
  latest public sample is what the job reports;
* a contact-completion gripper close that never contacts (empty closing) or
  stops making progress stays failed/recoverable and is never a success, while
  a fresh bilateral-contact completion keeps its existing succeeded outcome;
* stale feedback, stale contact feedback, held velocity, moving hold, changed
  connection identity, a fault and an unconfirmed hold stay unknown;
* cancellation stays cancellation and a non-MuJoCo backend is unchanged;
* a confirmed stop leaves no stale active job.
"""
from __future__ import annotations

import threading
import time

import pytest

from piperx_middleware import service as service_module
from piperx_middleware.backends import SimBackend
from piperx_middleware.kinematics import PiperKinematics
from piperx_middleware.models import (ExecuteRequest, GripperMove, JointMove,
                                      LeaseRequest, MoveLinear, Settings)
from piperx_middleware.service import RobotService

WAIT_S = 20.0


class ScriptedMujoco(SimBackend):
    """MuJoCo-named backend whose measured samples are scripted, never teleported.

    ``step_deg``/``step_width_m`` are the measured progress per fresh sample;
    ``max_progress_samples`` blocks further measured progress, which models a
    real contact block. The hold only replaces the servo reference with the
    measured pose, exactly like ``MujocoBackend.hold_position``.
    """

    name = "mujoco"

    def __init__(self, q=None, width=0.02):
        super().__init__()
        self.q = list(q) if q is not None else [0.0] * 6
        self.width = width
        self.follow = False
        self.target = list(self.q)
        self.target_width = width
        self.step_deg = 0.0
        self.step_width_m = 0.0
        self.max_progress_samples = None
        self.progress_samples = 0
        self.drift_deg = 0.0
        self.freeze_stamp = False
        self.raise_on_snapshot = False
        self.velocity = [0.0] * 6
        self.epoch = 0
        self.stamp = time.monotonic()
        self.hold_calls = []
        self.post_hold = None
        # Optional [(feedback_stamp_s, gripper_stamp_s), ...] armed by a
        # post_hold hook; consumed one entry per snapshot after the hold.
        self.stamp_plan = None
        self.stamp_plan_index = 0
        # None = no contact diagnostic at all (an empty/missed closing reports
        # no grasp); otherwise (supported, age_s) for the reported contact read.
        self.contact_report = None
        # (monotonic_s, measured q, measured width) for every measured read.
        self.snapshots = []

    # -- measured motion only: no qpos/qvel write, no teleport ----------------
    def joint_target(self, joints_deg):
        with self.lock:
            self._write_guard()
            self.target = [float(v) for v in joints_deg]
            self.commands.append(("joint", list(joints_deg)))

    def gripper_target(self, width_m, effort_protocol):
        with self.lock:
            self._write_guard()
            self.target_width = float(width_m)
            self.commands.append(("gripper", width_m, effort_protocol))

    def hold_position(self, hold_gripper=False):
        with self.lock:
            self.hold_calls.append(bool(hold_gripper))
            self.target = list(self.q)
            if hold_gripper:
                self.target_width = self.width
            if self.post_hold is not None:
                hook, self.post_hold = self.post_hold, None
                hook(self)

    def snapshot(self):
        with self.lock:
            if self.raise_on_snapshot and threading.current_thread().name == "piperx-action":
                raise RuntimeError("adapter read failed")
            if self.step_deg and (self.max_progress_samples is None
                                  or self.progress_samples < self.max_progress_samples):
                stepped = [a + max(-self.step_deg, min(self.step_deg, b - a))
                           for a, b in zip(self.q, self.target)]
                if stepped != self.q:
                    self.progress_samples += 1
                    self.q = stepped
            if self.step_width_m:
                self.width += max(-self.step_width_m, min(self.step_width_m, self.target_width - self.width))
            if self.drift_deg:
                self.q[0] += self.drift_deg
            state = super().snapshot()
            state.feedback_stamp_s = state.gripper_stamp_s = self.stamp
            if not self.freeze_stamp:
                self.stamp += 0.05
            if self.stamp_plan is not None:
                index = min(self.stamp_plan_index, len(self.stamp_plan) - 1)
                self.stamp_plan_index += 1
                state.feedback_stamp_s, state.gripper_stamp_s = self.stamp_plan[index]
            if self.contact_report is not None:
                supported, age_s = self.contact_report
                state.diagnostics["gripper_contact"] = {
                    "supported": bool(supported),
                    "finger1_force_n": 1.0 if supported else 0.0,
                    "finger2_force_n": 1.0 if supported else 0.0,
                    "source_timestamp_s": time.monotonic() - age_s,
                }
            state.velocity_deg_s = list(self.velocity)
            state.diagnostics["epoch"] = self.epoch
            self.snapshots.append((time.monotonic(), list(self.q), self.width))
            return state


def make_service(tmp_path, backend):
    service = RobotService(backend, Settings(backend="mujoco", data_dir=tmp_path))
    service.connect()
    return service


def wait(service, job, limit=WAIT_S):
    until = time.monotonic() + limit
    while job["status"] in ("accepted", "running"):
        assert time.monotonic() < until, job
        time.sleep(0.01)
        job = service.get_job(job["job_id"])
    return job


def fast_stall(monkeypatch, warmup=0.1, window=0.2):
    monkeypatch.setattr(service_module, "_STALL_WARMUP_S", warmup)
    monkeypatch.setattr(service_module, "_STALL_WINDOW_S", window)


def short_confirmation(monkeypatch, window=0.3):
    monkeypatch.setattr(service_module, "_RECOVERY_HOLD_MAX_S", window)
    monkeypatch.setattr(service_module, "_RECOVERY_HOLD_SAMPLE_INTERVAL_S", 0.01)


def flat_error_window(backend, final_deg):
    """Find a settled window where the scalar error to the final target is flat.

    Returns the window metrics when, after the watchdog warmup, the largest
    measured error to the final commanded joints improved by less than the
    stall epsilon over at least one full ``_STALL_WINDOW_S`` while the measured
    joints moved by more than that epsilon. That is exactly the shape a curved
    Cartesian/IK path can produce and the old scalar watchdog misread.
    """
    samples = backend.snapshots
    window = service_module._STALL_WINDOW_S
    epsilon = service_module._STALL_MIN_PROGRESS_DEG
    for i, (started_at, baseline_q, _) in enumerate(samples):
        if started_at - samples[0][0] < service_module._STALL_WARMUP_S:
            continue
        baseline_error = max(abs(a - b) for a, b in zip(baseline_q, final_deg))
        for j in range(i + 1, len(samples)):
            ended_at, latest_q, _ = samples[j]
            if ended_at - started_at < window:
                continue
            if ended_at - started_at > 2 * window:
                break
            best_error = min(max(abs(a - b) for a, b in zip(q, final_deg))
                             for _, q, _ in samples[i:j + 1])
            moved = max(max(abs(a - b) for a, b in zip(q, baseline_q)) for _, q, _ in samples[i:j + 1])
            if baseline_error - best_error < epsilon and moved > epsilon:
                return {"baseline_error_deg": baseline_error, "best_error_deg": best_error,
                        "moved_deg": moved}
    return None


def preview_with_waypoints(service, command, waypoints):
    """Create a plan through the normal preview path, then use given waypoints.

    The worker consumes ``resolution.waypoints_deg`` exactly like a resolved
    primitive does; this only substitutes a deterministic legal joint path.
    """
    plan = service.preview(command)
    with service.lock:
        service.plans[plan["plan_id"]]["resolution"] = {
            "waypoints_deg": [list(point) for point in waypoints], "native_controller": False}
    return plan


def test_internal_limits_stay_bounded_and_inside_poll_grace():
    # Fixed internal limits, not operator/model knobs. The confirmation must fit
    # inside the console's 5 s post-timeout polling grace.
    assert service_module._RECOVERY_HOLD_MIN_SAMPLES >= 3
    assert 0 < service_module._RECOVERY_HOLD_MIN_SPAN_S <= service_module._RECOVERY_HOLD_MAX_S
    assert service_module._RECOVERY_HOLD_MAX_S < 5.0
    assert 0 < service_module._STALL_WARMUP_S < service_module._STALL_WINDOW_S
    assert service_module._STALL_MIN_PROGRESS_DEG > 0
    assert service_module._STALL_MIN_PROGRESS_M > 0
    assert set(service_module._BLOCKED_RECOVERY_CODES) == {"execution_timeout", "motion_stalled", "tracking_error"}


def test_stalled_joint_move_returns_failed_recoverable_after_confirmed_hold(tmp_path):
    backend = ScriptedMujoco()          # blocked from the first sample
    service = make_service(tmp_path, backend)
    try:
        started = time.monotonic()
        job = wait(service, service.move(JointMove(joints_deg=[3., 0., 0., 0., 0., 0.], timeout_s=30.),
                                         "stalled-joint-01"))
        elapsed = time.monotonic() - started
        assert elapsed < 5.0, elapsed   # stall detected long before the 30 s deadline
        assert job["status"] == "failed"
        assert job["error_code"] == "motion_blocked"
        assert job["recoverable"] is True
        assert job["original_error_code"] == "motion_stalled"
        assert "progress" in job["error"]
        assert job["command_attempted"] is True
        assert "verification" not in job            # never a success claim
        stop = job["stop_result"]
        assert stop["status"] == "hold_requested"
        assert stop["confirmed_stopped"] is True
        assert 0 < stop["hold_requested_monotonic_s"] <= time.monotonic()
        assert stop["confirmation"]["samples"] >= 3
        assert stop["confirmation"]["settled_span_s"] >= service_module._RECOVERY_HOLD_MIN_SPAN_S
        assert stop["confirmation"]["max_velocity_deg_s"] <= service_module._RECOVERY_HOLD_MAX_VELOCITY_DEG_S
        assert stop["confirmation"]["max_joint_delta_deg"] <= service_module._RECOVERY_HOLD_MAX_JOINT_DELTA_DEG
        assert job["after"]["q_deg"] == [0.0] * 6
        assert backend.hold_calls == [False]        # arm hold keeps the gripper reference
    finally:
        service.close()


def test_new_motion_is_possible_after_a_confirmed_blocked_stop(tmp_path):
    backend = ScriptedMujoco()
    service = make_service(tmp_path, backend)
    try:
        job = wait(service, service.move(JointMove(joints_deg=[3., 0., 0., 0., 0., 0.], timeout_s=30.),
                                         "stalled-joint-02"))
        assert job["error_code"] == "motion_blocked"
        assert service.active is None
        assert service.state()["active_job_id"] is None
        nxt = wait(service, service.move(JointMove(joints_deg=[0.] * 6, timeout_s=5.),
                                         "after-recovery-01"))
        assert nxt["status"] == "succeeded", nxt
        assert nxt["job_id"] != job["job_id"]
        assert service.active is None
    finally:
        service.close()


def test_progressing_motion_is_not_prematurely_blocked(tmp_path, monkeypatch):
    fast_stall(monkeypatch, warmup=0.1, window=0.5)
    backend = ScriptedMujoco()
    backend.step_deg = 0.015            # slow but real measured progress
    service = make_service(tmp_path, backend)
    try:
        job = wait(service, service.move(JointMove(joints_deg=[0.5, 0., 0., 0., 0., 0.], timeout_s=30.),
                                         "progressing-01"))
        assert job["status"] == "succeeded", job
        assert job.get("error_code") != "motion_blocked"
        assert "recoverable" not in job
        assert backend.hold_calls == []
        assert job["after"]["q_deg"][0] == pytest.approx(0.5, abs=0.001)
    finally:
        service.close()


def test_paused_linear_guard_stall_is_recovered(tmp_path, monkeypatch):
    # The sampled linear reference pauses while measured tracking sits in the
    # existing .12-.3 guard band; a measured block there must not wait out the
    # whole deadline.
    fast_stall(monkeypatch, warmup=0.1, window=0.25)
    monkeypatch.setattr(service_module, "_STALL_MIN_PROGRESS_DEG", 0.01)
    backend = ScriptedMujoco(q=[0., 45., -60., 0., 20., 0.])
    backend.step_deg = 0.06
    backend.max_progress_samples = 20   # contact block after a short path segment
    service = make_service(tmp_path, backend)
    try:
        target = PiperKinematics().pose(backend.q)["xyz_m"]
        target[2] += 0.004
        job = wait(service, service.primitive(MoveLinear(xyz_m=target, step_m=0.001, timeout_s=30.),
                                              "guard-stall-01"))
        assert job["status"] == "failed", job
        assert job["error_code"] == "motion_blocked"
        assert job["original_error_code"] == "motion_stalled"
        assert job["recoverable"] is True
        assert job["stop_result"]["confirmed_stopped"] is True
    finally:
        service.close()


def test_moving_waypoints_with_constant_final_target_error_are_not_stalled(tmp_path, monkeypatch):
    """A legal waypoint path may hold the largest final-target error flat.

    Joint 0 keeps an exact 2 degree error to the final commanded pose while
    joints 4 and 5 travel, then joint 0 closes the remaining distance. The
    measured joints move the whole time, so the scalar "distance to the final
    command" stays constant and must not be mistaken for a stalled motion.
    """
    fast_stall(monkeypatch, warmup=0.1, window=0.25)
    start = [0.0, 30.0, -70.0, 0.0, 10.0, 0.0]
    final = [2.0, 30.0, -70.0, 0.0, 12.0, -1.0]
    backend = ScriptedMujoco(q=start)
    backend.step_deg = 0.5              # measured tracking, never a teleport
    service = make_service(tmp_path, backend)
    try:
        waypoints = [[0.0, 30.0, -70.0, 0.0, 10.0 + 0.1 * k, -0.05 * k] for k in range(1, 21)]
        waypoints += [[0.1 * k, 30.0, -70.0, 0.0, 12.0, -1.0] for k in range(1, 21)]
        plan = preview_with_waypoints(service, JointMove(joints_deg=final, timeout_s=30.), waypoints)
        job = wait(service, service.execute(ExecuteRequest(plan_id=plan["plan_id"],
                                                          request_id="flat-waypoints-01")))
        assert job["status"] == "succeeded", job
        assert job.get("error_code") != "motion_blocked"
        assert backend.hold_calls == []
        flat = flat_error_window(backend, final)
        assert flat is not None, "the waypoint path did not hold a flat final-target error window"
        assert flat["baseline_error_deg"] == pytest.approx(2.0, abs=0.05)
        assert flat["moved_deg"] > service_module._STALL_MIN_PROGRESS_DEG
        assert backend.q == pytest.approx(final, abs=0.12)
    finally:
        service.close()


def test_real_curved_ik_line_path_is_not_falsely_stalled(tmp_path, monkeypatch):
    """A resolved MoveLinear path whose joint-space error stays flat is legal.

    The deterministic Cartesian line below produces a curved joint path: the
    largest error to the final commanded joints stays within the stall epsilon
    for over a full window while the measured joints keep moving. The scalar
    watchdog would have blocked it; the measured-displacement watchdog must let
    it finish.
    """
    fast_stall(monkeypatch, warmup=0.1, window=0.25)
    backend = ScriptedMujoco(q=[0.0, 30.0, -70.0, 0.0, 10.0, 0.0])
    backend.step_deg = 0.5              # measured tracking, never a teleport
    service = make_service(tmp_path, backend)
    try:
        job = wait(service, service.primitive(
            MoveLinear(xyz_m=[0.091, -0.018, 0.567], step_m=0.004, timeout_s=30.),
            "curved-ik-line-01"), limit=40.0)
        assert job["status"] == "succeeded", job
        assert job.get("error_code") != "motion_blocked"
        assert backend.hold_calls == []
        final = job["command"]["joints_deg"]
        flat = flat_error_window(backend, final)
        assert flat is not None, "the resolved IK path did not hold a flat final-target error window"
        assert flat["moved_deg"] > service_module._STALL_MIN_PROGRESS_DEG
    finally:
        service.close()


def test_slow_timeout_becomes_motion_blocked_only_after_the_hold(tmp_path):
    backend = ScriptedMujoco()
    backend.step_deg = 0.005            # far from the target at the short deadline
    service = make_service(tmp_path, backend)
    try:
        job = wait(service, service.move(JointMove(joints_deg=[1., 0., 0., 0., 0., 0.], timeout_s=0.4),
                                         "timeout-block-01"))
        assert job["status"] == "failed"
        assert job["error_code"] == "motion_blocked"
        assert job["original_error_code"] == "execution_timeout"
        assert job["recoverable"] is True
        assert job["stop_result"]["confirmed_stopped"] is True
    finally:
        service.close()


@pytest.mark.parametrize("width,target_width", [(0.02, 0.0), (0.02, 0.05)])
def test_gripper_width_stall_is_recoverable_and_retains_grip(tmp_path, width, target_width):
    backend = ScriptedMujoco(width=width)
    service = make_service(tmp_path, backend)
    try:
        # Closing (0.0) and opening (0.05) both stall when the measured width
        # cannot make progress toward the commanded width.
        job = wait(service, service.move(GripperMove(width_m=target_width, timeout_s=30.), "gripper-stall-01"))
        assert job["status"] == "failed"
        assert job["error_code"] == "motion_blocked"
        assert job["original_error_code"] == "motion_stalled"
        assert job["recoverable"] is True
        assert backend.hold_calls == [True]          # gripper hold keeps the grip
        assert job["after"]["gripper_width_m"] == pytest.approx(width, abs=1e-9)
    finally:
        service.close()


@pytest.mark.parametrize("hook,reason", [
    (lambda b: b.velocity.__setitem__(0, 3.0), "settle"),
    (lambda b: setattr(b, "drift_deg", 0.2), "settle"),
    (lambda b: setattr(b, "epoch", 1), "identity"),
    (lambda b: setattr(b, "fault", "driver"), "standstill"),
    (lambda b: setattr(b, "stale", True), "standstill"),
    (lambda b: setattr(b, "freeze_stamp", True), "advancing"),
    (lambda b: setattr(b, "raise_on_snapshot", True), "confirmation failed"),
])
def test_unconfirmed_hold_never_becomes_recoverable(tmp_path, monkeypatch, hook, reason):
    fast_stall(monkeypatch)
    short_confirmation(monkeypatch)
    backend = ScriptedMujoco()
    backend.post_hold = hook
    service = make_service(tmp_path, backend)
    try:
        job = wait(service, service.move(JointMove(joints_deg=[3., 0., 0., 0., 0., 0.], timeout_s=30.),
                                         "unconfirmed-01"))
        assert job["status"] == "outcome_unknown", job
        assert job["error_code"] == "motion_stalled"
        assert "recoverable" not in job
        assert job["stop_result"]["confirmed_stopped"] is False
        assert reason in job["stop_result"]["stop_confirmation"]["reason"]
    finally:
        service.close()


def test_pre_hold_cached_frames_are_not_stop_evidence(tmp_path, monkeypatch):
    """Source stamps from before the hold write never count as stop evidence.

    The cached frames advance monotonically (and span the settled interval) but
    were produced on the pre-hold side of the hold-request instant, so they must
    leave the outcome unknown.
    """
    fast_stall(monkeypatch)
    short_confirmation(monkeypatch, window=0.6)
    backend = ScriptedMujoco()

    def arm_cached(scripted):
        armed_at = time.monotonic()
        scripted.stamp_plan = [(armed_at - 0.40 + 0.03 * k, armed_at - 0.40 + 0.03 * k)
                               for k in range(12)]
    backend.post_hold = arm_cached
    service = make_service(tmp_path, backend)
    try:
        job = wait(service, service.move(JointMove(joints_deg=[3., 0., 0., 0., 0., 0.], timeout_s=30.),
                                         "cached-prehold-01"))
        assert job["status"] == "outcome_unknown", job
        assert job["error_code"] == "motion_stalled"
        assert "recoverable" not in job
        assert job["stop_result"]["confirmed_stopped"] is False
        assert "advancing" in job["stop_result"]["stop_confirmation"]["reason"]
    finally:
        service.close()


def test_replayed_out_of_order_frames_are_not_stop_evidence(tmp_path, monkeypatch):
    """Two replayed source stamps must not accumulate as fresh stop samples."""
    fast_stall(monkeypatch)
    short_confirmation(monkeypatch, window=0.6)
    backend = ScriptedMujoco()

    def arm_out_of_order(scripted):
        armed_at = time.monotonic()
        scripted.stamp_plan = [(armed_at + offset, armed_at + offset)
                               for _ in range(8) for offset in (0.10, 0.40)]
    backend.post_hold = arm_out_of_order
    service = make_service(tmp_path, backend)
    try:
        job = wait(service, service.move(JointMove(joints_deg=[3., 0., 0., 0., 0., 0.], timeout_s=30.),
                                         "out-of-order-01"))
        assert job["status"] == "outcome_unknown", job
        assert job["error_code"] == "motion_stalled"
        assert "recoverable" not in job
        assert job["stop_result"]["confirmed_stopped"] is False
        assert "advancing" in job["stop_result"]["stop_confirmation"]["reason"]
    finally:
        service.close()


def test_cached_frame_does_not_block_later_fresh_confirmation(tmp_path, monkeypatch):
    """One pre-hold cached frame is skipped; genuinely fresh samples still confirm."""
    fast_stall(monkeypatch)
    short_confirmation(monkeypatch, window=0.6)
    backend = ScriptedMujoco()

    def arm_cached_then_fresh(scripted):
        armed_at = time.monotonic()
        scripted.stamp_plan = [(armed_at - 0.20, armed_at - 0.20)] + [
            (armed_at + 0.05 * k, armed_at + 0.05 * k) for k in range(1, 12)]
    backend.post_hold = arm_cached_then_fresh
    service = make_service(tmp_path, backend)
    try:
        job = wait(service, service.move(JointMove(joints_deg=[3., 0., 0., 0., 0., 0.], timeout_s=30.),
                                         "cached-then-fresh-01"))
        assert job["status"] == "failed", job
        assert job["error_code"] == "motion_blocked"
        assert job["recoverable"] is True
        assert job["stop_result"]["confirmed_stopped"] is True
        assert job["stop_result"]["confirmation"]["samples"] >= 3
    finally:
        service.close()


def test_final_read_that_is_moving_does_not_certify_standstill(tmp_path, monkeypatch):
    """The read that would certify the hold is itself bound by the stop bounds.

    The first four post-hold reads are stationary and complete the settled
    window (>= 3 samples over >= 0.25 s at 0.1 s stamp steps). The next read
    reports measured velocity and displacement above the hold bounds, so it
    must reset the window and no stop may be certified.
    """
    fast_stall(monkeypatch)
    short_confirmation(monkeypatch, window=0.8)
    backend = ScriptedMujoco()

    def arm_moving_final_read(scripted):
        armed_at = time.monotonic()
        scripted.stamp_plan = [(armed_at + 0.1 * k, armed_at + 0.1 * k) for k in range(1, 20)]
        original = scripted.snapshot
        reads = {"n": 0}

        def counted():
            reads["n"] += 1
            if reads["n"] == 5:                # the certifying read
                scripted.velocity[0] = 3.0     # above the 0.5 deg/s hold bound
                scripted.drift_deg = 0.4       # above the 0.05 deg joint bound
            return original()

        scripted.snapshot = counted

    backend.post_hold = arm_moving_final_read
    service = make_service(tmp_path, backend)
    try:
        job = wait(service, service.move(JointMove(joints_deg=[3., 0., 0., 0., 0., 0.], timeout_s=30.),
                                         "final-read-moving-01"))
        assert job["status"] == "outcome_unknown", job
        assert job["error_code"] == "motion_stalled"
        assert "recoverable" not in job
        assert job["stop_result"]["confirmed_stopped"] is False
        assert "settle" in job["stop_result"]["stop_confirmation"]["reason"]
    finally:
        service.close()


def test_final_replayed_read_does_not_certify_standstill(tmp_path, monkeypatch):
    """A replayed certifying read must not ride on the completed window.

    Four strictly advancing stamps complete the settled window; every read
    after them replays the last stamp, so the extra read that used to be
    accepted unchecked is not fresh evidence and the outcome stays unknown.
    """
    fast_stall(monkeypatch)
    short_confirmation(monkeypatch, window=0.8)
    backend = ScriptedMujoco()

    def arm_replayed_final_read(scripted):
        armed_at = time.monotonic()
        scripted.stamp_plan = [(armed_at + 0.1 * k, armed_at + 0.1 * k) for k in range(1, 5)]
        scripted.stamp_plan += [(armed_at + 0.4, armed_at + 0.4)] * 40

    backend.post_hold = arm_replayed_final_read
    service = make_service(tmp_path, backend)
    try:
        job = wait(service, service.move(JointMove(joints_deg=[3., 0., 0., 0., 0., 0.], timeout_s=30.),
                                         "final-read-replayed-01"))
        assert job["status"] == "outcome_unknown", job
        assert job["error_code"] == "motion_stalled"
        assert "recoverable" not in job
        assert job["stop_result"]["confirmed_stopped"] is False
        assert "advancing" in job["stop_result"]["stop_confirmation"]["reason"]
    finally:
        service.close()


def test_accepted_hold_returns_the_actual_latest_sample(tmp_path, monkeypatch):
    """A confirmed hold returns the certifying read, not the earlier window sample."""
    fast_stall(monkeypatch)
    short_confirmation(monkeypatch, window=0.8)
    backend = ScriptedMujoco()

    def arm_sub_epsilon_final_move(scripted):
        armed_at = time.monotonic()
        scripted.stamp_plan = [(armed_at + 0.1 * k, armed_at + 0.1 * k) for k in range(1, 20)]
        original = scripted.snapshot
        reads = {"n": 0}

        def counted():
            reads["n"] += 1
            if reads["n"] == 5:                # the certifying read
                scripted.drift_deg = 0.01      # inside the 0.05 deg joint bound
                state = original()
                scripted.drift_deg = 0.0
                return state
            return original()

        scripted.snapshot = counted

    backend.post_hold = arm_sub_epsilon_final_move
    service = make_service(tmp_path, backend)
    try:
        job = wait(service, service.move(JointMove(joints_deg=[3., 0., 0., 0., 0., 0.], timeout_s=30.),
                                         "latest-sample-01"))
        assert job["status"] == "failed", job
        assert job["error_code"] == "motion_blocked"
        assert job["recoverable"] is True
        assert job["stop_result"]["confirmed_stopped"] is True
        # The certifying read measured 0.01 deg and is the actual latest public
        # sample; the older window sample measured 0.0.
        assert job["after"]["q_deg"][0] == pytest.approx(0.01, abs=1e-9)
        assert job["after"]["q_deg"] == backend.snapshots[-1][1]
        assert job["stop_result"]["confirmation"]["samples"] >= 4
    finally:
        service.close()


@pytest.mark.parametrize("completion", ["width", "bilateral_contact"])
def test_frozen_gripper_source_stamp_is_not_stop_evidence(tmp_path, monkeypatch, completion):
    """A GripperMove hold needs the older required stamp to keep advancing."""
    fast_stall(monkeypatch)
    short_confirmation(monkeypatch, window=0.6)
    backend = ScriptedMujoco(width=0.02)

    def arm_frozen_gripper(scripted):
        armed_at = time.monotonic()
        scripted.stamp_plan = [(armed_at + 0.05 * k, armed_at - 0.02) for k in range(1, 12)]
    backend.post_hold = arm_frozen_gripper
    service = make_service(tmp_path, backend)
    try:
        job = wait(service, service.move(GripperMove(width_m=0.0, timeout_s=30., completion=completion),
                                         f"frozen-gripper-{completion}-01"))
        assert job["status"] == "outcome_unknown", job
        assert job["error_code"] == "motion_stalled"
        assert "recoverable" not in job
        assert backend.hold_calls == [True]
        assert job["stop_result"]["confirmed_stopped"] is False
        assert "advancing" in job["stop_result"]["stop_confirmation"]["reason"]
    finally:
        service.close()


def test_cancel_during_hold_confirmation_is_not_recoverable(tmp_path, monkeypatch):
    """A stop request during the bounded hold confirmation blocks recovery."""
    fast_stall(monkeypatch)
    short_confirmation(monkeypatch, window=0.8)
    backend = ScriptedMujoco()
    service = make_service(tmp_path, backend)

    def request_stop(_scripted):
        threading.Timer(0.01, service.stop).start()
    backend.post_hold = request_stop
    try:
        job = wait(service, service.move(JointMove(joints_deg=[3., 0., 0., 0., 0., 0.], timeout_s=30.),
                                         "cancel-hold-01"))
        assert job["status"] == "outcome_unknown", job
        assert job["error_code"] == "motion_stalled"
        assert "recoverable" not in job
        assert job["stop_result"]["confirmed_stopped"] is False
        assert "stop requested" in job["stop_result"]["stop_confirmation"]["reason"]
    finally:
        service.close()


def test_journal_failure_during_the_hold_is_not_recoverable(tmp_path, monkeypatch):
    fast_stall(monkeypatch)
    backend = ScriptedMujoco()
    service = make_service(tmp_path, backend)
    try:
        real_event = service.store.event

        def failing_event(name, **data):
            if name == "hold_request":
                raise RuntimeError("journal down")
            return real_event(name, **data)

        monkeypatch.setattr(service.store, "event", failing_event)
        job = wait(service, service.move(JointMove(joints_deg=[3., 0., 0., 0., 0., 0.], timeout_s=30.),
                                         "journal-down-01"))
        assert job["status"] == "outcome_unknown"
        assert job["error_code"] == "motion_stalled"
        assert "recoverable" not in job
        assert job["stop_result"]["confirmed_stopped"] is False
        assert "journalled" in job["stop_result"]["stop_confirmation"]["reason"]
    finally:
        service.close()


def test_cancellation_stays_cancellation(tmp_path):
    backend = ScriptedMujoco()
    service = make_service(tmp_path, backend)
    try:
        job = service.move(JointMove(joints_deg=[3., 0., 0., 0., 0., 0.], timeout_s=30.), "cancel-blocked-01")
        until = time.monotonic() + 5.0
        while service.get_job(job["job_id"])["status"] != "running":
            assert time.monotonic() < until
            time.sleep(0.01)
        service.stop()
        job = wait(service, job)
        assert job["status"] == "cancelled", job
        assert job["error_code"] == "cancelled"
        assert "recoverable" not in job
    finally:
        service.close()


def test_non_mujoco_backend_keeps_its_timeout_behaviour(tmp_path):
    backend = SimBackend()
    backend.follow = False
    service = RobotService(backend, Settings(data_dir=tmp_path))
    service.connect()
    try:
        job = wait(service, service.move(JointMove(joints_deg=[2., 0., 0., 0., 0., 0.], timeout_s=2.0),
                                         "sim-timeout-01"), limit=10.0)
        # The 2 s timeout, not the ~1.6 s stall watchdog, decided this job.
        assert job["status"] == "outcome_unknown"
        assert job["error_code"] == "execution_timeout"
        assert "recoverable" not in job
        assert backend.commands[0][0] == "begin"
    finally:
        service.close()


def test_bilateral_contact_timeout_after_confirmed_hold_is_recoverable_not_success(tmp_path):
    """A contact completion that cannot reach contact is no longer fatal.

    The direct-profile gripper is held at its measured width and fresh feedback
    confirms it is stationary, so the blocked close is re-expressed as
    failed/recoverable -- never as a success claim.
    """
    backend = ScriptedMujoco(width=0.02)     # no contact diagnostic at all
    service = make_service(tmp_path, backend)
    try:
        job = wait(service, service.move(GripperMove(width_m=0.001, completion="bilateral_contact", timeout_s=0.3),
                                         "contact-timeout-01"))
        assert job["status"] == "failed", job
        assert job["error_code"] == "motion_blocked"
        assert job["original_error_code"] == "execution_timeout"
        assert job["recoverable"] is True
        assert "verification" not in job
        assert backend.hold_calls == [True]
        assert job["stop_result"]["confirmed_stopped"] is True
    finally:
        service.close()


def test_empty_bilateral_contact_close_stays_failed_then_open_succeeds(tmp_path, monkeypatch):
    """Closing to the commanded width without contact is never a success.

    The measured width reaches 0.0 with no contact reported, so the contact
    completion stays failed/recoverable after the confirmed hold and a later
    ordinary open request still succeeds.
    """
    fast_stall(monkeypatch)
    backend = ScriptedMujoco(width=0.02)
    backend.step_width_m = 0.005            # closes fully, never contacts
    service = make_service(tmp_path, backend)
    try:
        job = wait(service, service.move(GripperMove(width_m=0.0, completion="bilateral_contact", timeout_s=30.),
                                         "empty-contact-close-01"))
        assert job["status"] == "failed", job
        assert job["error_code"] == "motion_blocked"
        assert job["original_error_code"] == "motion_stalled"
        assert job["recoverable"] is True
        assert "verification" not in job             # empty closing is never a success
        assert backend.width == pytest.approx(0.0, abs=1e-9)
        assert backend.hold_calls == [True]
        assert job["stop_result"]["confirmed_stopped"] is True
        nxt = wait(service, service.move(GripperMove(width_m=0.05, timeout_s=30.),
                                         "open-after-missed-grasp-01"))
        assert nxt["status"] == "succeeded", nxt
        assert nxt["after"]["gripper_width_m"] == pytest.approx(0.05, abs=0.001)
        assert backend.hold_calls == [True]
    finally:
        service.close()


def test_supported_bilateral_contact_completion_succeeds(tmp_path):
    """Fresh bilateral contact on one dynamic object keeps its succeeded outcome."""
    backend = ScriptedMujoco(width=0.02)
    backend.contact_report = (True, 0.0)    # both fingers on the same object, fresh
    service = make_service(tmp_path, backend)
    try:
        job = wait(service, service.move(GripperMove(width_m=0.001, completion="bilateral_contact", timeout_s=5.),
                                         "contact-success-01"))
        assert job["status"] == "succeeded", job
        assert job["contact_evidence"]["supported"] is True
        assert "bilateral forces" in job["verification"]
        assert "recoverable" not in job
        assert backend.hold_calls == []
        assert backend.width == pytest.approx(0.02, abs=1e-9)   # blocked by the held object
    finally:
        service.close()


def test_stale_contact_feedback_keeps_the_outcome_unknown(tmp_path, monkeypatch):
    """A present but stale contact reading never underwrites a recovery claim."""
    fast_stall(monkeypatch)
    short_confirmation(monkeypatch, window=0.8)
    backend = ScriptedMujoco(width=0.02)
    backend.contact_report = (False, 5.0)   # contact channel reports an old read
    service = make_service(tmp_path, backend)
    try:
        job = wait(service, service.move(GripperMove(width_m=0.0, completion="bilateral_contact", timeout_s=30.),
                                         "stale-contact-01"))
        assert job["status"] == "outcome_unknown", job
        assert job["error_code"] == "motion_stalled"
        assert "recoverable" not in job
        assert job["stop_result"]["confirmed_stopped"] is False
        assert "contact" in job["stop_result"]["stop_confirmation"]["reason"]
    finally:
        service.close()


def test_calibration_stall_keeps_its_existing_tracking_outcome(tmp_path):
    backend = ScriptedMujoco()          # blocked while the bounded reference advances
    service = RobotService(backend, Settings(backend="mujoco", control_profile="calibration", data_dir=tmp_path))
    service.connect()
    try:
        service.arm_window(LeaseRequest(duration_s=30, joint_radius_deg=5))
        job = wait(service, service.move(
            JointMove(joints_deg=[3., 0., 0., 0., 0., 0.], speed_percent=1, timeout_s=7.),
            "calibration-stall-01"))
        assert job["status"] == "outcome_unknown", job
        assert job["error_code"] == "tracking_error"
        assert "recoverable" not in job
    finally:
        service.close()


def test_control_mode_transition_is_not_recovered(tmp_path):
    class StuckMode(ScriptedMujoco):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.ctrl_mode = 2

        def set_control_mode(self, current_deg, speed_percent, checkpoint):
            checkpoint()
            self.commands.append(("control_mode", list(current_deg), speed_percent))

    from piperx_middleware.models import ControlMode
    backend = StuckMode()
    service = make_service(tmp_path, backend)
    try:
        job = wait(service, service.move(ControlMode(timeout_s=0.3), "control-mode-stall-01"))
        assert job["status"] == "outcome_unknown", job
        assert job["error_code"] == "execution_timeout"
        assert "recoverable" not in job
    finally:
        service.close()


@pytest.mark.parametrize('unstable', [False, True])
def test_linear_tracking_error_confirms_hold_before_recovery(tmp_path, monkeypatch, unstable):
    """A contact-disturbed joint triggers tracking_error, never a replay or success."""
    short_confirmation(monkeypatch, window=.8)
    class Disturbed(ScriptedMujoco):
        def joint_target(self, q):
            super().joint_target(q)
            # Physical disturbance while first command is active; not a command replay.
            if not self.hold_calls:
                self.drift_deg=.4
    kin=PiperKinematics([0.,0.,.1425]);backend=Disturbed(q=[0.,20.,-25.,0.,25.,0.])
    def held(b):
        b.drift_deg=0.
        if unstable:b.velocity[0]=3.
    backend.post_hold=held
    service=make_service(tmp_path,backend)
    try:
        pose=kin.pose(backend.q);target=list(pose['xyz_m']);target[2]+=.01
        job=wait(service,service.primitive(MoveLinear(xyz_m=target,speed_percent=5,timeout_s=10.),'linear-track-test'))
        assert len(backend.hold_calls)==1
        if unstable:
            assert job['status']=='outcome_unknown' and job['error_code']=='tracking_error'
            assert not job['stop_result']['confirmed_stopped'] and 'recoverable' not in job
        else:
            assert job['status']=='failed' and job['error_code']=='motion_blocked'
            assert job['original_error_code']=='tracking_error' and job['recoverable']
            assert job['stop_result']['confirmed_stopped'] and job['after']
        assert service.active is None
    finally:
        service.close()
