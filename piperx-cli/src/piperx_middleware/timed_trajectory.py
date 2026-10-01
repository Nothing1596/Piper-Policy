"""Simulation-only transport for immutable, externally planned joint timelines.

The ordinary RobotService remains the single durable action owner. This adapter
never admits trajectories to CAN and never substitutes endpoint motion for a
Cartesian path. Every supplied sample is written once at its supplied time.
"""
from __future__ import annotations

import copy
import math
import time

from pydantic import Field, model_validator

from .models import DomainError, ExecuteRequest, JointMove, Number, StrictModel


class TimedTrajectory(StrictModel):
    instance_id: str = Field(min_length=1, max_length=80)
    connection_epoch: int = Field(ge=0, strict=True)
    parameter_version: int = Field(ge=0, strict=True)
    start_joints_deg: list[Number] = Field(min_length=6, max_length=6)
    start_gripper_m: Number = Field(ge=0, le=.09)
    joints_deg: list[list[Number]] = Field(min_length=1, max_length=4096)
    relative_times_s: list[Number] = Field(min_length=1, max_length=4096)
    settle_timeout_s: Number = Field(default=3., gt=0, le=30)
    settle_position_rad: Number = Field(default=.03, gt=0, le=.05)
    settle_velocity_rad_s: Number = Field(default=.05, gt=0, le=.1)
    settle_samples: int = Field(default=10, ge=3, le=100, strict=True)
    control_hz: Number = Field(default=30., ge=1, le=200)
    tracking_error_rad: Number = Field(default=.12, gt=0, le=.2)
    max_lateness_s: Number = Field(default=.1, gt=0, le=.2)
    note: str = Field(min_length=1, max_length=2000)

    @model_validator(mode='after')
    def timeline(self):
        if len(self.joints_deg) != len(self.relative_times_s):
            raise ValueError('One timestamp is required per six-joint sample')
        if any(len(q) != 6 for q in self.joints_deg):
            raise ValueError('Every trajectory sample must have six joints')
        if not self.note.strip():
            raise ValueError('A nonempty motion note is required')
        if not 0 < self.relative_times_s[0] or self.relative_times_s[-1] > 600:
            raise ValueError('Timeline must begin after zero and finish within 600 seconds')
        if any(b <= a for a, b in zip(self.relative_times_s, self.relative_times_s[1:])):
            raise ValueError('Timeline must be strictly increasing')
        return self


class TimedTrajectoryRequest(StrictModel):
    trajectory: TimedTrajectory
    request_id: str = Field(min_length=8, max_length=128, pattern=r'^[A-Za-z0-9_.:-]+$')


class TimedTrajectoryControls:
    def _timed_simulation_guard(self):
        self._require_open()
        if self.backend.name != 'mujoco' or self.settings.backend != 'mujoco':
            raise DomainError('simulation_only', 'Timed GPT-Policy trajectories require the MuJoCo backend', 422)
        if self.settings.control_profile != 'direct' or self.policy.mode != 'auto':
            raise DomainError('trajectory_policy', 'Baseline timelines require direct-profile simulation with operator-selected auto policy', 422)

    def preview_timed_trajectory(self, trajectory: TimedTrajectory):
        with self.lock:
            self._timed_simulation_guard()
            self._require_session()
            if (trajectory.instance_id != self.instance_id or trajectory.connection_epoch != self.epoch
                    or trajectory.parameter_version != self.parameter_version):
                raise DomainError('state_changed', 'Executor, connection or TCP parameters changed during planning')
            state = self.backend.snapshot()
            self._check(state, gripper=True)
            if (max(abs(a-b) for a,b in zip(state.q_deg, trajectory.start_joints_deg)) > .1
                    or abs(state.gripper_width_m-trajectory.start_gripper_m) > .001):
                raise DomainError('state_changed', 'Arm or gripper moved during planning')
            timeout = trajectory.relative_times_s[-1] + trajectory.settle_timeout_s + 1.
            endpoint = JointMove(joints_deg=trajectory.joints_deg[-1], speed_percent=100, timeout_s=timeout)
            # Validate every intermediate sample against the same hard site and
            # device limits. Approval of an endpoint is never approval of a path.
            from .models import JOINT_LIMITS_DEG
            for point in trajectory.joints_deg:
                if any(not lo <= q <= hi for q,(lo,hi) in zip(point, JOINT_LIMITS_DEG)):
                    raise DomainError('joint_limits', 'A timeline sample exceeds Piper joint limits', 422)
                self._evaluate_interaction(JointMove(joints_deg=point, speed_percent=100, timeout_s=timeout), state)
            self._validate_timeline_envelope(trajectory)
            plan = self.preview(endpoint)
            internal = self.plans[plan['plan_id']]
            internal['timed_trajectory'] = trajectory.model_dump()
            internal['parameter_version'] = self.parameter_version
            return {k: copy.deepcopy(v) for k,v in internal.items() if k != 'deadline'}

    @staticmethod
    def _validate_timeline_envelope(trajectory):
        import numpy as np
        points = np.deg2rad([trajectory.start_joints_deg, *trajectory.joints_deg])
        times = np.array([0., *trajectory.relative_times_s])
        # Independent admission envelope, not proof of Ruckig provenance or a
        # collision guarantee. Never retime, clamp, omit or modify a sample.
        interval_velocity = np.diff(points,axis=0)/np.diff(times)[:,None]
        if np.max(np.abs(interval_velocity)) > math.pi + 1e-8:
            raise DomainError('trajectory_velocity', 'Timeline exceeds the fixed simulation velocity envelope', 422)
        derivative = points
        for order, limit in enumerate((math.pi,20.,500.),start=1):
            derivative = np.gradient(derivative,times,axis=0,edge_order=2 if len(times)>2 else 1)
            if not np.isfinite(derivative).all() or np.max(np.abs(derivative)) > limit + 1e-8:
                raise DomainError('trajectory_derivative', f'Timeline derivative order {order} exceeds simulation admission envelope', 422)

    def preview_timed_trajectory_validation(self, value, state):
        trajectory = TimedTrajectory(**value)
        if (trajectory.instance_id != self.instance_id or trajectory.connection_epoch != self.epoch
                or trajectory.parameter_version != self.parameter_version):
            raise DomainError('state_changed', 'Timeline executor identity changed')
        self._validate_timeline_envelope(trajectory)
        self._check(state, gripper=True)
        if abs(state.gripper_width_m-trajectory.start_gripper_m) > .001:
            raise DomainError('state_changed', 'Gripper moved since timeline preview')
        for point in trajectory.joints_deg:
            self._evaluate_interaction(JointMove(joints_deg=point, speed_percent=100,
                timeout_s=trajectory.relative_times_s[-1]+trajectory.settle_timeout_s+1.), state)

    def timed_trajectory(self, trajectory: TimedTrajectory, request_id: str):
        with self.lock:
            self._timed_simulation_guard()
            previous = self.store.get(request_id=request_id)
            if previous:
                if previous.get('timed_trajectory') != trajectory.model_dump():
                    raise DomainError('idempotency_conflict', 'This request_id refers to another timeline')
                return previous
            plan = self.preview_timed_trajectory(trajectory)
            return self.execute(ExecuteRequest(plan_id=plan['plan_id'], request_id=request_id))

    def _run_timed_trajectory(self, job, command, plan):
        trajectory = TimedTrajectory(**plan['timed_trajectory'])
        started = time.monotonic()
        deadline = started + command.timeout_s
        last_write = started
        actual_times = []
        reference = list(trajectory.start_joints_deg)
        try:
            with self.lock:
                self._timed_simulation_guard()
                if self.parameter_version != trajectory.parameter_version or self.epoch != trajectory.connection_epoch:
                    raise DomainError('state_changed', 'Timeline identity changed before execution')
                state = self._guard_running(command, deadline)
                if max(abs(a-b) for a,b in zip(state.q_deg, reference)) > .1:
                    raise DomainError('state_changed', 'Arm moved before timeline execution')
                job['status'] = 'running'
                self.store.put(job)
                # Durable attempted marker always precedes the first backend write.
                job['command_attempted'] = True
                self.store.put(job)
                self.backend.begin_timed_trajectory(state.q_deg)
            for index, (relative, point) in enumerate(zip(trajectory.relative_times_s, trajectory.joints_deg)):
                target_time = started + relative
                if self.cancel.wait(max(0., target_time-time.monotonic())):
                    raise DomainError('cancelled', 'Stop requested')
                with self.lock:
                    state = self._guard_running(command, deadline)
                    if abs(state.gripper_width_m-trajectory.start_gripper_m) > .002:
                        raise DomainError('gripper_changed', 'Held gripper width changed during arm motion')
                    if max(abs(a-b) for a,b in zip(state.q_deg, reference)) > math.degrees(trajectory.tracking_error_rad):
                        raise DomainError('tracking_error', 'Measured arm did not track the previous timeline sample')
                    sent_at = time.monotonic()
                    if sent_at-target_time > trajectory.max_lateness_s:
                        raise DomainError('trajectory_late', 'Timeline deadline missed; no samples skipped or accelerated')
                    self._check_cancel_deadline(deadline)
                    self.backend.joint_target(point)
                    last_write = time.monotonic()
                    reference = list(point)
                    actual_times.append(sent_at-started)
            settle_started = time.monotonic()
            consecutive, previous_stamp = 0, None
            while time.monotonic()-settle_started < trajectory.settle_timeout_s:
                if self.cancel.wait(1./trajectory.control_hz):
                    raise DomainError('cancelled', 'Stop requested')
                with self.lock:
                    state = self._guard_running(command, deadline)
                    position_error = math.radians(max(abs(a-b) for a,b in zip(state.q_deg, reference)))
                    velocity = math.radians(max(abs(v) for v in state.velocity_deg_s))
                    stamp = state.feedback_stamp_s
                    fresh = stamp is not None and stamp > last_write and stamp != previous_stamp
                    if fresh:
                        consecutive = consecutive+1 if (position_error <= trajectory.settle_position_rad
                            and velocity <= trajectory.settle_velocity_rad_s) else 0
                        previous_stamp = stamp
                    if consecutive >= trajectory.settle_samples:
                        job.update(status='succeeded', after=state.public(), verification='Fresh simulation feedback settled; no grasp or task-success claim',
                                   settle={'settled': True, 'consecutive_samples': consecutive,
                                           'max_position_error_rad': position_error, 'max_velocity_rad_s': velocity})
                        break
            else:
                raise DomainError('settle_timeout', 'Endpoint simulation feedback did not settle')
        except Exception as exc:
            with self.lock:
                self.lease = None
                hold = self._hold(command) if job['command_attempted'] else {'status': 'not_needed'}
                job.update(status='cancelled' if isinstance(exc,DomainError) and exc.code == 'cancelled' and
                           hold['status'] in ('hold_requested','not_needed') else 'outcome_unknown' if job['command_attempted'] else 'failed',
                           error=str(exc), error_code=getattr(exc,'code','execution_error'), stop_result=hold)
        finally:
            with self.lock:
                job['execution_timing'] = {'planned_relative_times_s': trajectory.relative_times_s,
                    'sent_relative_times_s': actual_times, 'samples_sent': len(actual_times),
                    'samples_planned': len(trajectory.joints_deg), 'source_clock': 'executor_host_monotonic',
                    'interpolated': False, 'samples_skipped': False}
                job['finished_at'] = time.time()
                try:
                    self.store.put(job)
                    self.store.event('job_finished', job_id=job['job_id'], status=job['status'])
                finally:
                    self.active = None
                    self.native_linear_active = False
