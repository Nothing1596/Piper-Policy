"""Operator-owned approval workflow shared by every action transport.

An admitted running action may finish after its frontend disappears. New actions
need a live session. No transport error causes an action to be replayed.
"""
import copy
import json
import os
import time
from pathlib import Path

from .interaction_types import InteractionPolicy
from .models import ControlMode, DomainError, ExecuteRequest, GripperMove, JointMove
from .request_context import control_session_id


class InteractionControls:
    def _init_interaction(self):
        from .control_sessions import ControlSessions
        self.sessions = ControlSessions()
        self.pending_plans = {}
        self._shutdown_on_session_loss = False
        self._session_lost = False
        self.on_idle_session_loss = None
        self._managed_started = time.monotonic()
        self._session_ever_acquired = False
        self._startup_grace_s = 30.
        self._pending_session_loss = False
        self._last_expired_generation = None
        self.policy_file = Path(self.settings.data_dir) / 'interaction-policy.json'
        if self.policy_file.exists():
            self.policy = InteractionPolicy.model_validate_json(self.policy_file.read_text(encoding='utf-8'))
        elif self.settings.interaction_policy is not None:
            self.policy = self.settings.interaction_policy.model_copy(deep=True)
        else:
            self.policy = InteractionPolicy(mode='risk' if self.backend.name == 'agx' else 'auto')

    @property
    def interaction_required(self):
        return self.backend.name == 'agx' or self.settings.managed_control or self.policy.force

    def _require_session(self):
        if self.interaction_required:
            self.sessions.require(control_session_id.get())

    def acquire_session(self, owner, shutdown_on_loss=False):
        with self.lock:
            self._require_open()
            if self.active:
                raise DomainError('busy', 'Current action must finish before acquiring another control session.')
            self._expire_control()
            self._require_open()
            public = self.sessions.public()
            result = self.sessions.acquire(owner, drained=True)
            self._shutdown_on_session_loss = shutdown_on_loss
            self._session_lost = False
            self._session_ever_acquired = True
            self.store.event('control_session_acquired', owner=owner)
            return result

    def heartbeat_session(self, session_id):
        self._require_open()
        return self.sessions.heartbeat(session_id)

    def release_session(self, session_id):
        with self.lock:
            result = self.sessions.release(session_id)
            self._shutdown_on_session_loss = False
            self._cancel_pending('session_released')
            return dict(result, status='draining' if self.active else 'released', active_job_id=self.active)

    def _expire_control(self):
        expired_now = self.sessions.expire()
        public = self.sessions.public()
        if expired_now or (public.get('expired') and public.get('generation') != self._last_expired_generation):
            self._pending_session_loss = True
            self._last_expired_generation = public.get('generation')
            self._session_lost = True
        if self._pending_session_loss and self.lock.acquire(blocking=False):
            try:
                self._cancel_pending('session_expired')
                self._pending_session_loss = False
                self.store.event('control_session_expired', active_job_id=self.active,
                                 behavior='finish_current_no_new_actions')
            finally:
                self.lock.release()

        if self.pending_plans and self.lock.acquire(blocking=False):
            try:
                self._cancel_pending('stop_requested', stopped_only=True)
            finally:
                self.lock.release()
        unclaimed = (self.settings.managed_control and not self._session_ever_acquired
                     and time.monotonic() - self._managed_started >= self._startup_grace_s)
        abandoned = self._session_lost and self._shutdown_on_session_loss
        if (unclaimed or abandoned) and self.on_idle_session_loss is not None and self.lock.acquire(blocking=False):
            try:
                # Atomically stop admissions before asking uvicorn to close. A new
                # owner cannot acquire between detecting abandonment and exit.
                if not self.active and not self.closing and not self.closed:
                    self.closing = True
                    self.on_idle_session_loss()
            finally:
                self.lock.release()

    def _cancel_pending(self, reason, *, stopped_only=False):
        for job_id in list(self.pending_plans):
            if stopped_only and self.pending_plans[job_id]['stop_generation'] == self.stop_generation:
                continue
            job = self.store.get(job_id=job_id)
            if job and job['status'] == 'awaiting_approval':
                job.update(status='cancelled', error_code=reason, error='Pending approval cancelled; no command sent.',
                           finished_at=time.time())
                self.store.put(job)
            self.pending_plans.pop(job_id, None)

    def interaction_state(self):
        with self.lock:
            self._expire_control()
            pending = [self.store.get(job_id=i) for i in self.pending_plans]
            from .approval_policy import PolicyEngine
            effective = PolicyEngine(self.policy).effective_limits(settings=self.settings,
                measured_limits=self.measured_limits, connection_epoch=self.epoch)
            return {'policy': self.policy.model_dump(), 'pending': [j for j in pending if j],
                    'session': self.sessions.public(), 'required': self.interaction_required,
                    'scene_collision_checked': False,
                    'effective_limits': effective,
                    'device_limits_deg': self.capabilities()['limits']['joint_limits_deg']}

    def configure_interaction(self, policy):
        with self.lock:
            self._require_open()
            self._require_session()
            if self.active:
                raise DomainError('busy', 'Wait for the current action before changing limits or approval mode.')
            if policy.version != self.policy.version:
                raise DomainError("policy_changed", "Policy changed since this proposal; inspect and propose again.")
            candidate = policy.model_copy(deep=True)
            candidate.version = self.policy.version + 1
            # Validate the intersection even before CAN is opened, without sending commands.
            from .approval_policy import PolicyEngine
            PolicyEngine(candidate).effective_limits(settings=self.settings,
                measured_limits=self.measured_limits, connection_epoch=self.epoch)
            temporary = self.policy_file.with_suffix('.tmp')
            with temporary.open('w', encoding='utf-8') as f:
                f.write(candidate.model_dump_json(indent=2)); f.flush(); os.fsync(f.fileno())
            temporary.replace(self.policy_file)
            self.policy = candidate
            self._cancel_pending('policy_changed')
            self.plans.clear()
            self.store.event('interaction_policy_changed', version=candidate.version, mode=candidate.mode)
            return self.interaction_state()

    def configure_profile(self, changes):
        """Operator-only persistent host settings; no firmware writes or auto-connect."""
        from .models import Settings
        from .approval_policy import PolicyEngine
        forbidden = {'backend', 'data_dir', 'host', 'port', 'managed_profile_id',
                     'managed_control', 'interaction_policy', 'simulation_asset', 'simulation_seed'}
        allowed = set(Settings.model_fields) - forbidden
        if not changes or not set(changes) <= allowed:
            raise DomainError('invalid_settings', 'Use /mode for backend/target and /limits for policy; unsupported profile field.', 422)
        with self.lock:
            self._require_open()
            self._require_session()
            if self.active:
                raise DomainError('busy', 'Wait for the current action before changing profile settings.')
            from pydantic import ValidationError
            try:
                candidate = Settings.model_validate(self.settings.model_dump() | changes)
            except ValidationError as exc:
                raise DomainError("invalid_settings", "Settings do not match the profile schema.", 422) from exc
            PolicyEngine(self.policy).effective_limits(settings=candidate,
                measured_limits=self.measured_limits, connection_epoch=self.epoch)
            # Disconnect first; cleanup failure must leave settings untouched.
            self._disconnect_locked()
            path = Path(self.settings.data_dir) / 'config.json'
            temporary = path.with_suffix('.tmp')
            with temporary.open('w', encoding='utf-8') as f:
                f.write(candidate.model_dump_json(indent=2)); f.flush(); os.fsync(f.fileno())
            temporary.replace(path)
            self.settings = candidate
            if hasattr(self.backend, 'settings'):
                self.backend.settings = candidate
            self.parameter_version += 1
            self.plans.clear()
            self._cancel_pending('parameters_changed')
            self.store.event('profile_settings_changed', fields=sorted(changes), parameter_version=self.parameter_version)
            return {'status': 'saved', 'reconnect_required': True, 'motion_sent': False,
                    'parameter_version': self.parameter_version, 'configured': candidate.model_dump(mode='json')}

    def _approval_identity(self):
        return dict(instance_id=self.instance_id, connection_epoch=self.epoch,
                    policy_version=self.policy.version, parameter_version=self.parameter_version)

    def _evaluate_interaction(self, command, state, resolution=None):
        from .approval_policy import PolicyEngine
        limits = self.measured_limits
        if limits and limits.get('connection_epoch') != self.epoch:
            limits = None
        public = state.public() | {'connection_epoch': self.epoch,
                                  'observation_identity': {'connection_epoch': self.epoch}}
        return PolicyEngine(self.policy).evaluate(command.model_dump(), public, settings=self.settings,
                                                 resolution=resolution, measured_limits=limits)

    def _queue_approval(self, job, command, plan, state):
        if not self.interaction_required:
            return False
        self._require_session()
        decision = self._evaluate_interaction(command, state, plan.get('resolution'))
        job['approval'] = decision
        job['control_session_generation'] = self.sessions.public().get('generation')
        if decision['decision'] != 'ask':
            return False
        from .approval_binding import bind_approval
        job.update(status='awaiting_approval', approval_binding=bind_approval(
            command.model_dump(), state.public(), **self._approval_identity()))
        with self.control_lock:
            with self.cancel_lock:
                if plan['stop_generation'] != self.stop_generation:
                    raise DomainError('cancelled', 'Stop requested while preparing approval; no command sent.')
            self.store.put(job, new=True)
            plan['consumed'] = True
            self.pending_plans[job['job_id']] = copy.deepcopy(plan)
        return True

    def decide_approval(self, job_id, approved):
        with self.lock:
            self._require_open()
            self._require_session()
            job = self.get_job(job_id)
            if job['status'] != 'awaiting_approval':
                return job
            if job.get('control_session_generation') != self.sessions.public().get('generation'):
                raise DomainError('session_conflict', 'Approval belongs to another control session.')
            if not approved:
                self.pending_plans.pop(job_id, None)
                job.update(status='cancelled', error_code='approval_denied', finished_at=time.time())
                self.store.put(job)
                return job
            if self.active:
                raise DomainError('busy', 'Wait for the active action before approving another.')
            plan = self.pending_plans.get(job_id)
            if plan is None:
                raise DomainError('approval_changed', 'Approval cannot be recovered after executor restart.')
            command = {'joint': JointMove, 'gripper': GripperMove, 'control_mode': ControlMode}[job['command']['kind']](**job['command'])
            state = self.backend.snapshot()
            try:
                from .approval_binding import validate_approval
                self._check(state, gripper=isinstance(command, GripperMove),
                            require_position_mode=not isinstance(command, ControlMode))
                validate_approval(job['approval_binding'], command.model_dump(), state.public(), **self._approval_identity())
                self._evaluate_interaction(command, state, plan.get('resolution'))
                self._check_window(state, command)
                with self.cancel_lock:
                    if plan['stop_generation'] != self.stop_generation:
                        raise DomainError('approval_changed', 'A stop invalidated this approval.')
            except DomainError as exc:
                self.pending_plans.pop(job_id, None)
                job.update(status='cancelled', error_code=exc.code, error=exc.message, finished_at=time.time())
                self.store.put(job)
                return job
            self.pending_plans.pop(job_id, None)
            job['status'] = 'accepted'
            job['approved_at'] = time.time()
            return self._start_job(job, command, plan, new=False)
