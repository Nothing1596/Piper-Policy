"""Regression tests for approval/session interaction correctness.
Focused tests for verified behaviors identified during independent review."""
import threading
import time
from contextlib import contextmanager
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from piperx_middleware.backends import SimBackend
from piperx_middleware.http_api import create_app
from piperx_middleware.interaction_types import InteractionPolicy, ExecutionLimits
from piperx_middleware.models import Settings, DomainError
from piperx_middleware.service import RobotService

MODEL = {'Authorization': 'Bearer ' + 'm' * 48}
OPERATOR = {'Authorization': 'Bearer ' + 'o' * 48}
MOVE = {'kind': 'joint', 'joints_deg': [1., 0., 0., 0., 0., 0.], 'speed_percent': 5}


@contextmanager
def running(tmp_path, mode='always', limits=None):
    policy = InteractionPolicy(mode=mode, limits=limits or ExecutionLimits())
    s = RobotService(SimBackend(), Settings(data_dir=tmp_path, managed_control=True, interaction_policy=policy))
    with TestClient(create_app(s, 'm'*48, 'o'*48)) as c:
        assert c.post('/v1/connect', headers=MODEL).status_code == 200
        yield s, c


def acquire(c):
    response = c.post('/operator/session', headers=OPERATOR, json={'owner': 'test-console'})
    assert response.status_code == 200, response.text
    return response.json()['session_id'], {'X-Piper-Control-Session': response.json()['session_id']}


def move(c, sid_header, ident='review-test-job', command=None):
    return c.post('/v1/move', headers=MODEL | sid_header, json={'request_id': ident, 'command': command or MOVE})


def test_missing_session_rejects_approval_no_commands(tmp_path):
    """Verify that approval requests without a valid session are rejected,
    and critically, that no backend commands are issued."""
    with running(tmp_path) as (s, c):
        sid, sid_header = acquire(c)

        # Create a pending approval
        j = move(c, sid_header, 'no-session-job').json()
        assert j['status'] == 'awaiting_approval'
        job_id = j['job_id']

        # Release session
        c.post('/operator/session/release', headers=OPERATOR | sid_header, json={'session_id': sid})

        # Track backend commands before approval attempt
        initial_command_count = len(s.backend.commands)

        # Try to approve without session - should fail
        response = c.post(f'/operator/approvals/{job_id}',
                         headers=OPERATOR,  # No session header
                         json={'approved': True})

        assert response.status_code == 409, "Approval without session should fail"
        assert 'missing_control_session' in response.text or 'session' in response.text.lower()

        # Verify NO commands were sent to backend
        assert len(s.backend.commands) == initial_command_count, \
            "Missing session approval must not execute motion commands"


def test_session_expiry_cancels_and_rejects_new_approvals(tmp_path):
    """Verify that when a session expires:
    1. Pending approvals are cancelled
    2. New approval attempts with stale session are rejected"""
    with running(tmp_path) as (s, c):
        # Create fake clock that can be advanced
        fake_time = [time.monotonic()]
        def fake_clock():
            return fake_time[0]

        # Inject clock BEFORE acquiring session
        s.sessions._now = fake_clock

        sid, sid_header = acquire(c)

        # Create pending approval
        j = move(c, sid_header, 'expiry-job').json()
        assert j['status'] == 'awaiting_approval'
        job_id = j['job_id']

        # Advance time past TTL (default 5s)
        fake_time[0] += 6.0

        # Trigger expiry check
        s._expire_control()

        # Verify job was cancelled
        job_status = c.get(f'/v1/jobs/{job_id}', headers=MODEL).json()
        assert job_status['status'] == 'cancelled'
        assert job_status['error_code'] == 'session_expired'

        # Verify expired session cannot approve
        response = c.post(f'/operator/approvals/{job_id}',
                         headers=OPERATOR | sid_header,
                         json={'approved': True})
        assert response.status_code == 409
        assert 'session_expired' in response.text or 'expired' in response.text.lower()


def test_different_session_cannot_execute_stale_approval(tmp_path):
    """Verify that approvals are bound to specific session generations,
    and cannot be executed by a different session."""
    with running(tmp_path) as (s, c):
        sid1, sid1_header = acquire(c)

        # Create pending approval with session 1
        j = move(c, sid1_header, 'stale-approval-job').json()
        assert j['status'] == 'awaiting_approval'
        job_id = j['job_id']
        original_gen = j['control_session_generation']

        # Release first session - this cancels pending approvals
        c.post('/operator/session/release', headers=OPERATOR | sid1_header, json={'session_id': sid1})

        # Verify job was cancelled by session release
        job_status = c.get(f'/v1/jobs/{job_id}', headers=MODEL).json()
        assert job_status['status'] == 'cancelled'
        assert job_status['error_code'] == 'session_released'

        # Acquire new session
        sid2, sid2_header = acquire(c)
        new_gen = s.sessions.public()['generation']
        assert new_gen != original_gen, "New session should have different generation"

        # Attempting to approve already-cancelled job returns the cancelled job
        response = c.post(f'/operator/approvals/{job_id}',
                         headers=OPERATOR | sid2_header,
                         json={'approved': True})

        # Response is OK but returns the cancelled job (decide_approval early-exits at line 191-192)
        assert response.status_code == 200
        result = response.json()
        assert result['status'] == 'cancelled', "Stale approval cannot be re-approved"


def test_completed_request_retrieval_never_repeats_commands(tmp_path):
    """Verify that duplicate request_id returns completed job from ledger
    without replaying motion. This is intentional recovery semantics."""
    # Use 'auto' mode to avoid approval requirement
    with running(tmp_path, mode='auto') as (s, c):
        sid, sid_header = acquire(c)

        # Execute and complete a job
        j = move(c, sid_header, 'idempotent-job', MOVE).json()

        # Wait for completion
        deadline = time.monotonic() + 5
        while j['status'] in ('accepted', 'running'):
            assert time.monotonic() < deadline, "Job should complete"
            time.sleep(0.01)
            j = c.get(f'/v1/jobs/{j["job_id"]}', headers=MODEL).json()

        assert j['status'] == 'succeeded'
        job_id = j['job_id']

        # Record backend command count
        command_count_after_first = len(s.backend.commands)

        # Retry with same request_id - should return existing job
        retry = move(c, sid_header, 'idempotent-job', MOVE).json()

        assert retry['job_id'] == job_id, "Should return same job"
        assert retry['status'] == 'succeeded', "Should return completed status"

        # Verify NO new commands were sent
        assert len(s.backend.commands) == command_count_after_first, \
            "Duplicate request_id must not replay motion - intentional recovery semantics"


def test_fresh_simulation_auto_needs_no_threshold_setup(tmp_path):
    """First-use defaults admit a valid move through HTTP without approval setup."""
    service = RobotService(SimBackend(), Settings(data_dir=tmp_path, managed_control=True))
    with TestClient(create_app(service, 'm'*48, 'o'*48)) as client:
        assert client.post('/v1/connect', headers=MODEL).status_code == 200
        _, session = acquire(client)
        assert service.policy.mode == 'auto'
        assert service.policy.automatic.max_joint_step_deg is None
        response = move(client, session, 'first-simulation-move')
        assert response.status_code == 200, response.text
        job = response.json()
        deadline = time.monotonic() + 5
        while job['status'] in ('accepted', 'running'):
            assert time.monotonic() < deadline
            time.sleep(.01)
            job = client.get(f'/v1/jobs/{job["job_id"]}', headers=MODEL).json()
        assert job['status'] == 'succeeded'
        assert not service.pending_plans
        sent_before_rejection = len(service.backend.commands)
        # Auto still goes through the same hard-limit checks.
        bad = dict(MOVE, joints_deg=[999., 0., 0., 0., 0., 0.])
        response = move(client, session, 'invalid-simulation-move', bad)
        assert response.status_code == 422, response.text
        assert response.json()['error']['code'] == 'joint_limits'
        assert len(service.backend.commands) == sent_before_rejection


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
