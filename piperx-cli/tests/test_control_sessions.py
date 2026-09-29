"""ControlSessions: single owner, explicit expiry, no silent resume."""
import itertools
import threading

import pytest

from piperx_middleware.control_sessions import (
    HEARTBEAT_INTERVAL_S,
    SESSION_TTL_S,
    ControlSessions,
)
from piperx_middleware.models import DomainError


class Clock:
    def __init__(self, start=1000.0):
        self.value = float(start)

    def __call__(self):
        return self.value

    def advance(self, seconds):
        self.value += seconds


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def sessions(clock):
    counter = itertools.count(1)
    factory = lambda: f"sid-{next(counter):04d}-" + "x" * 32
    return ControlSessions(monotonic=clock, session_id_factory=factory)


def codes(exc):
    return exc.value.code


def test_defaults_match_contract():
    assert SESSION_TTL_S == 5.0
    assert HEARTBEAT_INTERVAL_S == 1.0
    sessions = ControlSessions()
    assert sessions.ttl_s == 5.0
    assert sessions.heartbeat_s == 1.0
    assert sessions.public()["state"] == "inactive"


def test_acquire_heartbeat_require_release(sessions):
    session = sessions.acquire("console")
    assert session["owner"] == "console"
    assert len(session["session_id"]) >= 32
    assert session["expires_in_s"] == 5.0
    assert session["heartbeat_interval_s"] == 1.0
    assert sessions.require(session["session_id"]) is None
    beat = sessions.heartbeat(session["session_id"])
    assert beat["session_id"] == session["session_id"]
    assert beat["status"] == "active"
    released = sessions.release(session["session_id"])
    assert released == {"status": "released", "session_id": session["session_id"], "owner": "console"}
    with pytest.raises(DomainError) as exc:
        sessions.require(session["session_id"])
    assert codes(exc) == "missing_control_session"
    assert sessions.public()["state"] == "inactive"


def test_single_owner_conflict(sessions):
    first = sessions.acquire("console")
    with pytest.raises(DomainError) as exc:
        sessions.acquire("console")
    assert codes(exc) == "session_conflict"
    with pytest.raises(DomainError) as exc:
        sessions.acquire("second-terminal")
    assert codes(exc) == "session_conflict"
    with pytest.raises(DomainError) as exc:
        sessions.heartbeat("some-other-session-id")
    assert codes(exc) == "session_conflict"
    with pytest.raises(DomainError) as exc:
        sessions.release("some-other-session-id")
    assert codes(exc) == "session_conflict"
    # The original session is untouched by the failed attempts.
    assert sessions.require(first["session_id"]) is None


def test_expiry_is_explicit_and_never_resumes(sessions, clock):
    session = sessions.acquire("console")
    clock.advance(4.9)
    assert sessions.expire() is False
    assert sessions.require(session["session_id"]) is None
    clock.advance(0.2)
    assert sessions.expire() is True
    # expire() reports the transition exactly once.
    assert sessions.expire() is False
    assert sessions.public() == {
        "state": "expired", "active": False, "expired": True, "owner": "console",
        "generation": 1, "expires_in_s": 0.0, "expired_for_s": pytest.approx(0.1),
        "heartbeat_interval_s": 1.0, "ttl_s": 5.0,
    }
    with pytest.raises(DomainError) as exc:
        sessions.heartbeat(session["session_id"])
    assert codes(exc) == "session_expired"
    with pytest.raises(DomainError) as exc:
        sessions.require(session["session_id"])
    assert codes(exc) == "session_expired"
    # A new session cannot silently replace the expired one.
    with pytest.raises(DomainError) as exc:
        sessions.acquire("console")
    assert codes(exc) == "session_expired"


def test_acquire_after_expiry_requires_drained(sessions, clock):
    old = sessions.acquire("console")
    clock.advance(5.1)
    with pytest.raises(DomainError) as exc:
        sessions.acquire("console")
    assert codes(exc) == "session_expired"
    fresh = sessions.acquire("console", drained=True)
    assert fresh["session_id"] != old["session_id"]
    assert fresh["generation"] == 2
    with pytest.raises(DomainError) as exc:
        sessions.require(old["session_id"])
    assert codes(exc) == "session_conflict"  # a different session now owns the lease
    assert sessions.require(fresh["session_id"]) is None


def test_release_expired_clears_residue(sessions, clock):
    session = sessions.acquire("console")
    clock.advance(6)
    released = sessions.release(session["session_id"])
    assert released["status"] == "expired_released"
    assert sessions.public()["state"] == "inactive"
    fresh = sessions.acquire("console")
    assert fresh["session_id"] != session["session_id"]


def test_missing_unknown_and_none(sessions):
    with pytest.raises(DomainError) as exc:
        sessions.require(None)
    assert codes(exc) == "missing_control_session"
    with pytest.raises(DomainError) as exc:
        sessions.require("")
    assert codes(exc) == "missing_control_session"
    with pytest.raises(DomainError) as exc:
        sessions.require("never-issued")
    assert codes(exc) == "missing_control_session"
    with pytest.raises(DomainError) as exc:
        sessions.heartbeat(None)
    assert codes(exc) == "missing_control_session"
    with pytest.raises(DomainError) as exc:
        sessions.release(None)
    assert codes(exc) == "missing_control_session"


def test_lazy_expiry_on_public_is_still_reported_by_expire(sessions, clock):
    sessions.acquire("console")
    clock.advance(5.5)
    # public() applies the clock even when expire() was never called, but the
    # transition stays pending until expire() reports it exactly once.
    assert sessions.public()["state"] == "expired"
    assert sessions.expire() is True
    assert sessions.expire() is False


def test_expiry_detected_by_require_is_reported_once_by_expire(sessions, clock):
    session = sessions.acquire("console")
    clock.advance(5.5)
    with pytest.raises(DomainError) as exc:
        sessions.require(session["session_id"])
    assert codes(exc) == "session_expired"
    assert sessions.expire() is True
    assert sessions.expire() is False


def test_expiry_detected_by_heartbeat_or_public_is_reported_once(sessions, clock):
    session = sessions.acquire("console")
    clock.advance(5.5)
    assert sessions.public()["expired"] is True
    with pytest.raises(DomainError) as exc:
        sessions.heartbeat(session["session_id"])
    assert codes(exc) == "session_expired"
    # Repeated read-only observation never consumes the notification.
    assert sessions.public()["expired"] is True
    assert sessions.expire() is True
    assert sessions.expire() is False


def test_refused_acquire_does_not_consume_pending_expiry(sessions, clock):
    sessions.acquire("console")
    clock.advance(5.5)
    assert sessions.public()["state"] == "expired"
    with pytest.raises(DomainError) as exc:
        sessions.acquire("second-terminal")
    assert codes(exc) == "session_expired"
    assert sessions.expire() is True
    assert sessions.expire() is False


def test_pending_expiry_survives_release_of_expired_residue(sessions, clock):
    session = sessions.acquire("console")
    clock.advance(5.5)
    assert sessions.release(session["session_id"])["status"] == "expired_released"
    assert sessions.public()["state"] == "inactive"
    # The supervisor that only polls expire() must still learn about the loss.
    assert sessions.expire() is True
    assert sessions.expire() is False


def test_new_session_after_drain_clears_pending_expiry(sessions, clock):
    sessions.acquire("console")
    clock.advance(5.5)
    assert sessions.public()["state"] == "expired"
    fresh = sessions.acquire("console", drained=True)
    # A superseded expiry must not make the supervisor act on the new session.
    assert sessions.expire() is False
    assert sessions.require(fresh["session_id"]) is None


def test_heartbeat_extends_beyond_original_deadline(sessions, clock):
    session = sessions.acquire("console")
    for _ in range(5):
        clock.advance(4)
        sessions.heartbeat(session["session_id"])
        assert sessions.require(session["session_id"]) is None
    assert sessions.public()["state"] == "active"


def test_public_never_exposes_the_session_id(sessions):
    session = sessions.acquire("console")
    public = sessions.public()
    assert "session_id" not in public
    assert session["session_id"] not in repr(public)
    assert set(public) == {"state", "active", "expired", "owner", "generation",
                           "expires_in_s", "heartbeat_interval_s", "ttl_s"}


def test_owner_validation(sessions):
    for bad in (None, "", "   ", "x" * 129, "bad\nowner", "bad\x00owner", 5):
        with pytest.raises(DomainError) as exc:
            sessions.acquire(bad)
        assert codes(exc) == "invalid_request"
    assert sessions.acquire("  operator one  ")["owner"] == "operator one"


def test_invalid_configuration_rejected():
    with pytest.raises(ValueError):
        ControlSessions(ttl_s=0)
    with pytest.raises(ValueError):
        ControlSessions(ttl_s=5, heartbeat_s=10)
    with pytest.raises(ValueError):
        ControlSessions(heartbeat_s=0)


def test_thread_safe_single_owner(clock):
    sessions = ControlSessions(monotonic=clock, ttl_s=3600, heartbeat_s=1)
    session = sessions.acquire("console")
    errors = []
    barrier = threading.Barrier(8)

    def worker():
        barrier.wait()
        try:
            for _ in range(200):
                sessions.require(session["session_id"])
                sessions.heartbeat(session["session_id"])
        except Exception as exc:  # pragma: no cover - failure path
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not errors
    assert sessions.public()["state"] == "active"
    sessions.release(session["session_id"])
    assert sessions.public()["state"] == "inactive"
