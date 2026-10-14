"""Single-owner operator control sessions with explicit, monotonic expiry.

The session object performs no hardware I/O and holds no operator credential.
It only tracks *who* may currently drive a managed real session:

* exactly one owner at a time (``session_conflict``);
* a heartbeat every ``heartbeat_interval_s`` refreshes the lease, and a silent
  session expires after ``ttl_s``;
* an expired session is never resumed silently - ``heartbeat``/``require``
  raise ``session_expired`` and the next ``acquire`` is refused until the caller
  states that no action is active (``drained=True``) or releases the expired
  session after draining;
* a missing session and an unknown session id are distinguished from a
  conflicting or expired one (``missing_control_session``);
* the expiry transition is *sticky*: whichever accessor notices it first
  (``require``/``heartbeat``/``public``/``release``), the next ``expire()``
  reports it exactly once.  A supervisor that only polls ``expire()`` therefore
  cannot miss an expiry that an HTTP handler happened to observe first, and it
  is never told about the same expiry twice.  Acquiring a new session after a
  verified drain ends the episode and clears the pending report.

The monotonic clock is injectable so expiry can be tested without sleeping.
"""
from __future__ import annotations

import hmac
import secrets
import threading
import time
from dataclasses import dataclass
from typing import Callable

from .models import DomainError

HEARTBEAT_INTERVAL_S = 1.0
SESSION_TTL_S = 5.0
MAX_OWNER_LENGTH = 128


@dataclass
class _Session:
    session_id: str
    owner: str
    created_at: float
    last_seen: float
    generation: int


class ControlSessions:
    """Thread-safe, single-owner, expiring control-session registry."""

    def __init__(self,
                 ttl_s: float = SESSION_TTL_S,
                 heartbeat_s: float = HEARTBEAT_INTERVAL_S,
                 *,
                 monotonic: Callable[[], float] | None = None,
                 session_id_factory: Callable[[], str] | None = None):
        if not isinstance(ttl_s, (int, float)) or isinstance(ttl_s, bool) or not 0 < ttl_s <= 3600:
            raise ValueError("ttl_s must be a positive number of seconds (<= 3600)")
        if not isinstance(heartbeat_s, (int, float)) or isinstance(heartbeat_s, bool) or not 0 < heartbeat_s <= ttl_s:
            raise ValueError("heartbeat_s must be positive and not greater than ttl_s")
        self.ttl_s = float(ttl_s)
        self.heartbeat_s = float(heartbeat_s)
        self._now = monotonic or time.monotonic
        self._new_id = session_id_factory or (lambda: secrets.token_urlsafe(32))
        self._lock = threading.RLock()
        self._active: _Session | None = None
        self._expired: _Session | None = None
        self._generation = 0
        # Generation of an expiry transition that no expire() call has reported
        # yet.  It survives accessors that merely observe the expiry and is
        # cleared only by the expire() that reports it or by a new acquire().
        self._pending_expiry_generation: int | None = None

    # -- internals ---------------------------------------------------------
    def _expire_locked(self, now: float) -> bool:
        session = self._active
        if session is None:
            return False
        if now - session.last_seen < self.ttl_s:
            return False
        self._active = None
        self._expired = session
        self._pending_expiry_generation = session.generation
        return True

    @staticmethod
    def _matches(session: _Session | None, session_id) -> bool:
        return (session is not None and isinstance(session_id, str)
                and hmac.compare_digest(session.session_id, session_id))

    def _describe(self, session: _Session, now: float) -> dict:
        return {
            "owner": session.owner,
            "expires_in_s": max(0.0, self.ttl_s - max(0.0, now - session.last_seen)),
            "heartbeat_interval_s": self.heartbeat_s,
            "generation": session.generation,
        }

    @staticmethod
    def _validate_owner(owner) -> str:
        if not isinstance(owner, str):
            raise DomainError("invalid_request", "owner must be a string.", 422)
        text = owner.strip()
        if not text or len(text) > MAX_OWNER_LENGTH or any(ord(char) < 32 or ord(char) == 127 for char in text):
            raise DomainError("invalid_request", "owner must be 1..128 printable characters.", 422)
        return text

    def _resolve_locked(self, session_id) -> _Session:
        """Return the active session for ``session_id`` or raise the right error."""
        if not isinstance(session_id, str) or not session_id:
            raise DomainError("missing_control_session", "A control session is required.", 409)
        if self._matches(self._active, session_id):
            return self._active
        if self._active is not None:
            raise DomainError("session_conflict",
                              "Another control session is active; it must be released before this one is used.", 409)
        if self._matches(self._expired, session_id):
            raise DomainError("session_expired",
                              "The control session expired and is never resumed silently; release it and acquire a "
                              "new session after verifying no action is active.", 409)
        raise DomainError("missing_control_session", "Unknown control session.", 409)

    # -- public API --------------------------------------------------------
    def acquire(self, owner: str, *, drained: bool = False) -> dict:
        """Create the single control session.

        ``drained=True`` is the caller's assertion that the root verified no
        action is active; it is required before replacing an expired session.
        """
        owner = self._validate_owner(owner)
        with self._lock:
            now = self._now()
            self._expire_locked(now)
            if self._active is not None:
                raise DomainError("session_conflict",
                                  f"A control session owned by {self._active.owner!r} is active; "
                                  "heartbeat or release it before acquiring.", 409)
            if self._expired is not None and not drained:
                raise DomainError("session_expired",
                                  "The previous control session expired; verify no action is active and release it, "
                                  "or acquire with drained=True.", 409)
            self._expired = None
            # A verified drain ends the previous expiry episode; a supervisor
            # must not later act on an expiry that this new session superseded.
            self._pending_expiry_generation = None
            self._generation += 1
            session = _Session(session_id=self._new_id(), owner=owner, created_at=now,
                               last_seen=now, generation=self._generation)
            self._active = session
            return {"session_id": session.session_id, "owner": session.owner,
                    "status": "active", **self._describe(session, now)}

    def heartbeat(self, session_id) -> dict:
        """Refresh an active session; never revives an expired one."""
        with self._lock:
            now = self._now()
            self._expire_locked(now)
            session = self._resolve_locked(session_id)
            session.last_seen = now
            return {"session_id": session.session_id, "status": "active", **self._describe(session, now)}

    def require(self, session_id) -> None:
        """Validate a session for one write; does not refresh the lease."""
        with self._lock:
            now = self._now()
            self._expire_locked(now)
            self._resolve_locked(session_id)
        return None

    def release(self, session_id) -> dict:
        """Release the active session, or clear an already expired one after a drain."""
        with self._lock:
            now = self._now()
            self._expire_locked(now)
            if not isinstance(session_id, str) or not session_id:
                raise DomainError("missing_control_session", "A control session is required.", 409)
            if self._matches(self._active, session_id):
                session = self._active
                self._active = None
                return {"status": "released", "session_id": session.session_id, "owner": session.owner}
            if self._active is not None:
                raise DomainError("session_conflict",
                                  "Another control session is active; it must be released first.", 409)
            if self._matches(self._expired, session_id):
                session = self._expired
                self._expired = None
                return {"status": "expired_released", "session_id": session.session_id, "owner": session.owner}
            raise DomainError("missing_control_session", "Unknown control session.", 409)

    def expire(self) -> bool:
        """Report and consume the pending expiry notification.

        Returns ``True`` exactly once for each expiry transition, regardless of
        which accessor (``require``/``heartbeat``/``public``/``release``) noticed
        the transition first, and ``False`` when there is nothing to report.
        """
        with self._lock:
            self._expire_locked(self._now())
            if self._pending_expiry_generation is None:
                return False
            self._pending_expiry_generation = None
            return True

    def public(self) -> dict:
        """Model-safe view: never exposes the session id capability."""
        with self._lock:
            now = self._now()
            self._expire_locked(now)
            if self._active is not None:
                return {"state": "active", "active": True, "expired": False,
                        "owner": self._active.owner, "generation": self._active.generation,
                        "expires_in_s": self._describe(self._active, now)["expires_in_s"],
                        "heartbeat_interval_s": self.heartbeat_s, "ttl_s": self.ttl_s}
            if self._expired is not None:
                return {"state": "expired", "active": False, "expired": True,
                        "owner": self._expired.owner, "generation": self._expired.generation,
                        "expires_in_s": 0.0,
                        "expired_for_s": max(0.0, now - (self._expired.last_seen + self.ttl_s)),
                        "heartbeat_interval_s": self.heartbeat_s, "ttl_s": self.ttl_s}
            return {"state": "inactive", "active": False, "expired": False,
                    "owner": None, "generation": self._generation,
                    "expires_in_s": None, "heartbeat_interval_s": self.heartbeat_s, "ttl_s": self.ttl_s}


__all__ = ["ControlSessions", "HEARTBEAT_INTERVAL_S", "SESSION_TTL_S"]
