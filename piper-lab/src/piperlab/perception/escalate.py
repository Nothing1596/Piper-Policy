"""Escalation to the slow model, and the degraded modes when it is not there.

The fast path must keep running while the slow model is thinking, blocked, or
absent. That only works if the degraded behaviour is *predefined* rather than
improvised, so the tiers below are configuration, not fallback logic scattered
through the loop.

Escalation also has to be rate-limited per subject. Without a de-duplication key
and a minimum gap, one uncertain object produces an escalation storm, and the
slow model's latency becomes the pipeline's latency.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Protocol, runtime_checkable

from .budget import Budget
from .events import Event, EventKind


class EscalationError(ValueError):
    """An escalation request or policy is malformed."""


class TriggerKind(str, Enum):
    """Why the slow model is being consulted."""

    HEARTBEAT = "heartbeat"
    WORLD_CHANGE = "world_change"
    OBSERVATION_GAP = "observation_gap"
    CONFLICT = "conflict"
    STALE_VIEW = "stale_view"
    EXECUTABILITY_FLIP = "executability_flip"
    GRIPPER_TRANSITION = "gripper_transition"
    ATTENTION = "attention"

    def __str__(self) -> str:  # pragma: no cover - convenience only
        return self.value


class DegradedTier(str, Enum):
    """Predefined behaviour as the slow path falls behind.

    These are behaviour contracts, not descriptions of data age:

    - ``NOMINAL``: the fast path may issue new primitives.
    - ``HOLD``: no new primitives; keep verifying the current target.
    - ``FREEZE``: freeze the last *verified* state; no new actions at all.
    - ``OBSERVE_ONLY``: observation continues, actions are refused, and the
      operator is told.
    """

    NOMINAL = "nominal"
    HOLD = "hold"
    FREEZE = "freeze"
    OBSERVE_ONLY = "observe_only"

    def __str__(self) -> str:  # pragma: no cover - convenience only
        return self.value


@dataclass(frozen=True)
class DegradedPolicy:
    """Thresholds for the tiers, in seconds of slow-path staleness."""

    hold_after_s: float = 0.15
    freeze_after_s: float = 1.0
    observe_only_after_s: float = 5.0

    def __post_init__(self) -> None:
        for name in ("hold_after_s", "freeze_after_s", "observe_only_after_s"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise EscalationError(f"DegradedPolicy.{name} must be finite")
            if value <= 0:
                raise EscalationError(f"DegradedPolicy.{name} must be positive")
        if not (self.hold_after_s <= self.freeze_after_s <= self.observe_only_after_s):
            raise EscalationError(
                "DegradedPolicy thresholds must be non-decreasing: "
                "hold <= freeze <= observe_only"
            )

    def tier_for(self, slow_staleness_s: float) -> DegradedTier:
        if slow_staleness_s < self.hold_after_s:
            return DegradedTier.NOMINAL
        if slow_staleness_s < self.freeze_after_s:
            return DegradedTier.HOLD
        if slow_staleness_s < self.observe_only_after_s:
            return DegradedTier.FREEZE
        return DegradedTier.OBSERVE_ONLY

    def may_issue_actions(self, tier: DegradedTier) -> bool:
        return tier is DegradedTier.NOMINAL

    def describe(self) -> dict:
        return {
            "hold_after_s": self.hold_after_s,
            "freeze_after_s": self.freeze_after_s,
            "observe_only_after_s": self.observe_only_after_s,
            "tiers": {
                tier.value: self.may_issue_actions(tier)
                for tier in DegradedTier
            },
        }


@dataclass(frozen=True)
class EscalationRequest:
    """A bounded request for the slow model's interpretation."""

    request_id: str
    trigger: TriggerKind
    subject: str
    requested_at_source_s: float
    summary: str
    evidence_sequences: tuple[int, ...] = ()
    state_version: int | None = None
    dedup_key: str = ""
    expires_at_source_s: float | None = None
    detail: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.request_id:
            raise EscalationError("EscalationRequest.request_id is required")
        if not isinstance(self.trigger, TriggerKind):
            raise EscalationError("EscalationRequest.trigger must be a TriggerKind")
        if not self.subject:
            raise EscalationError("EscalationRequest.subject is required")
        if not self.summary:
            raise EscalationError("EscalationRequest.summary is required")
        if isinstance(self.requested_at_source_s, bool) or not isinstance(
            self.requested_at_source_s, (int, float)
        ) or not math.isfinite(self.requested_at_source_s):
            raise EscalationError("EscalationRequest.requested_at_source_s must be finite")

    def describe(self) -> dict:
        return {
            "request_id": self.request_id,
            "trigger": self.trigger.value,
            "subject": self.subject,
            "requested_at_source_s": self.requested_at_source_s,
            "summary": self.summary,
            "evidence_sequences": list(self.evidence_sequences),
            "state_version": self.state_version,
            "dedup_key": self.dedup_key,
            "expires_at_source_s": self.expires_at_source_s,
            "detail": dict(self.detail),
        }


@runtime_checkable
class SlowModelPort(Protocol):
    """The only channel to the slow model. Deliberately tiny.

    A slow model may propose interpretation, attention and goals. It may not
    write state, and it may not authorise motion.
    """

    name: str

    def available(self) -> bool:
        ...

    def submit(self, request: EscalationRequest) -> dict:
        ...


@dataclass
class FakeSlowModel:
    """Deterministic stand-in used by tests and by the replay harness."""

    name: str = "fake_slow_model"
    is_available: bool = True
    submissions: list = field(default_factory=list)
    response: dict = field(
        default_factory=lambda: {
            "verdict": "unknown",
            "reason": "fake slow model returns no interpretation",
        }
    )

    def available(self) -> bool:
        return self.is_available

    def submit(self, request: EscalationRequest) -> dict:
        if not self.is_available:
            raise EscalationError("slow model is unavailable; degradation policy applies")
        self.submissions.append(request)
        return {"request_id": request.request_id, **self.response}


@dataclass
class Escalator:
    """Decides when the slow model is consulted, and enforces the tier policy."""

    budget: Budget = field(default_factory=Budget)
    policy: DegradedPolicy = field(default_factory=DegradedPolicy)
    slow_model: SlowModelPort | None = None
    _last_escalation: dict[str, float] = field(default_factory=dict)
    _backoff: dict[str, int] = field(default_factory=dict)
    _counter: int = 0

    # -- triggering --------------------------------------------------------
    def consider(
        self,
        *,
        subject: str,
        trigger: TriggerKind,
        now_source_s: float,
        summary: str,
        evidence_sequences: Iterable[int] = (),
        state_version: int | None = None,
        dedup_key: str = "",
        ttl_s: float | None = None,
        detail: dict | None = None,
    ) -> EscalationRequest | None:
        """Return a request when one is warranted, else ``None``.

        Suppression is reported by the caller through the returned ``None`` plus
        the counters in :meth:`describe`; a suppressed escalation is not an
        error, it is the rate limiter working.
        """
        if isinstance(now_source_s, bool) or not isinstance(now_source_s, (int, float)) \
                or not math.isfinite(now_source_s):
            raise EscalationError("now_source_s must be finite")
        key = dedup_key or f"{subject}:{trigger.value}"
        minimum_gap = self.budget.escalation_min_gap_s * (2 ** min(self._backoff.get(key, 0), 4))
        last = self._last_escalation.get(key)
        if last is not None and now_source_s - last < minimum_gap:
            return None
        self._counter += 1
        self._last_escalation[key] = now_source_s
        if ttl_s is not None and (not math.isfinite(ttl_s) or ttl_s <= 0):
            raise EscalationError("ttl_s must be positive and finite")
        return EscalationRequest(
            request_id=f"esc-{self._counter:05d}",
            trigger=trigger,
            subject=subject,
            requested_at_source_s=float(now_source_s),
            summary=summary,
            evidence_sequences=tuple(evidence_sequences),
            state_version=state_version,
            dedup_key=key,
            expires_at_source_s=(None if ttl_s is None else float(now_source_s) + ttl_s),
            detail=dict(detail or {}),
        )

    def heartbeat_due(self, *, now_source_s: float, last_escalation_s: float | None = None) -> bool:
        """A floor that keeps the loop from going blind when the scene is static."""
        reference = last_escalation_s
        if reference is None and self._last_escalation:
            reference = max(self._last_escalation.values())
        if reference is None:
            return True
        return now_source_s - reference >= self.budget.heartbeat_s

    def note_backoff(self, key: str) -> int:
        """Widen the gap after a failed or unhelpful escalation."""
        self._backoff[key] = min(self._backoff.get(key, 0) + 1, 4)
        return self._backoff[key]

    def clear_backoff(self, key: str) -> None:
        self._backoff.pop(key, None)

    # -- degraded mode -----------------------------------------------------
    def tier(self, *, now_monotonic_s: float, last_slow_response_monotonic_s: float | None) -> DegradedTier:
        if last_slow_response_monotonic_s is None:
            # No slow response has ever arrived: treat as the most cautious tier
            # rather than assuming nominal.
            return DegradedTier.OBSERVE_ONLY
        staleness = now_monotonic_s - last_slow_response_monotonic_s
        if staleness < 0:
            raise EscalationError("slow response timestamp is in the future")
        return self.policy.tier_for(staleness)

    def may_issue_actions(self, *, now_monotonic_s: float, last_slow_response_monotonic_s: float | None) -> bool:
        """Whether the fast path may issue new primitives right now."""
        tier = self.tier(
            now_monotonic_s=now_monotonic_s,
            last_slow_response_monotonic_s=last_slow_response_monotonic_s,
        )
        return self.policy.may_issue_actions(tier)

    def describe(self) -> dict:
        return {
            "policy": self.policy.describe(),
            "last_escalation": dict(self._last_escalation),
            "backoff": dict(self._backoff),
            "issued": self._counter,
            "slow_model": getattr(self.slow_model, "name", None),
            "slow_model_available": bool(self.slow_model and self.slow_model.available()),
        }


def escalation_event(request: EscalationRequest, *, event_id: str) -> Event:
    """Render an escalation as an operational event, never a world claim."""
    return Event(
        event_id=event_id,
        kind=EventKind.ESCALATION,
        source_s=request.requested_at_source_s,
        summary=f"escalated to slow model: {request.summary}",
        entity_id=request.subject if request.subject.startswith("ent-") else None,
        evidence_sequences=request.evidence_sequences,
        detail={"trigger": request.trigger.value, "request": request.describe()},
    )


def conflict_event(
    *,
    subject: str,
    source_s: float,
    slow_claim: str,
    slow_verdict: str,
    measured: str,
    event_id: str,
    evidence_sequences: Iterable[int] = (),
) -> Event:
    """Record that a model assertion disagrees with measurement.

    The measured side wins, the model claim stays ``unknown``, and the
    disagreement is an event so it cannot be quietly dropped.
    """
    return Event(
        event_id=event_id,
        kind=EventKind.CONFLICT,
        source_s=source_s,
        summary=(
            f"slow model asserted {slow_claim}={slow_verdict} but measurement says {measured}; "
            "the claim is not upgraded to supported"
        ),
        entity_id=subject,
        field_name=slow_claim,
        before=slow_verdict,
        after="unknown",
        revision="conflict_with_measurement",
        evidence_sequences=tuple(evidence_sequences),
        detail={"measured": measured, "resolution": "measurement_prevails"},
    )
