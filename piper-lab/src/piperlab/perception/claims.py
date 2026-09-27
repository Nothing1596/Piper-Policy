"""Claims, verdicts and verification windows.

A single ``verified = true`` flag cannot express what a robot system needs. A
claim must say *which proposition* was checked, *by what method*, *against which
instants*, and *until when* the answer holds. Two consequences follow, and both
are enforced here:

- ``unknown`` is a first-class verdict. Missing or expired evidence yields
  ``unknown``, never a lenient pass or a convenient failure.
- Verdicts are not inherited. ``gripper_width_reached`` says nothing about
  ``holding(cup)``; each predicate needs its own evidence and method.

The three levels stay separate end to end:

    what was issued  ->  what the device reported  ->  what the task achieved
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from enum import Enum

from .clockmap import ClockStamp


class ClaimError(ValueError):
    """A claim or verification request is malformed."""


class Verdict(str, Enum):
    """Three-valued outcome. ``UNKNOWN`` is a real answer, not a failure."""

    SUPPORTED = "supported"
    REFUTED = "refuted"
    UNKNOWN = "unknown"

    def __str__(self) -> str:  # pragma: no cover - convenience only
        return self.value


@dataclass(frozen=True)
class Predicate:
    """A named proposition, so verdicts can never be compared across subjects."""

    name: str

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ClaimError("Predicate.name is required")

    def __str__(self) -> str:
        return self.name


#: Stage-1 predicate vocabulary. New predicates need their own evidence method.
P_FRAME_AVAILABLE = Predicate("frame_available")
P_STATE_FRESH = Predicate("robot_state_fresh")
P_GAP_ACCOUNTED = Predicate("gap_accounted")
P_COMMAND_ACKNOWLEDGED = Predicate("command_acknowledged")
P_TARGET_REACHED = Predicate("target_reached")
P_ENTITY_OBSERVED = Predicate("entity_observed")
P_IDENTITY_SAME = Predicate("identity_same")


@dataclass(frozen=True)
class EvidenceRef:
    """A pointer to the raw material a verdict rests on."""

    source_id: str
    sequence: int
    source_s: float
    kind: str = "frame"

    def describe(self) -> dict:
        return {
            "source_id": self.source_id,
            "sequence": self.sequence,
            "source_s": self.source_s,
            "kind": self.kind,
        }


@dataclass(frozen=True)
class Claim:
    """One checked proposition with its verdict, method and validity window."""

    claim_id: str
    subject: str
    predicate: Predicate
    verdict: Verdict
    method: str
    evidence_refs: tuple[EvidenceRef, ...] = ()
    observed_at: float | None = None
    verified_at: float | None = None
    valid_until: float | None = None
    value: object = None
    confidence: float | None = None
    detail: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.claim_id, str) or not self.claim_id:
            raise ClaimError("Claim.claim_id is required")
        if not isinstance(self.subject, str) or not self.subject:
            raise ClaimError("Claim.subject is required")
        if not isinstance(self.predicate, Predicate):
            raise ClaimError("Claim.predicate must be a Predicate")
        if not isinstance(self.verdict, Verdict):
            raise ClaimError("Claim.verdict must be a Verdict")
        if not isinstance(self.method, str) or not self.method:
            raise ClaimError(
                "Claim.method is required: a verdict without a method cannot be reviewed"
            )
        for name in ("observed_at", "verified_at", "valid_until"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
            ):
                raise ClaimError(f"Claim.{name} must be finite when present")
        if self.confidence is not None and (
            not math.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0
        ):
            raise ClaimError("Claim.confidence must be within [0, 1] when present")
        if self.verdict in (Verdict.SUPPORTED, Verdict.REFUTED) and not self.evidence_refs:
            raise ClaimError(
                f"a {self.verdict.value} verdict requires at least one evidence reference"
            )

    @property
    def decided(self) -> bool:
        return self.verdict is not Verdict.UNKNOWN

    def expired_at(self, now: float) -> bool:
        if self.valid_until is None:
            return False
        return now > self.valid_until

    def holds_now(self, now: float) -> bool:
        """Whether the claim still stands. Expiry demotes to unknown, not false."""
        return self.verdict is Verdict.SUPPORTED and not self.expired_at(now)

    def describe(self) -> dict:
        return {
            "claim_id": self.claim_id,
            "subject": self.subject,
            "predicate": str(self.predicate),
            "verdict": self.verdict.value,
            "method": self.method,
            "evidence_refs": [ref.describe() for ref in self.evidence_refs],
            "observed_at": self.observed_at,
            "verified_at": self.verified_at,
            "valid_until": self.valid_until,
            "value": self.value,
            "confidence": self.confidence,
            "detail": dict(self.detail),
        }


class ClaimStore:
    """Holds claims and answers point-in-time questions about them."""

    def __init__(self, *, prefix: str = "clm") -> None:
        self.prefix = prefix
        self._claims: list[Claim] = []

    @property
    def claims(self) -> tuple[Claim, ...]:
        return tuple(self._claims)

    def next_id(self) -> str:
        return f"{self.prefix}-{len(self._claims) + 1:05d}"

    def record(self, claim: Claim) -> Claim:
        if not isinstance(claim, Claim):
            raise ClaimError("ClaimStore.record expects a Claim")
        self._claims.append(claim)
        return claim

    def unknown(
        self,
        subject: str,
        predicate: Predicate,
        *,
        method: str,
        reason: str,
        claim_id: str | None = None,
        observed_at: float | None = None,
        valid_until: float | None = None,
    ) -> Claim:
        """Record an explicitly undecided claim; the safe default."""
        return self.record(
            Claim(
                claim_id=claim_id or self.next_id(),
                subject=subject,
                predicate=predicate,
                verdict=Verdict.UNKNOWN,
                method=method,
                observed_at=observed_at,
                verified_at=observed_at,
                valid_until=valid_until,
                detail={"reason": reason},
            )
        )

    def verify_frame_available(
        self,
        *,
        subject: str,
        evidence,
        now: float,
        claim_id: str | None = None,
    ) -> Claim:
        """Turn an :class:`EvidenceResult` into a claim about evidence existence.

        Missing, evicted or gapped evidence yields ``unknown`` — never
        ``refuted``, because we did not observe the absence of the frame, only
        the absence of the record.
        """
        from .evidence import EvidenceStatus

        status = getattr(evidence, "status", None)
        refs: tuple[EvidenceRef, ...] = ()
        frame = getattr(evidence, "frame", None)
        if frame is not None:
            refs = (
                EvidenceRef(frame.source_id, frame.sequence, frame.stamp.source_s, "frame"),
            )
        if status is EvidenceStatus.OK and frame is not None:
            return self.record(
                Claim(
                    claim_id=claim_id or self.next_id(),
                    subject=subject,
                    predicate=P_FRAME_AVAILABLE,
                    verdict=Verdict.SUPPORTED,
                    method="buffer_lookup_exact_or_tolerance",
                    evidence_refs=refs,
                    observed_at=frame.stamp.source_s,
                    verified_at=now,
                    value={"exact": getattr(evidence, "exact", None),
                           "offset_s": getattr(evidence, "offset_s", None)},
                )
            )
        return self.unknown(
            subject,
            P_FRAME_AVAILABLE,
            method="buffer_lookup_exact_or_tolerance",
            reason=getattr(status, "value", str(status)),
            claim_id=claim_id,
            observed_at=getattr(evidence, "requested_source_s", None),
            valid_until=None,
        )

    def verify_state_fresh(
        self,
        *,
        subject: str,
        stamp: ClockStamp,
        now_monotonic_s: float,
        max_age_s: float,
        skew_s: float,
        claim_id: str | None = None,
    ) -> Claim:
        """Freshness is decided by age against the contract window, nothing else."""
        fresh = stamp.valid(max_age_s=max_age_s, skew_s=skew_s, now_monotonic_s=now_monotonic_s)
        if fresh:
            return self.record(
                Claim(
                    claim_id=claim_id or self.next_id(),
                    subject=subject,
                    predicate=P_STATE_FRESH,
                    verdict=Verdict.SUPPORTED,
                    method="clockstamp_window_check",
                    evidence_refs=(
                        EvidenceRef(stamp.mapping_id, 0, stamp.source_s, "stamp"),
                    ),
                    observed_at=stamp.source_s,
                    verified_at=now_monotonic_s,
                    valid_until=now_monotonic_s + max(0.0, max_age_s - stamp.age_at(now_monotonic_s)),
                    value={"age_s": stamp.age_at(now_monotonic_s),
                           "domain": stamp.domain.value,
                           "uncertainty_s": stamp.uncertainty_s},
                )
            )
        reason = (
            "stamp domain is not locally comparable"
            if not stamp.comparable
            else "age outside the contract window"
        )
        return self.unknown(
            subject,
            P_STATE_FRESH,
            method="clockstamp_window_check",
            reason=reason,
            claim_id=claim_id,
            observed_at=stamp.source_s,
        )

    def verify_command_versus_target(
        self,
        *,
        subject: str,
        request_id: str,
        acknowledged: bool,
        acknowledged_at: float | None,
        target,
        measured,
        tolerance,
        now: float,
        evidence_refs: tuple[EvidenceRef, ...] = (),
        claim_id_prefix: str | None = None,
    ) -> tuple[Claim, Claim]:
        """Record acknowledgement and arrival as two independent claims.

        ``command_acknowledged`` may be supported while ``target_reached`` is
        refuted — a device can accept a command and fail to arrive. Reporting a
        single combined verdict would erase exactly that case.
        """
        prefix = claim_id_prefix or self.next_id()
        import numpy as np

        if acknowledged:
            ack = self.record(
                Claim(
                    claim_id=f"{prefix}-ack",
                    subject=subject,
                    predicate=P_COMMAND_ACKNOWLEDGED,
                    verdict=Verdict.SUPPORTED,
                    method="executor_acknowledgement",
                    evidence_refs=evidence_refs
                    or (EvidenceRef(request_id, 0, acknowledged_at or now, "ack"),),
                    observed_at=acknowledged_at,
                    verified_at=now,
                    value={"request_id": request_id},
                    detail={
                        "note": (
                            "accepted is not completed; outcome_unknown is not unexecuted"
                        )
                    },
                )
            )
        else:
            ack = self.unknown(
                subject,
                P_COMMAND_ACKNOWLEDGED,
                method="executor_acknowledgement",
                reason="no acknowledgement was recorded",
                claim_id=f"{prefix}-ack",
                observed_at=now,
            )

        target_array = None if target is None else np.asarray(target, dtype=np.float64)
        measured_array = None if measured is None else np.asarray(measured, dtype=np.float64)
        if target_array is None or measured_array is None:
            reached = self.unknown(
                subject,
                P_TARGET_REACHED,
                method="measured_versus_target",
                reason="measured state or target is unavailable",
                claim_id=f"{prefix}-reach",
                observed_at=now,
            )
        elif not np.all(np.isfinite(target_array)) or not np.all(np.isfinite(measured_array)):
            reached = self.unknown(
                subject,
                P_TARGET_REACHED,
                method="measured_versus_target",
                reason="non-finite target or measurement",
                claim_id=f"{prefix}-reach",
                observed_at=now,
            )
        else:
            if target_array.shape != measured_array.shape:
                raise ClaimError(
                    f"target shape {target_array.shape} != measured shape {measured_array.shape}"
                )
            error = float(np.max(np.abs(target_array - measured_array)))
            refs = evidence_refs or (EvidenceRef(request_id, 0, now, "measurement"),)
            reached = self.record(
                Claim(
                    claim_id=f"{prefix}-reach",
                    subject=subject,
                    predicate=P_TARGET_REACHED,
                    verdict=Verdict.SUPPORTED if error <= tolerance else Verdict.REFUTED,
                    method="measured_versus_target",
                    evidence_refs=refs,
                    observed_at=now,
                    verified_at=now,
                    value={"max_abs_error": error, "tolerance": tolerance},
                    detail={
                        "note": (
                            "refuted here means the target was not reached at this instant; "
                            "it does not assert the command failed"
                        )
                    },
                )
            )
        return ack, reached

    # -- queries -----------------------------------------------------------
    def for_subject(self, subject: str) -> list[Claim]:
        return [claim for claim in self._claims if claim.subject == subject]

    def for_predicate(self, predicate: Predicate) -> list[Claim]:
        return [claim for claim in self._claims if claim.predicate == predicate]

    def latest(self, subject: str, predicate: Predicate) -> Claim | None:
        matches = [
            claim for claim in self._claims
            if claim.subject == subject and claim.predicate == predicate
        ]
        return matches[-1] if matches else None

    def standing(self, now: float) -> list[Claim]:
        """Claims that currently hold. Expired or unknown claims are excluded."""
        return [claim for claim in self._claims if claim.holds_now(now)]

    def expire(self, now: float) -> list[Claim]:
        """Demote expired supported claims to unknown, returning what changed."""
        demoted: list[Claim] = []
        for index, claim in enumerate(self._claims):
            if claim.verdict is Verdict.SUPPORTED and claim.expired_at(now):
                updated = replace(
                    claim,
                    verdict=Verdict.UNKNOWN,
                    detail={**claim.detail, "expired_at": now},
                )
                self._claims[index] = updated
                demoted.append(updated)
        return demoted

    def describe(self, now: float | None = None) -> dict:
        payload = {
            "count": len(self._claims),
            "claims": [claim.describe() for claim in self._claims],
            "by_verdict": {},
        }
        for claim in self._claims:
            payload["by_verdict"][claim.verdict.value] = (
                payload["by_verdict"].get(claim.verdict.value, 0) + 1
            )
        if now is not None:
            payload["standing"] = [claim.claim_id for claim in self.standing(now)]
        return payload
