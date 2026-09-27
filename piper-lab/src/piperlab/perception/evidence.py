"""Evidence retrieval with honest failure reporting.

A query for raw evidence must be able to say "the frame is gone", "the frame is
here but this branch never analysed it", or "nothing was ever captured here".
Returning a neighbouring frame instead is how a system starts lying about the
past, so an inexact match is reported with its offset and never presented as an
exact hit.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum

from .budget import Budget, BudgetExceeded
from .buffer import CaptureGap, FrameRecord, Gap, RingBuffer
from .clockmap import ClockStamp


class EvidenceStatus(str, Enum):
    """Outcome of an evidence query."""

    OK = "ok"
    """An exact or within-tolerance frame was returned."""

    NOT_SELECTED = "not_selected"
    """The frame exists; the requested analysis branch was never run on it."""

    CAPTURE_GAP = "capture_gap"
    """Nothing was ever captured for that interval. Treated as unknown downstream."""

    EVICTED = "evicted"
    """The frame existed and was removed. Irrecoverable; never substituted."""

    UNAVAILABLE = "unavailable"
    """Outside the buffer's known span, or the stamp cannot be compared locally."""

    BUDGET = "budget"
    """The request exceeded a configured budget and was refused entirely."""

    def __str__(self) -> str:  # pragma: no cover - convenience only
        return self.value


@dataclass(frozen=True)
class EvidenceResult:
    """Result of asking for the raw evidence behind a conclusion."""

    status: EvidenceStatus
    source_id: str
    requested_source_s: float
    frame: FrameRecord | None = None
    offset_s: float | None = None
    exact: bool = False
    reason: str = ""
    gap: Gap | None = None
    capture_gap: CaptureGap | None = None
    candidates: tuple[FrameRecord, ...] = ()
    detail: dict = field(default_factory=dict)

    @property
    def found(self) -> bool:
        return self.frame is not None

    @property
    def recoverable(self) -> bool:
        """True when re-running an analysis could still produce an answer."""
        return self.status is EvidenceStatus.NOT_SELECTED

    def to_claim_verdict(self) -> bool:
        """Whether this result can support a positive factual claim.

        Only an actual frame can. Every other status leaves the question open.
        """
        return self.status is EvidenceStatus.OK

    def describe(self) -> dict:
        payload = {
            "status": self.status.value,
            "source_id": self.source_id,
            "requested_source_s": self.requested_source_s,
            "exact": self.exact,
            "recoverable": self.recoverable,
            "reason": self.reason,
        }
        if self.frame is not None:
            payload["frame"] = self.frame.describe()
            payload["offset_s"] = self.offset_s
        if self.gap is not None:
            payload["gap"] = self.gap.describe()
        if self.capture_gap is not None:
            payload["capture_gap"] = self.capture_gap.describe()
        if self.candidates:
            payload["candidates"] = [
                {"sequence": item.sequence, "source_s": item.stamp.source_s,
                 "offset_s": item.stamp.source_s - self.requested_source_s}
                for item in self.candidates
            ]
        if self.detail:
            payload["detail"] = dict(self.detail)
        return payload


def resolve(
    buffer: RingBuffer,
    source_id: str,
    source_s: float,
    *,
    tolerance_s: float | None = None,
    analyzed_sequences: frozenset[int] | None = None,
    budget: Budget | None = None,
) -> EvidenceResult:
    """Resolve a request for a frame at ``source_s`` against the buffer.

    Order of explanation matters. A frame that is retained is returned (or
    reported as ``not_selected`` when the caller says that branch never ran on
    it). Only when nothing is retained do we distinguish an eviction, a capture
    gap, or a time outside the buffer's span.
    """
    if not math.isfinite(source_s):
        raise ValueError("source_s must be finite")
    active_budget = budget or buffer.budget
    if tolerance_s is None:
        tolerance_s = active_budget.evidence_window_s
    if isinstance(tolerance_s, bool) or not isinstance(tolerance_s, (int, float)) or not math.isfinite(tolerance_s):
        raise ValueError("tolerance_s must be finite")
    if tolerance_s < 0:
        raise ValueError("tolerance_s must be non-negative")

    candidates = _candidates(buffer, source_id, source_s, tolerance_s)
    if len(candidates) > active_budget.max_evidence_frames_per_query:
        raise BudgetExceeded(
            "max_evidence_frames_per_query",
            active_budget.max_evidence_frames_per_query,
            len(candidates),
            "frames",
        )

    exact = next(
        (item for item in candidates if item.stamp.source_s == source_s), None
    )
    if exact is not None:
        if analyzed_sequences is not None and exact.sequence not in analyzed_sequences:
            return EvidenceResult(
                status=EvidenceStatus.NOT_SELECTED,
                source_id=source_id,
                requested_source_s=source_s,
                frame=exact,
                offset_s=0.0,
                exact=True,
                reason=(
                    "frame retained but the requested analysis branch has not run on it; "
                    "re-running the selector over this frame is still possible"
                ),
                candidates=tuple(candidates),
            )
        return EvidenceResult(
            status=EvidenceStatus.OK,
            source_id=source_id,
            requested_source_s=source_s,
            frame=exact,
            offset_s=0.0,
            exact=True,
            candidates=tuple(candidates),
        )

    # The exact instant was asked for. If it was removed, say so: a retained
    # neighbour can be useful, but it must not be presented as this frame.
    evicted = buffer.evicted_covering(source_id, source_s)
    if evicted is not None:
        return EvidenceResult(
            status=EvidenceStatus.EVICTED,
            source_id=source_id,
            requested_source_s=source_s,
            reason=(
                f"evidence existed and was removed ({evicted.reason}); it cannot be "
                "recovered and must not be reported as absent"
            ),
            gap=evicted,
            candidates=tuple(candidates),
        )

    capture_gap = buffer.capture_gap_covering(source_id, source_s)
    if capture_gap is not None:
        return EvidenceResult(
            status=EvidenceStatus.CAPTURE_GAP,
            source_id=source_id,
            requested_source_s=source_s,
            reason=f"no data was ever captured in this interval: {capture_gap.reason}",
            capture_gap=capture_gap,
        )

    nearest = _nearest(candidates, source_s)
    if nearest is not None:
        chosen = nearest.frame
        if analyzed_sequences is not None and chosen.sequence not in analyzed_sequences:
            status = EvidenceStatus.NOT_SELECTED
            reason = "nearest retained frame exists but was never analysed by this branch"
        else:
            status = EvidenceStatus.OK
            reason = (
                "no frame at the exact source time; nearest retained frame returned "
                f"with offset {nearest.offset_s:+.6f}s (not an exact hit)"
            )
        return EvidenceResult(
            status=status,
            source_id=source_id,
            requested_source_s=source_s,
            frame=chosen,
            offset_s=nearest.offset_s,
            exact=False,
            reason=reason,
            candidates=tuple(candidates),
        )

    return EvidenceResult(
        status=EvidenceStatus.UNAVAILABLE,
        source_id=source_id,
        requested_source_s=source_s,
        reason=(
            "requested instant is outside the retained span and no gap was recorded; "
            "the observation history for this source does not cover it"
        ),
        detail={"span": _span(buffer, source_id)},
    )


def resolve_stamp(
    buffer: RingBuffer,
    stamp: ClockStamp,
    *,
    tolerance_s: float | None = None,
    analyzed_sequences: frozenset[int] | None = None,
    budget: Budget | None = None,
) -> EvidenceResult:
    """Resolve using a stamp, refusing sources that are not locally comparable."""
    source_id = getattr(stamp, "source_id", None)
    if source_id is None:
        source_id = ""
    if not stamp.comparable:
        return EvidenceResult(
            status=EvidenceStatus.UNAVAILABLE,
            source_id=source_id,
            requested_source_s=stamp.source_s,
            reason=(
                f"stamp domain {stamp.domain.value} has no established relation to local "
                "clocks; it may be logged but not used to locate local evidence"
            ),
            detail={"stamp": stamp.describe()},
        )
    return resolve(
        buffer,
        source_id,
        stamp.source_s,
        tolerance_s=tolerance_s,
        analyzed_sequences=analyzed_sequences,
        budget=budget,
    )


@dataclass(frozen=True)
class _Nearest:
    frame: FrameRecord
    offset_s: float


def _candidates(
    buffer: RingBuffer, source_id: str, source_s: float, tolerance_s: float
) -> list[FrameRecord]:
    return [
        record
        for record in buffer.records(source_id)
        if abs(record.stamp.source_s - source_s) <= tolerance_s
    ]


def _nearest(candidates: list[FrameRecord], source_s: float) -> _Nearest | None:
    if not candidates:
        return None
    best = min(candidates, key=lambda item: (abs(item.stamp.source_s - source_s), item.sequence))
    return _Nearest(frame=best, offset_s=best.stamp.source_s - source_s)


def _span(buffer: RingBuffer, source_id: str) -> dict:
    records = list(buffer.records(source_id))
    if not records:
        return {"retained": 0, "oldest_source_s": None, "newest_source_s": None}
    return {
        "retained": len(records),
        "oldest_source_s": records[0].stamp.source_s,
        "newest_source_s": records[-1].stamp.source_s,
    }
