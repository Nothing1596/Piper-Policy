"""Scorer output to *selection* to *report*.

Three responsibilities that were previously collapsed into one "cheap filter"
step are separated here:

- a **scorer** gives a number,
- the **selector** decides what is forwarded *this time*,
- the **SelectionReport** explains that decision, including everything that was
  left out and why.

The distinction between the three drop reasons is the point of this module. A
scorer must never be able to delete raw data, and a consumer must never have to
guess whether an absent item was never studied or was studied and rejected.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum

from .budget import Budget, BudgetExceeded
from .detect import Detection


class SelectionError(ValueError):
    """A selection request is malformed."""


class DropReason(str, Enum):
    """Why a candidate is not part of this selection."""

    BELOW_THRESHOLD = "below_threshold"
    """Scored, but under the score floor for this policy."""

    OVER_WINDOW_BUDGET = "over_window_budget"
    """Per-source window budget was already full. Exists; simply not forwarded."""

    OVER_TOTAL_BUDGET = "over_total_budget"
    """Total candidate budget was already full."""

    NOT_ANALYSABLE = "not_analysable"
    """The candidate carries no usable geometry (degenerate bbox)."""

    def __str__(self) -> str:  # pragma: no cover - convenience only
        return self.value


#: Drop reasons that mean "the data exists, we chose not to forward it".
#: These are recoverable by re-running the selector; the others are not.
RECOVERABLE_DROPS = frozenset(
    {DropReason.OVER_WINDOW_BUDGET, DropReason.OVER_TOTAL_BUDGET, DropReason.BELOW_THRESHOLD}
)


@dataclass(frozen=True)
class SelectionPolicy:
    """Policy for one selection round.

    ``min_score`` is an admission floor for *forwarding*, not a truth claim:
    a candidate under it is recorded as ``below_threshold`` and remains
    re-selectable.
    """

    min_score: float = 0.0
    max_per_source: int | None = None
    max_total: int | None = None
    scorers: tuple[str, ...] | None = None
    prefer_distinct_hints: bool = True
    dedupe_radius_px: float = 8.0

    def __post_init__(self) -> None:
        if isinstance(self.min_score, bool) or not isinstance(self.min_score, (int, float)) \
                or not math.isfinite(self.min_score):
            raise SelectionError("min_score must be finite")
        for name in ("max_per_source", "max_total"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value <= 0):
                raise SelectionError(f"{name} must be a positive int or None")
        if self.scorers is not None and not all(isinstance(name, str) and name for name in self.scorers):
            raise SelectionError("scorers must be non-empty names")
        if isinstance(self.dedupe_radius_px, bool) or not isinstance(self.dedupe_radius_px, (int, float)) \
                or not math.isfinite(self.dedupe_radius_px) or self.dedupe_radius_px < 0:
            raise SelectionError("dedupe_radius_px must be a finite non-negative number")


@dataclass(frozen=True)
class DroppedCandidate:
    """One candidate that did not make the selection, with its reason."""

    source_id: str
    sequence: int
    source_s: float
    scorer: str
    label: str
    score: float
    reason: DropReason
    detail: dict = field(default_factory=dict)

    @property
    def recoverable(self) -> bool:
        return self.reason in RECOVERABLE_DROPS

    def describe(self) -> dict:
        return {
            "source_id": self.source_id,
            "sequence": self.sequence,
            "source_s": self.source_s,
            "scorer": self.scorer,
            "label": self.label,
            "score": self.score,
            "reason": self.reason.value,
            "recoverable": self.recoverable,
            "candidate_exists": True,
            "detail": dict(self.detail),
        }


@dataclass(frozen=True)
class SelectionReport:
    """What was selected, what was not, and why.

    ``dropped`` never means "did not exist" — those candidates demonstrably
    existed and are described individually. Absence of data is a different
    question, answered by the buffer's capture gaps, not by this report.
    """

    selected: tuple[Detection, ...]
    dropped: tuple[DroppedCandidate, ...]
    policy: SelectionPolicy
    considered: int
    source_summary: dict = field(default_factory=dict)
    notes: tuple[str, ...] = ()
    #: Candidates a scorer capped away before selection ever saw them.
    #: ``{scorer: hidden_count}``. Recorded here rather than only on surviving
    #: detections, because a scorer limit is invisible when it hides everything.
    truncated_by_scorer: dict = field(default_factory=dict)

    @property
    def scorer_capped_total(self) -> int:
        return sum(int(value) for value in self.truncated_by_scorer.values())

    @property
    def selected_count(self) -> int:
        return len(self.selected)

    @property
    def dropped_count(self) -> int:
        return len(self.dropped)

    def dropped_by_reason(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for item in self.dropped:
            counts[item.reason.value] = counts.get(item.reason.value, 0) + 1
        return counts

    def recoverable_drops(self) -> tuple[DroppedCandidate, ...]:
        return tuple(item for item in self.dropped if item.recoverable)

    def describe(self) -> dict:
        return {
            "considered": self.considered,
            "selected_count": self.selected_count,
            "dropped_count": self.dropped_count,
            "dropped_by_reason": self.dropped_by_reason(),
            "selected": [item.describe() for item in self.selected],
            "dropped": [item.describe() for item in self.dropped],
            "source_summary": dict(self.source_summary),
            "policy": {
                "min_score": self.policy.min_score,
                "max_per_source": self.policy.max_per_source,
                "max_total": self.policy.max_total,
                "scorers": list(self.policy.scorers) if self.policy.scorers else None,
            },
            "notes": list(self.notes),
            "truncated_by_scorer": dict(self.truncated_by_scorer),
            "scorer_capped_total": self.scorer_capped_total,
        }

    def text_summary(self) -> str:
        """Compact human-readable line for logs and model context."""
        reasons = ", ".join(f"{k}={v}" for k, v in sorted(self.dropped_by_reason().items()))
        base = f"selected {self.selected_count}/{self.considered}"
        return base + (f" (dropped: {reasons})" if reasons else "")


def select_detections(
    detections: list[Detection],
    *,
    policy: SelectionPolicy | None = None,
    budget: Budget | None = None,
    window_key=None,
) -> SelectionReport:
    """Choose which candidates to forward, recording every exclusion.

    ``window_key`` is an optional callable mapping a detection to a window
    identity (default: the source id) so per-window budgets can be enforced.
    """
    active_policy = policy or SelectionPolicy()
    active_budget = budget or Budget()
    max_per_source = active_policy.max_per_source or active_budget.max_frames_per_source_per_window
    max_total = active_policy.max_total or active_budget.max_candidates_total
    key_of = window_key or (lambda detection: detection.source_id)

    if active_policy.scorers is not None:
        allowed = set(active_policy.scorers)
        unknown = {detection.scorer for detection in detections} - allowed
        if unknown:
            # An explicit scorer list is a promise about what was consulted.
            raise SelectionError(
                "detections from scorers outside the policy were supplied: "
                + ", ".join(sorted(unknown))
            )

    ordered = sorted(detections, key=lambda d: (-d.score, d.source_id, d.sequence, d.scorer, d.label))
    selected: list[Detection] = []
    dropped: list[DroppedCandidate] = []
    per_window: dict[object, int] = {}
    accepted: list[Detection] = []

    for detection in ordered:
        window = key_of(detection)
        reason: DropReason | None = None
        x0, y0, x1, y1 = detection.bbox_xyxy
        if x1 <= x0 or y1 <= y0:
            reason = DropReason.NOT_ANALYSABLE
        elif detection.score < active_policy.min_score:
            reason = DropReason.BELOW_THRESHOLD
        elif active_policy.prefer_distinct_hints and _duplicate_of(
            detection, accepted, active_policy.dedupe_radius_px
        ):
            # Only a spatially coincident detection of the same hint counts as a
            # duplicate. Two red blocks in one frame are two objects, not one.
            reason = DropReason.OVER_WINDOW_BUDGET
        elif per_window.get(window, 0) >= max_per_source:
            reason = DropReason.OVER_WINDOW_BUDGET
        elif len(selected) >= max_total:
            reason = DropReason.OVER_TOTAL_BUDGET

        if reason is None:
            selected.append(detection)
            accepted.append(detection)
            per_window[window] = per_window.get(window, 0) + 1
            continue

        dropped.append(
            DroppedCandidate(
                source_id=detection.source_id,
                sequence=detection.sequence,
                source_s=detection.source_s,
                scorer=detection.scorer,
                label=detection.label,
                score=detection.score,
                reason=reason,
                detail={"window": str(window)} if reason in (
                    DropReason.OVER_WINDOW_BUDGET, DropReason.OVER_TOTAL_BUDGET
                ) else {},
            )
        )

    summary: dict[str, dict] = {}
    for detection in detections:
        entry = summary.setdefault(
            detection.source_id, {"considered": 0, "selected": 0, "dropped": 0}
        )
        entry["considered"] += 1
    for detection in selected:
        summary[detection.source_id]["selected"] += 1
    for item in dropped:
        summary[item.source_id]["dropped"] += 1

    notes: list[str] = []
    if dropped:
        notes.append(
            "candidates listed under 'dropped' existed and were not forwarded; "
            "this is not evidence that they were absent"
        )
    if any(item.reason in (DropReason.OVER_WINDOW_BUDGET, DropReason.OVER_TOTAL_BUDGET)
           for item in dropped):
        notes.append(
            "budget-limited drops are recoverable: re-running the selector with a larger "
            "budget or a narrower window can still forward them"
        )

    # A scorer's own cap is information loss that happens *before* selection, so
    # it must be reported even when the surviving list is empty.
    capped: dict[str, int] = {}
    for detection in detections:
        if detection.truncated_by_scorer:
            capped[detection.scorer] = max(
                capped.get(detection.scorer, 0), int(detection.truncated_by_scorer)
            )
    if capped:
        notes.append(
            "one or more scorers capped their own output before selection saw it "
            f"({capped}); those candidates were never scored and are not listed as dropped"
        )

    return SelectionReport(
        selected=tuple(selected),
        dropped=tuple(dropped),
        policy=active_policy,
        considered=len(detections),
        source_summary=summary,
        notes=tuple(notes),
        truncated_by_scorer=capped,
    )


def _duplicate_of(detection: Detection, accepted: list[Detection], radius_px: float) -> bool:
    """True when an already-accepted detection is the same hint at the same spot.

    Spatial proximity is part of the test on purpose: deduplicating on the hint
    alone would silently merge two physically distinct objects that share a
    label.
    """
    for item in accepted:
        if item.source_id != detection.source_id or item.entity_hint != detection.entity_hint:
            continue
        dx = item.centroid_xy[0] - detection.centroid_xy[0]
        dy = item.centroid_xy[1] - detection.centroid_xy[1]
        if math.hypot(dx, dy) <= radius_px:
            return True
    return False


def enforce_budget(report: SelectionReport, budget: Budget) -> SelectionReport:
    """Refuse a report that claims more forwarded items than the budget allows.

    Selection already respects the budget, so this is an assertion against
    future changes rather than a normal path.
    """
    budget.check_count(
        "max_candidates_total", report.selected_count, budget.max_candidates_total
    )
    return report
