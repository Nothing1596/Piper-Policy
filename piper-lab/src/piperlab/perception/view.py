"""Versioned state view: the one artefact the fast path reads.

The view answers four questions about every field it publishes:

- what value (an entity state, a robot vector),
- which observation instant it describes,
- the evidence pointers that produced it,
- until when it is valid, and if it is not, which executor-level code
  invalidated it.

Invalidation reasons come from the shared vocabulary in ``clockmap`` rather than
from home-made booleans, so "why can the fast path not use this snapshot" always
traces to a concrete condition instead of a vibe.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Iterable

from .budget import Budget, BudgetExceeded
from .claims import Claim, ClaimStore
from .clockmap import INVALIDATION_CODES, ClockStamp, is_known_invalidation
from .events import Event, EventLog


class ViewError(ValueError):
    """A state view operation is malformed."""


class Staleness(str, Enum):
    """Usability of a value at a given instant."""

    FRESH = "fresh"
    STALE = "stale"
    UNKNOWN = "unknown"
    INVALIDATED = "invalidated"

    def __str__(self) -> str:  # pragma: no cover - convenience only
        return self.value


@dataclass(frozen=True)
class FieldValue:
    """One published value with full provenance."""

    name: str
    value: object
    stamp: ClockStamp | None = None
    evidence_sequences: tuple[int, ...] = ()
    evidence_refs: tuple[dict, ...] = ()
    valid_until_source_s: float | None = None
    valid_until_monotonic_s: float | None = None
    claim_ids: tuple[str, ...] = ()
    note: str = ""

    def staleness_at(
        self,
        *,
        now_source_s: float,
        now_monotonic_s: float,
        max_age_s: float,
        skew_s: float,
    ) -> Staleness:
        if self.stamp is None and self.valid_until_source_s is None \
                and self.valid_until_monotonic_s is None:
            return Staleness.UNKNOWN
        if self.valid_until_source_s is not None and now_source_s > self.valid_until_source_s:
            return Staleness.STALE
        if self.valid_until_monotonic_s is not None and now_monotonic_s > self.valid_until_monotonic_s:
            return Staleness.STALE
        if self.stamp is not None:
            if not self.stamp.valid(
                max_age_s=max_age_s, skew_s=skew_s, now_monotonic_s=now_monotonic_s
            ):
                return Staleness.STALE
        return Staleness.FRESH

    def describe(self) -> dict:
        return {
            "name": self.name,
            "value": self.value,
            "stamp": self.stamp.describe() if self.stamp else None,
            "evidence_sequences": list(self.evidence_sequences),
            "evidence_refs": [dict(ref) for ref in self.evidence_refs],
            "valid_until_source_s": self.valid_until_source_s,
            "valid_until_monotonic_s": self.valid_until_monotonic_s,
            "claim_ids": list(self.claim_ids),
            "note": self.note,
        }


@dataclass
class StateView:
    """An immutable-by-convention snapshot, replaced per version.

    ``version`` increases monotonically. The previous view is marked
    ``superseded_by`` rather than being rewritten, so an audit can reconstruct
    what the fast path saw at any point.
    """

    version: int
    created_at_source_s: float
    created_at_monotonic_s: float
    fields: dict[str, FieldValue] = field(default_factory=dict)
    entity_ids: tuple[str, ...] = ()
    invalidated: bool = False
    invalidation_code: str | None = None
    invalidation_detail: dict = field(default_factory=dict)
    invalidated_at_monotonic_s: float | None = None
    superseded_by: int | None = None
    claim_ids: tuple[str, ...] = ()
    event_ids: tuple[str, ...] = ()
    budget: Budget = field(default_factory=Budget)

    def __post_init__(self) -> None:
        if isinstance(self.version, bool) or not isinstance(self.version, int) or self.version < 1:
            raise ViewError("StateView.version must be a positive int")
        for name in ("created_at_source_s", "created_at_monotonic_s"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ViewError(f"StateView.{name} must be finite")
        if self.invalidation_code is not None and not is_known_invalidation(self.invalidation_code):
            raise ViewError(
                f"unknown invalidation code {self.invalidation_code!r}; "
                f"expected one of {', '.join(INVALIDATION_CODES)}"
            )

    # -- construction ------------------------------------------------------
    def put(self, field_value: FieldValue) -> "StateView":
        if not isinstance(field_value, FieldValue):
            raise ViewError("put expects a FieldValue")
        self.fields[field_value.name] = field_value
        return self

    def put_entity_summary(
        self,
        name: str,
        value,
        *,
        entity_ids: Iterable[str],
        evidence_sequences: Iterable[int],
        stamp: ClockStamp | None = None,
        valid_until_monotonic_s: float | None = None,
        note: str = "",
    ) -> "StateView":
        self.entity_ids = tuple(entity_ids)
        return self.put(
            FieldValue(
                name=name,
                value=value,
                stamp=stamp,
                evidence_sequences=tuple(evidence_sequences),
                valid_until_monotonic_s=valid_until_monotonic_s,
                note=note,
            )
        )

    def invalidate(
        self,
        code: str,
        *,
        now_monotonic_s: float,
        detail: dict | None = None,
    ) -> "StateView":
        """Mark the whole view unusable with an executor-level reason code."""
        if not is_known_invalidation(code):
            raise ViewError(
                f"unknown invalidation code {code!r}; expected one of "
                f"{', '.join(INVALIDATION_CODES)}"
            )
        self.invalidated = True
        self.invalidation_code = code
        self.invalidation_detail = dict(detail or {})
        self.invalidated_at_monotonic_s = float(now_monotonic_s)
        return self

    # -- queries -----------------------------------------------------------
    def staleness(
        self,
        name: str,
        *,
        now_source_s: float,
        now_monotonic_s: float,
        max_age_s: float | None = None,
        skew_s: float = 0.05,
    ) -> Staleness:
        if self.invalidated:
            return Staleness.INVALIDATED
        field_value = self.fields.get(name)
        if field_value is None:
            return Staleness.UNKNOWN
        window = max_age_s if max_age_s is not None else 0.2
        return field_value.staleness_at(
            now_source_s=now_source_s,
            now_monotonic_s=now_monotonic_s,
            max_age_s=window,
            skew_s=skew_s,
        )

    def usable(
        self,
        names: Iterable[str],
        *,
        now_source_s: float,
        now_monotonic_s: float,
        max_age_s: float | None = None,
        skew_s: float = 0.05,
    ) -> bool:
        """Whether the named fields may be relied on for a new decision."""
        return all(
            self.staleness(
                name,
                now_source_s=now_source_s,
                now_monotonic_s=now_monotonic_s,
                max_age_s=max_age_s,
                skew_s=skew_s,
            )
            is Staleness.FRESH
            for name in names
        )

    def require_fresh(
        self,
        names: Iterable[str],
        *,
        now_source_s: float,
        now_monotonic_s: float,
        max_age_s: float | None = None,
        skew_s: float = 0.05,
    ) -> None:
        """Raise when the fast path must not act on this view."""
        unusable = {
            name: self.staleness(
                name,
                now_source_s=now_source_s,
                now_monotonic_s=now_monotonic_s,
                max_age_s=max_age_s,
                skew_s=skew_s,
            ).value
            for name in names
            if self.staleness(
                name,
                now_source_s=now_source_s,
                now_monotonic_s=now_monotonic_s,
                max_age_s=max_age_s,
                skew_s=skew_s,
            )
            is not Staleness.FRESH
        }
        if unusable:
            reason = self.invalidation_code or "stale_or_missing"
            raise ViewError(
                f"state view v{self.version} is not usable ({reason}): "
                + ", ".join(f"{name}={state}" for name, state in sorted(unusable.items()))
            )

    def field(self, name: str) -> FieldValue | None:
        return self.fields.get(name)

    def describe(self) -> dict:
        return {
            "version": self.version,
            "created_at_source_s": self.created_at_source_s,
            "created_at_monotonic_s": self.created_at_monotonic_s,
            "invalidated": self.invalidated,
            "invalidation_code": self.invalidation_code,
            "invalidation_detail": dict(self.invalidation_detail),
            "superseded_by": self.superseded_by,
            "entity_ids": list(self.entity_ids),
            "claim_ids": list(self.claim_ids),
            "event_ids": list(self.event_ids),
            "fields": {
                name: field_value.describe() for name, field_value in sorted(self.fields.items())
            },
        }


class StateStore:
    """Owns version numbering, supersession and the audit trail of views."""

    def __init__(self, *, budget: Budget | None = None, start_version: int = 0) -> None:
        self.budget = budget or Budget()
        self._version = start_version
        self._views: list[StateView] = []
        self._current: StateView | None = None

    @property
    def current(self) -> StateView | None:
        return self._current

    @property
    def version(self) -> int:
        return self._version

    def publish(
        self,
        *,
        now_source_s: float,
        now_monotonic_s: float,
        fields: dict[str, FieldValue] | None = None,
        entity_ids: Iterable[str] = (),
        claim_ids: Iterable[str] = (),
        event_ids: Iterable[str] = (),
    ) -> StateView:
        """Create the next version and mark the previous one superseded."""
        if self._current is not None:
            self._current.superseded_by = self._version + 1
        self._version += 1
        view = StateView(
            version=self._version,
            created_at_source_s=float(now_source_s),
            created_at_monotonic_s=float(now_monotonic_s),
            fields=dict(fields or {}),
            entity_ids=tuple(entity_ids),
            claim_ids=tuple(claim_ids),
            event_ids=tuple(event_ids),
            budget=self.budget,
        )
        self._views.append(view)
        self._current = view
        return view

    def invalidate_current(self, code: str, *, now_monotonic_s: float, detail: dict | None = None) -> StateView:
        if self._current is None:
            raise ViewError("no current view to invalidate")
        return self._current.invalidate(code, now_monotonic_s=now_monotonic_s, detail=detail)

    def history(self) -> tuple[StateView, ...]:
        return tuple(self._views)

    def replay(self, at_version: int) -> StateView:
        """Retrieve exactly what the fast path saw at a given version."""
        for view in self._views:
            if view.version == at_version:
                return view
        raise ViewError(f"no state view with version {at_version}")

    def as_of_source(self, source_s: float) -> StateView | None:
        """Newest view created at or before ``source_s``."""
        candidate = None
        for view in self._views:
            if view.created_at_source_s <= source_s:
                candidate = view
            else:
                break
        return candidate

    def snapshot(self) -> dict:
        return {
            "version": self._version,
            "current": self._current.describe() if self._current else None,
            "history_versions": [view.version for view in self._views],
        }

    def check_budget(self) -> None:
        """Refuse to grow history without bound."""
        self.budget.check_count("state_view_history", len(self._views), 10_000)


def build_view(
    *,
    store: StateStore,
    entity_summary: dict,
    entity_ids: Iterable[str],
    evidence_sequences: Iterable[int],
    robot_fields: dict[str, FieldValue] | None = None,
    now_source_s: float,
    now_monotonic_s: float,
    claims: ClaimStore | None = None,
    events: EventLog | None = None,
    valid_until_monotonic_s: float | None = None,
) -> StateView:
    """Convenience assembly used by the pipeline and by tests."""
    fields: dict[str, FieldValue] = dict(robot_fields or {})
    fields["entities"] = FieldValue(
        name="entities",
        value=entity_summary,
        evidence_sequences=tuple(evidence_sequences),
        valid_until_monotonic_s=valid_until_monotonic_s,
        note=(
            "entity summary carries its own staleness; entity internal states "
            "'occluded'/'lost' mean unconfirmed, not absent"
        ),
    )
    view = store.publish(
        now_source_s=now_source_s,
        now_monotonic_s=now_monotonic_s,
        fields=fields,
        entity_ids=entity_ids,
        claim_ids=tuple(claim.claim_id for claim in (claims.claims if claims else ())),
        event_ids=tuple(event.event_id for event in (events.events if events else ())),
    )
    store.check_budget()
    return view
