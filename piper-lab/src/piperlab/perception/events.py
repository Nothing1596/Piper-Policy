"""Event classification: separate the world from our beliefs about it.

Three epistemic kinds are kept distinct, because collapsing them is how a
pipeline starts narrating its own bookkeeping as physical reality:

``world_change``
    Observations support a change in the state of the world.
``belief_revision``
    Identity, label or estimate was corrected. The world need not have changed:
    learning that two tracks are one cup reduces the entity count without any
    cup disappearing.
``observation_gap``
    We cannot currently confirm the state. This is an absence of knowledge, not
    evidence of absence.

``tool_action`` and ``escalation`` are operational records, not world events,
and are named separately so a reader can never mistake a command for an
observation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Iterable


class EventError(ValueError):
    """An event record is malformed."""


class EventKind(str, Enum):
    """What kind of statement an event is making."""

    WORLD_CHANGE = "world_change"
    BELIEF_REVISION = "belief_revision"
    OBSERVATION_GAP = "observation_gap"
    CONFLICT = "conflict"
    TOOL_ACTION = "tool_action"
    ESCALATION = "escalation"

    def __str__(self) -> str:  # pragma: no cover - convenience only
        return self.value


#: Kinds that assert something about the world. Only these may be summarised as
#: "what happened"; the others are bookkeeping and must be labelled as such.
WORLD_ASSERTING = frozenset({EventKind.WORLD_CHANGE})


@dataclass(frozen=True)
class Event:
    """One classified statement with its evidence pointers."""

    event_id: str
    kind: EventKind
    source_s: float
    summary: str
    entity_id: str | None = None
    field_name: str | None = None
    before: object = None
    after: object = None
    evidence_sequences: tuple[int, ...] = ()
    evidence_gaps: tuple[str, ...] = ()
    late: bool = False
    revision: str | None = None
    confidence: float | None = None
    detail: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.event_id:
            raise EventError("Event.event_id is required")
        if not isinstance(self.kind, EventKind):
            raise EventError("Event.kind must be an EventKind")
        if isinstance(self.source_s, bool) or not isinstance(self.source_s, (int, float)) \
                or not math.isfinite(self.source_s):
            raise EventError("Event.source_s must be finite")
        if not self.summary:
            raise EventError("Event.summary is required")
        if self.confidence is not None and (
            not math.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0
        ):
            raise EventError("Event.confidence must be within [0, 1] when present")
        if self.kind is EventKind.WORLD_CHANGE and not self.evidence_sequences:
            # A world claim without a frame pointer is unauditable.
            raise EventError("world_change requires at least one evidence sequence")

    @property
    def asserts_world_state(self) -> bool:
        return self.kind in WORLD_ASSERTING

    def describe(self) -> dict:
        return {
            "event_id": self.event_id,
            "kind": self.kind.value,
            "asserts_world_state": self.asserts_world_state,
            "source_s": self.source_s,
            "summary": self.summary,
            "entity_id": self.entity_id,
            "field_name": self.field_name,
            "before": self.before,
            "after": self.after,
            "evidence_sequences": list(self.evidence_sequences),
            "evidence_gaps": list(self.evidence_gaps),
            "late": self.late,
            "revision": self.revision,
            "confidence": self.confidence,
            "detail": dict(self.detail),
        }


def classify(
    *,
    previous_entities: Iterable,
    current_entities: Iterable,
    source_s: float,
    capture_gaps: Iterable = (),
    event_prefix: str = "evt",
    start_index: int = 0,
) -> list[Event]:
    """Diff two entity snapshots into classified events.

    Rules that matter:

    - An entity that becomes occluded or lost yields ``observation_gap``, never
      a disappearance. We do not get to claim an object vanished because our
      tracker lost it.
    - An entity that disappears while *visible* in both snapshots is the only
      case that yields a disappearance ``world_change``.
    - A drop in identity count with no visible disappearance is a
      ``belief_revision`` (identity merge), not objects disappearing.
    - Label changes are ``belief_revision``: the classification changed, the
      object did not.
    """
    before = {entity.entity_id: entity for entity in previous_entities}
    after = {entity.entity_id: entity for entity in current_entities}
    events: list[Event] = []
    counter = start_index

    def next_id() -> str:
        nonlocal counter
        counter += 1
        return f"{event_prefix}-{counter:05d}"

    gaps = tuple(capture_gaps)
    gap_ids = tuple(getattr(gap, "reason", str(gap)) for gap in gaps)
    if gaps:
        events.append(
            Event(
                event_id=next_id(),
                kind=EventKind.OBSERVATION_GAP,
                source_s=source_s,
                summary=(
                    "capture gap recorded; state during this interval is unknown, "
                    "which is not the same as unchanged"
                ),
                evidence_gaps=gap_ids,
                detail={"capture_gaps": [gap.describe() if hasattr(gap, "describe") else str(gap) for gap in gaps]},
            )
        )

    for entity_id, entity in after.items():
        prior = before.get(entity_id)
        if prior is None:
            events.append(
                Event(
                    event_id=next_id(),
                    kind=EventKind.WORLD_CHANGE,
                    source_s=entity.last_source_s,
                    summary=f"{entity_id} first observed as {entity.label}",
                    entity_id=entity_id,
                    field_name="presence",
                    before=None,
                    after="present",
                    evidence_sequences=entity.evidence_sequences,
                    confidence=entity.score if entity.score else None,
                )
            )
            continue

        if prior.state != entity.state:
            if entity.gap and not prior.gap:
                events.append(
                    Event(
                        event_id=next_id(),
                        kind=EventKind.OBSERVATION_GAP,
                        source_s=entity.last_source_s,
                        summary=(
                            f"{entity_id} can no longer be confirmed ({entity.state}); "
                            "existence is not asserted either way"
                        ),
                        entity_id=entity_id,
                        field_name="visibility",
                        before=prior.state,
                        after=entity.state,
                        evidence_sequences=entity.evidence_sequences,
                    )
                )
                continue
            if prior.gap and entity.visible:
                events.append(
                    Event(
                        event_id=next_id(),
                        kind=EventKind.BELIEF_REVISION,
                        source_s=entity.last_source_s,
                        summary=f"{entity_id} re-acquired; identity continued, not re-created",
                        entity_id=entity_id,
                        field_name="visibility",
                        before=prior.state,
                        after=entity.state,
                        revision="reacquired",
                        evidence_sequences=entity.evidence_sequences,
                    )
                )
                continue

        if prior.label != entity.label:
            events.append(
                Event(
                    event_id=next_id(),
                    kind=EventKind.BELIEF_REVISION,
                    source_s=entity.last_source_s,
                    summary=(
                        f"{entity_id} relabelled {prior.label} -> {entity.label}; "
                        "the classification changed, not the object"
                    ),
                    entity_id=entity_id,
                    field_name="label",
                    before=prior.label,
                    after=entity.label,
                    revision="classification",
                    evidence_sequences=entity.evidence_sequences,
                )
            )

        if prior.visible and entity.visible:
            moved = _displacement(prior, entity)
            if moved is not None and moved > 0.002:
                events.append(
                    Event(
                        event_id=next_id(),
                        kind=EventKind.WORLD_CHANGE,
                        source_s=entity.last_source_s,
                        summary=f"{entity_id} moved {moved:.4f} px in the image",
                        entity_id=entity_id,
                        field_name="centroid_xy",
                        before=list(prior.centroid_xy) if prior.centroid_xy else None,
                        after=list(entity.centroid_xy) if entity.centroid_xy else None,
                        evidence_sequences=entity.evidence_sequences,
                    )
                )

    for entity_id, entity in before.items():
        if entity_id in after:
            continue
        if entity.visible:
            events.append(
                Event(
                    event_id=next_id(),
                    kind=EventKind.WORLD_CHANGE,
                    source_s=source_s,
                    summary=(
                        f"{entity_id} left the tracked set while still visible; "
                        "this is a disappearance claim and requires the evidence below"
                    ),
                    entity_id=entity_id,
                    field_name="presence",
                    before="present",
                    after="absent",
                    evidence_sequences=entity.evidence_sequences,
                    detail={"caution": "verify before treating this as an object removal"},
                )
            )
        else:
            events.append(
                Event(
                    event_id=next_id(),
                    kind=EventKind.OBSERVATION_GAP,
                    source_s=source_s,
                    summary=f"{entity_id} dropped while unconfirmed; no existence claim is made",
                    entity_id=entity_id,
                    field_name="presence",
                    before=entity.state,
                    after="untracked",
                    evidence_sequences=entity.evidence_sequences,
                )
            )

    return events


def _displacement(prior, current) -> float | None:
    if prior.centroid_xy is None or current.centroid_xy is None:
        return None
    return math.dist(prior.centroid_xy, current.centroid_xy)


def identity_merge_event(
    *,
    entity_ids: Iterable[str],
    surviving_entity_id: str,
    association_ids: Iterable[str],
    source_s: float,
    confidence: float,
    event_id: str,
    evidence_sequences: Iterable[int] = (),
) -> Event:
    """Record that several entities were found to be one object.

    The entity count drops, yet nothing left the world. Emitting a
    ``world_change`` here would be wrong, so this helper exists to make the
    correct classification the easy one.
    """
    members = tuple(entity_ids)
    if surviving_entity_id not in members:
        raise EventError("surviving_entity_id must be one of entity_ids")
    return Event(
        event_id=event_id,
        kind=EventKind.BELIEF_REVISION,
        source_s=source_s,
        summary=(
            f"identity merge: {len(members)} entities -> 1 ({surviving_entity_id}); "
            "no object disappeared, the identity belief was corrected"
        ),
        entity_id=surviving_entity_id,
        field_name="identity",
        before=list(members),
        after=[surviving_entity_id],
        revision="identity_merge",
        confidence=confidence,
        evidence_sequences=tuple(evidence_sequences),
        detail={"association_ids": list(association_ids)},
    )


@dataclass
class EventLog:
    """Append-only event log with a revision-safe append API.

    ``append`` numbers events for the caller, so ids stay unique across a run
    without the caller tracking counters.
    """

    prefix: str = "evt"
    events: list[Event] = field(default_factory=list)

    def append(self, event: Event) -> Event:
        if not isinstance(event, Event):
            raise EventError("EventLog.append expects an Event")
        expected = f"{self.prefix}-{len(self.events) + 1:05d}"
        if event.event_id != expected:
            # Renumbering keeps ids unique per run without callers holding counters.
            event = replace(event, event_id=expected)
        self.events.append(event)
        return event

    def extend(self, events: Iterable[Event]) -> list[Event]:
        return [self.append(event) for event in events]

    def classify_and_append(self, **kwargs) -> list[Event]:
        kwargs.setdefault("event_prefix", self.prefix)
        kwargs.setdefault("start_index", len(self.events))
        produced = classify(**kwargs)
        self.events.extend(produced)
        return produced

    def by_kind(self, kind: EventKind) -> list[Event]:
        return [event for event in self.events if event.kind is kind]

    def world_claims(self) -> list[Event]:
        """Only events that assert something about the world."""
        return [event for event in self.events if event.asserts_world_state]

    def describe(self) -> dict:
        counts: dict[str, int] = {}
        for event in self.events:
            counts[event.kind.value] = counts.get(event.kind.value, 0) + 1
        return {
            "count": len(self.events),
            "by_kind": counts,
            "world_claims": len(self.world_claims()),
            "events": [event.describe() for event in self.events],
        }
