"""Track identity, entity fusion and association.

Two identifiers are deliberately kept apart:

- a **local track id** ``(source_id, tracker_epoch, local_track_id)``, which is
  what one tracker on one stream can actually vouch for;
- an **entity**, which is a belief that some local tracks refer to the same
  physical object.

Merging tracks into an entity is a reversible *belief*, not an observation.
Which is why :meth:`EntityTracker.associate` produces a
``possible_same_entity`` record with its confidence and supporting observations
instead of irreversibly rewriting history.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable

from .budget import Budget, BudgetExceeded
from .detect import Detection


class TrackError(ValueError):
    """A tracking request or record is malformed."""


@dataclass(frozen=True)
class TrackId:
    """What one tracker on one stream can vouch for."""

    source_id: str
    tracker_epoch: int
    local_track_id: int

    def __post_init__(self) -> None:
        if not self.source_id:
            raise TrackError("TrackId.source_id is required")
        for name in ("tracker_epoch", "local_track_id"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise TrackError(f"TrackId.{name} must be a non-negative int")

    @property
    def key(self) -> str:
        return f"{self.source_id}:{self.tracker_epoch}:{self.local_track_id}"

    def describe(self) -> dict:
        return {
            "source_id": self.source_id,
            "tracker_epoch": self.tracker_epoch,
            "local_track_id": self.local_track_id,
            "key": self.key,
        }


@dataclass
class Entity:
    """A tracked object with an explicit lifecycle.

    ``state`` is the honest answer to "can we see it": losing a track is an
    ``observation_gap``, never a claim that the object ceased to exist.
    """

    entity_id: str
    label: str
    track: TrackId
    first_source_s: float
    last_source_s: float
    state: str = "new"
    hits: int = 0
    misses: int = 0
    centroid_xy: tuple[float, float] | None = None
    bbox_xyxy: tuple[float, float, float, float] | None = None
    score: float = 0.0
    first_sequence: int = 0
    last_sequence: int = 0
    evidence_sequences: tuple[int, ...] = ()
    sources: tuple[str, ...] = ()
    base_xyz_m: tuple[float, float, float] | None = None
    projection_available: bool = False
    calibration_id: str | None = None
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.entity_id:
            raise TrackError("Entity.entity_id is required")
        if self.state not in ("new", "tracked", "occluded", "lost", "reacquired"):
            raise TrackError(
                "Entity.state must be one of new/tracked/occluded/lost/reacquired"
            )
        if self.last_source_s < self.first_source_s:
            raise TrackError("Entity.last_source_s must not precede first_source_s")

    @property
    def visible(self) -> bool:
        return self.state in ("new", "tracked", "reacquired")

    @property
    def gap(self) -> bool:
        """True while the entity cannot be confirmed; downstream must stay unsure."""
        return self.state in ("occluded", "lost")

    def describe(self) -> dict:
        return {
            "entity_id": self.entity_id,
            "label": self.label,
            "track": self.track.describe(),
            "state": self.state,
            "visible": self.visible,
            "first_source_s": self.first_source_s,
            "last_source_s": self.last_source_s,
            "hits": self.hits,
            "misses": self.misses,
            "centroid_xy": list(self.centroid_xy) if self.centroid_xy else None,
            "bbox_xyxy": list(self.bbox_xyxy) if self.bbox_xyxy else None,
            "score": self.score,
            "first_sequence": self.first_sequence,
            "last_sequence": self.last_sequence,
            "evidence_sequences": list(self.evidence_sequences),
            "sources": list(self.sources),
            "base_xyz_m": list(self.base_xyz_m) if self.base_xyz_m else None,
            "projection_available": self.projection_available,
            "calibration_id": self.calibration_id,
            "meta": dict(self.meta),
        }


@dataclass(frozen=True)
class Association:
    """A reversible claim that two local tracks are the same object."""

    association_id: str
    entity_id: str
    member: TrackId
    possible_same_entity: bool
    confidence: float
    supporting_observations: tuple[int, ...]
    method: str
    recorded_at_source_s: float
    notes: str = ""

    def __post_init__(self) -> None:
        if not math.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise TrackError("Association.confidence must be within [0, 1]")

    def describe(self) -> dict:
        return {
            "association_id": self.association_id,
            "entity_id": self.entity_id,
            "member": self.member.describe(),
            "possible_same_entity": self.possible_same_entity,
            "confidence": self.confidence,
            "supporting_observations": list(self.supporting_observations),
            "method": self.method,
            "recorded_at_source_s": self.recorded_at_source_s,
            "notes": self.notes,
        }


class EntityTracker:
    """Greedy centroid tracker producing entities with explicit lifecycles.

    Deliberately simple: Stage 1 needs *correct bookkeeping* (identity,
    lifecycle, evidence) rather than an accurate multi-object tracker. A better
    tracker can replace :meth:`_match` without touching the rest of the
    pipeline.
    """

    def __init__(
        self,
        *,
        tracker_epoch: int = 0,
        match_radius_px: float = 48.0,
        confirm_hits: int = 2,
        max_misses: int = 3,
        budget: Budget | None = None,
        source_id: str = "camera",
    ) -> None:
        if not math.isfinite(match_radius_px) or match_radius_px <= 0:
            raise TrackError("match_radius_px must be positive")
        self.tracker_epoch = tracker_epoch
        self.match_radius_px = float(match_radius_px)
        self.confirm_hits = int(confirm_hits)
        self.max_misses = int(max_misses)
        self.budget = budget or Budget()
        self.source_id = source_id
        self._next_local_id = 0
        self._next_entity = 0
        self._entities: dict[str, Entity] = {}
        self._associations: list[Association] = []

    # -- lifecycle ---------------------------------------------------------
    @property
    def entities(self) -> tuple[Entity, ...]:
        return tuple(self._entities.values())

    @property
    def associations(self) -> tuple[Association, ...]:
        return tuple(self._associations)

    def get(self, entity_id: str) -> Entity | None:
        return self._entities.get(entity_id)

    def reset_epoch(self) -> int:
        """Begin a new tracker epoch; local ids restart and never alias old ones."""
        self.tracker_epoch += 1
        self._next_local_id = 0
        return self.tracker_epoch

    # -- update ------------------------------------------------------------
    def update(
        self,
        detections: Iterable[Detection],
        *,
        source_s: float,
        cycle: int | None = None,
    ) -> list[Entity]:
        """Advance tracking by one observation step.

        Returns the entities touched by this step. Entities not matched
        accumulate misses and move ``tracked -> occluded -> lost``; they are
        never deleted outright, because deletion would look like the object
        ceasing to exist.
        """
        items = [d for d in detections]
        self.budget.check_entities(len(items))
        matched: dict[str, Detection] = {}
        used: set[int] = set()

        for entity in self._entities.values():
            if entity.state == "lost":
                continue
            best_index = self._match(entity, items, used)
            if best_index is None:
                continue
            used.add(best_index)
            matched[entity.entity_id] = items[best_index]

        touched: list[Entity] = []
        touched_ids: set[str] = set()
        for entity_id, detection in matched.items():
            entity = self._entities[entity_id]
            entity.hits += 1
            entity.misses = 0
            entity.last_source_s = detection.source_s
            entity.last_sequence = detection.sequence
            entity.centroid_xy = detection.centroid_xy
            entity.bbox_xyxy = detection.bbox_xyxy
            entity.score = detection.score
            entity.state = "tracked" if entity.hits >= self.confirm_hits else "new"
            entity.evidence_sequences = entity.evidence_sequences + (detection.sequence,)
            touched.append(entity)
            touched_ids.add(entity_id)

        for index, detection in enumerate(items):
            if index in used:
                continue
            spawned = self._spawn(detection, source_s=source_s)
            touched.append(spawned)
            touched_ids.add(spawned.entity_id)

        # Only entities that saw nothing this cycle accumulate misses. Ageing a
        # newly matched or newly spawned entity would report it as unconfirmed
        # in the same cycle it was observed.
        for entity in self._entities.values():
            if entity.entity_id in touched_ids:
                continue
            self._age(entity, source_s)

        self.budget.check_entities_total(len(self._entities))
        return touched

    def _match(self, entity: Entity, items: list[Detection], used: set[int]) -> int | None:
        if entity.centroid_xy is None:
            return None
        best_index, best_distance = None, None
        for index, detection in enumerate(items):
            if index in used:
                continue
            distance = math.dist(entity.centroid_xy, detection.centroid_xy)
            if distance > self.match_radius_px:
                continue
            if best_distance is None or distance < best_distance:
                best_index, best_distance = index, distance
        return best_index

    def _spawn(self, detection: Detection, *, source_s: float) -> Entity:
        entity_id = f"ent-{self._next_entity:04d}"
        self._next_entity += 1
        track = TrackId(self.source_id, self.tracker_epoch, self._next_local_id)
        self._next_local_id += 1
        entity = Entity(
            entity_id=entity_id,
            label=detection.label,
            track=track,
            first_source_s=detection.source_s,
            last_source_s=detection.source_s,
            state="new",
            hits=1,
            centroid_xy=detection.centroid_xy,
            bbox_xyxy=detection.bbox_xyxy,
            score=detection.score,
            first_sequence=detection.sequence,
            last_sequence=detection.sequence,
            evidence_sequences=(detection.sequence,),
            sources=(detection.source_id,),
            meta={"scorer": detection.scorer, "entity_hint": detection.entity_hint},
        )
        self._entities[entity_id] = entity
        return entity

    def _age(self, entity: Entity, source_s: float) -> None:
        """Advance the lifecycle for one cycle with no matching observation.

        Misses accumulate regardless of the current state: an already-occluded
        entity that is still unseen must keep ageing towards ``lost``, otherwise
        it would stay occluded forever.
        """
        if entity.state == "lost":
            return
        entity.misses += 1
        if entity.misses >= self.max_misses:
            entity.state = "lost"
        else:
            entity.state = "occluded"
        if source_s - entity.last_source_s > self.budget.max_track_age_s:
            entity.state = "lost"

    def mark_reacquired(self, entity_id: str, detection: Detection) -> Entity:
        """Re-acquire a lost entity from a new observation, keeping its history."""
        entity = self._entities.get(entity_id)
        if entity is None:
            raise TrackError(f"unknown entity {entity_id}")
        entity.state = "reacquired"
        entity.hits += 1
        entity.misses = 0
        entity.last_source_s = detection.source_s
        entity.last_sequence = detection.sequence
        entity.centroid_xy = detection.centroid_xy
        entity.bbox_xyxy = detection.bbox_xyxy
        entity.score = detection.score
        entity.evidence_sequences = entity.evidence_sequences + (detection.sequence,)
        entity.meta["reacquired_at_source_s"] = detection.source_s
        return entity

    # -- association -------------------------------------------------------
    def associate(
        self,
        entity_id: str,
        member: TrackId,
        *,
        confidence: float,
        supporting_observations: Iterable[int],
        method: str,
        source_s: float,
        same: bool = True,
        notes: str = "",
    ) -> Association:
        """Record a *revisable* belief that a local track belongs to an entity.

        Nothing is merged destructively. If the belief turns out wrong, the
        association is superseded and the event log records a
        ``belief_revision`` rather than an object disappearing.
        """
        if entity_id not in self._entities:
            raise TrackError(f"unknown entity {entity_id}")
        record = Association(
            association_id=f"assoc-{len(self._associations):04d}",
            entity_id=entity_id,
            member=member,
            possible_same_entity=bool(same),
            confidence=float(confidence),
            supporting_observations=tuple(int(item) for item in supporting_observations),
            method=method,
            recorded_at_source_s=float(source_s),
            notes=notes,
        )
        self._associations.append(record)
        if same and member.source_id not in self._entities[entity_id].sources:
            entity = self._entities[entity_id]
            entity.sources = entity.sources + (member.source_id,)
        return record

    def visible_count(self) -> int:
        return sum(1 for entity in self._entities.values() if entity.visible)

    def describe(self) -> dict:
        return {
            "tracker_epoch": self.tracker_epoch,
            "entities": [entity.describe() for entity in self._entities.values()],
            "associations": [item.describe() for item in self._associations],
            "visible": self.visible_count(),
            "total": len(self._entities),
        }
