"""Host-side perception infrastructure for Piper Lab.

Purpose is not extra cleverness but *accountability*: every downstream consumer
must be able to learn what a conclusion was based on, which instant it
describes, what was omitted, and whether it is still trustworthy.

Hard boundaries (see piper-lab/AGENTS.md and ../AGENTS.md):

- This package never enables, arms, moves or calibrates hardware.
- It never writes ``SafetyGate`` state and holds *no* motion authority; the
  gate in ``piperlab.safety`` remains the only admission point.
- With a missing timestamp it refuses the record; it never substitutes a wall
  clock for a source stamp.
- Over-budget work raises instead of silently truncating.

Only numpy / Pillow / PyYAML are used, so these tests run without torch or
LeRobot.
"""
from __future__ import annotations

from .budget import Budget, BudgetExceeded, Retention
from .clockmap import (
    ClockDomain,
    ClockError,
    ClockStamp,
    INVALIDATION_CODES,
    Mapping,
)
from .buffer import CaptureGap, FrameRecord, Gap, RingBuffer
from .evidence import EvidenceResult, EvidenceStatus
from .claims import (
    Claim,
    ClaimError,
    ClaimStore,
    EvidenceRef,
    P_COMMAND_ACKNOWLEDGED,
    P_ENTITY_OBSERVED,
    P_FRAME_AVAILABLE,
    P_GAP_ACCOUNTED,
    P_IDENTITY_SAME,
    P_STATE_FRESH,
    P_TARGET_REACHED,
    Predicate,
    Verdict,
)
from .detect import ColorBlobScorer, Detection, MotionEnergyScorer, Scorer
from .events import Event, EventError, EventKind, EventLog, classify, identity_merge_event
from .fuse import Calibration, CalibrationError, FusedEntity, Projection, fuse_entities, project
from .safety_critical import (
    ATTENTION_TUNABLE,
    SAFETY_CRITICAL,
    AttentionEffect,
    AttentionRequest,
    AttentionRejected,
    assert_critical_always_on,
    validate_attention,
)
from .select import (
    DropReason,
    DroppedCandidate,
    SelectionError,
    SelectionPolicy,
    SelectionReport,
    select_detections,
)
from .tracks import Association, Entity, EntityTracker, TrackError, TrackId
from .view import (
    FieldValue,
    StateStore,
    StateView,
    Staleness,
    ViewError,
    build_view,
)
from .escalate import (
    DegradedPolicy,
    DegradedTier,
    EscalationError,
    EscalationRequest,
    Escalator,
    FakeSlowModel,
    SlowModelPort,
    TriggerKind,
    conflict_event,
    escalation_event,
)

__all__ = [
    "ATTENTION_TUNABLE",
    "INVALIDATION_CODES",
    "P_COMMAND_ACKNOWLEDGED",
    "P_ENTITY_OBSERVED",
    "P_FRAME_AVAILABLE",
    "P_GAP_ACCOUNTED",
    "P_IDENTITY_SAME",
    "P_STATE_FRESH",
    "P_TARGET_REACHED",
    "SAFETY_CRITICAL",
    "Association",
    "AttentionEffect",
    "AttentionRejected",
    "AttentionRequest",
    "Budget",
    "BudgetExceeded",
    "Calibration",
    "CalibrationError",
    "CaptureGap",
    "Claim",
    "ClaimError",
    "ClaimStore",
    "ClockDomain",
    "ClockError",
    "ClockStamp",
    "ColorBlobScorer",
    "DegradedPolicy",
    "DegradedTier",
    "Detection",
    "DropReason",
    "DroppedCandidate",
    "Entity",
    "EntityTracker",
    "EscalationError",
    "EscalationRequest",
    "Escalator",
    "Event",
    "EventError",
    "EventKind",
    "EventLog",
    "EvidenceRef",
    "EvidenceResult",
    "EvidenceStatus",
    "FakeSlowModel",
    "FieldValue",
    "FrameRecord",
    "FusedEntity",
    "Gap",
    "Mapping",
    "MotionEnergyScorer",
    "Predicate",
    "Projection",
    "Retention",
    "RingBuffer",
    "Scorer",
    "SelectionError",
    "SelectionPolicy",
    "SelectionReport",
    "SlowModelPort",
    "Staleness",
    "StateStore",
    "StateView",
    "TrackError",
    "TrackId",
    "TriggerKind",
    "Verdict",
    "ViewError",
    "assert_critical_always_on",
    "build_view",
    "classify",
    "conflict_event",
    "escalation_event",
    "fuse_entities",
    "identity_merge_event",
    "project",
    "select_detections",
    "validate_attention",
]
