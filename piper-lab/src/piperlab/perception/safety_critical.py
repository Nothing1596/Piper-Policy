"""Attention requests that can change *focus* and nothing else.

A model-driven attention channel is also an injection surface. If "look only at
the cup" can switch off a safety-relevant analysis, then any text that reaches
the slow model can degrade monitoring.

So the analyses are split into two hard-coded sets:

- ``SAFETY_CRITICAL`` — never addressable by attention, at all.
- ``ATTENTION_TUNABLE`` — sampling density, branch enablement, event ordering.

Attention may adjust thresholds *within* a tunable analysis. It may never move
a threshold so far that the analysis stops running, and it may never write to
the critical set. There is no override flag: the whitelist is a module
constant, which is what makes it reviewable.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable

#: Analyses whose absence would make the pipeline unsafe or unauditable.
#: Attention cannot name these, disable these, or change their thresholds.
SAFETY_CRITICAL: frozenset[str] = frozenset(
    {
        "frame_liveness",      # is the stream actually alive
        "decode_integrity",    # did the frame decode
        "state_freshness",     # is robot state within the contract window
        "gap_accounting",      # are capture/eviction gaps recorded
        "clock_provenance",    # does every record keep its clock domain
    }
)

#: Analyses attention may re-prioritise or re-enable.
ATTENTION_TUNABLE: frozenset[str] = frozenset(
    {
        "object_detect",
        "pose",
        "embedding",
        "motion_energy",
        "frame_rate",
        "event_ordering",
        "evidence_prefetch",
    }
)


class AttentionRejected(ValueError):
    """An attention request tried to touch something it may not."""


@dataclass(frozen=True)
class AttentionRequest:
    """A time- and budget-bounded observation request from the slow model."""

    request_id: str
    entities: tuple[str, ...]
    changes: tuple[str, ...]
    scope: str
    issued_at_source_s: float
    valid_until_source_s: float
    budget_hint: dict = field(default_factory=dict)
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.request_id:
            raise AttentionRejected("AttentionRequest.request_id is required")
        if not self.entities:
            raise AttentionRejected(
                "AttentionRequest.entities is required; an unbounded scope is not a request"
            )
        if not self.scope:
            raise AttentionRejected("AttentionRequest.scope is required (task phase or window)")
        for name in ("issued_at_source_s", "valid_until_source_s"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise AttentionRejected(f"AttentionRequest.{name} must be finite")
        if self.valid_until_source_s <= self.issued_at_source_s:
            raise AttentionRejected(
                "AttentionRequest must expire: an unbounded attention grant is not accepted"
            )

    def expired_at(self, source_s: float) -> bool:
        return source_s > self.valid_until_source_s

    def describe(self) -> dict:
        return {
            "request_id": self.request_id,
            "entities": list(self.entities),
            "changes": list(self.changes),
            "scope": self.scope,
            "issued_at_source_s": self.issued_at_source_s,
            "valid_until_source_s": self.valid_until_source_s,
            "budget_hint": dict(self.budget_hint),
            "notes": self.notes,
        }


@dataclass(frozen=True)
class AttentionEffect:
    """The validated, bounded effect an attention request is allowed to have."""

    request_id: str
    enabled_analyses: tuple[str, ...]
    disabled_analyses: tuple[str, ...]
    threshold_overrides: dict
    max_extra_frames: int
    expires_at_source_s: float

    def describe(self) -> dict:
        return {
            "request_id": self.request_id,
            "enabled_analyses": list(self.enabled_analyses),
            "disabled_analyses": list(self.disabled_analyses),
            "threshold_overrides": dict(self.threshold_overrides),
            "max_extra_frames": self.max_extra_frames,
            "expires_at_source_s": self.expires_at_source_s,
            "cannot_touch": sorted(SAFETY_CRITICAL),
        }


def validate_attention(
    request: AttentionRequest,
    *,
    enable: Iterable[str] = (),
    disable: Iterable[str] = (),
    threshold_overrides: dict | None = None,
    max_extra_frames: int = 0,
) -> AttentionEffect:
    """Approve an attention request, or reject it outright.

    Rejection is the default for anything that touches a critical analysis or
    removes an analysis entirely by pushing its threshold out of range.
    """
    enable_tuple = tuple(dict.fromkeys(str(name) for name in enable))
    disable_tuple = tuple(dict.fromkeys(str(name) for name in disable))
    overrides = {str(key): value for key, value in (threshold_overrides or {}).items()}

    if isinstance(max_extra_frames, bool) or not isinstance(max_extra_frames, int) or max_extra_frames < 0:
        raise AttentionRejected("max_extra_frames must be a non-negative int")

    for name in (*enable_tuple, *disable_tuple, *overrides):
        if name in SAFETY_CRITICAL:
            raise AttentionRejected(
                f"attention may not reference the safety-critical analysis {name!r}; "
                "these are always active and not model-addressable"
            )
        if name not in ATTENTION_TUNABLE:
            raise AttentionRejected(
                f"unknown analysis {name!r}; attention is restricted to "
                + ", ".join(sorted(ATTENTION_TUNABLE))
            )

    if set(enable_tuple) & set(disable_tuple):
        raise AttentionRejected("the same analysis cannot be both enabled and disabled")

    for name, value in overrides.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise AttentionRejected(f"threshold override for {name!r} must be finite")
        if name in ("object_detect", "pose", "embedding", "motion_energy") and value <= 0.0:
            raise AttentionRejected(
                f"threshold for {name!r} must stay positive: attention may re-prioritise "
                "but must not switch an analysis off"
            )
        if name == "frame_rate" and value <= 0.0:
            raise AttentionRejected("frame_rate must stay positive")

    return AttentionEffect(
        request_id=request.request_id,
        enabled_analyses=enable_tuple,
        disabled_analyses=disable_tuple,
        threshold_overrides=overrides,
        max_extra_frames=max_extra_frames,
        expires_at_source_s=request.valid_until_source_s,
    )


def assert_critical_always_on(active: Iterable[str]) -> None:
    """Verify a running configuration still includes every critical analysis."""
    missing = SAFETY_CRITICAL - set(active)
    if missing:
        raise AttentionRejected(
            "safety-critical analyses are not active: " + ", ".join(sorted(missing))
        )
