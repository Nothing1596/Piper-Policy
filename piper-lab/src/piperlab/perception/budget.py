"""Explicit budgets for the perception layer.

Every arrow in the pipeline gets a number here, and exceeding it raises. Silent
truncation is the failure mode this module exists to prevent: a consumer that
receives fewer items than existed cannot tell "nothing was there" from "we
dropped it", which is exactly the confusion the pipeline is supposed to remove.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


class BudgetExceeded(RuntimeError):
    """A configured budget was exceeded; nothing was truncated to fit."""

    def __init__(self, scope: str, limit: float, observed: float, unit: str):
        self.scope = scope
        self.limit = limit
        self.observed = observed
        self.unit = unit
        super().__init__(
            f"{scope} exceeds budget: {observed:g} > {limit:g} {unit}. "
            "No data was truncated; raise the budget or reduce the request."
        )

    def report(self) -> dict:
        return {
            "error": "budget_exceeded",
            "scope": self.scope,
            "limit": self.limit,
            "observed": self.observed,
            "unit": self.unit,
        }


def _positive(name: str, value) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"Budget.{name} must be a finite number")
    if value <= 0:
        raise ValueError(f"Budget.{name} must be positive")
    return float(value)


def _non_negative(name: str, value) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"Budget.{name} must be a finite number")
    if value < 0:
        raise ValueError(f"Budget.{name} must be non-negative")
    return float(value)


@dataclass(frozen=True)
class Retention:
    """How long each storage tier keeps data.

    ``capacity_evicts_before_ttl`` documents the real behaviour when both a TTL
    and a capacity limit are configured: capacity wins, the TTL is a *maximum*
    retention rather than a promise. Evicted data must be reported as
    ``capacity_evicted``, never silently substituted with a neighbouring frame.
    """

    l1_s: float = 20.0
    l2_s: float = 300.0
    l3_s: float = 3600.0
    capacity_evicts_before_ttl: bool = True

    def __post_init__(self) -> None:
        for name in ("l1_s", "l2_s", "l3_s"):
            _positive(name, getattr(self, name))
        if not (self.l1_s <= self.l2_s <= self.l3_s):
            raise ValueError("Retention tiers must satisfy l1_s <= l2_s <= l3_s")

    def as_dict(self) -> dict:
        return {
            "l1_s": self.l1_s,
            "l2_s": self.l2_s,
            "l3_s": self.l3_s,
            "capacity_evicts_before_ttl": self.capacity_evicts_before_ttl,
        }


@dataclass(frozen=True)
class Budget:
    """All hard limits for the perception layer."""

    # --- ring buffer / L1 ---
    ring_bytes: int = 256 * 1024 * 1024
    """Hard cap on retained L1 payload bytes. This is the L1 *time* budget."""

    # --- per-window selection ---
    max_frames_per_source_per_window: int = 24
    max_candidates_total: int = 48
    window_s: float = 30.0

    # --- entity / tracking ---
    max_entities_per_frame: int = 16
    max_track_age_s: float = 5.0
    max_entities_total: int = 64

    # --- evidence queries ---
    max_evidence_frames_per_query: int = 8
    evidence_window_s: float = 1.0

    # --- escalation ---
    escalation_min_gap_s: float = 0.5
    heartbeat_s: float = 2.0

    # --- text surfaced to any model ---
    max_text_chars: int = 1024 * 1024

    retention: Retention = Retention()

    def __post_init__(self) -> None:
        for name in (
            "ring_bytes",
            "max_frames_per_source_per_window",
            "max_candidates_total",
            "max_entities_per_frame",
            "max_entities_total",
            "max_evidence_frames_per_query",
            "max_text_chars",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"Budget.{name} must be a positive int")
        for name in (
            "window_s",
            "max_track_age_s",
            "evidence_window_s",
            "escalation_min_gap_s",
            "heartbeat_s",
        ):
            _positive(name, getattr(self, name))
        if not isinstance(self.retention, Retention):
            raise ValueError("Budget.retention must be a Retention")

    # -- checks -------------------------------------------------------------
    def check_count(self, scope: str, observed: int, limit: int) -> int:
        """Return ``observed`` when within budget, else raise."""
        if observed > limit:
            raise BudgetExceeded(scope, limit, observed, "items")
        return observed

    def check_entities(self, observed: int) -> int:
        return self.check_count("max_entities_per_frame", observed, self.max_entities_per_frame)

    def check_entities_total(self, observed: int) -> int:
        return self.check_count("max_entities_total", observed, self.max_entities_total)

    def check_evidence_frames(self, observed: int) -> int:
        return self.check_count(
            "max_evidence_frames_per_query", observed, self.max_evidence_frames_per_query
        )

    def check_text(self, observed: int) -> int:
        return self.check_count("max_text_chars", observed, self.max_text_chars)

    def check_ring_bytes(self, observed: int) -> int:
        return self.check_count("ring_bytes", observed, self.ring_bytes)

    def check_single_payload(self, incoming_bytes: int) -> int:
        """Reject a payload that could never fit, before anything is evicted.

        This is deliberately *not* a cumulative capacity check: filling the ring
        legitimately evicts older unpinned records, so overshoot is expected and
        handled by the ring's eviction path, which raises when nothing may be
        removed.
        """
        _non_negative("incoming_bytes", incoming_bytes)
        if incoming_bytes > self.ring_bytes:
            raise BudgetExceeded("ring_bytes", self.ring_bytes, incoming_bytes, "bytes")
        return incoming_bytes

    def as_dict(self) -> dict:
        return {
            "ring_bytes": self.ring_bytes,
            "max_frames_per_source_per_window": self.max_frames_per_source_per_window,
            "max_candidates_total": self.max_candidates_total,
            "window_s": self.window_s,
            "max_entities_per_frame": self.max_entities_per_frame,
            "max_track_age_s": self.max_track_age_s,
            "max_entities_total": self.max_entities_total,
            "max_evidence_frames_per_query": self.max_evidence_frames_per_query,
            "evidence_window_s": self.evidence_window_s,
            "escalation_min_gap_s": self.escalation_min_gap_s,
            "heartbeat_s": self.heartbeat_s,
            "max_text_chars": self.max_text_chars,
            "retention": self.retention.as_dict(),
        }
