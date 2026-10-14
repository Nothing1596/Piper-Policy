"""Time provenance for perception records.

Recording two timestamps is not synchronization. This module keeps the *domain*
of every stamp explicit, carries the residual uncertainty, and separates the
three uses of time that Piper Lab already distinguishes in ``safety.py``:

- the source instant (what the camera/robot says, a wall clock in ROS),
- the local receipt instant (monotonic, used for all local decisions),
- the audit instant (walls clock, logs only).

The invariant that matters most: the instant a model finished processing must
never overwrite the instant the observation actually happened. A frame that is
understood late is still an old frame.

Follows the ``SafetyGate`` pattern (``safety.py:114-137``): local decisions use
the monotonic clock, the source stamp is preserved as-is, and a source that is
absent or stale is *refused* rather than repaired with the local clock.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from enum import Enum


class ClockError(ValueError):
    """A stamp cannot be trusted for the requested operation."""


class ClockDomain(str, Enum):
    """Which clock a stamp value belongs to.

    ``ROS_WALL`` is what reaches ``/lab/rgb`` and ``/lab/observation`` today:
    ``runtime.py`` and ``mock_device.py`` both publish ``get_clock().now()``.
    ``HOST_MONOTONIC`` is the only clock allowed for local timeout decisions.
    ``CROSS_HOST_UNSYNCHRONIZED`` marks a producer on another machine whose
    monotonic clock has no established relation to ours; such stamps may be
    logged but must not be differenced against local time.
    """

    ROS_WALL = "ros_wall"
    HOST_MONOTONIC = "host_monotonic"
    HOST_REALTIME = "host_realtime"
    CROSS_HOST_UNSYNCHRONIZED = "cross_host_unsynchronized"


#: Domains whose values may be compared with this host's clocks.
_COMPARABLE = frozenset({ClockDomain.ROS_WALL, ClockDomain.HOST_REALTIME})


@dataclass(frozen=True)
class Mapping:
    """An identified clock relation (identity for same-domain, offset otherwise).

    ``scale``/``offset`` describe ``reference = scale * source + offset``.
    A same-host, same-domain producer uses the identity mapping with zero
    uncertainty; nothing here guesses an offset that has not been established.
    """

    mapping_id: str
    scale: float = 1.0
    offset_s: float = 0.0
    uncertainty_s: float = 0.0

    def __post_init__(self) -> None:
        for name in ("scale", "offset_s", "uncertainty_s"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ClockError(f"Mapping.{name} must be a finite number")
        if self.scale == 0.0 or self.uncertainty_s < 0.0:
            raise ClockError("Mapping.scale must be non-zero and uncertainty non-negative")
        if not self.mapping_id:
            raise ClockError("Mapping.mapping_id is required for provenance")

    def to_reference(self, source_s: float) -> float:
        return self.scale * source_s + self.offset_s

    @property
    def identity(self) -> bool:
        return self.scale == 1.0 and self.offset_s == 0.0


IDENTITY = Mapping(mapping_id="identity")


@dataclass(frozen=True)
class ClockStamp:
    """One timestamp with its domain, age, provenance and uncertainty.

    ``age_s`` is the source-to-receipt delay, not the age at read time. Call
    :meth:`age_at` for the latter; re-reading a record never makes it younger.
    """

    source_s: float
    received_s: float
    domain: ClockDomain
    mapping_id: str
    acquisition_epoch: int
    mapping_uncertainty_s: float = 0.0
    receipt_uncertainty_s: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.domain, ClockDomain):
            raise ClockError("ClockStamp.domain must be a ClockDomain")
        for name in ("source_s", "received_s", "mapping_uncertainty_s", "receipt_uncertainty_s"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ClockError(f"ClockStamp.{name} must be a finite number")
        if isinstance(self.acquisition_epoch, bool) or not isinstance(self.acquisition_epoch, int):
            raise ClockError("ClockStamp.acquisition_epoch must be an int")
        if self.acquisition_epoch < 0:
            raise ClockError("ClockStamp.acquisition_epoch must be non-negative")
        if not self.mapping_id:
            raise ClockError("ClockStamp.mapping_id is required; provenance is not optional")
        for name in ("mapping_uncertainty_s", "receipt_uncertainty_s"):
            if getattr(self, name) < 0.0:
                raise ClockError(f"ClockStamp.{name} must be non-negative")

    @property
    def age_s(self) -> float:
        """Source-to-receipt delay in seconds (may be negative within skew)."""
        return self.received_s - self.source_s

    @property
    def uncertainty_s(self) -> float:
        """Worst-case error of :attr:`age_s`."""
        return self.mapping_uncertainty_s + self.receipt_uncertainty_s

    @property
    def comparable(self) -> bool:
        """True when this stamp may be differenced against local clocks."""
        return self.domain in _COMPARABLE

    def age_at(self, now_monotonic_s: float) -> float:
        """Age of the record now, floored at :attr:`age_s`.

        Receiving or re-reading an old frame does not make it fresh, so the
        result can never be smaller than the original transit delay.
        """
        if not math.isfinite(now_monotonic_s):
            raise ClockError("now_monotonic_s must be finite")
        return max(self.age_s, now_monotonic_s - self.received_s + self.age_s)

    def valid(
        self,
        *,
        max_age_s: float,
        skew_s: float,
        now_monotonic_s: float,
        epoch: int | None = None,
    ) -> bool:
        """Whether the record may be used for a time-bounded decision.

        Mirrors ``SafetyGate._fresh_stamp``: accept ages in
        ``[-skew_s, max_age_s]``, refuse anything else. A non-comparable
        domain, a mismatched acquisition epoch, or uncertainty large enough to
        make the answer meaningless all fail closed.
        """
        for name, value in (("max_age_s", max_age_s), ("skew_s", skew_s)):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ClockError(f"{name} must be a finite number")
        if skew_s < 0.0:
            raise ClockError("skew_s must be non-negative")
        if epoch is not None and epoch != self.acquisition_epoch:
            return False
        if not self.comparable:
            return False
        age = self.age_at(now_monotonic_s)
        if age < -skew_s or age > max_age_s:
            return False
        # If uncertainty alone exceeds the window, the stamp cannot decide.
        return age - self.uncertainty_s >= -skew_s and age + self.uncertainty_s <= max_age_s

    def with_epoch(self, acquisition_epoch: int) -> "ClockStamp":
        return replace(self, acquisition_epoch=acquisition_epoch)

    def describe(self) -> dict:
        """JSON-serializable provenance block for downstream consumers."""
        return {
            "source_s": self.source_s,
            "received_s": self.received_s,
            "domain": self.domain.value,
            "age_s": self.age_s,
            "mapping_id": self.mapping_id,
            "mapping_uncertainty_s": self.mapping_uncertainty_s,
            "receipt_uncertainty_s": self.receipt_uncertainty_s,
            "acquisition_epoch": self.acquisition_epoch,
            "comparable": self.comparable,
        }


def in_domain(source_s: float, received_s: float, domain: ClockDomain, *,
              mapping_id: str = "identity", acquisition_epoch: int = 0,
              uncertainty_s: float = 0.0) -> ClockStamp:
    """Build a stamp for a producer already in a known, established domain."""
    return ClockStamp(
        source_s=float(source_s),
        received_s=float(received_s),
        domain=domain,
        mapping_id=mapping_id,
        acquisition_epoch=acquisition_epoch,
        mapping_uncertainty_s=float(uncertainty_s),
    )


def from_mapping(source_s: float, received_s: float, mapping: Mapping, domain: ClockDomain, *,
                 acquisition_epoch: int = 0) -> ClockStamp:
    """Map an untrusted source stamp into the reference domain.

    The mapped value replaces ``source_s``; the original mapping identity and
    its uncertainty travel with the record so a consumer can tell an exact
    same-domain stamp from an estimated one.
    """
    return ClockStamp(
        source_s=mapping.to_reference(float(source_s)),
        received_s=float(received_s),
        domain=domain,
        mapping_id=mapping.mapping_id,
        acquisition_epoch=acquisition_epoch,
        mapping_uncertainty_s=mapping.uncertainty_s,
    )


#: Invalidation vocabulary shared with the executor layer (PiperX-style codes).
#: A stale state view must cite one of these instead of a home-made boolean.
INVALIDATION_CODES = (
    "stale_feedback",
    "invalid_feedback",
    "control_mode",
    "robot_not_ready",
    "velocity_limit",
    "gripper_not_ready",
    "not_connected",
    "busy",
    "reconnect_required",
    "clock_mismatch",
    "acquisition_epoch_changed",
    "capacity_evicted",
    "capture_gap",
    "superseded",
)


def is_known_invalidation(code: str) -> bool:
    return code in INVALIDATION_CODES
