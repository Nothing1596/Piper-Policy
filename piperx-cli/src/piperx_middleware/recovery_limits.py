"""Bounded recovery of an already out-of-range pose; never expands a new target.

The worker-local envelope is installed only after operator approval. It also
covers a cancellation hold, and is removed on every success/error path.
"""
from contextlib import contextmanager
from contextvars import ContextVar
import math

from .models import DomainError, JOINT_LIMITS_DEG

RECOVERY_MARGIN_DEG = 5.0
_worker_bounds = ContextVar("piper_recovery_bounds", default=None)


def recovery_bounds(current, lower, upper):
    if len(current) != 6 or any(not math.isfinite(q) for q in current):
        raise DomainError("invalid_state", "Recovery requires six finite measured joint angles.", 422)
    bounds = []
    for i, (q, lo, hi) in enumerate(zip(current, lower, upper)):
        if not lo - RECOVERY_MARGIN_DEG <= q <= hi + RECOVERY_MARGIN_DEG:
            raise DomainError("recovery_limit", f"J{i+1} exceeds the 5 degree recovery margin.", 422)
        bounds.append((min(q, lo), max(q, hi)))
    return bounds


def check_bounds(q, bounds, code="recovery_direction"):
    if len(q) != 6 or any(not math.isfinite(v) or not lo <= v <= hi
                         for v, (lo, hi) in zip(q, bounds)):
        raise DomainError(code, "Recovery may only hold or move inward; no new or larger joint-limit violation.", 422)


def worker_bounds():
    return _worker_bounds.get()


@contextmanager
def recovery_scope(bounds):
    if bounds is not None:
        if len(bounds) != 6 or any(not all(math.isfinite(v) for v in pair) or
                not lo - RECOVERY_MARGIN_DEG <= pair[0] <= pair[1] <= hi + RECOVERY_MARGIN_DEG
                for pair, (lo, hi) in zip(bounds, JOINT_LIMITS_DEG)):
            raise DomainError("recovery_limit", "Invalid approved recovery envelope.", 422)
    token = _worker_bounds.set(bounds)
    try:
        yield
    finally:
        _worker_bounds.reset(token)
