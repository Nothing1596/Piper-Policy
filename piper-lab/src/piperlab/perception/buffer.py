"""Bounded observation buffer with an explicit gap ledger.

The buffer answers three different questions that must never collapse into one
boolean:

``not_selected``
    The observation exists; this analysis branch simply was not run on it. It is
    recoverable by re-running the selector over the same data.
``capture_gap``
    Nothing was ever captured or received for that interval. This is a
    *knowledge gap*: the correct downstream conclusion is "unknown", not
    "nothing happened".
``evicted``
    The observation existed and was removed to respect capacity or retention.
    Irrecoverable, and reported as such.

Conflating these turns "we never looked" into "we looked and found nothing",
which is the exact class of error this package exists to prevent.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Iterable, Iterator

from .budget import Budget, BudgetExceeded
from .clockmap import ClockStamp


class BufferError(ValueError):
    """A record cannot be accepted into the buffer."""


@dataclass(frozen=True)
class FrameRecord:
    """One captured observation.

    ``payload`` is the L1 bytes when retained in memory. Records may keep only
    an ``l1_ref`` (segment plus byte offset) so the L1 tier can live on disk;
    ``payload_bytes`` always states how much L1 data the record stands for, so
    capacity accounting is correct either way.
    """

    source_id: str
    sequence: int
    stamp: ClockStamp
    payload_bytes: int = 0
    payload: bytes | None = None
    l1_ref: str | None = None
    format: str = "jpeg"
    camera_frame_id: str | None = None
    retention_hint_s: float | None = None
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.source_id:
            raise BufferError("FrameRecord.source_id is required")
        if isinstance(self.sequence, bool) or not isinstance(self.sequence, int) or self.sequence < 0:
            raise BufferError("FrameRecord.sequence must be a non-negative int")
        if not isinstance(self.stamp, ClockStamp):
            raise BufferError("FrameRecord.stamp must be a ClockStamp")
        if isinstance(self.payload_bytes, bool) or not isinstance(self.payload_bytes, int):
            raise BufferError("FrameRecord.payload_bytes must be an int")
        if self.payload_bytes < 0:
            raise BufferError("FrameRecord.payload_bytes must be non-negative")
        if self.payload is not None and len(self.payload) != self.payload_bytes:
            raise BufferError(
                "FrameRecord.payload length must equal payload_bytes for correct accounting"
            )

    @property
    def source_s(self) -> float:
        return self.stamp.source_s

    @property
    def received_s(self) -> float:
        return self.stamp.received_s

    def describe(self, *, include_payload: bool = False) -> dict:
        record = {
            "source_id": self.source_id,
            "sequence": self.sequence,
            "stamp": self.stamp.describe(),
            "payload_bytes": self.payload_bytes,
            "format": self.format,
            "camera_frame_id": self.camera_frame_id,
            "l1_ref": self.l1_ref,
            "meta": dict(self.meta),
        }
        if include_payload:
            record["payload"] = self.payload
        return record


@dataclass(frozen=True)
class CaptureGap:
    """A known interval in which nothing was captured or received.

    Recorded by the capture layer when it knows a source stopped or dropped
    frames. It is evidence of absence of data, never evidence of absence of
    events.
    """

    source_id: str
    start_source_s: float
    end_source_s: float
    reason: str
    detail: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.source_id:
            raise BufferError("CaptureGap.source_id is required")
        for name in ("start_source_s", "end_source_s"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise BufferError(f"CaptureGap.{name} must be finite")
        if self.end_source_s < self.start_source_s:
            raise BufferError("CaptureGap.end_source_s must not precede start")
        if not self.reason:
            raise BufferError("CaptureGap.reason is required; gaps are not anonymous")

    @property
    def duration_s(self) -> float:
        return self.end_source_s - self.start_source_s

    def describe(self) -> dict:
        return {
            "source_id": self.source_id,
            "start_source_s": self.start_source_s,
            "end_source_s": self.end_source_s,
            "duration_s": self.duration_s,
            "reason": self.reason,
            "detail": dict(self.detail),
        }


@dataclass(frozen=True)
class Gap:
    """A recorded interval where data once existed and no longer does."""

    source_id: str
    start_sequence: int
    end_sequence: int
    reason: str
    evicted_bytes: int
    start_source_s: float | None = None
    end_source_s: float | None = None

    def describe(self) -> dict:
        return {
            "source_id": self.source_id,
            "start_sequence": self.start_sequence,
            "end_sequence": self.end_sequence,
            "reason": self.reason,
            "evicted_bytes": self.evicted_bytes,
            "start_source_s": self.start_source_s,
            "end_source_s": self.end_source_s,
        }


class RingBuffer:
    """Per-source FIFO of :class:`FrameRecord` with a byte budget and gap ledger.

    Records are appended in strictly increasing ``source_s`` order per source.
    When the byte budget is exceeded the oldest unpinned record is evicted and
    the eviction is recorded as a :class:`Gap` so a later query can say
    ``evicted`` instead of quietly returning a neighbour.
    """

    def __init__(self, budget: Budget | None = None) -> None:
        self.budget = budget or Budget()
        self._records: deque[FrameRecord] = deque()
        self._gaps: deque[Gap] = deque()
        self._capture_gaps: list[CaptureGap] = []
        self._bytes = 0
        self._pinned: dict[tuple[str, int], float] = {}
        self._pushed = 0
        self._evicted = 0
        #: Current acquisition epoch per source; used to reject interleaving.
        self._open_epoch: dict[str, int] = {}
        #: Source-time of the most recent declared epoch boundary per source.
        self._epoch_boundary: dict[str, float] = {}
        #: Sequences not yet pushed (e.g. after a restart), per source.
        self._missing: dict[str, set[int]] = {}

    # -- ingest ------------------------------------------------------------
    def push(self, record: FrameRecord, *, pin: bool = False) -> FrameRecord:
        """Append a record, evicting older unpinned records if over budget.

        ``pin=True`` protects the *incoming* record from being the one evicted.
        That matters because eviction removes the oldest unpinned record, which
        for a large single payload would otherwise be the very observation being
        recorded: the ring would report success while dropping the frame it was
        just handed. With ``pin=True`` the buffer instead refuses when the
        protected set cannot fit.
        """
        self.budget.check_single_payload(record.payload_bytes)
        self._check_order(record)
        self._records.append(record)
        self._bytes += record.payload_bytes
        self._pushed += 1
        if pin:
            self.pin(
                record.source_id,
                record.sequence,
                record.stamp.source_s
                + (record.retention_hint_s if record.retention_hint_s is not None else 1e9),
            )
        try:
            self._evict_to_budget()
        except BudgetExceeded:
            # Roll the refused record back completely: it was never accepted, so
            # leaving it in the FIFO would put phantom data in the buffer and
            # corrupt the byte accounting.
            self._records.remove(record)
            self._bytes -= record.payload_bytes
            self._pushed -= 1
            self.unpin(record.source_id, record.sequence)
            raise
        return record

    def push_capture_gap(self, gap: CaptureGap) -> CaptureGap:
        """Record a known interval with no data; never merged into a negative claim."""
        self._capture_gaps.append(gap)
        self._capture_gaps.sort(key=lambda item: (item.source_id, item.start_source_s))
        return gap

    def _check_order(self, record: FrameRecord) -> None:
        last = self._last_for(record.source_id)
        declared = self._open_epoch.get(record.source_id)
        if declared is not None and record.stamp.acquisition_epoch != declared:
            raise BufferError(
                f"record epoch {record.stamp.acquisition_epoch} does not match the "
                f"declared epoch {declared} for source {record.source_id!r}; call "
                "begin_epoch() to change it"
            )
        if last is None:
            # After a declared boundary the clock and sequences restart, so no
            # ordering constraint applies yet.
            self._open_epoch[record.source_id] = record.stamp.acquisition_epoch
            return
        self._open_epoch.setdefault(record.source_id, record.stamp.acquisition_epoch)
        if record.stamp.acquisition_epoch != last.stamp.acquisition_epoch:
            raise BufferError(
                "acquisition_epoch changed without an intervening epoch boundary; "
                "old and new data must not be interleaved"
            )
        if record.sequence <= last.sequence:
            raise BufferError(
                f"sequence must increase per source: {record.sequence} after {last.sequence}"
            )
        if record.stamp.source_s <= last.stamp.source_s:
            raise BufferError(
                "source timestamps must strictly increase per source: "
                f"{record.stamp.source_s} after {last.stamp.source_s}"
            )

    def begin_epoch(self, acquisition_epoch: int) -> list[CaptureGap]:
        """Declare a device/driver restart boundary.

        A restart resets the source clock and sequence numbering, so records
        before and after it must not be interleaved or sorted together. This
        closes every currently open per-source stream with an explicit capture
        gap (``reason=acquisition_epoch_changed``) so a later query can explain
        why history stops there, and returns the gaps that were recorded.
        """
        if isinstance(acquisition_epoch, bool) or not isinstance(acquisition_epoch, int) \
                or acquisition_epoch < 0:
            raise BufferError("acquisition_epoch must be a non-negative int")
        last_source_s: dict[str, float] = {}
        for record in self._records:
            last_source_s[record.source_id] = max(
                last_source_s.get(record.source_id, -math.inf), record.stamp.source_s
            )
        for source_id in sorted(last_source_s):
            self._open_epoch[source_id] = acquisition_epoch
        self._epoch_boundary.clear()
        self._missing.clear()
        produced: list[CaptureGap] = []
        for source_id, source_s in sorted(last_source_s.items()):
            gap = CaptureGap(
                source_id=source_id,
                start_source_s=source_s,
                end_source_s=source_s,
                reason="acquisition_epoch_changed",
                detail={"new_acquisition_epoch": acquisition_epoch},
            )
            self.push_capture_gap(gap)
            # Nothing is retained after the boundary yet, so a query for a later
            # instant must say "unavailable" rather than resurrect the old epoch.
            self._epoch_boundary[source_id] = source_s
            produced.append(gap)
        return produced

    def _last_for(self, source_id: str) -> FrameRecord | None:
        """Newest retained record of the source in its *current* epoch.

        Records from a superseded epoch are deliberately invisible here: after a
        restart the clock and numbering begin again, so comparing against them
        would invent an ordering violation.
        """
        declared = self._open_epoch.get(source_id)
        for record in reversed(self._records):
            if record.source_id != source_id:
                continue
            if declared is not None and record.stamp.acquisition_epoch != declared:
                continue
            return record
        return None

    # -- retention ---------------------------------------------------------
    def pin(self, source_id: str, sequence: int, until_source_s: float) -> None:
        """Protect a record from eviction until a source-time deadline.

        Used for frames referenced by an emitted event that a slow consumer may
        still request. Pinning is a promise with an expiry, never permanent.
        """
        if not math.isfinite(until_source_s):
            raise BufferError("pin deadline must be finite")
        self._pinned[(source_id, sequence)] = until_source_s

    def unpin(self, source_id: str, sequence: int) -> None:
        self._pinned.pop((source_id, sequence), None)

    def _evict_to_budget(self) -> None:
        """Evict oldest *unpinned* records until the byte budget is met.

        A pin is a promise to a slow consumer that referenced frame will still be
        there, so pinned records are skipped. If every retained record is
        pinned, refusing is the only honest option: silently dropping one would
        break exactly the guarantee the pin exists to provide.
        """
        while self._bytes > self.budget.ring_bytes and self._records:
            candidate = next(
                (r for r in self._records if (r.source_id, r.sequence) not in self._pinned),
                None,
            )
            if candidate is None:
                raise BudgetExceeded(
                    "ring_bytes_pinned", self.budget.ring_bytes, self._bytes, "bytes"
                )
            self._records.remove(candidate)
            self._bytes -= candidate.payload_bytes
            self._evicted += 1
            self._record_gap(candidate, "capacity_evicted")

    def expire_pins(self, now_source_s: float) -> list[Gap]:
        """Drop expired pins and evict as needed, returning new gaps."""
        if not math.isfinite(now_source_s):
            raise BufferError("now_source_s must be finite")
        before = len(self._gaps)
        self._pinned = {
            key: deadline
            for key, deadline in self._pinned.items()
            if deadline > now_source_s
        }
        self._evict_to_budget()
        return list(self._gaps)[before:]

    def expire_retention(self, now_source_s: float, *, tier: str = "l1") -> list[Gap]:
        """Remove records older than the requested retention tier.

        ``tier`` is one of ``l1``/``l2``/``l3`` and selects the TTL. Removal is
        always reported as a gap with ``reason=retention_<tier>``.
        """
        if tier not in ("l1", "l2", "l3"):
            raise BufferError("tier must be l1, l2 or l3")
        ttl = {"l1": self.budget.retention.l1_s, "l2": self.budget.retention.l2_s,
               "l3": self.budget.retention.l3_s}[tier]
        before = len(self._gaps)
        for record in list(self._records):
            if (record.source_id, record.sequence) in self._pinned:
                continue
            if now_source_s - record.stamp.source_s <= ttl:
                continue
            self._records.remove(record)
            self._bytes -= record.payload_bytes
            self._evicted += 1
            self._note_removal(record, f"retention_{tier}")
        return list(self._gaps)[before:]

    def _note_removal(self, record: FrameRecord, reason: str) -> None:
        """Record that a record is gone without re-removing it from the queue.

        Used when the record was pulled out earlier in the same operation, e.g.
        a pinned record evicted after its pin expired.
        """
        self._record_gap(record, reason)

    def _record_gap(self, record: FrameRecord, reason: str) -> None:
        for gap in reversed(self._gaps):
            if gap.source_id == record.source_id and gap.end_sequence + 1 == record.sequence \
                    and gap.reason == reason:
                self._gaps.remove(gap)
                self._gaps.append(
                    Gap(
                        source_id=gap.source_id,
                        start_sequence=gap.start_sequence,
                        end_sequence=record.sequence,
                        reason=reason,
                        evicted_bytes=gap.evicted_bytes + record.payload_bytes,
                        start_source_s=gap.start_source_s,
                        end_source_s=record.stamp.source_s,
                    )
                )
                return
        self._gaps.append(
            Gap(
                source_id=record.source_id,
                start_sequence=record.sequence,
                end_sequence=record.sequence,
                reason=reason,
                evicted_bytes=record.payload_bytes,
                start_source_s=record.stamp.source_s,
                end_source_s=record.stamp.source_s,
            )
        )

    # -- query -------------------------------------------------------------
    def records(self, source_id: str | None = None, *, current_epoch_only: bool = False) -> Iterator[FrameRecord]:
        """Iterate retained records, optionally restricted to open epochs.

        ``current_epoch_only=True`` is what consumers that need a coherent
        timeline should use: records from a superseded acquisition epoch remain
        available for audit but must not be mixed into a live answer.
        """
        for record in self._records:
            if source_id is not None and record.source_id != source_id:
                continue
            declared = self._open_epoch.get(record.source_id)
            if current_epoch_only and declared is not None \
                    and record.stamp.acquisition_epoch != declared:
                continue
            yield record

    def latest(self, source_id: str) -> FrameRecord | None:
        for record in reversed(self._records):
            if record.source_id == source_id:
                return record
        return None

    def at_or_before(self, source_id: str, source_s: float) -> FrameRecord | None:
        """Newest retained record whose source time is ``<= source_s``."""
        found = None
        for record in self._records:
            if record.source_id != source_id:
                continue
            if record.stamp.source_s <= source_s:
                found = record
            else:
                break
        return found

    def gaps(self, source_id: str | None = None) -> list[Gap]:
        return [g for g in self._gaps if source_id is None or g.source_id == source_id]

    def capture_gaps(self, source_id: str | None = None) -> list[CaptureGap]:
        return [g for g in self._capture_gaps if source_id is None or g.source_id == source_id]

    def evicted_covering(self, source_id: str, source_s: float) -> Gap | None:
        """Return the eviction gap that covers ``source_s``, if any."""
        for gap in self._gaps:
            if gap.source_id != source_id:
                continue
            start = gap.start_source_s
            end = gap.end_source_s
            if start is not None and end is not None and start <= source_s <= end:
                return gap
        return None

    def capture_gap_covering(self, source_id: str, source_s: float) -> CaptureGap | None:
        for gap in self._capture_gaps:
            if gap.source_id == source_id and gap.start_source_s <= source_s <= gap.end_source_s:
                return gap
        return None

    def sources(self) -> list[str]:
        return sorted({record.source_id for record in self._records})

    # -- stats -------------------------------------------------------------
    def stats(self) -> dict:
        by_source: dict[str, dict] = {}
        for record in self._records:
            entry = by_source.setdefault(
                record.source_id,
                {"retained": 0, "bytes": 0, "oldest_source_s": None, "newest_source_s": None,
                 "oldest_sequence": None, "newest_sequence": None},
            )
            entry["retained"] += 1
            entry["bytes"] += record.payload_bytes
            if entry["oldest_source_s"] is None:
                entry["oldest_source_s"] = record.stamp.source_s
                entry["oldest_sequence"] = record.sequence
            entry["newest_source_s"] = record.stamp.source_s
            entry["newest_sequence"] = record.sequence
        return {
            "pushed": self._pushed,
            "retained": len(self._records),
            "bytes": self._bytes,
            "ring_bytes": self.budget.ring_bytes,
            "evicted": self._evicted,
            "gaps": [gap.describe() for gap in self._gaps],
            "capture_gaps": [gap.describe() for gap in self._capture_gaps],
            "by_source": by_source,
            "retention": self.budget.retention.as_dict(),
        }

    def describe(self) -> str:
        stats = self.stats()
        return (
            f"RingBuffer(retained={stats['retained']} bytes={stats['bytes']}"
            f"/{stats['ring_bytes']} evicted={stats['evicted']} "
            f"gaps={len(stats['gaps'])} capture_gaps={len(stats['capture_gaps'])})"
        )


def records_for(records: Iterable[FrameRecord], source_id: str) -> list[FrameRecord]:
    """Filter a sequence of records to one source, preserving order."""
    return [record for record in records if record.source_id == source_id]
