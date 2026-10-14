"""Buffering, gaps and evidence retrieval.

The central distinction under test is between three ways a frame can be absent:

- ``not_selected``  — it exists, this branch never analysed it, recoverable;
- ``capture_gap``   — nothing was ever captured there, a knowledge gap;
- ``evicted``       — it existed and was removed, irrecoverable.

Collapsing these into one "missing" flag is how a system reports "we never
looked" as "we looked and found nothing".
"""
import pytest

from piperlab.perception.budget import Budget, BudgetExceeded
from piperlab.perception.buffer import BufferError, CaptureGap, FrameRecord, RingBuffer
from piperlab.perception.clockmap import ClockDomain, in_domain
from piperlab.perception.evidence import EvidenceStatus, resolve, resolve_stamp


def record(sequence, source_s, *, payload=1000, source_id="camera", epoch=0):
    return FrameRecord(
        source_id=source_id,
        sequence=sequence,
        stamp=in_domain(
            source_s=source_s, received_s=source_s + 0.01,
            domain=ClockDomain.ROS_WALL, acquisition_epoch=epoch,
        ),
        payload_bytes=payload,
        l1_ref=f"seg/0000/{sequence}",
    )


def buffer_with(count=5, *, ring_bytes=10_000_000):
    buf = RingBuffer(Budget(ring_bytes=ring_bytes))
    for index in range(count):
        buf.push(record(index, 100.0 + index * 0.1))
    return buf


def test_push_requires_strictly_increasing_source_time_per_source():
    buf = RingBuffer(Budget())
    buf.push(record(0, 100.0))
    with pytest.raises(BufferError):
        buf.push(record(1, 100.0))
    with pytest.raises(BufferError):
        buf.push(record(0, 100.5))


def test_interleaving_across_an_acquisition_epoch_is_refused():
    buf = RingBuffer(Budget())
    buf.push(record(0, 100.0, epoch=0))
    with pytest.raises(BufferError, match="declared epoch"):
        buf.push(record(1, 100.1, epoch=1))
    # Declaring the boundary explicitly is the supported path.
    buf.begin_epoch(1)
    assert buf.push(record(0, 0.0, epoch=1)).stamp.acquisition_epoch == 1


def test_payload_length_must_match_declared_accounting():
    with pytest.raises(BufferError):
        FrameRecord(
            source_id="camera",
            sequence=0,
            stamp=in_domain(100.0, 100.01, ClockDomain.ROS_WALL),
            payload_bytes=10,
            payload=b"short",
        )


def test_capacity_eviction_is_recorded_as_a_gap_not_a_silent_drop():
    buf = RingBuffer(Budget(ring_bytes=2500))
    for index in range(5):
        buf.push(record(index, 100.0 + index * 0.1, payload=1000))
    stats = buf.stats()
    assert stats["bytes"] <= 2500
    assert stats["evicted"] >= 1
    gaps = buf.gaps()
    assert gaps and gaps[0].reason == "capacity_evicted"
    assert gaps[0].evicted_bytes > 0


def test_evicted_evidence_is_reported_as_evicted_and_never_substituted():
    buf = RingBuffer(Budget(ring_bytes=2500, evidence_window_s=1.0))
    for index in range(5):
        buf.push(record(index, 100.0 + index * 0.1, payload=1000))
    result = resolve(buf, "camera", 100.0)
    assert result.status is EvidenceStatus.EVICTED
    assert result.frame is None
    assert result.reason.startswith("evidence existed and was removed")
    assert result.gap is not None and result.gap.reason == "capacity_evicted"
    assert result.describe()["status"] == "evicted"
    assert not result.to_claim_verdict()


def test_capture_gap_is_distinct_from_eviction_and_from_absence():
    buf = buffer_with(3)
    buf.push_capture_gap(
        CaptureGap("camera", 500.0, 500.5, reason="usb_reconnect", detail={"attempts": 2})
    )
    result = resolve(buf, "camera", 500.25)
    assert result.status is EvidenceStatus.CAPTURE_GAP
    assert result.capture_gap is not None
    assert result.capture_gap.reason == "usb_reconnect"
    assert "never captured" in result.reason or "no data was ever captured" in result.reason
    # A gap is not a negative observation.
    assert not result.to_claim_verdict()

    outside = resolve(buf, "camera", 9999.0)
    assert outside.status is EvidenceStatus.UNAVAILABLE
    assert outside.describe()["detail"]["span"]["retained"] == 3


def test_not_selected_means_the_frame_exists_and_is_recoverable():
    buf = buffer_with(3)
    result = resolve(buf, "camera", 100.1, analyzed_sequences=frozenset({0, 2}))
    assert result.status is EvidenceStatus.NOT_SELECTED
    assert result.frame is not None and result.frame.sequence == 1
    assert result.recoverable
    assert "has not run on it" in result.reason


def test_exact_hit_is_marked_exact_and_inexact_hit_carries_its_offset():
    buf = buffer_with(5, ring_bytes=10_000_000)
    exact = resolve(buf, "camera", 100.2, tolerance_s=0.05)
    assert exact.status is EvidenceStatus.OK and exact.exact
    assert exact.offset_s == pytest.approx(0.0)

    inexact = resolve(buf, "camera", 100.26, tolerance_s=0.2)
    assert inexact.status is EvidenceStatus.OK
    assert not inexact.exact
    # Nearest retained frame is 100.3, so the offset is measured against the
    # request, and the result is explicitly not an exact hit.
    assert inexact.offset_s == pytest.approx(100.3 - 100.26)
    assert "not an exact hit" in inexact.reason


def test_evidence_query_over_budget_raises_instead_of_truncating():
    buf = buffer_with(10, ring_bytes=10_000_000)
    with pytest.raises(BudgetExceeded) as excinfo:
        resolve(buf, "camera", 100.45, tolerance_s=5.0,
                budget=Budget(max_evidence_frames_per_query=3))
    assert excinfo.value.report()["scope"] == "max_evidence_frames_per_query"


def test_protected_records_are_never_evicted_and_the_push_is_refused_cleanly():
    """A protected frame forces a refusal rather than a silent eviction.

    Eviction removes the oldest unpinned record, which for a protected payload
    would be the observation being recorded right now. The buffer must refuse
    instead of reporting success while dropping what it was handed.
    """
    buf = RingBuffer(Budget(ring_bytes=2200))
    buf.push(record(0, 100.0, payload=1000))
    buf.push(record(1, 100.1, payload=1000))
    buf.pin("camera", 0, until_source_s=200.0)
    buf.pin("camera", 1, until_source_s=200.0)

    with pytest.raises(BudgetExceeded) as excinfo:
        buf.push(record(2, 100.2, payload=1000), pin=True)
    assert excinfo.value.report()["scope"] == "ring_bytes_pinned"
    # The refused frame is not retained and no stale promise is left behind.
    assert [item.sequence for item in buf.records()] == [0, 1]
    assert ("camera", 2) not in buf._pinned

    # Once the pins expire, the same push fits.
    buf.expire_pins(now_source_s=250.0)
    assert [item.sequence for item in buf.records()] == [0, 1]
    buf.push(record(2, 100.2, payload=1000), pin=True)
    retained = [item.sequence for item in buf.records()]
    assert 2 in retained
    assert len(retained) <= 2


def test_fully_pinned_buffer_refuses_rather_than_breaking_a_promise():
    """With every retained record pinned, the buffer refuses instead of lying."""
    buf = RingBuffer(Budget(ring_bytes=2100))
    buf.push(record(0, 100.0, payload=1000))
    buf.push(record(1, 100.1, payload=1000))
    buf.pin("camera", 0, until_source_s=1e9)
    buf.pin("camera", 1, until_source_s=1e9)
    with pytest.raises(BudgetExceeded) as excinfo:
        buf.push(record(2, 100.2, payload=500), pin=True)
    assert excinfo.value.report()["scope"] == "ring_bytes_pinned"
    assert [item.sequence for item in buf.records()] == [0, 1]


def test_retention_expiry_reports_which_tier_removed_the_data():
    buf = RingBuffer(Budget(ring_bytes=10_000_000))
    for index in range(4):
        buf.push(record(index, 100.0 + index, payload=10))
    removed = buf.expire_retention(now_source_s=200.0, tier="l1")
    assert removed and all(gap.reason == "retention_l1" for gap in removed)
    assert not list(buf.records())


def test_a_single_payload_larger_than_the_ring_is_refused_before_evicting_anything():
    buf = RingBuffer(Budget(ring_bytes=2000))
    buf.push(record(0, 100.0, payload=1000))
    with pytest.raises(BudgetExceeded) as excinfo:
        buf.push(record(1, 100.1, payload=5000))
    assert excinfo.value.report()["observed"] == 5000
    # The earlier record is untouched: we did not clear the buffer to fit.
    assert list(buf.records())[0].sequence == 0


def test_resolve_stamp_refuses_a_non_comparable_domain():
    buf = buffer_with(3)
    stamp = in_domain(
        source_s=100.1, received_s=100.11, domain=ClockDomain.CROSS_HOST_UNSYNCHRONIZED
    )
    result = resolve_stamp(buf, stamp)
    assert result.status is EvidenceStatus.UNAVAILABLE
    assert "no established relation to local clocks" in result.reason


def test_stats_expose_every_source_and_the_retention_policy():
    buf = buffer_with(3)
    buf.push(record(0, 100.0, source_id="wrist"))
    stats = buf.stats()
    assert set(stats["by_source"]) == {"camera", "wrist"}
    assert stats["by_source"]["camera"]["retained"] == 3
    assert stats["retention"]["l1_s"] == 20.0
    assert "RingBuffer" in buf.describe()
