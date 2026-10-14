"""Failure-injection replay: what the pipeline must say when things go wrong.

Each case asserts on the *reported* content, not on internal state, because the
value of this layer is that a downstream consumer can tell old from new, absent
from unexamined, and unknown from false. The last two tests pin the boundary
with the existing motion gate: perception reports, ``SafetyGate`` decides.
"""
import math
from pathlib import Path

import pytest

from piperlab.perception.budget import Budget, BudgetExceeded
from piperlab.perception.buffer import CaptureGap, FrameRecord, RingBuffer
from piperlab.perception.claims import (
    ClaimStore,
    P_COMMAND_ACKNOWLEDGED,
    P_STATE_FRESH,
    P_TARGET_REACHED,
    Predicate,
    Verdict,
)
from piperlab.perception.clockmap import ClockDomain, ClockError, ClockStamp, in_domain
from piperlab.perception.escalate import (
    DegradedPolicy,
    DegradedTier,
    Escalator,
    FakeSlowModel,
    TriggerKind,
    conflict_event,
)
from piperlab.perception.evidence import EvidenceStatus, resolve, resolve_stamp
from piperlab.perception.events import Event, EventKind, EventLog
from piperlab.perception.fuse import NO_CALIBRATION, project
from piperlab.perception.safety_critical import (
    ATTENTION_TUNABLE,
    SAFETY_CRITICAL,
    AttentionRejected,
    AttentionRequest,
    assert_critical_always_on,
    validate_attention,
)
from piperlab.perception.tracks import Entity, TrackId
from piperlab.perception.view import FieldValue, StateStore, Staleness
from piperlab.safety import Rejected, SafetyGate, load_config


def record(sequence, source_s, *, source_id="camera", epoch=0, payload=1000):
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


def entity(entity_id="ent-0000", *, state="tracked", label="block", centroid=(10.0, 10.0),
           last_sequence=0, last_source_s=100.0, evidence=(0,)):
    return Entity(
        entity_id=entity_id, label=label, track=TrackId("camera", 0, 0),
        first_source_s=100.0, last_source_s=last_source_s, state=state, hits=2,
        centroid_xy=centroid, bbox_xyxy=(0.0, 0.0, 20.0, 20.0), score=0.5,
        last_sequence=last_sequence, evidence_sequences=evidence,
    )


# --- 1. device restart: native time resets, old and new must not interleave ----
def test_device_restart_starts_a_new_epoch_and_never_interleaves():
    buf = RingBuffer(Budget(ring_bytes=10_000_000))
    for index in range(3):
        buf.push(record(index, 100.0 + index * 0.1, epoch=0))

    with pytest.raises(Exception) as excinfo:
        buf.push(record(3, 0.0, epoch=1))
    assert "epoch" in str(excinfo.value).lower()

    # A declared epoch boundary closes the old stream and lets the restarted
    # source begin again from zero.
    fresh = RingBuffer(Budget(ring_bytes=10_000_000))
    fresh.push(record(0, 100.0, epoch=0))
    gaps = fresh.begin_epoch(1)
    assert [gap.reason for gap in gaps] == ["acquisition_epoch_changed"]
    assert fresh.push(record(0, 0.0, epoch=1)).stamp.acquisition_epoch == 1
    assert [item.sequence for item in fresh.records(current_epoch_only=True)] == [0]
    # The superseded record stays available for audit but is out of the live view.
    assert [item.sequence for item in fresh.records()] == [0, 0]

    older = in_domain(100.0, 100.01, ClockDomain.ROS_WALL, acquisition_epoch=0)
    assert not older.valid(max_age_s=0.2, skew_s=0.05, now_monotonic_s=100.02, epoch=1)


# --- 2. a late branch result is marked late, not backdated --------------------
def test_late_result_is_a_belief_revision_and_never_a_fresh_world_change():
    log = EventLog()
    log.classify_and_append(previous_entities=[], current_entities=[entity()], source_s=100.0)
    late = Event(
        event_id="x", kind=EventKind.BELIEF_REVISION, source_s=100.0,
        summary="late reclassification arrived after the fact",
        entity_id="ent-0000", before="block", after="cup", late=True,
        revision="classification", evidence_sequences=(0,),
    )
    stored = log.append(late)
    assert stored.late is True
    assert stored.kind is EventKind.BELIEF_REVISION
    assert stored.asserts_world_state is False
    # The observation instant is preserved; the arrival time did not overwrite it.
    assert stored.source_s == 100.0
    assert log.describe()["world_claims"] == 1


# --- 3. raw frames removed by capacity: reported unavailable, never substituted
def test_removed_raw_evidence_reports_evicted_and_never_returns_a_neighbour():
    buf = RingBuffer(Budget(ring_bytes=2100, evidence_window_s=1.0))
    for index in range(4):
        buf.push(record(index, 100.0 + index * 0.1, payload=1000))
    result = resolve(buf, "camera", 100.0)
    assert result.status is EvidenceStatus.EVICTED
    assert result.frame is None
    assert result.reason.startswith("evidence existed and was removed")
    assert result.describe()["reason"].startswith("evidence existed")

    # A frame that was never analysed is a different answer again, and it is
    # explicitly recoverable.
    retained = list(buf.records())[0]
    unexamined = resolve(buf, "camera", retained.stamp.source_s, analyzed_sequences=frozenset())
    assert unexamined.status is EvidenceStatus.NOT_SELECTED
    assert unexamined.recoverable


# --- 4. occlusion: unknown, never "the object is gone" ------------------------
def test_occlusion_becomes_a_gap_and_does_not_assert_disappearance():
    log = EventLog()
    log.classify_and_append(previous_entities=[], current_entities=[entity()], source_s=100.0)
    produced = log.classify_and_append(
        previous_entities=[entity(state="tracked")],
        current_entities=[entity(state="occluded", last_sequence=1, evidence=(0, 1))],
        source_s=100.1,
    )
    assert [event.kind for event in produced] == [EventKind.OBSERVATION_GAP]
    assert all(not event.asserts_world_state for event in produced)
    described = log.describe()
    assert described["by_kind"]["world_change"] == 1  # only the initial sighting
    assert "not asserted either way" in produced[0].summary

    # Identity count unchanged => no disappearance claim is produced.
    assert "presence" not in [event.field_name for event in produced]


# --- 5. accepted but not arrived: two independent claims -----------------------
def test_accepted_command_and_unreached_target_coexist_as_separate_claims():
    store = ClaimStore()
    ack, reached = store.verify_command_versus_target(
        subject="ent-0007", request_id="place-001", acknowledged=True, acknowledged_at=100.0,
        target=[0.0] * 6 + [0.02], measured=[0.0] * 6 + [0.06], tolerance=0.005, now=100.2,
    )
    assert ack.predicate is P_COMMAND_ACKNOWLEDGED and ack.verdict is Verdict.SUPPORTED
    assert reached.predicate is P_TARGET_REACHED and reached.verdict is Verdict.REFUTED
    described = store.describe()["by_verdict"]
    assert described["supported"] == 1 and described["refuted"] == 1
    assert "accepted is not completed" in ack.detail["note"]


# --- 6. slow model blocked: predefined degradation, fast path keeps its rules --
def test_slow_model_blocked_follows_predefined_tiers():
    policy = DegradedPolicy(hold_after_s=0.15, freeze_after_s=1.0, observe_only_after_s=5.0)
    escalator = Escalator(budget=Budget(), policy=policy, slow_model=FakeSlowModel())

    assert policy.tier_for(0.05) is DegradedTier.NOMINAL
    assert policy.tier_for(0.5) is DegradedTier.HOLD
    assert policy.tier_for(2.0) is DegradedTier.FREEZE
    assert policy.tier_for(30.0) is DegradedTier.OBSERVE_ONLY
    assert policy.may_issue_actions(DegradedTier.NOMINAL)
    assert not policy.may_issue_actions(DegradedTier.HOLD)
    assert not policy.may_issue_actions(DegradedTier.OBSERVE_ONLY)

    # No slow response has ever arrived: the cautious tier, not nominal.
    assert escalator.tier(
        now_monotonic_s=100.0, last_slow_response_monotonic_s=None
    ) is DegradedTier.OBSERVE_ONLY
    assert not escalator.may_issue_actions(
        now_monotonic_s=100.0, last_slow_response_monotonic_s=None
    )

    # A blocked slow model must not become an escalation storm.
    first = escalator.consider(
        subject="ent-0007", trigger=TriggerKind.WORLD_CHANGE, now_source_s=100.0,
        summary="object moved",
    )
    assert first is not None
    suppressed = escalator.consider(
        subject="ent-0007", trigger=TriggerKind.WORLD_CHANGE, now_source_s=100.1,
        summary="object moved again",
    )
    assert suppressed is None
    later = escalator.consider(
        subject="ent-0007", trigger=TriggerKind.WORLD_CHANGE, now_source_s=101.0,
        summary="object moved again",
    )
    assert later is not None
    assert escalator.heartbeat_due(now_source_s=200.0)


# --- 7. model asserts success, measurement disagrees ---------------------------
def test_model_success_assertion_is_not_upgraded_when_measurement_disagrees():
    store = ClaimStore()
    holding = Predicate("holding")
    claim = store.unknown(
        "cup-17", holding, method="slow_model_assertion",
        reason="model reported success but no measurement supports it",
    )
    assert claim.verdict is Verdict.UNKNOWN

    event = conflict_event(
        subject="cup-17", source_s=100.5, slow_claim="holding", slow_verdict="supported",
        measured="gripper open, object on table", event_id="x", evidence_sequences=[3],
    )
    assert event.kind is EventKind.CONFLICT
    assert event.after == "unknown"
    assert event.revision == "conflict_with_measurement"
    assert not event.asserts_world_state
    assert store.latest("cup-17", holding).verdict is Verdict.UNKNOWN


# --- 8. clock domain that cannot be related locally ---------------------------
def test_unestablished_clock_domain_refuses_to_locate_evidence():
    buf = RingBuffer(Budget(ring_bytes=10_000_000))
    buf.push(record(0, 100.0))
    foreign = in_domain(
        source_s=100.0, received_s=100.01, domain=ClockDomain.CROSS_HOST_UNSYNCHRONIZED
    )
    result = resolve_stamp(buf, foreign)
    assert result.status is EvidenceStatus.UNAVAILABLE
    assert result.frame is None
    assert "no established relation to local clocks" in result.reason

    store = ClaimStore()
    claim = store.verify_state_fresh(
        subject="robot", stamp=foreign, now_monotonic_s=100.02, max_age_s=0.2, skew_s=0.05
    )
    assert claim.verdict is Verdict.UNKNOWN


# --- 9. over budget: raise, never truncate -----------------------------------
def test_over_budget_work_raises_and_never_truncates():
    buf = RingBuffer(Budget(ring_bytes=10_000_000, max_evidence_frames_per_query=2))
    for index in range(10):
        buf.push(record(index, 100.0 + index * 0.1))
    with pytest.raises(BudgetExceeded) as excinfo:
        resolve(buf, "camera", 100.45, tolerance_s=5.0)
    report = excinfo.value.report()
    assert report["scope"] == "max_evidence_frames_per_query"
    assert report["observed"] > report["limit"]
    assert "truncated" in str(excinfo.value)


def test_a_payload_that_can_never_fit_is_refused_before_evicting_anything():
    buf = RingBuffer(Budget(ring_bytes=2000))
    buf.push(record(0, 100.0, payload=1000))
    with pytest.raises(BudgetExceeded) as excinfo:
        buf.push(record(1, 100.1, payload=5000))
    assert excinfo.value.report()["observed"] == 5000
    assert [item.sequence for item in buf.records()] == [0]


# --- 10. attention may change focus, never safety -----------------------------
def test_attention_cannot_touch_safety_critical_analyses():
    request = AttentionRequest(
        request_id="att-1", entities=("ent-0007",), changes=("relative_motion", "contact_loss"),
        scope="grasp_phase", issued_at_source_s=100.0, valid_until_source_s=110.0,
    )
    effect = validate_attention(
        request, enable=("object_detect",), threshold_overrides={"object_detect": 0.02},
        max_extra_frames=4,
    )
    assert effect.enabled_analyses == ("object_detect",)
    assert effect.max_extra_frames == 4
    assert effect.expires_at_source_s == 110.0
    assert "frame_liveness" in effect.describe()["cannot_touch"]

    for critical in sorted(SAFETY_CRITICAL):
        with pytest.raises(AttentionRejected, match="safety-critical"):
            validate_attention(request, disable=(critical,))
        with pytest.raises(AttentionRejected, match="safety-critical"):
            validate_attention(request, threshold_overrides={critical: 0.0})

    # Attention cannot switch an analysis off by pushing its threshold away.
    with pytest.raises(AttentionRejected, match="must stay positive"):
        validate_attention(request, threshold_overrides={"object_detect": 0.0})
    with pytest.raises(AttentionRejected, match="unknown analysis"):
        validate_attention(request, enable=("brand_new_thing",))

    # An unbounded attention grant is not a request.
    with pytest.raises(AttentionRejected, match="expire"):
        AttentionRequest(
            request_id="att-2", entities=("e",), changes=(), scope="s",
            issued_at_source_s=1.0, valid_until_source_s=1.0,
        )

    assert_critical_always_on(SAFETY_CRITICAL | ATTENTION_TUNABLE)
    with pytest.raises(AttentionRejected, match="not active"):
        assert_critical_always_on({"object_detect"})


# --- explicit boundaries -----------------------------------------------------
def test_a_source_stamp_is_never_replaced_by_the_local_clock():
    with pytest.raises(ClockError):
        in_domain(source_s=float("nan"), received_s=100.0, domain=ClockDomain.ROS_WALL)
    missing = ClockStamp(
        source_s=0.0, received_s=100.0, domain=ClockDomain.CROSS_HOST_UNSYNCHRONIZED,
        mapping_id="unavailable", acquisition_epoch=0,
    )
    assert not missing.valid(max_age_s=0.2, skew_s=0.05, now_monotonic_s=100.01)


def test_perception_reports_do_not_grant_permission_the_gate_owns():
    """Perception reports; SafetyGate decides. Neither the gate's state nor its
    admission decisions may be influenced by the perception layer."""
    config = load_config(Path(__file__).resolve().parents[1] / "config/hardware.yaml")
    gate = SafetyGate(config, clock=lambda: 100.0, wall_clock=lambda: 100.0)

    # A stale camera makes the gate refuse to reset, regardless of what a state
    # view or claim store believes.
    gate.observe(gate.names, [0, 1, -1, 0, 0, 0, 0.05], 100.0)
    assert not gate.observe_camera(90.0)
    with pytest.raises(Rejected, match="Fresh feedback and camera"):
        gate.control("reset", "policy")

    store = ClaimStore()
    view_store = StateStore()
    view = view_store.publish(now_source_s=100.0, now_monotonic_s=100.0)
    view.put(FieldValue(name="entities", value={"visible": 1}, valid_until_monotonic_s=200.0))
    assert view.staleness("entities", now_source_s=100.0, now_monotonic_s=100.0) is Staleness.FRESH
    store.unknown(
        "robot", P_STATE_FRESH, method="test", reason="deliberately left unknown"
    )
    # The gate is still closed: no perception object can open it.
    assert not gate.armed
    with pytest.raises(Rejected, match="Motion is disabled"):
        gate.admit("policy", "", 0, gate.names, [0] * 7, 100.0)


def test_importing_perception_does_not_import_torch_or_lerobot():
    import importlib
    import sys as _sys

    importlib.import_module("piperlab.perception")
    assert "torch" not in _sys.modules
    assert "lerobot" not in _sys.modules


def test_missing_calibration_is_reported_not_guessed():
    projection = project((320.0, 240.0), calibration=None)
    assert projection.available is False
    assert projection.reason == NO_CALIBRATION
    assert projection.base_xyz_m is None
    assert "no base_link-from-camera transform" in projection.detail["hint"]
