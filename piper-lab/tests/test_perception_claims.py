"""Claims, three-valued verdicts and verification windows.

A boolean ``verified`` flag cannot express which proposition was checked, how,
or until when. These tests pin the three consequences: unknown is a real
answer, verdicts are never inherited between predicates, and an expired verdict
demotes instead of silently passing.
"""
import pytest

from piperlab.perception.claims import (
    P_COMMAND_ACKNOWLEDGED,
    P_FRAME_AVAILABLE,
    P_TARGET_REACHED,
    Claim,
    ClaimError,
    ClaimStore,
    EvidenceRef,
    Predicate,
    Verdict,
)
from piperlab.perception.clockmap import ClockDomain, in_domain
from piperlab.perception.evidence import EvidenceResult, EvidenceStatus


def ref(sequence=0, source_s=100.0):
    return EvidenceRef("camera", sequence, source_s, "frame")


def test_a_decided_verdict_must_cite_evidence():
    with pytest.raises(ClaimError, match="evidence reference"):
        Claim(
            claim_id="clm-1",
            subject="gripper",
            predicate=P_TARGET_REACHED,
            verdict=Verdict.SUPPORTED,
            method="measured_versus_target",
        )


def test_unknown_needs_no_evidence_and_is_a_first_class_answer():
    store = ClaimStore()
    claim = store.unknown(
        "ent-0001", P_FRAME_AVAILABLE, method="buffer_lookup", reason="capacity_evicted"
    )
    assert claim.verdict is Verdict.UNKNOWN
    assert not claim.decided
    assert claim.detail["reason"] == "capacity_evicted"
    assert store.latest("ent-0001", P_FRAME_AVAILABLE) is claim
    assert store.standing(now=0.0) == []


def test_a_verdict_without_a_method_is_refused():
    with pytest.raises(ClaimError, match="method"):
        Claim(
            claim_id="clm-1", subject="s", predicate=P_FRAME_AVAILABLE,
            verdict=Verdict.UNKNOWN, method="",
        )


def test_frame_available_supported_only_for_an_actual_frame():
    store = ClaimStore()
    frame_like = type("Frame", (), {"source_id": "camera", "sequence": 3, "stamp": in_domain(
        100.3, 100.31, ClockDomain.ROS_WALL)})
    ok = EvidenceResult(
        status=EvidenceStatus.OK, source_id="camera", requested_source_s=100.3,
        frame=frame_like, offset_s=0.0, exact=True,
    )
    claim = store.verify_frame_available(subject="ent-0001", evidence=ok, now=100.4)
    assert claim.verdict is Verdict.SUPPORTED
    assert claim.evidence_refs[0].sequence == 3

    # Evicted and gap results must not become a negative claim.
    for status in (EvidenceStatus.EVICTED, EvidenceStatus.CAPTURE_GAP, EvidenceStatus.UNAVAILABLE):
        evidence = EvidenceResult(status=status, source_id="camera", requested_source_s=100.3)
        claim = store.verify_frame_available(subject="ent-0002", evidence=evidence, now=100.4)
        assert claim.verdict is Verdict.UNKNOWN
        assert claim.detail["reason"] == status.value


def test_state_freshness_is_decided_by_age_and_domain():
    store = ClaimStore()
    fresh = in_domain(100.0, 100.02, ClockDomain.ROS_WALL)
    claim = store.verify_state_fresh(
        subject="robot", stamp=fresh, now_monotonic_s=100.05, max_age_s=0.2, skew_s=0.05
    )
    assert claim.verdict is Verdict.SUPPORTED
    assert claim.valid_until == pytest.approx(100.05 + (0.2 - 0.05))

    stale = in_domain(100.0, 100.02, ClockDomain.ROS_WALL)
    claim = store.verify_state_fresh(
        subject="robot", stamp=stale, now_monotonic_s=100.90, max_age_s=0.2, skew_s=0.05
    )
    assert claim.verdict is Verdict.UNKNOWN
    assert "outside the contract window" in claim.detail["reason"]

    foreign = in_domain(100.0, 100.02, ClockDomain.CROSS_HOST_UNSYNCHRONIZED)
    claim = store.verify_state_fresh(
        subject="robot", stamp=foreign, now_monotonic_s=100.03, max_age_s=0.2, skew_s=0.05
    )
    assert claim.verdict is Verdict.UNKNOWN
    assert "not locally comparable" in claim.detail["reason"]


def test_acknowledgement_and_arrival_are_independent_claims():
    """Accepted is not completed; a measured shortfall must not erase the ack."""
    store = ClaimStore()
    ack, reached = store.verify_command_versus_target(
        subject="ent-0007",
        request_id="req-00012345",
        acknowledged=True,
        acknowledged_at=100.0,
        target=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.02],
        measured=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.05],
        tolerance=0.005,
        now=100.2,
    )
    assert ack.verdict is Verdict.SUPPORTED
    assert reached.verdict is Verdict.REFUTED
    assert reached.value["max_abs_error"] == pytest.approx(0.03)
    assert ack.predicate == P_COMMAND_ACKNOWLEDGED and reached.predicate == P_TARGET_REACHED
    assert "does not assert the command failed" in reached.detail["note"]
    assert "accepted is not completed" in ack.detail["note"]


def test_arrival_is_unknown_without_a_measurement_rather_than_assumed_true():
    store = ClaimStore()
    ack, reached = store.verify_command_versus_target(
        subject="ent-0007", request_id="req-1", acknowledged=True, acknowledged_at=100.0,
        target=[0.0, 0.02], measured=None, tolerance=0.001, now=100.1,
    )
    assert ack.verdict is Verdict.SUPPORTED
    assert reached.verdict is Verdict.UNKNOWN
    assert reached.detail["reason"] == "measured state or target is unavailable"


def test_verdicts_are_not_inherited_across_predicates():
    store = ClaimStore()
    store.verify_command_versus_target(
        subject="cup-17", request_id="req-2", acknowledged=True, acknowledged_at=100.0,
        target=[0.0, 0.02], measured=[0.0, 0.02], tolerance=0.001, now=100.1,
    )
    holding = Predicate("holding")
    assert store.latest("cup-17", holding) is None
    claim = store.unknown("cup-17", holding, method="not_evaluated", reason="no predicate implementation")
    assert claim.verdict is Verdict.UNKNOWN
    assert store.latest("cup-17", holding).verdict is Verdict.UNKNOWN


def test_expiry_demotes_a_supported_claim_instead_of_passing_it_silently():
    store = ClaimStore()
    claim = Claim(
        claim_id="clm-1", subject="robot", predicate=P_TARGET_REACHED,
        verdict=Verdict.SUPPORTED, method="measured_versus_target",
        evidence_refs=(ref(),), observed_at=100.0, verified_at=100.0, valid_until=100.5,
    )
    store.record(claim)
    assert claim.holds_now(100.4)
    assert not claim.holds_now(100.6)
    demoted = store.expire(now=100.6)
    assert [item.claim_id for item in demoted] == ["clm-1"]
    assert store.claims[0].verdict is Verdict.UNKNOWN
    assert store.claims[0].detail["expired_at"] == 100.6
    assert store.standing(100.6) == []
    assert store.describe(now=100.6)["standing"] == []
