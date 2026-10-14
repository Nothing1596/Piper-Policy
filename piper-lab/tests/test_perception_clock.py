"""Clock provenance: domains, ages, uncertainty and fail-closed validity.

These tests exist because "we recorded two timestamps" is routinely mistaken for
"the two streams are synchronised". The layer must instead be able to say *which
clock domain* a value came from, how stale it is, how uncertain the mapping is,
and refuse to answer when those are not good enough.
"""
import pytest

from piperlab.perception.clockmap import (
    IDENTITY,
    ClockDomain,
    ClockError,
    ClockStamp,
    INVALIDATION_CODES,
    Mapping,
    from_mapping,
    in_domain,
    is_known_invalidation,
)


def test_same_domain_stamp_age_and_validity():
    stamp = in_domain(source_s=100.0, received_s=100.05, domain=ClockDomain.ROS_WALL)
    assert stamp.comparable
    assert stamp.age_s == pytest.approx(0.05)
    assert stamp.valid(max_age_s=0.2, skew_s=0.05, now_monotonic_s=100.06)
    assert not stamp.valid(max_age_s=0.01, skew_s=0.05, now_monotonic_s=100.06)


def test_reading_a_frame_later_never_makes_it_fresher():
    stamp = in_domain(source_s=100.0, received_s=100.04, domain=ClockDomain.ROS_WALL)
    assert stamp.age_at(100.04) == pytest.approx(0.04)
    assert stamp.age_at(101.00) == pytest.approx(1.00)
    # The floor is the original transit delay: re-reading cannot reduce age.
    assert stamp.age_at(100.01) == pytest.approx(0.04)


def test_future_dated_stamp_within_skew_is_accepted_and_beyond_is_not():
    stamp = in_domain(source_s=100.30, received_s=100.25, domain=ClockDomain.ROS_WALL)
    assert stamp.age_s == pytest.approx(-0.05)
    assert stamp.valid(max_age_s=0.2, skew_s=0.05, now_monotonic_s=100.26)
    assert not stamp.valid(max_age_s=0.2, skew_s=0.01, now_monotonic_s=100.26)


def test_cross_host_domain_is_never_treated_as_comparable():
    stamp = in_domain(
        source_s=100.0, received_s=100.05, domain=ClockDomain.CROSS_HOST_UNSYNCHRONIZED
    )
    assert not stamp.comparable
    assert not stamp.valid(max_age_s=0.2, skew_s=0.05, now_monotonic_s=100.06)
    assert stamp.describe()["domain"] == "cross_host_unsynchronized"


def test_epoch_mismatch_invalidates_the_stamp():
    stamp = in_domain(
        source_s=100.0, received_s=100.02, domain=ClockDomain.ROS_WALL, acquisition_epoch=2
    )
    assert stamp.valid(max_age_s=0.2, skew_s=0.05, now_monotonic_s=100.03, epoch=2)
    assert not stamp.valid(max_age_s=0.2, skew_s=0.05, now_monotonic_s=100.03, epoch=1)


def test_uncertainty_wide_enough_to_swamp_the_window_fails_closed():
    stamp = in_domain(
        source_s=100.0,
        received_s=100.05,
        domain=ClockDomain.ROS_WALL,
        uncertainty_s=0.30,
    )
    assert stamp.uncertainty_s == pytest.approx(0.30)
    assert not stamp.valid(max_age_s=0.2, skew_s=0.05, now_monotonic_s=100.06)


def test_missing_mapping_id_is_refused():
    with pytest.raises(ClockError):
        ClockStamp(
            source_s=1.0,
            received_s=1.0,
            domain=ClockDomain.ROS_WALL,
            mapping_id="",
            acquisition_epoch=0,
        )


def test_mapping_must_be_identified_and_is_applied_explicitly():
    mapping = Mapping(mapping_id="exec-host-offset-v1", offset_s=0.25, uncertainty_s=0.01)
    assert not mapping.identity
    stamp = from_mapping(100.0, 100.3, mapping, ClockDomain.HOST_REALTIME, acquisition_epoch=1)
    assert stamp.source_s == pytest.approx(100.25)
    assert stamp.mapping_id == "exec-host-offset-v1"
    assert stamp.mapping_uncertainty_s == pytest.approx(0.01)
    assert IDENTITY.identity

    with pytest.raises(ClockError):
        Mapping(mapping_id="bad", scale=0.0)
    with pytest.raises(ClockError):
        Mapping(mapping_id="bad", uncertainty_s=-1.0)


def test_non_finite_inputs_are_refused():
    with pytest.raises(ClockError):
        in_domain(source_s=float("nan"), received_s=1.0, domain=ClockDomain.ROS_WALL)
    with pytest.raises(ClockError):
        in_domain(source_s=1.0, received_s=1.0, domain=ClockDomain.ROS_WALL).valid(
            max_age_s=0.2, skew_s=0.05, now_monotonic_s=float("inf")
        )


def test_invalidation_vocabulary_is_shared_and_closed():
    # A stale view must cite an executor-level reason, not a home-made boolean.
    assert "stale_feedback" in INVALIDATION_CODES
    assert "capacity_evicted" in INVALIDATION_CODES
    assert is_known_invalidation("reconnect_required")
    assert not is_known_invalidation("looks_fine_to_me")
