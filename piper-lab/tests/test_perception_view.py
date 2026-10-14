"""State view versioning, validity windows and invalidation vocabulary."""
import pytest

from piperlab.perception.budget import Budget
from piperlab.perception.claims import ClaimStore, P_ENTITY_OBSERVED
from piperlab.perception.clockmap import ClockDomain, in_domain
from piperlab.perception.events import EventLog
from piperlab.perception.view import (
    FieldValue,
    StateStore,
    Staleness,
    ViewError,
    build_view,
)


def stamp(source_s=100.0, received_s=100.01, *, epoch=0, domain=ClockDomain.ROS_WALL):
    return in_domain(
        source_s=source_s, received_s=received_s, domain=domain, acquisition_epoch=epoch
    )


def test_versions_increase_and_the_previous_view_is_superseded_not_rewritten():
    store = StateStore()
    first = store.publish(now_source_s=100.0, now_monotonic_s=100.0, entity_ids=["ent-0000"])
    second = store.publish(now_source_s=100.1, now_monotonic_s=100.1, entity_ids=["ent-0000"])
    assert (first.version, second.version) == (1, 2)
    assert first.superseded_by == 2
    assert second.superseded_by is None
    replayed = store.replay(1)
    assert replayed.entity_ids == ("ent-0000",)
    assert replayed.superseded_by == 2
    assert store.snapshot()["history_versions"] == [1, 2]


def test_replay_unknown_version_is_refused():
    store = StateStore()
    store.publish(now_source_s=1.0, now_monotonic_s=1.0)
    with pytest.raises(ViewError, match="no state view with version"):
        store.replay(99)


def test_field_staleness_uses_the_contract_window_and_expiry():
    store = StateStore()
    view = store.publish(now_source_s=100.0, now_monotonic_s=100.0)
    view.put(FieldValue(name="robot_state", value=[0.0] * 7, stamp=stamp(100.0, 100.0)))
    view.put(
        FieldValue(name="entities", value={"visible": 1}, valid_until_monotonic_s=100.5)
    )
    assert view.staleness("robot_state", now_source_s=100.05, now_monotonic_s=100.05) is Staleness.FRESH
    assert view.staleness("robot_state", now_source_s=100.5, now_monotonic_s=100.5) is Staleness.STALE
    assert view.staleness("entities", now_source_s=100.2, now_monotonic_s=100.2) is Staleness.FRESH
    assert view.staleness("entities", now_source_s=100.9, now_monotonic_s=100.9) is Staleness.STALE
    assert view.staleness("missing", now_source_s=100.1, now_monotonic_s=100.1) is Staleness.UNKNOWN


def test_require_fresh_names_the_reason_and_the_offending_fields():
    store = StateStore()
    view = store.publish(now_source_s=100.0, now_monotonic_s=100.0)
    view.put(FieldValue(name="robot_state", value=[0.0] * 7, stamp=stamp(100.0, 100.0)))
    view.require_fresh(["robot_state"], now_source_s=100.05, now_monotonic_s=100.05)

    with pytest.raises(ViewError) as excinfo:
        view.require_fresh(["robot_state"], now_source_s=100.9, now_monotonic_s=100.9)
    assert "robot_state=stale" in str(excinfo.value)

    view.invalidate("stale_feedback", now_monotonic_s=100.95, detail={"age_s": 0.9})
    with pytest.raises(ViewError, match="stale_feedback"):
        view.require_fresh(["robot_state"], now_source_s=100.96, now_monotonic_s=100.96)
    assert view.staleness("robot_state", now_source_s=100.96, now_monotonic_s=100.96) is Staleness.INVALIDATED


def test_invalidation_reasons_come_from_the_shared_vocabulary():
    store = StateStore()
    store.publish(now_source_s=1.0, now_monotonic_s=1.0)
    with pytest.raises(ViewError, match="unknown invalidation code"):
        store.invalidate_current("because_i_said_so", now_monotonic_s=1.0)

    view = store.invalidate_current(
        "reconnect_required", now_monotonic_s=1.1, detail={"note": "executor restart"}
    )
    assert view.invalidated and view.invalidation_code == "reconnect_required"
    assert view.describe()["invalidation_detail"]["note"] == "executor restart"


def test_a_non_comparable_domain_makes_the_field_stale_not_fresh():
    store = StateStore()
    view = store.publish(now_source_s=100.0, now_monotonic_s=100.0)
    view.put(
        FieldValue(
            name="robot_state",
            value=[0.0] * 7,
            stamp=stamp(100.0, 100.0, domain=ClockDomain.CROSS_HOST_UNSYNCHRONIZED),
        )
    )
    assert view.staleness("robot_state", now_source_s=100.01, now_monotonic_s=100.01) is Staleness.STALE


def test_as_of_source_returns_what_the_fast_path_saw_at_that_time():
    store = StateStore()
    store.publish(now_source_s=100.0, now_monotonic_s=100.0)
    store.publish(now_source_s=100.5, now_monotonic_s=100.5)
    assert store.as_of_source(100.2).version == 1
    assert store.as_of_source(100.5).version == 2
    assert store.as_of_source(99.0) is None


def test_build_view_links_claims_and_events_and_marks_entity_staleness():
    store = StateStore()
    claims = ClaimStore()
    events = EventLog()
    claims.unknown("ent-0000", P_ENTITY_OBSERVED, method="tracker", reason="occluded")
    view = build_view(
        store=store,
        entity_summary={"visible": 0, "occluded": 1},
        entity_ids=["ent-0000"],
        evidence_sequences=[3],
        now_source_s=100.0,
        now_monotonic_s=100.0,
        claims=claims,
        events=events,
        valid_until_monotonic_s=100.4,
    )
    assert view.claim_ids == ("clm-00001",)
    assert view.version == 1
    assert view.field("entities").evidence_sequences == (3,)
    assert "unconfirmed, not absent" in view.field("entities").note


def test_history_growth_is_bounded_by_budget():
    store = StateStore(budget=Budget())
    for index in range(5):
        store.publish(now_source_s=float(index), now_monotonic_s=float(index))
    store.check_budget()
    assert len(store.history()) == 5
