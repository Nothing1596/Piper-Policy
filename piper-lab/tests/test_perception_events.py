"""Event classification, and the world/belief boundary.

The failure this guards against is a pipeline narrating its own bookkeeping as
physical reality: a lost track becoming "the object vanished", a relabel
becoming a change in the world, an identity merge becoming two objects
disappearing.
"""
import pytest

from piperlab.perception.buffer import CaptureGap
from piperlab.perception.events import (
    Event,
    EventError,
    EventKind,
    EventLog,
    classify,
    identity_merge_event,
)
from piperlab.perception.tracks import Entity, TrackId


def entity(entity_id="ent-0000", *, label="block", state="tracked", centroid=(10.0, 10.0),
           last_sequence=0, last_source_s=100.0, score=0.5, evidence=(0,)):
    return Entity(
        entity_id=entity_id,
        label=label,
        track=TrackId("camera", 0, 0),
        first_source_s=100.0,
        last_source_s=last_source_s,
        state=state,
        hits=2,
        centroid_xy=centroid,
        bbox_xyxy=(0.0, 0.0, 20.0, 20.0),
        score=score,
        last_sequence=last_sequence,
        evidence_sequences=evidence,
    )


def kinds(events):
    return [event.kind for event in events]


def test_first_observation_is_a_world_change_with_evidence():
    events = classify(previous_entities=[], current_entities=[entity()], source_s=100.0)
    assert kinds(events) == [EventKind.WORLD_CHANGE]
    assert events[0].evidence_sequences == (0,)
    assert events[0].asserts_world_state


def test_world_change_without_evidence_is_refused_by_construction():
    with pytest.raises(EventError, match="evidence"):
        Event(
            event_id="evt-1",
            kind=EventKind.WORLD_CHANGE,
            source_s=1.0,
            summary="something happened",
        )


def test_tracker_loss_yields_observation_gap_never_a_disappearance():
    before = entity(state="tracked")
    after = entity(state="occluded", last_sequence=1, evidence=(0, 1))
    events = classify(previous_entities=[before], current_entities=[after], source_s=100.1)
    assert kinds(events) == [EventKind.OBSERVATION_GAP]
    assert not events[0].asserts_world_state
    assert "not asserted either way" in events[0].summary


def test_entity_that_drops_out_while_unconfirmed_is_a_gap_not_a_removal():
    events = classify(
        previous_entities=[entity(state="lost")], current_entities=[], source_s=100.2
    )
    assert kinds(events) == [EventKind.OBSERVATION_GAP]
    assert "no existence claim" in events[0].summary


def test_entity_that_drops_out_while_visible_is_a_flagged_disappearance_claim():
    events = classify(
        previous_entities=[entity(state="tracked")], current_entities=[], source_s=100.2
    )
    assert kinds(events) == [EventKind.WORLD_CHANGE]
    assert events[0].field_name == "presence"
    assert "verify before treating this as an object removal" in events[0].detail["caution"]


def test_relabel_is_a_belief_revision_not_a_world_change():
    before = entity(label="block")
    after = entity(label="cup", last_sequence=1, evidence=(0, 1))
    events = classify(previous_entities=[before], current_entities=[after], source_s=100.1)
    assert kinds(events) == [EventKind.BELIEF_REVISION]
    assert events[0].revision == "classification"
    assert events[0].before == "block" and events[0].after == "cup"


def test_reacquisition_is_a_belief_revision_about_continuity():
    before = entity(state="occluded")
    after = entity(state="reacquired", last_sequence=2, evidence=(0, 2))
    events = classify(previous_entities=[before], current_entities=[after], source_s=100.2)
    assert kinds(events) == [EventKind.BELIEF_REVISION]
    assert events[0].revision == "reacquired"
    assert "not re-created" in events[0].summary


def test_movement_produces_a_world_change_with_before_and_after():
    before = entity(centroid=(10.0, 10.0))
    after = entity(centroid=(13.0, 10.0), last_sequence=1, evidence=(0, 1))
    events = classify(previous_entities=[before], current_entities=[after], source_s=100.1)
    assert kinds(events) == [EventKind.WORLD_CHANGE]
    assert events[0].field_name == "centroid_xy"
    assert events[0].after == [13.0, 10.0]


def test_sub_pixel_jitter_is_not_reported_as_movement():
    before = entity(centroid=(10.0, 10.0))
    after = entity(centroid=(10.001, 10.0), last_sequence=1, evidence=(0, 1))
    assert classify(previous_entities=[before], current_entities=[after], source_s=100.1) == []


def test_identity_merge_reduces_the_count_without_anything_disappearing():
    event = identity_merge_event(
        entity_ids=["ent-0001", "ent-0002"],
        surviving_entity_id="ent-0001",
        association_ids=["assoc-0000"],
        source_s=100.5,
        confidence=0.8,
        event_id="evt-ignored",
        evidence_sequences=[1, 2],
    )
    assert event.kind is EventKind.BELIEF_REVISION
    assert event.revision == "identity_merge"
    assert not event.asserts_world_state
    assert event.before == ["ent-0001", "ent-0002"]
    assert "no object disappeared" in event.summary

    with pytest.raises(EventError):
        identity_merge_event(
            entity_ids=["ent-0001"], surviving_entity_id="ent-0009",
            association_ids=[], source_s=1.0, confidence=0.5, event_id="x",
        )


def test_a_capture_gap_is_reported_as_unknown_not_as_unchanged():
    gap = CaptureGap("camera", 100.0, 100.5, reason="usb_reconnect")
    events = classify(
        previous_entities=[], current_entities=[entity()], source_s=100.6, capture_gaps=[gap]
    )
    assert EventKind.OBSERVATION_GAP in kinds(events)
    gap_event = next(event for event in events if event.kind is EventKind.OBSERVATION_GAP)
    assert gap_event.evidence_gaps == ("usb_reconnect",)
    assert "not the same as unchanged" in gap_event.summary


def test_event_log_numbers_events_and_separates_world_claims():
    log = EventLog()
    log.classify_and_append(previous_entities=[], current_entities=[entity()], source_s=100.0)
    log.classify_and_append(
        previous_entities=[entity(state="tracked")],
        current_entities=[entity(state="occluded", last_sequence=1, evidence=(0, 1))],
        source_s=100.1,
    )
    described = log.describe()
    assert described["count"] == 2
    assert described["by_kind"] == {"world_change": 1, "observation_gap": 1}
    assert described["world_claims"] == 1
    assert [event.event_id for event in log.events] == ["evt-00001", "evt-00002"]
    assert log.by_kind(EventKind.OBSERVATION_GAP)[0].asserts_world_state is False


def test_appending_an_event_with_a_foreign_id_renumbers_it_instead_of_duplicating():
    log = EventLog()
    event = Event(
        event_id="something-else", kind=EventKind.BELIEF_REVISION, source_s=1.0,
        summary="revision",
    )
    stored = log.append(event)
    assert stored.event_id == "evt-00001"
    assert log.events[0].event_id == "evt-00001"
