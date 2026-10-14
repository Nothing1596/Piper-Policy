"""Scorers, selection and the selection report.

Two properties matter here: a scorer can never delete raw data, and every
exclusion is reported with a reason that says whether the data still exists.
"""
import numpy as np
import pytest

from piperlab.perception.budget import Budget
from piperlab.perception.detect import (
    ColorBlobScorer,
    Detection,
    DetectionError,
    MotionEnergyScorer,
    _label_components,
)
from piperlab.perception.select import (
    DropReason,
    SelectionError,
    SelectionPolicy,
    select_detections,
)


def image_with_square(color, *, size=(64, 64), box=(10, 10, 30, 30), background=(0, 0, 0)):
    frame = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    frame[:, :] = background
    x0, y0, x1, y1 = box
    frame[y0:y1, x0:x1] = color
    return frame


def detection(sequence, score, *, source_id="camera", label="block", centroid=(20.0, 20.0)):
    return Detection(
        source_id=source_id,
        sequence=sequence,
        source_s=100.0 + sequence * 0.1,
        scorer="color_blob",
        label=label,
        score=score,
        bbox_xyxy=(10.0, 10.0, 30.0, 30.0),
        centroid_xy=centroid,
        area_px=400.0,
    )


def test_color_blob_scorer_finds_a_red_square_and_reports_geometry():
    scorer = ColorBlobScorer((255, 0, 0), tolerance=0.2, min_area_px=16)
    found = scorer.score(image_with_square((255, 0, 0)), source_id="camera", sequence=0, source_s=100.0)
    assert len(found) == 1
    blob = found[0]
    assert blob.label == "blob"
    assert blob.bbox_xyxy == (10.0, 10.0, 29.0, 29.0)
    assert blob.centroid_xy == pytest.approx((19.5, 19.5))
    assert 0.0 <= blob.score <= 1.0
    assert blob.entity_hint == "color_blob:blob"


def test_color_blob_scorer_ignores_unrelated_colour():
    scorer = ColorBlobScorer((255, 0, 0), tolerance=0.05)
    assert scorer.score(image_with_square((0, 0, 255)), source_id="camera", sequence=0, source_s=100.0) == []


def test_scorer_input_contract_is_enforced_not_coerced():
    scorer = ColorBlobScorer((255, 0, 0))
    with pytest.raises(DetectionError):
        scorer.score(np.zeros((8, 8), dtype=np.uint8), source_id="c", sequence=0, source_s=1.0)
    with pytest.raises(DetectionError):
        scorer.score(np.zeros((8, 8, 3), dtype=np.float32), source_id="c", sequence=0, source_s=1.0)


def test_motion_scorer_needs_a_previous_frame_and_then_reports_change():
    scorer = MotionEnergyScorer(threshold=0.1, min_area_px=16)
    first = scorer.score(image_with_square((0, 0, 0)), source_id="camera", sequence=0, source_s=100.0)
    assert first == []
    moved = image_with_square((255, 255, 255), box=(40, 40, 60, 60))
    second = scorer.score(moved, source_id="camera", sequence=1, source_s=100.1)
    assert second and second[0].label == "motion"
    assert second[0].detail["previous_sequence"] == 0


def test_motion_scorer_does_not_compare_across_sources_or_after_reset():
    scorer = MotionEnergyScorer(threshold=0.1, min_area_px=16)
    scorer.score(image_with_square((0, 0, 0)), source_id="left", sequence=0, source_s=100.0)
    assert scorer.score(image_with_square((255, 255, 255)), source_id="right", sequence=1, source_s=100.1) == []
    scorer.reset()
    assert scorer.score(image_with_square((255, 255, 255)), source_id="left", sequence=2, source_s=100.2) == []


def test_label_components_are_four_connected_and_area_filtered():
    mask = np.zeros((10, 10), dtype=bool)
    mask[1:3, 1:3] = True
    mask[7:9, 7:9] = True
    components = _label_components(mask, min_area=3)
    assert len(components) == 2
    assert _label_components(mask, min_area=5) == []


def test_selection_forwards_within_budget_and_reports_the_rest():
    candidates = [
        detection(index, 0.9 - index * 0.1, centroid=(20.0 + index * 40.0, 20.0))
        for index in range(6)
    ]
    report = select_detections(candidates, policy=SelectionPolicy(max_total=3, max_per_source=10))
    assert report.selected_count == 3
    assert report.dropped_count == 3
    assert report.dropped_by_reason() == {DropReason.OVER_TOTAL_BUDGET.value: 3}
    assert all(item.recoverable for item in report.dropped)
    assert report.considered == 6


def test_threshold_drops_are_recorded_as_existing_candidates_not_as_absence():
    report = select_detections(
        [detection(0, 0.5), detection(1, 0.01)],
        policy=SelectionPolicy(min_score=0.2),
    )
    assert report.selected_count == 1
    dropped = report.dropped[0]
    assert dropped.reason is DropReason.BELOW_THRESHOLD
    assert dropped.recoverable
    described = report.describe()["dropped"][0]
    assert described["candidate_exists"] is True
    assert described["reason"] == "below_threshold"
    assert any("not evidence that they were absent" in note for note in report.notes)


def test_degenerate_bbox_is_dropped_as_not_analysable_and_is_not_recoverable():
    broken = Detection(
        source_id="camera", sequence=0, source_s=100.0, scorer="color_blob", label="blob",
        score=0.9, bbox_xyxy=(10.0, 10.0, 10.0, 30.0), centroid_xy=(10.0, 20.0),
    )
    report = select_detections([broken])
    assert report.selected_count == 0
    assert report.dropped[0].reason is DropReason.NOT_ANALYSABLE
    assert not report.dropped[0].recoverable


def test_explicit_scorer_list_is_a_promise_about_what_was_consulted():
    with pytest.raises(SelectionError, match="outside the policy"):
        select_detections(
            [detection(0, 0.5)],
            policy=SelectionPolicy(scorers=("yolo",)),
        )


def test_per_window_budget_is_applied_per_window_key():
    candidates = [
        detection(index, 0.9, source_id="camera", centroid=(20.0 + index * 4.0, 20.0 + (index % 2) * 30.0))
        for index in range(4)
    ]
    report = select_detections(
        candidates,
        policy=SelectionPolicy(max_per_source=2, max_total=10),
        window_key=lambda item: "window-a",
    )
    assert report.selected_count == 2
    assert report.dropped_by_reason() == {DropReason.OVER_WINDOW_BUDGET.value: 2}
    assert "recoverable" in report.text_summary() or report.text_summary().startswith("selected")


def test_distinct_hint_deduplication_does_not_merge_different_labels():
    left = detection(0, 0.9, label="red_block")
    right = detection(1, 0.8, label="blue_block")
    report = select_detections([left, right], policy=SelectionPolicy(max_total=10))
    assert report.selected_count == 2


def test_selection_never_exceeds_the_configured_budget():
    candidates = [
        detection(index, 0.5, centroid=(10.0 + (index % 8) * 60.0, 10.0 + (index // 8) * 60.0))
        for index in range(80)
    ]
    report = select_detections(candidates, budget=Budget(max_candidates_total=5,
                                                        max_frames_per_source_per_window=5))
    assert report.selected_count == 5
    assert report.considered == 80


def test_scorer_cap_is_reported_even_when_it_hides_every_candidate():
    """A scorer's own limit must stay visible when the surviving list is empty."""
    scorer = ColorBlobScorer((255, 0, 0), tolerance=0.2, min_area_px=4, max_blobs=2)
    frame = np.zeros((40, 200, 3), dtype=np.uint8)
    for column in range(5):
        frame[5:20, column * 40: column * 40 + 20] = (255, 0, 0)
    found = scorer.score(frame, source_id="camera", sequence=0, source_s=100.0)
    assert len(found) == 2
    assert all(item.truncated_by_scorer == 3 for item in found)

    report = select_detections(found, policy=SelectionPolicy(min_score=1.1))
    assert report.selected_count == 0
    assert report.truncated_by_scorer == {"color_blob": 3}
    assert report.scorer_capped_total == 3
    assert any("before selection saw it" in note for note in report.notes)
