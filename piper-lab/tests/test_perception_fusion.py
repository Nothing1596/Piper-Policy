"""Track identity, association and pixel-to-base projection.

Two boundaries are under test: a local track id is not a cross-sensor entity id,
and a pixel is not a position. Both are places where a pipeline can quietly
invent certainty it does not have.
"""
import numpy as np
import pytest

from piperlab.perception.budget import Budget, BudgetExceeded
from piperlab.perception.detect import Detection
from piperlab.perception.fuse import (
    BEHIND_CAMERA,
    Calibration,
    CalibrationError,
    DEGENERATE_PIXEL,
    DEPTH_OUT_OF_RANGE,
    NO_CALIBRATION,
    NO_DEPTH,
    fuse_entities,
    project,
)
from piperlab.perception.tracks import EntityTracker, TrackError, TrackId


def detection(sequence, source_s, centroid=(10.0, 10.0), label="block", score=0.8, source_id="camera"):
    return Detection(
        source_id=source_id, sequence=sequence, source_s=source_s, scorer="color_blob",
        label=label, score=score,
        bbox_xyxy=(centroid[0] - 5, centroid[1] - 5, centroid[0] + 5, centroid[1] + 5),
        centroid_xy=centroid, area_px=100.0,
    )


def identity_calibration(*, depth_scale_m=0.001, translation=(0.0, 0.0, 0.0)):
    return Calibration(
        calibration_id="test-calib-1",
        intrinsics=[[200.0, 0.0, 320.0], [0.0, 200.0, 240.0], [0.0, 0.0, 1.0]],
        base_from_camera=np.array(
            [[1.0, 0.0, 0.0, translation[0]],
             [0.0, 1.0, 0.0, translation[1]],
             [0.0, 0.0, 1.0, translation[2]],
             [0.0, 0.0, 0.0, 1.0]]
        ),
        distortion=[0.0] * 5,
        depth_scale_m=depth_scale_m,
        method="unit_test_identity",
    )


# --- projection -------------------------------------------------------------
def test_projection_without_calibration_refuses_and_explains():
    result = project((320.0, 240.0), calibration=None)
    assert result.available is False
    assert result.reason == NO_CALIBRATION
    assert result.base_xyz_m is None
    assert "no base_link-from-camera transform" in result.detail["hint"]


def test_rgb_only_yields_a_ray_not_a_point():
    result = project((320.0, 240.0), calibration=identity_calibration())
    assert result.available is False
    assert result.reason == NO_DEPTH
    assert result.ray_direction_base_xyz is not None
    # Optical axis through the principal point points along +z of the camera.
    assert result.ray_direction_base_xyz == pytest.approx((0.0, 0.0, 1.0), abs=1e-6)
    assert "ray, not a point" in result.detail["hint"]


def test_calibration_plus_depth_yields_base_coordinates_with_identified_calibration():
    calibration = identity_calibration(translation=(0.2, 0.0, 0.5))
    result = project((320.0, 240.0), calibration=calibration, depth_m=1.0)
    assert result.available is True
    assert result.calibration_id == "test-calib-1"
    assert result.base_xyz_m == pytest.approx((0.2, 0.0, 1.5), abs=1e-9)
    # No per-pixel noise model has been established, so uncertainty is unknown
    # rather than a fabricated small number.
    assert result.uncertainty_m is None
    assert "unqualified" in result.detail["note"]


def test_pixel_outside_the_image_and_bad_depth_are_reported_not_clamped():
    calibration = identity_calibration()
    outside = project((700.0, 240.0), calibration=calibration, image_size=(640, 480))
    assert outside.available is False and outside.reason == DEGENERATE_PIXEL

    bad_depth = project((320.0, 240.0), calibration=calibration, depth_m=0.0)
    assert bad_depth.available is False and bad_depth.reason == DEPTH_OUT_OF_RANGE

    behind = project((320.0, 240.0), calibration=identity_calibration(depth_scale_m=None), depth_m=-0.5)
    assert behind.available is False
    assert behind.reason in (DEPTH_OUT_OF_RANGE, BEHIND_CAMERA)


def test_calibration_validation_rejects_a_non_orthonormal_rotation():
    with pytest.raises(CalibrationError, match="orthonormal"):
        Calibration(
            calibration_id="bad",
            intrinsics=[[200.0, 0.0, 320.0], [0.0, 200.0, 240.0], [0.0, 0.0, 1.0]],
            base_from_camera=[[2.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0],
                              [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
        )
    with pytest.raises(CalibrationError, match="calibration_id"):
        Calibration(
            calibration_id="",
            intrinsics=[[200.0, 0.0, 320.0], [0.0, 200.0, 240.0], [0.0, 0.0, 1.0]],
            base_from_camera=np.eye(4),
        )


def test_distortion_is_unapplied_before_projecting():
    calibration = identity_calibration()
    distorted = Calibration(
        calibration_id="test-calib-distorted",
        intrinsics=calibration.intrinsics,
        base_from_camera=calibration.base_from_camera,
        distortion=[0.2, 0.01, 0.0, 0.0, 0.0],
        depth_scale_m=0.001,
    )
    plain = project((400.0, 300.0), calibration=calibration, depth_m=1.0)
    corrected = project((400.0, 300.0), calibration=distorted, depth_m=1.0)
    assert plain.base_xyz_m != corrected.base_xyz_m


# --- tracking ---------------------------------------------------------------
def test_tracker_keeps_identity_across_frames_and_accumulates_evidence():
    tracker = EntityTracker(match_radius_px=20.0, confirm_hits=2)
    first = tracker.update([detection(0, 100.0)], source_s=100.0)
    assert len(first) == 1 and first[0].state == "new"
    second = tracker.update([detection(1, 100.1, centroid=(12.0, 10.0))], source_s=100.1)
    assert second[0].entity_id == first[0].entity_id
    assert second[0].state == "tracked"
    assert second[0].evidence_sequences == (0, 1)
    assert second[0].track.key == "camera:0:0"


def test_loss_becomes_occluded_then_lost_and_is_never_deleted():
    tracker = EntityTracker(match_radius_px=5.0, confirm_hits=1, max_misses=3)
    tracker.update([detection(0, 100.0)], source_s=100.0)
    tracker.update([], source_s=100.1)
    entity = tracker.get("ent-0000")
    assert entity.state == "occluded"
    assert entity.visible is False and entity.gap is True
    tracker.update([], source_s=100.2)
    tracker.update([], source_s=100.3)
    assert tracker.get("ent-0000").state == "lost"
    # Still present: losing a track is not evidence the object ceased to exist.
    assert tracker.get("ent-0000") is not None
    assert tracker.visible_count() == 0


def test_reacquisition_continues_the_same_identity():
    tracker = EntityTracker(match_radius_px=5.0, confirm_hits=1, max_misses=1)
    tracker.update([detection(0, 100.0)], source_s=100.0)
    tracker.update([], source_s=100.1)
    assert tracker.get("ent-0000").state == "lost"
    entity = tracker.mark_reacquired("ent-0000", detection(2, 100.2))
    assert entity.state == "reacquired"
    assert entity.hits == 2
    assert entity.meta["reacquired_at_source_s"] == 100.2


def test_two_objects_close_together_stay_two_entities():
    tracker = EntityTracker(match_radius_px=50.0, confirm_hits=1)
    tracker.update(
        [detection(0, 100.0, centroid=(10.0, 10.0)), detection(0, 100.0, centroid=(200.0, 200.0))],
        source_s=100.0,
    )
    assert len(tracker.entities) == 2


def test_association_is_reversible_and_records_its_support():
    tracker = EntityTracker(confirm_hits=1)
    tracker.update([detection(0, 100.0)], source_s=100.0)
    association = tracker.associate(
        "ent-0000",
        TrackId("wrist", 0, 7),
        confidence=0.6,
        supporting_observations=[0],
        method="appearance_similarity",
        source_s=100.0,
        same=True,
        notes="possible match, not confirmed",
    )
    assert association.possible_same_entity is True
    assert tracker.get("ent-0000").sources == ("camera", "wrist")
    described = tracker.describe()["associations"][0]
    assert described["possible_same_entity"] is True
    assert described["confidence"] == 0.6

    with pytest.raises(TrackError):
        tracker.associate(
            "ent-0000", TrackId("wrist", 0, 8), confidence=2.0,
            supporting_observations=[], method="x", source_s=100.0,
        )


def test_epoch_reset_prevents_local_ids_from_aliasing_old_ones():
    tracker = EntityTracker(confirm_hits=1)
    tracker.update([detection(0, 100.0)], source_s=100.0)
    old_key = tracker.get("ent-0000").track.key
    tracker.reset_epoch()
    tracker.update([detection(1, 100.1, centroid=(200.0, 200.0))], source_s=100.1)
    new_key = tracker.get("ent-0001").track.key
    assert old_key == "camera:0:0" and new_key == "camera:1:0"
    assert old_key != new_key


def test_tracker_respects_entity_budgets():
    tracker = EntityTracker(confirm_hits=1, budget=Budget(max_entities_per_frame=1))
    with pytest.raises(BudgetExceeded):
        tracker.update(
            [detection(0, 100.0), detection(0, 100.0, centroid=(200.0, 200.0))],
            source_s=100.0,
        )


# --- fusion -----------------------------------------------------------------
def test_fusion_never_averages_across_different_calibrations():
    tracker = EntityTracker(confirm_hits=1)
    tracker.update([detection(0, 100.0)], source_s=100.0)
    camera_entity = tracker.get("ent-0000")
    camera_entity.projection_available = True
    camera_entity.base_xyz_m = (0.2, 0.0, 0.1)
    camera_entity.calibration_id = "calib-a"

    other = EntityTracker(confirm_hits=1, source_id="wrist")
    other.update([detection(0, 100.0, source_id="wrist")], source_s=100.0)
    wrist_entity = other.get("ent-0000")
    wrist_entity.entity_id = camera_entity.entity_id  # same believed object
    wrist_entity.projection_available = True
    wrist_entity.base_xyz_m = (0.3, 0.1, 0.2)
    wrist_entity.calibration_id = "calib-b"

    fused = fuse_entities([camera_entity, wrist_entity])
    assert len(fused) == 1
    record = fused[0]
    assert "projections_use_different_calibrations_not_averaged" in record.conflicts
    assert record.base_xyz_m is None
    assert record.projection_available is False
    # Source count is not evidence count: both views are counted per source.
    assert record.detail["independent_evidence_count"] == 2


def test_fusion_averages_only_within_one_calibration_and_flags_the_spread():
    tracker = EntityTracker(confirm_hits=1)
    tracker.update([detection(0, 100.0)], source_s=100.0)
    a = tracker.get("ent-0000")
    a.projection_available = True
    a.base_xyz_m = (0.20, 0.0, 0.10)
    a.calibration_id = "calib-a"

    other = EntityTracker(confirm_hits=1, source_id="wrist")
    other.update([detection(0, 100.0, source_id="wrist")], source_s=100.0)
    b = other.get("ent-0000")
    b.entity_id = a.entity_id
    b.projection_available = True
    b.base_xyz_m = (0.24, 0.0, 0.10)
    b.calibration_id = "calib-a"

    fused = fuse_entities([a, b])[0]
    assert fused.base_xyz_m == pytest.approx((0.22, 0.0, 0.10))
    assert "multi_view_average_reported_with_spread" in fused.conflicts


def test_fusion_counts_repeated_processing_of_one_frame_once():
    tracker = EntityTracker(confirm_hits=1)
    tracker.update([detection(0, 100.0)], source_s=100.0)
    entity = tracker.get("ent-0000")
    # Three analyses of the same frame is still one observation.
    assert entity.evidence_sequences == (0,)
    fused = fuse_entities([entity])[0]
    assert fused.evidence_sequences == (0,)
    assert fused.detail["independent_evidence_count"] == 1
