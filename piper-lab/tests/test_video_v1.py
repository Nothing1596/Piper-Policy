"""Human-video v1 ingestion tests.

All videos are generated synthetically in tmp dirs (solid background plus a
moving rectangle); no real footage, no network, no hardware. Decode/scoring
integration tests skip cleanly until root installs the video extras (PyAV,
OpenCV) into the video environment; the timestamp-validity and selection
logic is tested without them.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from piperlab.video.candidates import (
    CandidateBudget,
    _plan_selection,
    _validate_budget,
    build_candidates,
)
from piperlab.video.source import (
    FrameRecord,
    VideoError,
    VideoSource,
    _check_rotation,
    _PtsGuard,
    _require_av,
    _require_pts,
    _rotation_degrees,
)

HAVE_AV = importlib.util.find_spec("av") is not None
HAVE_CV2 = importlib.util.find_spec("cv2") is not None
needs_av = pytest.mark.skipif(not HAVE_AV, reason="PyAV not installed (video extra)")
needs_cv2 = pytest.mark.skipif(not HAVE_CV2, reason="OpenCV not installed (video extra)")

SYNTHETIC_NOTE = "synthetic test fixture generated in a tmp dir; not real footage"


def _write_synthetic_video(path: Path, *, frames: int = 24, fps: int = 10,
                           size: tuple[int, int] = (96, 64),
                           moving_start: int = 8, moving_stop: int = 16) -> Path:
    """Write a small MPEG-4 video: static scene, then a moving red square.

    Fully synthetic; ``moving_start``/``moving_stop`` delimit the motion burst
    so candidate tests know where the events live.
    """
    import av  # local: only called from av-gated tests

    container = av.open(str(path), "w")
    stream = container.add_stream("mpeg4", rate=Fraction(fps, 1))
    stream.width, stream.height = size
    stream.pix_fmt = "yuv420p"
    width, height = size
    for index in range(frames):
        image = np.full((height, width, 3), (30, 30, 40), dtype=np.uint8)
        if moving_start <= index <= moving_stop:
            # bright 32px square moving 8 px/frame; clipped so it stays inside
            x = min(4 + (index - moving_start) * 8, width - 32 - 2)
            image[12:44, x:x + 32] = (235, 235, 235)
        frame = av.VideoFrame.from_ndarray(image, format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()
    return path


# ---------------------------------------------------------------------------
# timestamp validity (no PyAV needed)


class TestPtsValidity:
    def test_duplicate_or_backwards_pts_rejected(self):
        guard = _PtsGuard()
        guard.check(10)
        with pytest.raises(VideoError, match="non-monotonic"):
            guard.check(10)
        with pytest.raises(VideoError, match="non-monotonic"):
            guard.check(9)
        assert guard.check(11) == 11

    def test_missing_pts_rejected_never_substituted(self):
        with pytest.raises(VideoError, match="no PTS"):
            _require_pts(None, sequence=3)

    def test_frame_record_timestamp_must_be_pts_times_time_base(self):
        image = np.zeros((4, 4, 3), dtype=np.uint8)
        record = FrameRecord(
            sequence=0, pts=1536, time_base="1/15360", timestamp_s=0.1, image=image
        )
        assert record.timestamp_s == pytest.approx(0.1)
        with pytest.raises(VideoError, match="does not equal"):
            FrameRecord(
                sequence=0, pts=1536, time_base="1/15360", timestamp_s=0.5, image=image
            )

    def test_frame_record_requires_rgb_uint8(self):
        with pytest.raises(VideoError):
            FrameRecord(
                sequence=0, pts=0, time_base="1/15360", timestamp_s=0.0,
                image=np.zeros((4, 4), dtype=np.uint8),
            )


class TestRotationPolicy:
    def test_rotation_metadata_is_read(self):
        stream = SimpleNamespace(metadata={"rotate": "90"}, side_data=None)
        container = SimpleNamespace(metadata={})
        assert _rotation_degrees(container, stream) == 90.0

    def test_zero_rotation_accepted(self):
        stream = SimpleNamespace(metadata={"rotate": "0"}, side_data=None)
        container = SimpleNamespace(metadata={})
        assert _check_rotation(container, stream) == 0.0

    def test_nonzero_rotation_explicitly_rejected(self):
        stream = SimpleNamespace(metadata={"rotate": "-90"}, side_data=None)
        container = SimpleNamespace(metadata={})
        with pytest.raises(VideoError, match="unsupported rotation"):
            _check_rotation(container, stream)

    def test_unparseable_rotation_rejected(self):
        stream = SimpleNamespace(metadata={"rotate": "sideways"}, side_data=None)
        container = SimpleNamespace(metadata={})
        with pytest.raises(VideoError, match="unparseable"):
            _check_rotation(container, stream)

    def test_missing_av_is_a_clear_error(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "av", None)
        with pytest.raises(VideoError, match="PyAV"):
            _require_av()


# ---------------------------------------------------------------------------
# decoding (needs PyAV)


@needs_av
class TestDecode:
    def test_frames_decode_all_with_strict_monotonic_timestamps(self, tmp_path):
        video = _write_synthetic_video(tmp_path / "synthetic.mp4", frames=24, fps=10)
        source = VideoSource(video)
        records = list(source.frames())
        assert len(records) == 24
        timestamps = [r.timestamp_s for r in records]
        assert timestamps == sorted(timestamps)
        assert len(set(timestamps)) == len(timestamps)
        # 10 fps: last frame near 23/10 s; from PTS*time_base, not from fps math
        assert timestamps[-1] == pytest.approx(2.3, abs=0.05)
        for index, record in enumerate(records):
            assert record.sequence == index
            assert record.image.shape == (64, 96, 3)
            assert record.image.dtype == np.uint8
            num, den = record.time_base.split("/")
            expected = float(Fraction(record.pts) * Fraction(int(num), int(den)))
            assert record.timestamp_s == pytest.approx(expected, abs=1e-12)

    def test_sample_hz_bounds_yield_rate(self, tmp_path):
        video = _write_synthetic_video(tmp_path / "synthetic.mp4", frames=24, fps=10)
        source = VideoSource(video)
        records = list(source.frames(sample_hz=5))
        assert 10 <= len(records) <= 14  # ~half of 24, boundaries included
        timestamps = [r.timestamp_s for r in records]
        for earlier, later in zip(timestamps, timestamps[1:]):
            assert later - earlier >= 0.2 - 1e-6
        assert records[0].timestamp_s == pytest.approx(0.0, abs=1e-9)

    def test_at_returns_first_actual_pts_at_or_after_request(self, tmp_path):
        video = _write_synthetic_video(tmp_path / "synthetic.mp4", frames=24, fps=10)
        source = VideoSource(video)
        record = source.at(0.35)
        assert record.timestamp_s >= 0.35
        assert record.timestamp_s == pytest.approx(0.4, abs=1e-6)
        assert source.at(0.0).timestamp_s == pytest.approx(0.0, abs=1e-9)
        assert source.at(0.4).timestamp_s == pytest.approx(0.4, abs=1e-6)

    def test_sampling_can_preserve_the_actual_final_frame(self,tmp_path):
        video=_write_synthetic_video(tmp_path/'last-frame.mp4',frames=24,fps=10)
        source=VideoSource(video)
        sampled=list(source.frames(sample_hz=5,include_last=True))
        assert sampled[-1].sequence==23
        assert sampled[-1].timestamp_s==pytest.approx(2.3)
        assert sampled[-1].timestamp_s-sampled[-2].timestamp_s<.2

    def test_at_outside_source_raises(self, tmp_path):
        video = _write_synthetic_video(tmp_path / "synthetic.mp4", frames=24, fps=10)
        source = VideoSource(video)
        with pytest.raises(VideoError, match="beyond the last decoded frame"):
            source.at(99.0)
        with pytest.raises(VideoError, match="finite >= 0"):
            source.at(-0.5)

    def test_metadata_reports_hash_timebase_and_no_fps_timestamps(self, tmp_path):
        video = _write_synthetic_video(tmp_path / "synthetic.mp4", frames=24, fps=10)
        source = VideoSource(video)
        metadata = source.metadata()
        expected_sha = hashlib.sha256(video.read_bytes()).hexdigest()
        assert metadata["sha256"] == expected_sha == source.sha256
        assert metadata["rotation_degrees"] == 0.0
        assert metadata["rotation_policy"] == "reject_nonzero"
        assert "pts*time_base" in metadata["timestamp_policy"]
        assert "/" in metadata["time_base"]
        assert metadata["width"] == 96 and metadata["height"] == 64

    def test_missing_file_rejected(self, tmp_path):
        with pytest.raises(VideoError, match="does not exist"):
            VideoSource(tmp_path / "nope.mp4")


# ---------------------------------------------------------------------------
# selection planning (pure, no cv2)


def _samples(count: int, hz: float = 10.0, spikes: dict[int, float] | None = None,
             stride: int = 1):
    """Synthetic sampled records; ``spikes`` keys are sequence numbers.

    ``stride`` > 1 produces non-consecutive sequences (e.g. 30 fps decoded,
    10 Hz sampled -> sequences 0, 3, 6, ...), proving neighbour selection is
    chronological and never sequence +/- 1.
    """
    spikes = spikes or {}
    return [
        {
            "sequence": i * stride,
            "timestamp_s": i / hz,
            "motion": spikes.get(i * stride, 0.0 if i else None),
        }
        for i in range(count)
    ]


class TestSelectionPlanner:
    def test_first_last_baseline_and_event_neighbours_kept(self):
        samples = _samples(200, hz=10.0, spikes={120: 0.9})
        reasons, dropped = _plan_selection(
            samples, window_s=10.0, max_per_window=24, baseline_hz=1.0
        )
        assert dropped == []
        assert "first" in reasons[0]
        assert "last" in reasons[199]
        assert "event" in reasons[120]
        # neighbours are the sampled frames nearest to event +/- 0.5 s:
        # event at t=12.0 s -> t-0.5=11.5 s (seq 115), t+0.5=12.5 s (seq 125)
        assert "event_before" in reasons[115]
        assert "event_after" in reasons[125]
        baseline_count = sum(1 for r in reasons.values() if "baseline" in r)
        assert baseline_count >= 18  # ~1 Hz over 19.9 s

    def test_neighbours_follow_chronology_not_sequence_ids(self):
        # 30 fps decode sampled at 10 Hz -> sequences 0, 3, 6, ...
        samples = _samples(200, hz=10.0, spikes={120: 0.9}, stride=3)
        reasons, dropped = _plan_selection(
            samples, window_s=10.0, max_per_window=24, baseline_hz=1.0
        )
        assert dropped == []
        assert "event" in reasons[120]  # index 40, t=4.0 s
        # nearest sampled frames to 3.5 s / 4.5 s: indices 35 / 45
        assert "event_before" in reasons[105]
        assert "event_after" in reasons[135]
        # sequence +/- 1 (119/121) does not even exist in this source
        assert 119 not in reasons and 121 not in reasons

    def test_impossible_budget_fails_up_front(self):
        with pytest.raises(ValueError, match="cannot hold mandatory"):
            _validate_budget(window_s=10.0, max_per_window=8, baseline_hz=1.0)
        with pytest.raises(ValueError, match="cannot hold mandatory"):
            _plan_selection(
                _samples(100), window_s=10.0, max_per_window=8, baseline_hz=1.0
            )

    def test_window_budget_overflow_reports_every_drop(self):
        # 12 strong events inside window 1 (of 100 samples): all qualify, but
        # max_per_window=20 leaves only 9 event/neighbour slots after the
        # mandatory+baseline frames of that window.
        spikes = {i: 0.9 + (i - 100) * 0.001 for i in range(100, 112)}
        samples = _samples(200, hz=10.0, spikes=spikes)
        reasons, dropped = _plan_selection(
            samples, window_s=10.0, max_per_window=20, baseline_hz=1.0
        )
        per_window: dict[int, int] = {}
        for seq in reasons:
            window = int(samples[seq]["timestamp_s"] // 10.0)
            per_window[window] = per_window.get(window, 0) + 1
        assert all(count <= 20 for count in per_window.values())
        dropped_events = [d for d in dropped if d["kind"] == "event"]
        dropped_neighbours = [d for d in dropped if d["kind"] == "event_neighbour"]
        assert dropped_events, "over-budget events must be reported"
        assert all(d["reason"] == "window_budget" for d in dropped)
        kept_events = sum(1 for r in reasons.values() if "event" in r)
        assert kept_events + len(dropped_events) == 12
        # highest-motion event survives; lower ones are reported dropped
        assert "event" in reasons[111]
        assert {d["sequence"] for d in dropped_events} <= set(range(100, 112))
        # neighbour losses are explicit too
        assert dropped_neighbours
        assert all(d["reason"] == "window_budget" for d in dropped_neighbours)
        assert all("event_sequence" in d for d in dropped_neighbours)
        # mandatory/baseline never dropped
        assert 0 in reasons and 199 in reasons
        assert not any(d.get("sequence") in (0, 199) for d in dropped)

    def test_neighbours_across_window_boundary_charged_to_own_window(self):
        samples = _samples(200, hz=10.0, spikes={100: 0.9})  # exactly at window edge
        reasons, dropped = _plan_selection(
            samples, window_s=10.0, max_per_window=24, baseline_hz=1.0
        )
        assert dropped == []
        assert "event" in reasons[100]
        # before-neighbour nearest to 9.5 s lives in window 0, still kept
        assert "event_before" in reasons[95]
        assert int(9.5 // 10.0) == 0
        assert "event_after" in reasons[105]

    def test_event_at_video_start_reports_missing_before_sample(self):
        samples = _samples(100, hz=10.0, spikes={0: 0.9})
        reasons, dropped = _plan_selection(
            samples, window_s=10.0, max_per_window=24, baseline_hz=1.0
        )
        assert "event" in reasons[0] and "first" in reasons[0]
        missing = [d for d in dropped if d["reason"] == "no_sample_exists"]
        assert any(d["side"] == "before" for d in missing)


# ---------------------------------------------------------------------------
# candidate building (needs PyAV + OpenCV)


@needs_av
@needs_cv2
class TestBuildCandidates:
    def test_manifest_candidates_and_budget(self, tmp_path):
        video = _write_synthetic_video(
            tmp_path / "synthetic.mp4", frames=40, fps=10, moving_start=15, moving_stop=25
        )
        output = tmp_path / "candidates"
        manifest = build_candidates(str(video), str(output), window_s=2.0,
                                    max_per_window=24, analysis_hz=10, baseline_hz=1)
        on_disk = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        assert on_disk == manifest
        assert manifest["schema_version"] == 1
        assert manifest["source"]["sha256"] == hashlib.sha256(video.read_bytes()).hexdigest()
        assert manifest["source"]["path"] == str(video.absolute())
        candidates = manifest["candidates"]
        assert candidates, "expected candidates"
        # first/last preserved
        reasons_first = candidates[0]["reasons"]
        reasons_last = candidates[-1]["reasons"]
        assert "first" in reasons_first and "last" in reasons_last
        # every candidate has an absolute image path to an existing JPEG
        for cand in candidates:
            image_path = Path(cand["image_path"])
            assert image_path.is_absolute() and image_path.is_file()
            assert image_path.suffix == ".jpg"
            assert {"motion", "clarity", "camera"} <= set(cand["scores"])
            assert cand["window"] == int(cand["timestamp_s"] // 2.0)
        # per-window budget honoured
        counts: dict[int, int] = {}
        for cand in candidates:
            counts[cand["window"]] = counts.get(cand["window"], 0) + 1
        assert all(count <= 24 for count in counts.values())
        assert manifest["budget"]["per_window_candidate_counts"] == {
            str(k): v for k, v in sorted(counts.items())
        }
        # the motion burst is represented as an event with neighbours
        event_ids = {c["frame_id"] for c in candidates if "event" in c["reasons"]}
        neighbour_ids = {c["frame_id"] for c in candidates
                         if {"event_before", "event_after"} & set(c["reasons"])}
        assert event_ids, "expected at least one motion event candidate"
        assert neighbour_ids, "expected before/after neighbours around events"
        assert manifest["budget"]["dropped_event_candidates"] == len(manifest["dropped"])

    def test_output_dir_refuses_overwrite(self, tmp_path):
        video = _write_synthetic_video(tmp_path / "synthetic.mp4", frames=12, fps=10)
        output = tmp_path / "candidates"
        build_candidates(str(video), str(output))
        with pytest.raises(FileExistsError, match="refusing to overwrite"):
            build_candidates(str(video), str(output))

    def test_bad_budget_combination_rejected_before_decoding(self, tmp_path):
        video = _write_synthetic_video(tmp_path / "synthetic.mp4", frames=12, fps=10)
        with pytest.raises(ValueError, match="cannot hold mandatory"):
            build_candidates(str(video), str(tmp_path / "out"),
                             window_s=10, max_per_window=5, baseline_hz=1)

    def test_sampled_frame_budget_raises_explicitly(self, tmp_path):
        video = _write_synthetic_video(tmp_path / "synthetic.mp4", frames=24, fps=10)
        with pytest.raises(VideoError, match="sampled frames exceed budget"):
            build_candidates(str(video), str(tmp_path / "out"),
                             budget=CandidateBudget(max_sampled_frames=5))
        assert not (tmp_path / "out").exists()  # nothing committed on failure

    def test_duration_budget_raises_explicitly(self, tmp_path):
        video = _write_synthetic_video(tmp_path / "synthetic.mp4", frames=24, fps=10)
        with pytest.raises(VideoError, match="duration exceeds budget"):
            build_candidates(str(video), str(tmp_path / "out"),
                             budget=CandidateBudget(max_duration_s=0.5))
        assert not (tmp_path / "out").exists()

    def test_pts_and_paths_valid_after_commit(self, tmp_path):
        video = _write_synthetic_video(tmp_path / "synthetic.mp4", frames=24, fps=10)
        manifest = build_candidates(str(video), str(tmp_path / "out"), window_s=2.0)
        for cand in manifest["candidates"]:
            # exact PTS preserved: timestamp_s == pts * time_base
            num, den = cand["time_base"].split("/")
            assert cand["timestamp_s"] == pytest.approx(
                float(Fraction(cand["pts"]) * Fraction(int(num), int(den))), abs=1e-12
            )
            # path is valid post-rename and lives under the committed output
            path = Path(cand["image_path"])
            assert path.is_file()
            assert str(tmp_path / "out") in str(path)
            assert ".tmp-" not in str(path)


class FakeScorer:
    """In-test scorer conforming to the perception Scorer protocol.

    Synthetic stand-in for root's ONNX detector; no torch, no ONNX.
    """

    name = "fake_square"

    def __init__(self):
        self.calls: list[tuple[int, float]] = []

    def score(self, image, *, source_id: str, sequence: int, source_s: float):
        from piperlab.perception.detect import Detection

        self.calls.append((sequence, source_s))
        return [
            Detection(
                source_id=source_id,
                sequence=sequence,
                source_s=source_s,
                scorer=self.name,
                label="square",
                score=0.9,
                bbox_xyxy=(1.0, 2.0, 11.0, 12.0),
                centroid_xy=(6.0, 7.0),
            )
        ]


@needs_av
@needs_cv2
class TestDetectorIntegration:
    def test_detector_summaries_and_report(self, tmp_path):
        video = _write_synthetic_video(tmp_path / "synthetic.mp4", frames=24, fps=10)
        scorer = FakeScorer()
        manifest = build_candidates(str(video), str(tmp_path / "out"),
                                    detector=scorer, detector_hz=2)
        report = manifest["detector"]
        assert report["name"] == "fake_square"
        assert report["detector_hz"] == 2.0
        # 2 Hz over ~2.3 s of video
        assert 4 <= report["analyzed_frames"] <= 6
        assert report["total_detections"] == report["analyzed_frames"]
        assert report["truncated_by_scorer"] == 0
        # cadence honoured: detector calls ~0.5 s apart on the source clock
        call_times = [t for _seq, t in sorted(scorer.calls, key=lambda c: c[1])]
        for earlier, later in zip(call_times, call_times[1:]):
            assert later - earlier >= 0.5 - 1e-6
        # per-candidate bounded summaries: no raw boxes retained
        with_detections = [c for c in manifest["candidates"] if c["detections"]]
        assert with_detections
        for cand in with_detections:
            summary = cand["detections"]
            assert summary["count"] == 1
            assert summary["labels"] == ["square"]
            assert summary["max_score"] == pytest.approx(0.9)
            assert len(summary['regions'])==1
            assert summary['regions'][0]['bbox_xyxy']==[1.0,2.0,11.0,12.0]
        # detector identity recorded in parameters for cache identity
        assert manifest["parameters"]["detector"] == "fake_square"
        assert manifest["parameters"]["detector_hz"] == 2.0

    def test_no_detector_still_works(self, tmp_path):
        video = _write_synthetic_video(tmp_path / "synthetic.mp4", frames=12, fps=10)
        manifest = build_candidates(str(video), str(tmp_path / "out"))
        assert manifest["detector"] is None
        assert all(c["detections"] is None for c in manifest["candidates"])

    def test_detector_failure_is_explicit(self, tmp_path):
        class BadScorer:
            name = "bad"

            def score(self, image, *, source_id, sequence, source_s):
                raise RuntimeError("synthetic detector failure")

        video = _write_synthetic_video(tmp_path / "synthetic.mp4", frames=12, fps=10)
        with pytest.raises(VideoError, match="detector failed"):
            build_candidates(str(video), str(tmp_path / "out"), detector=BadScorer())
        assert not (tmp_path / "out").exists()

    def test_malformed_detector_output_rejected(self, tmp_path):
        class OpaqueScorer:
            name = "opaque"

            def score(self, image, *, source_id, sequence, source_s):
                return [object()]  # not a Detection

        video = _write_synthetic_video(tmp_path / "synthetic.mp4", frames=12, fps=10)
        with pytest.raises(VideoError, match="malformed detections"):
            build_candidates(str(video), str(tmp_path / "out"), detector=OpaqueScorer())
