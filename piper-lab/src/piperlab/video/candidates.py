"""Traditional-CV candidate selection for human-video v1.

Two-pass, memory-bounded design:

1. Analysis pass: decode at ``analysis_hz``, downscale to gray at
   ``SCORING_LONG_EDGE`` and score with OpenCV (motion energy after
   best-effort camera-motion compensation, Laplacian clarity). Only numeric
   metadata plus the single previous low-resolution gray image are retained;
   full-resolution RGB frames are never accumulated. An optional detector
   (e.g. root's ONNX scorer, anything conforming to the perception
   ``Scorer.score(image, source_id=..., sequence=..., source_s=...)``
   protocol) runs at ``detector_hz`` on the full-resolution sampled frames;
   only bounded per-frame detection *summaries* are retained.
2. Materialization pass: after selection, the source is decoded a second
   time and only selected frames are written as JPEGs. The PTS decoded in
   pass 2 must equal the PTS recorded in pass 1 for the same sequence;
   a mismatch fails loudly instead of substituting anything.

Budgets are explicit (``CandidateBudget``): sampled-frame count, source
duration and retained gray bytes all raise on exceed; nothing is silently
truncated. Per-window selection keeps first/last and baseline samples,
events keep their ~+/-0.5 s before/after sampled neighbours, and every
event or neighbour that does not fit is reported in ``dropped`` with its
reason. The output directory must not exist; files land in a staging
directory renamed into place only after the manifest is written.
"""
from __future__ import annotations

import bisect
import json
import math
import os
import statistics
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .source import VideoError, VideoSource

SCORING_LONG_EDGE = 320
"""Frames are downscaled to this long edge for scoring (plan section A.2)."""

MOTION_EVENT_FLOOR = 0.02
"""Absolute mean-abs-diff floor (0..1 gray scale) below which nothing is an event."""

NEIGHBOUR_OFFSET_S = 0.5
"""Events keep the sampled frames nearest to event_time +/- this offset."""


@dataclass(frozen=True)
class CandidateBudget:
    """Hard analysis budgets; exceeding any of them raises, never truncates."""

    max_sampled_frames: int = 100_000
    """Cap on retained per-sample metadata records (pass 1)."""

    max_duration_s: float = 3_600.0
    """Cap on source duration; checked in both decode passes."""

    max_gray_retained_bytes: int = SCORING_LONG_EDGE * SCORING_LONG_EDGE
    """Cap on the one retained low-res gray image (previous frame only)."""

    def __post_init__(self) -> None:
        for name in ("max_sampled_frames", "max_gray_retained_bytes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"CandidateBudget.{name} must be a positive int")
        value = self.max_duration_s
        if isinstance(value, bool) or not isinstance(value, (int, float)) \
                or not math.isfinite(value) or value <= 0:
            raise ValueError("CandidateBudget.max_duration_s must be a positive finite number")

    def as_dict(self) -> dict:
        return {
            "max_sampled_frames": self.max_sampled_frames,
            "max_duration_s": self.max_duration_s,
            "max_gray_retained_bytes": self.max_gray_retained_bytes,
        }


def _require_cv2():
    """Import OpenCV lazily; the video extra is not a core dependency."""
    try:
        import cv2  # noqa: PLC0415 - intentional lazy import
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise VideoError(
            "OpenCV (package 'opencv-python', importable as 'cv2') is required "
            "for candidate scoring; install the piper-lab video extras first"
        ) from exc
    return cv2


def _fail_if_exists(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"output already exists, refusing to overwrite: {path}")


def _write_json_atomic(path: Path, payload: dict) -> None:
    tmp = path.with_name(path.name + f".tmp-{os.getpid()}")
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp, path)


def _check_float(name: str, value) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) \
            or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite number")
    return float(value)


def _check_int(name: str, value) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive int")
    return value


# ---------------------------------------------------------------------------
# pass 1: streaming analysis (retains metadata + one low-res gray image)


@dataclass
class _SampleMeta:
    """Bounded per-sample record; the only thing pass 1 retains per frame."""

    sequence: int
    pts: int
    time_base: str
    timestamp_s: float
    motion: float | None = None
    clarity: float = 0.0
    camera: dict | None = None
    detections: dict | None = None
    scene_cut_score: float = 0.
    track_events: list | None = None
    """Bounded summary only: count/labels/max_score/truncated; never raw boxes."""


def _downscale_gray(cv2, gray: np.ndarray, long_edge: int) -> np.ndarray:
    height, width = gray.shape
    longest = max(height, width)
    if longest <= long_edge:
        return gray
    scale = long_edge / float(longest)
    return cv2.resize(
        gray,
        (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
        interpolation=cv2.INTER_AREA,
    )


def _estimate_camera(cv2, prev_gray: np.ndarray, gray: np.ndarray) -> tuple[np.ndarray | None, dict]:
    """Sparse optical flow + partial-affine camera motion estimate.

    Returns (transform, report). ``transform`` maps prev -> cur and is None
    when the estimate is not trustworthy; the report says why. Failures mark
    camera motion unknown; they never force a compensation.
    """
    points = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=8)
    if points is None or len(points) < 12:
        return None, {"compensated": False, "reason": "too_few_features", "features": 0}
    tracked, status, _err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, points, None)
    if tracked is None or status is None:
        return None, {"compensated": False, "reason": "optical_flow_failed", "features": int(len(points))}
    good_prev = points[status.flatten() == 1]
    good_cur = tracked[status.flatten() == 1]
    if len(good_prev) < 12:
        return None, {
            "compensated": False,
            "reason": "too_few_tracks",
            "features": int(len(points)),
            "tracks": int(len(good_prev)),
        }
    transform, inliers = cv2.estimateAffinePartial2D(
        good_prev, good_cur, method=cv2.RANSAC, ransacReprojThreshold=2.0
    )
    if transform is None or inliers is None:
        return None, {
            "compensated": False,
            "reason": "affine_estimate_failed",
            "features": int(len(points)),
            "tracks": int(len(good_prev)),
        }
    inlier_count = int(inliers.sum())
    inlier_ratio = inlier_count / float(len(good_prev))
    if inlier_ratio < 0.3:
        return None, {
            "compensated": False,
            "reason": "inlier_ratio_too_low",
            "features": int(len(points)),
            "tracks": int(len(good_prev)),
            "inliers": inlier_count,
        }
    report = {
        "compensated": True,
        "features": int(len(points)),
        "tracks": int(len(good_prev)),
        "inliers": inlier_count,
        "dx": float(transform[0, 2]),
        "dy": float(transform[1, 2]),
    }
    return transform, report


def _motion_clarity(cv2, gray: np.ndarray, prev_gray: np.ndarray | None,
                    budget: CandidateBudget) -> tuple[float | None, float, dict | None, bool]:
    """Motion (vs previous gray, camera-compensated when feasible) + clarity."""
    if gray.nbytes > budget.max_gray_retained_bytes:
        raise VideoError(
            f"retained gray image is {gray.nbytes} bytes > "
            f"budget {budget.max_gray_retained_bytes}; scoring resolution "
            "must be lowered, memory is never grown silently"
        )
    clarity = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if prev_gray is None:
        return None, clarity, None, False
    transform, report = _estimate_camera(cv2, prev_gray, gray)
    if transform is not None:
        height, width = gray.shape
        warped_prev = cv2.warpAffine(prev_gray, transform, (width, height))
        valid = cv2.warpAffine(np.full_like(prev_gray, 255), transform, (width, height)) > 0
        difference = np.abs(gray.astype(np.float64) - warped_prev.astype(np.float64))
        motion = float(difference[valid].mean() / 255.0) if valid.any() else 0.0
        return motion, clarity, report, True
    difference = np.abs(gray.astype(np.float64) - prev_gray.astype(np.float64))
    return float(difference.mean() / 255.0), clarity, report, False


def _summarize_detections(detections) -> dict:
    """Bounded per-frame detection summary; raw boxes are not retained."""
    try:
        if len(detections)>128:raise VideoError('detector region metadata budget exceeded: 128')
        return {
            "count": len(detections),
            "labels": sorted({str(d.label) for d in detections}),
            "max_score": max((float(d.score) for d in detections), default=None),
            "truncated_by_scorer": sum(int(d.truncated_by_scorer) for d in detections),
            "regions":[{'label':d.label,'score':float(d.score),'bbox_xyxy':list(d.bbox_xyxy),
                        'track_id':d.detail.get('track_id'),'track_epoch':d.detail.get('track_epoch'),
                        'entity_key':d.detail.get('entity_key'),'entity_state':d.detail.get('entity_state')}
                       for d in detections],
        }
    except (AttributeError, TypeError, ValueError) as exc:
        raise VideoError(
            "detector returned malformed detections; expected the perception "
            f"Scorer Detection contract: {exc}"
        ) from exc


def _scene_cut_score(previous,current):
    """Histogram and broad pixel change proposal; exposure changes can also trigger it."""
    if previous is None:return 0.
    a=np.histogram(previous,bins=32,range=(0,256))[0]/previous.size
    b=np.histogram(current,bins=32,range=(0,256))[0]/current.size
    distance=float(np.abs(a-b).sum()/2)
    change=float(np.abs(previous.astype(float)-current.astype(float)).mean()/255)
    return distance if distance>=.6 and change>=.12 else 0.


def _track_changes(previous,current,width,height):
    """Detector-tick changes only. Lost visibility never means physical disappearance."""
    if previous is None:return []
    old_epoch=previous.get('detector_report',{}).get('tracker_epoch')
    epoch=current.get('detector_report',{}).get('tracker_epoch')
    if epoch!=old_epoch:
        return [{'kind':'tracker_epoch_reset','previous_epoch':old_epoch,'epoch':epoch}]
    def keyed(summary):
        return {(r['track_epoch'],r['track_id']):r for r in summary.get('regions',[])
                if isinstance(r.get('track_id'),int) and r['track_id']>=0}
    old,new=keyed(previous),keyed(current)
    # An epoch may be carried on each region by scorers without last_report.
    if old and new and {k[0] for k in old}!={k[0] for k in new}:
        return [{'kind':'tracker_epoch_reset','scope':'region_epochs'}]
    events=[]
    for key in sorted(old.keys()|new.keys(),key=str):
        if key not in old:events.append({'kind':'track_newly_observed','track':key});continue
        if key not in new:events.append({'kind':'track_not_observed','track':key,'meaning':'observation_gap'});continue
        a=np.asarray(old[key]['bbox_xyxy']);b=np.asarray(new[key]['bbox_xyxy'])
        displacement=float(np.linalg.norm(((a[:2]+a[2:]-b[:2]-b[2:])/2)/[width,height]))
        area_a=max(1.,float(np.prod(a[2:]-a[:2])));area_b=max(1.,float(np.prod(b[2:]-b[:2])))
        scale_change=abs(math.log(area_b/area_a))
        if displacement>=.04 or scale_change>=.3 or old[key]['label']!=new[key]['label']:
            events.append({'kind':'track_measurement_change','track':key,
                'normalized_displacement':displacement,'absolute_log_area_ratio':scale_change})
    return events


def _analyze(
    source: VideoSource,
    cv2,
    *,
    analysis_hz: float,
    detector,
    detector_hz: float,
    budget: CandidateBudget,
) -> tuple[list[_SampleMeta], dict | None, int]:
    """Pass 1: stream decode, score, detect; retain only bounded metadata.

    Returns (samples, detector_report, compensated_frame_count)."""
    samples: list[_SampleMeta] = []
    prev_gray: np.ndarray | None = None
    previous_detections=None
    compensated = 0
    detector_calls = 0
    detector_total = 0
    detector_truncated = 0
    next_detection_s: float | None = None
    detector_period = 1.0 / detector_hz if detector is not None else None

    for record in source.frames(sample_hz=analysis_hz,include_last=True):
        if len(samples) >= budget.max_sampled_frames:
            raise VideoError(
                f"sampled frames exceed budget {budget.max_sampled_frames}; "
                "raise CandidateBudget.max_sampled_frames or reduce analysis_hz"
            )
        if record.timestamp_s > budget.max_duration_s:
            raise VideoError(
                f"source duration exceeds budget {budget.max_duration_s}s at "
                f"{record.timestamp_s:.3f}s; refusing to analyse further"
            )
        gray = cv2.cvtColor(record.image, cv2.COLOR_RGB2GRAY)
        gray = _downscale_gray(cv2, gray, SCORING_LONG_EDGE)
        motion, clarity, camera, was_compensated = _motion_clarity(cv2, gray, prev_gray, budget)
        cut_score=_scene_cut_score(prev_gray,gray)
        prev_gray = gray
        compensated += int(was_compensated)

        detections_summary = None
        track_events=[]
        if detector is not None:
            if next_detection_s is None or record.timestamp_s >= next_detection_s - 1e-12:
                try:
                    detections = detector.score(
                        record.image,
                        source_id=source.sha256,
                        sequence=record.sequence,
                        source_s=record.timestamp_s,
                    )
                except VideoError:
                    raise
                except Exception as exc:
                    raise VideoError(
                        f"detector failed at sequence {record.sequence} "
                        f"({record.timestamp_s:.3f}s): {type(exc).__name__}: {exc}"
                    ) from exc
                detections_summary = _summarize_detections(detections)
                detections_summary['detector_report']=getattr(detector,'last_report',None) or {}
                height,width=record.image.shape[:2]
                track_events=_track_changes(previous_detections,detections_summary,width,height)
                previous_detections=detections_summary
                detector_calls += 1
                detector_total += detections_summary["count"]
                detector_truncated += detections_summary["truncated_by_scorer"]
                next_detection_s = record.timestamp_s + detector_period

        samples.append(_SampleMeta(
            sequence=record.sequence,
            pts=record.pts,
            time_base=record.time_base,
            timestamp_s=record.timestamp_s,
            motion=motion,
            clarity=clarity,
            camera=camera,
            detections=detections_summary,
            scene_cut_score=cut_score,track_events=track_events,
        ))

    if not samples:
        raise VideoError(f"video source decoded zero frames: {source.path}")
    detector_report = None
    if detector is not None:
        detector_report = {
            "name": getattr(detector, "name", type(detector).__name__),
            "detector_hz": detector_hz,
            "analyzed_frames": detector_calls,
            "total_detections": detector_total,
            "truncated_by_scorer": detector_truncated,
        }
    return samples, detector_report, compensated


# ---------------------------------------------------------------------------
# selection (pure; unit-testable without cv2)


def _window_index(timestamp_s: float, window_s: float) -> int:
    return int(timestamp_s // window_s)


def _validate_budget(window_s: float, max_per_window: int, baseline_hz: float) -> None:
    """Fail up front when mandatory + baseline can never fit the window budget."""
    baseline_ticks_per_window = int(window_s * baseline_hz) + 1
    required = baseline_ticks_per_window + 2  # + first/last
    if required > max_per_window:
        raise ValueError(
            f"window budget cannot hold mandatory frames: baseline_hz={baseline_hz} "
            f"over window_s={window_s} needs up to {baseline_ticks_per_window} baseline "
            f"samples plus first/last = {required} > max_per_window={max_per_window}. "
            "Raise max_per_window or lower baseline_hz; nothing was silently dropped."
        )


def _nearest_before(ordered: list[dict], timestamps: list[float], index: int,
                    target_s: float) -> dict | None:
    """Sampled frame strictly before ``index`` nearest to ``target_s``."""
    if index <= 0:
        return None
    pos = bisect.bisect_left(timestamps, target_s, 0, index)
    candidates = []
    if pos < index:
        candidates.append(ordered[pos])
    if pos > 0:
        candidates.append(ordered[pos - 1])
    if not candidates:
        return None
    return min(candidates, key=lambda s: (abs(s["timestamp_s"] - target_s), s["timestamp_s"]))


def _nearest_after(ordered: list[dict], timestamps: list[float], index: int,
                   target_s: float) -> dict | None:
    """Sampled frame strictly after ``index`` nearest to ``target_s``."""
    if index + 1 >= len(ordered):
        return None
    pos = bisect.bisect_left(timestamps, target_s, index + 1, len(ordered))
    candidates = []
    if pos < len(ordered):
        candidates.append(ordered[pos])
    if pos > index + 1:
        candidates.append(ordered[pos - 1])
    if not candidates:
        return None
    return min(candidates, key=lambda s: (abs(s["timestamp_s"] - target_s), s["timestamp_s"]))


def _plan_selection(
    samples: list[dict],
    *,
    window_s: float,
    max_per_window: int,
    baseline_hz: float,
) -> tuple[dict[int, list[str]], list[dict]]:
    """Pure selection planner over ``{sequence, timestamp_s, motion}`` dicts.

    Returns (reasons_by_sequence, dropped). Sequences may be sparse or
    non-consecutive: event before/after neighbours are the *sampled* frames
    nearest to event_time -/+ ``NEIGHBOUR_OFFSET_S`` (strictly before /
    strictly after the event), never ``sequence +/- 1``.

    Budget accounting: mandatory (first/last) and baseline frames are never
    dropped (up-front validation guarantees they fit). Event frames are kept
    highest-motion first until the event's window is full; neighbours are
    charged against their *own* timestamp window, so ``max_per_window`` is a
    hard invariant per window even at boundaries. Every event or neighbour
    that does not fit lands in ``dropped`` with an explicit reason;
    neighbours that do not exist (video start/end) are reported as
    ``no_sample_exists``.
    """
    if not samples:
        return {}, []
    _validate_budget(window_s, max_per_window, baseline_hz)

    ordered = sorted(samples, key=lambda s: s["timestamp_s"])
    timestamps = [s["timestamp_s"] for s in ordered]
    position = {s["sequence"]: i for i, s in enumerate(ordered)}
    first_seq, last_seq = ordered[0]["sequence"], ordered[-1]["sequence"]

    reasons: dict[int, list[str]] = {}
    counts: dict[int, int] = {}

    def keep(seq: int, reason: str) -> None:
        if seq not in reasons:
            reasons[seq] = []
            window = _window_index(ordered[position[seq]]["timestamp_s"], window_s)
            counts[window] = counts.get(window, 0) + 1
        if reason not in reasons[seq]:
            reasons[seq].append(reason)

    keep(first_seq, "first")
    keep(last_seq, "last")

    # Periodic baseline: nearest sampled frame to each baseline tick.
    duration_s = ordered[-1]["timestamp_s"] - ordered[0]["timestamp_s"]
    tick_count = int(math.floor(duration_s * baseline_hz)) + 1
    for k in range(tick_count + 1):
        tick_s = ordered[0]["timestamp_s"] + k / baseline_hz
        if tick_s > ordered[-1]["timestamp_s"]:
            break
        nearest = min(ordered, key=lambda s: abs(s["timestamp_s"] - tick_s))
        keep(nearest["sequence"], "baseline")

    # Motion events per window: motion >= max(floor, median + 2*scaled MAD).
    # Median/MAD stays robust when a motion burst fills a large fraction of a
    # window, where mean + 2*std would swallow exactly the events we want.
    windows: dict[int, list[dict]] = {}
    for s in ordered:
        windows.setdefault(_window_index(s["timestamp_s"], window_s), []).append(s)

    dropped: list[dict] = []
    for window_idx in sorted(windows):
        members = windows[window_idx]
        motions = [s["motion"] for s in members if s["motion"] is not None]
        events: list[dict] = []
        if motions:
            median = statistics.median(motions)
            mad = statistics.median([abs(m - median) for m in motions])
            threshold = max(MOTION_EVENT_FLOOR, median + 2.0 * 1.4826 * mad)
            events = [s for s in members if s["motion"] is not None and s["motion"] >= threshold]
        chosen={s['sequence']:s for s in events}
        for s in members:
            if s.get('scene_cut_score',0)>0 or s.get('track_events'):chosen[s['sequence']]=s
        events=list(chosen.values())
        events.sort(key=lambda s:(-max(s['motion'] or 0,s.get('scene_cut_score',0),.5 if s.get('track_events') else 0),s['sequence']))

        slots = max_per_window - counts.get(window_idx, 0)
        # slots >= 0 for baseline+mandatory is guaranteed by _validate_budget.
        for event in events:
            event_seq = event["sequence"]
            event_ts = event["timestamp_s"]
            if event_seq not in reasons and slots <= 0:
                dropped.append({
                    "kind": "event",
                    "sequence": event_seq,
                    "timestamp_s": event_ts,
                    "motion": event["motion"],
                    "window": window_idx,
                    "reason": "window_budget",
                    "event_kinds":['scene_cut_candidate'] if event.get('scene_cut_score',0)>0 else [e['kind'] for e in event.get('track_events',[])],
                    "detail": (
                        f"event at {event_ts:.3f}s did not fit "
                        f"window {window_idx} (max_per_window={max_per_window})"
                    ),
                })
                continue
            if event_seq not in reasons:
                slots -= 1
            keep(event_seq, "event")
            if event.get('scene_cut_score',0)>0:keep(event_seq,'scene_cut_after')
            if event.get('track_events'):keep(event_seq,'track_change')

            index = position[event_seq]
            neighbours = (
                ("event_before", _nearest_before(ordered, timestamps, index,
                                                 event_ts - NEIGHBOUR_OFFSET_S)),
                ("event_after", _nearest_after(ordered, timestamps, index,
                                               event_ts + NEIGHBOUR_OFFSET_S)),
            )
            if event.get('scene_cut_score',0)>0:
                neighbours+=(('scene_cut_before',ordered[index-1] if index else None),)
            for tag, neighbour in neighbours:
                side = "before" if tag.endswith('before') else "after"
                if neighbour is None:
                    dropped.append({
                        "kind": "event_neighbour",
                        "event_sequence": event_seq,
                        "event_timestamp_s": event_ts,
                        "side": side,
                        "sequence": None,
                        "timestamp_s": None,
                        "window": None,
                        "reason": "no_sample_exists",
                        "detail": f"event at {event_ts:.3f}s has no {side} sample "
                                  "(video boundary); reported, not substituted",
                    })
                    continue
                if neighbour["sequence"] in reasons:
                    keep(neighbour['sequence'],tag)
                    continue  # already kept for another reason; no extra slot
                nwin = _window_index(neighbour["timestamp_s"], window_s)
                if nwin == window_idx:
                    if slots <= 0:
                        dropped.append({
                            "kind": "event_neighbour",
                            "event_sequence": event_seq,
                            "event_timestamp_s": event_ts,
                            "side": side,
                            "sequence": neighbour["sequence"],
                            "timestamp_s": neighbour["timestamp_s"],
                            "window": nwin,
                            "reason": "window_budget",
                            "detail": f"{side} neighbour of event at {event_ts:.3f}s "
                                      f"did not fit window {window_idx}",
                        })
                        continue
                    slots -= 1
                elif counts.get(nwin, 0) >= max_per_window:
                    dropped.append({
                        "kind": "event_neighbour",
                        "event_sequence": event_seq,
                        "event_timestamp_s": event_ts,
                        "side": side,
                        "sequence": neighbour["sequence"],
                        "timestamp_s": neighbour["timestamp_s"],
                        "window": nwin,
                        "reason": "window_budget",
                        "detail": f"{side} neighbour of event at {event_ts:.3f}s "
                                  f"did not fit its own window {nwin}",
                    })
                    continue
                keep(neighbour["sequence"], tag)

    return reasons, dropped


# ---------------------------------------------------------------------------
# pass 2: materialize selected frames (bounded: one full-res frame at a time)


def _materialize(
    source: VideoSource,
    selected: dict[int, tuple[int, str]],
    staging_frames_dir: Path,
    final_frames_dir: Path,
    budget: CandidateBudget,
) -> dict[int, str]:
    """Decode again and write JPEGs for ``selected`` {sequence: (pts, time_base)}.

    The pass-2 PTS at each selected sequence must equal the pass-1 PTS; any
    drift fails loudly. Returns sequence -> final (post-rename) absolute
    image path. Timestamps are never re-derived; pass-1 values stand.
    """
    remaining = dict(selected)
    written: dict[int, str] = {}
    for record in source.frames():
        if record.timestamp_s > budget.max_duration_s:
            raise VideoError(
                f"source duration exceeds budget {budget.max_duration_s}s; "
                "second decode disagrees with the budgeted extent"
            )
        expected = remaining.pop(record.sequence, None)
        if expected is None:
            continue
        expected_pts, expected_time_base = expected
        if record.pts != expected_pts or record.time_base != expected_time_base:
            raise VideoError(
                f"decode drift between passes at sequence {record.sequence}: "
                f"pass 1 recorded pts={expected_pts} {expected_time_base}, pass 2 "
                f"decoded pts={record.pts} {record.time_base}; refusing to guess"
            )
        frame_id = f"f{record.sequence:06d}"
        selected_image=Image.fromarray(record.image)
        selected_image.thumbnail((768,768))
        selected_image.save(
            staging_frames_dir / f"{frame_id}.jpg", format="JPEG", quality=90
        )
        # Record the path the file will have after staging is renamed into
        # place, so the manifest stays valid after commit.
        written[record.sequence] = str((final_frames_dir / f"{frame_id}.jpg").absolute())
    if remaining:
        raise VideoError(
            f"second decode ended before sequences {sorted(remaining)}; "
            "the source changed between passes or is undecodable"
        )
    return written


# ---------------------------------------------------------------------------
# entry point


def build_candidates(
    path: str,
    output_dir: str,
    window_s: float = 10,
    max_per_window: int = 24,
    analysis_hz: float = 10,
    baseline_hz: float = 1,
    *,
    detector=None,
    detector_hz: float = 2,
    budget: CandidateBudget | None = None,
) -> dict:
    """Analyse ``path`` and persist budgeted candidate frames + manifest.

    ``detector`` is an optional scorer conforming to the perception
    ``Scorer.score(image, source_id=..., sequence=..., source_s=...)``
    protocol (e.g. root's ONNX detector); it runs at ``detector_hz`` on
    full-resolution sampled frames and only bounded summaries are kept.
    Works without any detector and never imports torch.

    Returns the manifest dict (also written to ``output_dir/manifest.json``).
    All candidate image paths in the manifest are absolute and valid after
    the atomic commit. The source file is retained in place and referenced
    by path + SHA-256; it is never modified or copied.
    """
    window_s = _check_float("window_s", window_s)
    analysis_hz = _check_float("analysis_hz", analysis_hz)
    baseline_hz = _check_float("baseline_hz", baseline_hz)
    max_per_window = _check_int("max_per_window", max_per_window)
    detector_hz = _check_float("detector_hz", detector_hz)
    if budget is None:
        budget = CandidateBudget()
    elif not isinstance(budget, CandidateBudget):
        raise ValueError("budget must be a CandidateBudget")
    if detector is not None and not callable(getattr(detector, "score", None)):
        raise ValueError("detector must expose a callable .score(image, source_id=..., "
                         "sequence=..., source_s=...) per the perception Scorer protocol")
    _validate_budget(window_s, max_per_window, baseline_hz)

    cv2 = _require_cv2()
    source = VideoSource(path)

    # pass 1: streaming analysis, bounded retention
    samples, detector_report, compensated = _analyze(
        source, cv2, analysis_hz=analysis_hz, detector=detector,
        detector_hz=detector_hz, budget=budget,
    )

    reasons, dropped = _plan_selection(
        [
            {"sequence": s.sequence, "timestamp_s": s.timestamp_s, "motion": s.motion,
             'scene_cut_score':s.scene_cut_score,'track_events':s.track_events}
            for s in samples
        ],
        window_s=window_s,
        max_per_window=max_per_window,
        baseline_hz=baseline_hz,
    )

    output = Path(output_dir).absolute()
    _fail_if_exists(output)
    staging = output.parent / (output.name + f".tmp-{os.getpid()}")
    if staging.exists():
        raise FileExistsError(f"stale staging directory in the way: {staging}")
    staging.mkdir(parents=True)
    frames_dir = staging / "frames"
    frames_dir.mkdir()

    # pass 2: materialize only the selected frames, verifying PTS identity
    selected = {
        s.sequence: (s.pts, s.time_base) for s in samples if s.sequence in reasons
    }
    image_paths = _materialize(source, selected, frames_dir, output / "frames", budget)

    neighbour_drop_counts: dict[int, int] = {}
    for entry in dropped:
        if entry["kind"] == "event_neighbour":
            seq = entry["event_sequence"]
            neighbour_drop_counts[seq] = neighbour_drop_counts.get(seq, 0) + 1

    candidates: list[dict] = []
    for s in samples:
        if s.sequence not in reasons:
            continue
        candidates.append({
            "frame_id": f"f{s.sequence:06d}",
            "sequence": s.sequence,
            "pts": s.pts,
            "time_base": s.time_base,
            "timestamp_s": s.timestamp_s,
            "image_path": image_paths[s.sequence],
            "scores": {
                "motion": s.motion,
                "clarity": s.clarity,
                "camera": s.camera,
                "scene_cut_score":s.scene_cut_score,
            },
            'track_events':s.track_events,
            "detections": s.detections,
            "reasons": sorted(reasons[s.sequence]),
            "event_neighbours_dropped": neighbour_drop_counts.get(s.sequence, 0),
            "window": _window_index(s.timestamp_s, window_s),
        })

    windows_hit = {c["window"] for c in candidates}
    motion_known = [s for s in samples if s.motion is not None]
    manifest = {
        "schema_version": 1,
        "kind": "piperlab.video.candidates",
        "created_by": "piperlab.video.candidates.build_candidates",
        "source": {"path": str(source.path), "sha256": source.sha256},
        "parameters": {
            'candidate_version':'1.4',
            'scene_cut_histogram_l1_threshold':.6,'scene_cut_mean_difference_threshold':.12,
            'track_normalized_displacement_threshold':.04,'track_log_area_ratio_threshold':.3,
            'include_actual_final_frame':True,
            "window_s": window_s,
            "max_per_window": max_per_window,
            "analysis_hz": analysis_hz,
            "baseline_hz": baseline_hz,
            "detector_hz": detector_hz if detector is not None else None,
            "detector": detector_report["name"] if detector_report else None,
            "scoring_long_edge_px": SCORING_LONG_EDGE,
            "motion_event_floor": MOTION_EVENT_FLOOR,
            "neighbour_offset_s": NEIGHBOUR_OFFSET_S,
        },
        "analysis": {
            "sampled_frames": len(samples),
            "duration_s": samples[-1].timestamp_s,
            "windows": len(windows_hit),
            "camera_compensation": (
                "applied" if compensated == len(motion_known) and motion_known
                else "partial" if compensated
                else "unavailable"
            ),
            "camera_compensated_frames": compensated,
            "scoring": "opencv absdiff motion + laplacian clarity on CPU; "
                       "best-effort RANSAC partial-affine camera compensation; "
                       "streaming two-pass decode, full-resolution frames are "
                       "never retained",
        },
        "detector": detector_report,
        "candidates": candidates,
        "dropped": dropped,
        "budget": {
            "max_per_window": max_per_window,
            "dropped_event_candidates": sum(1 for d in dropped if d["kind"] == "event"),
            "dropped_event_neighbours": sum(
                1 for d in dropped if d["kind"] == "event_neighbour"
            ),
            "per_window_candidate_counts": {
                str(w): sum(1 for c in candidates if c["window"] == w)
                for w in sorted(windows_hit)
            },
            "analysis_budget": budget.as_dict(),
        },
    }
    index=source.frame_index()
    _write_json_atomic(staging / 'source-index.json',index)
    manifest['source']['index']='source-index.json'
    _write_json_atomic(staging / "manifest.json", manifest)
    os.rename(staging, output)
    return manifest
