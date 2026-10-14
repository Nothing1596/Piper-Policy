"""Bounded human-video source access (human-video v1).

Rules implemented here:

- every timestamp is ``pts * time_base`` of the decoded frame; frame index,
  container FPS hints and the wall clock are never substituted for a missing
  or implausible timestamp;
- frames with missing PTS are rejected, never stamped from anything else;
- non-monotonic decoded PTS rejects the whole source;
- rotation metadata is accounted: v1 explicitly rejects any non-zero rotation
  instead of silently returning unrotated pixels;
- PyAV is imported lazily, so ``import piperlab.video`` works in the core
  environment without the video extras installed.

``at()`` builds a bounded timestamp/keyframe index, seeks backwards to a
keyframe, and verifies the actual decoded PTS against that index.
"""
from __future__ import annotations

import hashlib
import bisect
import math
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np


class VideoError(RuntimeError):
    """A video source operation failed explicitly; nothing was repaired."""


def _require_av():
    """Import PyAV lazily; the video extra is not a core dependency."""
    try:
        import av  # noqa: PLC0415 - intentional lazy import
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise VideoError(
            "PyAV (package 'av') is required for video decoding; "
            "install the piper-lab video extras first"
        ) from exc
    return av


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class _PtsGuard:
    """Strictly-increasing decoded-PTS guard.

    Duplicate or backwards PTS means the container's timing cannot be trusted
    for ordering evidence, so the source is rejected rather than repaired.
    """

    def __init__(self) -> None:
        self._last: int | None = None

    def check(self, pts: int) -> int:
        if isinstance(pts, bool) or not isinstance(pts, int):
            raise VideoError(f"decoded PTS must be an int, got {pts!r}")
        if self._last is not None and pts <= self._last:
            raise VideoError(
                f"non-monotonic decoded PTS: {pts} after {self._last}; "
                "the source is rejected, timestamps are not invented"
            )
        self._last = pts
        return pts


def _require_pts(pts, sequence: int) -> int:
    if pts is None:
        raise VideoError(
            f"decoded frame at decode order {sequence} has no PTS; "
            "refusing to substitute a frame index, FPS hint or wall clock"
        )
    return pts


def _rotation_degrees(container, stream) -> float:
    """Best-effort rotation probe.

    Checks the classic ``rotate`` metadata tag on stream and container, then
    display-matrix side data when the installed PyAV exposes it. Returns 0.0
    when no rotation signal is found.
    """
    for meta_holder in (stream, container):
        metadata = getattr(meta_holder, "metadata", None) or {}
        raw = metadata.get("rotate")
        if raw is None:
            continue
        try:
            return float(raw)
        except (TypeError, ValueError) as exc:
            raise VideoError(f"unparseable 'rotate' metadata value: {raw!r}") from exc
    side_data = getattr(stream, "side_data", None)
    if side_data:
        for item in side_data:
            rotation = getattr(item, "rotation", None)
            if rotation is not None:
                return float(rotation)
    return 0.0


def _check_rotation(container, stream) -> float:
    """Reject any non-zero rotation; v1 does not transpose pixels.

    PyAV does not apply display-matrix rotation to decoded frames, so
    pretending the pixels are upright would corrupt every downstream
    geometric claim. Rejection is the honest behaviour.
    """
    rotation = _rotation_degrees(container, stream)
    if not math.isfinite(rotation):
        raise VideoError(f"non-finite rotation metadata: {rotation!r}")
    if rotation % 360.0 != 0.0:
        raise VideoError(
            f"unsupported rotation metadata: {rotation:g} degrees; "
            "human-video v1 rejects rotated sources instead of returning "
            "unrotated pixels with a wrong geometry claim"
        )
    return rotation


@dataclass(frozen=True)
class FrameRecord:
    """One decoded frame with its container-true timestamp."""

    sequence: int
    """Decode order, 0-based. Informational only; never used as a timestamp."""

    pts: int
    """Raw presentation timestamp in ``time_base`` units."""

    time_base: str
    """Rational time base as ``"num/den"``, e.g. ``"1/15360"``."""

    timestamp_s: float
    """``pts * time_base`` in seconds, computed via exact Fraction math."""

    image: np.ndarray
    """RGB uint8 HWC, per the AGENTS.md image contract."""

    def __post_init__(self) -> None:
        if isinstance(self.sequence, bool) or not isinstance(self.sequence, int) or self.sequence < 0:
            raise VideoError("FrameRecord.sequence must be a non-negative int")
        if isinstance(self.pts, bool) or not isinstance(self.pts, int):
            raise VideoError("FrameRecord.pts must be an int")
        if not isinstance(self.time_base, str) or "/" not in self.time_base:
            raise VideoError("FrameRecord.time_base must look like 'num/den'")
        if not math.isfinite(self.timestamp_s) or self.timestamp_s < 0:
            raise VideoError("FrameRecord.timestamp_s must be finite and >= 0")
        if not isinstance(self.image, np.ndarray) or self.image.ndim != 3 \
                or self.image.shape[2] != 3 or self.image.dtype != np.uint8:
            raise VideoError("FrameRecord.image must be an RGB uint8 HWC ndarray")
        # Verify the declared timestamp really is pts * time_base.
        num, den = self.time_base.split("/", 1)
        expected = float(Fraction(self.pts) * Fraction(int(num), int(den)))
        if abs(expected - self.timestamp_s) > 1e-12:
            raise VideoError(
                "FrameRecord.timestamp_s does not equal pts * time_base; "
                "timestamps must not be derived from anything else"
            )


class VideoSource:
    """One local video file, decoded with strict timestamp validity."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path).absolute()
        if not self._path.is_file():
            raise VideoError(f"video source does not exist: {self._path}")
        self._sha256 = _sha256_file(self._path)
        stat=self._path.stat()
        self._fingerprint=(stat.st_size,stat.st_mtime_ns)
        self._index_cache=None

    @property
    def path(self) -> Path:
        return self._path

    @property
    def sha256(self) -> str:
        """SHA-256 of the exact file bytes; part of every downstream identity."""
        return self._sha256

    # -- internal ---------------------------------------------------------
    def _open(self):
        stat=self._path.stat()
        if (stat.st_size,stat.st_mtime_ns)!=self._fingerprint:
            raise VideoError('video source changed after hashing')
        av = _require_av()
        try:
            container = av.open(str(self._path))
        except Exception as exc:
            raise VideoError(f"cannot open video source {self._path}: {exc}") from exc
        video_streams = [s for s in container.streams if s.type == "video"]
        if not video_streams:
            container.close()
            raise VideoError(f"no video stream in {self._path}")
        return container, video_streams[0]

    def _decode(self, sample_hz: float | None, include_last=False):
        container, stream = self._open()
        try:
            _check_rotation(container, stream)
            guard = _PtsGuard()
            sequence = 0
            next_yield_s: float | None = None
            period_s = (1.0 / sample_hz) if sample_hz is not None else None
            pending=None
            last_yield_sequence=None
            last_timestamp=None
            for frame in container.decode(stream):
                rotation=float(getattr(frame,'rotation',0))
                if not math.isfinite(rotation) or abs(rotation%360)>1e-6:
                    raise VideoError('non-zero decoded-frame rotation is unsupported; refusing unrotated evidence')
                pts = guard.check(_require_pts(frame.pts, sequence))
                time_base = frame.time_base or stream.time_base
                if time_base is None:
                    raise VideoError("decoded frame and stream both lack a time base")
                tb = Fraction(time_base.numerator, time_base.denominator)
                if tb<=0:raise VideoError('decoded time base must be positive')
                timestamp_s = float(Fraction(pts) * tb)
                if last_timestamp is not None and timestamp_s<=last_timestamp:
                    raise VideoError('non-monotonic decoded source time')
                last_timestamp=timestamp_s
                pending=(frame,sequence,pts,tb,timestamp_s)
                if period_s is not None and next_yield_s is not None \
                        and timestamp_s < next_yield_s - 1e-12:
                    sequence += 1
                    continue
                image = frame.to_ndarray(format="rgb24")
                record = FrameRecord(
                    sequence=sequence,
                    pts=pts,
                    time_base=f"{tb.numerator}/{tb.denominator}",
                    timestamp_s=timestamp_s,
                    image=image,
                )
                yield record
                last_yield_sequence=sequence
                sequence += 1
                if period_s is not None:
                    next_yield_s = timestamp_s + period_s
            if include_last and pending is not None and pending[1]!=last_yield_sequence:
                frame,seq,pts,tb,stamp=pending
                yield FrameRecord(sequence=seq,pts=pts,time_base=f'{tb.numerator}/{tb.denominator}',
                                  timestamp_s=stamp,image=frame.to_ndarray(format='rgb24'))
        finally:
            container.close()

    # -- public -----------------------------------------------------------
    def frames(self, sample_hz: float | None = None, *, include_last=False):
        """Yield decoded frames in order.

        ``sample_hz`` bounds the yield rate: the first frame is always
        yielded, then the first frame whose true timestamp is at least
        ``1/sample_hz`` after the previously yielded one. Sub-sampling never
        rewrites timestamps. ``include_last=True`` additionally retains the real
        final frame even if it falls less than one sampling period after a sample.
        """
        if sample_hz is not None:
            if isinstance(sample_hz, bool) or not isinstance(sample_hz, (int, float)) \
                    or not math.isfinite(sample_hz) or sample_hz <= 0:
                raise ValueError("sample_hz must be a positive finite number")
            sample_hz = float(sample_hz)
        yield from self._decode(sample_hz,include_last=include_last)

    def frame_index(self,max_frames=180000):
        """Exact decoded PTS and keyframe flags; no RGB arrays retained."""
        if self._index_cache is None:
            container,stream=self._open();rows=[]
            guard=_PtsGuard()
            try:
                _check_rotation(container,stream)
                for sequence,frame in enumerate(container.decode(stream)):
                    if sequence>=max_frames:raise VideoError('frame index budget exceeded')
                    rotation=float(getattr(frame,'rotation',0))
                    if not math.isfinite(rotation) or abs(rotation%360)>1e-6:
                        raise VideoError('unsupported decoded-frame rotation')
                    pts=guard.check(_require_pts(frame.pts,sequence))
                    tb=frame.time_base or stream.time_base
                    if tb is None or tb<=0:raise VideoError('missing or invalid frame time base')
                    stamp=float(pts*tb)
                    if stamp<0 or (rows and stamp<=rows[-1]['timestamp_s']):
                        raise VideoError('non-monotonic or negative source timestamp')
                    rows.append({'sequence':sequence,'pts':pts,'time_base':f'{tb.numerator}/{tb.denominator}',
                                 'timestamp_s':stamp,'key_frame':bool(frame.key_frame)})
            finally:container.close()
            if not rows:raise VideoError('video source decoded zero frames')
            self._index_cache=rows
        if len(self._index_cache)>max_frames:raise VideoError('frame index budget exceeded')
        return {'source_hash':self.sha256,'frames':[dict(row) for row in self._index_cache],
                'first_timestamp_s':self._index_cache[0]['timestamp_s'],
                'last_timestamp_s':self._index_cache[-1]['timestamp_s'],'max_frames':max_frames}

    def at(self, timestamp_s: float) -> FrameRecord:
        """Return the first decoded frame whose actual PTS is >= ``timestamp_s``.

        Builds a bounded index once, then seeks from an earlier keyframe.
        Requests before the first frame or beyond the last frame raise
        VideoError ("outside source"); nothing is extrapolated.
        """
        if isinstance(timestamp_s, bool) or not isinstance(timestamp_s, (int, float)) \
                or not math.isfinite(timestamp_s) or timestamp_s < 0:
            raise VideoError(f"requested timestamp is not a finite >= 0 number: {timestamp_s!r}")
        requested = float(timestamp_s)
        if self._index_cache is None:self.frame_index()
        rows=self._index_cache
        if requested<rows[0]['timestamp_s']-1e-12:raise VideoError('requested time before the first decoded frame')
        times=[row['timestamp_s'] for row in rows]
        selected=bisect.bisect_left(times,requested-1e-12)
        if selected>=len(rows):raise VideoError('requested time beyond the last decoded frame')
        expected=rows[selected]
        key=next((rows[i] for i in range(selected,-1,-1) if rows[i]['key_frame']),rows[0])
        container,stream=self._open()
        try:
            offset=int(Fraction(key['pts'])*Fraction(key['time_base'])/stream.time_base)
            container.seek(offset,stream=stream,backward=True,any_frame=False)
            guard=_PtsGuard()
            for frame in container.decode(stream):
                pts=guard.check(_require_pts(frame.pts,0));tb=frame.time_base or stream.time_base
                stamp=float(pts*tb)
                if stamp<expected['timestamp_s']-1e-12:continue
                if pts!=expected['pts'] or str(tb)!=expected['time_base']:
                    raise VideoError('seek result does not match indexed PTS')
                return FrameRecord(sequence=expected['sequence'],pts=pts,time_base=expected['time_base'],
                                   timestamp_s=stamp,image=frame.to_ndarray(format='rgb24'))
        finally:container.close()
        raise VideoError('seek ended before indexed target frame')

    def metadata(self) -> dict:
        """Container facts. No frame is decoded; unknowns stay ``None``."""
        container, stream = self._open()
        try:
            rotation = _check_rotation(container, stream)
            time_base = stream.time_base
            average_rate = getattr(stream, "average_rate", None)
            duration_s = None
            if container.duration is not None:
                duration_s = float(container.duration) / 1_000_000.0
            reported_frames = getattr(stream, "frames", None)
            return {
                "path": str(self._path),
                "sha256": self._sha256,
                "codec": getattr(stream.codec_context, "name", None),
                "width": getattr(stream.codec_context, "width", None),
                "height": getattr(stream.codec_context, "height", None),
                "time_base": (
                    f"{time_base.numerator}/{time_base.denominator}" if time_base else None
                ),
                # Informational container hints only; never used for timestamps.
                "container_duration_s": duration_s,
                "reported_frame_count": reported_frames if reported_frames else None,
                "average_rate_hint": str(average_rate) if average_rate else None,
                "rotation_degrees": rotation,
                "rotation_policy": "reject_nonzero",
                "timestamp_policy": "pts*time_base only; no fps-derived timestamps",
            }
        finally:
            container.close()
