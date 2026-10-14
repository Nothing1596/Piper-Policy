"""Bounded live-frame selection: endpoints and visual change, never contact claims.

Uses the offline video selector's small grayscale difference/clarity idea. Live
sources cannot be decoded twice, so compressed RGB and aligned depth are kept
in a short ring instead of importing the independent video distribution.
"""
from collections import deque
from dataclasses import dataclass
import math

import numpy as np


def positive(value, name, maximum):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 < value <= maximum:
        raise ValueError(f'{name} must be >0 and <= {maximum}')
    return value


def checked_roi(roi, width, height):
    if roi is None:
        return None
    if not isinstance(roi, (list, tuple)) or len(roi) != 4 or any(type(v) is not int for v in roi):
        raise ValueError('roi must be [left, top, right, bottom] in integer pixels')
    x1, y1, x2, y2 = roi
    if not 0 <= x1 < x2 <= width or not 0 <= y1 < y2 <= height:
        raise ValueError('roi outside image')
    return tuple(roi)


@dataclass
class Frame:
    sequence: int
    received_mono_s: float
    metadata: dict
    jpeg: bytes
    depth: np.ndarray
    gray: np.ndarray
    clarity: float

    @property
    def nbytes(self):
        return len(self.jpeg) + self.depth.nbytes + self.gray.nbytes


class FrameRing:
    def __init__(self, seconds=20, max_frames=60, max_bytes=128 * 1024 * 1024):
        positive(seconds, 'buffer_s', 120)
        if type(max_frames) is not int or not 2 <= max_frames <= 360:
            raise ValueError('max_frames must be 2..360')
        self.seconds, self.max_frames, self.max_bytes = seconds, max_frames, max_bytes
        self.frames = deque()
        self.bytes = 0
        self.dropped = 0

    def add(self, frame):
        if frame.nbytes > self.max_bytes:
            raise ValueError('one frame exceeds buffer memory budget')
        self.frames.append(frame)
        self.bytes += frame.nbytes
        while len(self.frames) > self.max_frames or self.bytes > self.max_bytes or frame.received_mono_s - self.frames[0].received_mono_s > self.seconds:
            self.bytes -= self.frames.popleft().nbytes
            self.dropped += 1


def select(frames, *, since_mono_s, now_mono_s, max_frames=3, roi=None, max_age_s=2):
    positive(max_age_s, 'max_age_s', 10)
    if type(max_frames) is not int or not 1 <= max_frames <= 6:
        raise ValueError('max_frames must be 1..6')
    if not frames or now_mono_s - frames[-1].received_mono_s > max_age_s:
        raise ValueError('camera_stale: no fresh video frame; restart /video explicitly')
    latest = frames[-1]
    height, width = latest.depth.shape
    roi = checked_roi(roi, width, height)
    candidates = [f for f in frames if f.received_mono_s >= since_mono_s]
    if not candidates:
        raise ValueError('no frames in requested interval')
    coverage = {'requested_since_mono_s': since_mono_s,
                'first_mono_s': candidates[0].received_mono_s,
                'last_mono_s': candidates[-1].received_mono_s,
                'history_truncated': frames[0].received_mono_s > since_mono_s + .5,
                'latest_age_s': max(0., now_mono_s - latest.received_mono_s)}
    if max_frames == 1 or len(candidates) == 1:
        return [(latest, 'latest', 0.)], coverage
    picked = {0: ('before', 0.), len(candidates)-1: ('after', 0.)}
    scores = []
    for i in range(1, len(candidates)-1):
        a, b = candidates[i-1].gray, candidates[i].gray
        if roi:
            x1, y1, x2, y2 = roi
            gh, gw = b.shape
            ys = slice(int(y1*gh/height), max(int(y1*gh/height)+1, int(y2*gh/height)))
            xs = slice(int(x1*gw/width), max(int(x1*gw/width)+1, int(x2*gw/width)))
            a, b = a[ys, xs], b[ys, xs]
        motion = float(np.abs(b.astype(float)-a.astype(float)).mean()/255)
        scores.append((motion, candidates[i].clarity, i))
    for motion, clarity, i in sorted(scores, reverse=True):
        if len(picked) >= max_frames:
            break
        if motion < .02 or any(abs(candidates[i].received_mono_s-candidates[j].received_mono_s) < .3 for j in picked):
            continue
        picked[i] = ('visual_change', motion)
    return [(candidates[i], *picked[i]) for i in sorted(picked)], coverage
