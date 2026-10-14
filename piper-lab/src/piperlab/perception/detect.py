"""Cheap local scorers.

Stage 1 deliberately ships no learned detector. Adding one is a separate
decision with its own measurable purpose. The cheap filters here are
deterministic, CPU-only and testable offline, so the surrounding infrastructure
(buffer, selection report, evidence) can be proven before any model enters the
loop.

Scorers are *not* allowed to hold veto power over raw data. They produce scores
and hints; the selector decides what to forward, and every drop is reported.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Protocol, Sequence, runtime_checkable

import numpy as np


class DetectionError(ValueError):
    """A detection or scorer input is malformed."""


@dataclass(frozen=True)
class Detection:
    """One candidate region proposed by a cheap scorer.

    ``source_id``/``sequence`` tie the detection back to a retained frame so a
    downstream consumer can always ask for the pixels behind it.
    """

    source_id: str
    sequence: int
    source_s: float
    scorer: str
    label: str
    score: float
    bbox_xyxy: tuple[float, float, float, float]
    centroid_xy: tuple[float, float]
    area_px: float = 0.0
    detail: dict = field(default_factory=dict)
    truncated_by_scorer: int = 0
    """How many further candidates this scorer itself capped away.

    Kept as a first-class field rather than a free-form detail key: a scorer
    limit is a real information loss and must stay visible even when the
    surviving list is empty.
    """

    def __post_init__(self) -> None:
        if not self.source_id:
            raise DetectionError("Detection.source_id is required")
        if not self.scorer:
            raise DetectionError("Detection.scorer is required")
        if isinstance(self.score, bool) or not isinstance(self.score, (int, float)) \
                or not math.isfinite(self.score):
            raise DetectionError("Detection.score must be finite")
        if len(self.bbox_xyxy) != 4 or not all(math.isfinite(v) for v in self.bbox_xyxy):
            raise DetectionError("Detection.bbox_xyxy must be four finite numbers")
        x0, y0, x1, y1 = self.bbox_xyxy
        if x1 < x0 or y1 < y0:
            raise DetectionError("Detection.bbox_xyxy must satisfy x1>=x0 and y1>=y0")
        if len(self.centroid_xy) != 2 or not all(math.isfinite(v) for v in self.centroid_xy):
            raise DetectionError("Detection.centroid_xy must be two finite numbers")

    @property
    def entity_hint(self) -> str:
        """Stable-ish hint used only to seed identity association, never an ID."""
        return f"{self.scorer}:{self.label}"

    def describe(self) -> dict:
        return {
            "source_id": self.source_id,
            "sequence": self.sequence,
            "source_s": self.source_s,
            "scorer": self.scorer,
            "label": self.label,
            "score": self.score,
            "bbox_xyxy": list(self.bbox_xyxy),
            "centroid_xy": list(self.centroid_xy),
            "area_px": self.area_px,
            "detail": dict(self.detail),
        }


@runtime_checkable
class Scorer(Protocol):
    """A cheap, dependency-light proposal source."""

    name: str

    def score(self, image: np.ndarray, *, source_id: str, sequence: int, source_s: float) -> list[Detection]:
        ...


def _as_rgb_uint8(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3:
        raise DetectionError("expected an HWC RGB image")
    if array.dtype != np.uint8:
        raise DetectionError("expected uint8 RGB (the AGENTS.md image contract)")
    if array.size == 0:
        raise DetectionError("empty image")
    return array


def _label_components(mask: np.ndarray, min_area: int) -> list[tuple[int, int, int, int, int]]:
    """Return (x0, y0, x1, y1, area) for 4-connected components of ``mask``.

    Small flood fill over a boolean mask; no scipy or OpenCV dependency.
    """
    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    components: list[tuple[int, int, int, int, int]] = []
    for start_y in range(height):
        row = mask[start_y]
        for start_x in range(width):
            if not row[start_x] or visited[start_y, start_x]:
                continue
            stack = [(start_y, start_x)]
            visited[start_y, start_x] = True
            x0 = x1 = start_x
            y0 = y1 = start_y
            area = 0
            while stack:
                y, x = stack.pop()
                area += 1
                if x < x0:
                    x0 = x
                if x > x1:
                    x1 = x
                if y < y0:
                    y0 = y
                if y > y1:
                    y1 = y
                for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                    if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((ny, nx))
            if area >= min_area:
                components.append((x0, y0, x1, y1, area))
    return components


class ColorBlobScorer:
    """Deterministic colour-blob proposals.

    Intended for the lab's coloured blocks and markers, and as a reproducible
    fixture generator for tests. It reports blobs, never identities.
    """

    name = "color_blob"

    def __init__(
        self,
        rgb: Sequence[float],
        *,
        tolerance: float = 0.15,
        min_area_px: int = 64,
        max_blobs: int = 8,
        label: str = "blob",
    ) -> None:
        if len(rgb) != 3 or not all(math.isfinite(v) and 0 <= v <= 255 for v in rgb):
            raise DetectionError("rgb must be three finite values in 0..255")
        if not math.isfinite(tolerance) or not 0 < tolerance <= 1:
            raise DetectionError("tolerance must be in (0, 1]")
        if not isinstance(min_area_px, int) or min_area_px <= 0:
            raise DetectionError("min_area_px must be a positive int")
        if not isinstance(max_blobs, int) or max_blobs <= 0:
            raise DetectionError("max_blobs must be a positive int")
        self.rgb = np.asarray(rgb, dtype=np.float64)
        self.tolerance = float(tolerance)
        self.min_area_px = min_area_px
        self.max_blobs = max_blobs
        self.label = label

    def score(self, image: np.ndarray, *, source_id: str, sequence: int, source_s: float) -> list[Detection]:
        array = _as_rgb_uint8(image)
        scaled = array.astype(np.float64) / 255.0
        target = self.rgb / 255.0
        distance = np.sqrt(np.sum((scaled - target) ** 2, axis=2))
        mask = distance <= self.tolerance
        components = _label_components(mask, self.min_area_px)
        components.sort(key=lambda item: item[4], reverse=True)
        capped = max(0, len(components) - self.max_blobs)
        out: list[Detection] = []
        for x0, y0, x1, y1, area in components[: self.max_blobs]:
            # Score is proximity to the target colour, not a probability.
            region = distance[y0:y1 + 1, x0:x1 + 1]
            score = float(max(0.0, 1.0 - (region.mean() / self.tolerance if self.tolerance else 1.0)))
            out.append(
                Detection(
                    source_id=source_id,
                    sequence=sequence,
                    source_s=source_s,
                    scorer=self.name,
                    label=self.label,
                    score=score,
                    bbox_xyxy=(float(x0), float(y0), float(x1), float(y1)),
                    centroid_xy=((x0 + x1) / 2.0, (y0 + y1) / 2.0),
                    area_px=float(area),
                    detail={"mean_color_distance": float(region.mean())},
                    truncated_by_scorer=capped,
                )
            )
        return out


class MotionEnergyScorer:
    """Frame-difference proposals against a previous frame.

    Detects "something changed here", which is the cheapest useful signal for
    deciding *when* to spend model attention. It carries no notion of what the
    object is and must not be treated as evidence about identity.
    """

    name = "motion_energy"

    def __init__(
        self,
        *,
        threshold: float = 0.08,
        min_area_px: int = 48,
        max_regions: int = 8,
        blur: bool = True,
    ) -> None:
        if not math.isfinite(threshold) or not 0 < threshold <= 1:
            raise DetectionError("threshold must be in (0, 1]")
        if not isinstance(min_area_px, int) or min_area_px <= 0:
            raise DetectionError("min_area_px must be a positive int")
        if not isinstance(max_regions, int) or max_regions <= 0:
            raise DetectionError("max_regions must be a positive int")
        self.threshold = float(threshold)
        self.min_area_px = min_area_px
        self.max_regions = max_regions
        self.blur = bool(blur)
        self._previous: np.ndarray | None = None
        self._previous_key: tuple[str, int] | None = None

    def reset(self) -> None:
        self._previous = None
        self._previous_key = None

    def score(self, image: np.ndarray, *, source_id: str, sequence: int, source_s: float) -> list[Detection]:
        array = _as_rgb_uint8(image)
        gray = (
            0.299 * array[:, :, 0] + 0.587 * array[:, :, 1] + 0.114 * array[:, :, 2]
        ) / 255.0
        if self.blur:
            gray = _box_blur3(gray)
        key = (source_id, sequence)
        previous, previous_key = self._previous, self._previous_key
        self._previous, self._previous_key = gray, key
        if previous is None or previous_key is None or previous_key[0] != source_id \
                or previous.shape != gray.shape:
            return []
        difference = np.abs(gray - previous)
        mask = difference >= self.threshold
        components = _label_components(mask, self.min_area_px)
        components.sort(key=lambda item: item[4], reverse=True)
        out: list[Detection] = []
        for x0, y0, x1, y1, area in components[: self.max_regions]:
            region = difference[y0:y1 + 1, x0:x1 + 1]
            out.append(
                Detection(
                    source_id=source_id,
                    sequence=sequence,
                    source_s=source_s,
                    scorer=self.name,
                    label="motion",
                    score=float(min(1.0, region.mean() / max(self.threshold, 1e-9))),
                    bbox_xyxy=(float(x0), float(y0), float(x1), float(y1)),
                    centroid_xy=((x0 + x1) / 2.0, (y0 + y1) / 2.0),
                    area_px=float(area),
                    detail={
                        "mean_abs_diff": float(region.mean()),
                        "previous_sequence": previous_key[1],
                    },
                )
            )
        return out


def _box_blur3(gray: np.ndarray) -> np.ndarray:
    """3x3 box blur via slicing; keeps the dependency surface at numpy only."""
    padded = np.pad(gray, 1, mode="edge")
    total = np.zeros_like(gray)
    for dy in (0, 1, 2):
        for dx in (0, 1, 2):
            total += padded[dy:dy + gray.shape[0], dx:dx + gray.shape[1]]
    return total / 9.0
