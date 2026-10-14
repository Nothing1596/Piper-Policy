"""Project pixel observations into the robot base frame.

A pixel is not a position. One RGB frame determines a ray, not a point, and
turning a ray into metres requires a calibration that nothing in this repository
currently produces (there is no ``base_link``-from-camera transform and no TCP
pose publisher; ``runtime.py`` only queries ``tcp_link`` for MoveIt validity
checks).

So this module's main job is to **refuse honestly**. When calibration or depth
is missing it returns ``available=False`` with a reason and, where possible, the
ray; it never guesses a coordinate and never substitutes a plausible default.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import yaml


class CalibrationError(ValueError):
    """A calibration artifact is malformed or unusable."""


def _matrix(value, shape: tuple[int, int], name: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise CalibrationError(f"{name} must be numeric") from exc
    if array.shape != shape:
        raise CalibrationError(f"{name} must have shape {shape}, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise CalibrationError(f"{name} must be finite")
    return array


@dataclass(frozen=True)
class Calibration:
    """Everything needed to relate pixels to the base frame, with provenance.

    ``calibration_id`` is mandatory: a projection without an identified
    calibration cannot be audited, and no downstream claim may depend on it.
    """

    calibration_id: str
    intrinsics: np.ndarray
    base_from_camera: np.ndarray
    distortion: np.ndarray = field(default_factory=lambda: np.zeros(5))
    depth_scale_m: float | None = None
    camera_frame_id: str = "camera_color_optical_frame"
    method: str = ""
    measured_at: str = ""
    notes: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.calibration_id, str) or not self.calibration_id.strip():
            raise CalibrationError("calibration_id is required")
        object.__setattr__(self, "intrinsics", _matrix(self.intrinsics, (3, 3), "intrinsics"))
        object.__setattr__(
            self, "base_from_camera", _matrix(self.base_from_camera, (4, 4), "base_from_camera")
        )
        object.__setattr__(self, "distortion", _matrix(self.distortion, (5,), "distortion"))
        k = self.intrinsics
        if k[0, 0] == 0.0 or k[1, 1] == 0.0:
            raise CalibrationError("intrinsics focal lengths must be non-zero")
        rotation = self.base_from_camera[:3, :3]
        if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-6):
            raise CalibrationError("base_from_camera rotation must be orthonormal")
        if abs(np.linalg.det(rotation) - 1.0) > 1e-6:
            raise CalibrationError("base_from_camera rotation must have determinant +1")
        if self.depth_scale_m is not None:
            if not math.isfinite(self.depth_scale_m) or self.depth_scale_m <= 0:
                raise CalibrationError("depth_scale_m must be positive and finite")

    @property
    def supports_metric(self) -> bool:
        """True only when a metric unprojection is defined at all."""
        return self.depth_scale_m is not None

    def describe(self) -> dict:
        return {
            "calibration_id": self.calibration_id,
            "camera_frame_id": self.camera_frame_id,
            "method": self.method,
            "measured_at": self.measured_at,
            "depth_scale_m": self.depth_scale_m,
            "supports_metric": self.supports_metric,
            "intrinsics": self.intrinsics.tolist(),
            "distortion": self.distortion.tolist(),
            "base_from_camera": self.base_from_camera.tolist(),
            "notes": self.notes,
        }

    @classmethod
    def from_mapping(cls, payload: dict) -> "Calibration":
        if not isinstance(payload, dict):
            raise CalibrationError("calibration must be a mapping")
        try:
            calibration_id = str(payload["calibration_id"])
            intrinsics = payload["intrinsics"]
            base_from_camera = payload["base_from_camera"]
        except KeyError as exc:
            raise CalibrationError(f"calibration is missing required key {exc.args[0]!r}") from exc
        return cls(
            calibration_id=calibration_id,
            intrinsics=intrinsics,
            base_from_camera=base_from_camera,
            distortion=payload.get("distortion", [0.0] * 5),
            depth_scale_m=payload.get("depth_scale_m"),
            camera_frame_id=str(payload.get("camera_frame_id", "camera_color_optical_frame")),
            method=str(payload.get("method", "")),
            measured_at=str(payload.get("measured_at", "")),
            notes=str(payload.get("notes", "")),
        )

    @classmethod
    def load(cls, path: str | Path) -> "Calibration":
        source = Path(path)
        with source.open(encoding="utf-8") as handle:
            payload = yaml.safe_load(handle)
        calibration = cls.from_mapping(payload)
        if not calibration.calibration_id:
            raise CalibrationError(f"{source} does not declare a calibration_id")
        return calibration


@dataclass(frozen=True)
class Projection:
    """Result of asking where a pixel is in space."""

    available: bool
    reason: str
    calibration_id: str | None = None
    base_xyz_m: tuple[float, float, float] | None = None
    ray_origin_base_xyz: tuple[float, float, float] | None = None
    ray_direction_base_xyz: tuple[float, float, float] | None = None
    depth_m: float | None = None
    uncertainty_m: float | None = None
    detail: dict = field(default_factory=dict)

    def describe(self) -> dict:
        return {
            "available": self.available,
            "reason": self.reason,
            "calibration_id": self.calibration_id,
            "base_xyz_m": list(self.base_xyz_m) if self.base_xyz_m else None,
            "ray_origin_base_xyz": list(self.ray_origin_base_xyz) if self.ray_origin_base_xyz else None,
            "ray_direction_base_xyz": (
                list(self.ray_direction_base_xyz) if self.ray_direction_base_xyz else None
            ),
            "depth_m": self.depth_m,
            "uncertainty_m": self.uncertainty_m,
            "detail": dict(self.detail),
        }


#: Stable reason vocabulary so callers can branch without string matching on prose.
NO_CALIBRATION = "no_calibration"
NO_DEPTH = "no_depth"
DEGENERATE_PIXEL = "degenerate_pixel"
DEPTH_OUT_OF_RANGE = "depth_out_of_range"
BEHIND_CAMERA = "behind_camera"


def project(
    pixel_xy,
    *,
    calibration: Calibration | None,
    depth_m: float | None = None,
    image_size: tuple[int, int] | None = None,
) -> Projection:
    """Relate one pixel to the base frame, or explain why that is impossible.

    With a calibration and a metric depth, returns ``base_xyz_m``. With a
    calibration but no depth, returns the ray in base coordinates and
    ``available=False``: a single RGB frame genuinely has no depth, and saying
    so is more useful than a fabricated point.
    """
    if calibration is None:
        return Projection(
            available=False,
            reason=NO_CALIBRATION,
            detail={
                "hint": (
                    "no base_link-from-camera transform is published by this repository; "
                    "provide a calibration file with intrinsics and base_from_camera"
                )
            },
        )

    x, y = (float(pixel_xy[0]), float(pixel_xy[1]))
    if not math.isfinite(x) or not math.isfinite(y):
        raise CalibrationError("pixel_xy must be finite")
    if image_size is not None:
        width, height = int(image_size[0]), int(image_size[1])
        if not (0 <= x < width and 0 <= y < height):
            return Projection(
                available=False,
                reason=DEGENERATE_PIXEL,
                calibration_id=calibration.calibration_id,
                detail={"pixel_xy": [x, y], "image_size": [width, height]},
            )

    k = calibration.intrinsics
    normalized = np.array(
        [(x - k[0, 2]) / k[0, 0], (y - k[1, 2]) / k[1, 1]], dtype=np.float64
    )
    undistorted = _undistort(normalized, calibration.distortion)
    ray_camera = np.array([undistorted[0], undistorted[1], 1.0], dtype=np.float64)
    ray_camera /= np.linalg.norm(ray_camera)
    rotation = calibration.base_from_camera[:3, :3]
    origin = calibration.base_from_camera[:3, 3]
    ray_base = rotation @ ray_camera

    if depth_m is None:
        supports = "metric unprojection requires depth" if calibration.supports_metric else \
            "no depth channel was supplied"
        return Projection(
            available=False,
            reason=NO_DEPTH,
            calibration_id=calibration.calibration_id,
            ray_origin_base_xyz=(float(origin[0]), float(origin[1]), float(origin[2])),
            ray_direction_base_xyz=(float(ray_base[0]), float(ray_base[1]), float(ray_base[2])),
            detail={
                "hint": supports + "; a single RGB frame determines a ray, not a point",
                "undistorted_normalized_xy": undistorted.tolist(),
            },
        )

    depth = float(depth_m)
    if not math.isfinite(depth) or depth <= 0:
        return Projection(
            available=False,
            reason=DEPTH_OUT_OF_RANGE,
            calibration_id=calibration.calibration_id,
            ray_origin_base_xyz=(float(origin[0]), float(origin[1]), float(origin[2])),
            ray_direction_base_xyz=(float(ray_base[0]), float(ray_base[1]), float(ray_base[2])),
            detail={"depth_m": depth},
        )

    point_camera = ray_camera * depth
    if point_camera[2] <= 0:
        return Projection(
            available=False,
            reason=BEHIND_CAMERA,
            calibration_id=calibration.calibration_id,
            detail={"point_camera": point_camera.tolist()},
        )
    homogeneous = np.r_[point_camera, 1.0]
    point_base = (calibration.base_from_camera @ homogeneous)[:3]
    return Projection(
        available=True,
        reason="projected_with_calibration_and_depth",
        calibration_id=calibration.calibration_id,
        base_xyz_m=(float(point_base[0]), float(point_base[1]), float(point_base[2])),
        ray_origin_base_xyz=(float(origin[0]), float(origin[1]), float(origin[2])),
        ray_direction_base_xyz=(float(ray_base[0]), float(ray_base[1]), float(ray_base[2])),
        depth_m=depth,
        uncertainty_m=None,
        detail={
            "depth_source": "depth_frame",
            "note": (
                "uncertainty_m is None because no per-pixel depth noise model has been "
                "established for this rig; treat the point as unqualified"
            ),
        },
    )


def _undistort(point: np.ndarray, coefficients: np.ndarray, iterations: int = 8) -> np.ndarray:
    """Iterative inverse of the plumb-bob distortion model (mirrors GPT-Policy)."""
    if coefficients is None or np.allclose(coefficients, 0.0):
        return point
    k1, k2, p1, p2, k3 = coefficients
    x, y = float(point[0]), float(point[1])
    for _ in range(iterations):
        radius = x * x + y * y
        radial = 1 + k1 * radius + k2 * radius ** 2 + k3 * radius ** 3
        if radial == 0:
            break
        dx = 2 * p1 * x * y + p2 * (radius + 2 * x * x)
        dy = p1 * (radius + 2 * y * y) + 2 * p2 * x * y
        x, y = (float(point[0]) - dx) / radial, (float(point[1]) - dy) / radial
    return np.array([x, y])


@dataclass(frozen=True)
class FusedEntity:
    """One entity after merging per-source views, with its uncertainties kept."""

    entity_id: str
    label: str
    state: str
    sources: tuple[str, ...]
    track_keys: tuple[str, ...]
    base_xyz_m: tuple[float, float, float] | None
    projection_available: bool
    calibration_id: str | None
    evidence_sequences: tuple[int, ...]
    confidence: float
    conflicts: tuple[str, ...] = ()
    detail: dict = field(default_factory=dict)

    def describe(self) -> dict:
        return {
            "entity_id": self.entity_id,
            "label": self.label,
            "state": self.state,
            "sources": list(self.sources),
            "track_keys": list(self.track_keys),
            "base_xyz_m": list(self.base_xyz_m) if self.base_xyz_m else None,
            "projection_available": self.projection_available,
            "calibration_id": self.calibration_id,
            "evidence_sequences": list(self.evidence_sequences),
            "confidence": self.confidence,
            "conflicts": list(self.conflicts),
            "detail": dict(self.detail),
        }


def fuse_entities(entities: Iterable, *, calibration: Calibration | None = None) -> list[FusedEntity]:
    """Merge per-source entity views into fused records.

    Two rules the earlier design got wrong and this function enforces:

    1. Coordinates from different frames are never averaged. A merged position
       is only reported when a single calibration produced it.
    2. Multi-source agreement is not multiple independent evidence. Repeated
       processing of one image is counted once, via ``evidence_sequences``.
    """
    grouped: dict[str, list] = {}
    for entity in entities:
        grouped.setdefault(entity.entity_id, []).append(entity)

    fused: list[FusedEntity] = []
    for entity_id, members in grouped.items():
        primary = members[0]
        sources = tuple(sorted({source for member in members for source in member.sources}))
        track_keys = tuple(member.track.key for member in members)
        sequences: list[int] = []
        for member in members:
            for sequence in member.evidence_sequences:
                if sequence not in sequences:
                    sequences.append(sequence)

        projected = [member for member in members if member.projection_available and member.base_xyz_m]
        calibration_ids = {member.calibration_id for member in projected}
        conflicts: list[str] = []
        point: tuple[float, float, float] | None = None
        if len(projected) > 1:
            spread = max(
                math.dist(projected[0].base_xyz_m, other.base_xyz_m) for other in projected[1:]
            )
            if spread > 0.01:
                conflicts.append(f"projection_disagreement_{spread:.3f}m")
        if len(projected) == 1:
            point = projected[0].base_xyz_m
        elif len(projected) > 1 and len(calibration_ids) == 1:
            # Same calibration, so the frames are commensurate; still report the
            # worst-case spread rather than implying a precise consensus.
            stacked = np.asarray([member.base_xyz_m for member in projected], dtype=np.float64)
            point = tuple(float(v) for v in stacked.mean(axis=0))
            conflicts.append("multi_view_average_reported_with_spread")
        elif len(projected) > 1:
            conflicts.append("projections_use_different_calibrations_not_averaged")

        labels = {member.label for member in members}
        if len(labels) > 1:
            conflicts.append("label_disagreement")

        fused.append(
            FusedEntity(
                entity_id=entity_id,
                label=primary.label,
                state=primary.state,
                sources=sources,
                track_keys=track_keys,
                base_xyz_m=point,
                projection_available=point is not None,
                calibration_id=(projected[0].calibration_id if projected else None),
                evidence_sequences=tuple(sorted(sequences)),
                confidence=min(member.score for member in members) if members else 0.0,
                conflicts=tuple(conflicts),
                detail={
                    "member_count": len(members),
                    "labels": sorted(labels),
                    "independent_evidence_count": len({member.track.source_id for member in members}),
                },
            )
        )
    return fused
