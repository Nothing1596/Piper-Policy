"""Human-video ingestion (human-video v1).

Host-side, offline, simulation-only: this package never touches hardware,
never publishes commands and never substitutes clocks for missing timestamps.
PyAV and OpenCV are imported lazily inside the functions that need them, so
the package imports in the core environment without the video extras.
"""
from __future__ import annotations

from .candidates import build_candidates
from .source import FrameRecord, VideoError, VideoSource

__all__ = ["FrameRecord", "VideoError", "VideoSource", "build_candidates"]
