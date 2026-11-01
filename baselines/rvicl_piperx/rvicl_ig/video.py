"""Read actual presentation frames inside one LeRobot MP4 episode window."""

import bisect
import hashlib
import json
import math
import shutil
import subprocess
from pathlib import Path

import numpy as np

from .io import save


def binary(name):
    found = shutil.which(name)
    if found:
        return found
    import sys
    adjacent = Path(sys.executable).parent / name
    if adjacent.is_file():
        return str(adjacent)
    raise RuntimeError(f"Missing {name}; install ffmpeg in the evaluation environment")


class EpisodeVideo:
    def __init__(self, source, directory, *, timeline=None):
        self.source = source
        self.path = Path(source["path"])
        self.directory = Path(directory)
        if timeline is None:
            stat = self.path.stat()
            identity = f"{self.path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}"
            cache = self.directory.parent / "timelines" / (hashlib.sha256(identity.encode()).hexdigest() + ".json")
            if cache.is_file():
                timeline = json.loads(cache.read_text())
            else:
                result = subprocess.run([binary("ffprobe"), "-v", "error", "-select_streams", "v:0",
                    "-show_entries", "frame=best_effort_timestamp_time", "-of", "json", str(self.path)],
                    capture_output=True, text=True, check=True, timeout=300)
                timeline = [float(f["best_effort_timestamp_time"]) for f in json.loads(result.stdout)["frames"]]
                save(cache, timeline)
        if not timeline or any(not math.isfinite(t) for t in timeline) or any(
                a >= b for a, b in zip(timeline, timeline[1:])):
            raise ValueError("Source PTS must be finite and strictly increasing")
        self.timeline = timeline
        start, end = source["start_s"], source["end_s"]
        if not 0 <= start < end or end > timeline[-1] + max(.1, timeline[-1] - timeline[-2] if len(timeline) > 1 else .1):
            raise ValueError("Invalid episode window")
        # PTS are microsecond-rounded; the shared shard's next episode is excluded.
        self.indices = [i for i, t in enumerate(timeline) if start - 1e-6 <= t < end - 1e-6]
        if not self.indices:
            raise ValueError("Episode window contains no frames")
        self.duration = end - start

    def decode(self, indices, directory, width=512):
        indices = sorted(set(indices))
        if not indices or not set(indices) <= set(self.indices):
            raise ValueError("Requested frames fall outside this human episode")
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        expression = "+".join(f"eq(n\\,{i})" for i in indices)
        subprocess.run([binary("ffmpeg"), "-nostdin", "-v", "error", "-y", "-threads", "1",
            "-i", str(self.path), "-vf", f"select={expression},scale={width}:-2", "-vsync", "0",
            "-q:v", "3", "-threads", "1", "-frames:v", str(len(indices)), str(directory / "%06d.jpg")],
            capture_output=True, check=True, timeout=300)
        frames = [{"index": j, "source_frame": i,
                   "timestamp_s": max(0., self.timeline[i] - self.source["start_s"]),
                   "path": str((directory / f"{j + 1:06d}.jpg").resolve())}
                  for j, i in enumerate(indices)]
        if any(not Path(f["path"]).is_file() for f in frames):
            raise RuntimeError("FFmpeg did not decode every requested frame")
        return frames

    def candidates(self, fps=2., limit=128):
        if fps <= 0 or limit < 2:
            raise ValueError("candidate_fps must be positive and max_candidates >= 2")
        count = min(len(self.indices), limit, max(2, math.ceil(self.duration * fps) + 1))
        picks = [self.indices[i] for i in sorted(set(np.linspace(0, len(self.indices)-1, count).astype(int)))]
        return self.decode(picks, self.directory / "candidates")

    def clip(self, start_s, end_s, name):
        if not 0 <= start_s < end_s <= self.duration + 1e-6:
            raise ValueError("Clip outside episode")
        times = [self.timeline[i] - self.source["start_s"] for i in self.indices]
        lo, hi = bisect.bisect_left(times, start_s - 1e-6), bisect.bisect_left(times, end_s - 1e-6)
        available = self.indices[lo:hi]
        if not available:
            raise ValueError("Clip contains no source frames")
        picks = [available[i] for i in sorted(set(np.linspace(0, len(available)-1, min(4, len(available))).astype(int)))]
        return self.decode(picks, self.directory / "clips" / name, width=256)


def contact_sheet(frames, path):
    """A bounded grid, matching the original RV-ICL clip presentation."""
    from PIL import Image, ImageDraw
    images = [Image.open(frame["path"]).convert("RGB") for frame in frames]
    width, height = images[0].size
    columns, caption = min(5, len(images)), 20
    sheet = Image.new("RGB", (columns*width, math.ceil(len(images)/columns)*(height+caption)), "white")
    draw = ImageDraw.Draw(sheet)
    for i, (frame, image) in enumerate(zip(frames, images)):
        x, y = (i % columns)*width, (i // columns)*(height+caption)
        draw.text((x+3, y+3), f"t={frame['timestamp_s']:.3f}s", fill="black")
        sheet.paste(image, (x, y+caption))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)
    return str(path.resolve())
