"""Stream the complete rollout and one aligned video per tool invocation."""

import contextlib
import re
import subprocess
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from .io import save
from .video import binary


class ExecutionFrames:
    """Bounded samples of native camera RGB during one physical primitive."""

    def __init__(self, directory, tool, cameras, *, turn=None, call_id=None, sample_count=4):
        self.directory, self.tool, self.cameras = Path(directory), tool, tuple(cameras)
        self.turn, self.call_id, self.sample_count = turn, call_id, sample_count
        self.samples, self.last = [], None
        self.start_step, self.stride = None, 1

    def capture(self, views, step):
        if self.last is not None and self.last[0] == step:
            return
        if self.start_step is None:
            self.start_step = step
        self.last = (step, {camera: np.asarray(views[camera], dtype=np.uint8).copy()
                            for camera in self.cameras})
        if (step - self.start_step) % self.stride == 0:
            self.samples.append(self.last)
            if len(self.samples) > 2 * self.sample_count:
                self.samples = self.samples[::2]
                self.stride *= 2

    def finish(self):
        # A planning rejection or read-only call has no execution process.
        if self.last is None or self.last[0] == self.start_step:
            return []
        if self.samples[-1][0] != self.last[0]:
            self.samples.append(self.last)
        count = min(self.sample_count, len(self.samples))
        indices = np.linspace(0, len(self.samples)-1, count).astype(int)
        images = []
        self.directory.mkdir(parents=True, exist_ok=True)
        for order, index in enumerate(indices, 1):
            step, views = self.samples[index]
            phase = "start" if step == self.start_step else "end" if step == self.last[0] else "during"
            for camera, rgb in views.items():
                path = self.directory / f"{order:02d}-step-{step:05d}-{camera}.png"
                Image.fromarray(rgb).save(path)
                images.append({"path": str(path.resolve()), "image_kind": "execution_process",
                    "camera": camera, "simulation_step": step, "sequence_order": order,
                    "sample_count": count, "phase": phase, "source_tool": self.tool,
                    "source_turn": self.turn, "source_call_id": self.call_id})
        save(self.directory / "manifest.json", {"start_step": self.start_step, "end_step": self.last[0],
            "source_tool": self.tool, "source_turn": self.turn, "source_call_id": self.call_id,
            "sample_cameras": list(self.cameras), "sample_count": count, "images": images})
        return images


class VideoWriter:
    def __init__(self, path, frame, fps):
        self.path, self.frames = Path(path), 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.shape = frame.shape
        self.log = self.path.with_suffix(".ffmpeg.log").open("wb")
        height, width = frame.shape[:2]
        self.process = subprocess.Popen([binary("ffmpeg"), "-nostdin", "-v", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(fps),
            "-i", "pipe:0", "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
            "-threads", "1", "-pix_fmt", "yuv420p", "-movflags", "+frag_keyframe+empty_moov+default_base_moof",
            str(self.path)], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=self.log)
        self.closed = False

    def append(self, frame):
        if self.closed or frame.shape != self.shape:
            raise ValueError("Video frame shape changed or writer is closed")
        try:
            self.process.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())
        except (BrokenPipeError, OSError) as error:
            raise RuntimeError(f"Video encoder failed; inspect {self.path.with_suffix('.ffmpeg.log')}") from error
        self.frames += 1

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            self.process.stdin.close()
            status = self.process.wait(timeout=60)
            if status:
                raise RuntimeError(f"Video encoder exited {status}; inspect {self.path.with_suffix('.ffmpeg.log')}")
        finally:
            if self.process.poll() is None:
                self.process.kill()
                self.process.wait()
            self.log.close()


def compose_frame(overview, views, step, fps):
    """Overview and all three sensor views, letterboxed without distortion."""
    canvas = Image.new("RGB", (1280,720), (24,24,24))
    draw = ImageDraw.Draw(canvas)
    draw.text((12,8), f"Panda execution | control step {step} | simulated time {step/fps:.3f}s", fill="white")
    overview = Image.fromarray(np.asarray(overview, dtype=np.uint8)).convert("RGB")
    overview = ImageOps.contain(overview, (960,680))
    canvas.paste(overview, ((960-overview.width)//2, 36+(680-overview.height)//2))
    for i, camera in enumerate(("head", "left_wrist", "right_wrist")):
        image = Image.fromarray(np.asarray(views[camera], dtype=np.uint8)).convert("RGB")
        image = ImageOps.contain(image, (320,210))
        y = 30+i*230
        draw.text((968,y), camera, fill="white")
        canvas.paste(image, (960+(320-image.width)//2,y+18+(210-image.height)//2))
    return np.asarray(canvas)


class ExecutionRecorder:
    def __init__(self, directory, fps, enabled=True):
        self.directory, self.fps, self.enabled = Path(directory), float(fps), enabled
        initial, attempt = self.directory, 2
        while (self.directory / "videos/execution.mp4").exists() or (self.directory / "tool_calls/index.json").exists():
            self.directory = initial / "recordings" / f"attempt-{attempt:03d}"
            attempt += 1
        self.writer = self.tool_writer = None
        self.active = None
        self.records = []
        self.last_frame, self.step = None, 0
        self.closed = False

    def capture(self, frame, step):
        self.step = int(step)
        if not self.enabled:
            return
        self.last_frame = np.asarray(frame, dtype=np.uint8)
        if self.writer is None:
            self.writer = VideoWriter(self.directory / "videos/execution.mp4", self.last_frame, self.fps)
        self.writer.append(self.last_frame)
        if self.tool_writer is not None:
            self.tool_writer.append(self.last_frame)

    @contextlib.contextmanager
    def tool(self, name, arguments, *, turn=None, call_id=None, source="agent"):
        if self.active is not None:
            raise RuntimeError("Tool video scopes cannot overlap")
        index = len(self.records)+1
        safe = re.sub(r"[^A-Za-z0-9_-]", "_", name)[:80]
        directory = self.directory / "tool_calls" / f"{index:04d}_{safe}"
        record = {"index": index, "turn": turn, "call_id": call_id, "name": name, "arguments": arguments,
            "source": source, "start_step": self.step, "status": "running", "video": None}
        self.active = record
        self.records.append(record)
        started = time.monotonic()
        try:
            save(directory / "request.json", {key: record[key] for key in ("index", "turn", "call_id", "name", "arguments", "source", "start_step")})
            if self.enabled:
                if self.last_frame is None:
                    raise RuntimeError("Record the initial scene before a tool call")
                self.tool_writer = VideoWriter(directory / "video.mp4", self.last_frame, self.fps)
                self.tool_writer.append(self.last_frame)
                record["video"] = str((directory / "video.mp4").resolve())
            yield record
            record["status"] = "complete"
        except BaseException as error:
            record.update(status="error", error={"type": type(error).__name__, "message": str(error)})
            raise
        finally:
            record.update(end_step=self.step, physics_steps=self.step-record["start_step"],
                          read_only=self.step == record["start_step"], elapsed_wall_s=time.monotonic()-started)
            writer = self.tool_writer
            self.tool_writer = None
            try:
                if writer is not None:
                    # Read-only tools get a half-second still clip, explicitly marked in metadata.
                    if record["read_only"]:
                        for _ in range(max(2, round(self.fps*.5))-writer.frames):
                            writer.append(self.last_frame)
                    writer.close()
                    record.update(video_frames=writer.frames, video_fps=self.fps,
                                  static_video=record["read_only"], video_duration_s=writer.frames/self.fps)
            except Exception as error:
                record.update(status="error", video_error=str(error))
                raise
            finally:
                self.active = None
                result = record.pop("result", None)
                if result is None and record.get("error"):
                    result = {"tool_rejected": record["error"]["message"]}
                if result is not None:
                    save(directory / "result.json", result)
                save(directory / "recording.json", record)
                save(self.directory / "tool_calls/index.json", self.records)
                self._manifest()

    def _manifest(self):
        save(self.directory / "videos/manifest.json", {"enabled": self.enabled,
            "execution_video": str((self.directory / "videos/execution.mp4").resolve()) if self.writer else None,
            "frames": self.writer.frames if self.writer else 0, "fps": self.fps, "last_step": self.step,
            "closed": self.closed, "tool_index": str((self.directory / "tool_calls/index.json").resolve()),
            "timeline": "one full-video frame per control step plus initial frame; VLM waiting time excluded"})

    def close(self):
        if self.closed:
            return
        try:
            if self.writer:
                self.writer.close()
        finally:
            self.closed = True
            self._manifest()
