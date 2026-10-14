"""Data pipeline for Piper Lab: mock data, MCAP conversion, LeRobot export.

Data contract (see AGENTS.md):
- ordered state and target action are 7 float32 values: joint1..joint6 in
  radians, gripper total opening in metres.
- images are RGB uint8 HWC; raw depth is uint16 with ``depth_scale_m`` retained.
- training samples are aligned at fps=20 with a max source-message age of
  200 ms; messages from the future are never used.
- targets come only from ``/lab/applied_action``; a target is never invented
  from the observed state.
- datasets are split by whole episode, never frame-random.

All writers stage into a temporary sibling directory and rename into place;
existing outputs are never overwritten.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import warnings
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.lib import format as npy_format

JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "gripper"]
STATE_DIM = 7
DEFAULT_FPS = 20
MAX_SAMPLE_AGE_NS = 200_000_000  # 200 ms observation/command freshness window
DEFAULT_DEPTH_SCALE_M = 0.001  # RealSense default: 1 depth LSB = 1 mm
MOCK_SEED = 20260916
MOCK_IMAGE_SIZE = 224  # matches the ACT training image size
MOCK_BASE_STAMP_NS = 1_700_000_000_000_000_000
SYNTHETIC_MAGIC = 0xC0DE  # bit pattern rendered into every mock RGB frame

TOPIC_OBSERVATION = "/lab/observation"
TOPIC_RGB = "/lab/rgb"
TOPIC_DEPTH = "/lab/depth"
TOPIC_CAMERA_INFO = "/lab/camera_info"
TOPIC_APPLIED_ACTION = "/lab/applied_action"
TOPIC_EVENTS = "/lab/events"

STREAM_TOPICS = (TOPIC_OBSERVATION, TOPIC_APPLIED_ACTION, TOPIC_RGB, TOPIC_DEPTH)

MANIFEST_NAME = "manifest.json"
EPISODE_GLOB = "episode_*.npz"


# ---------------------------------------------------------------------------
# shared helpers


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fail_if_exists(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"output already exists, refusing to overwrite: {path}")


def _stage_dir(output: Path) -> Path:
    """Return a fresh staging directory next to ``output``."""
    _fail_if_exists(output)
    staging = output.parent / (output.name + f".tmp-{os.getpid()}")
    if staging.exists():
        raise FileExistsError(f"stale staging directory in the way: {staging}")
    staging.mkdir(parents=True)
    return staging


def _commit(staging: Path, output: Path) -> None:
    os.rename(staging, output)


def _abort(staging: Path) -> None:
    # Keep failed staging directories for inspection. Never recursively remove data.
    pass


def _write_json_atomic(path: Path, payload: dict) -> None:
    tmp = path.with_name(path.name + f".tmp-{os.getpid()}")
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp, path)


_NPZ_EPOCH = (2026, 1, 1, 0, 0, 0)  # fixed zip timestamps: byte-deterministic outputs


def _write_npz(path: Path, arrays: dict) -> None:
    """Write an .npz with a fixed member timestamp so identical arrays hash
    identically across runs (np.savez embeds the wall clock)."""
    with zipfile.ZipFile(path, "w") as zipf:
        for key in sorted(arrays):
            value = np.asanyarray(arrays[key])
            zinfo = zipfile.ZipInfo(f"{key}.npy", date_time=_NPZ_EPOCH)
            zinfo.compress_type = zipfile.ZIP_DEFLATED
            zinfo.external_attr = 0o600 << 16
            with zipf.open(zinfo, "w", force_zip64=True) as member:
                npy_format.write_array(member, value, allow_pickle=False)


def _joint_vector(names: list, positions: list, topic: str) -> np.ndarray:
    """Map a JointState name/position pair onto the contract 7-vector."""
    if len(names) != len(positions):
        raise ValueError(f"{topic}: name/position length mismatch ({len(names)} != {len(positions)})")
    mapping = {str(name): float(pos) for name, pos in zip(names, positions)}
    missing = [name for name in JOINT_NAMES if name not in mapping]
    if missing:
        raise ValueError(f"{topic}: message is missing joints {missing}")
    return np.array([mapping[name] for name in JOINT_NAMES], dtype=np.float32)


def _assign_splits(n_episodes: int, val_fraction: float = 0.2) -> dict:
    """Whole-episode deterministic split: the last episodes go to validation.

    Never frame-random: an episode is entirely in train or entirely in val.
    """
    indices = list(range(n_episodes))
    if n_episodes < 2 or val_fraction <= 0.0:
        return {"train": indices, "val": []}
    n_val = min(n_episodes - 1, max(1, int(round(n_episodes * val_fraction))))
    return {"train": indices[:-n_val], "val": indices[-n_val:]}


# ---------------------------------------------------------------------------
# normalized episode container


@dataclass
class RawEpisode:
    """Messages of one recording episode, grouped per topic, stamp-sorted."""

    index: int
    task: str
    start_ns: int
    end_ns: int
    streams: dict = field(default_factory=dict)  # topic -> [(stamp_ns, payload)]
    camera_infos: list = field(default_factory=list)  # [(stamp_ns, payload)]
    depth_scale_m: float | None = None


@dataclass
class AlignedEpisode:
    index: int
    task: str
    rgb: np.ndarray  # (T, H, W, 3) uint8
    states: np.ndarray  # (T, 7) float32
    actions: np.ndarray  # (T, 7) float32
    stamps_ns: np.ndarray  # (T,) int64
    depth: np.ndarray | None  # (T, H, W) uint16 or None
    depth_scale_m: float | None
    intrinsics: np.ndarray | None  # (3, 3) float64 or None
    dropped_frames: int
    drop_reasons: dict
    source_stamps_ns: np.ndarray | None = None
    sample_grid_ns: np.ndarray | None = None
    sample_valid: np.ndarray | None = None


def _latest_not_after(messages: list, stamp_ns: int, pointer: int) -> tuple:
    """Return (message, pointer) for the newest message with stamp <= stamp_ns.

    ``messages`` must be stamp-sorted. Messages stamped after ``stamp_ns`` are
    never returned: future data is not allowed to fill a sample.
    """
    while pointer < len(messages) and messages[pointer][0] <= stamp_ns:
        pointer += 1
    if pointer == 0:
        return None, pointer
    return messages[pointer - 1], pointer


def _align_episode(raw: RawEpisode, fps: int, max_age_ns: int = MAX_SAMPLE_AGE_NS) -> AlignedEpisode:
    """Resample one raw episode onto the fps grid using source timestamps."""
    period_ns = int(round(1e9 / fps))
    has_depth = bool(raw.streams.get(TOPIC_DEPTH))
    required = [TOPIC_OBSERVATION, TOPIC_APPLIED_ACTION, TOPIC_RGB] + ([TOPIC_DEPTH] if has_depth else [])

    pointers = {topic: 0 for topic in STREAM_TOPICS}
    info_pointer = 0

    frames_rgb, frames_state, frames_action, frames_depth, frames_stamp = [], [], [], [], []
    intrinsics = None
    dropped = 0
    drop_reasons: dict = {}
    sources, grid, validity = [], [], []

    t = raw.start_ns
    # half-period tolerance lets the last sample land on the end stamp
    while t <= raw.end_ns + period_ns // 2:
        sample = {}
        source_times = {topic: -1 for topic in STREAM_TOPICS}
        missing_reason = None
        for topic in required:
            message, pointers[topic] = _latest_not_after(raw.streams.get(topic, []), t, pointers[topic])
            if message is None:
                missing_reason = f"no_message:{topic}"
                break
            stamp, payload = message
            age = t - stamp
            if age < 0:  # defensive: _latest_not_after already forbids this
                missing_reason = f"future_message:{topic}"
                break
            if age > max_age_ns:
                missing_reason = f"stale_message:{topic}"
                break
            sample[topic] = payload
            source_times[topic] = stamp
        grid.append(t); validity.append(missing_reason is None)
        if missing_reason is not None:
            dropped += 1
            drop_reasons[missing_reason] = drop_reasons.get(missing_reason, 0) + 1
        else:
            info, info_pointer = _latest_not_after(raw.camera_infos, t, info_pointer)
            if info is not None:
                intrinsics = np.asarray(info[1]["k"], dtype=np.float64).reshape(3, 3)
            frames_rgb.append(sample[TOPIC_RGB])
            frames_state.append(sample[TOPIC_OBSERVATION])
            frames_action.append(sample[TOPIC_APPLIED_ACTION])
            if has_depth:
                frames_depth.append(sample[TOPIC_DEPTH])
            frames_stamp.append(t)
            sources.append([source_times[topic] for topic in STREAM_TOPICS])
        t += period_ns

    if not frames_state:
        raise ValueError(
            f"episode {raw.index}: no frame satisfied the freshness window "
            f"({dropped} dropped; reasons={drop_reasons})"
        )

    return AlignedEpisode(
        index=raw.index,
        task=raw.task,
        rgb=np.stack(frames_rgb).astype(np.uint8),
        states=np.stack(frames_state).astype(np.float32),
        actions=np.stack(frames_action).astype(np.float32),
        stamps_ns=np.asarray(frames_stamp, dtype=np.int64),
        depth=np.stack(frames_depth).astype(np.uint16) if has_depth else None,
        depth_scale_m=raw.depth_scale_m if has_depth else None,
        intrinsics=intrinsics,
        dropped_frames=dropped,
        drop_reasons=drop_reasons,
        source_stamps_ns=np.asarray(sources, dtype=np.int64),
        sample_grid_ns=np.asarray(grid, dtype=np.int64),
        sample_valid=np.asarray(validity, dtype=bool),
    )


def _write_dataset(
    staging: Path,
    episodes: list,
    *,
    provenance: str,
    synthetic: bool,
    fps: int,
    source: str | None,
    extra_manifest: dict | None = None,
) -> dict:
    """Write normalized episode NPZ files plus manifest into ``staging``."""
    ep_dir = staging / "episodes"
    ep_dir.mkdir()
    splits = _assign_splits(len(episodes))
    split_of = {i: ("val" if i in splits["val"] else "train") for i in splits["train"] + splits["val"]}

    manifest_episodes = []
    total_frames = 0
    total_dropped = 0
    for ep in episodes:
        rel = f"episodes/episode_{ep.index:04d}.npz"
        path = staging / rel
        arrays = {
            "rgb": ep.rgb,
            "states": ep.states,
            "actions": ep.actions,
            "stamps_ns": ep.stamps_ns,
            "task": np.asarray(ep.task),
        }
        if ep.depth is not None:
            arrays["depth"] = ep.depth
            arrays["depth_scale_m"] = np.asarray(ep.depth_scale_m, dtype=np.float64)
        arrays['source_stamps_ns'] = ep.source_stamps_ns if ep.source_stamps_ns is not None else np.repeat(ep.stamps_ns[:,None],len(STREAM_TOPICS),axis=1)
        arrays['source_topics'] = np.asarray(STREAM_TOPICS)
        arrays['sample_grid_ns'] = ep.sample_grid_ns if ep.sample_grid_ns is not None else ep.stamps_ns
        arrays['sample_valid'] = ep.sample_valid if ep.sample_valid is not None else np.ones(len(ep.stamps_ns),dtype=bool)
        if ep.intrinsics is not None:
            arrays["camera_intrinsics"] = ep.intrinsics
        _write_npz(path, arrays)
        frames = int(ep.states.shape[0])
        total_frames += frames
        total_dropped += ep.dropped_frames
        manifest_episodes.append(
            {
                "index": ep.index,
                "file": rel,
                "frames": frames,
                "dropped_frames": ep.dropped_frames,
                "drop_reasons": ep.drop_reasons,
                "split": split_of[ep.index],
                "task": ep.task,
                "sha256": _sha256_file(path),
                "has_depth": ep.depth is not None,
                "duration_s": round(frames / fps, 6),
            }
        )

    manifest = {
        "format_version": 1,
        "generator": "piperlab.data",
        "provenance": provenance,
        "synthetic": bool(synthetic),
        "source": source,
        "fps": fps,
        "max_sample_age_ms": MAX_SAMPLE_AGE_NS // 1_000_000,
        "joint_names": JOINT_NAMES,
        "state_dim": STATE_DIM,
        "num_episodes": len(episodes),
        "total_frames": total_frames,
        "total_dropped_frames": total_dropped,
        "splits": splits,
        "episodes": manifest_episodes,
    }
    if extra_manifest:
        manifest.update(extra_manifest)
    _write_json_atomic(staging / MANIFEST_NAME, manifest)
    return manifest


def load_manifest(dataset: str | Path) -> dict:
    path = Path(dataset) / MANIFEST_NAME
    if not path.is_file():
        raise FileNotFoundError(f"not a piperlab normalized dataset (missing {MANIFEST_NAME}): {dataset}")
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def load_episode(dataset: str | Path, entry: dict) -> dict:
    path = Path(dataset) / entry["file"]
    if not path.is_file():
        raise FileNotFoundError(f"episode file listed in manifest is missing: {path}")
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


# ---------------------------------------------------------------------------
# mock data


def _mock_rgb_frame(ep_index: int, frame_index: int, size: int) -> np.ndarray:
    """Deterministic, visibly synthetic RGB frame.

    Unnatural flat colors, an episode-colored border, a moving cyan block and
    a fixed 0xC0DE bit marker make synthetic origin obvious at a glance.
    """
    img = np.zeros((size, size, 3), dtype=np.uint8)
    # vertical gradient background tinted per episode
    base = np.linspace(30, 120, size, dtype=np.int32)
    img[..., 0] = (base[:, None] // 2 + 17 * ep_index) % 256
    img[..., 1] = base[:, None] % 256
    img[..., 2] = (base[None, :] // 3 + 60) % 256
    # episode-colored 4px border (bright, saturated, clearly artificial)
    palette = np.array(
        [[255, 0, 255], [0, 255, 0], [255, 128, 0], [0, 128, 255], [255, 255, 0], [255, 0, 128]],
        dtype=np.uint8,
    )
    color = palette[ep_index % len(palette)]
    img[:4, :] = color
    img[-4:, :] = color
    img[:, :4] = color
    img[:, -4:] = color
    # moving cyan block (position encodes the frame index)
    block = 24
    span = size - block - 8
    x = 4 + (frame_index * 7) % max(span, 1)
    y = 4 + (frame_index * 3) % max(span, 1)
    img[y : y + block, x : x + block] = (0, 255, 255)
    # fixed synthetic marker: 16 bits of SYNTHETIC_MAGIC as black/white 4px cells
    for bit in range(16):
        row, col = divmod(bit, 8)
        value = 255 if (SYNTHETIC_MAGIC >> bit) & 1 else 0
        img[8 + row * 4 : 12 + row * 4, 8 + col * 4 : 12 + col * 4] = value
    # frame index as white cells in the bottom strip (above the border)
    for bit in range(16):
        if (frame_index >> bit) & 1:
            img[size - 8 : size - 4, 8 + bit * 4 : 12 + bit * 4] = 255
    return img


def _mock_depth_frame(frame_index: int, size: int) -> np.ndarray:
    """Deterministic uint16 depth in raw units (1 unit = depth_scale_m)."""
    yy, xx = np.mgrid[0:size, 0:size]
    depth = 800 + (xx + yy) // 4  # slanted plane in mm
    block = 24
    span = size - block - 8
    x = 4 + (frame_index * 7) % max(span, 1)
    y = 4 + (frame_index * 3) % max(span, 1)
    depth[y : y + block, x : x + block] += 300  # raised block matching the RGB block
    return depth.astype(np.uint16)


def _mock_joint_trajectories(ep_index: int, frames: int, fps: int, rng: np.random.Generator):
    """Return (measured, commanded) trajectories, distinct by construction.

    The commanded action leads the measured state by one sample and carries an
    additional per-joint bias plus a higher-frequency term, so actions can
    never be confused with a copy of the observed state.
    """
    t = np.arange(frames, dtype=np.float64) / fps
    phases = rng.uniform(0.0, 2.0 * np.pi, size=STATE_DIM)
    amps = np.linspace(0.15, 0.35, STATE_DIM) * (1.0 + 0.05 * ep_index)
    freqs = np.linspace(0.25, 0.6, STATE_DIM)
    bias = np.linspace(-0.04, 0.04, STATE_DIM)

    def trajectory(time):
        signal = amps[None, :] * np.sin(2.0 * np.pi * freqs[None, :] * time[:, None] + phases[None, :])
        signal[:, -1] = 0.04 + 0.02 * np.sin(2.0 * np.pi * 0.25 * time + phases[-1])  # gripper, metres
        return signal

    commanded = trajectory(t) + bias[None, :] + 0.01 * np.sin(2.0 * np.pi * 1.5 * t[:, None])
    commanded[:, -1] = np.clip(commanded[:, -1], 0.0, 0.085)
    lead = np.vstack([commanded[1:], commanded[-1:]])  # action leads measurement
    measured = 0.85 * commanded + 0.002 * np.sin(2.0 * np.pi * 3.0 * t[:, None] + phases[None, :])
    measured[:, -1] = np.clip(measured[:, -1], 0.0, 0.085)
    return measured.astype(np.float32), lead.astype(np.float32)


def generate_mock(output: str, episodes: int = 5, frames: int = 40, fps: int = DEFAULT_FPS) -> dict:
    """Generate a deterministic, visibly synthetic normalized dataset.

    Five episodes (default) of RGB uint8 frames, uint16 depth with
    ``depth_scale_m``, camera intrinsics, measured states and distinct
    commanded actions, written as episode NPZ files plus a manifest with a
    whole-episode train/val split.
    """
    if episodes < 1 or frames < 2 or fps < 1:
        raise ValueError("episodes >= 1, frames >= 2 and fps >= 1 are required")
    out = Path(output)
    staging = _stage_dir(out)
    try:
        aligned = []
        period_ns = int(round(1e9 / fps))
        size = MOCK_IMAGE_SIZE
        intrinsics = np.array(
            [[600.0, 0.0, size / 2.0], [0.0, 600.0, size / 2.0], [0.0, 0.0, 1.0]], dtype=np.float64
        )
        for ep_index in range(episodes):
            rng = np.random.default_rng(MOCK_SEED + ep_index)
            measured, commanded = _mock_joint_trajectories(ep_index, frames, fps, rng)
            stamps = MOCK_BASE_STAMP_NS + ep_index * 3_600_000_000_000 + np.arange(frames) * period_ns
            rgb = np.stack([_mock_rgb_frame(ep_index, k, size) for k in range(frames)])
            depth = np.stack([_mock_depth_frame(k, size) for k in range(frames)])
            aligned.append(
                AlignedEpisode(
                    index=ep_index,
                    task=f"SYNTHETIC_mock_task_ep{ep_index}",
                    rgb=rgb,
                    states=measured,
                    actions=commanded,
                    stamps_ns=stamps.astype(np.int64),
                    depth=depth,
                    depth_scale_m=DEFAULT_DEPTH_SCALE_M,
                    intrinsics=intrinsics,
                    dropped_frames=0,
                    drop_reasons={},
                )
            )
        manifest = _write_dataset(
            staging,
            aligned,
            provenance="mock",
            synthetic=True,
            fps=fps,
            source=None,
            extra_manifest={"seed": MOCK_SEED, "synthetic_marker": hex(SYNTHETIC_MAGIC)},
        )
        _commit(staging, out)
    except Exception:
        _abort(staging)
        raise
    return {
        "output": str(out),
        "provenance": "mock",
        "synthetic": True,
        "episodes": episodes,
        "frames_per_episode": frames,
        "total_frames": episodes * frames,
        "fps": fps,
        "seed": MOCK_SEED,
        "splits": manifest["splits"],
        "files": [entry["file"] for entry in manifest["episodes"]],
    }


# ---------------------------------------------------------------------------
# MCAP conversion (synthetic JSON records or ROS 2 CDR)


def _decode_jpeg(data: bytes) -> np.ndarray:
    from PIL import Image

    with Image.open(io.BytesIO(data)) as img:
        return np.asarray(img.convert("RGB"), dtype=np.uint8)


def _decode_depth_payload(width: int, height: int, step: int, data: bytes) -> np.ndarray:
    raw = np.frombuffer(data, dtype=np.uint8)
    if step and step >= width * 2 and raw.size >= step * height:
        rows = raw[: step * height].reshape(height, step)
        return rows[:, : width * 2].copy().view(np.uint16).reshape(height, width)
    return raw[: width * height * 2].reshape(height, width, 2).view(np.uint16).reshape(height, width)


def _parse_event(payload) -> dict | None:
    """Normalize an /lab/events payload (std_msgs/String of JSON) to a dict."""
    data = payload
    if isinstance(payload, dict) and "data" in payload and len(payload) == 1:
        data = payload["data"]
    if isinstance(data, (bytes, bytearray)):
        data = data.decode("utf-8", errors="replace")
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except ValueError:
            return None
    if isinstance(data, dict) and "event" in data:
        return data
    return None


def _message_stamp_ns(message: dict) -> int:
    header = message.get("header") or {}
    stamp = header.get("stamp") or {}
    sec = int(stamp.get("sec", stamp.get("secs", 0)))
    nsec = int(stamp.get("nanosec", stamp.get("nsec", stamp.get("nsecs", 0))))
    return sec * 1_000_000_000 + nsec


def _normalize_payload(topic: str, message: dict, depth_scale_default: float):
    """Convert one decoded message into a normalized payload, or None to skip."""
    if topic == TOPIC_OBSERVATION or topic == TOPIC_APPLIED_ACTION:
        return _joint_vector(list(message.get("name", [])), list(message.get("position", [])), topic)
    if topic == TOPIC_RGB:
        if "data_b64" in message:
            blob = base64.b64decode(message["data_b64"])
        else:
            blob = bytes(message.get("data", b""))
        fmt = str(message.get("format", "jpeg")).lower()
        if "jpeg" in fmt or "jpg" in fmt or "png" in fmt:
            return _decode_jpeg(blob)
        width, height = int(message["width"]), int(message["height"])
        return np.frombuffer(blob, dtype=np.uint8).reshape(height, width, 3).copy()
    if topic == TOPIC_DEPTH:
        if "data_b64" in message:
            blob = base64.b64decode(message["data_b64"])
        else:
            blob = bytes(message.get("data", b""))
        encoding = str(message.get("encoding", "16UC1"))
        if encoding != "16UC1":
            raise ValueError(f"{topic}: expected encoding 16UC1, got {encoding!r}")
        width, height = int(message["width"]), int(message["height"])
        step = int(message.get("step", width * 2))
        return _decode_depth_payload(width, height, step, blob)
    if topic == TOPIC_CAMERA_INFO:
        k = [float(v) for v in message.get("k", message.get("K", []))]
        if len(k) != 9:
            raise ValueError(f"{topic}: expected 9 intrinsics values, got {len(k)}")
        return {"k": k, "width": int(message.get("width", 0)), "height": int(message.get("height", 0))}
    raise ValueError(f"unsupported topic: {topic}")


def _collect_records(records, depth_scale_default):
    """Group flat (stamp, topic, message) records into episodes via /lab/events.

    Returns (episodes, stats). Without any episode events the whole stream is
    one implicit episode; records outside event bounds are counted and skipped.
    """
    sorted_records = sorted(records, key=lambda item: item[0])
    has_events = any(topic == TOPIC_EVENTS and (_parse_event(msg) or {}).get('event') == 'episode_start'
                     for _, topic, msg in sorted_records)

    episodes = []
    skipped_bad = 0
    outside = 0

    def append_to(ep, topic, stamp, payload):
        if topic == TOPIC_CAMERA_INFO:
            ep.camera_infos.append((stamp, payload))
        else:
            ep.streams.setdefault(topic, []).append((stamp, payload))

    if not has_events:
        ep = RawEpisode(index=0, task="recorded_task", start_ns=0, end_ns=0, depth_scale_m=depth_scale_default)
        stamps = []
        for stamp, topic, message in sorted_records:
            if topic != TOPIC_CAMERA_INFO and topic not in STREAM_TOPICS:
                continue
            try:
                append_to(ep, topic, stamp, _normalize_payload(topic, message, depth_scale_default))
            except ValueError:
                skipped_bad += 1
                continue
            if topic != TOPIC_CAMERA_INFO:
                stamps.append(stamp)
        if not stamps:
            return [], {"skipped_bad_messages": skipped_bad, "records_outside_episodes": outside}
        ep.start_ns, ep.end_ns = min(stamps), max(stamps)
        episodes.append(ep)
    else:
        current = None
        for stamp, topic, message in sorted_records:
            if topic == TOPIC_EVENTS:
                event = _parse_event(message)
                if event is None:
                    continue
                kind = str(event.get("event"))
                if kind == "episode_start":
                    if current is not None:  # previous episode was never closed
                        stamps = [s for msgs in current.streams.values() for s, _ in msgs]
                        if stamps:
                            current.end_ns = max(current.end_ns, max(stamps))
                            episodes.append(current)
                    index = int(event.get("episode_index", event.get("episode", len(episodes))))
                    current = RawEpisode(
                        index=index,
                        task=str(event.get("task", "recorded_task")),
                        start_ns=stamp,
                        end_ns=stamp,
                        depth_scale_m=depth_scale_default,
                    )
                elif kind == "episode_end" and current is not None:
                    current.end_ns = stamp
                    episodes.append(current)
                    current = None
                continue
            if topic != TOPIC_CAMERA_INFO and topic not in STREAM_TOPICS:
                continue
            if current is None:
                outside += 1
                continue
            try:
                append_to(current, topic, stamp, _normalize_payload(topic, message, depth_scale_default))
            except ValueError:
                skipped_bad += 1
        if current is not None:  # unterminated episode: close at its last stream stamp
            stamps = [s for msgs in current.streams.values() for s, _ in msgs]
            if stamps:
                current.end_ns = max(current.end_ns, max(stamps))
                episodes.append(current)
        if not episodes:
            return [], {"skipped_bad_messages": skipped_bad, "records_outside_episodes": outside}
        episodes.sort(key=lambda ep: ep.start_ns)
    for ep in episodes:  # keep streams stamp-sorted for the aligner, index densely
        for topic in ep.streams:
            ep.streams[topic].sort(key=lambda item: item[0])
        ep.camera_infos.sort(key=lambda item: item[0])
        if not ep.streams.get(TOPIC_OBSERVATION):
            raise ValueError(f"episode {ep.index}: no {TOPIC_OBSERVATION} messages")
    for new_index, ep in enumerate(episodes):
        ep.index = new_index
    return episodes, {"skipped_bad_messages": skipped_bad, "records_outside_episodes": outside}


def _read_json_fixture(path: Path):
    with open(path, encoding="utf-8") as handle:
        doc = json.load(handle)
    if not isinstance(doc, dict):
        raise ValueError(f"JSON fixture must be an object: {path}")
    depth_scale = float(doc.get("depth_scale_m", DEFAULT_DEPTH_SCALE_M))
    records = []
    if "episodes" in doc:
        for ep_pos, ep in enumerate(doc["episodes"]):
            ep_index = int(ep.get("episode_index", ep_pos))
            task = str(ep.get("task", "recorded_task"))
            ep_records = ep.get("records", [])
            stamps = [int(r["stamp_ns"]) for r in ep_records if r.get("topic") != TOPIC_EVENTS]
            start = min(stamps) if stamps else 0
            records.append((start, TOPIC_EVENTS, {"data": json.dumps({
                "event": "episode_start", "episode_index": ep_index, "task": task})}))
            for record in ep_records:
                records.append((int(record["stamp_ns"]), str(record["topic"]), record["message"]))
            end = max(stamps) if stamps else start
            records.append((end, TOPIC_EVENTS, {"data": json.dumps({"event": "episode_end", "episode_index": ep_index})}))
    else:
        for record in doc.get("records", []):
            records.append((int(record["stamp_ns"]), str(record["topic"]), record["message"]))
    return records, depth_scale, bool(doc.get("synthetic", False))


def _read_mcap_cdr(path: Path):
    """Read a rosbag2 MCAP file: CDR payloads decoded via mcap_ros2.reader.

    Returns (records, dropped_no_stamp). Alignment uses header stamps only;
    a message without a source timestamp is counted and never used — no
    wall-clock substitution.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from mcap_ros2.reader import read_ros2_messages

        records = []
        dropped_no_stamp = 0
        for wrapped in read_ros2_messages(str(path)):
            topic = wrapped.channel.topic
            if topic != TOPIC_CAMERA_INFO and topic not in STREAM_TOPICS + (TOPIC_EVENTS,):
                continue
            ros_msg = wrapped.ros_msg
            message = {
                name: getattr(ros_msg, name)
                for name in getattr(ros_msg, "__slots__", ())
                if not name.startswith("_")
            }
            header = message.get("header")
            if header is not None:
                stamp = header.stamp
                message["header"] = {"stamp": {"sec": stamp.sec, "nanosec": stamp.nanosec}}
            if topic == TOPIC_EVENTS:
                records.append((wrapped.publish_time_ns, topic, {"data": message.get("data", "")}))
                continue
            stamp_ns = _message_stamp_ns(message)
            if stamp_ns == 0:
                dropped_no_stamp += 1
                continue
            records.append((stamp_ns, topic, message))
    return records, dropped_no_stamp


def convert_mcap(source: str, output: str, fps: int = DEFAULT_FPS) -> dict:
    """Convert an MCAP source (synthetic JSON records or ROS 2 CDR bag) into a
    normalized local dataset of episode NPZ files plus a manifest.

    Samples are aligned on the fps grid using source timestamps only; a source
    message older than 200 ms is never used and future messages are never
    used. Target actions come exclusively from /lab/applied_action.
    """
    src = Path(source)
    if src.is_dir():
        return convert_directory(src, Path(output), fps)
    if not src.is_file():
        raise FileNotFoundError(f"MCAP source not found: {source}")
    suffix = src.suffix.lower()
    if suffix == ".json":
        provenance = "mcap-json"
        records, depth_scale, synthetic = _read_json_fixture(src)
        extra = {}
    elif suffix == ".mcap":
        provenance = "mcap-cdr"
        records, dropped_no_stamp = _read_mcap_cdr(src)
        depth_scale = DEFAULT_DEPTH_SCALE_M
        synthetic = any(topic == TOPIC_EVENTS and (_parse_event(msg) or {}).get('synthetic') is True
                        for _, topic, msg in records)
        extra = {"dropped_no_stamp_messages": dropped_no_stamp}
    else:
        raise ValueError(f"unsupported MCAP source type {suffix!r} (expected .json or .mcap)")

    raw_episodes, stats = _collect_records(records, depth_scale)
    if not raw_episodes:
        raise ValueError(f"no episodes found in source: {source}")

    out = Path(output)
    staging = _stage_dir(out)
    try:
        aligned = [_align_episode(raw, fps) for raw in raw_episodes]
        manifest = _write_dataset(
            staging,
            aligned,
            provenance=provenance,
            synthetic=synthetic,
            fps=fps,
            source=str(src),
            extra_manifest={"depth_scale_m": depth_scale, **extra, **stats},
        )
        _commit(staging, out)
    except Exception:
        _abort(staging)
        raise
    drop_reasons: dict = {}
    for entry in manifest["episodes"]:
        for reason, count in entry["drop_reasons"].items():
            drop_reasons[reason] = drop_reasons.get(reason, 0) + count
    return {
        "output": str(out),
        "provenance": provenance,
        "source": str(src),
        "synthetic": synthetic,
        "episodes": len(aligned),
        "total_frames": manifest["total_frames"],
        "total_dropped_frames": manifest["total_dropped_frames"],
        "drop_reasons": drop_reasons,
        "fps": fps,
        "splits": manifest["splits"],
        "files": [entry["file"] for entry in manifest["episodes"]],
        **extra,
        **stats,
    }


def convert_directory(source: Path, output: Path, fps=20):
    """Convert one bag at a time and merge whole episodes, bounding decoded memory.

    Prefer one bag per episode. MCAP split files from one long episode require
    reassembly before conversion; automatic splitting of one episode is rejected.
    """
    import shutil
    files=sorted(source.rglob('*.mcap'))
    if not files: raise ValueError('No MCAP files in directory')
    output=output.resolve(); source=source.resolve()
    if output.is_relative_to(source): raise ValueError('Output must be outside source directory')
    for bag in source.rglob('metadata.yaml'):
        import yaml
        info=yaml.safe_load(bag.read_text(encoding='utf-8'))['rosbag2_bagfile_information']
        if len(info.get('relative_file_paths',[]))>1:
            raise ValueError(f'Split bag must be reassembled before episode conversion: {bag}')
    staging=_stage_dir(output); entries=[]; synthetic=None; total=0; dropped=0
    (staging/'episodes').mkdir()
    for index, file in enumerate(files):
        part=staging/'source_segments'/f'bag-{index:04d}'
        convert_mcap(str(file),str(part),fps); manifest=load_manifest(part)
        if synthetic is None: synthetic=manifest['synthetic']
        if synthetic != manifest['synthetic']: raise ValueError('Do not mix real and synthetic episodes')
        for entry in manifest['episodes']:
            new={**entry,'index':len(entries),'source_bag':str(file)}
            new['file']=f"episodes/episode_{len(entries):04d}.npz"
            shutil.copy2(part/entry['file'],staging/new['file'])
            entries.append(new); total+=entry['frames']; dropped+=entry['dropped_frames']
    splits=_assign_splits(len(entries))
    for entry in entries: entry['split']='val' if entry['index'] in splits['val'] else 'train'
    merged={**manifest,'source':str(source),'source_files':[str(f) for f in files],'episodes':entries,'splits':splits,
            'num_episodes':len(entries),'total_frames':total,'total_dropped_frames':dropped}
    _write_json_atomic(staging/MANIFEST_NAME,merged); _commit(staging,output)
    return {'output':str(output),'episodes':len(entries),'frames':total,'splits':splits,'synthetic':synthetic}


# ---------------------------------------------------------------------------
# LeRobot export (v0.6.1 public API)


def _require_lerobot():
    try:
        import lerobot  # noqa: PLC0415 - intentionally lazy
    except ImportError as exc:
        raise RuntimeError(
            "LeRobot is required for export_lerobot but is not installed in this "
            "environment (root owns virtualenvs; expected lerobot==0.6.1)"
        ) from exc
    version = getattr(lerobot, "__version__", "0.0.0")
    if version != '0.6.1':
        raise RuntimeError(f"export_lerobot was implemented against LeRobot v0.6.1 APIs, found {version}")
    return lerobot


def export_lerobot(source: str, output: str, repo_id: str = "local/piper_mock") -> dict:
    """Export a normalized piperlab dataset as a real LeRobot v0.6.1 dataset.

    Frames are written through ``LeRobotDataset.create`` / ``add_frame`` /
    ``save_episode`` / ``finalize`` with RGB as an image feature (no video
    encoding dependency). Depth stays in the NPZ episodes: ACT consumes RGB +
    state + action only.
    """
    _require_lerobot()
    from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: PLC0415

    manifest = load_manifest(source)
    out = Path(output)
    staging = _stage_dir(out)
    staging.rmdir()  # LeRobot.create owns creation; this directory is newly created and empty.

    first = load_episode(source, manifest["episodes"][0])
    height, width = first["rgb"].shape[1:3]
    features = {
        "observation.images.rgb": {
            "dtype": "image",
            "shape": [height, width, 3],
            "names": ["height", "width", "channels"],
        },
        "observation.state": {"dtype": "float32", "shape": [STATE_DIM], "names": list(JOINT_NAMES)},
        "action": {"dtype": "float32", "shape": [STATE_DIM], "names": list(JOINT_NAMES)},
    }

    try:
        dataset = LeRobotDataset.create(
            repo_id=repo_id,
            fps=int(manifest["fps"]),
            features=features,
            root=staging,
            robot_type="piper",
            use_videos=False,
        )
        exported = []
        for entry in manifest["episodes"]:
            episode = load_episode(source, entry)
            task = str(episode["task"])
            n_frames = int(episode["states"].shape[0])
            for k in range(n_frames):
                dataset.add_frame(
                    {
                        "observation.images.rgb": episode["rgb"][k],
                        "observation.state": episode["states"][k].astype(np.float32),
                        "action": episode["actions"][k].astype(np.float32),
                        "task": task,
                    }
                )
            dataset.save_episode()
            exported.append({"source_index": entry["index"], "lerobot_index": len(exported), "split": entry["split"]})
        dataset.finalize()
        sidecar = {
            "repo_id": repo_id,
            "source": str(source),
            "source_provenance": manifest["provenance"],
            "synthetic": manifest["synthetic"],
            "episodes": exported,
            "splits": {
                "train": [e["lerobot_index"] for e in exported if e["split"] == "train"],
                "val": [e["lerobot_index"] for e in exported if e["split"] == "val"],
            },
        }
        _write_json_atomic(staging / "piperlab_export.json", sidecar)
        _commit(staging, out)
    except Exception:
        _abort(staging)
        raise

    import lerobot  # noqa: PLC0415

    return {
        "output": str(out),
        "repo_id": repo_id,
        "lerobot_version": getattr(lerobot, "__version__", "unknown"),
        "episodes": len(exported),
        "total_frames": manifest["total_frames"],
        "fps": manifest["fps"],
        "features": sorted(features),
        "splits": sidecar["splits"],
        "depth_exported": False,
    }
