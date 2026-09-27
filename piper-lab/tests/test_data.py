"""Tests for piperlab.data: mock generation, MCAP conversion, split rules.

Runs with the core environment only (numpy, Pillow, mcap, mcap-ros2-support,
pytest). The LeRobot export test skips itself when lerobot is unavailable;
the missing-dependency behaviour is always tested.
"""

import base64
import io
import json
import warnings
from pathlib import Path

import numpy as np
import pytest

from piperlab import data

FPS = 20
PERIOD_NS = 50_000_000  # 20 Hz
T0 = 1_700_000_000_000_000_000
SIZE = 64  # small frames keep the fixtures fast


# ---------------------------------------------------------------------------
# fixture builders


def _jpeg_b64(array: np.ndarray) -> str:
    from PIL import Image

    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="JPEG", quality=90)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _rgb_frame(seed: int) -> np.ndarray:
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    frame = np.stack(
        [(xx * 3 + seed) % 256, (yy * 5 + seed) % 256, np.full((SIZE, SIZE), seed % 256)],
        axis=-1,
    )
    return frame.astype(np.uint8)


def _depth_bytes(seed: int) -> bytes:
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    depth = (900 + xx + yy + seed).astype("<u2")
    return depth.tobytes()


def _joint_message(joint1: float, gripper: float = 0.04) -> dict:
    positions = [joint1, 0.1, -0.1, 0.2, -0.2, 0.05, gripper]
    return {"name": list(data.JOINT_NAMES), "position": positions}


def _record(stamp_ns: int, topic: str, message: dict) -> dict:
    return {"stamp_ns": int(stamp_ns), "topic": topic, "message": message}


def _rgb_record(stamp_ns: int, seed: int) -> dict:
    return _record(stamp_ns, data.TOPIC_RGB, {"format": "jpeg", "data_b64": _jpeg_b64(_rgb_frame(seed))})


def _depth_record(stamp_ns: int, seed: int) -> dict:
    return _record(
        stamp_ns,
        data.TOPIC_DEPTH,
        {
            "height": SIZE,
            "width": SIZE,
            "encoding": "16UC1",
            "step": SIZE * 2,
            "data_b64": base64.b64encode(_depth_bytes(seed)).decode("ascii"),
        },
    )


def _camera_info_record(stamp_ns: int) -> dict:
    return _record(
        stamp_ns,
        data.TOPIC_CAMERA_INFO,
        {"width": SIZE, "height": SIZE, "k": [500.0, 0.0, 32.0, 0.0, 500.0, 32.0, 0.0, 0.0, 1.0]},
    )


def _episode_records(ep: int, frames: int, *, base: int = T0, with_events: bool = True) -> list:
    """Full-rate records for one episode: 4 streams fresh at every sample."""
    records = []
    start = base + ep * 3_600_000_000_000
    if with_events:
        records.append(
            _record(
                start,
                data.TOPIC_EVENTS,
                {"data": json.dumps({"event": "episode_start", "episode_index": ep, "task": f"task_ep{ep}"})},
            )
        )
    records.append(_camera_info_record(start))
    for k in range(frames):
        t = start + k * PERIOD_NS
        records.append(_record(t, data.TOPIC_OBSERVATION, _joint_message(0.11 * (ep + 1) + 0.001 * k)))
        records.append(_record(t, data.TOPIC_APPLIED_ACTION, _joint_message(0.50 * (ep + 1) + 0.001 * k, 0.07)))
        records.append(_rgb_record(t, k + ep * 100))
        records.append(_depth_record(t, k))
    if with_events:
        end = start + (frames - 1) * PERIOD_NS
        records.append(
            _record(end, data.TOPIC_EVENTS, {"data": json.dumps({"event": "episode_end", "episode_index": ep})})
        )
    return records


def _write_json_fixture(path: Path, doc: dict) -> Path:
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# mock generation


def test_generate_mock_contract(tmp_path):
    report = data.generate_mock(str(tmp_path / "mock"))
    assert report["episodes"] == 5
    assert report["frames_per_episode"] == 40
    assert report["synthetic"] is True
    assert report["provenance"] == "mock"

    manifest = data.load_manifest(tmp_path / "mock")
    assert manifest["num_episodes"] == 5
    assert manifest["splits"]["val"] == [4]  # 20% of 5 episodes, whole episodes
    assert sorted(manifest["splits"]["train"] + manifest["splits"]["val"]) == list(range(5))

    for entry in manifest["episodes"]:
        episode = data.load_episode(tmp_path / "mock", entry)
        assert episode["rgb"].shape == (40, 224, 224, 3)
        assert episode["rgb"].dtype == np.uint8
        assert episode["depth"].shape == (40, 224, 224)
        assert episode["depth"].dtype == np.uint16
        assert episode["states"].shape == (40, 7)
        assert episode["actions"].shape == (40, 7)
        assert episode["states"].dtype == np.float32
        assert episode["actions"].dtype == np.float32
        assert float(episode["depth_scale_m"]) == pytest.approx(0.001)
        assert episode["camera_intrinsics"].shape == (3, 3)
        assert str(episode["task"]).startswith("SYNTHETIC_")
        # commanded actions are distinct from measured states
        diff = np.abs(episode["actions"] - episode["states"])
        assert diff.mean() > 5e-3
        assert not np.allclose(episode["actions"], episode["states"])


def test_generate_mock_deterministic(tmp_path):
    data.generate_mock(str(tmp_path / "a"))
    data.generate_mock(str(tmp_path / "b"))
    manifest_a = json.loads((tmp_path / "a" / "manifest.json").read_text())
    manifest_b = json.loads((tmp_path / "b" / "manifest.json").read_text())
    assert manifest_a == manifest_b  # includes per-file sha256
    for entry in manifest_a["episodes"]:
        bytes_a = (tmp_path / "a" / entry["file"]).read_bytes()
        bytes_b = (tmp_path / "b" / entry["file"]).read_bytes()
        assert bytes_a == bytes_b


def test_generate_mock_visibly_synthetic(tmp_path):
    data.generate_mock(str(tmp_path / "mock"), episodes=1, frames=4)
    manifest = data.load_manifest(tmp_path / "mock")
    episode = data.load_episode(tmp_path / "mock", manifest["episodes"][0])
    frame = episode["rgb"][0]
    # the fixed 0xC0DE marker must be present in the top-left cells
    bits = 0
    for bit in range(16):
        row, col = divmod(bit, 8)
        if frame[8 + row * 4, 8 + col * 4, 0] > 127:
            bits |= 1 << bit
    assert bits == data.SYNTHETIC_MAGIC
    # flat saturated border: not a natural image
    assert frame[0, 0].max() == 255 and frame[0, 0].min() == 0


def test_generate_mock_refuses_overwrite(tmp_path):
    data.generate_mock(str(tmp_path / "mock"), episodes=1, frames=4)
    with pytest.raises(FileExistsError):
        data.generate_mock(str(tmp_path / "mock"), episodes=1, frames=4)


# ---------------------------------------------------------------------------
# JSON fixture conversion


def test_convert_json_fixture_episodes_and_provenance(tmp_path):
    doc = {
        "provenance": "unit-test",
        "synthetic": True,
        "depth_scale_m": 0.001,
        "episodes": [
            {"episode_index": 0, "task": "task_ep0", "records": _episode_records(0, 6, with_events=False)},
            {"episode_index": 1, "task": "task_ep1", "records": _episode_records(1, 6, with_events=False)},
        ],
    }
    source = _write_json_fixture(tmp_path / "fixture.json", doc)
    report = data.convert_mcap(str(source), str(tmp_path / "out"))
    assert report["provenance"] == "mcap-json"
    assert report["episodes"] == 2
    assert report["total_frames"] == 12
    assert report["total_dropped_frames"] == 0

    manifest = data.load_manifest(tmp_path / "out")
    assert manifest["synthetic"] is True
    assert [e["task"] for e in manifest["episodes"]] == ["task_ep0", "task_ep1"]
    ep0 = data.load_episode(tmp_path / "out", manifest["episodes"][0])
    ep1 = data.load_episode(tmp_path / "out", manifest["episodes"][1])
    # episode separation: joint1 of ep1 must never appear in ep0 states
    assert ep0["states"][:, 0].max() < ep1["states"][:, 0].min()
    # actions come from /lab/applied_action (0.50 * (ep+1)), not from states
    assert ep0["actions"][0, 0] == pytest.approx(0.50, abs=1e-6)
    assert ep0["states"][0, 0] == pytest.approx(0.11, abs=1e-6)
    assert ep1["actions"][0, 0] == pytest.approx(1.00, abs=1e-6)
    assert ep0["depth"].dtype == np.uint16
    assert ep0["camera_intrinsics"][0, 0] == pytest.approx(500.0)


def test_convert_json_flat_records_with_events(tmp_path):
    records = _episode_records(0, 5) + _episode_records(1, 5)
    source = _write_json_fixture(tmp_path / "fixture.json", {"records": records})
    report = data.convert_mcap(str(source), str(tmp_path / "out"))
    assert report["episodes"] == 2
    manifest = data.load_manifest(tmp_path / "out")
    assert manifest["splits"]["val"] == [1]
    ep0 = data.load_episode(tmp_path / "out", manifest["episodes"][0])
    assert str(ep0["task"]) == "task_ep0"


def test_convert_stale_frames_dropped(tmp_path):
    # observation only at k=0; every other stream fresh at 20 Hz for 9 samples
    start = T0
    records = [
        _record(start, data.TOPIC_EVENTS,
                {"data": json.dumps({"event": "episode_start", "episode_index": 0, "task": "stale"})}),
        _record(start, data.TOPIC_OBSERVATION, _joint_message(0.3)),
    ]
    for k in range(9):
        t = start + k * PERIOD_NS
        records.append(_record(t, data.TOPIC_APPLIED_ACTION, _joint_message(0.6)))
        records.append(_rgb_record(t, k))
        records.append(_depth_record(t, k))
    end = start + 8 * PERIOD_NS
    records.append(
        _record(end, data.TOPIC_EVENTS, {"data": json.dumps({"event": "episode_end", "episode_index": 0})})
    )
    source = _write_json_fixture(tmp_path / "fixture.json", {"records": records})
    report = data.convert_mcap(str(source), str(tmp_path / "out"))
    # samples k=0..4 have observation age 0,50,100,150,200 ms (kept);
    # k=5..8 have age 250..400 ms and must be dropped, never backfilled
    assert report["total_frames"] == 5
    assert report["total_dropped_frames"] == 4
    manifest = data.load_manifest(tmp_path / "out")
    reasons = manifest["episodes"][0]["drop_reasons"]
    assert reasons == {f"stale_message:{data.TOPIC_OBSERVATION}": 4}
    episode = data.load_episode(tmp_path / "out", manifest["episodes"][0])
    assert episode["states"].shape[0] == 5


def test_convert_future_messages_never_used(tmp_path):
    # observation A at k=0, observation B at k=5; samples k<5 must use A only
    start = T0
    records = [
        _record(start, data.TOPIC_EVENTS,
                {"data": json.dumps({"event": "episode_start", "episode_index": 0, "task": "future"})}),
        _record(start, data.TOPIC_OBSERVATION, _joint_message(1.0)),
        _record(start + 5 * PERIOD_NS, data.TOPIC_OBSERVATION, _joint_message(2.0)),
    ]
    for k in range(6):
        t = start + k * PERIOD_NS
        records.append(_record(t, data.TOPIC_APPLIED_ACTION, _joint_message(0.6)))
        records.append(_rgb_record(t, k))
        records.append(_depth_record(t, k))
    records.append(
        _record(start + 5 * PERIOD_NS, data.TOPIC_EVENTS,
                {"data": json.dumps({"event": "episode_end", "episode_index": 0})})
    )
    source = _write_json_fixture(tmp_path / "fixture.json", {"records": records})
    report = data.convert_mcap(str(source), str(tmp_path / "out"))
    assert report["total_frames"] == 6
    manifest = data.load_manifest(tmp_path / "out")
    episode = data.load_episode(tmp_path / "out", manifest["episodes"][0])
    joint1 = episode["states"][:, 0]
    assert np.allclose(joint1[:5], 1.0)  # B (2.0) is in the future for k<5
    assert joint1[5] == pytest.approx(2.0, abs=1e-6)


def test_convert_never_invents_action_from_observation(tmp_path):
    # full observation/rgb/depth streams but no /lab/applied_action at all
    start = T0
    records = [
        _record(start, data.TOPIC_EVENTS,
                {"data": json.dumps({"event": "episode_start", "episode_index": 0, "task": "noact"})}),
    ]
    for k in range(4):
        t = start + k * PERIOD_NS
        records.append(_record(t, data.TOPIC_OBSERVATION, _joint_message(0.3)))
        records.append(_rgb_record(t, k))
        records.append(_depth_record(t, k))
    records.append(
        _record(start + 3 * PERIOD_NS, data.TOPIC_EVENTS,
                {"data": json.dumps({"event": "episode_end", "episode_index": 0})})
    )
    source = _write_json_fixture(tmp_path / "fixture.json", {"records": records})
    with pytest.raises(ValueError, match="no frame satisfied"):
        data.convert_mcap(str(source), str(tmp_path / "out"))
    # nothing may be materialized on failure (atomic output)
    assert not (tmp_path / "out").exists()
    # Failed staging data is retained for inspection; the published output is absent.
    assert list(tmp_path.glob("out.tmp-*"))


def test_convert_refuses_overwrite_and_unknown_suffix(tmp_path):
    doc = {"episodes": [{"episode_index": 0, "records": _episode_records(0, 3, with_events=False)}]}
    source = _write_json_fixture(tmp_path / "fixture.json", doc)
    data.convert_mcap(str(source), str(tmp_path / "out"))
    with pytest.raises(FileExistsError):
        data.convert_mcap(str(source), str(tmp_path / "out"))
    weird = _write_json_fixture(tmp_path / "fixture.txt", doc)
    with pytest.raises(ValueError, match="unsupported"):
        data.convert_mcap(str(weird), str(tmp_path / "out2"))


def test_split_is_whole_episode_never_frame_random(tmp_path):
    doc = {
        "episodes": [
            {"episode_index": i, "records": _episode_records(i, 4, with_events=False)} for i in range(5)
        ]
    }
    source = _write_json_fixture(tmp_path / "fixture.json", doc)
    data.convert_mcap(str(source), str(tmp_path / "out"))
    manifest = data.load_manifest(tmp_path / "out")
    train = set(manifest["splits"]["train"])
    val = set(manifest["splits"]["val"])
    assert not train & val
    assert train | val == set(range(5))
    for entry in manifest["episodes"]:
        expected = "val" if entry["index"] in val else "train"
        assert entry["split"] == expected  # the whole episode shares one split


# ---------------------------------------------------------------------------
# real rosbag2 MCAP (CDR) conversion via mcap_ros2


HEADER_MSGDEF = """builtin_interfaces/Time stamp
string frame_id"""

TIME_MSGDEF = """int32 sec
uint32 nanosec"""

JOINTSTATE_MSGDEF = f"""std_msgs/Header header
string[] name
float64[] position
float64[] velocity
float64[] effort
================================================================================
MSG: std_msgs/Header
{HEADER_MSGDEF}
================================================================================
MSG: builtin_interfaces/Time
{TIME_MSGDEF}"""

COMPRESSEDIMAGE_MSGDEF = f"""std_msgs/Header header
string format
uint8[] data
================================================================================
MSG: std_msgs/Header
{HEADER_MSGDEF}
================================================================================
MSG: builtin_interfaces/Time
{TIME_MSGDEF}"""

IMAGE_MSGDEF = f"""std_msgs/Header header
uint32 height
uint32 width
string encoding
uint8 is_bigendian
uint32 step
uint8[] data
================================================================================
MSG: std_msgs/Header
{HEADER_MSGDEF}
================================================================================
MSG: builtin_interfaces/Time
{TIME_MSGDEF}"""

CAMERAINFO_MSGDEF = f"""std_msgs/Header header
uint32 height
uint32 width
string distortion_model
float64[] d
float64[9] k
float64[9] r
float64[12] p
uint32 binning_x
uint32 binning_y
sensor_msgs/RegionOfInterest roi
================================================================================
MSG: std_msgs/Header
{HEADER_MSGDEF}
================================================================================
MSG: builtin_interfaces/Time
{TIME_MSGDEF}
================================================================================
MSG: sensor_msgs/RegionOfInterest
uint32 x_offset
uint32 y_offset
uint32 height
uint32 width
bool do_rectify"""

STRING_MSGDEF = "string data"


def _header(stamp_ns: int) -> dict:
    return {"stamp": {"sec": stamp_ns // 1_000_000_000, "nanosec": stamp_ns % 1_000_000_000}, "frame_id": ""}


def _write_mcap_fixture(path: Path) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from mcap_ros2.writer import Writer

        frames = 5
        with open(path, "wb") as stream:
            writer = Writer(stream)
            schemas = {
                data.TOPIC_OBSERVATION: writer.register_msgdef("sensor_msgs/msg/JointState", JOINTSTATE_MSGDEF),
                data.TOPIC_APPLIED_ACTION: writer.register_msgdef("sensor_msgs/msg/JointState", JOINTSTATE_MSGDEF),
                data.TOPIC_RGB: writer.register_msgdef("sensor_msgs/msg/CompressedImage", COMPRESSEDIMAGE_MSGDEF),
                data.TOPIC_DEPTH: writer.register_msgdef("sensor_msgs/msg/Image", IMAGE_MSGDEF),
                data.TOPIC_CAMERA_INFO: writer.register_msgdef("sensor_msgs/msg/CameraInfo", CAMERAINFO_MSGDEF),
                data.TOPIC_EVENTS: writer.register_msgdef("std_msgs/msg/String", STRING_MSGDEF),
            }

            def put(topic, stamp_ns, message):
                writer.write_message(
                    topic, schemas[topic], message, log_time=stamp_ns, publish_time=stamp_ns
                )

            for ep in range(2):
                start = T0 + ep * 3_600_000_000_000
                put(data.TOPIC_EVENTS, start,
                    {"data": json.dumps({"event": "episode_start", "episode_index": ep, "task": f"cdr_ep{ep}"})})
                put(data.TOPIC_CAMERA_INFO, start, {
                    "header": _header(start),
                    "height": SIZE,
                    "width": SIZE,
                    "distortion_model": "plumb_bob",
                    "d": [],
                    "k": [510.0, 0.0, 32.0, 0.0, 510.0, 32.0, 0.0, 0.0, 1.0],
                    "r": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                    "p": [510.0, 0.0, 32.0, 0.0, 0.0, 510.0, 32.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                    "binning_x": 0,
                    "binning_y": 0,
                    "roi": {"x_offset": 0, "y_offset": 0, "height": 0, "width": 0, "do_rectify": False},
                })
                for k in range(frames):
                    t = start + k * PERIOD_NS
                    obs = {"header": _header(t), "name": list(data.JOINT_NAMES),
                           "position": [0.21 * (ep + 1) + 0.001 * k, 0.0, 0.0, 0.0, 0.0, 0.0, 0.03],
                           "velocity": [], "effort": []}
                    act = {"header": _header(t), "name": list(data.JOINT_NAMES),
                           "position": [0.77 * (ep + 1) + 0.001 * k, 0.0, 0.0, 0.0, 0.0, 0.0, 0.06],
                           "velocity": [], "effort": []}
                    rgb = {"header": _header(t), "format": "jpeg",
                           "data": base64.b64decode(_jpeg_b64(_rgb_frame(k + ep * 50)))}
                    depth = {"header": _header(t), "height": SIZE, "width": SIZE,
                             "encoding": "16UC1", "is_bigendian": 0, "step": SIZE * 2,
                             "data": _depth_bytes(k)}
                    put(data.TOPIC_OBSERVATION, t, obs)
                    put(data.TOPIC_APPLIED_ACTION, t, act)
                    put(data.TOPIC_RGB, t, rgb)
                    put(data.TOPIC_DEPTH, t, depth)
                end = start + (frames - 1) * PERIOD_NS
                put(data.TOPIC_EVENTS, end,
                    {"data": json.dumps({"event": "episode_end", "episode_index": ep})})
            writer.finish()


def test_convert_real_mcap_cdr_roundtrip(tmp_path):
    source = tmp_path / "bag.mcap"
    _write_mcap_fixture(source)
    report = data.convert_mcap(str(source), str(tmp_path / "out"))
    assert report["provenance"] == "mcap-cdr"
    assert report["episodes"] == 2
    assert report["total_frames"] == 10
    assert report["total_dropped_frames"] == 0

    manifest = data.load_manifest(tmp_path / "out")
    assert manifest["synthetic"] is False
    assert manifest["splits"]["val"] == [1]
    ep0 = data.load_episode(tmp_path / "out", manifest["episodes"][0])
    ep1 = data.load_episode(tmp_path / "out", manifest["episodes"][1])
    assert str(ep0["task"]) == "cdr_ep0"
    assert str(ep1["task"]) == "cdr_ep1"
    # measured states decoded from CDR JointState
    assert ep0["states"][0, 0] == pytest.approx(0.21, abs=1e-6)
    assert ep1["states"][0, 0] == pytest.approx(0.42, abs=1e-6)
    # actions come from /lab/applied_action, distinct from states
    assert ep0["actions"][0, 0] == pytest.approx(0.77, abs=1e-6)
    assert not np.allclose(ep0["actions"], ep0["states"])
    # images decoded from JPEG, depth from 16UC1 bytes
    assert ep0["rgb"].shape == (5, SIZE, SIZE, 3)
    assert ep0["rgb"].dtype == np.uint8
    assert ep0["depth"].dtype == np.uint16
    assert int(ep0["depth"][0, 0, 0]) == 900
    assert float(ep0["depth_scale_m"]) == pytest.approx(0.001)
    # intrinsics from /lab/camera_info
    assert ep0["camera_intrinsics"][0, 0] == pytest.approx(510.0)
    # episode separation holds on real CDR data too
    assert ep0["states"][:, 0].max() < ep1["states"][:, 0].min()


def test_convert_real_mcap_stale_observation_dropped(tmp_path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from mcap_ros2.writer import Writer

        source = tmp_path / "stale.mcap"
        with open(source, "wb") as stream:
            writer = Writer(stream)
            js = writer.register_msgdef("sensor_msgs/msg/JointState", JOINTSTATE_MSGDEF)
            ci = writer.register_msgdef("sensor_msgs/msg/CompressedImage", COMPRESSEDIMAGE_MSGDEF)
            st = writer.register_msgdef("std_msgs/msg/String", STRING_MSGDEF)
            start = T0

            def put(topic, schema, stamp_ns, message):
                writer.write_message(topic, schema, message, log_time=stamp_ns, publish_time=stamp_ns)

            put(data.TOPIC_EVENTS, st, start,
                {"data": json.dumps({"event": "episode_start", "episode_index": 0, "task": "cdr_stale"})})
            # single observation at k=0 only
            put(data.TOPIC_OBSERVATION, js, start,
                {"header": _header(start), "name": list(data.JOINT_NAMES),
                 "position": [0.3, 0, 0, 0, 0, 0, 0.03], "velocity": [], "effort": []})
            for k in range(8):
                t = start + k * PERIOD_NS
                put(data.TOPIC_APPLIED_ACTION, js, t,
                    {"header": _header(t), "name": list(data.JOINT_NAMES),
                     "position": [0.6, 0, 0, 0, 0, 0, 0.05], "velocity": [], "effort": []})
                put(data.TOPIC_RGB, ci, t,
                    {"header": _header(t), "format": "jpeg", "data": base64.b64decode(_jpeg_b64(_rgb_frame(k)))})
            end = start + 7 * PERIOD_NS
            put(data.TOPIC_EVENTS, st, end, {"data": json.dumps({"event": "episode_end", "episode_index": 0})})
            writer.finish()

    report = data.convert_mcap(str(source), str(tmp_path / "out"))
    # age 0..200 ms kept (k=0..4), 250..350 ms dropped (k=5..7)
    assert report["total_frames"] == 5
    assert report["total_dropped_frames"] == 3


# ---------------------------------------------------------------------------
# LeRobot export


def test_export_lerobot_reports_missing_dependency(tmp_path, monkeypatch):
    data.generate_mock(str(tmp_path / "mock"), episodes=1, frames=4)

    real_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == "lerobot" or name.startswith("lerobot."):
            raise ImportError("No module named 'lerobot'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)
    with pytest.raises(RuntimeError, match="LeRobot"):
        data.export_lerobot(str(tmp_path / "mock"), str(tmp_path / "lerobot_ds"))
    assert not (tmp_path / "lerobot_ds").exists()


def test_export_lerobot_roundtrip(tmp_path):
    pytest.importorskip("lerobot", reason="lerobot not installed in this environment (root owns DL envs)")
    data.generate_mock(str(tmp_path / "mock"), episodes=3, frames=6)
    report = data.export_lerobot(str(tmp_path / "mock"), str(tmp_path / "lerobot_ds"))
    assert report["episodes"] == 3
    assert report["total_frames"] == 18
    assert (tmp_path / "lerobot_ds" / "meta" / "info.json").is_file()
    sidecar = json.loads((tmp_path / "lerobot_ds" / "piperlab_export.json").read_text())
    assert sidecar["repo_id"] == "local/piper_mock"
    assert sorted(sidecar["splits"]["train"] + sidecar["splits"]["val"]) == [0, 1, 2]

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(report["repo_id"], root=tmp_path / "lerobot_ds")
    assert len(dataset) == 18
    item = dataset[0]
    assert item["observation.state"].shape == (7,)
    assert item["action"].shape == (7,)
