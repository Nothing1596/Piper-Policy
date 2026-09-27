"""Tests for piperlab.adapter: read-only connect, freshness checks, explicit
acquire/arm, and no auto-reconnect / no action replay.

A fake rosbridge transport is injected, so these tests need no ROS and no
network. The real roslibpy wiring is exercised only by root against a live
rosbridge.
"""

import json

import pytest

from piperlab import adapter
from piperlab.adapter import (
    COMMAND_TOPIC,
    CONTROL_SERVICE,
    EVENTS_TOPIC,
    OBSERVATION_TOPIC,
    BridgeError,
    NotArmedError,
    PiperBridgeAdapter,
    StaleObservationError,
)


class FakeTopic:
    def __init__(self, ros, name, message_type):
        self.ros = ros
        self.name = name
        self.message_type = message_type
        self.callback = None
        self.advertised = False
        self.unsubscribed = False
        self.published = []

    def subscribe(self, callback):
        self.callback = callback

    def unsubscribe(self):
        self.unsubscribed = True
        self.callback = None

    def advertise(self):
        self.advertised = True

    def unadvertise(self):
        self.advertised = False

    def publish(self, message):
        self.published.append(dict(message))

    # test helper
    def deliver(self, message):
        assert self.callback is not None, f"no subscription on {self.name}"
        self.callback(message)


class FakeService:
    responses = {}

    def __init__(self, ros, name, service_type):
        self.ros = ros
        self.name = name
        self.service_type = service_type
        self.calls = []

    def call(self, request, timeout=None):
        self.calls.append(dict(request))
        response = FakeService.responses.get(
            request["operation"], {"accepted": True, "session_id": "sess-1", "reason": ""}
        )
        return dict(response)


class FakeRos:
    def __init__(self):
        self.handlers = {}
        self.is_connected = False
        self.run_calls = 0

    def on(self, event, callback):
        self.handlers.setdefault(event, []).append(callback)

    def run(self, timeout=None):
        self.run_calls += 1
        self.is_connected = True

    def close(self, timeout=None):
        self.is_connected = False
        self.emit("close")

    def emit(self, event, payload=None):
        for callback in self.handlers.get(event, []):
            callback(payload)


@pytest.fixture()
def rig():
    now = [1_700_000_000.0]  # mutable fake wall clock (seconds)
    fake_ros = FakeRos()
    topics = []
    services = []
    FakeService.responses = {}

    def topic_factory(ros, name, message_type):
        topic = FakeTopic(ros, name, message_type)
        topics.append(topic)
        return topic

    def service_factory(ros, name, service_type):
        service = FakeService(ros, name, service_type)
        services.append(service)
        return service

    client = PiperBridgeAdapter(
        ros_client=fake_ros,
        topic_factory=topic_factory,
        service_factory=service_factory,
        clock=lambda: now[0],
    )
    return client, fake_ros, topics, services, now


def _obs_message(stamp_s: float, joint1: float = 0.11, shuffle: bool = False) -> dict:
    names = list(adapter.JOINT_NAMES)
    positions = [joint1, 0.1, -0.1, 0.2, -0.2, 0.05, 0.04]
    if shuffle:  # joint order on the wire must not matter; mapping is by name
        order = [3, 0, 6, 1, 5, 2, 4]
        names = [names[i] for i in order]
        positions = [positions[i] for i in order]
    sec = int(stamp_s)
    return {
        "header": {"stamp": {"sec": sec, "nanosec": int((stamp_s - sec) * 1e9)}, "frame_id": "base"},
        "name": names,
        "position": positions,
        "velocity": [],
        "effort": [],
    }


def _topic(rig_topics, name):
    matches = [t for t in rig_topics if t.name == name]
    assert matches, f"topic {name} was never created"
    return matches[-1]


# ---------------------------------------------------------------------------
# connect() is read-only


def test_connect_is_read_only(rig):
    client, fake_ros, topics, services, _ = rig
    client.connect()
    assert fake_ros.run_calls == 1
    assert client.is_connected
    # only the two read subscriptions exist; no command topic, no service
    assert sorted(t.name for t in topics) == [EVENTS_TOPIC, OBSERVATION_TOPIC]
    assert all(not t.advertised for t in topics)
    assert all(not t.published for t in topics)
    assert services == []
    assert not client.is_armed


def test_close_before_connect_is_safe(rig):
    client, *_ = rig
    client.close()
    assert not client.is_connected


# ---------------------------------------------------------------------------
# timestamp freshness checks


def test_fresh_observation_accepted(rig):
    client, _, topics, _, now = rig
    client.connect()
    _topic(topics, OBSERVATION_TOPIC).deliver(_obs_message(now[0] - 0.05, shuffle=True))
    obs = client.latest_observation()
    assert obs.vector.shape == (7,)
    assert obs.vector.dtype.name == "float32"
    assert obs.vector[0] == pytest.approx(0.11, abs=1e-6)  # mapped by joint name
    assert obs.vector[-1] == pytest.approx(0.04, abs=1e-6)
    assert 0.0 <= obs.age_s <= 0.2


def test_stale_observation_rejected(rig):
    client, _, topics, _, now = rig
    client.connect()
    _topic(topics, OBSERVATION_TOPIC).deliver(_obs_message(now[0] - 0.5))
    with pytest.raises(StaleObservationError, match="stale"):
        client.latest_observation()


def test_boundary_age_accepted(rig):
    # float64 cannot represent an exact 200.0 ms delta at 1.7e9 s magnitude,
    # so use 199.9 ms here; the exact 200 ms boundary is covered in integer
    # nanoseconds by test_data.test_convert_stale_frames_dropped (age 200 ms
    # kept, 250 ms dropped).
    client, _, topics, _, now = rig
    client.connect()
    _topic(topics, OBSERVATION_TOPIC).deliver(_obs_message(now[0] - 0.1999))
    obs = client.latest_observation()
    assert obs.age_s == pytest.approx(0.1999, abs=1e-3)


def test_just_over_boundary_rejected(rig):
    client, _, topics, _, now = rig
    client.connect()
    _topic(topics, OBSERVATION_TOPIC).deliver(_obs_message(now[0] - 0.2001))
    with pytest.raises(StaleObservationError, match="stale"):
        client.latest_observation()


def test_future_observation_rejected(rig):
    client, _, topics, _, now = rig
    client.connect()
    _topic(topics, OBSERVATION_TOPIC).deliver(_obs_message(now[0] + 1.0))
    with pytest.raises(StaleObservationError, match="future"):
        client.latest_observation()


def test_missing_stamp_dropped_never_wallclock_substituted(rig):
    client, _, topics, _, now = rig
    client.connect()
    topic = _topic(topics, OBSERVATION_TOPIC)
    message = _obs_message(now[0])
    message["header"]["stamp"] = {"sec": 0, "nanosec": 0}
    topic.deliver(message)
    assert client.dropped_observations == 1
    with pytest.raises(StaleObservationError, match="no observation"):
        client.latest_observation()


# ---------------------------------------------------------------------------
# explicit acquire / arm only


def test_commands_require_explicit_arm(rig):
    client, _, topics, services, _ = rig
    client.connect()
    with pytest.raises(NotArmedError):
        client.send_command([0.0] * 7)
    # acquire alone must not arm: acquiring is not arming
    response = client.acquire()
    assert response["accepted"] is True
    assert client.session_id == "sess-1"
    assert not client.is_armed
    with pytest.raises(NotArmedError):
        client.send_command([0.0] * 7)
    # explicit arm enables commands
    response = client.arm()
    assert response["accepted"] is True
    assert client.is_armed
    assert [s.name for s in services] == [CONTROL_SERVICE]


def test_control_request_payloads(rig):
    client, _, _, services, _ = rig
    client.connect()
    client.acquire()
    client.arm()
    client.release()
    ops = [call["operation"] for call in services[0].calls]
    assert ops == ["acquire", "arm", "release"]
    assert services[0].calls[0]["owner"] == "policy"
    # the stored session id is forwarded once acquired
    assert services[0].calls[1]["session_id"] == "sess-1"
    assert not client.is_armed
    assert client.session_id is None


def test_rejected_arm_does_not_enable_commands(rig):
    client, _, topics, _, _ = rig
    FakeService.responses = {"arm": {"accepted": False, "session_id": "", "reason": "busy"}}
    client.connect()
    client.acquire()
    response = client.arm()
    assert response["accepted"] is False
    assert response["reason"] == "busy"
    assert not client.is_armed
    with pytest.raises(NotArmedError):
        client.send_command([0.0] * 7)


# ---------------------------------------------------------------------------
# command publishing: once, sequenced, never replayed


def test_send_command_shape_and_sequence(rig):
    client, _, topics, _, now = rig
    client.connect()
    client.acquire()
    client.arm()
    target = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.07]
    first = client.send_command(target, source="unit-test")
    second = client.send_command(target)
    assert first["sequence"] == 0 and second["sequence"] == 1
    command_topic = _topic(topics, COMMAND_TOPIC)
    assert command_topic.advertised
    assert len(command_topic.published) == 2  # exactly one publish per call
    message = command_topic.published[0]
    assert message["session_id"] == "sess-1"
    assert message["sequence"] == 0
    assert message["source"] == "unit-test"
    assert message["target"]["name"] == adapter.JOINT_NAMES
    assert message["target"]["position"] == pytest.approx(target)
    stamp = message["header"]["stamp"]
    assert stamp["sec"] > 0  # stamped at send time (command freshness contract)


def test_send_command_validates_target(rig):
    client, *_ = rig
    client.connect()
    client.acquire()
    client.arm()
    with pytest.raises(ValueError):
        client.send_command([0.0] * 6)
    with pytest.raises(ValueError):
        client.send_command([float("nan")] * 7)


def test_no_replay_after_disconnect(rig):
    client, fake_ros, topics, services, _ = rig
    client.connect()
    client.acquire()
    client.arm()
    client.send_command([0.0] * 7)
    published_before = sum(len(t.published) for t in topics)

    fake_ros.close()  # connection drops
    assert not client.is_connected
    assert not client.is_armed

    # reconnecting does not resurrect anything: commands need explicit re-arm
    client.connect()
    assert sum(len(t.published) for t in topics) == published_before  # nothing resent
    with pytest.raises(NotArmedError):
        client.send_command([0.0] * 7)

    # after explicit re-acquire/re-arm the sequence restarts at 0
    client.acquire()
    client.arm()
    result = client.send_command([0.0] * 7)
    assert result["sequence"] == 0
    assert sum(len(t.published) for t in topics) == published_before + 1


def test_control_calls_require_connection(rig):
    client, *_ = rig
    with pytest.raises(BridgeError):
        client.acquire()
    with pytest.raises(BridgeError):
        client.arm()


# ---------------------------------------------------------------------------
# events


def test_events_are_recorded(rig):
    client, _, topics, _, _ = rig
    client.connect()
    _topic(topics, EVENTS_TOPIC).deliver({"data": json.dumps({"event": "episode_start", "episode_index": 3})})
    assert client.events == [{"event": "episode_start", "episode_index": 3}]
