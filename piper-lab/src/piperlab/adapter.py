"""ROS bridge adapter for Piper Lab (rosbridge ``ws://127.0.0.1:9090``).

Safety properties (see AGENTS.md):

- ``connect()`` is strictly read-only: it subscribes to observation/event
  topics and nothing else. It never advertises the command topic and never
  calls the control service. LeRobot-side users get data only.
- ``acquire``/``arm``/``release``/``reset`` happen only when the caller
  explicitly invokes them. Acquiring is not arming: an accepted ``acquire``
  does not enable commands; only an accepted ``arm`` does.
- Observation freshness is checked against the 200 ms contract window using
  the message ``header.stamp``. Stale, future-dated, or stamp-less messages
  are rejected; the wall clock is never substituted for a missing stamp.
- No auto-reconnect and no action replay: when the connection closes, the
  armed state, session and command sequence are dropped. Nothing is buffered
  or resent, and reconnecting requires a fresh explicit ``connect()``,
  ``acquire()`` and ``arm()``.
"""

from __future__ import annotations

import json
import math
import time
from collections import deque
from dataclasses import dataclass

import numpy as np

OBSERVATION_TOPIC = "/lab/observation"
EVENTS_TOPIC = "/lab/events"
COMMAND_TOPIC = "/lab/command"
CONTROL_SERVICE = "/lab/control"

JOINT_STATE_TYPE = "sensor_msgs/JointState"
STRING_TYPE = "std_msgs/String"
COMMAND_TYPE = "piperlab_msgs/Command"
CONTROL_TYPE = "piperlab_msgs/Control"

JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "gripper"]
STATE_DIM = 7
DEFAULT_MAX_AGE_S = 0.2  # 200 ms observation/command freshness contract


class BridgeError(RuntimeError):
    """Connection or protocol failure against the rosbridge server."""


class StaleObservationError(BridgeError):
    """Latest observation is stale, future-dated, or missing its stamp."""


class NotArmedError(BridgeError):
    """A command was attempted without an accepted explicit arm."""


class ControlRejectedError(BridgeError):
    """The control service answered but did not accept the operation."""


@dataclass(frozen=True)
class Observation:
    """One fresh /lab/observation sample."""

    stamp_ns: int
    age_s: float
    names: tuple
    positions: tuple
    vector: np.ndarray  # (7,) float32 in contract order
    raw: dict


def _stamp_ns(message: dict) -> int:
    header = message.get("header") or {}
    stamp = header.get("stamp") or {}
    sec = int(stamp.get("sec", stamp.get("secs", 0)) or 0)
    nsec = int(stamp.get("nanosec", stamp.get("nsec", stamp.get("nsecs", 0))) or 0)
    return sec * 1_000_000_000 + nsec


def _joint_vector(names, positions) -> np.ndarray:
    if len(names) != len(positions) or len(set(names)) != len(names):
        raise BridgeError('Invalid or duplicate joint mapping')
    mapping = {str(name): float(pos) for name, pos in zip(names, positions)}
    missing = [name for name in JOINT_NAMES if name not in mapping]
    if missing:
        raise BridgeError(f"observation is missing joints {missing}")
    return np.array([mapping[name] for name in JOINT_NAMES], dtype=np.float32)


class PiperBridgeAdapter:
    """Thin, safe client for the Piper Lab rosbridge contract."""

    def __init__(
        self,
        url: str = "ws://127.0.0.1:9090",
        *,
        owner: str = "policy",
        max_message_age_s: float = DEFAULT_MAX_AGE_S,
        future_tolerance_s: float = 0.05,
        timeout_s: float = 10.0,
        ros_client=None,
        topic_factory=None,
        service_factory=None,
        clock=time.time,
    ):
        if not url.startswith("ws://"):
            raise ValueError(f"expected a ws:// rosbridge URL, got {url!r}")
        host_port = url[len("ws://") :].split("/", 1)[0]
        host, _, port = host_port.partition(":")
        self.url = url
        self.host = host
        self.port = int(port or 9090)
        self.owner = owner
        self.max_message_age_s = float(max_message_age_s)
        self.future_tolerance_s = float(future_tolerance_s)
        self.timeout_s = float(timeout_s)
        self._clock = clock

        self._ros_client = ros_client
        self._topic_factory = topic_factory
        self._service_factory = service_factory

        self._ros = None
        self._connected = False
        self._armed = False
        self._session_id = None
        self._sequence = 0
        self._observation_topic = None
        self._events_topic = None
        self._command_topic = None
        self._control_service = None
        self._latest_observation = None  # (receipt_s, stamp_ns, message)
        self._dropped_observations = 0
        self._events = deque(maxlen=100)

    # -- transport ----------------------------------------------------------

    def _make_ros(self):
        if self._ros_client is not None:
            return self._ros_client
        import roslibpy  # noqa: PLC0415 - lazy so tests can run without a bridge

        return roslibpy.Ros(host=self.host, port=self.port)

    def _make_topic(self, name, message_type):
        if self._topic_factory is not None:
            return self._topic_factory(self._ros, name, message_type)
        import roslibpy  # noqa: PLC0415

        return roslibpy.Topic(self._ros, name, message_type, reconnect_on_close=False)

    def _make_service(self, name, service_type):
        if self._service_factory is not None:
            return self._service_factory(self._ros, name, service_type)
        import roslibpy  # noqa: PLC0415

        return roslibpy.Service(self._ros, name, service_type, reconnect_on_close=False)

    # -- lifecycle ----------------------------------------------------------

    def connect(self) -> None:
        """Connect read-only: subscribe to observation/event topics only.

        No command topic is advertised and no control service is called here;
        arming requires explicit caller requests after connecting.
        """
        if self._connected:
            raise BridgeError("already connected; close() before reconnecting")
        self._ros = self._make_ros()
        self._ros.on("close", self._on_transport_close)
        self._ros.on("error", self._on_transport_close)
        self._ros.run(timeout=self.timeout_s)
        if not self._ros.is_connected:
            raise BridgeError(f"could not connect to rosbridge at {self.url}")
        self._observation_topic = self._make_topic(OBSERVATION_TOPIC, JOINT_STATE_TYPE)
        self._observation_topic.subscribe(self._on_observation)
        self._events_topic = self._make_topic(EVENTS_TOPIC, STRING_TYPE)
        self._events_topic.subscribe(self._on_event)
        self._connected = True

    def close(self) -> None:
        if self.is_connected and self._session_id:
            try:
                self.release()
            except Exception:
                pass  # The independent runtime watchdog handles lost transport.
        if self._observation_topic is not None:
            try:
                self._observation_topic.unsubscribe()
            except Exception:  # noqa: BLE001 - closing must not raise
                pass
        if self._events_topic is not None:
            try:
                self._events_topic.unsubscribe()
            except Exception:  # noqa: BLE001
                pass
        self._drop_control_state()
        if self._ros is not None:
            try:
                self._ros.close()
            except Exception:  # noqa: BLE001
                pass
        self._connected = False

    def _on_transport_close(self, _event=None) -> None:
        # no auto-reconnect, no replay: drop everything that could resend
        self._drop_control_state()
        self._connected = False

    def _drop_control_state(self) -> None:
        self._armed = False
        self._session_id = None
        self._sequence = 0
        self._command_topic = None
        self._control_service = None
        self._latest_observation = None

    @property
    def is_connected(self) -> bool:
        ros_live = bool(self._ros is not None and getattr(self._ros, "is_connected", False))
        return self._connected and ros_live

    @property
    def is_armed(self) -> bool:
        return self._armed and self.is_connected

    @property
    def session_id(self):
        return self._session_id

    @property
    def events(self) -> list:
        return list(self._events)

    @property
    def dropped_observations(self) -> int:
        return self._dropped_observations

    # -- subscriptions --------------------------------------------------------

    def _on_observation(self, message: dict) -> None:
        receipt = self._clock()
        stamp = _stamp_ns(message)
        if stamp == 0:
            # no wall-clock substitution: a stamp-less observation is unusable
            self._dropped_observations += 1
            return
        self._latest_observation = (receipt, stamp, message)

    def _on_event(self, message: dict) -> None:
        data = message.get("data")
        try:
            event = json.loads(data) if isinstance(data, str) else data
        except ValueError:
            event = {"raw": data}
        if isinstance(event, dict):
            self._events.append(event)
            if event.get('kind') == 'stop':
                self._drop_control_state()

    def latest_observation(self, *, max_age_s: float | None = None) -> Observation:
        """Return the newest observation if it passes the freshness checks.

        Raises StaleObservationError when the newest message is older than the
        200 ms contract window or is dated in the future.
        """
        if not self.is_connected:
            raise BridgeError("not connected")
        if self._latest_observation is None:
            raise StaleObservationError("no observation received yet")
        max_age = self.max_message_age_s if max_age_s is None else float(max_age_s)
        _, stamp, message = self._latest_observation
        age = self._clock() - stamp / 1e9
        if age > max_age:
            raise StaleObservationError(
                f"observation is stale: age {age * 1000:.0f} ms exceeds {max_age * 1000:.0f} ms"
            )
        if age < -self.future_tolerance_s:
            raise StaleObservationError(
                f"observation is dated in the future ({-age * 1000:.0f} ms ahead); refusing it"
            )
        names = tuple(message.get("name", ()))
        positions = tuple(message.get("position", ()))
        return Observation(
            stamp_ns=stamp,
            age_s=age,
            names=names,
            positions=positions,
            vector=_joint_vector(names, positions),
            raw=message,
        )

    def wait_observation(self, timeout_s: float = 5.0, *, max_age_s: float | None = None) -> Observation:
        """Poll until a fresh observation is available or the timeout expires."""
        deadline = self._clock() + timeout_s
        last_error = None
        while self._clock() < deadline:
            try:
                return self.latest_observation(max_age_s=max_age_s)
            except StaleObservationError as exc:
                last_error = exc
                time.sleep(0.005)
        raise StaleObservationError(
            f"no fresh observation within {timeout_s:.1f} s"
        ) from last_error

    # -- control service (explicit caller requests only) ----------------------

    def _control(self, operation: str, owner: str | None, session_id) -> dict:
        if not self.is_connected:
            raise BridgeError(f"cannot {operation}: not connected")
        if self._control_service is None:
            self._control_service = self._make_service(CONTROL_SERVICE, CONTROL_TYPE)
        request = {
            "operation": operation,
            "owner": owner or self.owner,
            "session_id": session_id or self._session_id or "",
        }
        result = self._control_service.call(request, timeout=self.timeout_s)
        response = {
            "accepted": bool(result.get("accepted", False)),
            "session_id": str(result.get("session_id", "")),
            "reason": str(result.get("reason", "")),
        }
        if response["accepted"]:
            if operation == "acquire":
                self._session_id = response["session_id"] or request["session_id"]
            elif operation == "arm":
                # acquiring is not arming: only here does the session become armed
                self._armed = True
                if response["session_id"]:
                    self._session_id = response["session_id"]
            elif operation in ("release", "reset"):
                self._armed = False
                self._session_id = None
                self._sequence = 0
                self._command_topic = None
        return response

    def acquire(self, owner: str | None = None, session_id=None) -> dict:
        """Explicitly request control ownership. Does NOT arm the bridge."""
        return self._control("acquire", owner, session_id)

    def arm(self, owner: str | None = None, session_id=None) -> dict:
        """Explicitly request arming. Only an accepted arm enables send_command."""
        return self._control("arm", owner, session_id)

    def release(self, owner: str | None = None, session_id=None) -> dict:
        """Explicitly release ownership and disarm."""
        return self._control("release", owner, session_id)

    def reset(self, owner: str | None = None, session_id=None) -> dict:
        """Explicitly request a bridge reset (also disarms locally)."""
        return self._control("reset", owner, session_id)

    # -- commands ---------------------------------------------------------------

    def send_command(self, target, *, source: str | None = None, session_id=None) -> dict:
        """Publish one /lab/command for an armed session. Never replayed.

        ``target`` is the ordered 7-value contract vector (joint1..joint6 in
        radians, gripper opening in metres). The command is stamped at send
        time and published exactly once; there is no buffering and no resend
        after a disconnect.
        """
        if not self.is_connected:
            raise BridgeError("cannot send a command: not connected")
        if not self._armed:
            raise NotArmedError(
                "refusing to send a command: the bridge was not explicitly armed by this adapter"
            )
        values = [float(v) for v in target]
        if len(values) != STATE_DIM or not all(math.isfinite(v) for v in values):
            raise ValueError(f"target must be {STATE_DIM} finite floats, got {target!r}")
        session = session_id or self._session_id or ""
        now = self._clock()
        sec = int(now)
        nanosec = int((now - sec) * 1e9)
        if self._command_topic is None:
            self._command_topic = self._make_topic(COMMAND_TOPIC, COMMAND_TYPE)
            self._command_topic.advertise()
        message = {
            "header": {"stamp": {"sec": sec, "nanosec": nanosec}, "frame_id": "piperlab"},
            "session_id": session,
            "sequence": self._sequence,
            "source": source or self.owner,
            "target": {
                "header": {"stamp": {"sec": sec, "nanosec": nanosec}, "frame_id": ""},
                "name": list(JOINT_NAMES),
                "position": values,
                "velocity": [],
                "effort": [],
            },
        }
        self._command_topic.publish(message)
        self._sequence += 1
        return {
            "published": True,
            "topic": COMMAND_TOPIC,
            "session_id": session,
            "sequence": self._sequence - 1,
            "stamp_ns": sec * 1_000_000_000 + nanosec,
        }
