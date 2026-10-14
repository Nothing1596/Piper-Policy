"""Managed runtime: exclusive startup, verified identity, graceful release, SSH.

All processes, HTTP calls and SSH calls are faked; no hardware, no network, and
no real executor is ever started.
"""
import asyncio
import io
import itertools
import json
import logging
import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest

from piperx_middleware import managed_runtime, profiles, ssh_runtime
from piperx_middleware.interaction_types import RuntimeConnection
from piperx_middleware.managed_runtime import RuntimeManager
from piperx_middleware.models import DomainError
from piperx_middleware.process_lock import ProcessLock


# ---------------------------------------------------------------------------
# fake executor harness
# ---------------------------------------------------------------------------

class FakeServer:
    _counter = itertools.count(1)

    def __init__(self, harness, *, url, instance_id, profile_id, mode, backend, pid=None):
        self.harness = harness
        self.url = url.rstrip("/")
        self.instance_id = instance_id
        self.profile_id = profile_id
        self.mode = mode
        self.backend = backend
        self.pid = pid if pid is not None else 900000 + next(FakeServer._counter)
        self.active_job_id = None
        self.state_override = None
        self.health_override = {}
        self.shutdown_error = None
        self.shutdown_calls = []
        self.state_calls = 0
        self.requests = []
        self.alive = True

    @classmethod
    def from_profile(cls, harness, profile):
        port = 20000 + next(cls._counter)
        return cls(harness, url=f"http://127.0.0.1:{port}",
                   instance_id=f"instance-{next(cls._counter):04d}",
                   profile_id=profile.profile_id, mode=profile.mode, backend=profile.backend)

    def record(self):
        return {"url": self.url, "instance_id": self.instance_id, "profile_id": self.profile_id,
                "mode": self.mode, "backend": self.backend, "pid": self.pid}

    def health(self):
        payload = {"service": "piperx-middleware", "api_version": "1", "process_id": self.pid,
                   "instance_id": self.instance_id, "profile_id": self.profile_id,
                   "mode": self.mode, "backend": self.backend}
        payload.update(self.health_override)
        return payload

    def state(self):
        self.state_calls += 1
        if self.state_override is not None:
            return self.state_override
        # Mirrors the root /v1/state shape: the instance identity lives inside
        # observation_identity and active_job_id is present (None when idle).
        return {"active_job_id": self.active_job_id, "connection_epoch": 1,
                "observation_identity": {"instance_id": self.instance_id, "connection_epoch": 1}}


class FakeProcess:
    def __init__(self, harness, argv, **kwargs):
        self.argv = list(argv)
        self.kwargs = kwargs
        self.returncode = None
        self.terminated = False
        self.killed = False
        self.reaped = False
        self.server = None
        harness.spawned.append(self)
        if harness.behavior.get("exit_immediately"):
            self.returncode = harness.behavior.get("exit_code", 2)
            return
        root = Path(self.argv[self.argv.index("--root") + 1])
        if not harness.behavior.get("publish", True):
            return
        profile = profiles.load_profile(root.parents[2], root.parent.name, root.name)
        self.server = FakeServer.from_profile(harness, profile)
        harness.servers[self.server.url] = self.server
        payload = self.server.record()
        if harness.behavior.get("record_profile_id"):
            payload["profile_id"] = harness.behavior["record_profile_id"]
        if harness.behavior.get("record_backend"):
            payload["backend"] = harness.behavior["record_backend"]
        (root / "runtime.json").write_text(json.dumps(payload), encoding="utf-8")

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        self.reaped = True
        if self.returncode is None:
            self.returncode = 0
        return self.returncode

    def terminate(self):
        self.terminated = True
        self.returncode = -15

    def kill(self):
        self.killed = True
        self.returncode = -9


class FakeTunnel:
    def __init__(self):
        self.closed = False
        self.terminated = False

    def poll(self):
        return 0 if self.closed else None

    def terminate(self):
        self.terminated = True
        self.closed = True

    def wait(self, timeout=None):
        return 0

    def kill(self):
        self.closed = True


class Harness:
    def __init__(self):
        self.servers = {}
        self.spawned = []
        self.behavior = {}

    def install(self, monkeypatch):
        harness = self

        def popen(argv, **kwargs):
            return FakeProcess(harness, argv, **kwargs)

        async def request_json(method, url, token_file, path, body=None, *, timeout_s=5.0):
            server = harness.servers.get(url.rstrip("/"))
            if server is None:
                raise DomainError("runtime_unreachable", "fake executor is not listening", 503)
            server.requests.append((method, path, body))
            if path == "/health":
                return server.health()
            if path == "/v1/state":
                return server.state()
            if path == "/v1/shutdown":
                server.shutdown_calls.append(body)
                if server.shutdown_error is not None:
                    return {"error": dict(server.shutdown_error)}
                server.alive = False
                return {"status": "resources_released", "process_id": server.pid,
                        "instance_id": server.instance_id, "robot_stop_command_sent": False}
            raise AssertionError(f"unexpected fake path {path}")

        monkeypatch.setattr(managed_runtime.subprocess, "Popen", popen)
        monkeypatch.setattr(managed_runtime, "_request_json", request_json)
        return harness


def manager(tmp_path, mode="simulation", **kwargs):
    options = {"startup_timeout_s": 2.0, "shutdown_grace_s": 0.2, "drain_timeout_s": 0.3,
               "poll_interval_s": 0.005, "lock_timeout_s": 2.0}
    options.update(kwargs)
    return RuntimeManager(tmp_path, mode, **options)


def run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# local startup and reuse
# ---------------------------------------------------------------------------

def test_ensure_starts_one_verified_executor(monkeypatch, tmp_path):
    harness = Harness().install(monkeypatch)
    runtime = manager(tmp_path)
    connection = run(runtime.ensure())

    assert connection.owned is True
    assert connection.mode == "simulation"
    assert connection.target == "local"
    assert connection.profile_id == profiles.profile_id("simulation", "local")
    assert connection.instance_id == harness.spawned[0].server.instance_id
    assert connection.url == harness.spawned[0].server.url
    assert len(harness.spawned) == 1

    argv = harness.spawned[0].argv
    assert argv[:3] == [sys.executable, "-m", "piperx_middleware.cli"]
    assert argv[3:5] == ["--root", str(runtime.profile.root)]
    assert argv[5:] == ["serve", "--managed"]
    assert "model.token" not in " ".join(argv), "credentials must never appear on a command line"

    assert connection.model_token_file == runtime.profile.model_token_file
    assert len(connection.model_token_file.read_text().strip()) >= 32
    record = json.loads(runtime.profile.runtime_path.read_text())
    assert record["pid"] == harness.spawned[0].server.pid
    # Startup never opens CAN: the fake harness raises on any other endpoint.
    assert all(path != "/v1/connect" for _method, path, _body in harness.spawned[0].server.requests)
    assert run(runtime.shutdown())["status"] == "released"


def test_ensure_is_idempotent_and_keeps_ownership(monkeypatch, tmp_path):
    harness = Harness().install(monkeypatch)
    runtime = manager(tmp_path)
    first = run(runtime.ensure())
    second = run(runtime.ensure())
    assert first.instance_id == second.instance_id
    assert second.owned is True
    assert len(harness.spawned) == 1
    assert run(runtime.shutdown())["status"] == "released"


def test_second_frontend_reuses_shared_executor(monkeypatch, tmp_path):
    harness = Harness().install(monkeypatch)
    owner = manager(tmp_path)
    shared = manager(tmp_path)
    owned = run(owner.ensure())
    reused = run(shared.ensure())
    assert reused.instance_id == owned.instance_id
    assert reused.owned is False
    assert len(harness.spawned) == 1
    # A shared instance is detached, never shut down.
    assert run(shared.release(reused))["status"] == "detached"
    assert harness.spawned[0].server.shutdown_calls == []
    assert harness.spawned[0].poll() is None
    assert run(owner.release(owned))["status"] == "released"


def test_concurrent_ensure_starts_exactly_one_process(monkeypatch, tmp_path):
    harness = Harness().install(monkeypatch)

    async def race():
        first = manager(tmp_path)
        second = manager(tmp_path)
        return await asyncio.gather(first.ensure(), second.ensure()), first, second

    (one, two), first, second = run(race())
    assert one.instance_id == two.instance_id
    assert len(harness.spawned) == 1
    assert {one.owned, two.owned} == {True, False}
    owner, owner_connection, shared, shared_connection = (
        (first, one, second, two) if one.owned else (second, two, first, one))
    assert run(shared.release(shared_connection))["status"] == "detached"
    assert run(owner.release(owner_connection))["status"] == "released"


def test_startup_lock_is_exclusive_across_handles(tmp_path):
    profile = profiles.ensure_profile(tmp_path, "simulation")
    first = ProcessLock(profile.startup_lock_path)
    try:
        with pytest.raises(RuntimeError):
            ProcessLock(profile.startup_lock_path)
    finally:
        first.close()
    second = ProcessLock(profile.startup_lock_path)
    second.close()


def test_startup_lock_timeout_reports_busy(monkeypatch, tmp_path):
    Harness().install(monkeypatch)
    profile = profiles.ensure_profile(tmp_path, "simulation")
    held = ProcessLock(profile.startup_lock_path)
    try:
        with pytest.raises(DomainError) as exc:
            run(manager(tmp_path, lock_timeout_s=0.1).ensure())
        assert exc.value.code == "runtime_busy"
    finally:
        held.close()


def test_startup_failure_reaps_child_and_reports(monkeypatch, tmp_path):
    harness = Harness().install(monkeypatch)
    harness.behavior["exit_immediately"] = True
    harness.behavior["exit_code"] = 3
    with pytest.raises(DomainError) as exc:
        run(manager(tmp_path).ensure())
    assert exc.value.code == "runtime_start_failed"
    assert "3" in exc.value.message
    assert harness.spawned[0].poll() == 3
    assert harness.spawned[0].reaped is True
    assert not (profiles.profile_root(tmp_path, "simulation") / "runtime.json").exists()


def test_startup_timeout_terminates_unpublished_child(monkeypatch, tmp_path):
    harness = Harness().install(monkeypatch)
    harness.behavior["publish"] = False
    with pytest.raises(DomainError) as exc:
        run(manager(tmp_path, startup_timeout_s=0.1).ensure())
    assert exc.value.code == "runtime_start_failed"
    assert harness.spawned[0].terminated is True


def test_real_mode_profile_is_physical(monkeypatch, tmp_path):
    harness = Harness().install(monkeypatch)
    runtime = manager(tmp_path, "real")
    connection = run(runtime.ensure())
    assert connection.profile_id == profiles.profile_id("real", "local")
    assert harness.spawned[0].server.backend == "agx"
    assert run(runtime.shutdown())["status"] == "released"


# ---------------------------------------------------------------------------
# identity verification and stale state
# ---------------------------------------------------------------------------

def test_stale_record_with_dead_pid_is_replaced(monkeypatch, tmp_path):
    harness = Harness().install(monkeypatch)
    profile = profiles.ensure_profile(tmp_path, "simulation")
    stale = FakeServer(harness, url="http://127.0.0.1:29999", instance_id="stale-instance",
                       profile_id=profile.profile_id, mode="simulation", backend="mujoco", pid=999999)
    profile.runtime_path.write_text(json.dumps(stale.record()))
    connection = run(manager(tmp_path).ensure())
    assert connection.instance_id != "stale-instance"
    assert len(harness.spawned) == 1
    assert json.loads(profile.runtime_path.read_text())["instance_id"] == connection.instance_id
    assert run(manager(tmp_path).release(connection))["status"] == "released"


def test_corrupt_record_is_refused_and_no_duplicate_started(monkeypatch, tmp_path):
    harness = Harness().install(monkeypatch)
    profile = profiles.ensure_profile(tmp_path, "simulation")
    profile.runtime_path.write_text("{not json", encoding="utf-8")
    with pytest.raises(DomainError) as exc:
        run(manager(tmp_path).ensure())
    assert exc.value.code == "runtime_state_corrupt"
    assert harness.spawned == []


def test_record_with_foreign_profile_identity_is_refused(monkeypatch, tmp_path):
    harness = Harness().install(monkeypatch)
    profile = profiles.ensure_profile(tmp_path, "simulation")
    record = {"url": "http://127.0.0.1:29999", "instance_id": "x", "profile_id": "deadbeefdeadbeef",
              "mode": "simulation", "backend": "mujoco", "pid": 999999}
    profile.runtime_path.write_text(json.dumps(record))
    with pytest.raises(DomainError) as exc:
        run(manager(tmp_path).ensure())
    assert exc.value.code == "runtime_identity_mismatch"
    assert harness.spawned == []


def test_health_instance_mismatch_is_refused(monkeypatch, tmp_path):
    harness = Harness().install(monkeypatch)
    owner = manager(tmp_path)
    connection = run(owner.ensure())
    server = harness.servers[connection.url]
    server.health_override = {"instance_id": "someone-else"}
    with pytest.raises(DomainError) as exc:
        run(manager(tmp_path).ensure())
    assert exc.value.code == "runtime_identity_mismatch"
    assert len(harness.spawned) == 1, "a mismatched live executor must never be duplicated"
    server.health_override = {}
    assert run(owner.release(connection))["status"] == "released"


def test_health_backend_mismatch_is_refused(monkeypatch, tmp_path):
    harness = Harness().install(monkeypatch)
    runtime = manager(tmp_path)
    connection = run(runtime.ensure())
    harness.servers[connection.url].health_override = {"backend": "agx"}
    with pytest.raises(DomainError) as exc:
        run(manager(tmp_path).ensure())
    assert exc.value.code == "runtime_identity_mismatch"
    assert len(harness.spawned) == 1


@pytest.mark.parametrize("key", ["instance_id", "process_id", "profile_id", "mode", "backend"])
def test_reuse_refuses_health_missing_any_identity_field(monkeypatch, tmp_path, key):
    """A half-upgraded or foreign service must not be reused just because the
    field it omits used to be optional."""
    harness = Harness().install(monkeypatch)
    profile = profiles.ensure_profile(tmp_path, "simulation")
    url = "http://127.0.0.1:28111"
    server = FakeServer(harness, url=url, instance_id="reuse-instance", profile_id=profile.profile_id,
                        mode="simulation", backend="mujoco", pid=900123)
    server.health_override = {key: None}
    harness.servers[url] = server
    profile.runtime_path.write_text(json.dumps(server.record()), encoding="utf-8")
    with pytest.raises(DomainError) as exc:
        run(manager(tmp_path).ensure())
    assert exc.value.code == "runtime_identity_mismatch"
    assert harness.spawned == [], "an identity-less listener must never be reused or duplicated"


@pytest.mark.parametrize("key,value", [
    ("instance_id", "someone-else"), ("profile_id", "0" * 16), ("mode", "real"), ("backend", "agx"),
])
def test_reuse_refuses_health_foreign_identity(monkeypatch, tmp_path, key, value):
    harness = Harness().install(monkeypatch)
    profile = profiles.ensure_profile(tmp_path, "simulation")
    url = "http://127.0.0.1:28112"
    server = FakeServer(harness, url=url, instance_id="expected-instance", profile_id=profile.profile_id,
                        mode="simulation", backend="mujoco", pid=900124)
    server.health_override = {key: value}
    harness.servers[url] = server
    profile.runtime_path.write_text(json.dumps(server.record()), encoding="utf-8")
    with pytest.raises(DomainError) as exc:
        run(manager(tmp_path).ensure())
    assert exc.value.code == "runtime_identity_mismatch"
    assert harness.spawned == []
    assert server.shutdown_calls == [], "a foreign service is never shut down"


def test_connection_healthy_requires_every_identity_field(monkeypatch, tmp_path):
    harness = Harness().install(monkeypatch)
    runtime = manager(tmp_path)
    connection = run(runtime.ensure())
    server = harness.servers[connection.url]
    assert run(runtime._connection_healthy(connection)) is True
    for key in ("instance_id", "profile_id", "mode", "backend"):
        server.health_override = {key: None}
        assert run(runtime._connection_healthy(connection)) is False, key
    server.health_override = {"service": "another-service"}
    assert run(runtime._connection_healthy(connection)) is False
    server.health_override = {}
    assert run(runtime._connection_healthy(connection)) is True
    assert run(runtime.release(connection))["status"] == "released"


def test_probe_health_rejects_a_foreign_service_payload(monkeypatch, tmp_path):
    harness = Harness().install(monkeypatch)
    profile = profiles.ensure_profile(tmp_path, "simulation")
    server = FakeServer(harness, url="http://127.0.0.1:28113", instance_id="instance-x",
                        profile_id=profile.profile_id, mode="simulation", backend="mujoco", pid=900125)
    harness.servers[server.url] = server
    token = profile.model_token_file
    assert run(managed_runtime.probe_health(server.url, token))["instance_id"] == "instance-x"
    for override in ({"service": "other-service"}, {"service": None}):
        server.health_override = override
        assert run(managed_runtime.probe_health(server.url, token)) is None


def test_live_pid_without_identity_is_not_duplicated(monkeypatch, tmp_path):
    harness = Harness().install(monkeypatch)
    profile = profiles.ensure_profile(tmp_path, "simulation")
    record = {"url": "http://127.0.0.1:29998", "instance_id": "live-but-silent",
              "profile_id": profile.profile_id, "mode": "simulation", "backend": "mujoco",
              "pid": os.getpid()}
    profile.runtime_path.write_text(json.dumps(record))
    with pytest.raises(DomainError) as exc:
        run(manager(tmp_path).ensure())
    assert exc.value.code == "runtime_unhealthy"
    assert harness.spawned == []


def test_unverified_listener_is_refused(monkeypatch, tmp_path):
    harness = Harness().install(monkeypatch)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        port = listener.getsockname()[1]
        profile = profiles.ensure_profile(tmp_path, "simulation")
        record = {"url": f"http://127.0.0.1:{port}", "instance_id": "foreign-listener",
                  "profile_id": profile.profile_id, "mode": "simulation", "backend": "mujoco",
                  "pid": os.getpid()}
        profile.runtime_path.write_text(json.dumps(record))
        with pytest.raises(DomainError) as exc:
            run(manager(tmp_path).ensure())
        assert exc.value.code == "runtime_identity_mismatch"
        assert harness.spawned == []


def test_record_publishing_a_non_loopback_url_is_refused(monkeypatch, tmp_path):
    harness = Harness().install(monkeypatch)
    profile = profiles.ensure_profile(tmp_path, "simulation")
    record = {"url": "http://10.0.0.5:8765", "instance_id": "remoteish",
              "profile_id": profile.profile_id, "mode": "simulation", "backend": "mujoco", "pid": 999999}
    profile.runtime_path.write_text(json.dumps(record))
    with pytest.raises(DomainError) as exc:
        run(manager(tmp_path).ensure())
    assert exc.value.code == "invalid_runtime_url"
    assert harness.spawned == []


# ---------------------------------------------------------------------------
# release: drain, detach, refuse
# ---------------------------------------------------------------------------

def test_release_drains_active_action_before_shutdown(monkeypatch, tmp_path):
    harness = Harness().install(monkeypatch)
    runtime = manager(tmp_path)
    connection = run(runtime.ensure())
    server = harness.servers[connection.url]
    server.active_job_id = "job-0001"
    steps = {"count": 0}

    async def finish_action():
        while server.state_calls < 3:
            await asyncio.sleep(0.005)
        server.active_job_id = None

    async def scenario():
        finisher = asyncio.create_task(finish_action())
        result = await runtime.release(connection)
        await finisher
        return result

    result = run(scenario())
    assert result["status"] == "released"
    assert result["drained"] is True
    assert server.state_calls >= 3, "release must poll the executor until the action completes"
    assert server.shutdown_calls == [{"expected_instance_id": connection.instance_id}]


def test_release_busy_never_shuts_down_or_kills(monkeypatch, tmp_path):
    harness = Harness().install(monkeypatch)
    runtime = manager(tmp_path, drain_timeout_s=0.05)
    connection = run(runtime.ensure())
    server = harness.servers[connection.url]
    server.active_job_id = "job-running"
    result = run(runtime.release(connection))
    assert result == {"status": "busy", "drained": False, "shutdown": False,
                      "instance_id": connection.instance_id, "active_job_id": "job-running"}
    assert server.shutdown_calls == []
    assert server.alive is True
    assert harness.spawned[0].terminated is False
    # Once the action completes the same connection can be released.
    server.active_job_id = None
    assert run(runtime.release(connection))["status"] == "released"


def test_release_without_shutdown_owned_only_detaches(monkeypatch, tmp_path):
    harness = Harness().install(monkeypatch)
    runtime = manager(tmp_path)
    connection = run(runtime.ensure())
    result = run(runtime.release(connection, shutdown_owned=False))
    assert result["status"] == "detached"
    assert harness.servers[connection.url].shutdown_calls == []
    assert harness.spawned[0].terminated is False
    # Ownership is gone: re-attaching reuses the same verified executor as shared.
    again = run(runtime.ensure())
    assert again.instance_id == connection.instance_id
    assert again.owned is False
    assert len(harness.spawned) == 1


def test_release_identity_mismatch_refuses_shutdown(monkeypatch, tmp_path):
    harness = Harness().install(monkeypatch)
    runtime = manager(tmp_path)
    connection = run(runtime.ensure())
    server = harness.servers[connection.url]
    server.health_override = {"instance_id": "different-instance"}
    with pytest.raises(DomainError) as exc:
        run(runtime.release(connection))
    assert exc.value.code == "instance_mismatch"
    assert server.shutdown_calls == []
    assert harness.spawned[0].terminated is False


def test_release_unreachable_executor_refuses_shutdown(monkeypatch, tmp_path):
    harness = Harness().install(monkeypatch)
    runtime = manager(tmp_path)
    connection = run(runtime.ensure())
    del harness.servers[connection.url]
    with pytest.raises(DomainError) as exc:
        run(runtime.release(connection))
    assert exc.value.code == "runtime_unreachable"
    assert harness.spawned[0].terminated is False


@pytest.mark.parametrize("key", ["instance_id", "process_id", "profile_id", "mode", "backend"])
def test_release_refuses_health_missing_any_identity_field(monkeypatch, tmp_path, key):
    harness = Harness().install(monkeypatch)
    runtime = manager(tmp_path)
    connection = run(runtime.ensure())
    server = harness.servers[connection.url]
    server.health_override = {key: None}
    with pytest.raises(DomainError) as exc:
        run(runtime.release(connection))
    assert exc.value.code == "instance_mismatch"
    assert server.shutdown_calls == []
    assert server.state_calls == 0, "identity is verified before any drain poll"
    assert harness.spawned[0].terminated is False
    server.health_override = {}
    assert run(runtime.release(connection))["status"] == "released"


def test_release_refuses_health_foreign_profile_identity(monkeypatch, tmp_path):
    harness = Harness().install(monkeypatch)
    runtime = manager(tmp_path)
    connection = run(runtime.ensure())
    server = harness.servers[connection.url]
    server.health_override = {"profile_id": "0" * 16, "mode": "real", "backend": "agx"}
    with pytest.raises(DomainError) as exc:
        run(runtime.release(connection))
    assert exc.value.code == "instance_mismatch"
    assert server.shutdown_calls == []
    assert harness.spawned[0].terminated is False


# ---------------------------------------------------------------------------
# drain: malformed state and instance replacement
# ---------------------------------------------------------------------------

def test_state_instance_id_extraction():
    assert managed_runtime._state_instance_id({"instance_id": "top"}) == "top"
    assert managed_runtime._state_instance_id(
        {"observation_identity": {"instance_id": "nested"}}) == "nested"
    assert managed_runtime._state_instance_id(
        {"instance_id": "same", "observation_identity": {"instance_id": "same"}}) == "same"
    for malformed in ({}, {"active_job_id": None}, {"observation_identity": {}},
                      {"observation_identity": "nope"}, {"observation_identity": {"instance_id": ""}},
                      {"instance_id": 7}, {"instance_id": ""},
                      {"instance_id": "a", "observation_identity": {"instance_id": "b"}}):
        assert managed_runtime._state_instance_id(malformed) is None, malformed


@pytest.mark.parametrize("bad_state", [
    {},
    {"active_job_id": None},
    {"active_job_id": None, "observation_identity": {}},
    {"active_job_id": None, "observation_identity": "not-an-object"},
    {"active_job_id": None, "observation_identity": {"instance_id": ""}},
    {"active_job_id": None, "observation_identity": {"instance_id": 5}},
])
def test_release_refuses_state_without_instance_identity(monkeypatch, tmp_path, bad_state):
    harness = Harness().install(monkeypatch)
    runtime = manager(tmp_path)
    connection = run(runtime.ensure())
    server = harness.servers[connection.url]
    server.state_override = bad_state
    with pytest.raises(DomainError) as exc:
        run(runtime.release(connection))
    assert exc.value.code == "runtime_invalid_response"
    assert server.shutdown_calls == []
    assert harness.spawned[0].terminated is False
    server.state_override = None
    assert run(runtime.release(connection))["status"] == "released"


def test_release_refuses_state_missing_active_job_id(monkeypatch, tmp_path):
    """A state document with identity but no idleness field is not idle."""
    harness = Harness().install(monkeypatch)
    runtime = manager(tmp_path)
    connection = run(runtime.ensure())
    server = harness.servers[connection.url]
    server.state_override = {"observation_identity": {"instance_id": connection.instance_id}}
    with pytest.raises(DomainError) as exc:
        run(runtime.release(connection))
    assert exc.value.code == "runtime_invalid_response"
    assert server.shutdown_calls == []
    server.state_override = None
    assert run(runtime.release(connection))["status"] == "released"


@pytest.mark.parametrize("invalid", [0, "", [], {}, True, 3.5])
def test_release_refuses_state_with_invalid_active_job_id(monkeypatch, tmp_path, invalid):
    harness = Harness().install(monkeypatch)
    runtime = manager(tmp_path)
    connection = run(runtime.ensure())
    server = harness.servers[connection.url]
    server.state_override = {"active_job_id": invalid,
                             "observation_identity": {"instance_id": connection.instance_id}}
    with pytest.raises(DomainError) as exc:
        run(runtime.release(connection))
    assert exc.value.code == "runtime_invalid_response"
    assert server.shutdown_calls == []
    server.state_override = None
    assert run(runtime.release(connection))["status"] == "released"


def test_release_refuses_state_with_conflicting_instance_identity(monkeypatch, tmp_path):
    harness = Harness().install(monkeypatch)
    runtime = manager(tmp_path)
    connection = run(runtime.ensure())
    server = harness.servers[connection.url]
    server.state_override = {"active_job_id": None, "instance_id": connection.instance_id,
                             "observation_identity": {"instance_id": "conflicting-instance"}}
    with pytest.raises(DomainError) as exc:
        run(runtime.release(connection))
    assert exc.value.code == "runtime_invalid_response"
    assert server.shutdown_calls == []
    server.state_override = None
    assert run(runtime.release(connection))["status"] == "released"


def test_release_detects_instance_replaced_while_draining(monkeypatch, tmp_path):
    """A restart while waiting must stop the release, not shut down the new run."""
    harness = Harness().install(monkeypatch)
    runtime = manager(tmp_path)
    connection = run(runtime.ensure())
    server = harness.servers[connection.url]
    server.active_job_id = "job-0001"
    steps = {"count": 0}

    async def restart_mid_drain():
        while server.state_calls < 2:
            await asyncio.sleep(0.005)
        server.instance_id = "restarted-instance"

    async def scenario():
        restarter = asyncio.create_task(restart_mid_drain())
        try:
            return await runtime.release(connection)
        finally:
            await restarter

    with pytest.raises(DomainError) as exc:
        run(scenario())
    assert exc.value.code == "instance_mismatch"
    assert server.state_calls >= 2, "identity is re-checked on every drain poll"
    assert server.shutdown_calls == [], "the restarted executor is never shut down"
    assert harness.spawned[0].terminated is False


def test_release_refuses_shutdown_when_instance_changed_after_drain(monkeypatch, tmp_path):
    """A restart between the last drain poll and /v1/shutdown is refused."""
    harness = Harness().install(monkeypatch)
    runtime = manager(tmp_path)
    connection = run(runtime.ensure())
    server = harness.servers[connection.url]
    server.shutdown_error = {"code": "instance_mismatch",
                             "message": "Executor instance changed; nothing was shut down."}
    with pytest.raises(DomainError) as exc:
        run(runtime.release(connection))
    assert exc.value.code == "instance_mismatch"
    assert server.shutdown_calls == [{"expected_instance_id": connection.instance_id}]
    assert server.alive is True
    # The manager keeps ownership so a later, correctly identified release can retry.
    assert runtime.connection is not None
    server.shutdown_error = None
    assert run(runtime.release(connection))["status"] == "released"


def test_release_foreign_connection_is_refused(monkeypatch, tmp_path):
    Harness().install(monkeypatch)
    runtime = manager(tmp_path)
    foreign = RuntimeConnection(url="http://127.0.0.1:1", model_token_file=tmp_path / "m",
                                operator_token_file=tmp_path / "o", instance_id="x", owned=True,
                                profile_root=tmp_path / "elsewhere", profile_id="p",
                                mode="simulation", target="local")
    with pytest.raises(DomainError) as exc:
        run(runtime.release(foreign))
    assert exc.value.code == "foreign_connection"


def test_shutdown_without_connection_is_idle(tmp_path):
    assert run(manager(tmp_path).shutdown()) == {"status": "idle", "shutdown": False}


# ---------------------------------------------------------------------------
# frontend failure / owned-executor lifecycle
# ---------------------------------------------------------------------------

def test_owned_executor_is_reaped_when_frontend_fails_before_session_acquire(monkeypatch, tmp_path):
    """ensure() -> operator session acquire fails -> release() must not leave a
    server, a published record, or a manager ownership claim behind."""
    harness = Harness().install(monkeypatch)
    runtime = manager(tmp_path)

    class SessionAcquireFailed(Exception):
        pass

    def start_frontend():
        connection = run(runtime.ensure())  # owned executor is now running
        try:
            raise SessionAcquireFailed("operator session acquire failed")
        except BaseException:
            # The cleanup the managed frontend performs when setup fails before
            # any control session exists.
            run(runtime.release(connection))
            raise

    with pytest.raises(SessionAcquireFailed):
        start_frontend()
    server = harness.spawned[0].server
    assert server.shutdown_calls == [{"expected_instance_id": server.instance_id}]
    assert runtime.connection is None
    assert not runtime.profile.runtime_path.exists(), "no stale record may advertise a dead owned executor"


def test_shared_attachment_failure_detaches_without_shutdown(monkeypatch, tmp_path):
    """A shared executor has shutdown_on_loss false: cleanup only detaches."""
    harness = Harness().install(monkeypatch)
    owner = manager(tmp_path)
    owned = run(owner.ensure())
    shared = manager(tmp_path)
    reused = run(shared.ensure())
    assert reused.owned is False
    assert run(shared.release(reused))["status"] == "detached"
    assert harness.spawned[0].server.shutdown_calls == []
    assert harness.spawned[0].terminated is False
    assert run(owner.release(owned))["status"] == "released"


def test_frontend_cleanup_failure_is_loud_and_keeps_ownership(monkeypatch, tmp_path):
    """An unverifiable owned executor is never silently reported as cleaned up."""
    harness = Harness().install(monkeypatch)
    runtime = manager(tmp_path)
    connection = run(runtime.ensure())
    del harness.servers[connection.url]
    with pytest.raises(DomainError) as exc:
        run(runtime.release(connection))
    assert exc.value.code == "runtime_unreachable"
    assert runtime.connection is not None
    assert harness.spawned[0].server.shutdown_calls == []


# ---------------------------------------------------------------------------
# secrets never leave their files / logs
# ---------------------------------------------------------------------------

def test_no_secret_logging_or_connection_leak(monkeypatch, tmp_path, caplog):
    harness = Harness().install(monkeypatch)
    caplog.set_level(logging.DEBUG)
    runtime = manager(tmp_path)
    connection = run(runtime.ensure())
    model = connection.model_token_file.read_text().strip()
    operator = connection.operator_token_file.read_text().strip()
    result = run(runtime.release(connection))
    payload = json.dumps(result) + repr(connection) + caplog.text
    assert model not in payload
    assert operator not in payload
    assert "token" in connection.model_token_file.name  # only the path is exposed
    assert len(harness.spawned) == 1


def test_error_messages_do_not_leak_credentials(monkeypatch, tmp_path):
    harness = Harness().install(monkeypatch)
    runtime = manager(tmp_path)
    connection = run(runtime.ensure())
    harness.servers[connection.url].health_override = {"instance_id": "nope"}
    with pytest.raises(DomainError) as exc:
        run(runtime.release(connection))
    token = connection.model_token_file.read_text().strip()
    assert token not in str(exc.value) and token not in exc.value.message


# ---------------------------------------------------------------------------
# remote target over (faked) SSH
# ---------------------------------------------------------------------------

class FakeSshClient:
    plan = {}
    instances = []

    def __init__(self, host, **kwargs):
        self.host = host
        self.kwargs = kwargs
        self.requests = []
        self.tunnels = []
        FakeSshClient.instances.append(self)

    def request(self, payload):
        self.requests.append(payload)
        plan = FakeSshClient.plan
        if payload["action"] == "start":
            return {"version": 1, "status": "ok", "runtime": dict(plan["runtime"])}
        if payload["action"] == "release":
            handler = plan.get("release")
            if handler is not None:
                return handler(payload)
            return {"version": 1, "status": "ok",
                    "released": {"status": "released", "shutdown": True, "stopped": True,
                                 "drained": True, "instance_id": payload.get("instance_id")}}
        raise AssertionError(payload["action"])

    def open_tunnel(self, local_port, remote_port, *, stderr_path=None):
        plan = FakeSshClient.plan
        tunnel = FakeTunnel()
        self.tunnels.append({"local_port": local_port, "remote_port": remote_port, "process": tunnel,
                             "stderr_path": stderr_path})
        server = FakeServer(plan["harness"], url=f"http://127.0.0.1:{local_port}",
                            instance_id=plan["runtime"]["instance_id"],
                            profile_id=plan["runtime"]["profile_id"],
                            mode=plan["runtime"]["mode"], backend=plan["runtime"]["backend"],
                            pid=plan["runtime"]["pid"])
        plan["harness"].servers[server.url] = server
        return tunnel

    @staticmethod
    def close_tunnel(process, *, timeout_s=5.0):
        if process is not None:
            process.terminated = True
            process.closed = True


def install_remote(monkeypatch, tmp_path, *, mode="simulation", backend="mujoco"):
    harness = Harness().install(monkeypatch)
    profiles.save_remote(tmp_path, "lab", "piper-lab.local")
    tokens = ("m" * 48, "o" * 48)
    plan = {
        "harness": harness,
        "runtime": {"url": "http://127.0.0.1:9", "instance_id": "remote-instance-1",
                    "profile_id": "remote-profile-1", "mode": mode, "backend": backend,
                    "pid": 4242, "port": 32123, "owned": True,
                    "model_token": tokens[0], "operator_token": tokens[1]},
    }
    FakeSshClient.plan = plan
    FakeSshClient.instances = []
    monkeypatch.setattr(managed_runtime, "SshRuntimeClient", FakeSshClient)
    return harness, plan, tokens


def test_remote_ensure_uses_tunnel_and_private_cache(monkeypatch, tmp_path):
    harness, plan, tokens = install_remote(monkeypatch, tmp_path)
    runtime = manager(tmp_path, target="lab")
    connection = run(runtime.ensure())

    client = FakeSshClient.instances[0]
    assert client.host == "piper-lab.local"
    start = client.requests[0]
    assert start["action"] == "start" and start["mode"] == "simulation"
    assert start["target"] == "lab"
    assert "token" not in json.dumps(start)
    assert harness.spawned == [], "a remote target must never start a local executor"

    assert connection.owned is True
    assert connection.instance_id == "remote-instance-1"
    assert connection.profile_id == "remote-profile-1"
    assert connection.url == f"http://127.0.0.1:{client.tunnels[0]['local_port']}"
    assert client.tunnels[0]["remote_port"] == 32123

    from piperx_middleware.profiles import profile_root
    cache_root = profile_root(tmp_path, "simulation", "lab")
    assert cache_root in connection.model_token_file.parents
    assert connection.model_token_file.read_text().strip() == tokens[0]
    assert connection.operator_token_file.read_text().strip() == tokens[1]
    assert (connection.model_token_file.stat().st_mode & 0o777) == 0o600
    assert (connection.model_token_file.parent.stat().st_mode & 0o777) == 0o700

    result = run(runtime.release(connection))
    assert result["status"] == "released"
    assert result["stopped"] is True and result["shutdown"] is True
    assert result["instance_id"] == connection.instance_id
    assert client.requests[-1]["action"] == "release"
    assert client.requests[-1]["instance_id"] == "remote-instance-1"
    assert client.tunnels[0]["process"].closed is True
    assert not connection.model_token_file.parent.exists(), "temporary credential cache must be removed"


def remote_cache_dirs(tmp_path):
    return list(profiles.profile_root(tmp_path, "simulation", "lab").glob("remote-session-*"))


def release_response(fields, *, with_instance=True):
    """Fake remote ``release`` handler answering with a ``released`` payload.

    ``with_instance`` mirrors the real protocol: every status except the
    record-less ``absent`` case names the instance it is reporting on.
    """
    def handler(payload):
        released = dict(fields)
        if with_instance:
            released.setdefault("instance_id", payload["instance_id"])
        return {"version": 1, "status": "ok", "released": released}
    return handler


def test_remote_release_success_confirms_identity_and_cleans_up(monkeypatch, tmp_path):
    harness, plan, tokens = install_remote(monkeypatch, tmp_path)
    runtime = manager(tmp_path, target="lab")
    connection = run(runtime.ensure())
    client = FakeSshClient.instances[0]

    result = run(runtime.release(connection))
    assert result["status"] == "released"
    assert result["stopped"] is True and result["shutdown"] is True and result["drained"] is True
    assert result["instance_id"] == connection.instance_id
    assert result["remote"]["instance_id"] == connection.instance_id
    assert [request["action"] for request in client.requests] == ["start", "release"]
    assert client.tunnels[0]["process"].closed is True
    assert not connection.model_token_file.parent.exists()
    assert runtime.connection is None and runtime._remote is None
    assert remote_cache_dirs(tmp_path) == []


def test_remote_release_busy_retains_session_for_status_and_retry(monkeypatch, tmp_path):
    harness, plan, tokens = install_remote(monkeypatch, tmp_path)
    runtime = manager(tmp_path, target="lab")
    connection = run(runtime.ensure())
    client = FakeSshClient.instances[0]
    cache = connection.model_token_file.parent
    plan["release"] = release_response({"status": "busy", "shutdown": False, "drained": False,
                                        "active_job_id": "job-remote-1"})

    result = run(runtime.release(connection))
    assert result["status"] == "busy"
    assert result["shutdown"] is False and result["drained"] is False
    assert result["instance_id"] == connection.instance_id
    assert result["remote"]["active_job_id"] == "job-remote-1"
    assert client.tunnels[0]["process"].closed is False, "a busy executor keeps its tunnel"
    assert cache.exists(), "a busy executor keeps its credential cache"
    assert runtime.connection is connection and runtime._remote is not None
    # The retained session is still the live one: status/reuse must not open a second tunnel.
    assert run(runtime.ensure()) is connection
    assert len(FakeSshClient.instances) == 1

    # The remote action finished: the explicit retry (/stop again) now confirms.
    plan["release"] = None
    retried = run(runtime.release(connection))
    assert retried["status"] == "released"
    assert client.tunnels[0]["process"].closed is True
    assert not cache.exists()
    assert runtime.connection is None and runtime._remote is None
    assert [request["action"] for request in client.requests] == ["start", "release", "release"]


@pytest.mark.parametrize("fields,with_instance", [
    ({"status": "shutdown_unconfirmed", "shutdown": True, "stopped": False}, True),
    ({"status": "released", "shutdown": True, "stopped": False}, True),
    ({"status": "released", "shutdown": True}, True),
    ({"status": "absent", "shutdown": False}, False),  # _host_release: no record to report on
])
def test_remote_release_unconfirmed_stop_is_never_success(monkeypatch, tmp_path, fields, with_instance):
    harness, plan, tokens = install_remote(monkeypatch, tmp_path)
    runtime = manager(tmp_path, target="lab")
    connection = run(runtime.ensure())
    client = FakeSshClient.instances[0]
    plan["release"] = release_response(fields, with_instance=with_instance)

    result = run(runtime.release(connection))
    assert result["status"] == "shutdown_unconfirmed", fields
    assert result["stopped"] is False, fields
    assert client.tunnels[0]["process"].closed is False, fields
    assert connection.model_token_file.parent.exists(), fields
    assert runtime.connection is connection and runtime._remote is not None, fields

    # The retained session supports an explicit retry.
    plan["release"] = None
    assert run(runtime.release(connection))["status"] == "released"
    assert client.tunnels[0]["process"].closed is True
    assert not connection.model_token_file.parent.exists()


@pytest.mark.parametrize("response", [
    {"version": 1, "status": "ok"},                                     # no released payload at all
    {"version": 1, "status": "ok", "released": "released"},             # wrong payload type
    {"version": 1, "status": "ok", "released": {"status": "unknown"}},  # unrecognized status
])
def test_remote_release_malformed_payload_is_refused_and_retained(monkeypatch, tmp_path, response):
    harness, plan, tokens = install_remote(monkeypatch, tmp_path)
    runtime = manager(tmp_path, target="lab")
    connection = run(runtime.ensure())
    client = FakeSshClient.instances[0]
    plan["release"] = lambda payload: dict(response)

    with pytest.raises(DomainError) as exc:
        run(runtime.release(connection))
    assert exc.value.code == "ssh_protocol", response
    assert client.tunnels[0]["process"].closed is False, response
    assert connection.model_token_file.parent.exists(), response
    assert runtime.connection is connection and runtime._remote is not None, response

    plan["release"] = None
    assert run(runtime.release(connection))["status"] == "released"


@pytest.mark.parametrize("released", [
    {"status": "released", "shutdown": True, "stopped": True, "instance_id": "some-other-instance"},
    {"status": "released", "shutdown": True, "stopped": True},  # identity missing is a mismatch
])
def test_remote_release_foreign_or_missing_instance_is_not_success(monkeypatch, tmp_path, released):
    harness, plan, tokens = install_remote(monkeypatch, tmp_path)
    runtime = manager(tmp_path, target="lab")
    connection = run(runtime.ensure())
    client = FakeSshClient.instances[0]
    plan["release"] = lambda payload: {"version": 1, "status": "ok", "released": dict(released)}

    with pytest.raises(DomainError) as exc:
        run(runtime.release(connection))
    assert exc.value.code == "instance_mismatch", released
    assert client.tunnels[0]["process"].closed is False
    assert connection.model_token_file.parent.exists()
    assert runtime.connection is connection and runtime._remote is not None

    # A correctly identified retry still works.
    plan["release"] = None
    assert run(runtime.release(connection))["status"] == "released"


def test_remote_release_transport_loss_retains_session(monkeypatch, tmp_path):
    harness, plan, tokens = install_remote(monkeypatch, tmp_path)
    runtime = manager(tmp_path, target="lab")
    connection = run(runtime.ensure())
    client = FakeSshClient.instances[0]

    def lost(payload):
        raise DomainError("ssh_timeout", "SSH command timed out.")

    plan["release"] = lost
    with pytest.raises(DomainError) as exc:
        run(runtime.release(connection))
    assert exc.value.code == "ssh_timeout"
    assert client.tunnels[0]["process"].closed is False, "a lost request must not close the tunnel"
    assert connection.model_token_file.parent.exists(), "a lost request must not delete the credentials"
    assert runtime.connection is connection and runtime._remote is not None

    plan["release"] = None
    assert run(runtime.release(connection))["status"] == "released"
    assert client.tunnels[0]["process"].closed is True
    assert not connection.model_token_file.parent.exists()
    assert runtime.connection is None and runtime._remote is None


def test_remote_shared_detach_cleans_up_without_contacting_executor(monkeypatch, tmp_path):
    harness, plan, tokens = install_remote(monkeypatch, tmp_path)
    runtime = manager(tmp_path, target="lab")
    connection = run(runtime.ensure())
    client = FakeSshClient.instances[0]

    result = run(runtime.release(connection, shutdown_owned=False))
    assert result == {"status": "detached", "shutdown": False, "instance_id": connection.instance_id}
    assert [request["action"] for request in client.requests] == ["start"], (
        "an explicit detach never sends a release or shutdown request")
    assert client.tunnels[0]["process"].closed is True
    assert not connection.model_token_file.parent.exists()
    assert runtime.connection is None and runtime._remote is None


def test_remote_release_detached_result_cleans_up(monkeypatch, tmp_path):
    harness, plan, tokens = install_remote(monkeypatch, tmp_path)
    runtime = manager(tmp_path, target="lab")
    connection = run(runtime.ensure())
    client = FakeSshClient.instances[0]
    plan["release"] = release_response({"status": "detached", "shutdown": False})

    result = run(runtime.release(connection))
    assert result["status"] == "detached"
    assert result["shutdown"] is False
    assert result["instance_id"] == connection.instance_id
    assert client.tunnels[0]["process"].closed is True
    assert not connection.model_token_file.parent.exists()
    assert runtime.connection is None and runtime._remote is None


def test_remote_reconnect_replaces_tunnel_and_cache_without_robot_action(monkeypatch, tmp_path):
    harness, plan, tokens = install_remote(monkeypatch, tmp_path)
    runtime = manager(tmp_path, target="lab", startup_timeout_s=0.5)

    first = run(runtime.ensure())
    first_client = FakeSshClient.instances[-1]
    first_tunnel = first_client.tunnels[0]["process"]
    first_cache = first.model_token_file.parent
    first_server = harness.servers[first.url]

    # The tunnel stops answering: each ensure() is an explicit reconnect.
    del harness.servers[first.url]
    second = run(runtime.ensure())
    second_client = FakeSshClient.instances[-1]
    second_tunnel = second_client.tunnels[0]["process"]
    second_server = harness.servers[second.url]
    assert first_tunnel.closed is True, "the replaced tunnel is closed"
    assert not first_cache.exists(), "the replaced credential cache is deleted"

    del harness.servers[second.url]
    third = run(runtime.ensure())
    third_client = FakeSshClient.instances[-1]
    third_tunnel = third_client.tunnels[0]["process"]
    third_server = harness.servers[third.url]
    assert second_tunnel.closed is True, "each reconnect closes the tunnel it replaces"
    assert not second.model_token_file.parent.exists()

    assert len(FakeSshClient.instances) == 3
    live = [entry["process"] for instance in FakeSshClient.instances
            for entry in instance.tunnels if not entry["process"].closed]
    assert live == [third_tunnel], "exactly one tunnel survives repeated reconnects"
    caches = remote_cache_dirs(tmp_path)
    assert len(caches) == 1, "exactly one credential cache survives repeated reconnects"
    assert third.model_token_file.parent == caches[0]
    assert runtime.connection is third and runtime._remote is not None

    # Reconnecting is local-only: no robot action, no remote shutdown, no duplicate executor.
    for instance in FakeSshClient.instances:
        assert [request["action"] for request in instance.requests] == ["start"], instance.requests
    assert harness.spawned == []
    for server in (first_server, second_server, third_server):
        assert all(path == "/health" for _method, path, _body in server.requests), server.requests

    assert run(runtime.release(third))["status"] == "released"


def test_remote_identity_mismatch_cleans_up_tunnel_and_cache(monkeypatch, tmp_path):
    harness, plan, tokens = install_remote(monkeypatch, tmp_path)
    plan["runtime"]["instance_id"] = "bootstrap-instance"
    # The tunnel health answers with a different instance than the bootstrap.
    original_open = FakeSshClient.open_tunnel

    def open_with_mismatch(self, local_port, remote_port, *, stderr_path=None):
        tunnel = original_open(self, local_port, remote_port, stderr_path=stderr_path)
        server = harness.servers[f"http://127.0.0.1:{local_port}"]
        server.instance_id = "tunnel-impostor"
        return tunnel

    monkeypatch.setattr(FakeSshClient, "open_tunnel", open_with_mismatch)
    runtime = manager(tmp_path, target="lab", startup_timeout_s=0.2)
    with pytest.raises(DomainError) as exc:
        run(runtime.ensure())
    assert exc.value.code == "runtime_identity_mismatch"
    client = FakeSshClient.instances[0]
    assert client.tunnels[0]["process"].closed is True
    caches = list(profiles.profile_root(tmp_path, "simulation", "lab").glob("remote-session-*"))
    assert caches == []


@pytest.mark.parametrize("override", [
    {"profile_id": None}, {"mode": None}, {"backend": None},
    {"profile_id": "some-other-profile"}, {"mode": "real"}, {"backend": "agx"},
])
def test_remote_tunnel_health_missing_or_foreign_identity_is_refused(monkeypatch, tmp_path, override):
    harness, plan, tokens = install_remote(monkeypatch, tmp_path)
    original_open = FakeSshClient.open_tunnel

    def open_with_override(self, local_port, remote_port, *, stderr_path=None):
        tunnel = original_open(self, local_port, remote_port, stderr_path=stderr_path)
        harness.servers[f"http://127.0.0.1:{local_port}"].health_override = dict(override)
        return tunnel

    monkeypatch.setattr(FakeSshClient, "open_tunnel", open_with_override)
    runtime = manager(tmp_path, target="lab", startup_timeout_s=0.2)
    with pytest.raises(DomainError) as exc:
        run(runtime.ensure())
    assert exc.value.code == "runtime_identity_mismatch"
    client = FakeSshClient.instances[0]
    assert client.tunnels[0]["process"].closed is True, "a refused tunnel is always closed"
    assert list(profiles.profile_root(tmp_path, "simulation", "lab").glob("remote-session-*")) == []


@pytest.mark.parametrize("field,code", [
    ("profile_id", "ssh_protocol"), ("instance_id", "ssh_protocol"),
    ("mode", "runtime_identity_mismatch"), ("backend", "mode_restriction"),
])
def test_remote_bootstrap_missing_identity_is_refused(monkeypatch, tmp_path, field, code):
    harness, plan, tokens = install_remote(monkeypatch, tmp_path)
    plan["runtime"].pop(field)
    with pytest.raises(DomainError) as exc:
        run(manager(tmp_path, target="lab").ensure())
    assert exc.value.code == code
    assert FakeSshClient.instances[0].tunnels == [], "no tunnel is opened without a full bootstrap identity"


def test_remote_simulation_refuses_physical_backend(monkeypatch, tmp_path):
    install_remote(monkeypatch, tmp_path, mode="simulation", backend="agx")
    with pytest.raises(DomainError) as exc:
        run(manager(tmp_path, target="lab").ensure())
    assert exc.value.code == "mode_restriction"
    assert FakeSshClient.instances[0].tunnels == []


def test_remote_real_refuses_simulator_backend(monkeypatch, tmp_path):
    install_remote(monkeypatch, tmp_path, mode="real", backend="mujoco")
    with pytest.raises(DomainError) as exc:
        run(manager(tmp_path, "real", target="lab").ensure())
    assert exc.value.code == "mode_restriction"


def test_unknown_remote_target_is_refused(monkeypatch, tmp_path):
    Harness().install(monkeypatch)
    with pytest.raises(DomainError) as exc:
        run(manager(tmp_path, target="lab").ensure())
    assert exc.value.code == "unknown_remote"


# ---------------------------------------------------------------------------
# SSH command construction and protocol
# ---------------------------------------------------------------------------

def test_build_ssh_argv_is_fixed_and_strict():
    argv = ssh_runtime.build_ssh_argv("piper-lab.local")
    assert argv[0] == "ssh"
    assert "StrictHostKeyChecking=yes" in argv
    assert "BatchMode=yes" in argv
    assert not any(option in argv for option in ("StrictHostKeyChecking=no", "StrictHostKeyChecking=accept-new",
                                                 "UserKnownHostsFile=/dev/null"))
    assert argv[-5:] == ["--", "piper-lab.local", "piper-robot", "host", "--stdio"]


def test_build_ssh_tunnel_argv_is_loopback_only():
    argv = ssh_runtime.build_ssh_argv("lab", remote_argv=None, local_forward=(45678, 8765))
    assert "-N" in argv
    assert "ExitOnForwardFailure=yes" in argv
    assert argv[argv.index("-L") + 1] == "127.0.0.1:45678:127.0.0.1:8765"
    assert argv[-2:] == ["--", "lab"]
    assert "piper-robot" not in argv
    with pytest.raises(DomainError):
        ssh_runtime.build_ssh_argv("lab", local_forward=(0, 8765), remote_argv=None)
    with pytest.raises(DomainError):
        ssh_runtime.build_ssh_argv("lab", local_forward=(45678, 70000), remote_argv=None)


@pytest.mark.parametrize("host", ["host; id", "-oProxyCommand=x", "host\nx", "user:pw@host", "a b"])
def test_build_ssh_argv_rejects_injection(host):
    with pytest.raises((DomainError, ValueError)):
        ssh_runtime.build_ssh_argv(host)


def test_ssh_request_sends_json_on_stdin_not_argv():
    captured = {}

    def runner(argv, input_text, timeout):
        captured.update(argv=list(argv), input=input_text, timeout=timeout)
        return subprocess.CompletedProcess(list(argv), 0, stdout='{"version":1,"status":"ok","runtime":{}}', stderr="")

    client = ssh_runtime.SshRuntimeClient("piper-lab.local", runner=runner)
    response = client.request({"action": "start", "mode": "simulation", "target": "lab"})
    assert response["status"] == "ok"
    assert json.loads(captured["input"]) == {"action": "start", "mode": "simulation", "target": "lab"}
    assert captured["argv"][-3:] == ["piper-robot", "host", "--stdio"]
    assert "start" not in captured["argv"]


def test_ssh_failure_is_structured_and_redacted():
    secret = "s" * 60

    def runner(argv, input_text, timeout):
        return subprocess.CompletedProcess(list(argv), 255,
                                           stdout="", stderr=f"Host key verification failed. key={secret}")

    client = ssh_runtime.SshRuntimeClient("piper-lab.local", runner=runner)
    with pytest.raises(DomainError) as exc:
        client.request({"action": "ping"})
    assert exc.value.code == "ssh_failed"
    assert "host key" in exc.value.message.lower()
    assert secret not in exc.value.message
    assert "<redacted>" in exc.value.message
    assert "known_hosts" in exc.value.message


def test_ssh_remote_error_is_mapped():
    def runner(argv, input_text, timeout):
        return subprocess.CompletedProcess(
            list(argv), 0,
            stdout=json.dumps({"version": 1, "status": "error",
                               "error": {"code": "mode_restriction", "message": "no simulator for real"}}),
            stderr="")

    client = ssh_runtime.SshRuntimeClient("lab", runner=runner)
    with pytest.raises(DomainError) as exc:
        client.request({"action": "start"})
    assert exc.value.code == "mode_restriction"


@pytest.mark.parametrize("stdout", ["", "   ", "not json", '{"a":1}{"b":2}', "[1,2]", '"text"'])
def test_parse_host_response_rejects_bad_documents(stdout):
    with pytest.raises(DomainError) as exc:
        ssh_runtime.parse_host_response(stdout)
    assert exc.value.code == "ssh_protocol"
    assert "not json" not in exc.value.message


def test_credential_cache_is_private_and_cleanable(tmp_path):
    profile = profiles.ensure_profile(tmp_path, "simulation")
    directory, model, operator = ssh_runtime.write_credential_cache(profile.root, "m" * 48, "o" * 48)
    assert (directory.stat().st_mode & 0o777) == 0o700
    assert (model.stat().st_mode & 0o777) == 0o600
    assert (operator.stat().st_mode & 0o777) == 0o600
    assert model.read_text().strip() == "m" * 48
    ssh_runtime.cleanup_credential_cache(directory)
    assert not directory.exists()
    with pytest.raises(DomainError) as exc:
        ssh_runtime.write_credential_cache(profile.root, "short", "o" * 48)
    assert exc.value.code == "ssh_protocol"


# ---------------------------------------------------------------------------
# remote side: piper-robot host --stdio
# ---------------------------------------------------------------------------

class FakeHostManager:
    """Stands in for RuntimeManager inside host_main; never touches hardware."""

    def __init__(self, base, mode, target="local", simulation_backend="mujoco", **kwargs):
        self.base = Path(base)
        self.mode = mode
        self.target = target
        self._profile = profiles.ensure_profile(base, mode, target, simulation_backend=simulation_backend)
        self._connection = None

    @property
    def profile(self):
        return self._profile

    def ensure_sync(self):
        profile = self._profile
        record = {"url": "http://127.0.0.1:23456", "instance_id": "host-instance",
                  "profile_id": profile.profile_id, "mode": self.mode,
                  "backend": profile.backend, "pid": 4242}
        profile.runtime_path.write_text(json.dumps(record))
        self._connection = RuntimeConnection(
            url=record["url"], model_token_file=profile.model_token_file,
            operator_token_file=profile.operator_token_file, instance_id=record["instance_id"],
            owned=True, profile_root=profile.root, profile_id=profile.profile_id,
            mode=self.mode, target=self.target)
        return self._connection

    def release_sync(self, connection, shutdown_owned=True):
        return {"status": "released", "shutdown": True, "instance_id": connection.instance_id}


def test_host_main_start_returns_one_json_response(monkeypatch, tmp_path):
    base = tmp_path / "remote-base"
    monkeypatch.setattr(managed_runtime, "RuntimeManager", FakeHostManager)
    request = {"version": 1, "action": "start", "mode": "simulation", "simulation_backend": "mujoco"}
    out = io.StringIO()
    code = ssh_runtime.host_main(base, stdin=io.StringIO(json.dumps(request)), stdout=out)
    assert code == 0
    response = json.loads(out.getvalue())
    assert response["status"] == "ok"
    runtime = response["runtime"]
    assert runtime["instance_id"] == "host-instance"
    assert runtime["profile_id"] == profiles.profile_id("simulation", "local")
    assert runtime["mode"] == "simulation"
    assert runtime["backend"] == "mujoco"
    assert runtime["target"] == "local"
    assert runtime["port"] == 23456
    assert runtime["owned"] is True
    assert len(runtime["model_token"]) >= 32 and len(runtime["operator_token"]) >= 32
    assert runtime["model_token"] != runtime["operator_token"]


def test_host_main_without_arguments_reads_sys_stdin_and_stdout(monkeypatch, tmp_path):
    """``piper-robot host --stdio`` calls host_main() with no arguments."""
    monkeypatch.setattr(managed_runtime, "RuntimeManager", FakeHostManager)
    default_base = tmp_path / "remote-base"
    monkeypatch.setattr(ssh_runtime, "_default_remote_root", lambda: default_base)
    out = io.StringIO()
    monkeypatch.setattr(ssh_runtime.sys, "stdin", io.StringIO('{"action":"ping"}'))
    monkeypatch.setattr(ssh_runtime.sys, "stdout", out)
    assert ssh_runtime.host_main() == 0
    assert json.loads(out.getvalue()) == {"version": 1, "status": "ok"}

    out = io.StringIO()
    monkeypatch.setattr(ssh_runtime.sys, "stdin",
                        io.StringIO(json.dumps({"version": 1, "action": "start", "mode": "simulation"})))
    monkeypatch.setattr(ssh_runtime.sys, "stdout", out)
    assert ssh_runtime.host_main() == 0
    response = json.loads(out.getvalue())
    assert response["status"] == "ok"
    assert response["runtime"]["instance_id"] == "host-instance"
    assert (default_base / "profiles" / "simulation" / "local" / "runtime.json").is_file()


def test_host_main_ping_and_unknown_action(tmp_path):
    out = io.StringIO()
    assert ssh_runtime.host_main(tmp_path, stdin=io.StringIO('{"action":"ping"}'), stdout=out) == 0
    assert json.loads(out.getvalue()) == {"version": 1, "status": "ok"}
    out = io.StringIO()
    assert ssh_runtime.host_main(tmp_path, stdin=io.StringIO('{"action":"launch-missiles"}'), stdout=out) == 1
    response = json.loads(out.getvalue())
    assert response["status"] == "error"
    assert response["error"]["code"] == "unsupported_action"


def test_host_main_never_guesses_a_mode(monkeypatch, tmp_path):
    monkeypatch.setattr(managed_runtime, "RuntimeManager", FakeHostManager)
    out = io.StringIO()
    code = ssh_runtime.host_main(tmp_path, stdin=io.StringIO('{"action":"start"}'), stdout=out)
    assert code == 1
    assert json.loads(out.getvalue())["error"]["code"] == "invalid_request"


def test_host_main_invalid_json_writes_one_error_document(tmp_path):
    out = io.StringIO()
    assert ssh_runtime.host_main(tmp_path, stdin=io.StringIO("not json"), stdout=out) == 1
    assert json.loads(out.getvalue())["error"]["code"] == "invalid_request"


def test_host_main_release_uses_instance_identity(monkeypatch, tmp_path):
    base = tmp_path / "remote-base"
    monkeypatch.setattr(managed_runtime, "RuntimeManager", FakeHostManager)
    FakeHostManager(base, "simulation").ensure_sync()
    profile_id = profiles.profile_id("simulation", "local")
    request = {"action": "release", "mode": "simulation", "instance_id": "host-instance",
               "profile_id": profile_id}
    out = io.StringIO()
    assert ssh_runtime.host_main(base, stdin=io.StringIO(json.dumps(request)), stdout=out) == 0
    assert json.loads(out.getvalue())["released"]["status"] == "released"
    # A process restart is refused instead of shutting down a different run.
    for change in ({"instance_id": "old-instance"}, {"profile_id": "0" * 16}):
        hostile = dict(request, **change)
        out = io.StringIO()
        assert ssh_runtime.host_main(base, stdin=io.StringIO(json.dumps(hostile)), stdout=out) == 1
        assert json.loads(out.getvalue())["error"]["code"] == "instance_mismatch", change


@pytest.mark.parametrize("drop", ["instance_id", "profile_id"])
def test_host_main_release_refuses_missing_identity(monkeypatch, tmp_path, drop):
    base = tmp_path / "remote-base"
    monkeypatch.setattr(managed_runtime, "RuntimeManager", FakeHostManager)
    FakeHostManager(base, "simulation").ensure_sync()
    request = {"action": "release", "mode": "simulation", "instance_id": "host-instance",
               "profile_id": profiles.profile_id("simulation", "local")}
    request.pop(drop)
    out = io.StringIO()
    assert ssh_runtime.host_main(base, stdin=io.StringIO(json.dumps(request)), stdout=out) == 1
    response = json.loads(out.getvalue())
    assert response["error"]["code"] == "invalid_request", drop
    # The installed executor is untouched: its record is still published.
    assert (base / "profiles" / "simulation" / "local" / "runtime.json").is_file()


def test_host_main_never_prints_tracebacks_or_paths(monkeypatch, tmp_path):
    base = tmp_path / "remote-base"

    class ExplodingManager(FakeHostManager):
        def ensure_sync(self):
            raise RuntimeError("boom /secret/path/token-value")

    monkeypatch.setattr(managed_runtime, "RuntimeManager", ExplodingManager)
    out = io.StringIO()
    code = ssh_runtime.host_main(base, stdin=io.StringIO('{"action":"start","mode":"simulation"}'), stdout=out)
    assert code == 1
    response = json.loads(out.getvalue())
    assert response["error"]["code"] == "host_error"
    assert "boom" not in response["error"]["message"]
    assert "/secret/path" not in out.getvalue()


# ---------------------------------------------------------------------------
# real loopback HTTP path (no hardware, no external network)
# ---------------------------------------------------------------------------

def test_probe_health_uses_real_http_with_bearer_token(tmp_path):
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    profile = profiles.ensure_profile(tmp_path, "simulation")
    seen = {}
    payload = {"service": "piperx-middleware", "api_version": "1", "instance_id": "real-http-instance",
               "process_id": os.getpid()}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            seen["path"] = self.path
            seen["authorization"] = self.headers.get("Authorization")
            body = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}"
        health = run(managed_runtime.probe_health(url, profile.model_token_file, timeout_s=2.0))
        assert health["instance_id"] == "real-http-instance"
        assert seen["path"] == "/health"
        assert seen["authorization"] == "Bearer " + profile.model_token_file.read_text().strip()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_probe_health_returns_none_when_unreachable(tmp_path):
    profile = profiles.ensure_profile(tmp_path, "simulation")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]  # bound then closed: nothing listens
    assert run(managed_runtime.probe_health(f"http://127.0.0.1:{port}", profile.model_token_file,
                                            timeout_s=0.5)) is None


def test_foreign_real_service_on_recorded_port_is_refused_and_never_stopped(monkeypatch, tmp_path):
    """A real listening piperx service with a foreign identity is refused before
    any duplicate is started and is never sent a shutdown request."""
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    profile = profiles.ensure_profile(tmp_path, "simulation")
    seen = {"shutdown": 0}
    payload = {"service": "piperx-middleware", "api_version": "1", "process_id": os.getpid(),
               "instance_id": "foreign-instance", "profile_id": "0" * 16, "mode": "real", "backend": "agx"}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            body = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            seen["shutdown"] += 1
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"{}")

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    def no_spawn(*args, **kwargs):
        raise AssertionError("a foreign listener must never cause a duplicate executor")

    monkeypatch.setattr(managed_runtime.subprocess, "Popen", no_spawn)
    try:
        profile.runtime_path.write_text(json.dumps({
            "url": f"http://127.0.0.1:{server.server_address[1]}", "instance_id": "expected-instance",
            "profile_id": profile.profile_id, "mode": "simulation", "backend": "mujoco",
            "pid": os.getpid()}), encoding="utf-8")
        with pytest.raises(DomainError) as exc:
            run(managed_runtime.RuntimeManager(tmp_path, "simulation", startup_timeout_s=0.5).ensure())
        assert exc.value.code == "runtime_identity_mismatch"
        assert seen["shutdown"] == 0
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


# ---------------------------------------------------------------------------
# process liveness probe: non-destructive on Windows, POSIX kill(0) elsewhere
# ---------------------------------------------------------------------------

class FakeKernel32:
    """Minimal kernel32 double for the Windows probe.

    ``error`` is reported as the last error whenever ``OpenProcess`` fails
    (returns 0); ``GetExitCodeProcess`` writes ``exit_code`` through the DWORD
    pointer unless ``exit_ok`` is false.
    """

    def __init__(self, *, handle=0x1234, error=0, exit_code=None, exit_ok=True):
        self.handle = handle
        self.error = error
        self.exit_code = managed_runtime.STILL_ACTIVE if exit_code is None else exit_code
        self.exit_ok = exit_ok
        self.opened = []
        self.closed = []

    def OpenProcess(self, access, inherit, pid):
        self.opened.append((access, inherit, pid))
        return 0 if self.error else self.handle

    def GetExitCodeProcess(self, handle, pointer):
        if not self.exit_ok:
            return 0
        pointer._obj.value = self.exit_code
        return 1

    def CloseHandle(self, handle):
        self.closed.append(handle)
        return 1


def install_kernel32(monkeypatch, fake):
    """Route the Windows probe to ``fake`` without patching global ``os.name``.

    Replacing ``os.name`` would also change ``pathlib``'s flavour selection and
    break every ``Path`` in the process, so only the probe's own seams are
    replaced.
    """
    monkeypatch.setattr(managed_runtime, "_windows_kernel32", lambda: fake)
    monkeypatch.setattr(managed_runtime, "_windows_last_error", lambda: fake.error)
    return fake


def test_windows_pid_alive_reports_still_active_and_exited(monkeypatch):
    live = install_kernel32(monkeypatch, FakeKernel32(exit_code=managed_runtime.STILL_ACTIVE))
    assert managed_runtime._windows_pid_alive(4321) is True
    assert live.opened == [(managed_runtime.PROCESS_QUERY_LIMITED_INFORMATION, False, 4321)]
    assert live.closed == [live.handle], "the queried process handle must always be closed"

    dead = install_kernel32(monkeypatch, FakeKernel32(exit_code=1))
    assert managed_runtime._windows_pid_alive(4321) is False
    assert dead.closed == [dead.handle]


def test_windows_pid_alive_access_denied_is_conservatively_alive(monkeypatch):
    fake = install_kernel32(monkeypatch, FakeKernel32(error=managed_runtime.ERROR_ACCESS_DENIED))
    assert managed_runtime._windows_pid_alive(4321) is True
    assert fake.closed == [], "no handle was opened, so none is closed"


def test_windows_pid_alive_invalid_parameter_is_dead(monkeypatch):
    install_kernel32(monkeypatch, FakeKernel32(error=managed_runtime.ERROR_INVALID_PARAMETER))
    assert managed_runtime._windows_pid_alive(4321) is False


@pytest.mark.parametrize("error", [0, 6, 8, 1234, managed_runtime.ERROR_INVALID_PARAMETER + 1])
def test_windows_pid_alive_unknown_error_is_never_dead(monkeypatch, error):
    """An unexplained OpenProcess failure must not start a duplicate owner."""
    install_kernel32(monkeypatch, FakeKernel32(error=error))
    assert managed_runtime._windows_pid_alive(4321) is True


def test_windows_pid_alive_query_failure_is_conservatively_alive(monkeypatch):
    fake = install_kernel32(monkeypatch, FakeKernel32(exit_ok=False))
    assert managed_runtime._windows_pid_alive(4321) is True
    assert fake.closed == [fake.handle], "a failed query must still close the handle"


def test_windows_probe_constants_match_the_documented_api():
    assert managed_runtime.PROCESS_QUERY_LIMITED_INFORMATION == 0x1000
    assert managed_runtime.STILL_ACTIVE == 259
    assert managed_runtime.ERROR_ACCESS_DENIED == 5
    assert managed_runtime.ERROR_INVALID_PARAMETER == 87


def test_pid_alive_windows_branch_never_calls_os_kill(monkeypatch):
    """The defect: os.kill(pid, 0) TerminateProcesses the executor on Windows."""
    install_kernel32(monkeypatch, FakeKernel32(exit_code=managed_runtime.STILL_ACTIVE))
    monkeypatch.setattr(managed_runtime, "_is_windows", lambda: True)

    def forbidden_kill(*args):
        raise AssertionError("os.kill must never be used as a liveness probe on Windows")

    monkeypatch.setattr(managed_runtime.os, "kill", forbidden_kill)
    assert managed_runtime._pid_alive(4321) is True


def test_pid_alive_rejects_invalid_pids_without_probing(monkeypatch):
    def forbidden(pid):
        raise AssertionError(f"pid {pid!r} must be rejected before any probe")

    monkeypatch.setattr(managed_runtime, "_windows_pid_alive", forbidden)
    monkeypatch.setattr(managed_runtime, "_posix_pid_alive", forbidden)
    for bad in (0, -1, -4321, True, False, "4321", 43.21, None):
        assert managed_runtime._pid_alive(bad) is False


def test_pid_alive_dispatches_by_platform(monkeypatch):
    probed = {"windows": [], "posix": []}
    monkeypatch.setattr(managed_runtime, "_windows_pid_alive",
                        lambda pid: probed["windows"].append(pid) or True)
    monkeypatch.setattr(managed_runtime, "_posix_pid_alive",
                        lambda pid: probed["posix"].append(pid) or True)

    monkeypatch.setattr(managed_runtime, "_is_windows", lambda: True)
    assert managed_runtime._pid_alive(4321) is True
    assert probed == {"windows": [4321], "posix": []}, "Windows never calls os.kill"

    monkeypatch.setattr(managed_runtime, "_is_windows", lambda: False)
    assert managed_runtime._pid_alive(1234) is True
    assert probed == {"windows": [4321], "posix": [1234]}


@pytest.mark.skipif(os.name == "nt", reason="POSIX kill(0) semantics")
def test_posix_pid_alive_sees_self_and_a_reaped_child():
    assert managed_runtime._posix_pid_alive(os.getpid()) is True
    assert managed_runtime._pid_alive(os.getpid()) is True
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait()
    assert managed_runtime._pid_alive(child.pid) is False


@pytest.mark.skipif(os.name != "nt", reason="validates the real Windows OpenProcess probe")
def test_windows_pid_alive_probe_never_terminates_a_live_child():
    """The defect this guards: os.kill(pid, 0) would TerminateProcess here."""
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        for _ in range(5):
            assert managed_runtime._pid_alive(child.pid) is True
            assert child.poll() is None, "a liveness probe must never terminate the child"
        child.terminate()
        child.wait(timeout=10)
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=10)
    assert managed_runtime._pid_alive(child.pid) is False


# ---------------------------------------------------------------------------
# unit coverage of the record/url helpers
# ---------------------------------------------------------------------------

def test_read_runtime_json_requires_full_identity(tmp_path):
    path = tmp_path / "runtime.json"
    assert managed_runtime.read_runtime_json(path) is None
    path.write_text(json.dumps({"url": "http://127.0.0.1:1", "instance_id": "i"}))
    with pytest.raises(DomainError) as exc:
        managed_runtime.read_runtime_json(path)
    assert exc.value.code == "runtime_state_corrupt"
    path.write_text(json.dumps({"url": "http://127.0.0.1:1", "instance_id": "i", "profile_id": "p",
                                "mode": "simulation", "backend": "mujoco", "pid": True}))
    with pytest.raises(DomainError):
        managed_runtime.read_runtime_json(path)


@pytest.mark.parametrize("url", [
    "http://10.0.0.5:8765", "http://user:pw@127.0.0.1:8765", "http://127.0.0.1:8765?x=1",
    "ftp://127.0.0.1:8765", "", None, "http://127.0.0.1:notaport",
])
def test_validate_runtime_url_rejects_unsafe_urls(url):
    with pytest.raises(DomainError):
        managed_runtime.validate_runtime_url(url)


def test_validate_runtime_url_accepts_loopback():
    assert managed_runtime.validate_runtime_url("http://127.0.0.1:8765/") == "http://127.0.0.1:8765"
    assert managed_runtime.validate_runtime_url("http://localhost:1234") == "http://localhost:1234"


def test_manager_rejects_unknown_mode_and_bad_simulation_backend(tmp_path):
    with pytest.raises((DomainError, ValueError)):
        RuntimeManager(tmp_path, "physical")
    with pytest.raises((DomainError, ValueError)):
        RuntimeManager(tmp_path, "simulation", simulation_backend="agx")
