"""Managed executor lifecycle for the single-terminal frontend.

``RuntimeManager`` owns exactly one managed executor per isolated profile
(mode/target).  Its guarantees:

* **exclusive startup/reuse** - a cross-process :class:`ProcessLock` plus an
  in-process asyncio lock means concurrent frontends can never start two
  executors for the same profile;
* **identity-verified reuse** - a listening port is not enough; the published
  ``runtime.json`` *and* the live ``/health`` response must both match the
  expected profile, instance, process, mode and backend exactly.  A missing
  identity field is a mismatch, never a match;
* **no real fallback** - a real profile refuses a simulator backend and vice
  versa, and a remote target never silently falls back to a local executor;
* **graceful release** - an owned executor is drained (current accepted action
  completes) before shutdown; a busy executor is never killed, a shared
  executor is only detached, an identity mismatch refuses shutdown, and every
  drain poll must confirm that the executor instance has not been replaced;
* **no premature cleanup** - a remote session is only forgotten (tunnel closed,
  credential cache deleted) after a release that proves this exact instance
  stopped, or after an explicit shared detach.  A busy, unconfirmed, malformed
  or failed release keeps the session so status, stop and an explicit retry
  remain usable;
* **local-only reconnect** - replacing an unhealthy remote session closes only
  this manager's tunnel and credential cache; a reconnect never sends a robot
  action or a remote shutdown;

No method here connects CAN or enables physical hardware; hardware connection
stays an explicit operator action in the console.
"""
from __future__ import annotations

import asyncio
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

import httpx

from .interaction_types import RuntimeConnection
from .models import DomainError
from .process_lock import ProcessLock
from .profiles import (
    LOCAL_TARGET,
    REAL_BACKEND,
    SIMULATION_BACKENDS,
    Profile,
    ProfileError,
    ensure_profile,
    get_remote,
    list_remotes,
    profile_id,
    save_remote,
    validate_mode,
    validate_target,
)
from .ssh_runtime import (
    PROTOCOL_VERSION,
    SshRuntimeClient,
    cleanup_credential_cache,
    free_local_port,
    write_credential_cache,
)

HEALTH_TIMEOUT_S = 2.0
STARTUP_TIMEOUT_S = 30.0
DRAIN_TIMEOUT_S = 120.0
SHUTDOWN_GRACE_S = 15.0
POLL_INTERVAL_S = 0.1
LOCK_TIMEOUT_S = 30.0
SSH_TIMEOUT_S = 30.0

RUNTIME_KEYS = ("url", "instance_id", "profile_id", "mode", "backend", "pid")


# --------------------------------------------------------------------------
# runtime record and HTTP helpers (module level so tests can inject fakes)
# --------------------------------------------------------------------------

def read_runtime_json(path) -> dict | None:
    """Read the atomically published runtime record, or ``None`` when absent.

    A malformed record is refused rather than ignored: the server publishes it
    atomically, so corruption means external interference, and guessing could
    double-drive a robot.
    """
    path = Path(path)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise DomainError("runtime_state_corrupt",
                          f"{path.name} is not readable JSON; refusing to reuse or delete it blindly.", 409) from None
    if not isinstance(data, dict):
        raise DomainError("runtime_state_corrupt", f"{path.name} must contain a JSON object.", 409)
    for key in RUNTIME_KEYS:
        if key not in data:
            raise DomainError("runtime_state_corrupt", f"{path.name} is missing {key!r}; refusing to guess.", 409)
    if not isinstance(data["url"], str) or not data["url"]:
        raise DomainError("runtime_state_corrupt", f"{path.name} has an invalid url.", 409)
    for key in ("instance_id", "profile_id", "mode", "backend"):
        if not isinstance(data[key], str) or not data[key]:
            raise DomainError("runtime_state_corrupt", f"{path.name} has an invalid {key}.", 409)
    if not isinstance(data["pid"], int) or isinstance(data["pid"], bool) or data["pid"] <= 0:
        raise DomainError("runtime_state_corrupt", f"{path.name} has an invalid pid.", 409)
    return data


def validate_runtime_url(url) -> str:
    if not isinstance(url, str) or not url:
        raise DomainError("invalid_runtime_url", "Managed runtime URL must be a non-empty string.", 502)
    parsed = urlparse(url)
    if (parsed.scheme not in ("http", "https") or not parsed.hostname
            or parsed.username or parsed.password or parsed.query or parsed.fragment):
        raise DomainError("invalid_runtime_url",
                          "Managed runtime URL must be an HTTP(S) origin without credentials, query, or fragment.", 502)
    try:
        parsed.port
    except ValueError:
        raise DomainError("invalid_runtime_url", "Managed runtime URL has a malformed port.", 502) from None
    if parsed.scheme == "http" and parsed.hostname not in ("127.0.0.1", "localhost", "::1"):
        raise DomainError("invalid_runtime_url",
                          "Managed executors are reached over loopback HTTP or an SSH tunnel.", 502)
    return url.rstrip("/")


def _read_token(token_file) -> str:
    path = Path(token_file)
    try:
        token = path.read_text(encoding="utf-8").strip()
    except OSError:
        raise DomainError("invalid_credentials", f"Credential file {path.name} is unreadable.", 500) from None
    if len(token) < 32 or any(char.isspace() for char in token):
        raise DomainError("invalid_credentials", f"Credential file {path.name} is invalid.", 500)
    return token


async def _request_json(method: str, url: str, token_file, path: str, body: dict | None = None,
                        *, timeout_s: float = 5.0) -> dict:
    """Authenticated JSON request; transport failures raise, HTTP errors return ``error``."""
    base = validate_runtime_url(url)
    token = _read_token(token_file)
    try:
        async with httpx.AsyncClient(
                base_url=base,
                headers={"Authorization": "Bearer " + token},
                timeout=httpx.Timeout(float(timeout_s), connect=min(2.0, float(timeout_s))),
                trust_env=False,
                follow_redirects=False) as client:
            response = await client.request(method, path, json=body)
    except httpx.HTTPError as exc:
        raise DomainError("runtime_unreachable",
                          f"Managed executor is unreachable ({type(exc).__name__}).", 503) from None
    try:
        payload = response.json()
    except ValueError:
        raise DomainError("runtime_invalid_response",
                          f"Managed executor returned HTTP {response.status_code} without JSON.", 502) from None
    if not isinstance(payload, dict):
        raise DomainError("runtime_invalid_response",
                          f"Managed executor returned HTTP {response.status_code} with a non-object body.", 502)
    if response.is_error and "error" not in payload:
        return {"error": {"code": "http_error", "message": f"Managed executor returned HTTP {response.status_code}."}}
    return payload


async def probe_health(url: str, token_file, *, timeout_s: float = HEALTH_TIMEOUT_S) -> dict | None:
    """Return the health object only when it proves this service; ``None`` otherwise.

    A JSON object that does not declare ``service == "piperx-middleware"`` is a
    foreign service on that port, not an executor, so it never counts as
    healthy.  Field-level identity (instance/profile/mode/backend) is checked by
    the caller with :meth:`RuntimeManager._verify_health`.
    """
    try:
        payload = await _request_json("GET", url, token_file, "/health", timeout_s=timeout_s)
    except DomainError as exc:
        if exc.code == "invalid_credentials":
            raise
        return None
    if not isinstance(payload, dict) or "error" in payload:
        return None
    if payload.get("service") != "piperx-middleware":
        return None
    return payload


# ``os.kill(pid, 0)`` is *not* a liveness probe on Windows: CPython maps zero (unlike CTRL_C/CTRL_BREAK events)
# to TerminateProcess, so probing an executor with it would kill it.
# OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION) + GetExitCodeProcess is the
# non-destructive equivalent.  Process handles are pointer-sized on 64-bit
# Windows, so every signature must declare HANDLE explicitly; the ctypes
# default ``c_int`` restype would truncate a 64-bit handle.
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
STILL_ACTIVE = 259
ERROR_ACCESS_DENIED = 5
ERROR_INVALID_PARAMETER = 87

_kernel32_library = None


def _is_windows() -> bool:
    return os.name == "nt"


def _windows_last_error() -> int:
    import ctypes
    return ctypes.get_last_error()


def _windows_kernel32():
    """Bind kernel32 once with explicit, 64-bit-safe signatures."""
    global _kernel32_library
    if _kernel32_library is None:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        kernel32.GetExitCodeProcess.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL
        _kernel32_library = kernel32
    return _kernel32_library


def _windows_pid_alive(pid: int) -> bool:
    """Windows process-existence probe that never signals the process.

    Only ``ERROR_INVALID_PARAMETER`` (no such process) proves the pid is dead.
    ``ERROR_ACCESS_DENIED`` means the process exists but is protected, and any
    other unexplained failure stays conservative too: reporting a live executor
    as dead would drop its runtime record and start a duplicate owner.
    """
    import ctypes

    kernel32 = _windows_kernel32()
    handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        error = _windows_last_error()
        if error == ERROR_ACCESS_DENIED:
            return True
        if error == ERROR_INVALID_PARAMETER:
            return False
        return True
    try:
        exit_code = ctypes.c_ulong()  # == wintypes.DWORD
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
            return True  # unknown failure: conservatively alive
        return exit_code.value == STILL_ACTIVE
    finally:
        kernel32.CloseHandle(handle)


def _posix_pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _pid_alive(pid) -> bool:
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        return False
    if _is_windows():
        return _windows_pid_alive(pid)
    return _posix_pid_alive(pid)


def _port_open(url: str) -> bool:
    parsed = urlparse(url)
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        with socket.create_connection((parsed.hostname, port), timeout=0.5):
            return True
    except OSError:
        return False


def _state_instance_id(state: dict) -> str | None:
    """Extract the executor instance identity from a ``/v1/state`` document.

    The service reports its instance identity inside ``observation_identity``
    and may also expose it top-level; both are accepted, but a malformed or
    contradictory document yields ``None`` so callers must refuse instead of
    treating it as an idle executor.
    """
    candidates: list[str] = []
    top = state.get("instance_id")
    if top is not None:
        if not isinstance(top, str) or not top:
            return None
        candidates.append(top)
    nested = state.get("observation_identity")
    if nested is not None:
        if not isinstance(nested, dict):
            return None
        value = nested.get("instance_id")
        if value is not None:
            if not isinstance(value, str) or not value:
                return None
            candidates.append(value)
    if not candidates or len(set(candidates)) != 1:
        return None
    return candidates[0]


class RuntimeManager:
    """Start, reuse and gracefully release the managed executor for one profile."""

    def __init__(self, base, mode: str, target: str = LOCAL_TARGET, simulation_backend: str = "mujoco", *,
                 startup_timeout_s: float = STARTUP_TIMEOUT_S,
                 drain_timeout_s: float = DRAIN_TIMEOUT_S,
                 shutdown_grace_s: float = SHUTDOWN_GRACE_S,
                 poll_interval_s: float = POLL_INTERVAL_S,
                 health_timeout_s: float = HEALTH_TIMEOUT_S,
                 lock_timeout_s: float = LOCK_TIMEOUT_S,
                 ssh_timeout_s: float = SSH_TIMEOUT_S,
                 ssh_binary: str | None = None):
        self.base = Path(base).expanduser()
        self.mode = validate_mode(mode)
        self.target = validate_target(target)
        if self.mode == "simulation" and simulation_backend not in SIMULATION_BACKENDS:
            raise ProfileError("invalid_backend", "Simulation backend must be 'mujoco' or 'sim'.")
        self.simulation_backend = simulation_backend
        self.startup_timeout_s = float(startup_timeout_s)
        self.drain_timeout_s = float(drain_timeout_s)
        self.shutdown_grace_s = float(shutdown_grace_s)
        self.poll_interval_s = float(poll_interval_s)
        self.health_timeout_s = float(health_timeout_s)
        self.lock_timeout_s = float(lock_timeout_s)
        self.ssh_timeout_s = float(ssh_timeout_s)
        self.ssh_binary = ssh_binary
        self._profile: Profile | None = None
        self._connection: RuntimeConnection | None = None
        self._process = None
        self._remote: dict | None = None
        self._lifecycle = asyncio.Lock()

    # -- introspection -----------------------------------------------------
    @property
    def profile(self) -> Profile | None:
        return self._profile

    @property
    def connection(self) -> RuntimeConnection | None:
        return self._connection

    @property
    def backend(self) -> str:
        if self._profile is not None:
            return self._profile.backend
        return self.simulation_backend if self.mode == "simulation" else REAL_BACKEND

    def remotes(self) -> list[dict]:
        return list_remotes(self.base)

    def save_remote(self, name: str, ssh_host: str) -> dict:
        return save_remote(self.base, name, ssh_host)

    def expected_profile_id(self) -> str:
        return profile_id(self.mode, self.target)

    # -- lifecycle ---------------------------------------------------------
    async def ensure(self) -> RuntimeConnection:
        """Start or reuse the verified executor for this profile."""
        async with self._lifecycle:
            profile = await asyncio.to_thread(
                ensure_profile, self.base, self.mode, self.target,
                simulation_backend=self.simulation_backend)
            self._profile = profile
            if self._connection is not None and await self._connection_healthy(self._connection):
                return self._connection
            if self.target != LOCAL_TARGET:
                if self._connection is not None or self._remote is not None:
                    # The cached remote session no longer answers a health check:
                    # reconnect explicitly, discarding only the local tunnel and
                    # credential cache.  No release request is sent, so a local
                    # reconnect can never stop or re-drive the remote executor.
                    await self._discard_remote_session(self._connection)
                return await self._ensure_remote(profile)
            lock = await self._acquire_startup_lock(profile)
            try:
                reused = await self._reuse_verified(profile)
                if reused is not None:
                    return reused
                return await self._start_local(profile)
            finally:
                await asyncio.to_thread(lock.close)

    async def release(self, connection: RuntimeConnection, shutdown_owned: bool = True) -> dict:
        """Release a connection: shared detaches, owned drains then shuts down."""
        if not isinstance(connection, RuntimeConnection):
            raise DomainError("invalid_request", "release() requires a RuntimeConnection.", 422)
        async with self._lifecycle:
            profile = await self._profile_for_release()
            if (Path(connection.profile_root) != profile.root
                    or connection.mode != self.mode or connection.target != self.target):
                raise DomainError("foreign_connection",
                                  "Connection does not belong to this managed profile; refusing to release it.", 409)
            if self.target != LOCAL_TARGET:
                return await self._release_remote(connection, shutdown_owned)
            if not connection.owned or not shutdown_owned:
                self._forget(connection)  # an explicit detach ends this manager's ownership claim
                return {"status": "detached", "owned": bool(connection.owned), "shutdown": False,
                        "instance_id": connection.instance_id, "url": connection.url}
            health = await probe_health(connection.url, connection.model_token_file, timeout_s=self.health_timeout_s)
            if health is None:
                raise DomainError("runtime_unreachable",
                                  "Managed executor did not answer a health check; refusing to shut down an "
                                  "unverified process.", 409)
            record = await asyncio.to_thread(read_runtime_json, profile.runtime_path)
            expected = {"instance_id": connection.instance_id}
            if record is not None and record.get("instance_id") == connection.instance_id:
                expected["pid"] = record["pid"]
            self._verify_health(health, expected, profile, code="instance_mismatch")
            drain = await self._drain(connection)
            if not drain["drained"]:
                return {"status": "busy", "drained": False, "shutdown": False,
                        "instance_id": connection.instance_id, "active_job_id": drain.get("active_job_id")}
            result = await _request_json("POST", connection.url, connection.model_token_file, "/v1/shutdown",
                                         {"expected_instance_id": connection.instance_id}, timeout_s=10.0)
            if isinstance(result, dict) and "error" in result:
                error = result["error"] if isinstance(result["error"], dict) else {}
                raise DomainError(error.get("code") or "shutdown_failed",
                                  error.get("message") or "Managed executor refused shutdown.", 409)
            stopped = await self._await_exit(profile, connection, health)
            self._forget(connection)
            return {"status": "released" if stopped else "shutdown_unconfirmed", "stopped": stopped,
                    "drained": True, "shutdown": True, "instance_id": connection.instance_id, "server": result}

    async def shutdown(self, *, shutdown_owned: bool = True) -> dict:
        """Release the connection this manager currently holds, if any."""
        connection = self._connection
        if connection is None:
            return {"status": "idle", "shutdown": False}
        return await self.release(connection, shutdown_owned=shutdown_owned)

    def ensure_sync(self) -> RuntimeConnection:
        return asyncio.run(self.ensure())

    def release_sync(self, connection: RuntimeConnection, shutdown_owned: bool = True) -> dict:
        return asyncio.run(self.release(connection, shutdown_owned=shutdown_owned))

    def shutdown_sync(self, *, shutdown_owned: bool = True) -> dict:
        return asyncio.run(self.shutdown(shutdown_owned=shutdown_owned))

    async def __aenter__(self) -> "RuntimeManager":
        await self.ensure()
        return self

    async def __aexit__(self, *exc_info) -> None:
        await self.shutdown()

    # -- local startup -----------------------------------------------------
    async def _acquire_startup_lock(self, profile: Profile) -> ProcessLock:
        deadline = time.monotonic() + self.lock_timeout_s
        while True:
            try:
                return await asyncio.to_thread(ProcessLock, profile.startup_lock_path)
            except RuntimeError:
                if time.monotonic() >= deadline:
                    raise DomainError(
                        "runtime_busy",
                        f"Another frontend is starting the {self.mode} executor for this profile; retry shortly.",
                        409) from None
                await asyncio.sleep(0.05)

    async def _reuse_verified(self, profile: Profile) -> RuntimeConnection | None:
        record = await asyncio.to_thread(read_runtime_json, profile.runtime_path)
        if record is None:
            return None
        self._verify_record(record, profile)
        url = validate_runtime_url(record["url"])
        health = await probe_health(url, profile.model_token_file, timeout_s=self.health_timeout_s)
        if health is not None:
            self._verify_health(health, record, profile)
            connection = RuntimeConnection(
                url=url,
                model_token_file=profile.model_token_file,
                operator_token_file=profile.operator_token_file,
                instance_id=record["instance_id"],
                owned=False,
                profile_root=profile.root,
                profile_id=profile.profile_id,
                mode=self.mode,
                target=self.target,
            )
            self._connection = connection
            return connection
        if _pid_alive(record["pid"]):
            if _port_open(url):
                raise DomainError(
                    "runtime_identity_mismatch",
                    f"A process is listening on {url} but did not prove the expected executor identity; "
                    "refusing to start a second executor or to shut it down.", 409)
            raise DomainError(
                "runtime_unhealthy",
                f"The recorded executor (pid {record['pid']}) is not serving; refusing to start a duplicate. "
                f"Stop that process manually, or remove the stale {profile.runtime_path.name} if the pid was "
                "recycled, and retry.", 409)
        await self._remove_stale_record(profile, record["instance_id"])
        return None

    async def _start_local(self, profile: Profile) -> RuntimeConnection:
        process = await asyncio.to_thread(self._spawn_process, self._local_command(profile), profile.log_path)
        try:
            record, health = await self._await_runtime(profile, process)
        except BaseException:
            await asyncio.to_thread(self._terminate_process, process)
            raise
        connection = RuntimeConnection(
            url=validate_runtime_url(record["url"]),
            model_token_file=profile.model_token_file,
            operator_token_file=profile.operator_token_file,
            instance_id=record["instance_id"],
            owned=True,
            profile_root=profile.root,
            profile_id=profile.profile_id,
            mode=self.mode,
            target=self.target,
        )
        self._process = process
        self._connection = connection
        return connection

    def _local_command(self, profile: Profile) -> list[str]:
        return [sys.executable, "-m", "piperx_middleware.cli",
                "--root", str(profile.root), "serve", "--managed"]

    def _child_env(self) -> dict:
        env = dict(os.environ)
        source_root = str(Path(__file__).resolve().parents[1])
        existing = env.get("PYTHONPATH")
        env["PYTHONPATH"] = source_root + (os.pathsep + existing if existing else "")
        env["PYTHONUNBUFFERED"] = "1"
        return env

    def _spawn_process(self, command: list[str], log_path: Path):
        log_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        descriptor = os.open(str(log_path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            return subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=descriptor,
                stderr=descriptor,
                cwd=str(log_path.parent),
                env=self._child_env(),
                start_new_session=True,
                close_fds=True,
            )
        finally:
            os.close(descriptor)

    @staticmethod
    def _terminate_process(process) -> None:
        if process is None:
            return
        try:
            if process.poll() is not None:
                process.wait(timeout=0)  # already exited: reap it, never leave a zombie
                return
            process.terminate()
            process.wait(timeout=3)
        except Exception:
            try:
                process.kill()
                process.wait(timeout=3)
            except Exception:
                pass

    async def _await_runtime(self, profile: Profile, process) -> tuple[dict, dict]:
        deadline = time.monotonic() + self.startup_timeout_s
        while True:
            if process is not None and process.poll() is not None:
                raise DomainError(
                    "runtime_start_failed",
                    f"Managed executor exited during startup (exit code {process.returncode}); "
                    f"see {profile.log_path.name}.", 503)
            record = await asyncio.to_thread(read_runtime_json, profile.runtime_path)
            if record is not None:
                self._verify_record(record, profile)
                url = validate_runtime_url(record["url"])
                health = await probe_health(url, profile.model_token_file, timeout_s=self.health_timeout_s)
                if health is not None:
                    self._verify_health(health, record, profile)
                    return record, health
            if time.monotonic() >= deadline:
                raise DomainError(
                    "runtime_start_failed",
                    f"Managed executor did not publish a verified runtime record within "
                    f"{self.startup_timeout_s:g}s; see {profile.log_path.name}.", 503)
            await asyncio.sleep(self.poll_interval_s)

    # -- verification ------------------------------------------------------
    def _verify_record(self, record: dict, profile: Profile) -> None:
        if (record.get("profile_id") != profile.profile_id
                or record.get("mode") != self.mode
                or record.get("backend") != profile.backend):
            raise DomainError(
                "runtime_identity_mismatch",
                f"{profile.runtime_path.name} does not match the {self.mode}/{self.target} profile identity; "
                "refusing to reuse it.", 409)

    def _verify_health(self, health: dict, record: dict, profile: Profile, *, code: str = "runtime_identity_mismatch") -> None:
        """Require an exact identity match; a missing field is a mismatch.

        The published record and the live ``/health`` response must agree on the
        instance, the process, the profile, the mode and the backend.  Treating
        an absent field as acceptable would let a foreign or half-upgraded
        service keep an executor alive and be driven as the real robot.
        """
        instance_id = health.get("instance_id")
        if not isinstance(instance_id, str) or not instance_id or instance_id != record.get("instance_id"):
            raise DomainError(code, "Executor health identity does not match the published instance; refusing.", 409)
        record_pid = record.get("pid")
        if isinstance(record_pid, int) and not isinstance(record_pid, bool):
            health_pid = health.get("process_id")
            if not isinstance(health_pid, int) or isinstance(health_pid, bool) or health_pid != record_pid:
                raise DomainError(code, "Executor health process does not match the published instance; refusing.", 409)
        for key, expected in (("profile_id", profile.profile_id), ("mode", self.mode), ("backend", profile.backend)):
            if health.get(key) != expected:
                raise DomainError(code, f"Executor health {key} does not match the expected profile; refusing.", 409)

    async def _connection_healthy(self, connection: RuntimeConnection) -> bool:
        if (Path(connection.profile_root) != (self._profile.root if self._profile else Path(connection.profile_root))
                or connection.mode != self.mode or connection.target != self.target):
            return False
        try:
            health = await probe_health(connection.url, connection.model_token_file, timeout_s=self.health_timeout_s)
        except DomainError:
            return False
        if health is None or health.get("instance_id") != connection.instance_id:
            return False
        if health.get("profile_id") != connection.profile_id:
            return False
        if health.get("mode") != self.mode or health.get("backend") != self.backend:
            return False
        return True

    # -- graceful release --------------------------------------------------
    async def _profile_for_release(self) -> Profile:
        if self._profile is None:
            self._profile = await asyncio.to_thread(
                ensure_profile, self.base, self.mode, self.target,
                simulation_backend=self.simulation_backend)
        return self._profile

    async def _drain(self, connection: RuntimeConnection) -> dict:
        """Wait for the current accepted action to finish, verifying identity.

        Every poll must carry a complete state document whose instance identity
        still matches the connection.  A malformed or foreign state is refused
        instead of being read as "idle", so a process restart while waiting can
        never make this manager shut down an executor it does not own.
        """
        started = time.monotonic()
        deadline = started + self.drain_timeout_s
        while True:
            state = await _request_json("GET", connection.url, connection.model_token_file, "/v1/state",
                                        timeout_s=self.health_timeout_s)
            if not isinstance(state, dict):
                raise DomainError("runtime_invalid_response",
                                  "Executor state is not a JSON object; refusing to treat it as idle.", 502)
            if "error" in state:
                error = state["error"] if isinstance(state["error"], dict) else {}
                raise DomainError(error.get("code") or "state_failed",
                                  error.get("message") or "Cannot verify executor idleness; refusing to shut down.",
                                  409)
            instance_id = _state_instance_id(state)
            if instance_id is None:
                raise DomainError(
                    "runtime_invalid_response",
                    "Executor state did not carry a valid instance identity; refusing to treat it as idle.", 502)
            if instance_id != connection.instance_id:
                raise DomainError(
                    "instance_mismatch",
                    "Executor instance changed while draining; refusing to shut down a foreign process.", 409)
            if "active_job_id" not in state:
                raise DomainError(
                    "runtime_invalid_response",
                    "Executor state is missing active_job_id; refusing to treat malformed state as idle.", 502)
            active = state["active_job_id"]
            if active is not None and (not isinstance(active, str) or not active):
                raise DomainError(
                    "runtime_invalid_response",
                    "Executor state reported an invalid active_job_id; refusing to treat it as idle.", 502)
            if active is None:
                return {"drained": True, "active_job_id": None, "waited_s": round(time.monotonic() - started, 3)}
            if time.monotonic() >= deadline:
                return {"drained": False, "active_job_id": active}
            await asyncio.sleep(self.poll_interval_s)

    async def _await_exit(self, profile: Profile, connection: RuntimeConnection, health: dict) -> bool:
        pid = health.get("process_id")
        if not isinstance(pid, int) or isinstance(pid, bool):
            record = await asyncio.to_thread(read_runtime_json, profile.runtime_path)
            pid = (record or {}).get("pid")
        deadline = time.monotonic() + self.shutdown_grace_s
        while True:
            alive = _pid_alive(pid)
            record = await asyncio.to_thread(read_runtime_json, profile.runtime_path)
            if record is not None and record.get("instance_id") != connection.instance_id:
                record = None  # a new executor already published its own record; leave it alone
            if not alive or record is None:
                await self._remove_record_if_matches(profile, connection.instance_id)
                return True
            if time.monotonic() >= deadline:
                return False
            await asyncio.sleep(self.poll_interval_s)

    async def _remove_stale_record(self, profile: Profile, instance_id: str) -> None:
        try:
            current = await asyncio.to_thread(read_runtime_json, profile.runtime_path)
        except DomainError:
            current = None
        if current is None or current.get("instance_id") != instance_id:
            return
        try:
            profile.runtime_path.unlink(missing_ok=True)
        except OSError:
            raise DomainError("runtime_state_corrupt",
                              f"Stale {profile.runtime_path.name} could not be removed; refusing to start a duplicate.",
                              409) from None

    async def _remove_record_if_matches(self, profile: Profile, instance_id: str) -> None:
        try:
            current = await asyncio.to_thread(read_runtime_json, profile.runtime_path)
        except DomainError:
            return
        if current is None or current.get("instance_id") != instance_id or _pid_alive(current.get("pid")):
            return
        try:
            profile.runtime_path.unlink(missing_ok=True)
        except OSError:
            pass

    def _forget(self, connection: RuntimeConnection) -> None:
        if self._connection is not None and self._connection.instance_id == connection.instance_id:
            self._connection = None
        self._process = None

    # -- remote target -----------------------------------------------------
    async def _ensure_remote(self, profile: Profile) -> RuntimeConnection:
        remote = await asyncio.to_thread(get_remote, self.base, self.target)
        if remote is None:
            raise DomainError("unknown_remote", f"No saved remote named {self.target!r}; add it before switching.", 404)
        client = SshRuntimeClient(remote["ssh_host"], ssh_binary=self.ssh_binary,
                                  request_timeout_s=self.ssh_timeout_s)
        request = {
            "version": PROTOCOL_VERSION,
            "action": "start",
            "mode": self.mode,
            "target": self.target,
            "simulation_backend": self.simulation_backend,
            "profile_id": profile.profile_id,
        }
        response = await asyncio.to_thread(client.request, request)
        runtime = response.get("runtime") if isinstance(response, dict) else None
        if not isinstance(runtime, dict):
            raise DomainError("ssh_protocol", "Remote host did not return runtime identity.", 502)
        instance_id = runtime.get("instance_id")
        remote_profile = runtime.get("profile_id")
        backend = runtime.get("backend")
        port = runtime.get("port")
        if not isinstance(instance_id, str) or not instance_id:
            raise DomainError("ssh_protocol", "Remote host returned an invalid instance identity.", 502)
        if not isinstance(remote_profile, str) or not remote_profile:
            raise DomainError("ssh_protocol", "Remote host returned an invalid profile identity.", 502)
        if runtime.get("mode") != self.mode:
            raise DomainError("runtime_identity_mismatch", "Remote executor mode does not match the requested mode.", 409)
        if not isinstance(backend, str) or (
                self.mode == "simulation" and backend not in SIMULATION_BACKENDS) or (
                self.mode == "real" and backend != REAL_BACKEND):
            raise DomainError("mode_restriction", "Remote executor backend does not match the requested mode.", 409)
        if not isinstance(port, int) or isinstance(port, bool) or not 1 <= port <= 65535:
            raise DomainError("ssh_protocol", "Remote host returned an invalid port.", 502)

        cache_dir, model_path, operator_path = await asyncio.to_thread(
            write_credential_cache, profile.root, runtime.get("model_token"), runtime.get("operator_token"))
        local_port = free_local_port()
        tunnel = None
        try:
            tunnel = await asyncio.to_thread(client.open_tunnel, local_port, port,
                                             stderr_path=profile.root / "tunnel.log")
            url = f"http://127.0.0.1:{local_port}"
            await self._await_remote_health(url, model_path, instance_id, remote_profile, backend)
        except BaseException:
            await asyncio.to_thread(SshRuntimeClient.close_tunnel, tunnel)
            await asyncio.to_thread(cleanup_credential_cache, cache_dir)
            raise
        connection = RuntimeConnection(
            url=url,
            model_token_file=model_path,
            operator_token_file=operator_path,
            instance_id=instance_id,
            owned=bool(runtime.get("owned")),
            profile_root=profile.root,
            profile_id=remote_profile,
            mode=self.mode,
            target=self.target,
        )
        self._remote = {"client": client, "tunnel": tunnel, "cache_dir": cache_dir, "runtime": runtime}
        self._connection = connection
        return connection

    async def _await_remote_health(self, url: str, token_file, instance_id: str,
                                   remote_profile: str, backend: str) -> dict:
        deadline = time.monotonic() + self.startup_timeout_s
        while True:
            health = await probe_health(url, token_file, timeout_s=self.health_timeout_s)
            if health is not None:
                if health.get("instance_id") != instance_id:
                    raise DomainError("runtime_identity_mismatch",
                                      "Tunnel health identity does not match the SSH bootstrap response.", 409)
                if health.get("profile_id") != remote_profile:
                    raise DomainError("runtime_identity_mismatch",
                                      "Tunnel profile identity does not match the SSH bootstrap response.", 409)
                if health.get("backend") != backend or health.get("mode") != self.mode:
                    raise DomainError("runtime_identity_mismatch",
                                      "Tunnel backend or mode does not match the SSH bootstrap response.", 409)
                return health
            if time.monotonic() >= deadline:
                raise DomainError("runtime_start_failed",
                                  "SSH tunnel did not expose a verified executor within "
                                  f"{self.startup_timeout_s:g}s.", 503)
            await asyncio.sleep(self.poll_interval_s)

    async def _discard_remote_session(self, connection: RuntimeConnection | None = None) -> None:
        """Close only the local half of a remote session.

        Closes the SSH tunnel and deletes the cached credentials, and
        deliberately sends **no** request to the remote host: discarding a local
        session must never stop, start, or re-drive the robot.  Used by an
        explicit reconnect, by a shared detach and by a release that has been
        confirmed by the remote host.
        """
        session, self._remote = self._remote, None
        if session is not None:
            await asyncio.to_thread(SshRuntimeClient.close_tunnel, session["tunnel"])
            await asyncio.to_thread(cleanup_credential_cache, session["cache_dir"])
        if connection is not None:
            self._forget(connection)

    @staticmethod
    def _remote_unconfirmed(connection: RuntimeConnection, released: dict) -> dict:
        """An honest non-success result for a release the host did not confirm."""
        return {"status": "shutdown_unconfirmed",
                "shutdown": released.get("shutdown") is True,
                "stopped": False,
                "instance_id": connection.instance_id,
                "remote": released}

    async def _release_remote(self, connection: RuntimeConnection, shutdown_owned: bool) -> dict:
        """Release a remote session without ever assuming an unconfirmed stop.

        Only two outcomes may end the local session: the remote host confirms
        this exact instance stopped (``released`` with ``stopped`` true and a
        matching identity), or this is an explicit shared detach.  A busy,
        absent, unconfirmed, malformed or failed release keeps the tunnel, the
        credential cache and the manager connection so ``/status``, ``/stop``
        and an explicit retry stay usable.
        """
        session = self._remote
        if session is None or session["runtime"].get("instance_id") != connection.instance_id:
            raise DomainError("foreign_connection",
                              "No matching remote session is open; refusing to release this connection.", 409)
        if not connection.owned or not shutdown_owned:
            # Explicit shared detach: the remote executor keeps running and is
            # never contacted; only this manager's tunnel and cache go away.
            result = {"status": "detached", "shutdown": False, "instance_id": connection.instance_id}
            await self._discard_remote_session(connection)
            return result
        request = {
            "version": PROTOCOL_VERSION,
            "action": "release",
            "mode": self.mode,
            "instance_id": connection.instance_id,
            "profile_id": connection.profile_id,
            "shutdown_owned": True,
        }
        response = await asyncio.to_thread(session["client"].request, request)
        if isinstance(response, dict) and response.get("status") == "error":
            error = response.get("error") if isinstance(response.get("error"), dict) else {}
            raise DomainError(error.get("code") or "remote_error",
                              error.get("message") or "Remote host refused the release request.", 409)
        released = response.get("released") if isinstance(response, dict) else None
        if not isinstance(released, dict):
            raise DomainError(
                "ssh_protocol",
                "Remote host did not return a release result; the executor state is unconfirmed and the "
                "session is kept for status or retry.", 502)

        status = released.get("status")
        if "instance_id" in released and released.get("instance_id") != connection.instance_id:
            raise DomainError(
                "instance_mismatch",
                "Remote release result names a different executor instance; refusing to treat this "
                "session as released.", 409)
        if status == "busy":
            return {"status": "busy", "shutdown": False, "drained": False,
                    "instance_id": connection.instance_id, "remote": released}
        if status == "detached":
            result = {"status": "detached", "shutdown": False,
                      "instance_id": connection.instance_id, "remote": released}
            await self._discard_remote_session(connection)
            return result
        if status in ("absent", "shutdown_unconfirmed"):
            # "absent" means the remote host has no runtime record at all: it can
            # never prove that *this* instance stopped, so it is not success.
            return self._remote_unconfirmed(connection, released)
        if status != "released":
            raise DomainError(
                "ssh_protocol",
                "Remote host returned an unrecognized release result; the executor state is unconfirmed "
                "and the session is kept for status or retry.", 502)
        if released.get("instance_id") != connection.instance_id:
            # A missing identity is a mismatch, never a confirmation.
            raise DomainError(
                "instance_mismatch",
                "Remote release result does not name this exact instance; refusing to treat it as "
                "released.", 409)
        if released.get("stopped") is not True or released.get("shutdown") is False:
            # Shutdown was requested but the host did not confirm the stop (or
            # reported a contradictory document): the session is retained.
            return self._remote_unconfirmed(connection, released)
        result = {"status": "released", "shutdown": True, "stopped": True, "drained": True,
                  "instance_id": connection.instance_id, "remote": released}
        await self._discard_remote_session(connection)
        return result


__all__ = [
    "RuntimeManager", "read_runtime_json", "validate_runtime_url", "probe_health",
    "HEALTH_TIMEOUT_S", "STARTUP_TIMEOUT_S", "DRAIN_TIMEOUT_S", "SHUTDOWN_GRACE_S", "POLL_INTERVAL_S",
]
