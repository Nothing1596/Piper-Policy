"""Managed SSH transport for remote executors.

Design rules enforced here:

* host key verification is never disabled, relaxed, or bypassed;
* the remote command is a fixed literal (``piper-robot host --stdio``) and all
  parameters travel as one JSON document on stdin, so no code path, profile
  name, or parameter is ever interpolated into a remote shell;
* credentials returned by a remote host travel only inside the SSH pipe and are
  cached in 0600 files under a 0700 directory; they are never logged;
* the module never installs software or edits SSH configuration.

``host_main`` is the remote side of the protocol: it reads exactly one JSON
request from stdin and writes exactly one JSON response to stdout.  The local
side is :class:`SshRuntimeClient`.
"""
from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Sequence
from urllib.parse import urlparse

from .models import DomainError
from .profiles import (
    LOCAL_TARGET,
    load_profile,
    validate_mode,
    validate_ssh_host,
)

SSH_BINARY = "ssh"
REMOTE_EXECUTABLE = "piper-robot"
REMOTE_SUBCOMMAND = ("host", "--stdio")
REMOTE_ARGV = (REMOTE_EXECUTABLE,) + REMOTE_SUBCOMMAND
PROTOCOL_VERSION = 1

# BatchMode is required because stdin carries the bootstrap document, so an
# interactive prompt could not be answered.  Host key checking stays strict.
SSH_OPTIONS = (
    "-o", "BatchMode=yes",
    "-o", "StrictHostKeyChecking=yes",
    "-o", "ConnectTimeout=10",
    "-o", "ServerAliveInterval=15",
    "-o", "ServerAliveCountMax=3",
)

Runner = Callable[[Sequence[str], str, float], Any]
PopenFactory = Callable[..., Any]


def setup_guidance(host: str) -> str:
    """Clear operator guidance; this never performs the setup itself."""
    return (
        f"Set up SSH access to {host!r} manually: install the key in your agent, "
        f"then verify the host key once (for example `ssh {host}`) so it is present in known_hosts. "
        "This tool never disables host key verification and never installs software. "
        "The remote host must already have a matching piper-robot package installed."
    )


def free_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def build_ssh_argv(host: str,
                   *,
                   remote_argv: Sequence[str] | None = REMOTE_ARGV,
                   local_forward: tuple[int, int] | None = None,
                   ssh_binary: str | None = None,
                   options: Sequence[str] = SSH_OPTIONS) -> list[str]:
    """Build the ssh argument vector for a bootstrap or a tunnel.

    ``host`` is validated and passed as a single argument after ``--``; the
    remote argv is a fixed literal tuple.  When ``local_forward`` is given the
    command becomes a loopback ``-L`` tunnel (``-N``).
    """
    host = validate_ssh_host(host)
    if local_forward is not None and remote_argv is not None:
        raise DomainError("invalid_ssh_command", "A tunnel and a remote command are mutually exclusive.")
    argv = [ssh_binary or SSH_BINARY, *options]
    if local_forward is not None:
        local_port, remote_port = local_forward
        for port in (local_port, remote_port):
            if not isinstance(port, int) or isinstance(port, bool) or not 1 <= port <= 65535:
                raise DomainError("invalid_ssh_command", "Tunnel ports must be integers between 1 and 65535.")
        argv += ["-N", "-o", "ExitOnForwardFailure=yes",
                 "-L", f"127.0.0.1:{local_port}:127.0.0.1:{remote_port}"]
    argv += ["--", host]
    if local_forward is None and remote_argv is not None:
        argv += [str(part) for part in remote_argv]
    return argv


def parse_host_response(stdout: str | bytes | None) -> dict:
    """Parse exactly one JSON object; never echo unexpected content."""
    if isinstance(stdout, bytes):
        try:
            text = stdout.decode("utf-8")
        except UnicodeDecodeError:
            raise DomainError("ssh_protocol", "Remote host response is not valid UTF-8.") from None
    elif isinstance(stdout, str):
        text = stdout
    else:
        raise DomainError("ssh_protocol", "Remote host returned no response.")
    stripped = text.strip()
    if not stripped:
        raise DomainError("ssh_protocol", "Remote host returned an empty response.")
    try:
        payload, end = json.JSONDecoder().raw_decode(stripped)
    except ValueError:
        raise DomainError("ssh_protocol", "Remote host response is not valid JSON.") from None
    if stripped[end:].strip():
        raise DomainError("ssh_protocol", "Remote host returned more than one JSON document.")
    if not isinstance(payload, dict):
        raise DomainError("ssh_protocol", "Remote host response must be a JSON object.")
    return payload


def _sanitize(text: str, limit: int = 300) -> str:
    """Strip token-shaped runs and control characters before an error message."""
    cleaned = []
    run = 0
    for char in text:
        if char.isalnum() or char in "-_":
            run += 1
            cleaned.append(char)
            continue
        if run >= 32:
            cleaned = cleaned[:len(cleaned) - run] + ["<redacted>"]
        run = 0
        cleaned.append(" " if char in "\r\n\t" else char)
    if run >= 32:
        cleaned = cleaned[:len(cleaned) - run] + ["<redacted>"]
    return "".join(cleaned).strip()[:limit]


def _default_runner(argv: Sequence[str], input_text: str, timeout: float) -> subprocess.CompletedProcess:
    return subprocess.run(list(argv), input=input_text, capture_output=True, text=True,
                          timeout=timeout, check=False)


class SshRuntimeClient:
    """One saved remote host; block calls are meant to run in a worker thread."""

    def __init__(self, host: str, *,
                 ssh_binary: str | None = None,
                 request_timeout_s: float = 30.0,
                 tunnel_timeout_s: float = 15.0,
                 runner: Runner | None = None,
                 popen: PopenFactory | None = None):
        self.host = validate_ssh_host(host)
        self.ssh_binary = ssh_binary or SSH_BINARY
        self.request_timeout_s = float(request_timeout_s)
        self.tunnel_timeout_s = float(tunnel_timeout_s)
        self._runner = runner
        self._popen = popen

    def request(self, payload: dict, *, timeout_s: float | None = None) -> dict:
        """Send one JSON document to ``piper-robot host --stdio`` over SSH."""
        argv = build_ssh_argv(self.host, ssh_binary=self.ssh_binary)
        document = json.dumps(payload, separators=(",", ":"), allow_nan=False)
        runner = self._runner or _default_runner
        try:
            completed = runner(argv, document, float(timeout_s or self.request_timeout_s))
        except FileNotFoundError:
            raise DomainError("ssh_unavailable", f"The ssh client is not available. {setup_guidance(self.host)}") from None
        except subprocess.TimeoutExpired:
            raise DomainError("ssh_timeout", f"SSH bootstrap to {self.host!r} timed out. {setup_guidance(self.host)}") from None
        returncode = getattr(completed, "returncode", None)
        if returncode != 0:
            stderr = getattr(completed, "stderr", "") or ""
            hint = " Host key verification failed; verify the host key manually." if "host key" in stderr.lower() else ""
            detail = _sanitize(stderr)
            message = f"SSH command to {self.host!r} failed with exit code {returncode}.{hint}"
            if detail:
                message += f" ssh said: {detail}"
            raise DomainError("ssh_failed", message + " " + setup_guidance(self.host))
        response = parse_host_response(getattr(completed, "stdout", None))
        if response.get("status") == "error":
            error = response.get("error") if isinstance(response.get("error"), dict) else {}
            raise DomainError(error.get("code") or "remote_error",
                              _sanitize(str(error.get("message") or "Remote host reported an error.")))
        return response

    def open_tunnel(self, local_port: int, remote_port: int, *, stderr_path: Path | None = None) -> Any:
        """Start ``ssh -N -L 127.0.0.1:local:127.0.0.1:remote`` and return the process."""
        argv = build_ssh_argv(self.host, remote_argv=None,
                              local_forward=(local_port, remote_port), ssh_binary=self.ssh_binary)
        popen = self._popen or subprocess.Popen
        stderr = subprocess.DEVNULL
        handle = None
        if stderr_path is not None:
            handle = os.open(str(stderr_path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            stderr = handle
        try:
            process = popen(argv, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                            stderr=stderr, start_new_session=True, close_fds=True)
        except FileNotFoundError:
            raise DomainError("ssh_unavailable", f"The ssh client is not available. {setup_guidance(self.host)}") from None
        finally:
            if handle is not None:
                os.close(handle)
        return process

    @staticmethod
    def close_tunnel(process: Any, *, timeout_s: float = 5.0) -> None:
        if process is None:
            return
        poll = getattr(process, "poll", None)
        try:
            if poll is not None and poll() is not None:
                return
        except Exception:
            return
        try:
            process.terminate()
            wait = getattr(process, "wait", None)
            if wait is not None:
                wait(timeout=timeout_s)
        except Exception:
            try:
                process.kill()
            except Exception:
                pass


# --------------------------------------------------------------------------
# remote credential cache (0600 files inside a 0700 temporary directory)
# --------------------------------------------------------------------------

def _secret_value(name: str, value: Any) -> str:
    if not isinstance(value, str) or len(value) < 32 or any(char.isspace() for char in value):
        raise DomainError("ssh_protocol", f"Remote host returned an invalid {name}.")
    return value


def _write_secret(path: Path, value: str) -> Path:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(value + "\n")
    return path


def write_credential_cache(profile_root: Path, model_token: Any, operator_token: Any) -> tuple[Path, Path, Path]:
    """Cache remote credentials in a fresh 0700 directory; values stay in files."""
    model_token = _secret_value("model token", model_token)
    operator_token = _secret_value("operator token", operator_token)
    directory = Path(tempfile.mkdtemp(prefix="remote-session-", dir=str(profile_root)))
    try:
        os.chmod(directory, 0o700)
        model_path = _write_secret(directory / "model.token", model_token)
        operator_path = _write_secret(directory / "operator.token", operator_token)
    except BaseException:
        cleanup_credential_cache(directory)
        raise
    return directory, model_path, operator_path


def cleanup_credential_cache(directory: Path | None) -> None:
    if directory is None:
        return
    path = Path(directory)
    for name in ("model.token", "operator.token"):
        try:
            (path / name).unlink(missing_ok=True)
        except OSError:
            pass
    shutil.rmtree(path, ignore_errors=True)


def _read_secret_file(path: Path) -> str:
    try:
        value = Path(path).read_text(encoding="utf-8").strip()
    except OSError:
        raise DomainError("invalid_credentials", f"Credential file {Path(path).name} is unreadable.") from None
    return _secret_value(Path(path).name, value)


# --------------------------------------------------------------------------
# remote side: piper-robot host --stdio
# --------------------------------------------------------------------------

def _default_remote_root():
    from .cli import default_root
    return default_root()


def _require_mode(request: dict) -> str:
    """Remote mode is never guessed; an unspecified mode is an invalid request."""
    if request.get("mode") is None:
        raise DomainError("invalid_request", "Host request must specify mode 'simulation' or 'real'.", 422)
    return validate_mode(request["mode"])


def _host_start(request: dict, root) -> dict:
    from .managed_runtime import RuntimeManager, read_runtime_json
    mode = _require_mode(request)
    simulation_backend = request.get("simulation_backend", "mujoco")
    manager = RuntimeManager(root, mode, LOCAL_TARGET, simulation_backend)
    connection = manager.ensure_sync()
    profile = manager.profile
    record = read_runtime_json(profile.runtime_path)
    port = urlparse(connection.url).port
    return {
        "version": PROTOCOL_VERSION,
        "status": "ok",
        "runtime": {
            "url": connection.url,
            "instance_id": connection.instance_id,
            "profile_id": connection.profile_id,
            "mode": connection.mode,
            "target": LOCAL_TARGET,
            "backend": profile.backend,
            "pid": (record or {}).get("pid"),
            "port": port,
            "owned": bool(connection.owned),
            "model_token": _read_secret_file(connection.model_token_file),
            "operator_token": _read_secret_file(connection.operator_token_file),
        },
    }


def _host_status(request: dict, root) -> dict:
    from .managed_runtime import read_runtime_json
    mode = _require_mode(request)
    profile = load_profile(root, mode, LOCAL_TARGET)
    record = read_runtime_json(profile.runtime_path)
    if record is None:
        return {"version": PROTOCOL_VERSION, "status": "ok", "runtime": None}
    health = probe_health_sync(record["url"], profile.model_token_file)
    runtime = {
        "url": record.get("url"),
        "instance_id": record.get("instance_id"),
        "profile_id": record.get("profile_id"),
        "mode": record.get("mode"),
        "backend": record.get("backend"),
        "pid": record.get("pid"),
        "target": LOCAL_TARGET,
        "healthy": health is not None,
    }
    if health is not None:
        runtime["health_instance_id"] = health.get("instance_id")
    return {"version": PROTOCOL_VERSION, "status": "ok", "runtime": runtime}


def _host_release(request: dict, root) -> dict:
    from .interaction_types import RuntimeConnection
    from .managed_runtime import RuntimeManager, read_runtime_json
    mode = _require_mode(request)
    profile = load_profile(root, mode, LOCAL_TARGET)
    record = read_runtime_json(profile.runtime_path)
    if record is None:
        return {"version": PROTOCOL_VERSION, "status": "ok",
                "released": {"status": "absent", "shutdown": False}}
    # The caller must name the exact instance *and* profile it believes it owns;
    # a missing identity is refused rather than assumed to mean "whatever is
    # running now", which could shut down a different operator's run.
    expected_instance = request.get("instance_id")
    if not isinstance(expected_instance, str) or not expected_instance:
        raise DomainError("invalid_request", "Release request must name the expected instance.", 422)
    expected_profile = request.get("profile_id")
    if not isinstance(expected_profile, str) or not expected_profile:
        raise DomainError("invalid_request", "Release request must name the expected profile.", 422)
    if expected_instance != record.get("instance_id") or expected_profile != record.get("profile_id"):
        raise DomainError("instance_mismatch", "Executor identity changed; nothing was shut down.")
    connection = RuntimeConnection(
        url=record["url"],
        model_token_file=profile.model_token_file,
        operator_token_file=profile.operator_token_file,
        instance_id=record["instance_id"],
        owned=True,
        profile_root=profile.root,
        profile_id=record["profile_id"],
        mode=record.get("mode", mode),
        target=LOCAL_TARGET,
    )
    manager = RuntimeManager(root, mode, LOCAL_TARGET)
    released = manager.release_sync(connection, shutdown_owned=bool(request.get("shutdown_owned", True)))
    return {"version": PROTOCOL_VERSION, "status": "ok", "released": released}


def handle_host_request(request: Any, *, root=None) -> dict:
    """Dispatch one already-parsed host request and return one response object."""
    root = Path(root).expanduser() if root is not None else _default_remote_root()
    if not isinstance(request, dict):
        return _host_error("invalid_request", "Host request must be a JSON object.")
    action = request.get("action")
    try:
        if action == "ping":
            return {"version": PROTOCOL_VERSION, "status": "ok"}
        if action == "start":
            return _host_start(request, root)
        if action == "status":
            return _host_status(request, root)
        if action == "release":
            return _host_release(request, root)
        return _host_error("unsupported_action", "Unsupported host action.")
    except DomainError as exc:
        return _host_error(exc.code, exc.message)
    except Exception as exc:  # never leak a traceback or a path into the pipe
        return _host_error("host_error", f"Remote host failed: {type(exc).__name__}")


def _host_error(code: str, message: str) -> dict:
    return {"version": PROTOCOL_VERSION, "status": "error",
            "error": {"code": code or "host_error", "message": _sanitize(str(message))}}


def host_main(root=None, *, base=None, stdin=None, stdout=None) -> int:
    """Read one JSON request from stdin, write one JSON response, return exit code.

    ``root``/``base`` select the per-user application data directory; ``stdin``
    and ``stdout`` are injectable for tests.
    """
    if root is None:
        root = base
    source = stdin if stdin is not None else sys.stdin
    sink = stdout if stdout is not None else sys.stdout
    try:
        raw = source.read()
    except Exception:
        raw = ""
    try:
        request = json.loads(raw)
    except ValueError:
        response = _host_error("invalid_request", "Host request is not valid JSON.")
    else:
        response = handle_host_request(request, root=root)
    sink.write(json.dumps(response, separators=(",", ":"), allow_nan=False) + "\n")
    sink.flush()
    return 0 if response.get("status") == "ok" else 1


def probe_health_sync(url: str, token_file) -> dict | None:
    from .managed_runtime import probe_health
    import asyncio
    return asyncio.run(probe_health(url, token_file))


__all__ = [
    "SSH_BINARY", "REMOTE_EXECUTABLE", "REMOTE_SUBCOMMAND", "REMOTE_ARGV", "PROTOCOL_VERSION",
    "SSH_OPTIONS", "SshRuntimeClient", "setup_guidance", "free_local_port", "build_ssh_argv",
    "parse_host_response", "write_credential_cache", "cleanup_credential_cache",
    "handle_host_request", "host_main",
]
