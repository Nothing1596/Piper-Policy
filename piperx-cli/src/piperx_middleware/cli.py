import argparse
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
import secrets
import sys
import time

from . import cli_views
from .client import RobotClient
from .models import Settings


def default_root():
    if os.name == "nt":
        return Path(os.environ["LOCALAPPDATA"]) / "PiperXMiddleware"
    return Path.home() / ".local" / "state" / "piperx-middleware"


def initialize(root: Path, backend: str, **overrides):
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    config = root / "config.json"
    if config.exists():
        raise SystemExit(f"Already initialized: {config}; existing configuration and tokens preserved.")
    # Exclusive creation; never overwrite identity or rotate a live credential implicitly.
    for name in ("model.token", "operator.token"):
        path = root / name
        if not path.exists():
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "w") as f:
                f.write(secrets.token_urlsafe(48) + "\n")
    platform_can = {"can_interface": "socketcan", "can_channel": "can0"} if sys.platform.startswith("linux") else {}
    settings = Settings(backend=backend, data_dir=root, **(platform_can | overrides))
    config.write_text(settings.model_dump_json(indent=2), encoding="utf-8")
    print(f"Created {config}. Profile: {settings.control_profile}; token values are never printed.")


def positive_seconds(text):
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a number: {text!r}") from None
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("must be a positive finite number of seconds")
    return value


def positive_count(text):
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not an integer: {text!r}") from None
    if value < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def record_limit(text):
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not an integer: {text!r}") from None
    if not 1 <= value <= 500:
        raise argparse.ArgumentTypeError("must be between 1 and 500")
    return value


INSPECTION_PATHS = {"status": "/v1/state", "tools": "/v1/capabilities", "params": "/v1/parameters"}
INSPECTION_COMMANDS = frozenset(INSPECTION_PATHS) | {"monitor", "calls", "jobs"}


def configured_url(root: Path):
    config = root / "config.json"
    if not config.is_file():
        return None
    try:
        settings = Settings.model_validate_json(config.read_text(encoding="utf-8-sig"))
    except Exception:
        print(f"warning: ignoring unreadable local config {config}", file=sys.stderr)
        return None
    host = "[::1]" if settings.host == "::1" else settings.host
    return f"http://{host}:{settings.port}"


def inspection_client(args, root: Path) -> RobotClient:
    # Explicit flag beats PIPERX_URL, which beats the local config default.
    url = args.url or os.environ.get("PIPERX_URL") or configured_url(root)
    token_file = args.token_file
    if token_file is None and os.environ.get("PIPERX_TOKEN_FILE"):
        token_file = Path(os.environ["PIPERX_TOKEN_FILE"])
    return RobotClient.from_env(url=url, token_file=token_file, root=root)


def error_text(result) -> str:
    error = result.get("error") if isinstance(result, dict) else None
    if isinstance(error, dict):
        code = error.get("code") or "error"
        message = error.get("message") or ""
        return f"{code}: {message}" if message else str(code)
    return "request failed"


def run_inspection(args, root: Path):
    try:
        client = inspection_client(args, root)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from None
    try:
        if args.command == "monitor":
            run_monitor(args, client)
            return
        path = (f"/v1/{args.command}?limit={args.limit}" if args.command in ("calls", "jobs")
                else INSPECTION_PATHS[args.command])
        result = client.call("GET", path)
        failed = "error" in result
        if args.json:
            print(json.dumps(result, indent=2, ensure_ascii=False))
        elif failed:
            print("error: " + error_text(result), file=sys.stderr)
        else:
            render = {"status": cli_views.render_status, "tools": cli_views.render_tools,
                      "params": cli_views.render_params, "calls": cli_views.render_calls,
                      "jobs": cli_views.render_jobs}[args.command]
            print(render(result))
        if failed:
            raise SystemExit(1)
    finally:
        client.close()


def run_monitor(args, client: RobotClient):
    remaining = args.count
    try:
        while True:
            result = client.call("GET", "/v1/state")
            if args.json:
                print(json.dumps(result, ensure_ascii=False), flush=True)
            else:
                stamp = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
                print(f"--- {stamp} ---")
                if "error" in result:
                    print("error: " + error_text(result), file=sys.stderr)
                else:
                    print(cli_views.render_status(result))
                    print(flush=True)
            if "error" in result:
                raise SystemExit(1)
            if remaining is not None:
                remaining -= 1
                if remaining <= 0:
                    return
            time.sleep(args.interval)
    except KeyboardInterrupt:
        # Ctrl-C only stops the local watch; nothing is sent to the robot.
        return


def main():
    parser = argparse.ArgumentParser(description="PiperX shared robot executor and operator controls")
    parser.add_argument("--root", type=Path, default=default_root())
    parser.add_argument("--url", help="Executor origin for the console and inspection commands "
                        "(env PIPERX_URL; default: local config, else http://127.0.0.1:8765)")
    parser.add_argument("--token-file", type=Path, help="Bearer token file for the console and inspection commands "
                        "(env PIPERX_TOKEN_FILE; default: <root>/model.token)")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("shell", help="Interactive robot console (also the default with no command)")
    init = sub.add_parser("init")
    init.add_argument("--backend", choices=["sim", "agx", "mujoco"], default="sim")
    init.add_argument("--port", type=int, default=8765)
    init.add_argument("--can-interface", choices=["socketcan", "agx_cando"])
    init.add_argument("--can-channel")
    init.add_argument("--sdk-root", type=Path)
    init.add_argument("--cando-source", type=Path)
    init.add_argument("--tcp-offset-m", type=float, nargs=3, metavar=("X", "Y", "Z"))
    init.add_argument("--tcp-offset-rpy-deg", type=float, nargs=3, metavar=("ROLL", "PITCH", "YAW"))
    init.add_argument("--profile", choices=["direct", "calibration"], default="direct")
    init.add_argument("--read-only", action="store_true")
    serve = sub.add_parser("serve")
    serve.add_argument("--config", type=Path)
    mode = serve.add_mutually_exclusive_group()
    mode.add_argument("--allow-motion", action="store_true", help="Enable configured action tools")
    mode.add_argument("--read-only", action="store_true", help="Run observations only")
    sub.add_parser("state")
    connect = sub.add_parser("connect")
    connect.add_argument("--reconnect", action="store_true", help="Release and reopen an idle CAN connection")
    connect.add_argument("--device-id", help="Explicit enumerated CAN device ID")
    sub.add_parser("disconnect", help="Release CAN without exiting the server")
    sub.add_parser("stop")
    sub.add_parser("shutdown", help="Release CAN and exit an idle executor gracefully; not a robot stop")
    arm = sub.add_parser("arm", help="Operator-only bounded control window; does not itself move")
    arm.add_argument("--seconds", type=int, default=120)
    arm.add_argument("--radius-deg", type=float, default=3)
    arm.add_argument("--gripper", action="store_true")
    status = sub.add_parser("status", help="Read-only summary: connection, robot, joints, gripper, TCP, diagnostics")
    status.add_argument("--json", action="store_true", help="Print the raw server response as JSON")
    monitor = sub.add_parser("monitor", help="Poll status repeatedly; Ctrl-C stops the watch only, never the robot")
    monitor.add_argument("--interval", type=positive_seconds, default=2.0,
                         help="Seconds between polls; positive and finite (default 2)")
    monitor.add_argument("--count", type=positive_count, default=None,
                         help="Stop after this many polls (default: until Ctrl-C)")
    monitor.add_argument("--json", action="store_true", help="Print one JSON state object per line")
    tools = sub.add_parser("tools", help="List MCP tools from executor capabilities")
    tools.add_argument("--json", action="store_true", help="Print the raw server response as JSON")
    params = sub.add_parser("params", help="Show configured parameters and reported state, distinctly labeled")
    params.add_argument("--json", action="store_true", help="Print the raw server response as JSON")
    calls = sub.add_parser("calls", help="Recorded executor calls, newest first; not proof of physical completion")
    calls.add_argument("--limit", type=record_limit, default=50, metavar="1..500")
    calls.add_argument("--json", action="store_true", help="Print the raw server response as JSON")
    jobs = sub.add_parser("jobs", help="Known jobs, newest first; job status is the execution result")
    jobs.add_argument("--limit", type=record_limit, default=50, metavar="1..500")
    jobs.add_argument("--json", action="store_true", help="Print the raw server response as JSON")
    from .command_cli import COMMANDS, CONTROL_COMMANDS, add_commands, run
    for name in CONTROL_COMMANDS:
        sub.choices[name].add_argument("--output", type=Path)
        sub.choices[name].add_argument("--json", action="store_true", help="JSON is the default")
    add_commands(sub)
    args = parser.parse_args()
    if args.command not in INSPECTION_COMMANDS | COMMANDS | CONTROL_COMMANDS | {None, "shell"} and (args.url is not None or args.token_file is not None):
        parser.error("--url and --token-file apply only to inspection commands; use --root for local control commands")
    root = args.root.resolve()
    if args.command in COMMANDS | CONTROL_COMMANDS:
        run(args, root)
        return
    if args.command in (None, "shell"):
        import asyncio
        from .console import run_console
        try:
            asyncio.run(run_console(args, root))
        except KeyboardInterrupt:
            pass
        return
    if args.command == "init":
        overrides = {key: getattr(args, key) for key in ("can_interface", "can_channel", "sdk_root", "cando_source", "tcp_offset_m", "tcp_offset_rpy_deg")
                     if getattr(args, key) is not None}
        initialize(root, args.backend, port=args.port, control_profile=args.profile, allow_motion=not args.read_only, **overrides)
        return
    if args.command in INSPECTION_COMMANDS:
        # Read-only inspection: no local config or SDK/backend required.
        run_inspection(args, root)
        return
    config = getattr(args, "config", None) or root / "config.json"
    settings = Settings.model_validate_json(config.read_text(encoding="utf-8-sig"))
    if settings.host not in ("127.0.0.1", "::1", "localhost"):
        raise SystemExit("Use a loopback listener with an authenticated SSH tunnel or HTTPS reverse proxy.")
    if args.command == "serve":
        import uvicorn
        from .backends import SimBackend
        from .http_api import create_app
        from .process_lock import ProcessLock
        from .service import RobotService
        lock = ProcessLock(settings.data_dir / "executor.lock")
        service = backend = None
        pid_path = root / "executor.pid"
        try:
            if args.allow_motion:
                settings.allow_motion = True
            if args.read_only:
                settings.allow_motion = False
            if settings.backend == "agx":
                if settings.can_interface == "socketcan" and not sys.platform.startswith("linux"):
                    raise SystemExit("SocketCAN requires Linux. Use a remote executor or --backend sim on this platform.")
                if settings.can_interface == "agx_cando" and os.name != "nt":
                    raise SystemExit("The bundled CANDO DLL requires Windows x64 Python (including WOA emulation).")
                from .agx_backend import AgxBackend
                backend = AgxBackend(settings)
            elif settings.backend == "mujoco":
                from .mujoco_backend import MujocoBackend
                backend = MujocoBackend(asset_path=settings.simulation_asset, seed=settings.simulation_seed)
            else:
                backend = SimBackend()
            service = RobotService(backend, settings)
            server = None
            def request_exit():
                server.should_exit = True
            app = create_app(service, (root / "model.token").read_text().strip(),
                             (root / "operator.token").read_text().strip(), on_shutdown=request_exit)
            server = uvicorn.Server(uvicorn.Config(app, host=settings.host, port=settings.port,
                                                 access_log=False, workers=1))
            temporary_pid = root / f"executor-{os.getpid()}.pid.tmp"
            temporary_pid.write_text(str(os.getpid()), encoding="ascii")
            temporary_pid.replace(pid_path)
            server.run()
        finally:
            # A failed cleanup must propagate; do not advertise a released lock.
            if service is not None:
                service.close()
            elif backend is not None:
                backend.close()
            if pid_path.exists() and pid_path.read_text().strip() == str(os.getpid()):
                pid_path.unlink()
            lock.close()
        return


if __name__ == "__main__":
    main()
