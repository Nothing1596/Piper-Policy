"""Scriptable CLI commands. stdout is JSON; progress/request IDs go to stderr."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import sys
import time
from urllib.parse import quote
import uuid

from .client import RobotClient

COMMANDS = {"move-joints", "gripper", "control-mode", "move-to", "move-by", "rotate", "move-linear",
            "preview", "execute", "job", "request", "events", "doctor", "devices", "estop",
            "clear-estop", "configure-runtime", "sim-fault", "model", "manual", "limits"}
CONTROL_COMMANDS = {"state", "connect", "disconnect", "stop", "shutdown", "arm"}


def add_commands(sub):
    from .cli import positive_seconds, record_limit
    for name in sorted(COMMANDS - {"model"}):
        p = sub.add_parser(name, help="JSON CLI: " + name)
        p.add_argument("--output", type=Path, help="Save the JSON result (exclusive create)")
        p.add_argument("--json", action="store_true", help="JSON is the default for this command")
        if name in {"move-joints", "gripper", "control-mode", "move-to", "move-by", "rotate", "move-linear", "execute"}:
            p.add_argument("--request-id", help="Stable ID for retries; generated once if omitted")
            p.add_argument("--wait", action="store_true", help="Wait for measured completion")
            p.add_argument("--wait-timeout", type=positive_seconds, default=130)
        if name in {"move-joints", "control-mode", "move-to", "move-by", "rotate", "move-linear"}:
            p.add_argument("--speed", type=int, default=5)
        if name in {"move-joints", "gripper", "control-mode", "move-to", "move-by", "rotate", "move-linear"}:
            p.add_argument("--timeout", type=positive_seconds, default=30)
        if name == "move-joints": p.add_argument("joints", type=float, nargs=6)
        if name == "gripper":
            p.add_argument("width", type=float)
            p.add_argument("--effort", type=float, default=.5)
            p.add_argument('--completion',choices=['width','bilateral_contact'],default='width',
                           help='MuJoCo closing grasp only: require fresh stable bilateral finger contact')
        if name in {"move-to", "move-by", "rotate", "move-linear"}: p.add_argument("vector", type=float, nargs=3)
        if name == "move-to": p.add_argument("--rpy", type=float, nargs=3)
        if name == "move-by": p.add_argument("--frame", choices=["base", "tcp"], default="base")
        if name == "move-linear":
            p.add_argument("--step", type=positive_seconds, default=.002)
            p.add_argument("--native", action="store_true", help="Use controller MOVE_L on agx; simulator still uses explicit synthetic samples")
        if name == "preview": p.add_argument("file", type=Path, help="JSON command object, not a request envelope")
        if name == "limits": p.add_argument("--refresh", action="store_true", help="Operator: actively query six controller limits")
        if name == "execute": p.add_argument("plan_id")
        if name in {"job", "request"}:
            p.add_argument("id")
            p.add_argument("--wait", action="store_true")
            p.add_argument("--wait-timeout", type=positive_seconds, default=130)
        if name == "events":
            p.add_argument("--after", type=int, default=0)
            p.add_argument("--limit", type=record_limit, default=100)
        if name == "configure-runtime":
            p.add_argument("--tcp-m", type=float, nargs=3)
            p.add_argument("--tcp-rpy", type=float, nargs=3)
            p.add_argument("--payload", choices=["empty", "half", "full"])
            p.add_argument("--collision-rating", type=int)
            p.add_argument("--joint-acc", type=float)
        if name == "sim-fault": p.add_argument("fault", choices=["none", "stale", "collision", "tracking", "driver", "teaching"])
        if name == "manual": p.add_argument("expression", help="Named literal MCP call, e.g. robot_status()")
    p = sub.add_parser("model", help="Local/OpenAI-compatible model configuration and bounded tool calling")
    ms = p.add_subparsers(dest="model_command", required=True)
    for name in ("show", "set", "list", "check", "run"):
        m = ms.add_parser(name)
        m.add_argument("--output", type=Path)
        m.add_argument("--endpoint")
        m.add_argument("--name")
        if name == "set":
            m.add_argument("--api-key-env", default="PIPERX_MODEL_API_KEY")
        if name == "run":
            m.add_argument("prompt")
            m.add_argument("--allow-motion", action="store_true", help="Allow motion proposals in this bounded turn")


def finish(result, output=None):
    rendered = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as f:
            f.write(rendered + "\n")
    print(rendered)
    failed = bool(result.get("error")) or result.get("status") in {"failed", "cancelled", "outcome_unknown", "error", "incomplete"}
    if failed:
        raise SystemExit(1)


def wait_job(client, result, timeout):
    deadline = time.monotonic() + timeout
    while result.get("status") in {"accepted", "running"}:
        if time.monotonic() >= deadline:
            return {"status": "outcome_unknown", "job_id": result.get("job_id"), "request_id": result.get("request_id"),
                    "error": {"code": "wait_timeout", "message": "Local wait expired; job may still run. Query original ID or stop explicitly."}}
        time.sleep(.05)
        result = client.call("GET", "/v1/jobs/" + quote(result["job_id"], safe=""))
    return result


def run(args, root):
    from .cli import configured_url
    if args.output and args.output.exists():
        raise SystemExit("Output exists; choose a new path before submitting a command.")
    try:
        if args.command == "model":
            finish(asyncio.run(run_model(args, root)), args.output)
            return
        if args.command == "manual":
            finish(asyncio.run(run_manual(args, root)), args.output)
            return
        operator = args.command in {"estop", "clear-estop", "configure-runtime", "sim-fault", "arm"} or (args.command == "limits" and args.refresh)
        token = args.token_file or (Path(os.environ["PIPERX_TOKEN_FILE"]) if os.environ.get("PIPERX_TOKEN_FILE") else
                                   root / ("operator.token" if operator else "model.token"))
        url = args.url or os.environ.get("PIPERX_URL") or configured_url(root) or "http://127.0.0.1:8765"
        client = RobotClient(url, token)
        try:
            result = dispatch(args, client)
        finally:
            client.close()
        finish(result, args.output)
    except (ValueError, OSError, RuntimeError) as exc:
        finish({"error": {"code": "cli_error", "message": str(exc)}}, None)
    except KeyboardInterrupt:
        finish({"status": "outcome_unknown", "error": {"code": "interrupted", "message": "Local wait cancelled. Query the emitted request ID; no automatic replay or stop."}})


def dispatch(a, c):
    from .models import MoveRequest, PrimitiveRequest, PreviewRequest, PrimitivePreviewRequest, RuntimeParameters, ExecuteRequest
    command = a.command
    if command == "state": return c.call("GET", "/v1/state")
    if command == "connect": return c.call("POST", "/v1/connect", {"reconnect": a.reconnect, **({"device_id": a.device_id} if a.device_id else {})})
    if command in {"stop", "disconnect"}: return c.call("POST", "/v1/"+command)
    if command == "arm": return c.call("POST", "/operator/control-window", {"duration_s": a.seconds, "joint_radius_deg": a.radius_deg, "allow_gripper": a.gripper})
    if command == "shutdown":
        health = c.call("GET", "/health")
        if "instance_id" not in health: return health
        return c.call("POST", "/v1/shutdown", {"expected_instance_id": health["instance_id"]})
    reads = {"doctor": "/v1/diagnostics", "devices": "/v1/devices",
             "events": f"/v1/events?after={getattr(a, 'after', 0)}&limit={getattr(a, 'limit', 100)}"}
    if command in reads: return c.call("GET", reads[command])
    if command == "limits": return c.call("POST", "/operator/query-limits") if a.refresh else c.call("GET", "/v1/limits")
    if command in {"estop", "clear-estop"}: return c.call("POST", "/operator/" + command)
    if command == "sim-fault": return c.call("POST", "/operator/sim-fault", {"fault": a.fault})
    if command == "configure-runtime":
        body = {k: v for k,v in dict(tcp_offset_m=a.tcp_m, tcp_offset_rpy_deg=a.tcp_rpy,
                payload=a.payload, collision_rating=a.collision_rating, joint_acc_rad_s2=a.joint_acc).items() if v is not None}
        return c.call("PATCH", "/operator/parameters", RuntimeParameters(**body).model_dump(exclude_none=True))
    if command in {"job", "request"}:
        result = c.call("GET", "/v1/" + ("jobs/" if command == "job" else "requests/") + quote(a.id, safe=""))
        return wait_job(c, result, a.wait_timeout) if a.wait else result
    if command == "preview":
        body = {"command": json.loads(a.file.read_text(encoding="utf-8-sig"))}
        primitive = body["command"].get("kind") in {"move_to", "move_by", "rotate", "set_gripper", "move_linear"}
        schema = PrimitivePreviewRequest if primitive else PreviewRequest
        body = schema.model_validate(body).model_dump()
        return c.call("POST", "/v1/primitives/preview" if primitive else "/v1/preview", body)
    rid = a.request_id or str(uuid.uuid4())
    print("request_id=" + rid, file=sys.stderr, flush=True)
    if command == "execute":
        result = c.call("POST", "/v1/execute", ExecuteRequest(plan_id=a.plan_id, request_id=rid).model_dump())
    else:
        values = {"kind": command.replace("-", "_"), "timeout_s": a.timeout}
        if hasattr(a, "speed"): values["speed_percent"] = a.speed
        if command == "move-joints": values.update(kind="joint", joints_deg=a.joints)
        if command == "gripper": values.update(width_m=a.width, effort_protocol=a.effort,completion=a.completion)
        if command in {"move-to", "move-linear"}: values["xyz_m"] = a.vector
        if command == "move-to": values["rpy_deg"] = a.rpy
        if command == "move-by": values.update(delta_m=a.vector, frame=a.frame)
        if command == "rotate": values["rpy_deg"] = a.vector
        if command == "move-linear": values.update(step_m=a.step, native_controller=a.native)
        primitive = command in {"move-to", "move-by", "rotate", "move-linear"}
        schema = PrimitiveRequest if primitive else MoveRequest
        body = schema(command=values, request_id=rid).model_dump()
        result = c.call("POST", "/v1/primitives" if primitive else "/v1/move", body)
    if a.wait: result = wait_job(c, result, a.wait_timeout)
    if isinstance(result.get("error"), dict): result.setdefault("request_id", rid)
    return result


def bridge_options(args, root):
    from .cli import configured_url
    return (args.url or os.environ.get("PIPERX_URL") or configured_url(root) or "http://127.0.0.1:8765",
            args.token_file or Path(os.environ.get("PIPERX_TOKEN_FILE", str(root / "model.token"))))


async def invoke_wait(bridge, name, arguments, tools):
    spec = next((t for t in tools if t["name"] == name), None)
    if spec is None:
        return {"error": {"code": "unknown_tool", "message": "Unknown MCP tool: " + name}}
    args = dict(arguments)
    if "request_id" in spec["inputSchema"].get("required", []):
        args.setdefault("request_id", str(uuid.uuid4()))
        print("request_id=" + args["request_id"], file=sys.stderr, flush=True)
    result = await bridge.call(name, args)
    deadline = time.monotonic() + min(float(args.get("timeout_s", 30)), 300) + 5
    while result.get("status") in {"accepted", "running"}:
        if time.monotonic() > deadline:
            return {"status": "outcome_unknown", "job_id": result.get("job_id"),
                    "request_id": args.get("request_id"), "error": "Local job wait expired"}
        await asyncio.sleep(.05)
        result = await bridge.rest("GET", "/v1/jobs/" + quote(result["job_id"], safe=""))
    return result


async def run_manual(args, root):
    from .console_bridge import MCPBridge
    from .manual_parser import parse_manual_command
    name, values = parse_manual_command(args.expression)
    async with MCPBridge(*bridge_options(args, root)) as bridge:
        return await invoke_wait(bridge, name, values, bridge.tools)


async def run_model(args, root):
    from .model_agent import load_model_config, save_model_config, check_model, run_turn, validate_endpoint, resolve_api_key
    config = load_model_config(root)
    if args.endpoint: config.endpoint = validate_endpoint(args.endpoint)
    if args.name: config.model = args.name
    if args.model_command == "set":
        config.api_key_env = args.api_key_env
        save_model_config(root, config)
        return {"status": "configured", "endpoint": config.endpoint, "model": config.model, "api_key_env": config.api_key_env}
    if args.model_command == "show":
        return {"endpoint": config.endpoint, "model": config.model, "api_key_env": config.api_key_env}
    if args.model_command == "check": return await check_model(config)
    if args.model_command == "list":
        import httpx
        headers = {}
        key = resolve_api_key(config)
        if key: headers["Authorization"] = "Bearer " + key
        async with httpx.AsyncClient(timeout=10, trust_env=False, follow_redirects=False) as client:
            r = await client.get(validate_endpoint(config.endpoint) + "/models", headers=headers)
            if r.status_code != 200: raise RuntimeError(f"Model list HTTP {r.status_code}")
            return r.json()
    from .console_bridge import MCPBridge
    report = {"status": "running", "requested_model": config.model, "endpoint": config.endpoint,
              "physical_validation": False, "tool_results": [], "model_events": []}
    async with MCPBridge(*bridge_options(args, root)) as bridge:
        initial = await bridge.rest("GET", "/v1/state")
        if "error" in initial: return initial
        report["simulation"] = initial.get("backend") == "sim"
        import copy
        tools = copy.deepcopy(bridge.tools if args.allow_motion else [t for t in bridge.tools if t["name"] in {"robot_status", "robot_diagnostics"}])
        for tool in tools:
            schema = tool["inputSchema"]
            if "request_id" in schema.get("required", []):
                schema["required"].remove("request_id")
                schema.get("properties", {}).pop("request_id", None)
            if tool["name"] == "robot_status":
                # This adapter already awaits jobs and supplies their receipts. Avoid
                # tempting the model to invent a request ID for an ordinary observation.
                schema["properties"] = {}
                schema["required"] = []
                schema["additionalProperties"] = False
                tool["description"] = "Read CURRENT robot state and capabilities. Call with empty arguments {}. This is not a historical job lookup."
        messages = [{"role": "system", "content": "You control a PiperX through tools. /no_think\nRead robot_status before actions. "
                    "Follow the user's exact targets. Do not invent success, measurements or calibration. "
                    "An accepted command is not completed. Tool invocation waits for a measured job result. "
                    "If a tool fails or outcome is unknown, stop proposing motion and explain it. "
                    "Never change safety settings. All simulation results are synthetic, not physical success."},
                    {"role": "user", "content": args.prompt + "\n/no_think"}]
        def emit(text):
            report["model_events"].append(text)
            print(text, file=sys.stderr, flush=True)
        halted = False
        async def invoke(name, values):
            nonlocal halted
            if halted and name not in {"robot_status", "robot_diagnostics", "robot_stop"}:
                return {"error": {"code": "turn_halted", "message": "An earlier tool failed; motion is disabled for this turn."}}
            result = await invoke_wait(bridge, name, values, bridge.tools)
            report["tool_results"].append({"name": name, "arguments": values, "result": result})
            if result.get("error") or result.get("status") in {"failed", "cancelled", "outcome_unknown"}:
                halted = True
            return result
        try:
            report["answer"] = await run_turn(config, messages, tools, invoke, emit)
            report["status"] = "failed" if halted else "incomplete" if report["answer"].startswith("Model reached maximum") else "succeeded"
        except Exception as exc:
            report.update(status="error", error={"code": "model_error", "message": str(exc)})
        report["final_state"] = await bridge.rest("GET", "/v1/state")
        report["messages"] = messages
    return report
