"""Interactive console for PiperX middleware 0.4.0.

Provides a local terminal interface with prompt_toolkit support, slash commands,
safe AST-based /manual tool execution, MCP tool schema validation, robot job polling,
and integration with OpenAI-compatible language models.
"""

from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import math
import os
import re
import secrets
import sys
import time
from pathlib import Path
from typing import Any, Callable

from jsonschema import Draft202012Validator

try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.completion import Completer, Completion
    from prompt_toolkit.history import InMemoryHistory
    from prompt_toolkit.patch_stdout import patch_stdout
    HAVE_PROMPT_TOOLKIT = True
except ImportError:
    HAVE_PROMPT_TOOLKIT = False
    Completer = object  # type: ignore

from . import cli_views
from .cli import configured_url
from .console_bridge import MCPBridge
from .manual_parser import parse_manual_command
from .model_agent import (
    ModelConfig,
    check_model,
    load_model_config,
    resolve_api_key,
    run_turn,
    save_model_config,
    validate_endpoint,
)


SLASH_COMMANDS = [
    ("/help", "Show available commands and usage guide"),
    ("/status", "Show read-only robot and connection telemetry"),
    ("/connect", "Enumerate and connect to device on executor host"),
    ("/disconnect", "Release CAN connection on executor"),
    ("/model", "Inspect or configure LLM endpoint and credentials"),
    ("/manual", "Execute an MCP tool directly: /manual TOOL(arg=val)"),
    ("/tools", "List exposed MCP tools or inspect a schema"),
    ("/calls", "Show recorded executor calls log"),
    ("/jobs", "Show recorded jobs and execution results"),
    ("/params", "Show configured parameters and reported state"),
    ("/stop", "Cancel local tasks and issue robot stop request (not hardware emergency stop)"),
    ("/quit", "Exit console (does not stop robot or disconnect CAN)"),
]

SYSTEM_INSTRUCTION = (
    "You are controlling a Piper robot arm via MCP tools. "
    "Read robot_status first to inspect actual robot state and connection diagnostics. "
    "Joint angles are six absolute values in degrees; gripper opening width is in metres. "
    "Use robot_connect to establish feedback and robot_disconnect to release the connection. "
    "Tools return job_id upon acceptance; acceptance is not completion. Query robot_status(job_id=...) "
    "for completion. Unknown action outcomes must never be replayed. "
    "move_to, move_by, and rotate target the TCP in base-frame metres and RPY degrees via the fixed numerical IK solver "
    "(joint-space endpoint move, not Cartesian straight line, no collision checking). "
    "set_gripper sets the opening width in metres."
)


class ConsoleCompleter(Completer):
    """Prompt-toolkit completer for slash commands, tool names, and parameter names."""

    def __init__(self, controller: ConsoleController):
        self.controller = controller

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor

        # Slash command completion
        if text.startswith("/") and " " not in text:
            for cmd, meta in SLASH_COMMANDS:
                if cmd.startswith(text):
                    yield Completion(cmd, start_position=-len(text), display_meta=meta)
            return

        # Tool inspection completion: /tools <name>
        if text.startswith("/tools "):
            prefix = text[len("/tools "):].strip()
            for tool in getattr(self.controller.bridge, "tools", []):
                name = tool.get("name", "")
                if name.startswith(prefix):
                    desc = tool.get("description", "")
                    yield Completion(name, start_position=-len(prefix), display_meta=desc[:40])
            return

        # Manual tool call completion: /manual TOOL(...)
        if text.startswith("/manual "):
            remainder = text[len("/manual "):].lstrip()
            if "(" not in remainder:
                for tool in getattr(self.controller.bridge, "tools", []):
                    name = tool.get("name", "")
                    if name.startswith(remainder):
                        desc = tool.get("description", "")
                        yield Completion(name, start_position=-len(remainder), display_meta=desc[:40])
            else:
                tool_name = remainder[:remainder.index("(")].strip()
                tool = next((t for t in getattr(self.controller.bridge, "tools", []) if t.get("name") == tool_name), None)
                if tool:
                    props = (tool.get("inputSchema") or {}).get("properties", {})
                    after_delim = re.split(r"[,(]", remainder)[-1].strip()
                    if "=" not in after_delim:
                        for p_name, p_info in props.items():
                            if p_name.startswith(after_delim):
                                desc = p_info.get("description", "")
                                unit = p_info.get("unit", "")
                                meta = f"[{unit}] {desc}" if unit else desc
                                yield Completion(f"{p_name}=", start_position=-len(after_delim), display_meta=meta[:40])
            return

        # Model subcommand completion: /model set <field>=
        if text.startswith("/model set "):
            prefix = text.split()[-1] if not text.endswith(" ") else ""
            allowed = ["endpoint=", "model=", "api_key_env=", "api_key_file="]
            for field in allowed:
                if field.startswith(prefix):
                    yield Completion(field, start_position=-len(prefix), display_meta="config setting")
            return

        if text.startswith("/model "):
            prefix = text[len("/model "):].strip()
            for sub in ["check", "set"]:
                if sub.startswith(prefix):
                    yield Completion(sub, start_position=-len(prefix))
            return


class ConsoleController:
    """State machine and controller for the PiperX interactive console."""

    def __init__(
        self,
        bridge: MCPBridge,
        root: Path,
        emit: Callable[[str], None] = print,
    ):
        self.bridge = bridge
        self.root = root
        self.emit = emit
        self.background_task: asyncio.Task | None = None
        self.messages: list[dict[str, Any]] = []
        self.model_config: ModelConfig = ModelConfig()
        self.last_model_check: dict[str, Any] | None = None
        self.cached_status: dict[str, Any] | None = None
        self._toolbar_task: asyncio.Task | None = None
        self._closing: bool = False

    async def start(self) -> None:
        """Initialize model config, attempt MCP bridge connection, display overview, and start toolbar poller."""
        try:
            self.model_config = load_model_config(self.root)
        except (ValueError, OSError):
            self.emit("Model configuration is invalid; use /model set to replace it.")
        try:
            await self.bridge.open()
            self.emit(f"Connected to executor at {self.bridge.url} (transport open, {len(self.bridge.tools)} tools discovered).")
            await self._cmd_status()
        except Exception as exc:
            self.emit(f"Notice: Executor connection unavailable: {type(exc).__name__}")
            self.emit("Local commands (/model, /help, /quit) remain available. Use /connect or /status to retry.")

        await self._cmd_model("")
        self._toolbar_task = asyncio.create_task(self._toolbar_poller())

    async def close(self) -> None:
        """Cancel active tasks and close bridge connection cleanly (idempotent)."""
        if self._closing:
            return
        self._closing = True
        try:
            if self.background_task and not self.background_task.done():
                self.background_task.cancel()
                try:
                    await self.background_task
                except (asyncio.CancelledError, Exception):
                    pass
        finally:
            try:
                if self._toolbar_task and not self._toolbar_task.done():
                    self._toolbar_task.cancel()
                    try:
                        await self._toolbar_task
                    except (asyncio.CancelledError, Exception):
                        pass
            finally:
                if hasattr(self.bridge, "close"):
                    try:
                        res = self.bridge.close()
                        if inspect.isawaitable(res):
                            await res
                    except Exception:
                        pass
                if hasattr(self.bridge, "connected") and not self.bridge.connected:
                    if hasattr(self.bridge, "tools"):
                        self.bridge.tools = []

    async def _toolbar_poller(self) -> None:
        while not self._closing:
            try:
                if getattr(self.bridge, "connected", False):
                    state = await self.bridge.rest("GET", "/v1/state")
                    if isinstance(state, dict) and "error" not in state:
                        self.cached_status = state
                    else:
                        self.cached_status = None
                else:
                    self.cached_status = None
                    if hasattr(self.bridge, "tools") and not getattr(self.bridge, "connected", False):
                        self.bridge.tools = []
            except Exception:
                self.cached_status = None
                if hasattr(self.bridge, "tools") and not getattr(self.bridge, "connected", False):
                    self.bridge.tools = []
            try:
                await asyncio.sleep(2.0)
            except asyncio.CancelledError:
                break

    @staticmethod
    def _validate_finite_literals(val: Any, depth: int = 0, state: dict[str, int] | None = None) -> None:
        if state is None:
            state = {"total_length": 0}
        if depth > 20:
            raise ValueError("Argument nesting exceeds maximum depth of 20")
        if isinstance(val, tuple):
            raise ValueError("Tuples are not allowed in arguments")
        if isinstance(val, bool) or val is None:
            return
        if isinstance(val, (int, float)):
            try:
                if not math.isfinite(val):
                    raise ValueError(f"Non-finite number: {val}")
            except OverflowError:
                raise ValueError("Number out of range (overflow)")
            return
        if isinstance(val, str):
            if len(val) > 100000:
                raise ValueError("String argument exceeds 100k length limit")
            state["total_length"] += len(val)
            if state["total_length"] > 200000:
                raise ValueError("Total argument length exceeds limit")
            return
        if isinstance(val, list):
            if len(val) > 1000:
                raise ValueError("Array argument exceeds 1000 elements limit")
            for item in val:
                ConsoleController._validate_finite_literals(item, depth + 1, state)
            return
        if isinstance(val, dict):
            if len(val) > 500:
                raise ValueError("Object argument exceeds 500 keys limit")
            for k, v in val.items():
                if not isinstance(k, str):
                    raise ValueError("Object keys must be strings")
                if len(k) > 1000:
                    raise ValueError("Object key exceeds length limit")
                state["total_length"] += len(k)
                if state["total_length"] > 200000:
                    raise ValueError("Total argument length exceeds limit")
                ConsoleController._validate_finite_literals(v, depth + 1, state)
            return
        raise ValueError(f"Unsupported argument value type: {type(val).__name__}")

    async def invoke_tool(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        """Validate schema, inject request_id, call tool, and poll job until terminal."""
        if not isinstance(args, dict):
            return {"error": {"code": "invalid_arguments", "message": "Arguments must be a dictionary"}}

        try:
            self._validate_finite_literals(args)
        except ValueError as exc:
            return {"error": {"code": "invalid_arguments", "message": str(exc)}}

        tool = next((t for t in getattr(self.bridge, "tools", []) if t.get("name") == name), None)
        if not tool:
            return {"error": {"code": "unknown_tool", "message": f"Tool '{name}' not found in exposed tools"}}

        schema = tool.get("inputSchema") or {}
        properties = schema.get("properties") or {}
        required = schema.get("required") or []

        # Reject unknown argument names even if schema allows extras
        for arg_key in args:
            if arg_key not in properties:
                return {
                    "error": {
                        "code": "schema_violation",
                        "message": f"Unknown argument '{arg_key}' for tool '{name}'",
                    }
                }

        call_args = dict(args)
        # Inject ONLY when request_id is REQUIRED by discovered schema
        if "request_id" in required and "request_id" not in call_args:
            call_args["request_id"] = f"req_{secrets.token_hex(8)}"
            args["request_id"] = call_args["request_id"]

        # Validate with jsonschema Draft202012Validator after injection
        try:
            validator = Draft202012Validator(schema)
            errors = sorted(validator.iter_errors(call_args), key=lambda e: e.path)
            if errors:
                first_err = errors[0]
                path_str = ".".join(str(p) for p in first_err.path)
                msg = f"{path_str}: {first_err.message}" if path_str else first_err.message
                return {"error": {"code": "schema_violation", "message": f"Schema validation failed: {msg}"}}
        except Exception as exc:
            return {"error": {"code": "schema_violation", "message": f"Schema validator error: {type(exc).__name__}"}}

        req_id = call_args.get("request_id")
        if req_id:
            self.emit(f"-> Invoking {name} (request_id={req_id})...")
        else:
            self.emit(f"-> Invoking {name}...")

        try:
            result = await self.bridge.call(name, call_args)
        except Exception as exc:
            err_type = type(exc).__name__
            self.emit(f"<- Tool {name} transport error: {err_type}")
            return {"error": {"code": "transport_unknown", "message": f"Transport error: {err_type}"}, "request_id": req_id}

        if not isinstance(result, dict):
            return {"error": {"code": "invalid_response", "message": "Tool returned non-dict response"}}

        if isinstance(result.get("error"), dict):
            err_code = result["error"].get("code", "unknown_error")
            self.emit(f"<- Tool {name} error: {err_code}")
            if req_id:
                result.setdefault("request_id", req_id)
            return result

        job_id = result.get("job_id") or (result.get("job") or {}).get("job_id")
        status = result.get("status") or (result.get("job") or {}).get("status")

        # Never return initial accepted as completed
        if status == "accepted" and not job_id:
            return {
                "error": {
                    "code": "outcome_unknown",
                    "message": f"Tool {name} returned accepted status without job_id",
                },
                "request_id": req_id,
            }

        if not job_id:
            self.emit(f"<- Tool {name} completed.")
            return result

        # Handle robot_status(job_id) result terminal immediately
        if name == "robot_status":
            if status not in ("accepted", "running"):
                self.emit(f"<- Tool {name} completed.")
                return result

        # If already terminal, do not repoll
        if status in ("succeeded", "completed", "failed", "cancelled", "rejected", "stopped", "outcome_unknown"):
            self.emit(f"<- Tool {name} completed: {status}")
            return result

        # Poll ONLY accepted/running or cancellation_requested
        if status in ("accepted", "running", "cancellation_requested"):
            self.emit(f"<- Tool {name} accepted job: {job_id}")
            return await self._poll_job_until_terminal(name, job_id, call_args)

        self.emit(f"<- Tool {name} completed.")
        return result

    async def _poll_job_until_terminal(self, name: str, job_id: str, call_args: dict[str, Any]) -> dict[str, Any]:
        req_id = call_args.get("request_id")
        timeout_s = 30.0
        if "timeout_s" in call_args and isinstance(call_args["timeout_s"], (int, float)):
            timeout_s = float(call_args["timeout_s"])
        grace_s = 5.0
        deadline = time.monotonic() + timeout_s + grace_s
        last_status = None

        while time.monotonic() < deadline:
            try:
                status_res = await self.bridge.call("robot_status", {"job_id": job_id})
            except Exception as exc:
                err_type = type(exc).__name__
                self.emit(f"Error polling job {job_id}: {err_type}")
                return {
                    "error": {"code": "outcome_unknown", "message": f"Lost contact polling job {job_id}: {err_type}"},
                    "job_id": job_id,
                    "request_id": req_id,
                }

            if not isinstance(status_res, dict):
                return {
                    "error": {"code": "outcome_unknown", "message": "Non-dict status response during polling"},
                    "job_id": job_id,
                    "request_id": req_id,
                }

            status = status_res.get("status") or (status_res.get("job") or {}).get("status") or "unknown"
            if status != last_status:
                self.emit(f"Job {job_id}: {status}")
                last_status = status

            # Outcome unknown is terminal, detect before generic error envelope
            if status in ("succeeded", "completed", "failed", "cancelled", "rejected", "stopped", "outcome_unknown"):
                return status_res

            # Check if status_res has an error envelope
            if isinstance(status_res.get("error"), dict):
                err_envelope = status_res["error"]
                err_code = err_envelope.get("code", "unknown")
                self.emit(f"Job {job_id} status error: {err_code}")
                return {
                    "error": {
                        "code": "outcome_unknown",
                        "message": f"Polling error for job {job_id}: {err_envelope.get('message', err_code)}",
                    },
                    "job_id": job_id,
                    "request_id": req_id,
                }

            # Check if status_res indicates not_found
            if status_res.get("error") == "not_found" or status == "not_found":
                self.emit(f"Job {job_id} not found")
                return {
                    "error": {
                        "code": "outcome_unknown",
                        "message": f"Job {job_id} not found during polling",
                    },
                    "job_id": job_id,
                    "request_id": req_id,
                }

            await asyncio.sleep(0.5)

        self.emit(f"Job {job_id} timed out after {timeout_s + grace_s}s.")
        return {
            "error": {
                "code": "outcome_unknown",
                "message": f"Job {job_id} timed out after {timeout_s + grace_s}s",
            },
            "job_id": job_id,
            "request_id": req_id,
        }

    async def handle_line(self, line: str) -> bool:
        """Handle one line of console input. Return False to exit console."""
        clean = line.strip()
        if not clean:
            return True

        if clean.startswith("/"):
            parts = clean.split(None, 1)
            cmd = parts[0]
            rest = parts[1] if len(parts) > 1 else ""

            if cmd == "/help":
                self._cmd_help()
                return True
            if cmd == "/status":
                await self._cmd_status()
                return True
            if cmd == "/connect":
                await self._cmd_connect(rest)
                return True
            if cmd == "/disconnect":
                await self._cmd_disconnect()
                return True
            if cmd == "/model":
                await self._cmd_model(rest)
                return True
            if cmd == "/manual":
                await self._cmd_manual(rest)
                return True
            if cmd == "/tools":
                await self._cmd_tools(rest)
                return True
            if cmd == "/calls":
                await self._cmd_calls(rest)
                return True
            if cmd == "/jobs":
                await self._cmd_jobs(rest)
                return True
            if cmd == "/params":
                await self._cmd_params()
                return True
            if cmd == "/stop":
                await self._cmd_stop()
                return True
            if cmd == "/quit":
                await self._cmd_quit()
                return False

            self.emit(f"Unknown command '{cmd}'. Type /help for available commands.")
            return True

        # Plain text initiates background agent turn
        if not self.model_config.endpoint or not self.model_config.model:
            self.emit("Model not configured. Use /model set endpoint=... model=... to configure.")
            return True

        if self.background_task and not self.background_task.done():
            self.emit("A command is already running. Use /stop to cancel it before starting a new one.")
            return True

        if not self.messages:
            self.messages.append({"role": "system", "content": SYSTEM_INSTRUCTION})

        self.messages.append({"role": "user", "content": clean})
        self.background_task = asyncio.create_task(self._run_model_turn())
        return True

    def _cmd_help(self) -> None:
        self.emit("PiperX Interactive Console Commands:")
        for cmd, desc in SLASH_COMMANDS:
            self.emit(f"  {cmd:<15} {desc}")
        self.emit("")
        self.emit("Plain text prompts are sent to the configured model agent.")
        self.emit("Ctrl-C cancels the active local task; use /stop for robot motion cancellation.")

    async def _cmd_status(self) -> None:
        try:
            if not self.bridge.connected:
                try:
                    await self.bridge.open()
                except Exception:
                    pass
            state = await self.bridge.rest("GET", "/v1/state")
            if isinstance(state, dict) and "error" in state:
                self.cached_status = None
                self.emit(f"Status error: {state['error']}")
            else:
                self.cached_status = state
                self.emit(cli_views.render_status(state))
        except Exception as exc:
            self.emit(f"Failed to retrieve status: {type(exc).__name__}")

    async def _cmd_connect(self, rest: str) -> None:
        if self.background_task and not self.background_task.done():
            self.emit("Cannot connect while a background task is running. Use /stop to cancel it first.")
            return

        tokens = rest.strip().split()
        reconnect = False
        target_id: str | None = None

        for t in tokens:
            if t == "--reconnect":
                reconnect = True
            elif t.startswith("-"):
                self.emit(f"Unknown option '{t}' to /connect. Usage: /connect [device_id] [--reconnect]")
                return
            elif target_id is None:
                target_id = t
            else:
                self.emit(f"Unexpected argument '{t}' to /connect. Usage: /connect [device_id] [--reconnect]")
                return

        if not self.bridge.connected:
            try:
                await self.bridge.open()
                self.emit(f"Executor MCP transport opened ({len(self.bridge.tools)} tools discovered).")
            except Exception as exc:
                self.emit(f"Cannot connect to executor MCP: {type(exc).__name__}")
                return

        try:
            res = await self.bridge.rest("GET", "/v1/devices")
        except Exception as exc:
            self.emit(f"Device enumeration failed: {type(exc).__name__}")
            return

        if isinstance(res, dict) and "error" in res:
            self.emit(f"Device enumeration error: {res['error']}")
            return

        notes = res.get("notes") or []
        if isinstance(notes, str):
            notes = [notes]
        for note in notes:
            self.emit(f"Discovery note: {note}")

        devices = res.get("devices") or []
        connected_id = res.get("connected_device_id")

        if not target_id:
            if connected_id:
                target_id = connected_id
                self.emit(f"Reusing currently connected device: {target_id}")
            elif not devices:
                self.emit("No devices discovered on executor host.")
                return
            elif len(devices) == 1:
                target_id = devices[0].get("id")
                self.emit(f"Auto-selected device: {target_id}")
            else:
                self.emit(f"Discovered {len(devices)} devices:")
                for d in devices:
                    status_tag = " (connected)" if d.get("connected") or d.get("id") == connected_id else ""
                    self.emit(f"  - {d.get('id')}: {d.get('label', 'unlabeled')}{status_tag}")
                self.emit("Specify device ID to connect: /connect <device_id>")
                return

        conn_args: dict[str, Any] = {"reconnect": reconnect}
        if target_id is not None:
            conn_args["device_id"] = target_id

        self.emit(f"Connecting to device '{target_id}' (reconnect={reconnect})...")
        try:
            connect_res = await self.bridge.call("robot_connect", conn_args)
            if isinstance(connect_res, dict) and "error" in connect_res:
                self.emit(f"robot_connect error: {connect_res['error']}")
                return
            self.emit(f"Connected to device '{target_id}'.")
        except Exception as exc:
            self.emit(f"robot_connect failed: {type(exc).__name__}")
            return

        try:
            state = await self.bridge.rest("GET", "/v1/state")
            if isinstance(state, dict) and "error" not in state:
                self.cached_status = state
                self.emit(cli_views.render_status(state))
            else:
                self.emit("Status telemetry unavailable after connect.")
        except Exception as exc:
            self.emit(f"Failed to query status after connect: {type(exc).__name__}")

    async def _cmd_disconnect(self) -> None:
        if self.background_task and not self.background_task.done():
            self.emit("Cannot disconnect while a background task is running. Use /stop to cancel it first.")
            return

        self.emit("Disconnecting from CAN...")
        try:
            if self.bridge.connected:
                res = await self.bridge.call("robot_disconnect", {})
                if isinstance(res, dict) and "error" in res:
                    self.emit(f"Disconnect error: {res['error']}")
                else:
                    self.emit(f"Disconnect result: {res}")
            else:
                res = await self.bridge.rest("POST", "/v1/disconnect")
                self.emit(f"Disconnect result: {res.get('error') or res}")
        except Exception as exc:
            self.emit(f"Disconnect failed: {type(exc).__name__}")

    async def _cmd_model(self, rest: str) -> None:
        trimmed = rest.strip()
        tokens = trimmed.split(None, 1)
        subcmd = tokens[0] if tokens else ""
        remainder = tokens[1] if len(tokens) > 1 else ""

        if not subcmd:
            self.emit("Model Configuration:")
            self.emit(f"  Endpoint: {self.model_config.endpoint or '(not set)'}")
            self.emit(f"  Model: {self.model_config.model or '(not set)'}")
            env_var = self.model_config.api_key_env or "PIPERX_MODEL_API_KEY"
            env_set = bool(os.environ.get(env_var))
            self.emit(f"  API Key Env: {env_var} ({'set' if env_set else 'not set'})")
            key_file = self.model_config.api_key_file
            file_set = bool(key_file and Path(key_file).is_file())
            self.emit(f"  API Key File: {key_file or '(none)'} ({'exists' if file_set else 'missing'})")
            has_key = bool(resolve_api_key(self.model_config))
            self.emit(f"  Resolved API Key: {'present' if has_key else 'absent'}")
            self.emit(f"  Last Model Check: {self.last_model_check or '(never run)'}")
            self.emit("")
            self.emit("MCP Bridge Status:")
            self.emit(f"  Connected: {getattr(self.bridge, 'connected', False)}")
            self.emit(f"  Tools Discovered: {len(getattr(self.bridge, 'tools', []))}")
            return

        if subcmd == "check":
            if self.background_task and not self.background_task.done():
                self.emit("A command is already running. Use /stop to cancel it before starting a new one.")
                return

            async def _run_model_check_bg():
                self.emit("Running model check...")
                try:
                    res = await check_model(self.model_config)
                    self.last_model_check = res
                    status = res.get("status", "unknown")
                    latency = res.get("latency_ms", "unknown")
                    model = res.get("model", self.model_config.model)
                    err = res.get("error")
                    if err or status != "ok":
                        self.emit(f"Model check: status={status}, model={model}, error={err or 'check failed'}")
                    else:
                        self.emit(f"Model check: status={status}, latency={latency}ms, model={model}")
                except asyncio.CancelledError:
                    self.emit("[Model check cancelled]")
                    raise
                except Exception as exc:
                    err_msg = f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__
                    self.emit(f"Model check failed: {err_msg}")

            self.background_task = asyncio.create_task(_run_model_check_bg())
            return

        if subcmd == "set":
            if self.background_task and not self.background_task.done():
                self.emit("Cannot change model configuration while a background task is running. Use /stop to cancel it first.")
                return

            if not remainder:
                self.emit("Usage: /model set endpoint=... model=... api_key_env=... api_key_file=...")
                return

            try:
                pattern = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)=(?:\"([^\"]*)\"|'([^']*)'|([^\s\"']*))(?:\s+|$)")
                items = []
                pos = 0
                while pos < len(remainder):
                    match = pattern.match(remainder, pos)
                    if not match:
                        raise ValueError("Use key=value or key=\"quoted value\".")
                    key = match.group(1)
                    value = next(v for v in match.groups()[1:] if v is not None)
                    items.append(key + "=" + value)
                    pos = match.end()
            except ValueError as exc:
                self.emit(f"Invalid parameter format: {exc}")
                return

            new_config = ModelConfig(
                endpoint=self.model_config.endpoint,
                model=self.model_config.model,
                api_key_env=self.model_config.api_key_env,
                api_key_file=self.model_config.api_key_file,
            )

            for item in items:
                if "=" not in item:
                    self.emit(f"Invalid parameter format '{item}', expected key=value")
                    return
                key, val = item.split("=", 1)
                key = key.strip()
                val = val.strip()
                if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
                    val = val[1:-1]

                if key in ("api_key", "key", "token", "secret", "password"):
                    self.emit("Error: Setting raw API key values is forbidden. Use api_key_env or api_key_file.")
                    return

                if key == "endpoint":
                    try:
                        validate_endpoint(val)
                    except ValueError as err:
                        self.emit(f"Invalid endpoint: {err}")
                        return
                    new_config.endpoint = val
                elif key == "model":
                    new_config.model = val
                elif key == "api_key_env":
                    new_config.api_key_env = val
                elif key == "api_key_file":
                    new_config.api_key_file = val
                else:
                    self.emit(f"Unknown configuration field '{key}'.")
                    return

            try:
                save_model_config(self.root, new_config)
            except Exception as exc:
                self.emit(f"Failed to save model configuration: {type(exc).__name__}: {exc}")
                return

            self.model_config = new_config
            self.last_model_check = None
            self.messages.clear()
            self.emit("Model configuration updated and saved to model.json.")
            return

        self.emit(f"Unknown /model subcommand '{subcmd}'. Use /model, /model check, or /model set ...")

    async def _cmd_manual(self, rest: str) -> None:
        if not rest:
            self.emit("Usage: /manual TOOL(keyword=literal, ...)")
            return
        try:
            tool_name, args = parse_manual_command(rest)
        except ValueError as exc:
            self.emit(f"Manual parser error: {exc}")
            return

        if self.background_task and not self.background_task.done():
            self.emit("A command is already running. Use /stop to cancel it before starting a new one.")
            return

        async def _run_manual():
            try:
                res = await self.invoke_tool(tool_name, args)
                self.emit(f"Result: {json.dumps(res, indent=2, ensure_ascii=False)}")
            except asyncio.CancelledError:
                self.emit(f"\n[Manual tool '{tool_name}' cancelled]")
                raise
            except Exception as exc:
                self.emit(f"Manual execution error: {type(exc).__name__}")

        self.background_task = asyncio.create_task(_run_manual())

    async def _cmd_tools(self, rest: str) -> None:
        tool_name = rest.strip()
        if not tool_name:
            tools = getattr(self.bridge, "tools", [])
            if not tools:
                self.emit("No tools exposed by MCP bridge.")
                return
            self.emit(f"Discovered MCP Tools ({len(tools)}):")
            for tool in tools:
                name = tool.get("name", "unnamed")
                desc = tool.get("description", "")
                self.emit(f"  - {name}: {desc}")
            return

        tool = next((t for t in getattr(self.bridge, "tools", []) if t.get("name") == tool_name), None)
        if not tool:
            self.emit(f"Tool '{tool_name}' not found.")
            return

        self.emit(f"Tool: {tool_name}")
        self.emit(f"Description: {tool.get('description', '(none)')}")
        schema = tool.get("inputSchema") or {}
        props = schema.get("properties") or {}
        required = schema.get("required") or []
        if props:
            self.emit("Parameters:")
            for p_name, p_info in props.items():
                req_tag = "[required]" if p_name in required else "[optional]"
                p_type = p_info.get("type", "any")
                constraints = []
                if "unit" in p_info:
                    constraints.append(f"unit: {p_info['unit']}")
                if "minimum" in p_info:
                    constraints.append(f">={p_info['minimum']}")
                if "maximum" in p_info:
                    constraints.append(f"<={p_info['maximum']}")
                if "exclusiveMinimum" in p_info:
                    constraints.append(f">{p_info['exclusiveMinimum']}")
                if "exclusiveMaximum" in p_info:
                    constraints.append(f"<{p_info['exclusiveMaximum']}")
                if "minItems" in p_info:
                    constraints.append(f"minItems: {p_info['minItems']}")
                if "maxItems" in p_info:
                    constraints.append(f"maxItems: {p_info['maxItems']}")
                if "minLength" in p_info:
                    constraints.append(f"minLength: {p_info['minLength']}")
                if "maxLength" in p_info:
                    constraints.append(f"maxLength: {p_info['maxLength']}")
                if "pattern" in p_info:
                    constraints.append(f"pattern: {p_info['pattern']}")
                if "enum" in p_info:
                    constraints.append(f"enum: {p_info['enum']}")
                if "default" in p_info:
                    constraints.append(f"default: {p_info['default']}")

                constraint_str = f" [{', '.join(constraints)}]" if constraints else ""
                p_desc = p_info.get("description", "")
                desc_str = f": {p_desc}" if p_desc else ""
                self.emit(f"  - {p_name} {req_tag} ({p_type}{constraint_str}){desc_str}")
        else:
            self.emit("Parameters: none")

    async def _cmd_calls(self, rest: str) -> None:
        limit = 50
        if rest.strip().isdigit():
            limit = max(1, min(500, int(rest.strip())))
        try:
            res = await self.bridge.rest("GET", f"/v1/calls?limit={limit}")
            if isinstance(res, dict) and "error" in res:
                self.emit(f"Error retrieving calls: {res['error']}")
            else:
                self.emit(cli_views.render_calls(res))
        except Exception as exc:
            self.emit(f"Calls request failed: {type(exc).__name__}")

    async def _cmd_jobs(self, rest: str) -> None:
        limit = 50
        if rest.strip().isdigit():
            limit = max(1, min(500, int(rest.strip())))
        try:
            res = await self.bridge.rest("GET", f"/v1/jobs?limit={limit}")
            if isinstance(res, dict) and "error" in res:
                self.emit(f"Error retrieving jobs: {res['error']}")
            else:
                self.emit(cli_views.render_jobs(res))
        except Exception as exc:
            self.emit(f"Jobs request failed: {type(exc).__name__}")

    async def _cmd_params(self) -> None:
        try:
            res = await self.bridge.rest("GET", "/v1/parameters")
            if isinstance(res, dict) and "error" in res:
                self.emit(f"Error retrieving parameters: {res['error']}")
            else:
                self.emit(cli_views.render_params(res))
        except Exception as exc:
            self.emit(f"Parameters request failed: {type(exc).__name__}")

    async def _cmd_stop(self) -> None:
        if self.background_task and not self.background_task.done():
            self.background_task.cancel()
            try:
                await self.background_task
            except asyncio.CancelledError:
                pass
            except Exception as exc:
                self.emit(f"Background task error on stop: {type(exc).__name__}")
            self.emit("Stopped active local task.")
        else:
            self.emit("No active local task.")

        self.emit("Calling robot_stop on executor...")
        try:
            stop_res = await self.bridge.call("robot_stop", {})
            if isinstance(stop_res, dict) and "error" in stop_res:
                self.emit(f"robot_stop error: {stop_res['error']}")
            else:
                status = stop_res.get("status") if isinstance(stop_res, dict) else None
                if status == "cancellation_requested":
                    self.emit(f"robot_stop result: cancellation requested (not confirmed stopped; status={status})")
                else:
                    self.emit(f"robot_stop result: {stop_res}")
        except Exception as exc:
            self.emit(f"robot_stop call failed: {type(exc).__name__}")

    async def _cmd_quit(self) -> None:
        self.emit("Exiting console.")
        await self.close()

    async def _run_model_turn(self) -> None:
        try:
            reply = await run_turn(
                config=self.model_config,
                messages=self.messages,
                tools=getattr(self.bridge, "tools", []),
                invoke=self.invoke_tool,
                emit=self.emit,
            )
            if reply:
                self.emit(f"\nAssistant: {reply}\n")
        except asyncio.CancelledError:
            self.emit("\n[Model turn cancelled]")
            raise
        except Exception as exc:
            self.emit(f"\nModel error: {type(exc).__name__}: {exc}\n")


async def _run_interactive_loop(controller: ConsoleController) -> None:
    def get_bottom_toolbar():
        if not getattr(controller.bridge, "connected", False):
            return " [MCP unreachable] "
        if controller.cached_status:
            backend = controller.cached_status.get("backend", "unknown")
            conn = (controller.cached_status.get("connection") or {}).get("status", "unknown")
            ready = controller.cached_status.get("ready", False)
            active_job = controller.cached_status.get("active_job_id") or "none"
            model = controller.model_config.model or "no-model"
            return f" [{backend}] Conn: {conn} | Ready: {ready} | Job: {active_job} | Model: {model} "
        return " [MCP unreachable] "

    session = PromptSession(
        history=InMemoryHistory(),
        completer=ConsoleCompleter(controller),
        bottom_toolbar=get_bottom_toolbar,
        refresh_interval=1,
    )

    with patch_stdout():
        while True:
            try:
                line = await session.prompt_async("piperx> ")
                line = line.strip()
                if not line:
                    continue
                cont = await controller.handle_line(line)
                if not cont:
                    break
            except KeyboardInterrupt:
                if controller.background_task and not controller.background_task.done():
                    controller.background_task.cancel()
                    try:
                        await controller.background_task
                    except asyncio.CancelledError:
                        pass
                    controller.emit("\n[Local task cancelled. Use /stop if a robot motion job is active.]")
                else:
                    controller.emit("\n(^C)")
            except EOFError:
                break


async def _run_piped_loop(controller: ConsoleController) -> None:
    loop = asyncio.get_running_loop()
    while True:
        try:
            line = await loop.run_in_executor(None, sys.stdin.readline)
            if not line:
                break
            line = line.strip()
            if not line:
                continue
            cont = await controller.handle_line(line)
            if not cont:
                break
            if controller.background_task and not controller.background_task.done():
                await controller.background_task
        except (EOFError, KeyboardInterrupt):
            break


async def run_console(args: argparse.Namespace, root: Path) -> None:
    """Run the interactive console from CLI main."""
    raw_url = getattr(args, "url", None) or os.environ.get("PIPERX_URL") or configured_url(root) or "http://127.0.0.1:8765"
    try:
        validate_endpoint(raw_url)
    except ValueError as exc:
        print(f"Error: Invalid executor URL: {exc}", file=sys.stderr)
        return

    token_file = getattr(args, "token_file", None)
    if token_file is None and os.environ.get("PIPERX_TOKEN_FILE"):
        token_file = Path(os.environ["PIPERX_TOKEN_FILE"])
    if token_file is None:
        token_file = root / "model.token"

    bridge = MCPBridge(url=raw_url, token_file=token_file)
    controller = ConsoleController(bridge=bridge, root=root)

    is_interactive = sys.stdin.isatty()
    try:
        await controller.start()
        if is_interactive and HAVE_PROMPT_TOOLKIT:
            await _run_interactive_loop(controller)
        else:
            await _run_piped_loop(controller)
    finally:
        await controller.close()
