"""Managed single-terminal interaction for Piper Robot.

Provides InteractiveConsoleController with operator-auth sessions, approval policies,
managed runtime switching/shutdown, interactive limits wizard, and live approval tracking.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
import secrets
import sys
from pathlib import Path
from typing import Any, Callable

from . import cli_views
from .console import ConsoleController, SLASH_COMMANDS
from .console_bridge import MCPBridge

INTERACTIVE_SLASH_COMMANDS = [
    ("/help", "Show available commands and usage guide"),
    ("/observe", "Capture local real RGB-D: /observe [SERIAL]; no robot motion"),
    ("/pixel", "Read recorded camera-frame depth: /pixel U V; not a motion target"),
    ("/status", "Show read-only robot and connection telemetry"),
    ("/connect", "Enumerate and connect to device on executor host"),
    ("/disconnect", "Release CAN connection on executor"),
    ("/approval", "Inspect or set approval mode [always|risk|auto]"),
    ("/approve", "Approve a pending job: /approve <job_id>"),
    ("/deny", "Deny a pending job: /deny <job_id>"),
    ("/limits", "Show limits and open a guided editor; JSON is optional"),
    ("/config", "Inspect config; /config force [on|off] for confirmed 5-degree recovery"),
    ("/confirm", "Confirm a pending configuration proposal: /confirm [code]"),
    ("/cancel", "Cancel a pending configuration proposal: /cancel [code]"),
    ("/remote", "List or add remote targets: /remote [add NAME SSH_HOST]"),
    ("/mode", "Switch execution mode: /mode [simulation|real] [TARGET]"),
    ("/shutdown", "Gracefully shut down managed executor and exit"),
    ("/model", "Inspect or configure LLM endpoint and credentials"),
    ("/manual", "Execute an MCP tool directly: /manual TOOL(arg=val)"),
    ("/tools", "List exposed MCP tools or inspect a schema"),
    ("/calls", "Show recorded executor calls log"),
    ("/jobs", "Show recorded jobs and execution results"),
    ("/params", "Show configured parameters and reported state"),
    ("/stop", "Cancel local tasks and issue robot stop request (not hardware emergency stop)"),
    ("/quit", "Exit console gracefully draining in-flight actions"),
]


@dataclass
class Proposal:
    code: str
    action: str  # "policy" or "config"
    description: str
    body: dict[str, Any]


async def choose_startup(prompt: Callable[..., Any] | None = None) -> tuple[str, str]:
    """Helper for selecting startup mode and target interactively.

    Piped/non-interactive startup requires explicit --mode flag.
    """
    if prompt is None:
        if not sys.stdin.isatty():
            raise ValueError("Piped startup requires --mode; cannot select interactively.")
        raise ValueError("choose_startup requires a prompt callable or explicit --mode.")

    selected_mode = await prompt("Select startup mode (simulation/real) [simulation]: ", default="simulation")
    selected_mode = (selected_mode or "").strip().lower()
    if not selected_mode:
        selected_mode = "simulation"
    if selected_mode not in ("simulation", "real"):
        raise ValueError(f"Invalid mode '{selected_mode}'. Allowed: simulation, real.")

    selected_target = await prompt("Select target (local/<remote_name>) [local]: ", default="local")
    selected_target = (selected_target or "").strip() or "local"
    return selected_mode, selected_target


class InteractiveConsoleController(ConsoleController):
    """Interactive console controller supporting operator policy, approvals, and managed executor lifecycle."""

    def __init__(
        self,
        bridge: MCPBridge,
        root: Path,
        emit: Callable[[str], None] = print,
        *,
        operator_call: Callable[..., Any] | None = None,
        managed: Any = None,
        prompt: Callable[..., Any] | None = None,
    ) -> None:
        super().__init__(bridge=bridge, root=root, emit=emit)
        self.slash_commands = INTERACTIVE_SLASH_COMMANDS
        self.managed = managed
        self.prompt = prompt
        self.mode: str = getattr(managed, "mode", "simulation") if managed else "simulation"
        self.target: str = getattr(managed, "target", "local") if managed else "local"
        self.approval_mode: str = "risk"
        self._seen_pending: set[tuple[str, Any]] = set()
        self._watcher_task: asyncio.Task[None] | None = None
        self._heartbeat_task: asyncio.Task[None] | None = None
        self._drain_task: asyncio.Task[None] | None = None
        self._proposals: dict[str, Proposal] = {}

        if operator_call is not None:
            self.operator_call = operator_call
        elif hasattr(bridge, "operator_call"):
            self.operator_call = bridge.operator_call
        else:
            async def _default_operator_call(method: str, path: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
                return await self.bridge.rest(method, path, body)
            self.operator_call = _default_operator_call

    async def start(self) -> None:
        if self._heartbeat_task is None or self._heartbeat_task.done():
            self._heartbeat_task = asyncio.create_task(self._operator_heartbeat())
        await super().start()
        if self._watcher_task is None or self._watcher_task.done():
            self._watcher_task = asyncio.create_task(self._pending_approvals_watcher())

    async def close(self) -> None:
        if self._closing:
            return
        if self._watcher_task and not self._watcher_task.done():
            self._watcher_task.cancel()
            try:
                await self._watcher_task
            except (asyncio.CancelledError, Exception):
                pass
            self._watcher_task = None
        if self._heartbeat_task and not self._heartbeat_task.done():
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except (asyncio.CancelledError, Exception):
                pass
            self._heartbeat_task = None
        await super().close()

    async def _pending_approvals_watcher(self) -> None:
        while not self._closing:
            try:
                res = await self.operator_call("GET", "/operator/interaction")
                if isinstance(res, dict):
                    pending = res.get("pending")
                    if isinstance(pending, list):
                        for job in pending:
                            if not isinstance(job, dict):
                                continue
                            job_id = job.get("job_id")
                            if not job_id:
                                continue
                            rev = job.get("revision", job.get("version", job.get("status", 1)))
                            key = (str(job_id), rev)
                            if key not in self._seen_pending:
                                self._seen_pending.add(key)
                                approval = job.get("approval") if isinstance(job.get("approval"), dict) else {}
                                reasons = approval.get("reasons") or job.get("reasons") or []
                                command = job.get("command") or approval.get("command") or job.get("action")
                                cmd_str = json.dumps(command) if isinstance(command, (dict, list)) else str(command or "")
                                r_str = f" ({', '.join(str(r) for r in reasons)})" if reasons else ""
                                cmd_info = f" [command: {cmd_str}]" if cmd_str else ""
                                self.emit(f"Notice: Job {job_id} requires operator approval{r_str}.{cmd_info} Use /approve {job_id} or /deny {job_id}.")
                    policy = res.get("policy")
                    if isinstance(policy, dict) and "mode" in policy:
                        self.approval_mode = str(policy["mode"])
            except asyncio.CancelledError:
                break
            except Exception:
                pass
            try:
                await asyncio.sleep(1.0)
            except asyncio.CancelledError:
                break

    async def _operator_heartbeat(self) -> None:
        while not self._closing:
            try:
                if not self._session_lost:
                    sess_id = getattr(self.bridge, "session_id", None) or getattr(self, "session_id", None)
                    if sess_id:
                        res = await self.operator_call("POST", "/operator/session/heartbeat", {"session_id": sess_id})
                        if isinstance(res, dict) and "error" in res:
                            if not self._session_lost:
                                self._session_lost = True
                                self.emit("Notice: Control session lost or expired. Explicit /connect or reconnect required.")
            except asyncio.CancelledError:
                break
            except Exception:
                if not self._session_lost:
                    self._session_lost = True
                    self.emit("Notice: Control session lost or expired. Explicit /connect or reconnect required.")
            try:
                await asyncio.sleep(1.0)
            except asyncio.CancelledError:
                break

    async def handle_line(self, line: str) -> bool:
        clean = line.strip()
        if not clean:
            return True

        if self._draining:
            parts = clean.split(None, 1)
            cmd = parts[0]
            rest = parts[1] if len(parts) > 1 else ""
            if cmd == "/stop":
                await self._cmd_stop()
                return True
            if cmd == "/status":
                await self._cmd_status()
                return True
            if cmd == "/jobs":
                await self._cmd_jobs(rest)
                return True
            if cmd in ("/shutdown", "/quit"):
                self.emit("Console is already shutting down / draining.")
                return True
            self.emit("Console is draining/shutting down; new commands rejected.")
            return True

        if clean.startswith("/"):
            parts = clean.split(None, 1)
            cmd = parts[0]
            rest = parts[1] if len(parts) > 1 else ""

            if cmd == "/help":
                self._cmd_help()
                return True
            if cmd in ("/observe", "/pixel"):
                if self.mode != "real" or self.target != "local":
                    self.emit("Camera commands currently support real/local only; no remote/local camera substitution.")
                    return True
                from . import console_camera
                try:
                    if cmd == "/observe":
                        if len(rest.split()) > 1:
                            raise ValueError("Usage: /observe [SERIAL]")
                        self.emit("Capturing RGB-D; no robot motion...")
                        result = await console_camera.observe(self.root, rest.strip() or None)
                    else:
                        fields = rest.split()
                        if len(fields) != 2:
                            raise ValueError("Usage: /pixel U V")
                        result = await console_camera.pixel(self.root, *map(int, fields))
                    self.emit(json.dumps(result, indent=2, ensure_ascii=False))
                except Exception as exc:
                    self.emit(f"Camera error: {exc}")
                return True
            if cmd == "/approval":
                await self._cmd_approval(rest)
                return True
            if cmd == "/approve":
                await self._cmd_approve(rest)
                return True
            if cmd == "/deny":
                await self._cmd_deny(rest)
                return True
            if cmd == "/limits":
                await self._cmd_limits(rest)
                return True
            if cmd == "/config":
                await self._cmd_config(rest)
                return True
            if cmd == "/confirm":
                await self._cmd_confirm(rest)
                return True
            if cmd == "/cancel":
                await self._cmd_cancel(rest)
                return True
            if cmd == "/remote":
                await self._cmd_remote(rest)
                return True
            if cmd == "/mode":
                await self._cmd_mode(rest)
                return True
            if cmd == "/shutdown":
                return await self._cmd_shutdown()
            if cmd == "/quit":
                return await self._cmd_quit()
            if cmd == "/connect":
                await self._cmd_connect(rest)
                return True

        return await super().handle_line(line)

    def _cmd_help(self) -> None:
        self.emit("PiperX Interactive Console Commands:")
        for cmd, desc in self.slash_commands:
            self.emit(f"  {cmd:<15} {desc}")
        self.emit("")
        self.emit("开始：/connect → /status → /tools；用 /manual 工具名(参数) 调用工具。")
        self.emit("新仿真默认 auto，无需配置审批阈值；真机默认 risk。已有配置保持不变。")
        self.emit("/limits 打开填写向导，回车保留原值；无需手写 JSON。")
        self.emit("配置修改后 /confirm 确认、/cancel 取消；多项待确认时需指定编号。")
        self.emit("等待动作审批时可用 /status、/jobs、/stop；/quit 等待动作结束后退出。")
        self.emit("Plain text prompts are sent to the configured model agent.")
        self.emit("Ctrl-C cancels the active local task; use /stop for robot motion cancellation.")

    def _generate_proposal_code(self) -> str:
        for _ in range(100):
            code = secrets.token_hex(2).upper()
            if code not in self._proposals:
                return code
        return secrets.token_hex(4).upper()

    def _show_retained_constraints(self, policy: dict[str, Any]) -> None:
        auto = policy.get("automatic") or {}
        limits = policy.get("limits") or {}
        self.emit("  Automatic approval thresholds (used only in risk mode):")
        self.emit(f"    max_joint_step_deg:  {auto.get('max_joint_step_deg', '[unconfigured - ask]')}")
        self.emit(f"    max_tcp_step_m:      {auto.get('max_tcp_step_m', '[unconfigured - ask]')}")
        self.emit(f"    max_speed_percent:   {auto.get('max_speed_percent', '[unconfigured - ask]')}")
        self.emit(f"    max_effort_protocol: {auto.get('max_effort_protocol', '[unconfigured - ask]')}")
        self.emit("  Execution limits:")
        self.emit(f"    max_speed_percent:   {limits.get('max_speed_percent', 100)}")
        self.emit(f"    gripper_max_m:       {limits.get('gripper_max_m', 0.07)}")
        self.emit(f"    gripper_min_m:       {limits.get('gripper_min_m', 0.0)}")

    async def _cmd_confirm(self, rest: str) -> None:
        code = rest.strip().upper()
        if not code and len(self._proposals) == 1:
            code = next(iter(self._proposals))
        if not code:
            self.emit("No single pending change. Use /confirm CODE when multiple changes are pending.")
            return
        prop = self._proposals.pop(code, None)
        if prop is None:
            self.emit(f"No pending proposal with code '{code}'.")
            return

        self._proposals.clear()

        if prop.action == "policy":
            try:
                res = await self.operator_call("PUT", "/operator/interaction", {"policy": prop.body})
                if isinstance(res, dict) and "error" in res:
                    self.emit(f"Failed to apply proposal {code}: {res['error']}")
                else:
                    new_mode = prop.body.get("mode")
                    if new_mode:
                        self.approval_mode = str(new_mode)
                    self.emit(f"Proposal {code} confirmed: {prop.description}.")
            except Exception as exc:
                self.emit(f"Failed to apply proposal {code}: {type(exc).__name__}: {exc}")
        elif prop.action == "config":
            try:
                res = await self.managed.configure(prop.body)
                self.emit(f"Proposal {code} confirmed: Configuration updated: {json.dumps(res, indent=2, default=str)}")
            except Exception as exc:
                self.emit(f"Failed to apply config proposal {code}: {type(exc).__name__}: {exc}")

    async def _cmd_cancel(self, rest: str) -> None:
        code = rest.strip().upper()
        if not code and len(self._proposals) == 1:
            code = next(iter(self._proposals))
        if not code:
            self.emit("No single pending change. Use /cancel CODE when multiple changes are pending.")
            return
        prop = self._proposals.pop(code, None)
        if prop is None:
            self.emit(f"No pending proposal with code '{code}'.")
            return
        self.emit(f"Proposal {code} cancelled ({prop.description}).")

    async def _read_interaction(self) -> dict[str, Any] | None:
        """Do not build edits from defaults when the current policy is unknown."""
        try:
            result = await self.operator_call("GET", "/operator/interaction")
            if (isinstance(result, dict) and "error" not in result
                    and isinstance(result.get("policy"), dict) and result["policy"]):
                return result
        except Exception as exc:
            self.emit(f"Failed to read current policy: {type(exc).__name__}")
        self.emit("Could not retrieve current policy; editor not opened. Retry after reconnecting.")
        return None

    async def _cmd_approval(self, rest: str) -> None:
        trimmed = rest.strip().lower()
        if trimmed and trimmed not in ("always", "risk", "auto"):
            self.emit(f"Invalid approval mode '{trimmed}'. Allowed: always, risk, auto.")
            return
        current = await self._read_interaction()
        if current is None:
            return
        current_policy = current["policy"]
        self.approval_mode = current_policy.get("mode", "unknown")
        if not trimmed:
            self.emit(f"Current approval mode: {self.approval_mode}")
            if self.approval_mode == "auto":
                self._show_retained_constraints(current_policy)
            return

        # With an older proposal pending, even a request to keep the current
        # setting must become an explicit alternative, not confirm the old edit.
        if current_policy.get("mode") == trimmed and not self._proposals:
            self.emit(f"Approval mode is already {trimmed}; no change needed.")
            return
        proposed_policy = dict(current_policy)
        proposed_policy["mode"] = trimmed

        if trimmed == "auto":
            self.emit("Retained constraints for auto mode:")
            self._show_retained_constraints(proposed_policy)

        code = self._generate_proposal_code()
        self._proposals[code] = Proposal(
            code=code,
            action="policy",
            description=f"Set approval mode to '{trimmed}'",
            body=proposed_policy,
        )
        self.emit(f"Confirmation required: Proposal {code} bound to exact policy change.")
        self.emit(f"Type /confirm to apply mode '{trimmed}', or /cancel to reject (if multiple pending: /confirm {code}).")

    async def _cmd_approve(self, rest: str) -> None:
        job_id = rest.strip()
        if not job_id:
            self.emit("Usage: /approve <job_id>")
            return

        try:
            cur = await self.operator_call("GET", "/operator/interaction")
            if isinstance(cur, dict) and isinstance(cur.get("pending"), list):
                match = next((j for j in cur["pending"] if isinstance(j, dict) and str(j.get("job_id")) == job_id), None)
                if match:
                    appr = match.get("approval") if isinstance(match.get("approval"), dict) else {}
                    reasons = appr.get("reasons") or match.get("reasons") or []
                    cmd = match.get("command") or appr.get("command") or match.get("action")
                    cmd_repr = json.dumps(cmd) if isinstance(cmd, (dict, list)) else str(cmd or "")
                    self.emit(f"Approving job {job_id}: command={cmd_repr}, reasons={reasons}")
        except Exception:
            pass

        try:
            res = await self.operator_call("POST", f"/operator/approvals/{job_id}", {"approved": True})
            if not isinstance(res, dict):
                self.emit(f"Job {job_id} approval returned unexpected response: {res}")
                return

            status = res.get("status") or (res.get("job") or {}).get("status")
            err = res.get("error")
            if status in ("cancelled", "rejected", "failed") or err:
                err_msg = err or res.get("reason") or status
                self.emit(f"Job {job_id} not approved: {status or 'failed'} ({err_msg})")
            else:
                self.emit(f"Job {job_id} approved (status: {status or 'accepted'}).")
        except Exception as exc:
            self.emit(f"Approval call failed: {type(exc).__name__}: {exc}")

    async def _cmd_deny(self, rest: str) -> None:
        job_id = rest.strip()
        if not job_id:
            self.emit("Usage: /deny <job_id>")
            return
        try:
            res = await self.operator_call("POST", f"/operator/approvals/{job_id}", {"approved": False})
            if isinstance(res, dict) and "error" in res:
                self.emit(f"Failed to deny job {job_id}: {res['error']}")
            else:
                status = res.get("status") if isinstance(res, dict) else None
                self.emit(f"Job {job_id} denied (status: {status or 'denied'}).")
        except Exception as exc:
            self.emit(f"Denial call failed: {type(exc).__name__}: {exc}")

    @staticmethod
    def _merge_limits_payload(current_policy: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
        updated_policy = dict(current_policy)
        cur_limits = dict(updated_policy.get("limits") or {})
        cur_auto = dict(updated_policy.get("automatic") or {})

        if "limits" in payload or "automatic" in payload:
            if "limits" in payload and isinstance(payload["limits"], dict):
                cur_limits.update(payload["limits"])
            if "automatic" in payload and isinstance(payload["automatic"], dict):
                cur_auto.update(payload["automatic"])
        else:
            auto_keys = {"max_joint_step_deg", "max_tcp_step_m", "max_effort_protocol"}
            for k, v in payload.items():
                if k in auto_keys:
                    cur_auto[k] = v
                else:
                    cur_limits[k] = v

        updated_policy["limits"] = cur_limits
        updated_policy["automatic"] = cur_auto
        return updated_policy

    async def _cmd_limits(self, rest: str) -> None:
        trimmed = rest.strip()
        if trimmed:
            try:
                new_payload = json.loads(trimmed)
                if not isinstance(new_payload, dict):
                    self.emit("Invalid limits format: expected JSON object.")
                    return
            except Exception as exc:
                self.emit(f"Invalid JSON for /limits: {exc}")
                return

        current = await self._read_interaction()
        if current is None:
            return
        current_policy = current["policy"]
        if trimmed:
            proposed_policy = self._merge_limits_payload(current_policy, new_payload)
            if proposed_policy == current_policy and not self._proposals:
                self.emit("Limits unchanged; nothing to confirm.")
                return

            code = self._generate_proposal_code()
            self._proposals[code] = Proposal(
                code=code,
                action="policy",
                description="Update interaction limits",
                body=proposed_policy,
            )
            self.emit(f"Confirmation required: Proposal {code} - Update interaction limits.")
            self.emit(f"Proposed limits diff:\n{json.dumps({'limits': proposed_policy.get('limits'), 'automatic': proposed_policy.get('automatic')}, indent=2)}")
            self.emit(f"Type /confirm to apply, or /cancel to reject (if multiple pending: /confirm {code}).")
            return

        eff = current.get("effective_limits") or current_policy.get("limits") or {}
        self.emit(f"Effective execution limits: {json.dumps(eff, indent=2)}")
        auto = current_policy.get("automatic") or {}
        if current_policy.get("mode") == "risk":
            self.emit(f"Automatic approval thresholds: {json.dumps(auto, indent=2)}")

        if self.prompt is None:
            self.emit("To update limits, pass a JSON object: /limits {\"limits\": {...}, \"automatic\": {...}}")
            return

        self.emit("Entering interactive limits wizard (press Enter to keep current value):")
        limits = dict(current_policy.get("limits") or {})
        auto = dict(current_policy.get("automatic") or {})
        try:
            self.emit("--- Execution Limits ---")
            speed_cur = limits.get("max_speed_percent", 100)
            speed_val = await self.prompt(f"Max speed percent [{speed_cur}]: ")
            if speed_val.strip():
                limits["max_speed_percent"] = int(speed_val.strip())

            g_max_cur = limits.get("gripper_max_m", 0.07)
            gripper_max = await self.prompt(f"Gripper max m [{g_max_cur}]: ")
            if gripper_max.strip():
                limits["gripper_max_m"] = float(gripper_max.strip())

            g_min_cur = limits.get("gripper_min_m", 0.0)
            gripper_min = await self.prompt(f"Gripper min m [{g_min_cur}]: ")
            if gripper_min.strip():
                limits["gripper_min_m"] = float(gripper_min.strip())

            if current_policy.get("mode") == "risk":
                self.emit("--- Automatic Approval Thresholds ---")
                j_step_cur = auto.get("max_joint_step_deg", "unconfigured")
                j_step = await self.prompt(f"Auto max joint step deg [{j_step_cur}]: ")
                if j_step.strip():
                    auto["max_joint_step_deg"] = float(j_step.strip())

                tcp_step_cur = auto.get("max_tcp_step_m", "unconfigured")
                tcp_step = await self.prompt(f"Auto max TCP step m [{tcp_step_cur}]: ")
                if tcp_step.strip():
                    auto["max_tcp_step_m"] = float(tcp_step.strip())

                auto_spd_cur = auto.get("max_speed_percent", "unconfigured")
                auto_spd = await self.prompt(f"Auto max speed percent (int) [{auto_spd_cur}]: ")
                if auto_spd.strip():
                    auto["max_speed_percent"] = int(auto_spd.strip())

                effort_cur = auto.get("max_effort_protocol", "unconfigured")
                effort = await self.prompt(f"Auto max effort protocol float [{effort_cur}]: ")
                if effort.strip():
                    auto["max_effort_protocol"] = float(effort.strip())

            else:
                self.emit("Automatic thresholds are only used in risk mode; existing values kept.")

            proposed_policy = dict(current_policy)
            proposed_policy["limits"] = limits
            proposed_policy["automatic"] = auto
            if proposed_policy == current_policy and not self._proposals:
                self.emit("Limits unchanged; nothing to confirm.")
                return

            code = self._generate_proposal_code()
            self._proposals[code] = Proposal(
                code=code,
                action="policy",
                description="Update limits via wizard",
                body=proposed_policy,
            )
            self.emit(f"Confirmation required: Proposal {code} - Update limits via wizard.")
            self.emit(f"Proposed policy:\n{json.dumps({'limits': limits, 'automatic': auto}, indent=2)}")
            self.emit(f"Type /confirm to apply, or /cancel to reject (if multiple pending: /confirm {code}).")
        except Exception as exc:
            self.emit(f"Limits wizard cancelled or failed: {type(exc).__name__}")

    async def _cmd_force(self, args: list[str]) -> None:
        if args not in ([], ["on"], ["off"]):
            self.emit("Usage: /config force [on|off]")
            return
        current = await self._read_interaction()
        if current is None:
            return
        policy = dict(current["policy"])
        if not args:
            self.emit(f"Force recovery: {'on' if policy.get('force', False) else 'off'}; existing overrun <=5 degrees, hold or inward only. Every request requires /approve, including auto mode. Controller limits remain enforced.")
            return
        policy["force"] = args[0] == "on"
        code = self._generate_proposal_code()
        self._proposals[code] = Proposal(code, "policy", f"Force recovery {args[0]}", policy)
        self.emit(f"Force recovery {args[0]}: <=5 degree existing overrun; hold or inward only. Each request needs /approve. Type /confirm {code} to apply or /cancel {code}.")

    async def _cmd_config(self, rest: str) -> None:
        if rest.strip().split()[:1] == ["force"]:
            await self._cmd_force(rest.strip().split()[1:])
            return
        if self.managed is None or not hasattr(self.managed, "configure"):
            self.emit("Managed runtime not available for configuration.")
            return

        trimmed = rest.strip()
        if not trimmed:
            try:
                curr = getattr(self.managed, "config", None)
                if curr is None:
                    curr = await self.managed.configure({})
                self.emit(f"Current managed config: {json.dumps(curr, indent=2, default=str)}")
            except Exception as exc:
                self.emit(f"Failed to inspect configuration: {type(exc).__name__}")
            return

        try:
            cfg = json.loads(trimmed)
            if not isinstance(cfg, dict):
                self.emit("Invalid config format: expected JSON object.")
                return
        except Exception as exc:
            self.emit(f"Invalid JSON for /config: {exc}")
            return

        code = self._generate_proposal_code()
        self._proposals[code] = Proposal(
            code=code,
            action="config",
            description="Update managed configuration",
            body=cfg,
        )
        self.emit(f"Confirmation required: Proposal {code} - Update managed configuration.")
        self.emit(f"Proposed config: {json.dumps(cfg, indent=2)}")
        self.emit(f"Type /confirm to apply, or /cancel to reject (if multiple pending: /confirm {code}).")

    async def _cmd_remote(self, rest: str) -> None:
        if self.managed is None or not hasattr(self.managed, "remotes"):
            self.emit("Managed runtime not available for remotes.")
            return

        trimmed = rest.strip()
        if not trimmed or trimmed == "list":
            try:
                remotes = await self.managed.remotes()
                if not remotes:
                    self.emit("No remote targets configured.")
                else:
                    self.emit(f"Configured remote targets ({len(remotes)}):")
                    for r in remotes:
                        self.emit(f"  - {r.get('name')}: {r.get('ssh_host')}")
            except Exception as exc:
                self.emit(f"Failed to list remotes: {type(exc).__name__}")
            return

        parts = trimmed.split()
        if parts[0] == "add":
            if len(parts) < 3:
                self.emit("Usage: /remote add <name> <ssh_host>")
                return
            name = parts[1]
            ssh_host = parts[2]
            try:
                await self.managed.save_remote(name, ssh_host)
                self.emit(f"Saved remote target '{name}' ({ssh_host}).")
            except Exception as exc:
                self.emit(f"Failed to save remote: {type(exc).__name__}: {exc}")
            return

        self.emit(f"Unknown /remote subcommand '{parts[0]}'. Usage: /remote or /remote add <name> <ssh_host>")

    async def _cmd_mode(self, rest: str) -> None:
        if not rest.strip():
            self.emit(f"Current mode: {self.mode}, target: {self.target}")
            return
        parts = rest.strip().split()
        new_mode = parts[0].lower()
        if new_mode not in ("simulation", "real"):
            self.emit(f"Invalid mode '{new_mode}'. Allowed: simulation, real.")
            return
        new_target = parts[1] if len(parts) > 1 else "local"

        if self.managed is None or not hasattr(self.managed, "switch"):
            self.emit("Managed runtime not available for mode switching.")
            return

        if self.background_task and not self.background_task.done():
            self.emit("Cannot switch mode while a background task is running. Wait for completion or use /stop first.")
            return

        self.emit(f"Switching mode to {new_mode} on target '{new_target}'...")
        try:
            await self.managed.switch(new_mode, new_target)
            self.mode = new_mode
            self.target = new_target
            self._proposals.clear()
            self.emit(f"Switched to {new_mode} mode on target '{new_target}'.")
        except Exception as exc:
            self.emit(f"Failed to switch mode: {type(exc).__name__}: {exc}")

    async def _run_lifecycle_shutdown(self) -> None:
        try:
            if self.managed is not None and hasattr(self.managed, "shutdown"):
                await self.managed.shutdown()

            if self.background_task and not self.background_task.done():
                try:
                    await asyncio.wait_for(asyncio.shield(self.background_task), timeout=5.0)
                except (asyncio.TimeoutError, asyncio.CancelledError, Exception):
                    pass

            await self.close()
            self.exit_requested = True
            self.emit("Shutdown complete.")
        except Exception as exc:
            self._draining = False
            self.emit(f"Managed shutdown error: {type(exc).__name__}: {exc}")

    async def _cmd_shutdown(self) -> bool:
        if self._draining:
            self.emit("Console is already shutting down.")
            return True
        self._draining = True
        self.emit("Shutting down managed executor (draining in-flight actions)...")
        self._drain_task = asyncio.create_task(self._run_lifecycle_shutdown())
        return True

    async def _cmd_quit(self) -> bool:
        if self._draining:
            self.emit("Console is already shutting down.")
            return True
        self._draining = True
        self.emit("Exiting console (draining in-flight actions)...")
        self._drain_task = asyncio.create_task(self._run_lifecycle_shutdown())
        return True

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

        if reconnect or self._session_lost:
            if self.managed is not None and hasattr(self.managed, "reconnect"):
                try:
                    self.emit("Reacquiring control session via managed runtime...")
                    await self.managed.reconnect()
                    self._session_lost = False
                    self.emit("Control session reacquired.")
                except Exception as exc:
                    self.emit(f"Managed reconnect failed: {type(exc).__name__}: {exc}")
                    return
            else:
                self._session_lost = False

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
        connected_id = res.get("connected_device_id") or res.get("connected_id")

        if not target_id:
            if connected_id:
                target_id = connected_id
                self.emit(f"Reusing currently connected device: {target_id}")
            elif not devices:
                self.emit("No devices discovered on executor host.")
                return
            elif len(devices) == 1:
                target_id = devices[0].get("device_id") or devices[0].get("id")
                self.emit(f"Auto-selected device: {target_id}")
            else:
                if self.prompt is not None:
                    self.emit(f"Discovered {len(devices)} devices:")
                    for d in devices:
                        dev_id = d.get("device_id") or d.get("id")
                        self.emit(f"  - {dev_id}: {d.get('label', 'unlabeled')}")
                    default_dev = devices[0].get("device_id") or devices[0].get("id") or ""
                    try:
                        chosen = await self.prompt(f"Select device ID [{default_dev}]: ", default=default_dev)
                        target_id = chosen.strip() or default_dev
                    except Exception:
                        target_id = default_dev
                else:
                    self.emit(f"Discovered {len(devices)} devices:")
                    for d in devices:
                        dev_id = d.get("device_id") or d.get("id")
                        status_tag = " (connected)" if d.get("connected") or dev_id == connected_id else ""
                        self.emit(f"  - {dev_id}: {d.get('label', 'unlabeled')}{status_tag}")
                    self.emit("Specify device ID to connect: /connect <device_id>")
                    return

        is_real = self.mode == "real" or (
            isinstance(self.cached_status, dict) and self.cached_status.get("backend") == "real"
        )
        if is_real and self.prompt is not None:
            try:
                ans = await self.prompt("Target is real robot. Confirm workspace is clear and ready to connect? (yes/no) [no]: ", default="no")
                if ans.strip().lower() not in ("yes", "y"):
                    self.emit("Connection cancelled by operator.")
                    return
            except Exception:
                self.emit("Connection cancelled by operator.")
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


ManagedConsoleController = InteractiveConsoleController
