"""Human-readable renderers for the read-only inspection commands.

Telemetry is only as fresh as the server reports: missing or null fields are
rendered as unknown, and nothing is presented as live unless the server says
the feedback is live. Renderers must never turn defaults into apparent health.
"""

from datetime import datetime, timezone
import json
import math


def _val(value):
    return "unknown" if value is None else str(value)


def _bool(value):
    if value is None:
        return "unknown"
    if isinstance(value, bool):
        return "yes" if value else "no"
    return str(value)


def _num(value, unit="", digits=3):
    if value is None:
        return "unknown"
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return "invalid"
    text = f"{value:.{digits}f}"
    if digits:
        text = text.rstrip("0").rstrip(".")
    if text in ("", "-0"):
        text = "0"
    return f"{text} {unit}".rstrip()


def _elem(value, digits=3):
    return _bool(value) if isinstance(value, bool) else _num(value, digits=digits)


def _list(values, digits=3):
    if not isinstance(values, (list, tuple)) or not values:
        return "unknown"
    return "[" + ", ".join(_elem(v, digits) for v in values) + "]"


def _at(epoch):
    if isinstance(epoch, bool) or not isinstance(epoch, (int, float)) or not math.isfinite(epoch):
        return "unknown-time"
    return datetime.fromtimestamp(epoch, tz=timezone.utc).astimezone().isoformat(timespec="seconds")


def render_status(state: dict) -> str:
    """Render a GET /v1/state payload; null telemetry stays visibly unknown."""
    robot = state.get("robot") or {}
    connection = state.get("connection") or {}
    lines = []
    backend = _val(state.get("backend"))
    lines.append(f"Backend: {backend}" + (" (simulation; no hardware)" if backend == "sim" else ""))
    lines.append(f"Connection: {_val(connection.get('status'))} "
                 f"(opened={_bool(connection.get('opened'))}, "
                 f"feedback_live={_bool(connection.get('feedback_live'))}, "
                 f"epoch={_val(state.get('connection_epoch'))})")
    if connection.get("reconnect_recommended"):
        lines.append("  The server recommends reconnecting before relying on this state.")
    lines.append("Ready: " + _bool(state.get("ready")))
    reason = state.get("not_ready_reason")
    if isinstance(reason, dict):
        lines.append(f"  Not ready: {_val(reason.get('code'))}: {reason.get('message', '')}".rstrip())
    lines.append("Feedback age: " + _num(robot.get("feedback_age_s"), "s"))
    if not connection.get("feedback_live"):
        lines.append("Telemetry below is not live; treat every value as stale or unavailable.")
    lines.append("Joints (deg): " + _list(robot.get("q_deg")))
    lines.append("Motor velocity telemetry (deg/s; joint speed unvalidated): " + _list(robot.get("velocity_deg_s")))
    lines.append("Joints enabled: " + _list(robot.get("enabled")))
    lines.append(f"Modes: ctrl={_val(robot.get('ctrl_mode'))}, motion={_val(robot.get('motion_mode'))}, "
                 f"teach={_val(robot.get('teach_status'))}, arm_status={_val(robot.get('arm_status'))}, "
                 f"error_code={_val(robot.get('error_code'))}")
    lines.append(f"Gripper: width={_num(robot.get('gripper_width_m'), 'm')}, "
                 f"enabled={_bool(robot.get('gripper_enabled'))}, "
                 f"error={_bool(robot.get('gripper_error'))}, "
                 f"mode={_val(robot.get('gripper_mode'))}, "
                 f"age={_num(robot.get('gripper_age_s'), 's')}")
    tcp = state.get("tcp")
    if isinstance(tcp, dict):
        lines.append(f"TCP ({_val(tcp.get('frame'))} frame): xyz_m={_list(tcp.get('xyz_m'))}, "
                     f"rpy_deg={_list(tcp.get('rpy_deg'))}")
        lines.append(f"  source={_val(tcp.get('source'))}, "
                     f"feedback_age={_num(tcp.get('feedback_age_s'), 's')}, "
                     f"live={_bool(tcp.get('feedback_live'))}")
    else:
        lines.append("TCP: unavailable (no valid joint feedback)")
    lines.append(f"Diagnostics: received_frames={_val(robot.get('received_frames'))}, "
                 f"tx_frames={_val(robot.get('tx_frames'))}")
    diagnostics = robot.get("diagnostics") or {}
    if diagnostics:
        lines.append("  " + ", ".join(f"{key}={diagnostics[key]}" for key in sorted(diagnostics)))
    lines.append("Active job: " + (_val(state.get("active_job_id")) if state.get("active_job_id") else "none"))
    window = state.get("control_window")
    if isinstance(window, dict):
        lines.append(f"Control window: remaining={_num(window.get('remaining_s'), 's')}, "
                     f"radius={_num(window.get('radius_deg'), 'deg')}, "
                     f"gripper={_bool(window.get('allow_gripper'))}")
    else:
        lines.append("Control window: none")
    return "\n".join(lines)


def render_tools(capabilities: dict) -> str:
    """Render GET /v1/capabilities, centered on the exposed MCP tools."""
    tools = capabilities.get("mcp_tools") or []
    lines = [f"MCP tools ({len(tools)}):"]
    lines.extend(f"  - {tool}" for tool in tools)
    backend = _val(capabilities.get("backend"))
    lines.append(f"Backend: {backend}" + (" (simulation)" if capabilities.get("simulation") else ""))
    lines.append(f"Motion configured: {_bool(capabilities.get('hardware_motion_configured'))}, "
                 f"control profile: {_val(capabilities.get('control_profile'))}, "
                 f"control window required: {_bool(capabilities.get('requires_control_window'))}")
    excluded = capabilities.get("excluded") or []
    if excluded:
        lines.append("Explicitly not exposed: " + ", ".join(str(item) for item in excluded))
    return "\n".join(lines)


_CONFIGURED_ORDER = ("backend", "control_profile", "can_interface", "can_channel", "firmware_profile",
                     "tcp_offset_m", "tcp_offset_rpy_deg", "joint_limits_deg", "gripper_max_m")


def render_params(payload: dict) -> str:
    """Render GET /v1/parameters with configuration and reported state distinct."""
    configured = payload.get("configured") or {}
    lines = ["Configured parameters (static server settings, not queried from the device):"]
    for key in _CONFIGURED_ORDER:
        if key not in configured:
            continue
        value = configured[key]
        text = json.dumps(value) if isinstance(value, (list, dict)) else str(value)
        if key == "firmware_profile":
            text += " (configured profile; the device firmware version is not queried)"
        lines.append(f"  {key}: {text}")
    for key in sorted(k for k in configured if k not in _CONFIGURED_ORDER):
        value = configured[key]
        lines.append(f"  {key}: {json.dumps(value) if isinstance(value, (list, dict)) else value}")
    lines.append("")
    lines.append("Reported state (cached feedback; check age and connection):")
    reported = payload.get("reported")
    if isinstance(reported, dict):
        lines.extend("  " + line for line in render_status(reported).splitlines())
    else:
        lines.append("  unavailable")
    notes = payload.get("notes") or []
    if notes:
        lines.append("")
        lines.append("Notes:")
        lines.extend(f"  - {note}" for note in notes)
    return "\n".join(lines)


def render_calls(payload: dict) -> str:
    """Render GET /v1/calls; records are operation logs, not physical proof."""
    calls = payload.get("calls") or []
    lines = [f"Recorded calls (newest first, {len(calls)} shown):"]
    if payload.get("semantics"):
        lines.append(f"  Semantics: {payload['semantics']}")
    lines.append("  These are executor operation records (REST+MCP), not proof of physical completion; "
                 "job status is the execution result.")
    for call in calls:
        if not isinstance(call, dict):
            continue
        line = f"  [{_at(call.get('at'))}] {_val(call.get('method'))} {_val(call.get('path'))} -> {_val(call.get('status'))}"
        if call.get("duration_ms") is not None:
            line += f" in {_num(call.get('duration_ms'), 'ms', 1)}"
        lines.append(line)
        details = [f"{key}={call[key]}" for key in ("source", "operation", "id", "request_id", "job_id", "error_code")
                   if call.get(key) is not None]
        if details:
            lines.append("    " + ", ".join(details))
    if not calls:
        lines.append("  (none recorded)")
    return "\n".join(lines)


def render_jobs(payload: dict) -> str:
    """Render GET /v1/jobs; job status is the execution result."""
    jobs = payload.get("jobs") or []
    lines = [f"Jobs (newest first, {len(jobs)} shown):"]
    lines.append("  Job status is the execution result; a recorded call alone is not proof of completion.")
    for job in jobs:
        if not isinstance(job, dict):
            continue
        lines.append(f"  {_val(job.get('job_id'))}: {_val(job.get('status'))}")
        details = []
        command = job.get("command")
        if isinstance(command, dict):
            details.append("command=" + _val(command.get("kind")))
        if job.get("request_id"):
            details.append(f"request_id={job['request_id']}")
        if job.get("accepted_at") is not None:
            details.append("accepted=" + _at(job.get("accepted_at")))
        if job.get("finished_at") is not None:
            details.append("finished=" + _at(job.get("finished_at")))
        if job.get("simulation"):
            details.append("simulation")
        if job.get("error"):
            details.append(f"error={job['error']}")
        if details:
            lines.append("    " + ", ".join(str(d) for d in details))
    if not jobs:
        lines.append("  (none)")
    return "\n".join(lines)
