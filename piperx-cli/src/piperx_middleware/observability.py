"""Observability helpers: whitelist parsing, safe ID validation, and parameters view."""
import re
from urllib.parse import unquote

from .models import JOINT_LIMITS_DEG

SAFE_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]+$")

CALLS_SEMANTICS = (
    "Executor operation call records track call acceptance/dispatch outcome, "
    "not physical motion completion; use job_id to inspect completion. "
    "Status indicates call execution success or failure; execution job status is separate."
)


def is_safe_id(val: str | None) -> bool:
    if not val or not isinstance(val, str):
        return False
    if len(val) > 128:
        return False
    return bool(SAFE_ID_RE.fullmatch(val))


def match_whitelisted_operation(method: str, path: str):
    """
    Match and whitelist REST and MCP operations.
    Returns (operation, normalized_path, safe_request_id, safe_job_id) or None.

    Whitelisted operations:
      - GET /v1/state
      - GET /v1/capabilities
      - POST /v1/connect
      - POST /v1/disconnect
      - POST /v1/move
      - POST /v1/primitives
      - POST /v1/stop
      - GET /v1/jobs/{ident} (safe ident only)
      - GET /v1/requests/{request_id} (safe request_id only)

    Explicitly excluded from recording:
      - /v1/parameters
      - /v1/jobs (inspection polling)
      - /v1/calls (inspection polling)
      - /v1/events
      - /health, /openapi.json, /mcp, and any unknown path.
    """
    clean_path = path
    extra = {("GET", "/v1/diagnostics"), ("POST", "/v1/primitives/preview"),
             ("POST", "/operator/query-limits"),
             ("POST", "/operator/estop"), ("POST", "/operator/clear-estop"),
             ("PATCH", "/operator/parameters"), ("POST", "/operator/sim-fault")}
    if (method, path) in extra:
        return path.rsplit("/", 1)[-1], path, None, None
    if method == "GET":
        if clean_path == "/v1/state":
            return "state", "/v1/state", None, None
        if clean_path == "/v1/capabilities":
            return "capabilities", "/v1/capabilities", None, None
        if path.startswith("/v1/jobs/"):
            ident = unquote(path.removeprefix("/v1/jobs/"))
            if ident and is_safe_id(ident):
                return "job", f"/v1/jobs/{ident}", None, ident
            return None
        if path.startswith("/v1/requests/"):
            ident = unquote(path.removeprefix("/v1/requests/"))
            if ident and is_safe_id(ident):
                return "request", f"/v1/requests/{ident}", ident, None
            return None
    elif method == "POST":
        if clean_path == "/v1/connect":
            return "connect", "/v1/connect", None, None
        if clean_path == "/v1/disconnect":
            return "disconnect", "/v1/disconnect", None, None
        if clean_path == "/v1/control-mode":
            return "control_mode", "/v1/control-mode", None, None
        if clean_path == "/v1/move":
            return "move", "/v1/move", None, None
        if clean_path == "/v1/primitives":
            return "primitives", "/v1/primitives", None, None
        if clean_path == "/v1/stop":
            return "stop", "/v1/stop", None, None
    return None


def get_parameters_response(service) -> dict:
    settings = service.settings
    configured = {
        "backend": settings.backend,
        "control_profile": settings.control_profile,
        "can_interface": settings.can_interface,
        "can_channel": settings.can_channel,
        "firmware_profile": settings.firmware_profile,
        "tcp_offset_m": list(settings.tcp_offset_m),
        "tcp_offset_rpy_deg": list(settings.tcp_offset_rpy_deg),
        "joint_limits_deg": [list(lim) for lim in JOINT_LIMITS_DEG],
        "gripper_max_m": float(settings.gripper_max_m),
    }
    reported = service.state()
    notes = [
        "Configured firmware profile reflects local middleware configuration and is not a live firmware query.",
        "Reported state data is cached from background feedback/service state and may be stale.",
        "No active SDK or CAN bus queries are performed during parameter inspection.",
    ]
    return {
        "configured": configured,
        "reported": reported,
        "notes": notes,
    }


def command_operation(path, kind, default):
    """Name validated action variants without recording arbitrary input."""
    if path == "/v1/primitives" and kind in ("move_to", "move_by", "rotate", "set_gripper", "move_linear"):
        return kind
    if path == "/v1/move" and kind in ("joint", "gripper"):
        return {"joint": "robot_move_joints", "gripper": "robot_gripper"}[kind]
    return default
