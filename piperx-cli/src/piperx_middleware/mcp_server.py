"""Shared tool definitions for stdio and Streamable HTTP; no hardware owner here."""
import json
import logging
import time
from typing import Annotated, Literal
from urllib.parse import quote, unquote

import anyio
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError
from mcp.server.transport_security import TransportSecuritySettings
from mcp.types import ToolAnnotations
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .client import RobotClient
from .models import ControlMode, ControlModeRequest, DomainError, MoveRequest, PrimitiveRequest
from .observability import command_operation, is_safe_id, match_whitelisted_operation


class ConnectOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")
    reconnect: bool = Field(default=False, strict=True)
    device_id: str | None = Field(default=None, strict=True, min_length=1, max_length=128)


class ServiceClient:
    """In-process equivalent of RobotClient, dispatching to the existing service."""
    def __init__(self, service):
        self.service = service

    def call(self, method, path, body=None):
        t0 = time.time()
        started = time.monotonic()
        op_info = match_whitelisted_operation(method, path)
        status = "failed"
        error_code = None
        action_kind = None
        req_id = None
        job_id = None

        if isinstance(body, dict):
            raw_req_id = body.get("request_id")
            if is_safe_id(raw_req_id):
                req_id = raw_req_id

        if op_info:
            _, _, path_req_id, path_job_id = op_info
            if path_req_id:
                req_id = path_req_id
            if path_job_id:
                job_id = path_job_id

        try:
            if method == "GET":
                if path == "/v1/state":
                    res = self.service.state()
                elif path == "/v1/capabilities":
                    res = self.service.capabilities()
                elif path == "/v1/diagnostics":
                    res = self.service.diagnostics()
                elif path.startswith("/v1/jobs/"):
                    res = self.service.get_job(unquote(path.removeprefix("/v1/jobs/")))
                elif path.startswith("/v1/requests/"):
                    res = self.service.get_request(unquote(path.removeprefix("/v1/requests/")))
                else:
                    raise DomainError("not_found", "Unknown service operation.", 404)
            elif method == "POST":
                if path == "/v1/connect":
                    options = ConnectOptions.model_validate(body or {})
                    res = self.service.connect(reconnect=options.reconnect, **({"device_id": options.device_id} if options.device_id is not None else {}))
                elif path == "/v1/disconnect":
                    res = self.service.disconnect()
                elif path == "/v1/control-mode":
                    request = ControlModeRequest.model_validate(body)
                    action_kind = "control_mode"
                    res = self.service.move(ControlMode(**request.model_dump(exclude={"request_id"})), request.request_id)
                elif path == "/v1/move":
                    request = MoveRequest.model_validate(body)
                    action_kind = request.command.kind
                    res = self.service.move(request.command, request.request_id)
                elif path == "/v1/primitives":
                    request = PrimitiveRequest.model_validate(body)
                    action_kind = request.command.kind
                    res = self.service.primitive(request.command, request.request_id)
                elif path == "/v1/stop":
                    res = self.service.stop()
                else:
                    raise DomainError("not_found", "Unknown service operation.", 404)
            else:
                raise DomainError("not_found", "Unknown service operation.", 404)

            if isinstance(res, dict) and isinstance(res.get("error"), dict):
                status = "failed"
                error_code = res["error"].get("code", "unknown_error")
            else:
                status = "succeeded"
                if isinstance(res, dict):
                    if not job_id and is_safe_id(res.get("job_id")):
                        job_id = res["job_id"]
                    if not req_id and is_safe_id(res.get("request_id")):
                        req_id = res["request_id"]
            return res
        except DomainError as exc:
            error_code = exc.code
            return {"error": {"code": exc.code, "message": exc.message}}
        except ValidationError:
            error_code = "invalid_request"
            return {"error": {"code": "invalid_request", "message": "Request does not match the schema."}}
        except Exception:
            error_code = "internal_error"
            raise
        finally:
            if op_info and hasattr(self.service, "store") and self.service.store:
                duration_ms = round((time.monotonic() - started) * 1000.0, 2)
                op, norm_path, _, _ = op_info
                try:
                    self.service.store.record_call(
                        at=t0,
                        source="mcp",
                        operation=command_operation(path, action_kind, op),
                        method=method,
                        path=norm_path,
                        status=status,
                        duration_ms=duration_ms,
                        request_id=req_id,
                        job_id=job_id,
                        error_code=error_code,
                    )
                except Exception:
                    logging.getLogger(__name__).warning("Call audit unavailable; operation response preserved.")


RequestId = Annotated[str, Field(min_length=8, max_length=128, pattern=r"^[A-Za-z0-9_.:-]+$")]
SpeedPercent = Annotated[int, Field(ge=0, le=100, strict=True)]
PositiveSeconds = Annotated[float, Field(gt=0, strict=True, allow_inf_nan=False)]
JointAngles = Annotated[list[Annotated[float, Field(strict=True, allow_inf_nan=False)]], Field(min_length=6, max_length=6)]
Vec3Meters = Annotated[list[Annotated[float, Field(strict=True, allow_inf_nan=False)]], Field(min_length=3, max_length=3)]
Vec3Degrees = Annotated[list[Annotated[float, Field(strict=True, allow_inf_nan=False)]], Field(min_length=3, max_length=3)]
GripperWidthM = Annotated[float, Field(ge=0, le=0.09, strict=True, allow_inf_nan=False)]
EffortProtocol = Annotated[float, Field(ge=0, le=32.767, strict=True, allow_inf_nan=False)]


def create_mcp(client: RobotClient | ServiceClient, *, allowed_http_hosts: list[str] | None = None, workspace=None):
    async def call(method, path, body=None):
        # SDK tool callbacks otherwise execute synchronous functions on its event
        # loop. Keep service/HTTP I/O off that loop and finish it before teardown.
        result = await anyio.to_thread.run_sync(client.call, method, path, body)
        # A completed job can contain an `error` string. That is a readable
        # execution outcome, not the API's {error: {code, message}} envelope.
        if isinstance(result.get("error"), dict):
            raise ToolError(json.dumps(result, ensure_ascii=False))
        return result

    security = TransportSecuritySettings(allowed_hosts=allowed_http_hosts) if allowed_http_hosts is not None else None
    mcp = FastMCP("piperx", stateless_http=True, json_response=True, transport_security=security, instructions=(
        "Read robot_status first. Joints are six absolute angles in degrees; gripper width is metres. "
        "Use robot_connect to establish feedback and robot_disconnect to release the connection. "
        "Keep request_id unchanged on retries. Query robot_status(job_id=...) for completion. "
        "Unknown action outcomes must not be replayed. "
        "move_to, move_by and rotate target the TCP in base-frame metres and RPY degrees via the "
        "fixed numerical IK solver; the arm follows a joint-space endpoint move, not a Cartesian straight line, "
        "with no collision checking. set_gripper sets the opening width in metres."))

    @mcp.tool(annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True))
    async def robot_status(job_id: str | None = None, request_id: str | None = None) -> dict:
        """Read connection diagnostics, robot state, or a job/request result. Does not connect or send commands."""
        if job_id and request_id:
            raise ToolError("Specify job_id or request_id, not both.")
        if job_id:
            return await call("GET", "/v1/jobs/" + quote(job_id, safe=""))
        if request_id:
            return await call("GET", "/v1/requests/" + quote(request_id, safe=""))
        state = await call("GET", "/v1/state")
        return state | {"capabilities": await call("GET", "/v1/capabilities")}

    @mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False))
    async def robot_connect(reconnect: Annotated[bool, Field(strict=True)] = False, device_id: str | None = None) -> dict:
        """Connect and report feedback readiness. reconnect=True closes the current connection first. Does not move the robot."""
        return await call("POST", "/v1/connect", {"reconnect": reconnect, **({"device_id": device_id} if device_id is not None else {})})

    @mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=True))
    async def robot_disconnect() -> dict:
        """Release the robot connection when idle. Does not stop a physical motion or send movement commands."""
        return await call("POST", "/v1/disconnect")

    @mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=True, idempotentHint=True))
    async def robot_set_control_mode(request_id: RequestId, speed_percent: SpeedPercent = 5, timeout_s: PositiveSeconds = 3) -> dict:
        """Explicitly select CAN/MOVE_J, preloading the measured pose. Requires live, enabled, fault-free feedback and inactive teaching. No reset, enable or homing. Returns job_id; verify its final result. Keep request_id on retries."""
        return await call("POST", "/v1/control-mode", {"request_id": request_id,
            "speed_percent": speed_percent, "timeout_s": timeout_s})

    @mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=True, idempotentHint=True))
    async def robot_move_joints(joints_deg: JointAngles, request_id: RequestId, speed_percent: SpeedPercent = 5, timeout_s: PositiveSeconds = 30) -> dict:
        """Move to six ABSOLUTE joint angles in DEGREES. Use a unique request_id, unchanged on retries. Returns job_id, not completion."""
        return await call("POST", "/v1/move", {"request_id": request_id, "command": {"kind": "joint",
            "joints_deg": joints_deg, "speed_percent": speed_percent, "timeout_s": timeout_s}})

    @mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=True, idempotentHint=True))
    async def robot_gripper(width_m: GripperWidthM, request_id: RequestId, effort_protocol: EffortProtocol = .5, timeout_s: PositiveSeconds = 10) -> dict:
        """Move gripper to absolute opening width in METRES. Effort is an uncalibrated SDK field. Keep request_id on retries; returns job_id."""
        return await call("POST", "/v1/move", {"request_id": request_id, "command": {"kind": "gripper",
            "width_m": width_m, "effort_protocol": effort_protocol, "timeout_s": timeout_s}})

    @mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=True, idempotentHint=True))
    async def robot_stop() -> dict:
        """Cancel the current action and request a fresh-position hold. This is not a hardware emergency stop."""
        return await call("POST", "/v1/stop")

    @mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=True, idempotentHint=True))
    async def move_to(xyz_m: Vec3Meters, request_id: RequestId, rpy_deg: Vec3Degrees | None = None,
                      speed_percent: SpeedPercent = 5, timeout_s: PositiveSeconds = 30) -> dict:
        """Move the TCP to an ABSOLUTE base-frame position in METRES. Optional rpy_deg is an absolute orientation
        in DEGREES (extrinsic xyz RPY: Rz@Ry@Rx); omit it to keep the measured TCP orientation. The target is
        solved analytically, then executed as a joint-space endpoint move, not a Cartesian straight line. No
        collision checking. Keep request_id unchanged on retries; returns job_id, not completion."""
        return await call("POST", "/v1/primitives", {"request_id": request_id, "command": {"kind": "move_to",
            "xyz_m": xyz_m, "rpy_deg": rpy_deg, "speed_percent": speed_percent, "timeout_s": timeout_s}})

    @mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=True, idempotentHint=True))
    async def move_by(delta_m: Vec3Meters, request_id: RequestId, frame: Literal["base", "tcp"] = "base",
                      speed_percent: SpeedPercent = 5, timeout_s: PositiveSeconds = 30) -> dict:
        """Translate the TCP by a RELATIVE offset in METRES. frame='base' adds the offset along base axes;
        frame='tcp' rotates it by the measured TCP orientation first. The TCP orientation is preserved.
        Executed as a joint-space endpoint move, not a Cartesian straight line. No collision checking.
        Keep request_id unchanged on retries; returns job_id, not completion."""
        return await call("POST", "/v1/primitives", {"request_id": request_id, "command": {"kind": "move_by",
            "delta_m": delta_m, "frame": frame, "speed_percent": speed_percent, "timeout_s": timeout_s}})

    @mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=True, idempotentHint=True))
    async def rotate(rpy_deg: Vec3Degrees, request_id: RequestId, speed_percent: SpeedPercent = 5,
                     timeout_s: PositiveSeconds = 30) -> dict:
        """Rotate the TCP to an ABSOLUTE base-frame orientation in DEGREES (extrinsic xyz RPY: Rz@Ry@Rx) while
        preserving the measured TCP position. Executed as a joint-space endpoint move. No collision checking.
        Keep request_id unchanged on retries; returns job_id, not completion."""
        return await call("POST", "/v1/primitives", {"request_id": request_id, "command": {"kind": "rotate",
            "rpy_deg": rpy_deg, "speed_percent": speed_percent, "timeout_s": timeout_s}})

    @mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=True, idempotentHint=True))
    async def set_gripper(width_m: GripperWidthM, request_id: RequestId, effort_protocol: EffortProtocol = .5,
                          timeout_s: PositiveSeconds = 10, completion: Literal['width','bilateral_contact']='width') -> dict:
        """Set the gripper to an absolute opening width in METRES. The width is continuous; reaching it is not
        a grasp success claim. Bilateral contact completion requires the MuJoCo force sensor and a closing move;
        it confirms contact only, not a successful lift. Effort is an uncalibrated SDK field. Keep request_id on retries; returns job_id."""
        return await call("POST", "/v1/primitives", {"request_id": request_id, "command": {"kind": "set_gripper",
            "width_m": width_m, "effort_protocol": effort_protocol, "timeout_s": timeout_s,'completion':completion}})

    @mcp.tool(annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True))
    async def robot_diagnostics() -> dict:
        """Read collision flags, temperatures, freshness, stop latch and configured/measured limit provenance."""
        return await call("GET", "/v1/diagnostics")

    @mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=True, idempotentHint=True))
    async def move_linear(xyz_m: Vec3Meters, request_id: RequestId,
                          speed_percent: SpeedPercent = 5, timeout_s: PositiveSeconds = 30) -> dict:
        """Fixed-orientation sampled TCP straight line in base metres. All IK samples are checked before execution;
        joints are interpolated between samples. No continuous-path collision or physical accuracy guarantee.
        Keep request_id on retries; query the returned job for its result."""
        return await call("POST", "/v1/primitives", {"request_id": request_id, "command": {
            "kind": "move_linear", "xyz_m": xyz_m, "speed_percent": speed_percent, "timeout_s": timeout_s}})

    if workspace is not None:
        from pathlib import Path
        from .simulation_camera import capture
        from mcp.server.fastmcp import Image
        from mcp.types import TextContent
        root = Path(workspace).resolve(strict=True)
        if not root.is_dir():
            raise ValueError('workspace must be an existing directory')

        @mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, openWorldHint=False))
        async def simulation_observe(output: str) -> list:
            """Capture current MuJoCo RGB-D and camera metadata into a new workspace folder; returns pixels, never hidden object state or scoring."""
            result = await anyio.to_thread.run_sync(lambda: capture(client, root, output))
            return [TextContent(type='text', text=json.dumps(result)),
                    Image(path=str(Path(result['evidence'])/'rgb.jpg')).to_image_content()]

    return mcp


def main(argv=None):
    import argparse
    from pathlib import Path
    parser = argparse.ArgumentParser(description="PiperX MCP stdio client for a shared executor; never opens CAN")
    parser.add_argument("--url", help="Executor URL; defaults to PIPERX_URL or http://127.0.0.1:8765")
    parser.add_argument("--root", type=Path, help="Local executor data directory (for its token file)")
    parser.add_argument("--token-file", type=Path, help="Model token file; value is never printed")
    parser.add_argument("--workspace", type=Path, help="Enable RGB-D capture inside this existing directory")
    args = parser.parse_args(argv)
    client = RobotClient.from_env(url=args.url, token_file=args.token_file, root=args.root)
    try:
        create_mcp(client, workspace=args.workspace).run(transport="stdio")
    finally:
        client.close()


if __name__ == "__main__":
    main()
