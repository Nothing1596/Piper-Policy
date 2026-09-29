from contextlib import asynccontextmanager
from functools import partial
import secrets
import logging
import time

import anyio
from fastapi import BackgroundTasks, Depends, FastAPI, Header, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.datastructures import Headers
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .models import ControlMode, ControlModeRequest, DomainError, ExecuteRequest, LeaseRequest, MoveRequest, PreviewRequest, PrimitiveRequest, ShutdownRequest
from .service import RobotService
from .request_context import control_session_id
from .interaction_types import SessionAcquire, SessionReference, ApprovalDecision, PolicyUpdate, ProfileSettingsUpdate
from .models import PrimitivePreviewRequest, RuntimeParameters, SimFault
from .piper_aio import AioAction, describe
from .mcp_server import ConnectOptions, ServiceClient, create_mcp
from .observability import CALLS_SEMANTICS, command_operation, get_parameters_response, is_safe_id, match_whitelisted_operation


class _MCPBearerAuth:
    """Authenticate the whole mounted app, including when ASGI root_path changes."""
    def __init__(self, app, token):
        self.app, self.token = app, token

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            authorization = Headers(scope=scope).get("authorization", "")
            if not secrets.compare_digest(authorization, "Bearer " + self.token):
                response = JSONResponse(
                    {"error": {"code": "unauthorized", "message": "A valid bearer token is required."}},
                    status_code=401, headers={"WWW-Authenticate": "Bearer"})
                await response(scope, receive, send)
                return
        await self.app(scope, receive, send)


def create_app(service: RobotService, model_token: str, operator_token: str, on_shutdown=None) -> FastAPI:
    if min(len(model_token), len(operator_token)) < 32 or model_token == operator_token:
        raise ValueError("Use distinct model and operator tokens of at least 32 characters.")
    mcp = create_mcp(ServiceClient(service), allowed_http_hosts=[
        "localhost:*", "127.0.0.1:*", "[::1]:*", "testserver", service.settings.host])
    mcp_app = mcp.streamable_http_app()

    @asynccontextmanager
    async def lifespan(app):
        try:
            async with mcp.session_manager.run():
                yield
        finally:
            camera_pool = getattr(app.state, "simulation_camera_pool", None)
            if camera_pool is not None:
                camera_pool.submit(service.backend.close_renderer).result(timeout=10)
                camera_pool.shutdown(wait=True, cancel_futures=True)
            await anyio.to_thread.run_sync(service.close)

    app = FastAPI(title="PiperX middleware", version="0.7.0", lifespan=lifespan,
                  docs_url=None, redoc_url=None, openapi_url=None)
    app.state.mcp = mcp
    service.on_idle_session_loss = on_shutdown
    app.add_middleware(TrustedHostMiddleware,
                       allowed_hosts=["localhost", "127.0.0.1", "[::1]", service.settings.host, "testserver"]
                       )

    def authorize(authorization, expected):
        if not authorization or not secrets.compare_digest(authorization, "Bearer " + expected):
            raise DomainError("unauthorized", "A valid bearer token is required.", 401)

    def model_auth(authorization: str | None = Header(default=None)):
        authorize(authorization, model_token)

    def operator_auth(authorization: str | None = Header(default=None)):
        authorize(authorization, operator_token)

    def write_call(**values):
        try:
            service.store.record_call(**values)
        except Exception:
            logging.getLogger(__name__).warning("Call audit unavailable; operation response preserved.")

    async def record_call(**values):
        # SQLite FULL commits and contention must not block the ASGI event loop.
        await anyio.to_thread.run_sync(partial(write_call, **values))

    @app.middleware("http")
    async def record_calls(request: Request, call_next):
        t0 = time.time()
        started = time.monotonic()
        try:
            response = await call_next(request)
        except Exception:
            duration_ms = round((time.monotonic() - started) * 1000.0, 2)
            match = match_whitelisted_operation(request.method, request.url.path)
            if match and hasattr(service, "store") and service.store:
                op, norm_path, safe_req_id, safe_job_id = match
                await record_call(
                    at=t0,
                    source="http",
                    operation=getattr(request.state, "operation", op),
                    method=request.method,
                    path=norm_path,
                    status="failed",
                    duration_ms=duration_ms,
                    request_id=getattr(request.state, "request_id", safe_req_id),
                    job_id=getattr(request.state, "job_id", safe_job_id),
                    error_code="internal_error",
                )
            raise

        duration_ms = round((time.monotonic() - started) * 1000.0, 2)
        match = match_whitelisted_operation(request.method, request.url.path)
        if match and hasattr(service, "store") and service.store:
            op, norm_path, safe_req_id, safe_job_id = match
            status = "succeeded" if response.status_code < 400 else "failed"
            error_code = None
            if status == "failed":
                error_code = getattr(request.state, "error_code", None)
                if not error_code:
                    if response.status_code == 401:
                        error_code = "unauthorized"
                    elif response.status_code == 403:
                        error_code = "origin_rejected"
                    elif response.status_code == 404:
                        error_code = "not_found"
                    elif response.status_code == 413:
                        error_code = "body_too_large"
                    elif response.status_code == 422:
                        error_code = "invalid_request"
                    else:
                        error_code = f"http_{response.status_code}"

            req_id = getattr(request.state, "request_id", None) or safe_req_id
            if req_id and not is_safe_id(req_id):
                req_id = None

            j_id = getattr(request.state, "job_id", None) or safe_job_id
            if j_id and not is_safe_id(j_id):
                j_id = None

            await record_call(
                at=t0,
                source="http",
                operation=getattr(request.state, "operation", op),
                method=request.method,
                path=norm_path,
                status=status,
                duration_ms=duration_ms,
                request_id=req_id,
                job_id=j_id,
                error_code=error_code,
            )
        return response

    @app.middleware("http")
    async def control_session_context(request: Request, call_next):
        token = control_session_id.set(request.headers.get("x-piper-control-session"))
        try:
            return await call_next(request)
        finally:
            control_session_id.reset(token)

    @app.middleware("http")
    async def reject_browser_origins(request: Request, call_next):
        # This API is for tool clients. No browser-origin requests or permissive CORS.
        if "origin" in request.headers:
            request.state.error_code = "origin_rejected"
            return JSONResponse({"error": {"code": "origin_rejected", "message": "Browser-origin requests are not supported."}}, 403)
        length = request.headers.get("content-length", "0")
        if not length.isdigit() or int(length) > 16384:
            request.state.error_code = "body_too_large"
            return JSONResponse({"error": {"code": "body_too_large", "message": "Request body exceeds 16 KiB."}}, 413)
        return await call_next(request)

    @app.exception_handler(DomainError)
    async def domain_error(request: Request, exc: DomainError):
        request.state.error_code = exc.code
        return JSONResponse({"error": {"code": exc.code, "message": exc.message}}, exc.status)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError):
        # Do not echo input (which can contain NaN, huge content, or secrets).
        request.state.error_code = "invalid_request"
        return JSONResponse({"error": {"code": "invalid_request", "message": "Request does not match the schema.",
            "fields": [{"loc": list(e["loc"]), "message": e["msg"]} for e in exc.errors()]}}, 422)

    auth = [Depends(model_auth)]

    @app.get("/health")
    def health():
        cap = service.capabilities()
        return {"service": "piperx-middleware", "api_version": "1",
                "process_id": cap["process_id"], "instance_id": service.instance_id,
                "profile_id": service.settings.managed_profile_id,
                "backend": service.backend.name,
                "mode": "real" if service.backend.name == "agx" else "simulation"}

    @app.post("/v1/shutdown", dependencies=auth)
    def shutdown(req: ShutdownRequest, background_tasks: BackgroundTasks):
        result = service.shutdown_idle(req.expected_instance_id)
        if on_shutdown is not None:
            background_tasks.add_task(on_shutdown)
        return result

    @app.get("/openapi.json", dependencies=auth)
    def schema():
        return app.openapi()

    @app.get("/v1/capabilities", dependencies=auth)
    def capabilities():
        return service.capabilities()

    @app.get("/v1/state", dependencies=auth)
    def state():
        return service.state()

    @app.get("/v1/parameters", dependencies=auth)
    def parameters():
        return get_parameters_response(service)

    @app.get("/v1/piper-aio", dependencies=auth)
    def piper_aio_schema():
        return describe()

    @app.post("/v1/piper-aio/preview-action", dependencies=auth)
    def piper_aio_preview(req: AioAction):
        return service.preview(req.to_command()) | {"source_layout": req.layout,
            "selected_arm": req.arm_side, "selected_component": req.component,
            "other_components_executed": False}

    @app.get("/v1/devices", dependencies=auth)
    def devices():
        return service.devices()

    @app.post("/v1/connect", dependencies=auth)
    def connect(req: ConnectOptions | None = None):
        return service.connect(reconnect=req.reconnect if req is not None else False,
                               **({"device_id": req.device_id} if req is not None and req.device_id is not None else {}))

    @app.post("/v1/disconnect", dependencies=auth)
    def disconnect():
        return service.disconnect()

    @app.post("/v1/preview", dependencies=auth)
    def preview(req: PreviewRequest):
        return service.preview(req.command)

    @app.post("/v1/execute", dependencies=auth)
    def execute(req: ExecuteRequest):
        return service.execute(req)

    @app.post("/v1/control-mode", dependencies=auth)
    def control_mode(req: ControlModeRequest, request: Request):
        request.state.request_id = req.request_id
        request.state.operation = "control_mode"
        result = service.move(ControlMode(**req.model_dump(exclude={"request_id"})), req.request_id)
        request.state.job_id = result.get("job_id")
        return result

    @app.post("/v1/move", dependencies=auth)
    def move(req: MoveRequest, request: Request):
        request.state.request_id = req.request_id
        request.state.operation = command_operation(request.url.path, req.command.kind, "move" if request.url.path == "/v1/move" else "primitives")
        result = service.move(req.command, req.request_id)
        if isinstance(result, dict):
            request.state.job_id = result.get("job_id")
            request.state.request_id = result.get("request_id", req.request_id)
        return result

    @app.post("/v1/primitives", dependencies=auth)
    def primitives(req: PrimitiveRequest, request: Request):
        request.state.request_id = req.request_id
        request.state.operation = command_operation(request.url.path, req.command.kind, "move" if request.url.path == "/v1/move" else "primitives")
        result = service.primitive(req.command, req.request_id)
        if isinstance(result, dict):
            request.state.job_id = result.get("job_id")
            request.state.request_id = result.get("request_id", req.request_id)
        return result

    @app.post("/v1/primitives/preview", dependencies=auth)
    def primitive_preview(req: PrimitivePreviewRequest):
        return service.preview_primitive(req.command)

    @app.get("/v1/diagnostics", dependencies=auth)
    def diagnostics():
        return service.diagnostics()

    @app.get("/v1/limits", dependencies=auth)
    def limits():
        return service.limits()

    @app.post("/operator/session", dependencies=[Depends(operator_auth)])
    def acquire_session(req: SessionAcquire):
        return service.acquire_session(req.owner, req.shutdown_on_loss)

    @app.post("/operator/session/heartbeat", dependencies=[Depends(operator_auth)])
    def heartbeat_session(req: SessionReference):
        return service.heartbeat_session(req.session_id)

    @app.post("/operator/session/release", dependencies=[Depends(operator_auth)])
    def release_session(req: SessionReference):
        return service.release_session(req.session_id)

    @app.get("/operator/interaction", dependencies=[Depends(operator_auth)])
    def interaction():
        return service.interaction_state()

    @app.put("/operator/interaction", dependencies=[Depends(operator_auth)])
    def configure_interaction(req: PolicyUpdate):
        return service.configure_interaction(req.policy)

    @app.post("/operator/approvals/{job_id}", dependencies=[Depends(operator_auth)])
    def decide_approval(job_id: str, req: ApprovalDecision):
        return service.decide_approval(job_id, req.approved)

    @app.get("/operator/settings", dependencies=[Depends(operator_auth)])
    def profile_settings():
        return {"configured": service.settings.model_dump(mode="json"),
                "parameter_version": service.parameter_version}

    @app.patch("/operator/settings", dependencies=[Depends(operator_auth)])
    def configure_profile(req: ProfileSettingsUpdate):
        return service.configure_profile(req.changes)

    @app.post("/operator/query-limits", dependencies=[Depends(operator_auth)])
    def query_limits():
        return service.limits(refresh=True)

    @app.post("/operator/estop", dependencies=[Depends(operator_auth)])
    def estop():
        return service.emergency_stop()

    @app.post("/operator/clear-estop", dependencies=[Depends(operator_auth)])
    def clear_estop():
        return service.clear_estop()

    @app.patch("/operator/parameters", dependencies=[Depends(operator_auth)])
    def configure(req: RuntimeParameters):
        return service.configure_runtime(req)

    @app.post("/operator/sim-fault", dependencies=[Depends(operator_auth)])
    def sim_fault(req: SimFault):
        return service.inject_fault(req)

    @app.get("/v1/jobs", dependencies=auth)
    def list_jobs(limit: int = Query(default=100, ge=1, le=500)):
        return {"jobs": service.store.jobs(limit)}

    @app.get("/v1/jobs/{ident}", dependencies=auth)
    def job(ident: str):
        return service.get_job(ident)

    @app.get("/v1/requests/{request_id}", dependencies=auth)
    def request_status(request_id: str):
        return service.get_request(request_id)

    @app.get("/v1/calls", dependencies=auth)
    def list_calls(limit: int = Query(default=100, ge=1, le=500)):
        return {"calls": service.store.calls(limit), "semantics": CALLS_SEMANTICS}

    @app.get("/v1/events", dependencies=auth)
    def events(after: int = Query(default=0, ge=0), limit: int = Query(default=100, ge=1, le=500)):
        return {"events": service.store.events(after, limit)}

    @app.post("/v1/stop", dependencies=auth)
    def stop():
        return service.stop()

    @app.post("/operator/control-window", dependencies=[Depends(operator_auth)], deprecated=True)
    def arm(req: LeaseRequest):
        return service.arm_window(req)

    if service.backend.name == "mujoco":
        from .simulation_api import attach_simulation_api
        attach_simulation_api(app, service, auth, operator_auth)

    # The SDK app owns its /mcp route; root mounting avoids /mcp/mcp and redirects.
    app.mount("/", _MCPBearerAuth(mcp_app, model_token))
    return app
