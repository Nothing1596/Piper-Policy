"""Console MCP client bridge for Streamable HTTP executor connection.

Owns the MCP ClientSession within a dedicated background task to ensure
AnyIO / SDK cancel scope lifetime matches the creation task.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client


class MCPBridge:
    """Real asynchronous MCP ClientSession bridge with dedicated task ownership."""

    def __init__(self, url: str, token_file: Path | str) -> None:
        parsed = urlparse(url)
        if (
            parsed.scheme not in ("http", "https")
            or not parsed.hostname
            or any(ord(c) < 32 for c in url)
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("Expected an HTTP(S) origin without credentials, query, or fragment.")
        if parsed.scheme == "http" and parsed.hostname not in ("localhost", "127.0.0.1", "::1"):
            raise ValueError("Remote access requires HTTPS or an SSH tunnel to localhost.")
        try:
            _ = parsed.port
        except ValueError as exc:
            raise ValueError("Malformed port in URL") from exc

        self.session_id: str | None = None
        self.url = url.rstrip("/")
        self.token_file = Path(token_file)
        self._token = ""
        self._lifecycle_lock = asyncio.Lock()

        self.connected: bool = False
        self.tools: list[dict[str, Any]] = []
        self.last_error: str | None = None

        self._owner_task: asyncio.Task[None] | None = None
        self._ready_event: asyncio.Event | None = None
        self._shutdown_event: asyncio.Event | None = None
        self._init_error: Exception | None = None
        self._session: ClientSession | None = None
        self._rest_client: httpx.AsyncClient | None = None

    def _read_token(self):
        try:
            token = self.token_file.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeError) as exc:
            raise ValueError("Invalid token file. Initialize an executor or pass --token-file.") from exc
        if len(token) < 32 or any(c in token for c in "\r\n"):
            raise ValueError("Invalid token file.")
        self._token = token
        return token

    async def __aenter__(self) -> MCPBridge:
        await self.open()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.close()

    async def _run_mcp_loop(self) -> None:
        try:
            headers = {"Authorization": f"Bearer {self._token}"}
            if self.session_id:
                headers["X-Piper-Control-Session"] = self.session_id
            mcp_url = f"{self.url}/mcp"
            async with httpx.AsyncClient(
                base_url=self.url,
                headers=headers,
                timeout=httpx.Timeout(30.0, connect=5.0),
                trust_env=False,
                follow_redirects=False,
            ) as http_client:
                async with streamable_http_client(mcp_url, http_client=http_client) as streams:
                    read_stream = streams[0]
                    write_stream = streams[1]
                    async with ClientSession(read_stream, write_stream) as session:
                        self._session = session
                        await session.initialize()
                        tools_result = await session.list_tools()
                        parsed_tools: list[dict[str, Any]] = []
                        for t in tools_result.tools:
                            schema = t.inputSchema
                            if hasattr(schema, "model_dump"):
                                schema = schema.model_dump()
                            elif not isinstance(schema, dict):
                                schema = dict(schema)
                            parsed_tools.append({
                                "name": t.name,
                                "description": t.description or "",
                                "inputSchema": schema,
                            })
                        self.tools = parsed_tools
                        self.connected = True
                        self.last_error = None
                        if self._ready_event and not self._ready_event.is_set():
                            self._ready_event.set()

                        if self._shutdown_event:
                            await self._shutdown_event.wait()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.last_error = type(exc).__name__
            self.connected = False
            self._session = None
            if self._ready_event and not self._ready_event.is_set():
                self._init_error = RuntimeError(f"MCP initialization failed ({type(exc).__name__}).")
                self._ready_event.set()
        finally:
            self.connected = False
            self.tools = []
            self._session = None
            if self._ready_event and not self._ready_event.is_set():
                self._init_error = RuntimeError("MCP initialization ended before readiness.")
                self._ready_event.set()

    async def _close_owner(self):
        if self._shutdown_event:
            self._shutdown_event.set()
        owner = self._owner_task
        if owner is not None:
            try:
                await asyncio.wait_for(asyncio.shield(owner), 5.0)
            except asyncio.TimeoutError:
                owner.cancel()
                await asyncio.gather(owner, return_exceptions=True)
            self._owner_task = None
        if self._rest_client is not None:
            await self._rest_client.aclose()
            self._rest_client = None
        self.connected = False
        self.tools = []
        self._session = None

    async def open(self) -> None:
        """Open one SDK session, with context entry/exit confined to its owner task."""
        async with self._lifecycle_lock:
            if self.connected and self._owner_task and not self._owner_task.done():
                return
            await self._close_owner()
            self._read_token()
            self._init_error = None
            self._ready_event = asyncio.Event()
            self._shutdown_event = asyncio.Event()
            self._owner_task = asyncio.create_task(self._run_mcp_loop())
            try:
                await asyncio.wait_for(self._ready_event.wait(), 15.0)
                if self._init_error:
                    raise self._init_error
            except BaseException:
                self._owner_task.cancel()
                await asyncio.gather(self._owner_task, return_exceptions=True)
                self._owner_task = None
                raise

    async def close(self) -> None:
        async with self._lifecycle_lock:
            await self._close_owner()

    async def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Calls an MCP tool without automatic retry; handles errors as {error:{code,message}}."""
        if not self.connected or self._session is None:
            return {"error": {"code": "not_connected", "message": "MCP bridge is not connected."}}
        try:
            res = await self._session.call_tool(name, arguments)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.last_error = type(exc).__name__
            self.connected = False
            self.tools = []
            return {"error": {"code": "transport_unknown", "message":
                f"{type(exc).__name__}: outcome unknown. Query the original request_id; do not replay automatically."}}

        if isinstance(res, dict):
            return res

        if getattr(res, "isError", False):
            structured = getattr(res, "structuredContent", None)
            if isinstance(structured, dict) and "error" in structured:
                return structured
            for c in getattr(res, "content", []):
                text = getattr(c, "text", "")
                if text:
                    try:
                        # FastMCP may prefix the JSON DomainError with a tool label.
                        start = text.find("{")
                        parsed = json.JSONDecoder().raw_decode(text[start:])[0] if start >= 0 else None
                        if isinstance(parsed, dict) and "error" in parsed:
                            return parsed
                    except Exception:
                        pass
            msg = " ".join(getattr(c, "text", str(c)) for c in getattr(res, "content", []))
            return {"error": {"code": "tool_error", "message": msg or "Tool execution failed."}}

        structured = getattr(res, "structuredContent", None)
        if structured is not None:
            return structured if isinstance(structured, dict) else {"result": structured}

        contents = getattr(res, "content", [])
        if contents:
            text = getattr(contents[0], "text", "")
            if text:
                try:
                    parsed = json.loads(text)
                    if isinstance(parsed, dict):
                        return parsed
                    return {"result": parsed}
                except Exception:
                    return {"result": text}
        return {}

    async def rest(self, method: str, path: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
        """Authenticated async REST call using the same model token; matching RobotClient error formats."""
        if not path.startswith(("/v1/", "/health")) or path.startswith("//"):
            return {"error": {"code": "invalid_path", "message": "Expected executor API path."}}
        if self._rest_client is None or self._rest_client.is_closed:
            try:
                self._read_token()
            except ValueError:
                return {"error": {"code": "missing_token", "message": "Executor token unavailable."}}
            self._rest_client = httpx.AsyncClient(
                base_url=self.url,
                headers={"Authorization": f"Bearer {self._token}", **({"X-Piper-Control-Session": self.session_id} if self.session_id else {})},
                timeout=httpx.Timeout(10.0, connect=3.0),
                trust_env=False,
                follow_redirects=False,
            )
        try:
            response = await self._rest_client.request(method, path, json=body)
        except httpx.TransportError as exc:
            self.connected = False
            self.tools = []
            return {"error": {"code": "transport_unknown", "message":
                f"{type(exc).__name__}: outcome unknown. Query the original request_id; do not replay automatically."}}
        except Exception as exc:
            return {"error": {"code": "transport_error", "message": f"{type(exc).__name__}"}}

        try:
            result = response.json()
        except ValueError:
            return {"error": {"code": "invalid_response", "message": f"HTTP {response.status_code}; outcome unknown."}}

        if not isinstance(result, dict) or response.is_redirect:
            return {"error": {"code": "invalid_response", "message": f"HTTP {response.status_code}; expected object, outcome unknown."}}

        if response.is_error and "error" not in result:
            return {"error": {"code": "http_error", "message": f"HTTP {response.status_code}"}}

        return result
