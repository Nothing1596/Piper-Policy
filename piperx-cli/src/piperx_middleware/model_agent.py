"""OpenAI-compatible model configuration and tool-calling execution loop."""
from __future__ import annotations

import asyncio
import copy
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Callable, Coroutine
from urllib.parse import urlparse

import httpx
from jsonschema import Draft202012Validator


UNCERTAINTY_CODES = {
    "transport_unknown",
    "outcome_unknown",
    "uncertain_outcome",
    "invalid_response",
}

MAX_INT = 2**63 - 1
MIN_INT = -(2**63)
MAX_DEPTH = 20


@dataclass
class ModelConfig:
    endpoint: str = ""
    model: str = ""
    api_key_env: str = "PIPERX_MODEL_API_KEY"
    api_key_file: str = ""
    max_tokens: int = 2048
    request_timeout_s: float = 90.0


def validate_endpoint(endpoint: str) -> str:
    """Validates the model endpoint; rejects remote HTTP, credentials, query, fragments, or control characters."""
    if not endpoint or not endpoint.strip():
        raise ValueError("Model endpoint is not configured.")
    if any(ord(c) < 32 or ord(c) == 127 for c in endpoint):
        raise ValueError("Endpoint contains invalid control characters.")
    endpoint = endpoint.strip()
    parsed = urlparse(endpoint)
    if parsed.scheme not in ("http", "https"):
        raise ValueError("Expected an HTTP(S) endpoint.")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("Endpoint must not contain credentials, query, or fragment.")
    if not parsed.hostname:
        raise ValueError("Endpoint missing hostname.")
    try:
        _ = parsed.port
    except ValueError as exc:
        raise ValueError("Malformed port in endpoint") from exc
    if parsed.scheme == "http" and parsed.hostname not in ("localhost", "127.0.0.1", "::1"):
        raise ValueError("Remote model access requires HTTPS or localhost.")
    return endpoint.rstrip("/")


def load_model_config(root: Path) -> ModelConfig:
    """Loads model configuration references from root/model.json with environment overrides."""
    config_file = root / "model.json"
    data: dict[str, Any] = {}
    if config_file.is_file():
        try:
            loaded = json.loads(config_file.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                data = loaded
        except Exception:
            pass

    config = ModelConfig(
        endpoint=str(data.get("endpoint", "")),
        model=str(data.get("model", "")),
        api_key_env=str(data.get("api_key_env", "PIPERX_MODEL_API_KEY")),
        api_key_file=str(data.get("api_key_file", "")),
    )

    if os.environ.get("PIPERX_MODEL_ENDPOINT"):
        ep = os.environ["PIPERX_MODEL_ENDPOINT"].strip()
        validate_endpoint(ep)
        config.endpoint = ep
    if os.environ.get("PIPERX_MODEL_NAME"):
        m = os.environ["PIPERX_MODEL_NAME"].strip()
        if any(ord(c) < 32 or ord(c) == 127 for c in m):
            raise ValueError("Environment PIPERX_MODEL_NAME contains invalid control characters.")
        config.model = m
    if os.environ.get("PIPERX_MODEL_API_KEY_ENV"):
        ake = os.environ["PIPERX_MODEL_API_KEY_ENV"].strip()
        if any(ord(c) < 32 or ord(c) == 127 for c in ake):
            raise ValueError("Environment PIPERX_MODEL_API_KEY_ENV contains invalid control characters.")
        config.api_key_env = ake

    if config.endpoint:
        validate_endpoint(config.endpoint)
    return config


def save_model_config(root: Path, config: ModelConfig) -> None:
    """Persists model configuration references (never key values) to root/model.json atomically."""
    if config.endpoint:
        validate_endpoint(config.endpoint)
    if any(ord(c) < 32 or ord(c) == 127 for c in config.model):
        raise ValueError("Model name contains invalid control characters.")
    if any(ord(c) < 32 or ord(c) == 127 for c in config.api_key_env):
        raise ValueError("API key env variable name contains invalid control characters.")

    root.mkdir(parents=True, exist_ok=True)
    config_file = root / "model.json"
    data = {
        "endpoint": config.endpoint,
        "model": config.model,
        "api_key_env": config.api_key_env,
        "api_key_file": config.api_key_file,
    }
    content = json.dumps(data, indent=2).encode("utf-8")
    with tempfile.NamedTemporaryFile(dir=root, prefix="model_", suffix=".tmp", delete=False) as tf:
        temp_path = Path(tf.name)
        try:
            tf.write(content)
            tf.flush()
            os.fsync(tf.fileno())
        except Exception:
            temp_path.unlink(missing_ok=True)
            raise
    temp_path.replace(config_file)


def resolve_api_key(config: ModelConfig) -> str:
    """Resolves API key strictly from the specified environment variable or key file."""
    if config.api_key_env and config.api_key_env in os.environ:
        val = os.environ[config.api_key_env].strip()
        if val:
            return val
    if config.api_key_file:
        key_path = Path(config.api_key_file)
        if key_path.is_file():
            try:
                val = key_path.read_text(encoding="utf-8").strip()
                if val:
                    return val
            except Exception:
                pass
    return ""


async def check_model(config: ModelConfig) -> dict[str, Any]:
    """Performs a minimal tools-free completion; reports sanitized status/latency/model without secrets or body."""
    if not config.model or not config.model.strip():
        return {"status": "error", "error": "Model name is not configured.", "latency_ms": 0.0, "model": config.model}
    if any(ord(c) < 32 or ord(c) == 127 for c in config.model):
        return {"status": "error", "error": "Model name contains invalid control characters.", "latency_ms": 0.0, "model": config.model}

    try:
        endpoint = validate_endpoint(config.endpoint)
    except ValueError as exc:
        return {"status": "error", "error": f"InvalidEndpoint: {exc}", "latency_ms": 0.0, "model": config.model}

    url = endpoint.rstrip("/") + "/chat/completions"
    key = resolve_api_key(config)
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"

    payload = {
        "model": config.model,
        "messages": [{"role": "user", "content": "ping"}],
        "max_tokens": 1,
    }

    t0 = time.monotonic()
    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(10.0, connect=3.0),
            trust_env=False,
            follow_redirects=False,
        ) as client:
            resp = await client.post(url, headers=headers, json=payload)
        latency_ms = round((time.monotonic() - t0) * 1000.0, 1)
        if resp.status_code != 200:
            return {"status": "error", "error": f"HTTP {resp.status_code}", "latency_ms": latency_ms, "model": config.model}

        try:
            data = resp.json()
        except Exception:
            return {"status": "error", "error": "Invalid JSON response from model API", "latency_ms": latency_ms, "model": config.model}

        if not isinstance(data, dict):
            return {"status": "error", "error": "Invalid response structure from model API", "latency_ms": latency_ms, "model": config.model}
        choices = data.get("choices")
        if not isinstance(choices, list) or len(choices) == 0:
            return {"status": "error", "error": "Missing or empty choices in model response", "latency_ms": latency_ms, "model": config.model}
        choice = choices[0]
        if not isinstance(choice, dict) or "message" not in choice or not isinstance(choice["message"], dict):
            return {"status": "error", "error": "Invalid choice message in model response", "latency_ms": latency_ms, "model": config.model}

        return {"status": "ok", "latency_ms": latency_ms, "model": config.model}
    except httpx.TransportError as exc:
        latency_ms = round((time.monotonic() - t0) * 1000.0, 1)
        return {"status": "error", "error": f"TransportError: {type(exc).__name__}", "latency_ms": latency_ms, "model": config.model}
    except Exception as exc:
        latency_ms = round((time.monotonic() - t0) * 1000.0, 1)
        return {"status": "error", "error": f"Error: {type(exc).__name__}", "latency_ms": latency_ms, "model": config.model}


def transform_mcp_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Transforms MCP JSON schemas into OpenAI ChatCompletions function schemas.

    Copies schema and removes REQUIRED request_id from model-facing required list,
    while retaining optional property in properties for explicit lookup/continuity.
    """
    openai_tools: list[dict[str, Any]] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        name = tool.get("name")
        if not name or not isinstance(name, str):
            continue
        raw_schema = tool.get("inputSchema")
        if isinstance(raw_schema, dict):
            schema = copy.deepcopy(raw_schema)
        else:
            schema = {"type": "object", "properties": {}}

        req = schema.get("required")
        if isinstance(req, list) and "request_id" in req:
            schema["required"] = [r for r in req if r != "request_id"]

        openai_tools.append({
            "type": "function",
            "function": {
                "name": name,
                "description": str(tool.get("description", "")),
                "parameters": schema,
            },
        })
    return openai_tools


def _is_bounded_finite_literal(val: Any, depth: int = 0, seen: set[int] | None = None) -> bool:
    """Verifies that all values are bounded finite literals without circular/deep structures."""
    if depth > MAX_DEPTH:
        return False
    if isinstance(val, bool) or val is None:
        return True
    if isinstance(val, int):
        return MIN_INT <= val <= MAX_INT
    if isinstance(val, float):
        return math.isfinite(val)
    if isinstance(val, str):
        return len(val) <= 1_000_000
    if seen is None:
        seen = set()
    val_id = id(val)
    if val_id in seen:
        return False
    seen.add(val_id)
    try:
        if isinstance(val, list):
            return all(_is_bounded_finite_literal(item, depth + 1, seen) for item in val)
        if isinstance(val, dict):
            return all(
                isinstance(k, str) and len(k) <= 1000 and _is_bounded_finite_literal(v, depth + 1, seen)
                for k, v in val.items()
            )
    finally:
        seen.remove(val_id)
    return False


def _validate_tool_arguments(args: dict[str, Any], schema: dict[str, Any]) -> str | None:
    """Validates tool arguments using Draft202012Validator.

    Rejects extra unknown kwargs even when MCP schema additionalProperties is missing.
    Allows missing request_id only when expected as an injector (required in schema).
    """
    if not isinstance(schema, dict):
        return None
    val_schema = copy.deepcopy(schema)
    if "type" not in val_schema:
        val_schema["type"] = "object"
    if "properties" not in val_schema:
        val_schema["properties"] = {}

    req = val_schema.get("required")
    if isinstance(req, list) and "request_id" in req:
        val_schema["required"] = [r for r in req if r != "request_id"]

    if "additionalProperties" not in val_schema or val_schema["additionalProperties"] is not False:
        val_schema["additionalProperties"] = False

    try:
        validator = Draft202012Validator(val_schema)
        errors = list(validator.iter_errors(args))
        if errors:
            err = errors[0]
            if err.validator == "additionalProperties":
                return f"Unrecognized parameter: {err.message}"
            if err.validator == "required":
                return err.message
            path_prefix = f"Parameter '{err.path[0]}': " if err.path else ""
            return f"{path_prefix}{err.message}"
    except Exception as exc:
        return f"Schema validation failed: {exc}"
    return None


def is_uncertain_result(result: Any) -> tuple[bool, str]:
    """Checks if a tool execution result indicates physical/transport uncertainty.

    Explicit uncertainty codes only: transport_unknown, outcome_unknown,
    uncertain_outcome, invalid_response. Safely inspects status, error_code,
    and error without calling .get on strings.
    """
    if not isinstance(result, dict):
        return False, ""

    status = result.get("status")
    if isinstance(status, str) and status.lower() in UNCERTAINTY_CODES:
        err = result.get("error")
        msg = str(err) if err else f"Status: {status}"
        return True, msg

    error_code = result.get("error_code")
    if isinstance(error_code, str) and error_code.lower() in UNCERTAINTY_CODES:
        err = result.get("error")
        msg = str(err) if err else f"Error code: {error_code}"
        return True, msg

    err_obj = result.get("error")
    if isinstance(err_obj, dict):
        code = err_obj.get("code")
        if isinstance(code, str) and code.lower() in UNCERTAINTY_CODES:
            msg = err_obj.get("message") or str(code)
            return True, str(msg)
    elif isinstance(err_obj, str) and err_obj.lower() in UNCERTAINTY_CODES:
        return True, err_obj

    return False, ""


def _sanitize_string(s: str) -> str:
    """Strips control characters and limits length for safe console emitting."""
    clean = "".join(c for c in str(s) if ord(c) >= 32 and ord(c) != 127)
    return clean[:200]


async def run_turn(
    config: ModelConfig,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    invoke: Callable[[str, dict[str, Any]], Coroutine[Any, Any, dict[str, Any]]],
    emit: Callable[[str], None] = print,
) -> str:
    """Runs a single user turn with OpenAI-compatible function calling loop (bounded to 12 rounds)."""
    endpoint = validate_endpoint(config.endpoint)
    if not config.model.strip():
        raise ValueError("Model name is not configured.")
    url = endpoint.rstrip("/") + "/chat/completions"
    api_key = resolve_api_key(config)
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    openai_tools = transform_mcp_tools(tools)
    history_backup = copy.deepcopy(messages)
    dispatched_any = False

    try:
        for _ in range(12):
            payload: dict[str, Any] = {
                "model": config.model,
                "messages": messages,
                "max_tokens": config.max_tokens,
                "temperature": 0,
            }
            if openai_tools:
                payload["tools"] = openai_tools

            try:
                async with httpx.AsyncClient(
                    timeout=httpx.Timeout(config.request_timeout_s, connect=10.0),
                    trust_env=False,
                    follow_redirects=False,
                ) as client:
                    resp = await client.post(url, headers=headers, json=payload)
            except httpx.TransportError as exc:
                raise RuntimeError(f"TransportError: {type(exc).__name__}") from exc

            if resp.status_code != 200:
                raise RuntimeError(f"ModelAPIError: HTTP {resp.status_code}")

            try:
                data = resp.json()
                if not isinstance(data, dict):
                    raise ValueError("Response root is not an object")
                choices = data.get("choices")
                if not isinstance(choices, list) or not choices:
                    raise ValueError("Missing or empty choices list")
                choice = choices[0]
                if not isinstance(choice, dict):
                    raise ValueError("Choice is not an object")
                msg = choice.get("message")
                if not isinstance(msg, dict):
                    raise ValueError("Message is not an object")
            except Exception as exc:
                raise RuntimeError("Invalid response structure from model API") from exc

            if choice.get("finish_reason") in ("length", "content_filter"):
                raise RuntimeError("Model output incomplete: " + choice["finish_reason"])
            if data.get("model"):
                emit("Model response: " + _sanitize_string(data["model"]))

            content = msg.get("content")
            tool_calls = msg.get("tool_calls")
            if content is not None and not isinstance(content, str):
                raise RuntimeError("Model message content must be text.")

            if not tool_calls:
                if not content or not content.strip():
                    raise RuntimeError("Model returned no text or tool calls.")
                assistant_msg = {"role": "assistant", "content": content}
                messages.append(assistant_msg)
                return str(content or "")

            if not isinstance(tool_calls, list):
                raise RuntimeError("Invalid response structure from model API: tool_calls must be a list")

            # Validate the entire protocol envelope before dispatching any call.
            ids = set()
            for tc in tool_calls:
                if not isinstance(tc, dict) or tc.get("type") != "function":
                    raise RuntimeError("Malformed model tool call.")
                ident, fn = tc.get("id"), tc.get("function")
                if not isinstance(ident, str) or not ident or ident in ids:
                    raise RuntimeError("Missing or duplicate model tool call ID.")
                ids.add(ident)
                if not isinstance(fn, dict) or not isinstance(fn.get("name"), str) or not fn["name"]:
                    raise RuntimeError("Malformed model tool function.")
                if not isinstance(fn.get("arguments"), str):
                    raise RuntimeError("Model tool arguments must be a JSON string.")

            assistant_msg = {"role": "assistant", "content": content, "tool_calls": tool_calls}
            messages.append(assistant_msg)

            id_counts: dict[str, int] = {}
            for tc in tool_calls:
                if isinstance(tc, dict) and isinstance(tc.get("id"), str) and tc.get("id"):
                    cid = tc["id"]
                    id_counts[cid] = id_counts.get(cid, 0) + 1

            for i, tc in enumerate(tool_calls):
                tool_error: dict[str, Any] | None = None
                call_id = ""
                fn_name = ""
                raw_args: Any = "{}"

                if not isinstance(tc, dict):
                    tool_error = {"error": {"code": "invalid_tool_call", "message": "Tool call must be an object."}}
                else:
                    call_id = tc.get("id", "")
                    if not isinstance(call_id, str) or not call_id:
                        tool_error = {"error": {"code": "invalid_tool_call", "message": "Tool call missing id."}}
                    elif id_counts.get(call_id, 0) > 1:
                        tool_error = {"error": {"code": "invalid_tool_call", "message": f"Duplicate tool call id '{call_id}'."}}
                    else:
                        fn = tc.get("function")
                        if not isinstance(fn, dict):
                            tool_error = {"error": {"code": "invalid_tool_call", "message": "Tool call missing function object."}}
                        else:
                            fn_name = fn.get("name", "")
                            if not isinstance(fn_name, str) or not fn_name:
                                tool_error = {"error": {"code": "invalid_tool_call", "message": "Tool function missing name."}}
                            else:
                                raw_args = fn.get("arguments", "{}")

                args: dict[str, Any] | None = None
                if tool_error is None:
                    if isinstance(raw_args, dict):
                        args = raw_args
                    elif isinstance(raw_args, str):
                        try:
                            def _reject_constant(val: Any) -> None:
                                raise ValueError(f"Non-finite literal: {val}")
                            args = json.loads(raw_args, parse_constant=_reject_constant)
                        except Exception as exc:
                            tool_error = {"error": {"code": "invalid_arguments", "message": f"Malformed arguments: {type(exc).__name__}"}}
                        else:
                            if not isinstance(args, dict):
                                tool_error = {"error": {"code": "invalid_arguments", "message": "Arguments must be a JSON object."}}
                    else:
                        tool_error = {"error": {"code": "invalid_arguments", "message": "Arguments must be a JSON object or string."}}

                if tool_error is None:
                    if not isinstance(args, dict) or not _is_bounded_finite_literal(args):
                        tool_error = {"error": {"code": "invalid_arguments", "message": "Arguments must be a JSON object of bounded finite literals."}}
                    else:
                        tool_def = next((t for t in tools if t.get("name") == fn_name), None)
                        if not tool_def:
                            tool_error = {"error": {"code": "unknown_tool", "message": f"Tool '{fn_name}' is not recognized."}}
                        else:
                            err_msg = _validate_tool_arguments(args, tool_def.get("inputSchema", {}))
                            if err_msg:
                                tool_error = {"error": {"code": "schema_violation", "message": err_msg}}

                if tool_error is not None:
                    tool_result = tool_error
                else:
                    clean_fn = _sanitize_string(fn_name)
                    emit(f"Invoking {clean_fn}...")
                    dispatched_any = True
                    try:
                        tool_result = await invoke(fn_name, args)
                    except asyncio.CancelledError:
                        tool_msg = {
                            "role": "tool",
                            "tool_call_id": call_id,
                            "name": fn_name,
                            "content": json.dumps({
                                "error": {
                                    "code": "outcome_unknown",
                                    "message": "Tool execution cancelled during invoke; outcome uncertain.",
                                    "request_id": args.get("request_id"),
                                }
                            }, ensure_ascii=False),
                        }
                        messages.append(tool_msg)
                        for remaining_tc in tool_calls[i + 1:]:
                            rem_id = remaining_tc.get("id", "") if isinstance(remaining_tc, dict) else ""
                            rem_fn = remaining_tc.get("function", {}) if isinstance(remaining_tc, dict) else {}
                            rem_name = rem_fn.get("name", "") if isinstance(rem_fn, dict) else ""
                            messages.append({
                                "role": "tool",
                                "tool_call_id": rem_id,
                                "name": rem_name,
                                "content": json.dumps({
                                    "error": {
                                        "code": "not_executed",
                                        "message": "Cancelled before execution.",
                                    }
                                }, ensure_ascii=False),
                            })
                        raise
                    except Exception as exc:
                        tool_result = {
                            "error": {
                                "code": "outcome_unknown",
                                "message": f"Tool invocation error: {type(exc).__name__}",
                            }
                        }

                tool_msg = {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": fn_name,
                    "content": json.dumps(tool_result, ensure_ascii=False),
                }
                messages.append(tool_msg)

                is_uncertain, uncertain_msg = is_uncertain_result(tool_result)
                if is_uncertain:
                    clean_fn = _sanitize_string(fn_name)
                    emit(f"Tool {clean_fn} outcome unknown; halting execution.")
                    for remaining_tc in tool_calls[i + 1:]:
                        rem_id = remaining_tc.get("id", "") if isinstance(remaining_tc, dict) else ""
                        rem_fn = remaining_tc.get("function", {}) if isinstance(remaining_tc, dict) else {}
                        rem_name = rem_fn.get("name", "") if isinstance(rem_fn, dict) else ""
                        messages.append({
                            "role": "tool",
                            "tool_call_id": rem_id,
                            "name": rem_name,
                            "content": json.dumps({
                                "error": {
                                    "code": "not_executed",
                                    "message": f"Previous tool execution had uncertain outcome ({uncertain_msg}); skipped.",
                                }
                            }, ensure_ascii=False),
                        })
                    return f"Action outcome unknown: {uncertain_msg}"

        return "Model reached maximum allowed rounds (12)."
    except asyncio.CancelledError:
        if not dispatched_any:
            messages[:] = history_backup
        raise
    except Exception:
        if not dispatched_any:
            messages[:] = history_backup
        raise
