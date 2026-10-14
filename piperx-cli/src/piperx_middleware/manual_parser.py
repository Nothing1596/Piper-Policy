"""AST-based manual command parser for PiperX interactive console.

Accepts bare identifier calls with keyword literal arguments only:
    TOOL(keyword=literal, ...)
Rejects positional arguments, attributes, unpacking, duplicate keys,
tuples, and non-finite or arbitrary expressions.
"""

from __future__ import annotations

import ast
import math
from typing import Any

MAX_DEPTH = 20
MAX_COLLECTION_SIZE = 500
MAX_STRING_LENGTH = 10000
MAX_TOTAL_LENGTH = 100000


def _validate_bounded(val: Any, depth: int = 0, state: dict[str, int] | None = None) -> None:
    if state is None:
        state = {"total_length": 0}

    if depth > MAX_DEPTH:
        raise ValueError(f"Argument nesting exceeds maximum depth of {MAX_DEPTH}")

    if isinstance(val, tuple):
        raise ValueError("Tuples are not allowed; use lists instead")

    if isinstance(val, bool) or val is None:
        return

    if isinstance(val, (int, float)):
        try:
            if not math.isfinite(val):
                raise ValueError(f"Non-finite numeric value not allowed: {val}")
        except OverflowError:
            raise ValueError("Numeric value too large (overflow)")
        return

    if isinstance(val, str):
        if len(val) > MAX_STRING_LENGTH:
            raise ValueError(f"String value exceeds maximum allowed length: {len(val)}")
        state["total_length"] += len(val)
        if state["total_length"] > MAX_TOTAL_LENGTH:
            raise ValueError("Total input string length exceeds limit")
        return

    if isinstance(val, list):
        if len(val) > MAX_COLLECTION_SIZE:
            raise ValueError(f"Collection exceeds maximum allowed size: {len(val)}")
        for elem in val:
            _validate_bounded(elem, depth + 1, state)
        return

    if isinstance(val, dict):
        if len(val) > MAX_COLLECTION_SIZE:
            raise ValueError(f"Dictionary exceeds maximum allowed size: {len(val)}")
        for k, v in val.items():
            if not isinstance(k, str):
                raise ValueError(f"Dictionary keys must be strings, got: {type(k).__name__}")
            if len(k) > 1000:
                raise ValueError("Dictionary key exceeds maximum allowed length")
            state["total_length"] += len(k)
            if state["total_length"] > MAX_TOTAL_LENGTH:
                raise ValueError("Total input string length exceeds limit")
            _validate_bounded(v, depth + 1, state)
        return

    raise ValueError(f"Unsupported value type in manual command: {type(val).__name__}")


def _eval_bounded_literal(node: ast.AST) -> Any:
    if isinstance(node, ast.Tuple):
        raise ValueError("Tuples are not allowed; use lists instead")
    try:
        val = ast.literal_eval(node)
    except Exception as exc:
        raw = getattr(node, "id", None) or type(node).__name__
        raise ValueError(f"Argument value must be a static literal, got: {raw}") from exc
    _validate_bounded(val)
    return val


def parse_manual_command(text: str) -> tuple[str, dict[str, Any]]:
    """Parse a manual tool invocation string.

    Supports:
        TOOL(arg1=val1, arg2=val2, ...)
        /manual TOOL(arg1=val1, ...)

    Returns:
        (tool_name, arguments_dict)

    Raises:
        ValueError on syntax error, positional args, attribute access,
        unpacking, duplicate keys, tuples, non-finite or overflowing values.
    """
    clean = text.strip()
    if clean.startswith("/manual"):
        clean = clean[len("/manual"):].strip()

    if not clean:
        raise ValueError("Usage: /manual TOOL(keyword=literal, ...)")

    if len(clean) > MAX_TOTAL_LENGTH:
        raise ValueError("Manual command exceeds maximum allowed length")

    try:
        tree = ast.parse(clean, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"Syntax error in manual command: {exc.msg}") from exc

    if not isinstance(tree.body, ast.Call):
        raise ValueError("Command must be a function call: TOOL(keyword=literal, ...)")

    call = tree.body
    if not isinstance(call.func, ast.Name):
        raise ValueError("Only bare tool identifiers are allowed (e.g. robot_move_to(...))")

    tool_name = call.func.id

    if call.args:
        raise ValueError("Positional arguments are not allowed; use keyword arguments only")

    args: dict[str, Any] = {}
    for kw in call.keywords:
        if kw.arg is None:
            raise ValueError("Dictionary unpacking (**kwargs) is not allowed")
        if kw.arg in args:
            raise ValueError(f"Duplicate argument name: {kw.arg!r}")
        val = _eval_bounded_literal(kw.value)
        args[kw.arg] = val

    return tool_name, args
