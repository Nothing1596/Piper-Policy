import asyncio
import json

import httpx
import pytest

from piperx_middleware.model_agent import ModelConfig, run_turn


TOOLS = [{"name": "move", "inputSchema": {"type": "object", "properties": {"target": {"type": "number"}},
            "required": ["target"], "additionalProperties": False}}]


def run_mock(monkeypatch, responses, result=None):
    actual = httpx.AsyncClient
    iterator = iter(responses)
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: actual(transport=httpx.MockTransport(lambda r: httpx.Response(200, json=next(iterator))), **kw))
    calls, messages = [], [{"role": "user", "content": "test"}]
    async def invoke(name, args):
        calls.append((name, args))
        return result or {"status": "succeeded"}
    return calls, messages, run_turn(ModelConfig(endpoint="http://127.0.0.1:1234/v1", model="fixture"), messages, TOOLS, invoke, lambda x: None)


def response(args='{"target": 1}', reason="tool_calls", tool="move", twice=False):
    tc = {"id": "t1", "type": "function", "function": {"name": tool, "arguments": args}}
    calls = [tc, dict(tc, id="t2")] if twice else [tc]
    return {"choices": [{"finish_reason": reason, "message": {"role": "assistant", "tool_calls": calls}}]}


def done(): return {"choices": [{"finish_reason": "stop", "message": {"content": "done"}}]}


def test_truncated_tool_never_executes(monkeypatch):
    calls, _, coro = run_mock(monkeypatch, [response(reason="length")])
    with pytest.raises(RuntimeError, match="incomplete"): asyncio.run(coro)
    assert not calls


@pytest.mark.parametrize("tool,args", [("operator_estop_clear", "{}"), ("move", '{"target":NaN}'), ("move", '{"target":1,"enable":true}')])
def test_model_invalid_arguments_no_dispatch(monkeypatch, tool, args):
    calls, messages, coro = run_mock(monkeypatch, [response(args=args, tool=tool), done()])
    asyncio.run(coro)
    assert not calls
    assert "error" in json.loads(messages[2]["content"])


def test_uncertainty_halts_remaining_calls(monkeypatch):
    calls, messages, coro = run_mock(monkeypatch, [response(twice=True)], {"status": "outcome_unknown"})
    assert "unknown" in asyncio.run(coro)
    assert len(calls) == 1
    assert "not_executed" in messages[-1]["content"]
