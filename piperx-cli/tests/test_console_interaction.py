"""Unit and integration tests for InteractiveConsoleController and console extensions."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from piperx_middleware.console_interaction import (
    InteractiveConsoleController,
    choose_startup,
)


class FakeBridge:
    def __init__(self, url: str = "http://127.0.0.1:8765") -> None:
        self.url = url
        self.connected = True
        self.session_id: str | None = "sess_test_123"
        self.tools: list[dict[str, Any]] = [
            {
                "name": "robot_status",
                "description": "Robot status",
                "inputSchema": {
                    "type": "object",
                    "properties": {"job_id": {"type": "string"}},
                },
            },
            {
                "name": "robot_stop",
                "description": "Stop motion",
                "inputSchema": {"type": "object", "properties": {}},
            },
            {
                "name": "robot_connect",
                "description": "Connect device",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "device_id": {"type": "string"},
                        "reconnect": {"type": "boolean"},
                    },
                },
            },
            {
                "name": "move_to",
                "description": "Move TCP",
                "inputSchema": {
                    "type": "object",
                    "properties": {"x": {"type": "number"}},
                    "required": [],
                },
            },
        ]
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.rest_calls: list[tuple[str, str, Any]] = []
        self.mock_job_sequence: list[dict[str, Any]] = []

    async def open(self) -> None:
        self.connected = True

    async def close(self) -> None:
        self.connected = False

    async def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((name, arguments))
        if name == "robot_status":
            if self.mock_job_sequence:
                return self.mock_job_sequence.pop(0)
            return {"status": "succeeded", "job_id": arguments.get("job_id")}
        if name == "move_to":
            return {"job_id": "job_move_1", "status": "accepted"}
        if name == "robot_stop":
            return {"status": "stopped"}
        if name == "robot_connect":
            return {"status": "connected"}
        return {"result": "ok"}

    async def rest(self, method: str, path: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
        self.rest_calls.append((method, path, body))
        if path == "/v1/devices":
            return {"devices": [{"device_id": "can0", "label": "CAN Bus 0"}], "connected_device_id": None}
        if path == "/v1/state":
            return {"backend": "simulation", "ready": True, "control_mode": "position", "connection": {"status": "connected"}}
        return {}


class FakeManaged:
    def __init__(self, mode: str = "simulation", target: str = "local") -> None:
        self.mode = mode
        self.target = target
        self.config = {"port": 8765}
        self.remotes_list = [{"name": "pi_arm", "ssh_host": "pi@192.168.1.50"}]
        self.shutdown_called = False
        self.reconnect_called = False
        self.switches: list[tuple[str, str]] = []

    async def switch(self, mode: str, target: str) -> dict[str, Any]:
        self.mode = mode
        self.target = target
        self.switches.append((mode, target))
        return {"mode": mode, "target": target}

    async def shutdown(self) -> dict[str, Any]:
        self.shutdown_called = True
        return {"status": "shutdown"}

    async def reconnect(self) -> dict[str, Any]:
        self.reconnect_called = True
        return {"status": "reconnected"}

    async def remotes(self) -> list[dict[str, str]]:
        return list(self.remotes_list)

    async def save_remote(self, name: str, ssh_host: str) -> dict[str, str]:
        entry = {"name": name, "ssh_host": ssh_host}
        self.remotes_list.append(entry)
        return entry

    async def configure(self, cfg: dict[str, Any]) -> dict[str, Any]:
        self.config.update(cfg)
        return dict(self.config)


class FakeOperator:
    def __init__(self) -> None:
        self.policy = {
            "mode": "risk",
            "limits": {"max_speed_percent": 100, "gripper_max_m": 0.07, "gripper_min_m": 0.0},
            "automatic": {"max_joint_step_deg": 10.0},
        }
        self.pending: list[dict[str, Any]] = []
        self.approvals: list[tuple[str, bool]] = []
        self.heartbeats: list[str] = []
        self.calls: list[tuple[str, str, Any]] = []
        self.heartbeat_error: bool = False
        self.approval_response: dict[str, Any] | None = None

    async def call(self, method: str, path: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
        self.calls.append((method, path, body))
        if method == "GET" and path == "/operator/interaction":
            return {
                "policy": dict(self.policy),
                "pending": list(self.pending),
                "session": {"owner": "test_owner"},
                "effective_limits": self.policy.get("limits", {}),
            }
        if method == "PUT" and path == "/operator/interaction":
            if body and "policy" in body:
                self.policy = body["policy"]
            return {"policy": dict(self.policy)}
        if method == "POST" and path.startswith("/operator/approvals/"):
            job_id = path.split("/")[-1]
            approved = bool(body.get("approved")) if body else False
            self.approvals.append((job_id, approved))
            if self.approval_response is not None:
                return self.approval_response
            return {"job_id": job_id, "approved": approved, "status": "accepted"}
        if method == "POST" and path == "/operator/session/heartbeat":
            if self.heartbeat_error:
                return {"error": "session_expired"}
            if body and "session_id" in body:
                self.heartbeats.append(body["session_id"])
            return {"status": "ok"}
        return {}


@pytest.fixture
def setup_env(tmp_path: Path):
    bridge = FakeBridge()
    managed = FakeManaged()
    operator = FakeOperator()
    emitted: list[str] = []

    def emit(msg: str) -> None:
        emitted.append(msg)

    controller = InteractiveConsoleController(
        bridge=bridge,
        root=tmp_path,
        emit=emit,
        operator_call=operator.call,
        managed=managed,
    )
    return controller, bridge, managed, operator, emitted


@pytest.mark.asyncio
async def test_approval_mode_confirmation_and_retained_constraints(setup_env):
    controller, bridge, managed, operator, emitted = setup_env
    # Readonly inspects current mode
    await controller.handle_line("/approval")
    assert any("Current approval mode: risk" in msg for msg in emitted)

    # Change requires confirmation proposal
    emitted.clear()
    await controller.handle_line("/approval auto")
    assert operator.policy["mode"] == "risk"  # Not changed yet
    assert any("Confirmation required: Proposal" in msg for msg in emitted)
    assert any("Retained constraints for auto mode:" in msg for msg in emitted)
    assert any("max_joint_step_deg:  10.0" in msg for msg in emitted)

    # Extract proposal code
    code = list(controller._proposals.keys())[0]
    await controller.handle_line(f"/confirm {code}")
    assert operator.policy["mode"] == "auto"
    assert controller.approval_mode == "auto"
    assert any(f"Proposal {code} confirmed" in msg for msg in emitted)


@pytest.mark.asyncio
async def test_approve_shows_reasons_and_handles_cancelled(setup_env):
    controller, bridge, managed, operator, emitted = setup_env
    operator.pending.append({
        "job_id": "job_xyz",
        "revision": 1,
        "approval": {"reasons": ["risk_exceeded"], "command": {"joint": 1}},
    })

    # Normal approval
    await controller.handle_line("/approve job_xyz")
    assert ("job_xyz", True) in operator.approvals
    assert any("Approving job job_xyz: command={\"joint\": 1}, reasons=['risk_exceeded']" in msg for msg in emitted)
    assert any("Job job_xyz approved (status: accepted)." in msg for msg in emitted)

    # Cancellation response: never claim approved
    emitted.clear()
    operator.approval_response = {"job_id": "job_xyz", "status": "cancelled", "reason": "approval_changed"}
    await controller.handle_line("/approve job_xyz")
    assert any("Job job_xyz not approved: cancelled (approval_changed)" in msg for msg in emitted)
    assert not any("Job job_xyz approved" in msg for msg in emitted)


@pytest.mark.asyncio
async def test_limits_patch_merge_and_wizard(setup_env):
    controller, bridge, managed, operator, emitted = setup_env
    # Nested limits JSON
    await controller.handle_line('/limits {"limits": {"max_speed_percent": 85}, "automatic": {"max_tcp_step_m": 0.05}}')
    assert len(controller._proposals) == 1
    code = list(controller._proposals.keys())[0]
    await controller.handle_line(f"/confirm {code}")

    assert operator.policy["limits"]["max_speed_percent"] == 85
    assert operator.policy["limits"]["gripper_max_m"] == 0.07  # Preserved unchanged
    assert operator.policy["automatic"]["max_tcp_step_m"] == 0.05
    assert operator.policy["automatic"]["max_joint_step_deg"] == 10.0  # Preserved unchanged

    # Interactive wizard with all 4 auto approval fields and execution limits
    prompt_inputs = ["90", "0.06", "0.01", "15.0", "0.04", "80", "12.5"]
    async def mock_prompt(text: str, default: str = "") -> str:
        return prompt_inputs.pop(0) if prompt_inputs else ""

    controller.prompt = mock_prompt
    emitted.clear()
    await controller.handle_line("/limits")
    assert any("Confirmation required: Proposal" in msg for msg in emitted)
    wizard_code = list(controller._proposals.keys())[0]
    await controller.handle_line(f"/confirm {wizard_code}")

    assert operator.policy["limits"]["max_speed_percent"] == 90.0
    assert operator.policy["automatic"]["max_joint_step_deg"] == 15.0
    assert operator.policy["automatic"]["max_tcp_step_m"] == 0.04
    assert operator.policy["automatic"]["max_speed_percent"] == 80
    assert operator.policy["automatic"]["max_effort_protocol"] == 12.5


@pytest.mark.asyncio
async def test_config_proposal_and_cancel(setup_env):
    controller, bridge, managed, operator, emitted = setup_env
    await controller.handle_line('/config {"speed": 10}')
    assert managed.config.get("speed") is None  # Not yet applied
    assert len(controller._proposals) == 1
    code = list(controller._proposals.keys())[0]
    await controller.handle_line(f"/cancel {code}")
    assert managed.config.get("speed") is None
    assert len(controller._proposals) == 0
    assert any(f"Proposal {code} cancelled" in msg for msg in emitted)


@pytest.mark.asyncio
async def test_mode_switch_rejects_active_task_and_preserves_bridge(setup_env):
    controller, bridge, managed, operator, emitted = setup_env
    async def long_task():
        await asyncio.sleep(10.0)

    controller.background_task = asyncio.create_task(long_task())
    # Attempting mode switch while active task running must be rejected
    await controller.handle_line("/mode real pi_arm")
    assert ("real", "pi_arm") not in managed.switches
    assert any("Cannot switch mode while a background task is running" in msg for msg in emitted)

    # Cancel task
    controller.background_task.cancel()
    try:
        await controller.background_task
    except asyncio.CancelledError:
        pass
    controller.background_task = None

    # Mode switch succeeds without frontend closing/reopening stale bridge
    emitted.clear()
    bridge.calls.clear()
    await controller.handle_line("/mode real pi_arm")
    assert ("real", "pi_arm") in managed.switches
    assert controller.mode == "real"
    assert controller.target == "pi_arm"
    assert bridge.connected is True


@pytest.mark.asyncio
async def test_shutdown_and_quit_lifecycle_drain(setup_env):
    controller, bridge, managed, operator, emitted = setup_env
    # /shutdown returns True, keeps handle_line returning True and accepting commands while draining
    res = await controller.handle_line("/shutdown")
    assert res is True
    assert controller._draining is True

    # Commands like /status, /jobs, /stop remain accepted
    assert await controller.handle_line("/status") is True
    assert await controller.handle_line("/stop") is True

    # Disallowed commands rejected
    assert await controller.handle_line("/limits {}") is True
    assert any("Console is draining/shutting down; new commands rejected." in msg for msg in emitted)

    # Wait for lifecycle task to finish
    if controller._drain_task:
        await controller._drain_task
    assert managed.shutdown_called is True
    assert controller.exit_requested is True


@pytest.mark.asyncio
async def test_heartbeat_session_lost_and_connect_reacquire(setup_env):
    controller, bridge, managed, operator, emitted = setup_env
    operator.heartbeat_error = True
    await controller.start()
    await asyncio.sleep(0.05)

    assert controller._session_lost is True
    assert any("Control session lost or expired" in msg for msg in emitted)

    # Tool invocation rejected while session lost
    res = await controller.invoke_tool("move_to", {"x": 0.1})
    assert res.get("error", {}).get("code") == "session_lost"

    # Reacquire via /connect --reconnect
    emitted.clear()
    await controller.handle_line("/connect --reconnect")
    assert managed.reconnect_called is True
    assert controller._session_lost is False
    assert any("Control session reacquired." in msg for msg in emitted)

    await controller.close()


@pytest.mark.asyncio
async def test_choose_startup_raises_on_invalid_mode():
    async def bad_prompt(text: str, default: str = "") -> str:
        return "invalid_mode"

    with pytest.raises(ValueError, match="Invalid mode 'invalid_mode'"):
        await choose_startup(bad_prompt)


@pytest.mark.asyncio
async def test_confirm_without_code_only_when_unambiguous(setup_env):
    controller, _, _, operator, _ = setup_env
    await controller.handle_line('/approval auto')
    await controller.handle_line('/confirm')
    assert operator.policy['mode'] == 'auto'
    await controller.handle_line('/approval risk')
    await controller.handle_line('/approval always')
    await controller.handle_line('/confirm')
    assert operator.policy['mode'] == 'auto'
    assert len(controller._proposals) == 2
    code = next(iter(controller._proposals))
    await controller.handle_line(f'/cancel {code}')
    await controller.handle_line('/cancel')
    assert not controller._proposals
    assert operator.policy['mode'] == 'auto'


@pytest.mark.asyncio
async def test_unchanged_policy_and_limits_do_not_prompt(setup_env):
    controller, _, _, operator, emitted = setup_env
    before = json.loads(json.dumps(operator.policy))
    await controller.handle_line('/approval risk')
    await controller.handle_line('/limits {}')
    async def keep(_):
        return ''
    controller.prompt = keep
    await controller.handle_line('/limits')
    assert not controller._proposals
    assert operator.policy == before
    assert any('nothing to confirm' in line for line in emitted)


@pytest.mark.asyncio
@pytest.mark.parametrize('failure', [None, {'error': {'code': 'session_lost'}}, {'policy': {}}])
@pytest.mark.parametrize('command', ['/limits', '/limits {"limits": {"max_speed_percent": 5}}', '/approval auto'])
async def test_policy_edits_require_current_policy(setup_env, failure, command):
    controller, _, _, _, emitted = setup_env
    async def unavailable(*args):
        return failure
    async def unexpected_prompt(_):
        pytest.fail('Cannot edit unknown policy')
    controller.operator_call = unavailable
    controller.prompt = unexpected_prompt
    await controller.handle_line(command)
    assert not controller._proposals
    assert any('editor not opened' in line for line in emitted)


@pytest.mark.asyncio
@pytest.mark.parametrize('command', ['/approval risk', '/limits {}', '/limits'])
async def test_noop_after_proposal_cannot_confirm_older_change(setup_env, command):
    controller, _, _, operator, _ = setup_env
    async def keep(_):
        return ''
    controller.prompt = keep
    await controller.handle_line('/approval auto')
    await controller.handle_line(command)
    await controller.handle_line('/confirm')
    assert operator.policy['mode'] == 'risk'
    assert len(controller._proposals) == 2  # Choose explicitly, never guess.
    latest = list(controller._proposals)[-1]
    await controller.handle_line(f'/confirm {latest}')
    assert operator.policy['mode'] == 'risk'
    assert not controller._proposals


@pytest.mark.asyncio
@pytest.mark.parametrize('mode', ['auto', 'always'])
async def test_wizard_skips_inactive_thresholds_and_preserves_them(setup_env, mode):
    controller, _, _, operator, _ = setup_env
    operator.policy['mode'] = mode
    before = dict(operator.policy['automatic'])
    prompts = []
    async def answer(text):
        prompts.append(text)
        return '50' if len(prompts) == 1 else ''
    controller.prompt = answer
    await controller.handle_line('/limits')
    assert len(prompts) == 3
    await controller.handle_line('/confirm')
    assert operator.policy['limits']['max_speed_percent'] == 50
    assert operator.policy['automatic'] == before
