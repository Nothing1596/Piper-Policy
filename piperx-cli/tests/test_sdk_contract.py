"""Actual bundled SDK encoder/decoder through GuardedBus, using ONLY an in-memory CAN peer."""
import os
import struct
import sys
import time
from types import SimpleNamespace

import can
import pytest

from piperx_middleware.agx_backend import AgxBackend, GuardedBus
from piperx_middleware.models import DomainError, Settings


@pytest.fixture
def backend(tmp_path):
    root = os.environ.get("PIPERX_SDK_ROOT")
    if not root:
        pytest.skip("Set PIPERX_SDK_ROOT to run the pinned SDK wire contract tests")
    sys.path.insert(0, root)
    from pyAgxArm import AgxArmFactory, ArmModel, create_agx_arm_config
    arm = AgxArmFactory.create_arm(create_agx_arm_config(robot=ArmModel.PIPER_X,
        firmeware_version="v189", interface="virtual", channel="offline-test", auto_connect=False))
    arm.set_auto_set_motion_mode_enabled(False)
    arm.set_joint_limits_enabled(True)
    b = AgxBackend(Settings(backend="agx", data_dir=tmp_path))
    b.arm = arm
    arm.is_connected = lambda: True
    frames = []
    ratings, acc = [3]*6, [100]*6
    def reply(ident, data):
        frame = can.Message(arbitration_id=ident, data=data, is_extended_id=False)
        arm._parser.parse_packet(frame)
        b._on_frame(frame)
    def send(frame, timeout=None):
        ident, data = frame.arbitration_id, bytes(frame.data)
        frames.append((ident, data))
        if ident == 0x477 and data[3] == 0xAE:
            reply(0x476, bytes([0x77])+bytes(7))
        if ident == 0x47A:
            ratings[:] = list(data[:6])
        if ident == 0x477 and data[0] == 2:
            reply(0x47B, bytes(ratings)+bytes(2))
        if ident == 0x475:
            acc[data[0]-1] = int.from_bytes(data[3:5], "big")
        if ident == 0x472 and data[1] == 2:
            reply(0x47C, bytes([data[0]])+struct.pack(">H", acc[data[0]-1])+bytes(5))
        if ident == 0x472 and data[1] == 1:
            reply(0x473, struct.pack(">BhhHB", data[0], 1500, -1400, 200, 0))
    # The real GuardedBus.send runs, but its inner transport never opens hardware.
    guard = SimpleNamespace(owner=b, _is_shutdown=False, inner=SimpleNamespace(send=send))
    comm = SimpleNamespace(send=lambda frame: GuardedBus.send(guard, frame), get_channel=lambda: "offline-test")
    arm._ctx.get_comm = lambda: comm
    b.test_frames = frames
    return b


def test_emergency_stop_exact_frame_and_latch(backend):
    backend.emergency_stop()
    assert backend.test_frames == [(0x150, bytes([1])+bytes(7))]
    with pytest.raises(DomainError) as exc: backend.joint_target([0.]*6)
    assert exc.value.code == "estop_latched"


def test_native_line_sdk_exact_frames(backend):
    backend.linear_target([0.]*6, {"xyz_m": [.2, 0., .3], "rpy_deg": [0., 45., 0.]}, 5, lambda: None)
    assert [i for i,_ in backend.test_frames] == [0x155, 0x156, 0x157, 0x151, 0x152, 0x153, 0x154]
    assert backend.test_frames[3][1] == bytes([1, 2, 5, 0, 0, 0, 0, 0])
    assert backend.test_frames[-1][1] == struct.pack(">ii", 45000, 0)


def test_sdk_parameter_writes_and_readback(backend):
    result = backend.apply_parameters({"payload": "half", "collision_rating": 4, "joint_acc_rad_s2": 1.25})
    assert result["status"] == "applied", result
    assert len(result["results"]) == 8
    assert result["results"][0]["acknowledged"] and not result["results"][0]["readback_verified"]
    assert all(r["readback_verified"] for r in result["results"][1:])
    assert all(i not in {0x471, 0x150} for i,_ in backend.test_frames)


def test_sdk_limit_query_actual_decoder(backend):
    result = backend.query_limits()
    assert result["available"], result
    assert len(result["joints"]) == 6
    assert result["joints"][0]["min_deg"] == pytest.approx(-140)
    assert [i for i,_ in backend.test_frames] == [0x472]*6


def test_unsolicited_frame_blocked(backend):
    with pytest.raises(RuntimeError): backend.arm.reset()
    assert not backend.test_frames
    assert backend.fault == "Unexpected CAN transmission blocked"


def test_recovery_exact_pose_no_clamp_and_sdk_restored(backend):
    from piperx_middleware.recovery_limits import recovery_bounds, recovery_scope, worker_bounds
    from piperx_middleware.models import JOINT_LIMITS_DEG
    from piperx_middleware.agx_backend import joint_frames
    q=[0.,-1.168,1.056,-39.249,-89.516,58.842]
    bounds=recovery_bounds(q, [lo for lo,hi in JOINT_LIMITS_DEG], [hi for lo,hi in JOINT_LIMITS_DEG])
    original=backend.arm._config
    with recovery_scope(bounds):
        backend.set_control_mode(q,5,lambda:None)
        assert backend.arm._config is original
        assert backend.arm.get_joint_limits_enabled()
        assert backend.test_frames==joint_frames(q)+[(0x151,bytes([1,1,5,0,0,0,0,0]))]
        with pytest.raises(DomainError):backend.joint_target([0.,-1.169,1.056,-39.249,-89.516,58.842])
    assert worker_bounds() is None
    with pytest.raises(DomainError):backend.joint_target(q)
    assert len(backend.test_frames)==4


def test_recovery_sdk_config_restored_on_error(backend,monkeypatch):
    from piperx_middleware.recovery_limits import recovery_scope, worker_bounds
    from piperx_middleware.models import JOINT_LIMITS_DEG
    original=backend.arm._config
    def fail(q):raise RuntimeError('transport failed')
    monkeypatch.setattr(backend.arm,'move_j',fail)
    with pytest.raises(RuntimeError):
        with recovery_scope(JOINT_LIMITS_DEG):backend.joint_target([0.]*6)
    assert backend.arm._config is original and worker_bounds() is None
