from pathlib import Path
import copy
import numpy as np
import pytest
from piperlab.safety import SafetyGate, Rejected, load_config


@pytest.fixture
def setup_gate():
    c = load_config(Path(__file__).resolve().parents[1] / 'config/hardware.yaml')
    now = [100.0]
    gate = SafetyGate(c, clock=lambda: now[0], wall_clock=lambda: now[0])
    state = [0, 1, -1, 0, 0, 0, .05]
    gate.observe(gate.names, state, now[0])
    gate.observe_camera(now[0])
    token = gate.control('acquire', 'policy')
    return gate, now, state, token


def test_acquire_does_not_arm(setup_gate):
    g, now, q, token = setup_gate
    with pytest.raises(Rejected):
        g.admit('policy', token, 0, g.names, q, now[0])


def test_exclusive_owner_and_speed_limit(setup_gate):
    g, now, q, token = setup_gate
    with pytest.raises(Rejected):
        g.control('acquire', 'teleop')
    g.control('arm', 'policy', token)
    now[0] += .05
    goal = q.copy(); goal[0] = 1
    applied = g.admit('policy', token, 0, g.names, goal, now[0])
    assert applied[0] == pytest.approx(.0075)
    g.mark_applied(applied)
    with pytest.raises(Rejected):
        g.admit('policy', token, 0, g.names, goal, now[0])


def test_watchdog_invalidates_session_and_does_not_rearm(setup_gate):
    g, now, q, token = setup_gate
    g.control('arm', 'policy', token)
    now[0] += .201
    assert g.tick()
    g.observe(g.names, q, now[0]); g.observe_camera(now[0])
    g.control('reset', '')
    assert not g.armed
    new = g.control('acquire', 'policy')
    assert new != token
    g.control('arm', 'policy', new)
    with pytest.raises(Rejected):
        g.admit('policy', token, 1, g.names, q, now[0])


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), 99])
def test_bad_action_latches_fault(setup_gate, bad):
    g, now, q, token = setup_gate
    g.control('arm', 'policy', token)
    q[0] = bad
    with pytest.raises(Rejected):
        g.admit('policy', token, 0, g.names, q, now[0])
    assert g.fault and not g.armed


def test_received_old_image_is_not_fresh(setup_gate):
    g, now, q, token = setup_gate
    now[0] += .15
    assert g.observe_camera(100.0)
    now[0] += .06
    with pytest.raises(Rejected):
        g.control('arm', 'policy', token)


def test_uncommissioned_hardware_cannot_arm(setup_gate):
    g, now, q, token = setup_gate
    g.config = copy.deepcopy(g.config); g.config['mode'] = 'real'
    with pytest.raises(Rejected, match='not commissioned'):
        g.control('arm', 'policy', token)


def test_future_and_stale_commands_trip(setup_gate):
    g, now, q, token = setup_gate
    g.control('arm', 'policy', token)
    with pytest.raises(Rejected):
        g.admit('policy', token, 0, g.names, q, now[0] + 10)
    assert not g.armed
