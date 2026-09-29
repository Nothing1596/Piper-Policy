from contextlib import contextmanager

import pytest

from piperx_middleware.approval_binding import (BINDING_VERSION, GRIPPER_TOLERANCE_M,
                                                JOINT_TOLERANCE_DEG, bind_approval,
                                                validate_approval)
from piperx_middleware.models import DomainError

Q0 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
IDENTITY = {'instance_id': 'instance-1', 'connection_epoch': 2,
            'policy_version': 3, 'parameter_version': 4}


@contextmanager
def expect_code(code):
    with pytest.raises(DomainError) as info:
        yield info
    assert info.value.code == code


def make_state(q_deg=Q0, gripper_width_m=0.02):
    return {'connected': True, 'q_deg': list(q_deg), 'gripper_width_m': gripper_width_m}


def joint_move(joints):
    return {'kind': 'joint', 'joints_deg': list(joints), 'speed_percent': 5, 'timeout_s': 30}


def gripper_move(width=0.03):
    return {'kind': 'gripper', 'width_m': width, 'effort_protocol': 0.5,
            'timeout_s': 10, 'completion': 'width'}


def test_bind_captures_canonical_command_identity_pose_and_tolerances():
    binding = bind_approval(joint_move([1.0, 2.0, 3.0, 0.0, 0.0, 0.0]),
                            make_state(q_deg=[0.5, 0, 0, 0, 0, 0]), **IDENTITY)
    assert binding['binding_version'] == BINDING_VERSION
    assert binding['command'] == joint_move([1.0, 2.0, 3.0, 0.0, 0.0, 0.0])
    for key, value in IDENTITY.items():
        assert binding[key] == value
    assert binding['measured']['q_deg'] == [0.5, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert binding['measured']['gripper_width_m'] is None
    assert binding['tolerances'] == {'joint_deg': JOINT_TOLERANCE_DEG,
                                     'gripper_m': GRIPPER_TOLERANCE_M}
    # No wall-clock expiry: bindings carry no timestamp fields.
    assert not any('time' in key or 'stamp' in key or 'expir' in key for key in binding)


def test_bind_canonicalizes_partial_command_with_model_defaults():
    binding = bind_approval({'kind': 'joint', 'joints_deg': Q0}, make_state(), **IDENTITY)
    assert binding['command']['speed_percent'] == 5
    assert binding['command']['timeout_s'] == 30


def test_bind_grips_gripper_width_only_for_gripper_commands():
    binding = bind_approval(gripper_move(), make_state(gripper_width_m=0.021), **IDENTITY)
    assert binding['measured']['gripper_width_m'] == 0.021
    binding = bind_approval(joint_move(Q0), make_state(gripper_width_m=0.021), **IDENTITY)
    assert binding['measured']['gripper_width_m'] is None


def test_validate_accepts_unchanged_command_identity_and_pose():
    command = joint_move([1.0, 0, 0, 0, 0, 0])
    state = make_state(q_deg=[0.5, 0, 0, 0, 0, 0])
    binding = bind_approval(command, state, **IDENTITY)
    assert validate_approval(binding, command, state, **IDENTITY) is None
    # Equal command with omitted optional fields canonicalizes identically.
    assert validate_approval(binding, {'kind': 'joint', 'joints_deg': [1.0, 0, 0, 0, 0, 0]},
                             state, **IDENTITY) is None
    # Small drift within tolerance is accepted.
    moved = make_state(q_deg=[0.5 + JOINT_TOLERANCE_DEG / 2, 0, 0, 0, 0, 0])
    assert validate_approval(binding, command, moved, **IDENTITY) is None


@pytest.mark.parametrize('field, value', [
    ('instance_id', 'instance-2'),
    ('connection_epoch', 3),
    ('policy_version', 4),
    ('parameter_version', 5),
])
def test_validate_rejects_identity_drift(field, value):
    command = joint_move(Q0)
    binding = bind_approval(command, make_state(), **IDENTITY)
    changed = dict(IDENTITY, **{field: value})
    with expect_code('approval_changed'):
        validate_approval(binding, command, make_state(), **changed)


def test_validate_rejects_command_change():
    binding = bind_approval(joint_move(Q0), make_state(), **IDENTITY)
    with expect_code('approval_changed'):
        validate_approval(binding, joint_move([1.0, 0, 0, 0, 0, 0]), make_state(), **IDENTITY)
    with expect_code('approval_changed'):
        validate_approval(binding, gripper_move(), make_state(), **IDENTITY)


def test_validate_rejects_pose_drift_beyond_tolerance():
    command = joint_move(Q0)
    binding = bind_approval(command, make_state(), **IDENTITY)
    moved = make_state(q_deg=[0.0, JOINT_TOLERANCE_DEG + 0.01, 0, 0, 0, 0])
    with expect_code('approval_changed'):
        validate_approval(binding, command, moved, **IDENTITY)


def test_validate_gripper_width_tolerance_only_for_gripper_commands():
    command = gripper_move()
    binding = bind_approval(command, make_state(gripper_width_m=0.020), **IDENTITY)
    close = make_state(gripper_width_m=0.020 + GRIPPER_TOLERANCE_M)
    assert validate_approval(binding, command, close, **IDENTITY) is None
    far = make_state(gripper_width_m=0.020 + GRIPPER_TOLERANCE_M + 0.0005)
    with expect_code('approval_changed'):
        validate_approval(binding, command, far, **IDENTITY)
    # A joint approval does not cover gripper motion.
    joint = bind_approval(joint_move(Q0), make_state(gripper_width_m=0.020), **IDENTITY)
    assert validate_approval(joint, joint_move(Q0),
                             make_state(gripper_width_m=0.05), **IDENTITY) is None


def test_validate_rejects_malformed_binding():
    command = joint_move(Q0)
    binding = bind_approval(command, make_state(), **IDENTITY)
    with expect_code('approval_changed'):
        validate_approval(dict(binding, command=None), command, make_state(), **IDENTITY)
    with expect_code('approval_changed'):
        validate_approval({k: v for k, v in binding.items() if k != 'measured'},
                          command, make_state(), **IDENTITY)
    with expect_code('approval_changed'):
        validate_approval(dict(binding, binding_version=BINDING_VERSION + 1),
                          command, make_state(), **IDENTITY)
    with expect_code('invalid_approval'):
        validate_approval('not-a-dict', command, make_state(), **IDENTITY)


def test_bind_strict_finite_validation():
    with expect_code('invalid_command'):
        bind_approval({'kind': 'joint', 'joints_deg': [float('nan')] * 6},
                      make_state(), **IDENTITY)
    with expect_code('invalid_command'):
        bind_approval({'kind': 'joint', 'joints_deg': Q0[:5]}, make_state(), **IDENTITY)
    with expect_code('invalid_state'):
        bind_approval(joint_move(Q0), {'q_deg': [0.0] * 5}, **IDENTITY)
    with expect_code('invalid_state'):
        bind_approval(joint_move(Q0), {'q_deg': [float('inf')] * 6}, **IDENTITY)
    with expect_code('invalid_state'):
        bind_approval(gripper_move(), {'q_deg': Q0, 'gripper_width_m': float('nan')}, **IDENTITY)


def test_identity_kwargs_are_strict():
    for field in ('connection_epoch', 'policy_version', 'parameter_version'):
        bad = dict(IDENTITY, **{field: True})
        with expect_code('invalid_approval'):
            bind_approval(joint_move(Q0), make_state(), **bad)
        with expect_code('invalid_approval'):
            validate_approval({}, joint_move(Q0), make_state(), **bad)
    with expect_code('invalid_approval'):
        bind_approval(joint_move(Q0), make_state(), **dict(IDENTITY, instance_id=''))
    with expect_code('invalid_approval'):
        bind_approval(joint_move(Q0), make_state(), **dict(IDENTITY, policy_version=0))


def test_validate_rejects_tampered_tolerances():
    # Tolerances are owned by this module; a binding carrying anything else is
    # rejected, whether wider (caller-adjustable) or otherwise different.
    command = joint_move(Q0)
    binding = bind_approval(command, make_state(), **IDENTITY)
    for tampered in ({'joint_deg': JOINT_TOLERANCE_DEG * 10, 'gripper_m': GRIPPER_TOLERANCE_M},
                     {'joint_deg': JOINT_TOLERANCE_DEG, 'gripper_m': GRIPPER_TOLERANCE_M * 10},
                     {'joint_deg': JOINT_TOLERANCE_DEG / 2, 'gripper_m': GRIPPER_TOLERANCE_M}):
        with expect_code('approval_changed'):
            validate_approval(dict(binding, tolerances=tampered), command, make_state(),
                              **IDENTITY)
    for malformed in ({'joint_deg': float('nan'), 'gripper_m': GRIPPER_TOLERANCE_M},
                      {'joint_deg': JOINT_TOLERANCE_DEG, 'gripper_m': float('inf')},
                      {'joint_deg': True, 'gripper_m': GRIPPER_TOLERANCE_M},
                      {'joint_deg': -0.1, 'gripper_m': GRIPPER_TOLERANCE_M}):
        with expect_code('invalid_approval'):
            validate_approval(dict(binding, tolerances=malformed), command, make_state(),
                              **IDENTITY)


def test_widened_tolerances_cannot_cover_pose_drift():
    command = joint_move(Q0)
    binding = bind_approval(command, make_state(), **IDENTITY)
    binding['tolerances'] = {'joint_deg': JOINT_TOLERANCE_DEG * 100,
                             'gripper_m': GRIPPER_TOLERANCE_M * 100}
    moved = make_state(q_deg=[0.0, JOINT_TOLERANCE_DEG * 10, 0, 0, 0, 0])
    with expect_code('approval_changed'):
        validate_approval(binding, command, moved, **IDENTITY)


def test_validate_rejects_non_finite_measured_gripper_width():
    command = gripper_move()
    binding = bind_approval(command, make_state(gripper_width_m=0.020), **IDENTITY)
    for bad in (float('nan'), float('inf'), True, '0.02'):
        tampered = dict(binding, measured=dict(binding['measured'], gripper_width_m=bad))
        with expect_code('approval_changed'):
            validate_approval(tampered, command, make_state(gripper_width_m=0.020), **IDENTITY)
