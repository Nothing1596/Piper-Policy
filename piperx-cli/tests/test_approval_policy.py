import math
from contextlib import contextmanager

import pytest

from piperx_middleware.approval_policy import PolicyEngine, evaluate_operation, parse_move
from piperx_middleware.interaction_types import AutoApproval, ExecutionLimits, InteractionPolicy
from piperx_middleware.kinematics import PiperKinematics
from piperx_middleware.models import DomainError, JOINT_LIMITS_DEG, Settings

Q0 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


@contextmanager
def expect_code(code):
    with pytest.raises(DomainError) as info:
        yield info
    assert info.value.code == code


def make_state(q_deg=Q0, gripper_width_m=0.02, epoch=0):
    return {'connected': True, 'q_deg': list(q_deg), 'gripper_width_m': gripper_width_m,
            'observation_identity': {'connection_epoch': epoch}, 'connection_epoch': epoch}


def joint_move(joints, speed=5):
    return {'kind': 'joint', 'joints_deg': list(joints), 'speed_percent': speed, 'timeout_s': 30}


def gripper_move(width=0.03, effort=0.5):
    return {'kind': 'gripper', 'width_m': width, 'effort_protocol': effort,
            'timeout_s': 10, 'completion': 'width'}


def control_mode(speed=5):
    return {'kind': 'control_mode', 'speed_percent': speed, 'timeout_s': 3}


def make_policy(mode='risk', *, automatic=None, limits=None):
    return InteractionPolicy(mode=mode,
                             automatic=automatic if automatic is not None else AutoApproval(),
                             limits=limits if limits is not None else ExecutionLimits())


def permissive_auto():
    return AutoApproval(max_joint_step_deg=180.0, max_tcp_step_m=10.0,
                        max_speed_percent=100, max_effort_protocol=32.767)


# --- strict input validation -------------------------------------------------

def test_rejects_unknown_and_malformed_command():
    engine = PolicyEngine(make_policy())
    with expect_code('invalid_command'):
        engine.evaluate({'kind': 'teleport'}, make_state(), settings=Settings())
    with expect_code('invalid_command'):
        engine.evaluate({'kind': 'joint', 'joints_deg': Q0[:5], 'speed_percent': 5},
                        make_state(), settings=Settings())
    with expect_code('invalid_command'):
        engine.evaluate(joint_move([float('nan')] * 6), make_state(), settings=Settings())
    with expect_code('invalid_command'):
        engine.evaluate(joint_move([float('inf')] * 6), make_state(), settings=Settings())
    with expect_code('invalid_command'):
        engine.evaluate({'kind': 'joint', 'joints_deg': Q0, 'speed_percent': True},
                        make_state(), settings=Settings())


def test_rejects_bad_state_resolution_and_measured_inputs():
    engine = PolicyEngine(make_policy())
    with expect_code('invalid_state'):
        engine.evaluate(joint_move(Q0), {'q_deg': Q0[:5]}, settings=Settings())
    with expect_code('invalid_state'):
        engine.evaluate(joint_move(Q0), {'q_deg': [False] * 6}, settings=Settings())
    with expect_code('invalid_state'):
        engine.evaluate(joint_move(Q0), {'q_deg': [0.0] * 5 + [float('nan')]}, settings=Settings())
    with expect_code('invalid_resolution'):
        engine.evaluate(joint_move(Q0), make_state(), settings=Settings(),
                        resolution={'waypoints_deg': 'not-a-list'})
    with expect_code('invalid_resolution'):
        engine.evaluate(joint_move(Q0), make_state(), settings=Settings(),
                        resolution={'waypoints_deg': [[0.0] * 5]})
    with expect_code('invalid_measured_limits'):
        engine.evaluate(joint_move(Q0), make_state(), settings=Settings(),
                        measured_limits={'available': True, 'joints': []})


def test_rejects_wrong_constructed_types():
    with expect_code('invalid_policy'):
        PolicyEngine({'mode': 'risk'})
    engine = PolicyEngine(make_policy())
    with expect_code('invalid_settings'):
        engine.evaluate(joint_move(Q0), make_state(), settings={'gripper_max_m': 0.07})


# --- hard limits in every mode ------------------------------------------------

@pytest.mark.parametrize('mode', ['always', 'risk', 'auto'])
def test_model_joint_limits_raise_in_all_modes(mode):
    engine = PolicyEngine(make_policy(mode))
    with expect_code('joint_limits'):
        engine.evaluate(joint_move([200.0, 0, 0, 0, 0, 0]), make_state(), settings=Settings())


@pytest.mark.parametrize('mode', ['always', 'risk', 'auto'])
def test_site_joint_limits_raise_in_all_modes(mode):
    limits = ExecutionLimits(joint_lower_deg=[-10.0] * 6, joint_upper_deg=[10.0] * 6)
    engine = PolicyEngine(make_policy(mode, limits=limits))
    with expect_code('site_joint_limits'):
        engine.evaluate(joint_move([20.0, 0, 0, 0, 0, 0]), make_state(), settings=Settings())
    result = engine.evaluate(joint_move([5.0, 0, 0, 0, 0, 0]), make_state(), settings=Settings())
    # Effective bounds are the intersection with the model limits.
    assert result['effective_limits']['joint_lower_deg'] == [-10.0, 0.0, -10.0, -10.0, -10.0, -10.0]
    assert result['effective_limits']['joint_upper_deg'] == [10.0, 10.0, 0.0, 10.0, 10.0, 10.0]


@pytest.mark.parametrize('mode', ['always', 'risk', 'auto'])
def test_measured_joint_limits_raise_only_when_valid_current_epoch(mode):
    measured = {'available': True, 'connection_epoch': 0,
                'joints': [{'joint': i + 1, 'available': True,
                            'min_deg': -100.0, 'max_deg': 100.0} for i in range(6)]}
    engine = PolicyEngine(make_policy(mode))
    with expect_code('measured_joint_limits'):
        engine.evaluate(joint_move([-120.0, 0, 0, 0, 0, 0]), make_state(epoch=0),
                        settings=Settings(), measured_limits=measured)
    # Stale epoch: measured limits must not be applied.
    result = engine.evaluate(joint_move([-120.0, 0, 0, 0, 0, 0]), make_state(epoch=1),
                             settings=Settings(), measured_limits=measured)
    assert result['effective_limits']['measured_limits_applied'] is False
    # Unavailable: ignored entirely.
    result = engine.evaluate(joint_move([-120.0, 0, 0, 0, 0, 0]), make_state(),
                             settings=Settings(),
                             measured_limits={'available': False, 'reason': 'no_physical_hardware'})
    assert result['effective_limits']['measured_limits_applied'] is False


@pytest.mark.parametrize('mode', ['always', 'risk', 'auto'])
def test_linear_waypoints_checked_in_all_modes(mode):
    engine = PolicyEngine(make_policy(mode))
    resolution = {'waypoints_deg': [[0.0] * 6, [160.0, 0, 0, 0, 0, 0]]}
    with expect_code('joint_limits'):
        engine.evaluate(joint_move(Q0), make_state(), settings=Settings(),
                        resolution=resolution)
    # Same waypoint via the contract's joint_waypoints_deg key.
    with expect_code('joint_limits'):
        engine.evaluate(joint_move(Q0), make_state(), settings=Settings(),
                        resolution={'joint_waypoints_deg': [[0.0] * 6, [0.0, 200.0, 0, 0, 0, 0]]})


@pytest.mark.parametrize('mode', ['always', 'risk', 'auto'])
def test_settings_gripper_limit_raises_in_all_modes(mode):
    engine = PolicyEngine(make_policy(mode))
    with expect_code('gripper_limits'):
        engine.evaluate(gripper_move(width=0.08), make_state(), settings=Settings())


def test_policy_gripper_window_and_effort_caps():
    limits = ExecutionLimits(gripper_min_m=0.01, gripper_max_m=0.05, max_effort_protocol=1.0)
    engine = PolicyEngine(make_policy(limits=limits))
    with expect_code('gripper_limits'):
        engine.evaluate(gripper_move(width=0.06), make_state(), settings=Settings())
    with expect_code('gripper_limits'):
        engine.evaluate(gripper_move(width=0.03, effort=2.0), make_state(), settings=Settings())
    with expect_code('gripper_limits'):
        engine.evaluate(gripper_move(width=0.005), make_state(), settings=Settings())


def test_calibration_profile_caps_from_settings():
    settings = Settings(control_profile='calibration', max_speed_percent=5,
                        gripper_effort_limit=0.5)
    engine = PolicyEngine(make_policy())
    with expect_code('speed_limit'):
        engine.evaluate(joint_move(Q0, speed=50), make_state(), settings=settings)
    with expect_code('speed_limit'):
        engine.evaluate(control_mode(speed=50), make_state(), settings=settings)
    with expect_code('gripper_limits'):
        engine.evaluate(gripper_move(effort=0.8), make_state(), settings=settings)


def test_policy_speed_cap_direct_profile():
    limits = ExecutionLimits(max_speed_percent=30)
    engine = PolicyEngine(make_policy(limits=limits))
    with expect_code('speed_limit'):
        engine.evaluate(joint_move(Q0, speed=50), make_state(), settings=Settings())
    with expect_code('speed_limit'):
        engine.evaluate(control_mode(speed=31), make_state(), settings=Settings())


# --- mode behaviour -----------------------------------------------------------

def test_risk_asks_when_relevant_threshold_missing():
    engine = PolicyEngine(make_policy('risk'))  # AutoApproval fully empty
    result = engine.evaluate(joint_move([1.0, 0, 0, 0, 0, 0]), make_state(), settings=Settings())
    assert result['decision'] == 'ask'
    assert any('max_joint_step_deg' in reason for reason in result['reasons'])
    assert any('max_tcp_step_m' in reason for reason in result['reasons'])
    assert any('max_speed_percent' in reason for reason in result['reasons'])
    gripper = engine.evaluate(gripper_move(), make_state(), settings=Settings())
    assert any('max_effort_protocol' in reason for reason in gripper['reasons'])


def test_risk_allows_when_thresholds_configured_and_respected():
    engine = PolicyEngine(make_policy('risk', automatic=permissive_auto()))
    result = engine.evaluate(joint_move([1.0, 0, 0, 0, 0, 0]), make_state(), settings=Settings())
    assert result['decision'] == 'allow'
    assert result['reasons'] == []


def test_risk_asks_on_threshold_exceeded():
    automatic = AutoApproval(max_joint_step_deg=0.5, max_tcp_step_m=10.0, max_speed_percent=100)
    engine = PolicyEngine(make_policy('risk', automatic=automatic))
    result = engine.evaluate(joint_move([2.0, 0, 0, 0, 0, 0]), make_state(), settings=Settings())
    assert result['decision'] == 'ask'
    assert any("exceeded" in reason and 'max_joint_step_deg' in reason
               for reason in result['reasons'])


def test_tcp_excursion_computed_from_current_pose_with_configured_tcp():
    settings = Settings(tcp_offset_m=[0.1, 0.0, 0.0], tcp_offset_rpy_deg=[0.0, 0.0, 0.0])
    kin = PiperKinematics(tcp_offset_m=list(settings.tcp_offset_m),
                          tcp_offset_rpy_deg=list(settings.tcp_offset_rpy_deg))
    measured_q = [10.0, 20.0, -30.0, 5.0, -5.0, 0.0]
    target_q = [12.0, 20.0, -30.0, 5.0, -5.0, 0.0]
    expected = math.dist(kin.pose(measured_q)['xyz_m'], kin.pose(target_q)['xyz_m'])
    threshold = expected - 1e-4
    engine = PolicyEngine(make_policy('risk', automatic=AutoApproval(
        max_joint_step_deg=180.0, max_tcp_step_m=threshold, max_speed_percent=100)))
    result = engine.evaluate(joint_move(target_q), make_state(q_deg=measured_q),
                             settings=settings)
    assert result['decision'] == 'ask'
    assert any('max_tcp_step_m' in reason for reason in result['reasons'])
    # The same excursion the test computes from kinematics appears in the reason.
    assert any(f'{expected:.4f}' in reason for reason in result['reasons'])
    # A slightly larger threshold allows the move.
    engine = PolicyEngine(make_policy('risk', automatic=AutoApproval(
        max_joint_step_deg=180.0, max_tcp_step_m=expected + 1e-3, max_speed_percent=100)))
    assert engine.evaluate(joint_move(target_q), make_state(q_deg=measured_q),
                           settings=settings)['decision'] == 'allow'


def test_risk_asks_for_control_mode_even_with_thresholds_configured():
    engine = PolicyEngine(make_policy('risk', automatic=permissive_auto()))
    result = engine.evaluate(control_mode(), make_state(), settings=Settings())
    assert result['decision'] == 'ask'
    assert any('control_mode' in reason for reason in result['reasons'])


def test_auto_mode_never_asks_but_keeps_hard_limits():
    engine = PolicyEngine(make_policy('auto'))  # empty AutoApproval: nothing may be missing
    move = joint_move([30.0, 0, 0, 0, 0, 0], speed=100)
    assert engine.evaluate(move, make_state(), settings=Settings())['decision'] == 'allow'
    assert engine.evaluate(control_mode(), make_state(), settings=Settings())['decision'] == 'allow'
    with expect_code('joint_limits'):
        engine.evaluate(joint_move([200.0, 0, 0, 0, 0, 0]), make_state(), settings=Settings())


def test_always_mode_asks_for_every_write():
    engine = PolicyEngine(make_policy('always', automatic=permissive_auto()))
    for command in (joint_move([0.5, 0, 0, 0, 0, 0]), gripper_move(), control_mode()):
        result = engine.evaluate(command, make_state(), settings=Settings())
        assert result['decision'] == 'ask'
        assert any("'always'" in reason for reason in result['reasons'])


def test_effective_limits_merge_all_sources():
    limits = ExecutionLimits(joint_lower_deg=[-90.0] * 6, joint_upper_deg=[90.0] * 6,
                             gripper_max_m=0.06)
    measured = {'available': True, 'connection_epoch': 3,
                'joints': [{'joint': i + 1, 'available': True,
                            'min_deg': -80.0, 'max_deg': 80.0} for i in range(6)]}
    engine = PolicyEngine(make_policy(limits=limits))
    result = engine.evaluate(joint_move(Q0), make_state(epoch=3), settings=Settings(),
                             measured_limits=measured)
    effective = result['effective_limits']
    # Site and measured bounds intersect with the PiperX model limits.
    assert effective['joint_lower_deg'] == [-80.0, 0.0, -80.0, -80.0, -80.0, -80.0]
    assert effective['joint_upper_deg'] == [80.0, 80.0, 0.0, 80.0, 80.0, 80.0]
    assert effective['gripper_max_m'] == 0.06
    assert effective['measured_limits_applied'] is True
    assert effective['scene_collision_checking'] is False
    with expect_code('measured_joint_limits'):
        engine.evaluate(joint_move([85.0, 0, 0, 0, 0, 0]), make_state(epoch=3),
                        settings=Settings(), measured_limits=measured)


def test_no_collision_claim_anywhere():
    engine = PolicyEngine(make_policy('risk'))
    result = engine.evaluate(joint_move([1.0, 0, 0, 0, 0, 0]), make_state(), settings=Settings())
    assert result['effective_limits']['scene_collision_checking'] is False


# --- operation classification -------------------------------------------------

def test_evaluate_operation_marks_operator_critical():
    for operation in ('estop', 'clear_estop', 'runtime_parameters', 'control_window',
                      'interaction_policy', 'limits', 'config', 'mode', 'shutdown',
                      'approve', 'deny'):
        result = evaluate_operation(operation)
        assert result['operator_critical'] is True
        assert result['model_visible'] is False
    result = evaluate_operation('robot_move_joints')
    assert result['operator_critical'] is False
    assert result['model_visible'] is True
    result = evaluate_operation('config', ['gripper_max_m'])
    assert result['operator_critical'] is True
    result = evaluate_operation('config', ['simulation_seed'])
    assert result['operator_critical'] is False


def test_evaluate_operation_strict_inputs():
    with expect_code('invalid_operation'):
        evaluate_operation('')
    with expect_code('invalid_operation'):
        evaluate_operation('estop', ['ok', 3])


def test_parse_move_canonicalizes_defaults():
    move = parse_move({'kind': 'joint', 'joints_deg': Q0})
    assert move.speed_percent == 5 and move.timeout_s == 30


# --- partial measured limits and strict epochs --------------------------------

def measured_rows(min_deg=-100.0, max_deg=100.0, unavailable=()):
    rows = []
    for i in range(6):
        if i in unavailable:
            rows.append({'joint': i + 1, 'available': False, 'reason': 'query_timeout'})
        else:
            rows.append({'joint': i + 1, 'available': True,
                         'min_deg': min_deg, 'max_deg': max_deg})
    return rows


@pytest.mark.parametrize('mode', ['always', 'risk', 'auto'])
def test_partial_measured_rows_apply_per_row(mode):
    # Top-level available is false because joint 6 never answered.
    measured = {'available': False, 'connection_epoch': 2,
                'joints': measured_rows(unavailable={5})}
    engine = PolicyEngine(make_policy(mode))
    result = engine.evaluate(joint_move(Q0), make_state(epoch=2), settings=Settings(),
                             measured_limits=measured)
    effective = result['effective_limits']
    assert effective['measured_limits_applied'] is True
    for index in range(5):
        low, high = JOINT_LIMITS_DEG[index]
        assert effective['joint_lower_deg'][index] == max(-100.0, float(low))
        assert effective['joint_upper_deg'][index] == min(100.0, float(high))
    # The unavailable row keeps the model bounds, not default zeros.
    assert effective['joint_lower_deg'][5] == float(JOINT_LIMITS_DEG[5][0])
    assert effective['joint_upper_deg'][5] == float(JOINT_LIMITS_DEG[5][1])
    # An available row still restricts even though overall available is false.
    with expect_code('measured_joint_limits'):
        engine.evaluate(joint_move([-120.0, 0, 0, 0, 0, 0]), make_state(epoch=2),
                        settings=Settings(), measured_limits=measured)


def test_measured_limits_strict_epochs_and_malformed_rows():
    engine = PolicyEngine(make_policy())
    good = measured_rows()
    # Available rows without a usable epoch are rejected, never accepted unbounded.
    with expect_code('invalid_measured_limits'):
        engine.evaluate(joint_move(Q0), make_state(epoch=2), settings=Settings(),
                        measured_limits={'available': True, 'joints': good})
    for bad_epoch in (True, float('nan'), float('inf'), 1.5, '2'):
        with expect_code('invalid_measured_limits'):
            engine.evaluate(joint_move(Q0), make_state(epoch=2), settings=Settings(),
                            measured_limits={'available': True, 'connection_epoch': bad_epoch,
                                             'joints': good})
    # A stale but well-formed epoch is ignored, not applied.
    result = engine.evaluate(joint_move(Q0), make_state(epoch=3), settings=Settings(),
                             measured_limits={'available': True, 'connection_epoch': 2,
                                              'joints': good})
    assert result['effective_limits']['measured_limits_applied'] is False
    # Malformed available rows are rejected.
    nan_row = measured_rows()
    nan_row[2] = {'joint': 3, 'available': True, 'min_deg': float('nan'), 'max_deg': 10.0}
    with expect_code('invalid_measured_limits'):
        engine.evaluate(joint_move(Q0), make_state(epoch=2), settings=Settings(),
                        measured_limits={'available': True, 'connection_epoch': 2,
                                         'joints': nan_row})
    inverted = measured_rows()
    inverted[1] = {'joint': 2, 'available': True, 'min_deg': 10.0, 'max_deg': -10.0}
    with expect_code('invalid_measured_limits'):
        engine.evaluate(joint_move(Q0), make_state(epoch=2), settings=Settings(),
                        measured_limits={'available': True, 'connection_epoch': 2,
                                         'joints': inverted})
    not_dict = measured_rows()
    not_dict[4] = ['joint', 5]
    with expect_code('invalid_measured_limits'):
        engine.evaluate(joint_move(Q0), make_state(epoch=2), settings=Settings(),
                        measured_limits={'available': True, 'connection_epoch': 2,
                                         'joints': not_dict})
    # A missing limits payload with an available top-level flag stays invalid.
    with expect_code('invalid_measured_limits'):
        engine.evaluate(joint_move(Q0), make_state(), settings=Settings(),
                        measured_limits={'available': True})


# --- public effective_limits ---------------------------------------------------

EFFECTIVE_FIELDS = {'joint_lower_deg', 'joint_upper_deg', 'max_speed_percent',
                    'gripper_min_m', 'gripper_max_m', 'max_effort_protocol',
                    'measured_limits_applied', 'scene_collision_checking'}


def test_effective_limits_method_matches_evaluate():
    limits = ExecutionLimits(joint_lower_deg=[-90.0] * 6, joint_upper_deg=[90.0] * 6,
                             gripper_max_m=0.06)
    measured = {'available': False, 'connection_epoch': 3,
                'joints': measured_rows(min_deg=-80.0, max_deg=80.0, unavailable={5})}
    engine = PolicyEngine(make_policy(limits=limits))
    result = engine.evaluate(joint_move(Q0), make_state(epoch=3), settings=Settings(),
                             measured_limits=measured)
    direct = engine.effective_limits(settings=Settings(), measured_limits=measured,
                                     connection_epoch=3)
    assert direct == result['effective_limits']
    assert set(direct) == EFFECTIVE_FIELDS
    # Works without measured limits too.
    assert engine.effective_limits(settings=Settings())['joint_lower_deg'] == \
        [max(-90.0, float(low)) for low, _ in JOINT_LIMITS_DEG]
    # The method validates its own epoch and settings strictly.
    with expect_code('invalid_measured_limits'):
        engine.effective_limits(settings=Settings(), measured_limits=measured,
                                connection_epoch=1.5)
    with expect_code('invalid_settings'):
        engine.effective_limits(settings={'gripper_max_m': 0.07})


@pytest.mark.parametrize('mode', ['always', 'risk', 'auto'])
def test_empty_joint_intersection_rejected_in_all_modes(mode):
    engine = PolicyEngine(make_policy(mode))
    # A measured window disjoint from the model limits empties the intersection.
    rows = measured_rows()
    rows[0] = {'joint': 1, 'available': True, 'min_deg': 200.0, 'max_deg': 300.0}
    measured = {'available': True, 'connection_epoch': 1, 'joints': rows}
    with expect_code('invalid_limits'):
        engine.evaluate(joint_move(Q0), make_state(epoch=1), settings=Settings(),
                        measured_limits=measured)
    with expect_code('invalid_limits'):
        engine.effective_limits(settings=Settings(), measured_limits=measured,
                                connection_epoch=1)


def test_empty_gripper_window_rejected():
    limits = ExecutionLimits(gripper_min_m=0.075, gripper_max_m=0.09)
    engine = PolicyEngine(make_policy(limits=limits))
    # The settings cap (default 0.07 m) sits below the configured minimum.
    with expect_code('invalid_limits'):
        engine.effective_limits(settings=Settings())
    with expect_code('invalid_limits'):
        engine.evaluate(gripper_move(width=0.06), make_state(), settings=Settings())


# --- excursion considers every waypoint ---------------------------------------

def test_risk_excursion_considers_every_waypoint():
    automatic = AutoApproval(max_joint_step_deg=5.0, max_tcp_step_m=0.01,
                             max_speed_percent=100)
    engine = PolicyEngine(make_policy('risk', automatic=automatic))
    # The final goal is small; an intermediate waypoint exceeds both thresholds.
    resolution = {'waypoints_deg': [[30.0, 0, 0, 0, 0, 0]]}
    result = engine.evaluate(joint_move([1.0, 0, 0, 0, 0, 0]), make_state(), settings=Settings(),
                             resolution=resolution)
    assert result['decision'] == 'ask'
    assert any('max_joint_step_deg' in reason for reason in result['reasons'])
    assert any('max_tcp_step_m' in reason for reason in result['reasons'])
    # Without the intermediate waypoint the same final goal is allowed.
    allowed = engine.evaluate(joint_move([1.0, 0, 0, 0, 0, 0]), make_state(),
                              settings=Settings())
    assert allowed['decision'] == 'allow'
    assert allowed['reasons'] == []
