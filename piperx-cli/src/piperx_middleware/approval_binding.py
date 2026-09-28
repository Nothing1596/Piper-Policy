"""Operator approval bindings.

A binding pins one operator approval to a canonical command, an executor
identity (instance/epoch/policy/parameter versions), and the measured pose at
approval time. Approvals do not expire on wall-clock time; the executor checks
fresh feedback at execution, and validate_approval rejects any drift beyond the
recorded tolerances (0.1 deg joints, 0.001 m gripper). The tolerances are
owned by this module: a binding carrying anything else is rejected, so a
caller cannot widen what an approval covers.
"""
from __future__ import annotations

import math

from .approval_policy import (_finite_float, _finite_vector, _is_number,
                              _robot_view, parse_move)
from .models import DomainError

BINDING_VERSION = 1
JOINT_TOLERANCE_DEG = 0.1
GRIPPER_TOLERANCE_M = 0.001
JOINT_COUNT = 6
# Exact-boundary drift is inclusive; the epsilon only absorbs float noise.
_DRIFT_EPSILON = 1e-9


def _check_identity(*, instance_id: str, connection_epoch: int,
                    policy_version: int, parameter_version: int) -> dict:
    if not isinstance(instance_id, str) or not instance_id:
        raise DomainError('invalid_approval', 'instance_id must be a non-empty string.', 400)
    for name, value, minimum in (('connection_epoch', connection_epoch, 0),
                                 ('policy_version', policy_version, 1),
                                 ('parameter_version', parameter_version, 0)):
        if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
            raise DomainError('invalid_approval',
                              f'{name} must be an integer >= {minimum}.', 400)
    return {'instance_id': instance_id, 'connection_epoch': connection_epoch,
            'policy_version': policy_version, 'parameter_version': parameter_version}


def _changed(message: str):
    raise DomainError('approval_changed', message, 409)


def bind_approval(command: dict, state: dict, *, instance_id: str,
                  connection_epoch: int, policy_version: int,
                  parameter_version: int) -> dict:
    """Bind an operator approval to the canonical command and measured pose."""
    canonical = parse_move(command).model_dump()
    binding = {'binding_version': BINDING_VERSION, 'command': canonical}
    binding.update(_check_identity(instance_id=instance_id, connection_epoch=connection_epoch,
                                   policy_version=policy_version,
                                   parameter_version=parameter_version))
    robot = _robot_view(state)
    measured = {'q_deg': _finite_vector(robot.get('q_deg'), JOINT_COUNT, 'state.q_deg',
                                        'invalid_state'),
                'gripper_width_m': None}
    if canonical['kind'] == 'gripper':
        measured['gripper_width_m'] = _finite_float(robot.get('gripper_width_m'),
                                                    'state.gripper_width_m', 'invalid_state')
    binding['measured'] = measured
    binding['tolerances'] = {'joint_deg': JOINT_TOLERANCE_DEG,
                             'gripper_m': GRIPPER_TOLERANCE_M}
    return binding


def validate_approval(binding: dict, command: dict, state: dict, *, instance_id: str,
                      connection_epoch: int, policy_version: int,
                      parameter_version: int) -> None:
    """Verify an approval still covers this exact command, identity and pose."""
    identity = _check_identity(instance_id=instance_id, connection_epoch=connection_epoch,
                               policy_version=policy_version,
                               parameter_version=parameter_version)
    if not isinstance(binding, dict):
        raise DomainError('invalid_approval', 'binding must be a dict.', 400)
    for key, value in identity.items():
        if binding.get(key) != value:
            _changed(f'Approval {key} no longer matches the current executor.')
    if binding.get('binding_version') != BINDING_VERSION:
        _changed('Approval binding version is not current.')
    tolerances = binding.get('tolerances')
    if not isinstance(tolerances, dict):
        _changed('Approval carries no tolerances.')
    joint_tolerance = _finite_float(tolerances.get('joint_deg'), 'tolerances.joint_deg',
                                    'invalid_approval')
    gripper_tolerance = _finite_float(tolerances.get('gripper_m'), 'tolerances.gripper_m',
                                      'invalid_approval')
    if joint_tolerance < 0 or gripper_tolerance < 0:
        raise DomainError('invalid_approval', 'Approval tolerances must be non-negative.', 400)
    if joint_tolerance != JOINT_TOLERANCE_DEG or gripper_tolerance != GRIPPER_TOLERANCE_M:
        _changed('Approval tolerances differ from the recorded instruction tolerances.')
    if parse_move(command).model_dump() != binding.get('command'):
        _changed('Command differs from the approved command.')
    measured = binding.get('measured')
    if not isinstance(measured, dict):
        _changed('Approval carries no measured pose.')
    bound_q = _finite_vector(measured.get('q_deg'), JOINT_COUNT, 'measured.q_deg',
                             'invalid_approval')
    robot = _robot_view(state)
    current_q = _finite_vector(robot.get('q_deg'), JOINT_COUNT, 'state.q_deg', 'invalid_state')
    for index, (bound, current) in enumerate(zip(bound_q, current_q)):
        if abs(bound - current) > JOINT_TOLERANCE_DEG + _DRIFT_EPSILON:
            _changed(f'Joint {index + 1} moved {abs(bound - current):g} deg since approval, '
                     f'beyond the {JOINT_TOLERANCE_DEG:g} deg tolerance.')
    if binding['command']['kind'] == 'gripper':
        bound_width = measured.get('gripper_width_m')
        if not _is_number(bound_width) or not math.isfinite(bound_width):
            _changed('Approval for a gripper command carries no measured width.')
        current_width = _finite_float(robot.get('gripper_width_m'), 'state.gripper_width_m',
                                      'invalid_state')
        if abs(float(bound_width) - current_width) > GRIPPER_TOLERANCE_M + _DRIFT_EPSILON:
            _changed(f'Gripper width moved {abs(float(bound_width) - current_width):g} m since '
                     f'approval, beyond the {GRIPPER_TOLERANCE_M:g} m tolerance.')
