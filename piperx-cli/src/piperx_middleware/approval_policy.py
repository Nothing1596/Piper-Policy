"""Deterministic operator approval policy.

Controller limits raise DomainError in every approval mode. Operator-enabled
force permits only an existing <=5 degree model/site overrun, with per-request
approval and no new or increasing overrun.
The mode only controls prompting: 'always' asks for every write, 'risk' asks
for critical control-mode transitions and for missing or exceeded automatic
thresholds, and 'auto' skips prompts while hard limits still apply.

TCP excursion for a joint move is always measured from the current measured
pose with the configured TCP transform (PiperKinematics), never assumed from
the original Cartesian request. This module makes no collision or world-model
claim; scene checks remain the service's job.
"""
from __future__ import annotations

import math

from pydantic import ValidationError

from .interaction_types import InteractionPolicy
from .kinematics import PiperKinematics
from .models import ControlMode, DomainError, GripperMove, JointMove, JOINT_LIMITS_DEG, Settings

JOINT_COUNT = 6

_COMMAND_MODELS = {'joint': JointMove, 'gripper': GripperMove,
                   'control_mode': ControlMode}

# Automatic thresholds checked per resolved command kind; an absent threshold
# is relevant for a kind only when listed here.
_RELEVANT_THRESHOLDS = {
    'joint': ('max_joint_step_deg', 'max_tcp_step_m', 'max_speed_percent'),
    'control_mode': ('max_speed_percent',),
    'gripper': ('max_effort_protocol',),
}

# Operator-critical configuration changes the model must never perform or
# trigger; the frontend routes these through operator channels only.
_OPERATOR_ONLY_OPERATIONS = frozenset({
    'estop', 'clear_estop', 'runtime_parameters', 'control_window',
    'interaction_policy', 'limits', 'config', 'mode', 'shutdown',
    'approve', 'deny', 'session',
})
# Operations whose criticality depends on which fields changed: with explicit
# fields they are operator-critical only when an operator-only field is among
# them (no fields means a whole-config replacement and stays critical).
_FIELD_GATED_OPERATIONS = frozenset({'config'})
_OPERATOR_ONLY_FIELDS = frozenset({
    'allow_motion', 'control_profile', 'firmware_profile', 'max_speed_percent',
    'max_move_deg', 'max_velocity_deg_s', 'gripper_max_m', 'gripper_effort_limit',
    'feedback_timeout_s', 'plan_ttl_s', 'tcp_offset_m', 'tcp_offset_rpy_deg',
    'interaction_policy', 'limits', 'automatic', 'approval_mode', 'mode',
    'target', 'policy', 'force',
})


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _finite_float(value, name: str, code: str = 'invalid_argument') -> float:
    if not _is_number(value) or not math.isfinite(value):
        raise DomainError(code, f'{name} must be a finite number.', 400)
    return float(value)


def _finite_vector(value, size: int, name: str, code: str = 'invalid_argument') -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != size:
        raise DomainError(code, f'{name} must contain exactly {size} numbers.', 400)
    return [_finite_float(item, f'{name}[{index}]', code) for index, item in enumerate(value)]


def parse_move(command: dict) -> JointMove | GripperMove | ControlMode:
    """Strictly validate a resolved move dict against the shared models."""
    if not isinstance(command, dict):
        raise DomainError('invalid_command', 'Command must be a resolved move dict.', 400)
    model = _COMMAND_MODELS.get(command.get('kind'))
    if model is None:
        raise DomainError('invalid_command',
                          f"Unknown command kind {command.get('kind')!r}.", 400)
    try:
        return model(**command)
    except ValidationError as exc:
        raise DomainError('invalid_command',
                          f'Command failed strict validation: {exc}', 400) from exc


def _robot_view(state: dict) -> dict:
    if not isinstance(state, dict):
        raise DomainError('invalid_state', 'State must be a State.public() dict.', 400)
    robot = state.get('robot')
    return robot if isinstance(robot, dict) else state


def _strict_epoch(value, name: str, code: str = 'invalid_measured_limits'):
    """A finite, whole epoch: bools, nan/inf and fractions are rejected."""
    if isinstance(value, bool) or not isinstance(value, int):
        if isinstance(value, float) and math.isfinite(value) and value.is_integer():
            return int(value)
        raise DomainError(code, f'{name} must be a finite integer epoch.', 400)
    return value


def _state_epoch(state: dict):
    ident = state.get('observation_identity')
    if isinstance(ident, dict) and _is_number(ident.get('connection_epoch')):
        candidate = ident['connection_epoch']
    elif _is_number(state.get('connection_epoch')):
        candidate = state['connection_epoch']
    else:
        return None
    try:
        return _strict_epoch(candidate, 'connection_epoch', 'invalid_state')
    except DomainError:
        return None


def _measured_rows(measured_limits: dict | None, epoch):
    """Six validated (min, max) rows, or None when limits must not be applied.

    Rows follow sdk_controls.query_limits: each joint row carries its own
    availability and the top-level flag is false when any row is missing. Every
    current-epoch available row is applied independently; unavailable rows keep
    the model/site bounds. Malformed available rows, unsafe epochs and limits
    without a verified current epoch are rejected, never approximated; a
    well-formed stale epoch is ignored.
    """
    if not isinstance(measured_limits, dict):
        return None
    rows = measured_limits.get('joints')
    if rows is None:
        if measured_limits.get('available'):
            raise DomainError('invalid_measured_limits',
                              'Available measured limits require six joint rows.', 400)
        return None
    if not isinstance(rows, list) or len(rows) != JOINT_COUNT:
        raise DomainError('invalid_measured_limits',
                          'Measured limits require six joint rows.', 400)
    parsed = []
    any_available = False
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise DomainError('invalid_measured_limits',
                              f'measured_limits.joints[{index}] must be a dict.', 400)
        if not row.get('available'):
            parsed.append(None)
            continue
        any_available = True
        low = _finite_float(row.get('min_deg'), f'measured_limits.joints[{index}].min_deg',
                            'invalid_measured_limits')
        high = _finite_float(row.get('max_deg'), f'measured_limits.joints[{index}].max_deg',
                             'invalid_measured_limits')
        if low > high:
            raise DomainError('invalid_measured_limits',
                              f'measured_limits.joints[{index}] min exceeds max.', 400)
        parsed.append((low, high))
    if not any_available:
        return None
    if 'connection_epoch' not in measured_limits:
        raise DomainError('invalid_measured_limits',
                          'Available measured limits require a connection_epoch.', 400)
    marker = _strict_epoch(measured_limits.get('connection_epoch'),
                           'measured_limits.connection_epoch')
    if epoch is None:
        raise DomainError('invalid_measured_limits',
                          'Available measured limits require a known current connection_epoch.', 400)
    if marker != epoch:
        return None
    return parsed


def _resolution_waypoints(resolution: dict | None) -> list[list[float]]:
    if resolution is None:
        return []
    if not isinstance(resolution, dict):
        raise DomainError('invalid_resolution', 'Resolution must be a dict.', 400)
    raw = resolution.get('joint_waypoints_deg')
    if raw is None:
        raw = resolution.get('waypoints_deg')
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise DomainError('invalid_resolution', 'Linear waypoints must be a list.', 400)
    return [_finite_vector(point, JOINT_COUNT, f'waypoints_deg[{index}]', 'invalid_resolution')
            for index, point in enumerate(raw)]


class PolicyEngine:
    """Evaluate resolved moves against an operator InteractionPolicy."""

    def __init__(self, policy: InteractionPolicy):
        if not isinstance(policy, InteractionPolicy):
            raise DomainError('invalid_policy', 'PolicyEngine requires an InteractionPolicy.', 400)
        self.policy = policy

    def _effective_bounds(self, rows):
        limits = self.policy.limits
        lower, upper = [], []
        for index, (model_low, model_high) in enumerate(JOINT_LIMITS_DEG):
            low, high = float(model_low), float(model_high)
            if limits.joint_lower_deg is not None:
                low = max(low, limits.joint_lower_deg[index])
            if limits.joint_upper_deg is not None:
                high = min(high, limits.joint_upper_deg[index])
            if rows is not None and rows[index] is not None:
                low = max(low, rows[index][0])
                high = min(high, rows[index][1])
            lower.append(low)
            upper.append(high)
        return lower, upper

    def _check_joint_points(self, points, rows):
        """Hard joint limits: model, then site, then valid measured."""
        limits = self.policy.limits
        for label, joints in points:
            for index, q in enumerate(joints):
                low, high = JOINT_LIMITS_DEG[index]
                if not low <= q <= high:
                    raise DomainError('joint_limits',
                        f'{label}: joint {index + 1} = {q:g} deg exceeds the PiperX model '
                        f'limits [{low:g}, {high:g}].', 422)
        for label, joints in points:
            for index, q in enumerate(joints):
                low = (limits.joint_lower_deg[index] if limits.joint_lower_deg is not None
                       else JOINT_LIMITS_DEG[index][0])
                high = (limits.joint_upper_deg[index] if limits.joint_upper_deg is not None
                        else JOINT_LIMITS_DEG[index][1])
                if not low <= q <= high:
                    raise DomainError('site_joint_limits',
                        f'{label}: joint {index + 1} = {q:g} deg exceeds the site limits '
                        f'[{low:g}, {high:g}].', 422)
        if rows is not None:
            for label, joints in points:
                for index, q in enumerate(joints):
                    if rows[index] is None:
                        continue
                    if not rows[index][0] <= q <= rows[index][1]:
                        raise DomainError('measured_joint_limits',
                            f'{label}: joint {index + 1} = {q:g} deg exceeds the queried '
                            f'controller limits [{rows[index][0]:g}, {rows[index][1]:g}].', 422)

    def _effective_limit_fields(self, rows, settings: Settings) -> dict:
        """Intersection of model, site, measured and settings limits.

        An empty window (any lower bound above its upper bound, or the gripper
        minimum above the effective maximum) is a configuration error in every
        mode: invalid bounds must never persist silently.
        """
        limits = self.policy.limits
        lower, upper = self._effective_bounds(rows)
        calibration = settings.control_profile == 'calibration'
        speed_cap = min(limits.max_speed_percent,
                        settings.max_speed_percent if calibration else 100)
        effort_cap = min(limits.max_effort_protocol,
                         settings.gripper_effort_limit if calibration else limits.max_effort_protocol)
        gripper_min = limits.gripper_min_m
        gripper_max = min(limits.gripper_max_m, settings.gripper_max_m)
        for index, (low, high) in enumerate(zip(lower, upper)):
            if low > high:
                raise DomainError('invalid_limits',
                    f'Joint {index + 1} effective window is empty: [{low:g}, {high:g}] deg.', 422)
        if gripper_min > gripper_max:
            raise DomainError('invalid_limits',
                f'Effective gripper window is empty: [{gripper_min:g}, {gripper_max:g}] m.', 422)
        return {
            'joint_lower_deg': lower,
            'joint_upper_deg': upper,
            'max_speed_percent': speed_cap,
            'gripper_min_m': gripper_min,
            'gripper_max_m': gripper_max,
            'max_effort_protocol': effort_cap,
            'measured_limits_applied': rows is not None,
            # No ROS/world collision claim is made here or anywhere downstream.
            'scene_collision_checking': False,
        }

    def effective_limits(self, *, settings: Settings,
                         measured_limits: dict | None = None,
                         connection_epoch=None) -> dict:
        """Public effective-limit intersection, identical to evaluate's result.

        Lets the caller display accurate limits and validate configuration
        without inventing a dummy motion. Rejects empty intersections.
        """
        if not isinstance(settings, Settings):
            raise DomainError('invalid_settings', 'settings must be a Settings instance.', 400)
        if connection_epoch is not None:
            connection_epoch = _strict_epoch(connection_epoch, 'connection_epoch')
        rows = _measured_rows(measured_limits, connection_epoch)
        return self._effective_limit_fields(rows, settings)

    def evaluate(self, command: dict, state: dict, *, settings: Settings,
                 resolution: dict | None = None,
                 measured_limits: dict | None = None) -> dict:
        if not isinstance(settings, Settings):
            raise DomainError('invalid_settings', 'settings must be a Settings instance.', 400)
        move = parse_move(command)
        robot = _robot_view(state)
        rows = _measured_rows(measured_limits, _state_epoch(state))
        effective_limits = self._effective_limit_fields(rows, settings)
        speed_cap = effective_limits['max_speed_percent']
        effort_cap = effective_limits['max_effort_protocol']
        gripper_min = effective_limits['gripper_min_m']
        gripper_max = effective_limits['gripper_max_m']

        recovery = None
        if self.policy.force and isinstance(move, (JointMove, ControlMode)):
            from .recovery_limits import recovery_bounds, check_bounds
            current = _finite_vector(robot.get('q_deg'), JOINT_COUNT, 'state.q_deg', 'invalid_state')
            recovery = recovery_bounds(current, effective_limits['joint_lower_deg'],
                                       effective_limits['joint_upper_deg'])
            targets = [list(move.joints_deg)] if isinstance(move, JointMove) else [current]
            for point in targets:
                check_bounds(point, recovery)
            # Hardware-reported bounds are not reprogrammed or relaxed by force.
            if rows is not None:
                for point in [current] + targets:
                    for i, row in enumerate(rows):
                        if row is not None and not row[0] <= point[i] <= row[1]:
                            raise DomainError('measured_joint_limits',
                                'Recovery cannot override queried controller limits.', 422)

        joint_step_deg = None
        tcp_step_m = None
        if isinstance(move, JointMove):
            measured_q = _finite_vector(robot.get('q_deg'), JOINT_COUNT, 'state.q_deg', 'invalid_state')
            waypoints = _resolution_waypoints(resolution)
            points = [('target', list(move.joints_deg))] + \
                     [(f'linear waypoint {i + 1}', point) for i, point in enumerate(waypoints)]
            if recovery is None:
                self._check_joint_points(points, rows)
            else:
                for _, point in points:
                    check_bounds(point, recovery)
            # Excursion is measured against every commanded point: an
            # intermediate waypoint may be farther from the measured pose than
            # the final goal.
            excursion_points = [list(move.joints_deg)] + waypoints
            joint_step_deg = max(max(abs(a - b) for a, b in zip(point, measured_q))
                                 for point in excursion_points)
            kinematics = PiperKinematics(tcp_offset_m=list(settings.tcp_offset_m),
                                         tcp_offset_rpy_deg=list(settings.tcp_offset_rpy_deg))
            current_xyz = kinematics.pose(measured_q)['xyz_m']
            tcp_step_m = max(math.dist(current_xyz, kinematics.pose(point)['xyz_m'])
                             for point in excursion_points)
            if move.speed_percent > speed_cap:
                raise DomainError('speed_limit',
                    f'Command speed {move.speed_percent} percent exceeds the effective cap '
                    f'{speed_cap} percent.', 422)
        elif isinstance(move, ControlMode):
            if move.speed_percent > speed_cap:
                raise DomainError('speed_limit',
                    f'Command speed {move.speed_percent} percent exceeds the effective cap '
                    f'{speed_cap} percent.', 422)
        else:
            if not gripper_min <= move.width_m <= gripper_max:
                raise DomainError('gripper_limits',
                    f'Requested width {move.width_m:g} m is outside the effective bounds '
                    f'[{gripper_min:g}, {gripper_max:g}] m.', 422)
            if move.effort_protocol > effort_cap:
                raise DomainError('gripper_limits',
                    f'Requested effort {move.effort_protocol:g} exceeds the effective cap '
                    f'{effort_cap:g}.', 422)

        reasons = []
        if self.policy.force:
            reasons.append('force recovery: each request requires separate operator confirmation; at most 5 degrees existing overrun, hold or inward only')
            if recovery is not None:
                reasons.append(f'measured joints (deg): {current}; approved recovery envelope (deg): {recovery}')
        if self.policy.mode == 'always':
            reasons.append("approval mode 'always': every write requires operator approval")
        elif self.policy.mode == 'risk':
            if isinstance(move, ControlMode):
                reasons.append('critical control_mode transition requires operator approval')
            automatic = self.policy.automatic
            for name in _RELEVANT_THRESHOLDS[move.kind]:
                limit = getattr(automatic, name)
                if name == 'max_joint_step_deg':
                    observed = joint_step_deg
                elif name == 'max_tcp_step_m':
                    observed = tcp_step_m
                elif name == 'max_speed_percent':
                    observed = move.speed_percent
                else:
                    observed = move.effort_protocol
                if limit is None:
                    reasons.append(f"missing automatic threshold '{name}': operator approval required")
                elif observed > limit:
                    reasons.append(f"automatic threshold '{name}' exceeded: {observed:g} > {limit:g}")

        return {'decision': 'ask' if reasons else 'allow',
                'reasons': reasons,
                'effective_limits': effective_limits,
                'recovery_bounds_deg': recovery}


def evaluate_operation(operation: str, changed_fields: list[str] | None = None) -> dict:
    """Classify an operation/field change as operator-critical or model-visible.

    Operator powers (estop, runtime parameters, interaction policy, mode
    switches, shutdown, approvals, ...) are never reported as model-visible.
    """
    if not isinstance(operation, str) or not operation.strip():
        raise DomainError('invalid_operation', 'operation must be a non-empty string.', 400)
    fields = list(changed_fields) if changed_fields is not None else []
    if any(not isinstance(field, str) or not field for field in fields):
        raise DomainError('invalid_operation', 'changed_fields must be non-empty strings.', 400)
    normalized = operation.strip().lower()
    if normalized in _FIELD_GATED_OPERATIONS:
        critical = not fields or any(field in _OPERATOR_ONLY_FIELDS for field in fields)
    else:
        critical = (normalized in _OPERATOR_ONLY_OPERATIONS or
                    any(field in _OPERATOR_ONLY_FIELDS for field in fields))
    return {'operation': operation, 'changed_fields': fields,
            'operator_critical': critical, 'model_visible': not critical,
            'reason': None if not critical else
                      'operator-critical configuration: the model must not perform or trigger it'}
