"""Operator-only scoring for one adapted can-to-basket task; no policy feedback."""
import numpy as np
from scipy.spatial.transform import Rotation

THRESHOLDS = {'lift_m', 'translation_m', 'distractor_motion_m', 'stability_m',
              'stability_rotation_deg', 'tcp_clearance_m', 'terminal_window_s'}


def vector(value, size, name):
    raw = np.asarray(value)
    if raw.dtype.kind not in 'iuf' or raw.shape != (size,) or not np.isfinite(raw).all():
        raise ValueError('Invalid measured ' + name)
    return raw.astype(float)


def validate_spec(spec):
    try:
        if spec['scope'] != 'watchact_imitation_piperx_v1':
            raise ValueError('Wrong scorer scope')
        if not all(isinstance(spec[k], str) and spec[k] for k in ('target', 'distractor')) or spec['target'] == spec['distractor']:
            raise ValueError('Distinct target and distractor required')
        limits = spec['thresholds']
        if not THRESHOLDS <= limits.keys():
            raise ValueError('Missing thresholds')
        shape = spec['target_shape']
        if shape['type'] != 'cylinder':
            raise ValueError('Only cylinder geometry supported')
        for name, value in [*limits.items(), ('radius_m', shape['radius_m']), ('half_height_m', shape['half_height_m'])]:
            if type(value) not in (int, float) or not np.isfinite(value) or value <= 0:
                raise ValueError('Invalid threshold or dimension: ' + name)
        low = vector(spec['container_bounds']['min'], 3, 'container minimum')
        high = vector(spec['container_bounds']['max'], 3, 'container maximum')
        if not (high > low).all():
            raise ValueError('Reversed container bounds')
        if 'auxiliary_pick_return' in spec:
            aux = spec['auxiliary_pick_return']
            if aux['body'] != spec['distractor']:
                raise ValueError('Auxiliary object must be the second measured body')
            half = vector(aux['half_size_m'], 3, 'auxiliary box half-size')
            low_xy = vector(aux['return_xy']['min'], 2, 'auxiliary region minimum')
            high_xy = vector(aux['return_xy']['max'], 2, 'auxiliary region maximum')
            if not (half > 0).all() or not (high_xy > low_xy).all():
                raise ValueError('Invalid auxiliary box or region')
            if type(aux['surface_z_m']) not in (int, float) or not np.isfinite(aux['surface_z_m']):
                raise ValueError('Invalid auxiliary support surface')
        return low, high
    except (KeyError, TypeError) as error:
        raise ValueError('Missing scorer configuration') from error


def pose(payload, key):
    try:
        item = payload['objects'][key]
        pos = vector(item['position'], 3, key+' position')
        quat = vector(item['quaternion'], 4, key+' quaternion')
        if abs(np.linalg.norm(quat)-1) > .001:
            raise ValueError('Invalid quaternion norm')
        if type(item['grasped']) is not bool:
            raise ValueError('Missing measured contact state')
        return pos, Rotation.from_quat(quat, scalar_first=True), item['grasped']
    except (KeyError, TypeError) as error:
        raise ValueError('Missing measured object: '+key) from error


def contained(payload, spec):
    pos, rotation, _ = pose(payload, spec['target'])
    axis = rotation.as_matrix()[:, 2]
    shape = spec['target_shape']
    extent = shape['radius_m']*np.sqrt(np.maximum(0, 1-axis**2)) + shape['half_height_m']*np.abs(axis)
    low, high = spec['container_bounds']['min'], spec['container_bounds']['max']
    return bool(np.all(pos-extent >= low) and np.all(pos+extent <= high))


def score_task(initial, trace, terminal, spec):
    validate_spec(spec)
    if not trace:
        raise ValueError('Insufficient trace samples')
    if len(terminal) < 2:
        raise ValueError('Insufficient terminal samples')
    target, distractor, th = spec['target'], spec['distractor'], spec['thresholds']
    all_samples = [initial, *trace, *terminal]
    for sample in all_samples:
        pose(sample, target); pose(sample, distractor)
        vector(sample.get('robot', {}).get('tcp_pos'), 3, 'TCP')
        t = sample.get('sim_time_s')
        if type(t) not in (int, float) or not np.isfinite(t):
            raise ValueError('Invalid measurement timestamp')
    for group in (trace, terminal):
        if any(b['sim_time_s'] <= a['sim_time_s'] for a, b in zip(group, group[1:])):
            raise ValueError('Timestamps not strictly increasing')
    if trace[0]['sim_time_s'] < initial['sim_time_s'] or terminal[0]['sim_time_s'] < trace[-1]['sim_time_s']:
        raise ValueError('Evidence chronology reversed')
    span = terminal[-1]['sim_time_s']-terminal[0]['sim_time_s']
    if span < th['terminal_window_s']:
        raise ValueError('Insufficient terminal window duration')
    start = pose(initial, target)[0]
    dist_start = pose(initial, distractor)[0]
    positions = [pose(p, target) for p in terminal]
    translation = max(float(np.linalg.norm(p-positions[0][0])) for p, _, _ in positions)
    rotation = max(float(np.degrees((positions[0][1].inv()*q).magnitude())) for _, q, _ in positions)
    distractor_motion = max(float(np.linalg.norm(pose(p, distractor)[0]-dist_start)) for p in trace+terminal)
    checks = {
        'initial_outside': not contained(initial, spec),
        'lift_and_grasp': any(grasped and pos[2]-start[2] >= th['lift_m']
                              for pos, _, grasped in [pose(p, target) for p in trace]),
        'translation': float(np.linalg.norm(positions[-1][0]-start)) >= th['translation_m'],
        'distractor_stable': distractor_motion <= th['distractor_motion_m'],
        'terminal_window': True,
        'terminal_not_grasped': all(not grasped for _, _, grasped in positions),
        'terminal_tcp_clear': all(float(np.linalg.norm(vector(p['robot']['tcp_pos'], 3, 'TCP')-pos)) >= th['tcp_clearance_m']
                                  for p, (pos, _, _) in zip(terminal, positions)),
        'stability_translation': translation <= th['stability_m'],
        'stability_rotation': rotation <= th['stability_rotation_deg'],
        'containment': all(contained(p, spec) for p in terminal),
    }
    final_other_ok = checks['distractor_stable']
    if 'auxiliary_pick_return' in spec:
        # A demonstrated pick-and-return is not an immobile distractor. This
        # evidence is operator-only and never controls the planner or actions.
        aux = spec['auxiliary_pick_return']
        def returned(sample):
            pos, rot, grasped = pose(sample, distractor)
            extent_z = float(np.abs(rot.as_matrix()[2]) @ aux['half_size_m'])
            return (not grasped and
                np.all(pos[:2] >= aux['return_xy']['min']) and
                np.all(pos[:2] <= aux['return_xy']['max']) and
                abs(pos[2]-extent_z-aux['surface_z_m']) <= th['stability_m'])
        lift_indices = [i for i, sample in enumerate(trace)
            if pose(sample, distractor)[2] and
               pose(sample, distractor)[0][2]-dist_start[2] >= th['lift_m']]
        target_lifts = [i for i, sample in enumerate(trace)
            if pose(sample, target)[2] and pose(sample, target)[0][2]-start[2] >= th['lift_m']]
        first_target_lift = min(target_lifts, default=-1)
        ordered = any(lift < ret < first_target_lift for lift in lift_indices
            for ret, sample in enumerate(trace) if returned(sample))
        other_poses = [pose(sample, distractor) for sample in terminal]
        final_other_ok = all(returned(sample) for sample in terminal)
        checks.pop('distractor_stable')
        checks.update(auxiliary_lift=bool(lift_indices), auxiliary_return_before_target_lift=ordered,
            auxiliary_terminal_on_region=final_other_ok,
            auxiliary_terminal_stable=(
                max(float(np.linalg.norm(pos-other_poses[0][0])) for pos, _, _ in other_poses) <= th['stability_m'] and
                max(float(np.degrees((other_poses[0][1].inv()*rot).magnitude())) for _, rot, _ in other_poses) <= th['stability_rotation_deg']))
    return {'checks': checks, 'success': all(checks.values()),
            'goal_satisfied': checks['containment'] and final_other_ok,
            'reasons': [k for k, holds in checks.items() if not holds],
            'measured': {'terminal_translation_m': translation, 'terminal_rotation_deg': rotation,
                         'distractor_motion_m': distractor_motion, 'terminal_span_s': span}}
