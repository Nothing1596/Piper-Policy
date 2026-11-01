import pytest
import numpy as np
from rvicl_ig.watchact_score import score_task

def make_payload(sim_time, target_pos, target_q, target_grasped, distractor_pos, tcp_pos):
    return {
        'sim_time_s': sim_time,
        'objects': {
            'cube_red': {'position': target_pos, 'quaternion': target_q, 'grasped': target_grasped, 'in_tray': False},
            'cube_blue': {'position': distractor_pos, 'quaternion': [1, 0, 0, 0], 'grasped': False, 'in_tray': False}
        },
        'robot': {'tcp_pos': tcp_pos, 'gripper_width_m': 0.0}
    }

def make_spec():
    return {
        'scope': 'watchact_imitation_piperx_v1',
        'target': 'cube_red',
        'distractor': 'cube_blue',
        'target_shape': {'type': 'cylinder', 'radius_m': 0.02, 'half_height_m': 0.03},
        'container_bounds': {'min': [0.3, 0.3, 0.8], 'max': [0.5, 0.5, 1.0]},
        'thresholds': {
            'lift_m': 0.02,
            'translation_m': 0.05,
            'distractor_motion_m': 0.015,
            'stability_m': 0.003,
            'stability_rotation_deg': 3.0,
            'tcp_clearance_m': 0.10,
            'terminal_window_s': 1.0
        }
    }

def test_valid_success():
    spec = make_spec()
    initial = make_payload(0.0, [0.2, 0.4, 0.7], [1, 0, 0, 0], False, [0.4, 0.3, 0.7], [0.15, 0.4, 0.9])
    trace = [
        make_payload(0.5, [0.25, 0.4, 0.75], [1, 0, 0, 0], True, [0.4, 0.3, 0.7], [0.25, 0.4, 0.95]),
        make_payload(1.0, [0.4, 0.4, 0.85], [1, 0, 0, 0], True, [0.4, 0.3, 0.7], [0.4, 0.4, 1.05])
    ]
    terminal = [
        make_payload(1.5, [0.4, 0.4, 0.88], [1, 0, 0, 0], False, [0.4, 0.3, 0.7], [0.3, 0.35, 0.75]),
        make_payload(2.5, [0.4, 0.4, 0.88], [1, 0, 0, 0], False, [0.4, 0.3, 0.7], [0.3, 0.35, 0.75])
    ]
    result = score_task(initial, trace, terminal, spec)
    assert result['success'] is True
    assert result['goal_satisfied'] is True

def test_centroid_inside_extent_outside():
    spec = make_spec()
    initial = make_payload(0.0, [0.2, 0.4, 0.7], [1, 0, 0, 0], False, [0.4, 0.3, 0.7], [0.15, 0.4, 0.9])
    trace = [
        make_payload(0.5, [0.495, 0.4, 0.75], [1, 0, 0, 0], True, [0.4, 0.3, 0.7], [0.495, 0.4, 0.95]),
        make_payload(1.0, [0.495, 0.4, 0.85], [1, 0, 0, 0], True, [0.4, 0.3, 0.7], [0.495, 0.4, 1.05])
    ]
    terminal = [
        make_payload(1.5, [0.495, 0.4, 0.88], [1, 0, 0, 0], False, [0.4, 0.3, 0.7], [0.3, 0.35, 0.75]),
        make_payload(2.5, [0.495, 0.4, 0.88], [1, 0, 0, 0], False, [0.4, 0.3, 0.7], [0.3, 0.35, 0.75])
    ]
    result = score_task(initial, trace, terminal, spec)
    assert result['checks']['containment'] is False
    assert result['success'] is False

def test_no_lift():
    spec = make_spec()
    initial = make_payload(0.0, [0.2, 0.4, 0.7], [1, 0, 0, 0], False, [0.4, 0.3, 0.7], [0.15, 0.4, 0.9])
    trace = [
        make_payload(0.5, [0.2, 0.4, 0.71], [1, 0, 0, 0], False, [0.4, 0.3, 0.7], [0.2, 0.4, 0.9]),
    ]
    terminal = [make_payload(1.5, [0.2, 0.4, 0.71], [1, 0, 0, 0], False, [0.4, 0.3, 0.7], [0.15, 0.4, 0.75])]
    terminal.append({**terminal[0], 'sim_time_s': 2.5})
    result = score_task(initial, trace, terminal, spec)
    assert result['checks']['lift_and_grasp'] is False
    assert result['success'] is False

def test_distractor_moved():
    spec = make_spec()
    initial = make_payload(0.0, [0.2, 0.4, 0.7], [1, 0, 0, 0], False, [0.4, 0.3, 0.7], [0.15, 0.4, 0.9])
    trace = [
        make_payload(0.5, [0.25, 0.4, 0.75], [1, 0, 0, 0], True, [0.4, 0.32, 0.7], [0.25, 0.4, 0.95]),
        make_payload(1.0, [0.4, 0.4, 0.85], [1, 0, 0, 0], True, [0.4, 0.3, 0.7], [0.4, 0.4, 1.05])
    ]
    terminal = [
        make_payload(1.5, [0.4, 0.4, 0.88], [1, 0, 0, 0], False, [0.4, 0.32, 0.7], [0.3, 0.35, 0.75]),
        make_payload(2.5, [0.4, 0.4, 0.88], [1, 0, 0, 0], False, [0.4, 0.3, 0.7], [0.3, 0.35, 0.75])
    ]
    result = score_task(initial, trace, terminal, spec)
    assert result['checks']['distractor_stable'] is False
    assert result['success'] is False

def test_initial_already_inside():
    spec = make_spec()
    initial = make_payload(0.0, [0.4, 0.4, 0.88], [1, 0, 0, 0], False, [0.4, 0.3, 0.7], [0.15, 0.4, 0.9])
    trace = [make_payload(0.5, [0.4, 0.4, 0.88], [1, 0, 0, 0], True, [0.4, 0.3, 0.7], [0.4, 0.4, 0.95])]
    terminal = [make_payload(1.5, [0.4, 0.4, 0.88], [1, 0, 0, 0], False, [0.4, 0.3, 0.7], [0.3, 0.35, 0.75])]
    terminal.append({**terminal[0], 'sim_time_s': 2.5})
    result = score_task(initial, trace, terminal, spec)
    assert result['checks']['initial_outside'] is False

def test_still_held():
    spec = make_spec()
    initial = make_payload(0.0, [0.2, 0.4, 0.7], [1, 0, 0, 0], False, [0.4, 0.3, 0.7], [0.15, 0.4, 0.9])
    trace = [
        make_payload(0.5, [0.25, 0.4, 0.75], [1, 0, 0, 0], True, [0.4, 0.3, 0.7], [0.25, 0.4, 0.95]),
        make_payload(1.0, [0.4, 0.4, 0.85], [1, 0, 0, 0], True, [0.4, 0.3, 0.7], [0.4, 0.4, 1.05])
    ]
    terminal = [
        make_payload(1.5, [0.4, 0.4, 0.88], [1, 0, 0, 0], True, [0.4, 0.3, 0.7], [0.4, 0.4, 0.85]),
        make_payload(2.5, [0.4, 0.4, 0.88], [1, 0, 0, 0], True, [0.4, 0.3, 0.7], [0.4, 0.4, 0.85])
    ]
    result = score_task(initial, trace, terminal, spec)
    assert result['checks']['terminal_not_grasped'] is False
    assert result['success'] is False

def test_tcp_not_clear():
    spec = make_spec()
    initial = make_payload(0.0, [0.2, 0.4, 0.7], [1, 0, 0, 0], False, [0.4, 0.3, 0.7], [0.15, 0.4, 0.9])
    trace = [
        make_payload(0.5, [0.25, 0.4, 0.75], [1, 0, 0, 0], True, [0.4, 0.3, 0.7], [0.25, 0.4, 0.95]),
        make_payload(1.0, [0.4, 0.4, 0.85], [1, 0, 0, 0], True, [0.4, 0.3, 0.7], [0.4, 0.4, 1.05])
    ]
    terminal = [
        make_payload(1.5, [0.4, 0.4, 0.88], [1, 0, 0, 0], False, [0.4, 0.3, 0.7], [0.40, 0.4, 0.88]),
        make_payload(2.5, [0.4, 0.4, 0.88], [1, 0, 0, 0], False, [0.4, 0.3, 0.7], [0.40, 0.4, 0.88])
    ]
    result = score_task(initial, trace, terminal, spec)
    assert result['checks']['terminal_tcp_clear'] is False
    assert result['success'] is False

def test_short_terminal_window():
    spec = make_spec()
    initial = make_payload(0.0, [0.2, 0.4, 0.7], [1, 0, 0, 0], False, [0.4, 0.3, 0.7], [0.15, 0.4, 0.9])
    trace = [
        make_payload(0.5, [0.25, 0.4, 0.75], [1, 0, 0, 0], True, [0.4, 0.3, 0.7], [0.25, 0.4, 0.95]),
        make_payload(1.0, [0.4, 0.4, 0.85], [1, 0, 0, 0], True, [0.4, 0.3, 0.7], [0.4, 0.4, 1.05])
    ]
    terminal = [make_payload(1.5, [0.4, 0.4, 0.88], [1, 0, 0, 0], False, [0.4, 0.3, 0.7], [0.3, 0.35, 0.75])]
    with pytest.raises(ValueError, match='terminal'):
        score_task(initial, trace, terminal, spec)

def test_invalid_missing_threshold():
    spec = make_spec()
    del spec['thresholds']['lift_m']
    initial = make_payload(0.0, [0.2, 0.4, 0.7], [1, 0, 0, 0], False, [0.4, 0.3, 0.7], [0.15, 0.4, 0.9])
    with pytest.raises(ValueError, match="Missing thresholds"):
        score_task(initial, [], [], spec)

def test_invalid_nonfinite_threshold():
    spec = make_spec()
    spec['thresholds']['lift_m'] = float('nan')
    initial = make_payload(0.0, [0.2, 0.4, 0.7], [1, 0, 0, 0], False, [0.4, 0.3, 0.7], [0.15, 0.4, 0.9])
    with pytest.raises(ValueError, match="Invalid threshold"):
        score_task(initial, [], [], spec)

def test_invalid_quaternion():
    spec = make_spec()
    initial = make_payload(0.0, [0.2, 0.4, 0.7], [2, 0, 0, 0], False, [0.4, 0.3, 0.7], [0.15, 0.4, 0.9])
    trace = [make_payload(0.5, [0.25, 0.4, 0.75], [1, 0, 0, 0], True, [0.4, 0.3, 0.7], [0.25, 0.4, 0.95])]
    with pytest.raises(ValueError, match="Invalid quaternion"):
        score_task(initial, trace, [make_payload(t, [0.4,0.4,0.88], [1,0,0,0], False, [0.4,0.3,0.7], [0.1,0.1,1.2]) for t in [1.5,2.5]], spec)

def test_insufficient_trace():
    spec = make_spec()
    initial = make_payload(0.0, [0.2, 0.4, 0.7], [1, 0, 0, 0], False, [0.4, 0.3, 0.7], [0.15, 0.4, 0.9])
    with pytest.raises(ValueError, match="Insufficient trace"):
        score_task(initial, [], [], spec)


def test_rotated_cylinder_extents_and_nonfinite_evidence():
    from rvicl_ig.watchact_score import contained
    spec=make_spec()
    # Tilted cylinder: full support extent grows along X and crosses the wall.
    spec['target_shape']={'type':'cylinder','radius_m':.01,'half_height_m':.08}
    sample=make_payload(0.,[.45,.4,.9],[2**-.5,0,2**-.5,0],False,[0,0,0],[0,0,1])
    assert not contained(sample,spec)
    sample['objects']['cube_red']['quaternion']=[1,0,0,0]
    assert contained(sample,spec)
    sample['objects']['cube_red']['position']=[float('nan'),0,0]
    with pytest.raises(ValueError):contained(sample,spec)


def auxiliary_case():
    from copy import deepcopy
    spec = make_spec()
    spec['auxiliary_pick_return'] = {'body': 'cube_blue', 'half_size_m': [.02, .02, .03],
        'return_xy': {'min': [.39, .29], 'max': [.41, .31]}, 'surface_z_m': .67}
    initial = make_payload(0., [.2,.4,.7], [1,0,0,0], False, [.4,.3,.7], [.1,.1,1.2])
    a = deepcopy(initial); a['sim_time_s'] = 1.
    a['objects']['cube_blue'].update(position=[.4,.3,.75],grasped=True)
    b = deepcopy(initial); b['sim_time_s'] = 2.
    c = deepcopy(initial); c['sim_time_s'] = 3.
    c['objects']['cube_red'].update(position=[.3,.4,.85],grasped=True)
    trace = [a,b,c]
    terminal = [make_payload(t,[.4,.4,.88],[1,0,0,0],False,[.4,.3,.7],[.1,.1,1.2]) for t in [4.,5.]]
    return initial,trace,terminal,spec


def test_auxiliary_pick_return_counts_motion_as_required_not_distractor_failure():
    args=auxiliary_case();result=score_task(*args)
    assert result['success'] and result['goal_satisfied']
    assert result['checks']['auxiliary_return_before_target_lift']
    assert 'distractor_stable' not in result['checks']


@pytest.mark.parametrize('error',['omitted','no_contact','target_first','not_returned','wrong_height','terminal_drift','terminal_rotated','terminal_grasped'])
def test_auxiliary_final_goal_does_not_hide_missing_or_reordered_actions(error):
    initial,trace,terminal,spec=auxiliary_case()
    if error=='omitted':trace[0]['objects']['cube_blue'].update(position=[.4,.3,.7],grasped=False)
    if error=='no_contact':trace[0]['objects']['cube_blue']['grasped']=False
    if error=='target_first':trace[0]['objects']['cube_red'].update(position=[.3,.4,.85],grasped=True)
    if error=='not_returned':trace[1]['objects']['cube_blue']['grasped']=True
    if error=='wrong_height':trace[1]['objects']['cube_blue']['position'][2]=.74
    if error=='terminal_drift':terminal[-1]['objects']['cube_blue']['position'][0]+=.008
    if error=='terminal_rotated':
        from scipy.spatial.transform import Rotation
        terminal[-1]['objects']['cube_blue']['quaternion']=Rotation.from_euler('z',10,degrees=True).as_quat(scalar_first=True).tolist()
    if error=='terminal_grasped':terminal[-1]['objects']['cube_blue']['grasped']=True
    result=score_task(initial,trace,terminal,spec)
    assert result['success'] is False
    if error in ['omitted','no_contact','target_first','not_returned','wrong_height']:
        assert result['checks']['auxiliary_return_before_target_lift'] is False
        assert result['goal_satisfied'] is True
    elif error=='terminal_grasped':assert result['checks']['auxiliary_terminal_on_region'] is False
    else:assert result['checks']['auxiliary_terminal_stable'] is False


@pytest.mark.parametrize('field,value',[('body','unknown'),('half_size_m',[.02,-.01,.03]),('surface_z_m',float('nan'))])
def test_invalid_auxiliary_spec_rejected(field,value):
    initial,trace,terminal,spec=auxiliary_case();spec['auxiliary_pick_return'][field]=value
    with pytest.raises(ValueError):score_task(initial,trace,terminal,spec)
