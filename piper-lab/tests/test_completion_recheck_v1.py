import time
import numpy as np
from piperlab.policy.verify import recheck_latest_effects


def observation(red_x=24):
    rgb = np.zeros((64, 64, 3), dtype=np.uint8)
    rgb[28:36, red_x-4:red_x+4] = [220, 0, 0]
    rgb[28:36, 36:44] = [0, 0, 220]
    transform = np.eye(4)
    transform[2, 3] = -.48
    return {'rgb': rgb, 'depth': np.full((64, 64), .5),
            'source_stamp_s': time.monotonic(), 'epoch': 3,
            'camera_info': {'intrinsics': [[100, 0, 32], [0, 100, 32], [0, 0, 1]],
                            'base_from_camera': transform, 'calibration_id': 'fixture'}}


def effect(label, x, stage):
    target = {'xyz_m': [x, 0, .02], 'pixel_xy': [0, 0],
              'calibration_id': 'fixture', 'source_stamp_s': 1., 'epoch': 3, 'valid_until': 2.}
    return {'decision': {'action': 'pick_place', 'object_label': label, 'stage_id': stage},
            'execution': {'status': 'executed_pending_visual_verification'},
            'grounding': [target, target], 'verification': {'verdict': 'supported'}}


def test_old_success_does_not_hide_displaced_object_at_completion():
    history = [effect('red_block', -.04, 'red'), effect('blue_block', .04, 'blue')]
    assert recheck_latest_effects(observation(), history)['verdict'] == 'supported'
    result = recheck_latest_effects(observation(red_x=54), history)
    assert result['verdict'] == 'refuted'
    assert [c['verdict'] for c in result['checks']] == ['refuted', 'supported']
    assert history[0]['verification']['verdict'] == 'supported'  # Preserve historical evidence.


def test_last_target_supersedes_previous_target_and_epoch_change_fails_closed():
    history = [effect('red_block', -.2, 'old'), effect('red cube', -.04, 'new')]
    result = recheck_latest_effects(observation(), history)
    assert result['verdict'] == 'supported'
    assert [c['stage_id'] for c in result['checks']] == ['new']
    changed = observation()
    changed['epoch'] = 4
    assert recheck_latest_effects(changed, history)['verdict'] == 'unknown'
    assert recheck_latest_effects(observation(), [])['verdict'] == 'unknown'
