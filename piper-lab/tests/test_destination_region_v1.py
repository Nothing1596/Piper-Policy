import time
import numpy as np
import pytest
from piperlab.policy.grounding import ground_pixel
from piperlab.policy.scene import color_candidates
from piperlab.policy.verify import verify_block_destination
from piperlab.policy.planner import VisionPlanner, DECISION_SCHEMA


def test_grid_covers_regions_missed_by_ranked_circle_proposals_and_is_bounded():
    from piperlab.policy.scene import free_region_pixels
    obs, _, region = scene()
    ranked=free_region_pixels(obs['rgb'],region,limit=1)
    fallback=free_region_pixels(obs['rgb'],region,limit=1,include_grid=True)
    assert fallback[:1]==ranked
    assert [32,32] in fallback and [32,32] not in ranked
    assert [64,64] not in fallback  # Occupied by red block.
    assert len(fallback)<1100
    assert all(obs['rgb'][y,x,1]>0 for x,y in fallback)


def scene(x=64):
    rgb = np.zeros((128, 128, 3), dtype=np.uint8)
    rgb[28:100, 28:100] = [0, 180, 0]
    rgb[60:68, x-4:x+4] = [220, 0, 0]
    depth = np.full((128, 128), .5)
    depth[60:68, x-4:x+4] = .47
    transform = np.diag([1., -1., -1., 1.])
    transform[2, 3] = .5
    obs = dict(rgb=rgb, depth=depth, epoch=1, source_stamp_s=time.monotonic(),
        camera_info=dict(intrinsics=[[100, 0, 64], [0, 100, 64], [0, 0, 1]],
                         base_from_camera=transform, calibration_id='fixture'))
    target = ground_pixel([x, 64], depth, obs['camera_info'],
        source_stamp_s=obs['source_stamp_s'], epoch=1, max_age_s=1)
    region = next(c for c in color_candidates(rgb) if c['color']=='green')
    return obs, target, region


@pytest.mark.parametrize('x,verdict', [(64, 'supported'), (112, 'refuted'), (97, 'refuted')])
def test_reaching_commanded_point_does_not_prove_region_goal(x, verdict):
    obs, target, region = scene(x)
    assert verify_block_destination(obs, 'red_block', target)['verdict']=='supported'
    result = verify_block_destination(obs, 'red_block', target, destination_region=region)
    assert result['verdict']==verdict


def test_invisible_region_cannot_confirm_success():
    obs, target, region = scene()
    obs['rgb'][obs['rgb'][:,:,1]>0] = 0
    assert verify_block_destination(obs, 'red_block', target,
        destination_region=region)['verdict']=='unknown'


@pytest.mark.parametrize('destination', ['', 'red-0', 'green-0'])
def test_planner_requires_observed_green_destination(destination):
    class Model:
        def infer(self, prompt, images, schema):
            assert schema['allOf'][0]['then']['properties']['destination_candidate_id']['enum']==['green-0']
            return dict(action='pick_place', stage_id='', evidence_refs=['live'],
                        object_candidate_id='red-0', destination_candidate_id=destination)
    planner = VisionPlanner(Model())
    kwargs = dict(task='put red in green', observation_path='live.jpg', robot_state={},
                  candidates=[dict(id='green-0', color='green'), dict(id='red-0', color='red')])
    if destination=='green-0':
        assert planner.decide(**kwargs)['destination_candidate_id']==destination
    else:
        with pytest.raises(ValueError, match='requires_measured_destination_region'):
            planner.decide(**kwargs)
    assert 'allOf' not in DECISION_SCHEMA
