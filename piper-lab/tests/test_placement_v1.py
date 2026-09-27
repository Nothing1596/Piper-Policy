import numpy as np
from piperlab.policy.scene import free_region_pixels,footprint_is_free
from piperlab.policy.grounding import GroundedPoint


def test_metric_footprint_rejects_occupied_and_outside_region():
    rgb=np.zeros((200,200,3),np.uint8)
    rgb[10:190,10:190]=[40,145,80]
    rgb[88:113,88:113]=[205,35,35]
    region={'color':'green','bbox_xyxy':[10,10,190,190],'pixel_xy':[100,100]}
    points=free_region_pixels(rgb,region)
    assert points and all(not (88<=x<113 and 88<=y<113) for x,y in points)
    camera={'intrinsics':[[100,0,100],[0,100,100],[0,0,1]],'base_from_camera':np.eye(4).tolist()}
    point=lambda x,y:GroundedPoint((x,y,1),(0,0),'test',0,0,1)
    assert not footprint_is_free(rgb,region,point(0,0),camera,half_size=.15)
    assert not footprint_is_free(rgb,region,point(.79,.79),camera,half_size=.15)
    assert footprint_is_free(rgb,region,point(-.5,-.5),camera,half_size=.15)


def test_push_candidates_use_measured_depth_and_reject_obstacle():
    from piperlab.policy.scene import measured_candidates
    rgb=np.zeros((200,200,3),np.uint8);rgb[95:106,95:106]=[205,35,35]
    depth=np.ones((200,200));depth[95:106,95:106]=.97
    transform=np.diag([1.,-1.,-1.,1.]);transform[2,3]=1
    camera={'intrinsics':[[200,0,100],[0,200,100],[0,0,1]],
            'base_from_camera':transform.tolist(),'calibration_id':'test'}
    obs={'rgb':rgb,'depth':depth,'camera_info':camera,'source_stamp_s':0.,'epoch':0}
    proposals=measured_candidates(obs)
    right=next(p for p in proposals if p['id']=='red-0-push-right')
    assert np.allclose(right['pixel_xy'],[116,100])
    depth[98:103,114:119]=.8
    assert not any(p['id']=='red-0-push-right' for p in measured_candidates(obs))
