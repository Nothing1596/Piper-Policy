import numpy as np
import pytest
from piperlab.policy.grounding import ground_pixel, GroundingError, scene_unchanged


def calibration():
    return {"intrinsics":[[100,0,2],[0,100,2],[0,0,1]],
            "base_from_camera":np.eye(4).tolist(),"calibration_id":"synthetic-test"}


def test_metric_grounding_and_timestamp():
    point=ground_pixel([2,2],np.ones((5,5)),calibration(),source_stamp_s=10,epoch=2,now=10.1)
    assert point.xyz_m==(0,0,1) and point.epoch==2
    with pytest.raises(GroundingError,match='stale'):
        ground_pixel([2,2],np.ones((5,5)),calibration(),source_stamp_s=10,epoch=2,now=11)


def test_unknown_depth_and_uncalibrated_rejected():
    with pytest.raises(GroundingError,match='calibration'):
        ground_pixel([2,2],np.ones((5,5)),{},source_stamp_s=10,epoch=0,now=10)
    with pytest.raises(GroundingError,match='depth'):
        ground_pixel([2,2],np.zeros((5,5)),calibration(),source_stamp_s=10,epoch=0,now=10)


def test_small_object_change_not_diluted():
    a=np.zeros((100,100,3),np.uint8);b=a.copy();b[30:40,30:40]=255
    assert not scene_unchanged(a,b)
    assert scene_unchanged(a,a)
