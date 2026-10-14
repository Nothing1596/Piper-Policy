"""Sensor contract without opening a graphics context; native proof is separate."""
import threading
from types import SimpleNamespace

import numpy as np
import pytest

import piperx_middleware.mujoco_backend as module


@pytest.mark.parametrize('depth_failure',[False,True])
def test_sensor_removes_debug_sites_and_restores_renderer_mode(monkeypatch,depth_failure):
    options=SimpleNamespace(sitegroup=np.ones(6,dtype=int),geomgroup=np.ones(6,dtype=int))
    monkeypatch.setattr(module,'mujoco',SimpleNamespace(MjvOption=lambda:options))
    class Renderer:
        depth=False
        def update_scene(self,data,*,camera,scene_option):
            assert camera=='overview_camera' and not scene_option.sitegroup.any()
            assert scene_option.geomgroup.all()  # Physical/rendered geometry stays visible.
        def render(self):
            if self.depth and depth_failure:raise RuntimeError('test depth failure')
            return np.ones((480,640),np.float32) if self.depth else np.zeros((480,640,3),np.uint8)
        def enable_depth_rendering(self):self.depth=True
        def disable_depth_rendering(self):self.depth=False
    renderer=Renderer();backend=module.MujocoBackend.__new__(module.MujocoBackend)
    backend._thread_local=SimpleNamespace(renderer=renderer,renderer_dims=(640,480))
    backend.lock=threading.RLock();backend.render_lock=threading.RLock()
    backend.model=SimpleNamespace(cam_fovy=[55.]);backend.camera_id=0;backend.epoch=1
    backend.data=SimpleNamespace(time=5.,cam_xpos=np.array([[.3,0,.9]]),cam_xmat=np.eye(3).reshape(1,9))
    backend.snapshot=lambda:SimpleNamespace(public=lambda:{'q_deg':[0.]*6})
    if depth_failure:
        with pytest.raises(RuntimeError,match='depth failure'):backend.observe()
    else:
        result=backend.observe()
        assert result['rgb'].shape==(480,640,3) and result['depth'].shape==(480,640)
        assert result['sim_time_s']==5. and result['robot_state']['q_deg']==[0.]*6
        assert (result['depth']==1.).all()  # no topology in this fake: depth stays unfiltered
        assert result['depth_filter']['applied'] is False
        assert result['depth_filter']['reason']=='robot_topology_unavailable'
        assert result['depth_filter']['filtered_pixels']==0
    assert not renderer.depth
