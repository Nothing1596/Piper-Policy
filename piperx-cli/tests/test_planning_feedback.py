"""Post-IK scheduling recovery must not admit stale or faulty feedback."""
import threading
import time
import pytest
from piperx_middleware.backends import SimBackend
from piperx_middleware.models import Settings,MoveLinear,JointMove,DomainError
from piperx_middleware.service import RobotService

class DelayedPhysics(SimBackend):
    name='mujoco'
    stalled=False
    fault=False
    def snapshot(self):
        state=super().snapshot()
        if self.stalled:
            state.feedback_stamp_s=time.monotonic()-1
            state.feedback_age_s=1.
        if self.fault:state.error_code=1
        return state

@pytest.mark.parametrize('recover',[True,False])
def test_post_planning_wait_requires_real_new_sample(tmp_path,monkeypatch,recover):
    b=DelayedPhysics();s=RobotService(b,Settings(backend='mujoco',data_dir=tmp_path));s.connect()
    worker=None
    def solve(command,q,settings):
        nonlocal worker
        b.stalled=True
        if recover:
            worker=threading.Timer(.025,lambda:setattr(b,'stalled',False));worker.start()
        return JointMove(joints_deg=q,speed_percent=5,timeout_s=10),{}
    monkeypatch.setattr('piperx_middleware.primitives.resolve_primitive',solve)
    try:
        cmd=MoveLinear(xyz_m=[.3,0,.2])
        if recover:assert s.preview_primitive(cmd)['plan_id']
        else:
            with pytest.raises(DomainError,match='no fresh feedback'):s.preview_primitive(cmd)
        assert not b.commands and not s.active
    finally:
        if worker:worker.join()
        s.close()

def test_initial_stale_sample_does_not_enter_solver(tmp_path,monkeypatch):
    b=DelayedPhysics();s=RobotService(b,Settings(backend='mujoco',data_dir=tmp_path));s.connect();b.stalled=True
    monkeypatch.setattr('piperx_middleware.primitives.resolve_primitive',lambda *a:pytest.fail('must not solve'))
    try:
        with pytest.raises(DomainError) as e:s.preview_primitive(MoveLinear(xyz_m=[.3,0,.2]))
        assert e.value.code=='stale_feedback'
        assert not b.commands
    finally:s.close()

@pytest.mark.parametrize('backend_name',['mujoco','agx'])
def test_fault_and_real_backend_are_not_recovered(tmp_path,monkeypatch,backend_name):
    b=DelayedPhysics();b.name=backend_name
    s=RobotService(b,Settings(backend=backend_name,data_dir=tmp_path));s.connect()
    def solve(command,q,settings):
        if backend_name=='mujoco':b.fault=True
        else:b.stalled=True
        return JointMove(joints_deg=q,speed_percent=5,timeout_s=10),{}
    monkeypatch.setattr('piperx_middleware.primitives.resolve_primitive',solve)
    try:
        with pytest.raises(DomainError):s.preview_primitive(MoveLinear(xyz_m=[.3,0,.2]))
        assert not b.commands and not s.active
    finally:s.close()
