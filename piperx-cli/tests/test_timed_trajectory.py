"""Safety and sample fidelity of the simulation-only timeline transport."""
import time
import pytest
from pydantic import ValidationError
from piperx_middleware.backends import SimBackend
from piperx_middleware.models import Settings, DomainError, ExecuteRequest, JointMove, RuntimeParameters
from piperx_middleware.service import RobotService
from piperx_middleware.timed_trajectory import TimedTrajectory
from piperx_middleware.request_context import control_session_id


class FakeMujoco(SimBackend):
    name = 'mujoco'
    def begin_timed_trajectory(self,current_deg):
        self.commands.append(('timed_begin',list(current_deg)))


@pytest.fixture
def service(tmp_path):
    s = RobotService(FakeMujoco(),Settings(backend='mujoco',data_dir=tmp_path))
    s.connect()
    yield s
    s.close()


def timeline(s, **updates):
    raw = s.state()
    return TimedTrajectory(**({'instance_id':raw['instance_id'],'connection_epoch':raw['connection_epoch'],
        'parameter_version':raw['parameter_version'],'start_joints_deg':raw['robot']['q_deg'],
        'start_gripper_m':raw['robot']['gripper_width_m'],
        'joints_deg':[[.1,0.,0.,0.,0.,0.],[.2,0.,0.,0.,0.,0.],[.1,0.,0.,0.,0.,0.]],
        'relative_times_s':[.05,.1,.15],'settle_samples':3,'control_hz':100.,'note':'immutable path'}|updates))


def wait(s,job):
    deadline=time.monotonic()+5
    while job['status'] in ('accepted','running'):
        assert time.monotonic()<deadline
        time.sleep(.01)
        job=s.get_job(job['job_id'])
    return job


def test_every_sample_including_nonendpoint_excursion(service):
    t=timeline(service)
    job=wait(service,service.timed_trajectory(t,'timed-request-001'))
    assert job['status']=='succeeded',job
    assert [c[1] for c in service.backend.commands if c[0]=='joint']==t.joints_deg
    actual=job['execution_timing']['sent_relative_times_s']
    assert len(actual)==len(t.relative_times_s)
    assert all(0 <= a-b <= .1 for a,b in zip(actual,t.relative_times_s))
    count=len(service.backend.commands)
    assert service.timed_trajectory(t,'timed-request-001')['job_id']==job['job_id']
    assert len(service.backend.commands)==count
    with pytest.raises(DomainError) as exc:
        service.move(JointMove(**job['command']),'timed-request-001')
    assert exc.value.code=='idempotency_conflict'


def test_hardware_and_kinematic_backends_rejected(tmp_path):
    s=RobotService(SimBackend(),Settings(data_dir=tmp_path));s.connect()
    try:
        with pytest.raises(DomainError) as exc:s.timed_trajectory(timeline(s),'timed-no-can-001')
        assert exc.value.code=='simulation_only'
        assert not s.backend.commands
    finally:s.close()


@pytest.mark.parametrize('updates',[{'relative_times_s':[.1,.1,.2]}, {'relative_times_s':[0.,.1,.2]},
    {'relative_times_s':[.1,.2]}, {'joints_deg':[[0.]*5]*3}, {'joints_deg':[[float('nan')]*6]*3}])
def test_strict_timeline(service,updates):
    with pytest.raises(ValidationError):timeline(service,**updates)


def test_invalid_intermediate_sample_rejected(service):
    with pytest.raises(DomainError) as exc:service.timed_trajectory(timeline(service,joints_deg=[[0.]*6,[151.,0.,0.,0.,0.,0.],[0.]*6]),'bad-path-001')
    assert exc.value.code=='joint_limits'
    assert not service.backend.commands


def test_preview_start_and_parameter_identity(service):
    t=timeline(service)
    p=service.preview_timed_trajectory(t)
    service.configure_runtime(RuntimeParameters(tcp_offset_m=[0.,0.,.01]))
    with pytest.raises(DomainError):service.execute(ExecuteRequest(plan_id=p['plan_id'],request_id='old-path-001'))
    with pytest.raises(DomainError):service.preview_timed_trajectory(t)
    assert not service.backend.commands


def test_stop_is_cancellable_and_never_replayed(service):
    t=timeline(service,relative_times_s=[.05,1.,2.])
    job=service.timed_trajectory(t,'stop-path-001')
    time.sleep(.15)
    service.stop()
    done=wait(service,job)
    assert done['status']=='cancelled',done
    assert done['execution_timing']['samples_sent']==1
    count=len(service.backend.commands)
    assert service.timed_trajectory(t,'stop-path-001')['status']=='cancelled'
    assert len(service.backend.commands)==count


def test_session_required_when_managed(tmp_path):
    s=RobotService(FakeMujoco(),Settings(backend='mujoco',managed_control=True,data_dir=tmp_path));s.connect()
    try:
        with pytest.raises(DomainError):s.preview_timed_trajectory(timeline(s))
        session=s.acquire_session('test')
        token=control_session_id.set(session['session_id'])
        try:assert wait(s,s.timed_trajectory(timeline(s),'managed-path-001'))['status']=='succeeded'
        finally:control_session_id.reset(token)
    finally:s.close()


def test_policy_hard_limits_apply_to_interior(service):
    service.policy.limits.joint_upper_deg=[.15,180.,0.,89.,89.,180.]
    with pytest.raises(DomainError) as exc:service.preview_timed_trajectory(timeline(service))
    assert exc.value.code=='site_joint_limits'
    assert not service.backend.commands


def test_tracking_and_lateness_fail_closed(service):
    service.backend.follow=False
    t=timeline(service,joints_deg=[[8.,0.,0.,0.,0.,0.]]*3,relative_times_s=[.2,.4,.6])
    done=wait(service,service.timed_trajectory(t,'tracking-path-001'))
    assert done['status']=='outcome_unknown'
    assert done['error_code']=='tracking_error'


def test_forged_fast_timeline_rejected(service):
    with pytest.raises(DomainError) as exc:
        service.preview_timed_trajectory(timeline(service,joints_deg=[[100.,0.,0.,0.,0.,0.]]*3))
    assert exc.value.code=='trajectory_velocity'
    assert not service.backend.commands


def test_chunked_body_limit(service):
    from fastapi.testclient import TestClient
    from piperx_middleware.http_api import create_app
    client=TestClient(create_app(service,'m'*40,'o'*40))
    response=client.post('/v1/simulation/trajectory',
        headers={'Authorization':'Bearer '+'m'*40},content=iter([b' '*600000,b' '*600000]))
    assert response.status_code==413
    assert not service.backend.commands


def test_expired_session_finishes_current_but_admits_no_new_work(tmp_path):
    s=RobotService(FakeMujoco(),Settings(backend='mujoco',managed_control=True,data_dir=tmp_path));s.connect()
    try:
        session=s.acquire_session('test')
        token=control_session_id.set(session['session_id'])
        try:
            t=timeline(s,relative_times_s=[.1,.2,.3])
            job=s.timed_trajectory(t,'expiring-path-001')
            s.sessions.ttl_s=.01
            time.sleep(.03);s._expire_control()
            assert wait(s,job)['status']=='succeeded'
            with pytest.raises(DomainError):s.preview_timed_trajectory(timeline(s))
        finally:control_session_id.reset(token)
    finally:s.close()
