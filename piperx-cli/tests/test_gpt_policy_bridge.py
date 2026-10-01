"""Optional integration with the separately supplied pinned upstream package."""
from pathlib import Path
import io
import json
import time
import numpy as np
import pytest

pytest.importorskip('gpt_policy',reason='Supply the pinned external GPT-Policy checkout to run bridge integration')
from gpt_policy_piper.robot import PiperRobot, PiperSolver
from gpt_policy_piper.camera import SimulationCameraSet
from gpt_policy_piper.transport import checked, BridgeError
from piperx_middleware.models import Settings, JointMove, GripperMove
from piperx_middleware.mujoco_backend import MujocoBackend
from piperx_middleware.service import RobotService
from piperx_middleware.timed_trajectory import TimedTrajectory
from piperx_middleware.observability import get_parameters_response


class LocalTransport:
    def __init__(self,service):self.service=service;self.calls=[]
    def call(self,method,path,body=None):
        self.calls.append((method,path,body))
        s=self.service
        if path=='/v1/state':return s.state()
        if path=='/v1/parameters':return get_parameters_response(s)
        if path=='/v1/simulation/trajectory':return s.timed_trajectory(TimedTrajectory(**body['trajectory']),body['request_id'])
        if path=='/v1/simulation/trajectory/preview':return s.preview_timed_trajectory(TimedTrajectory(**body))
        if path=='/v1/move':return s.move(GripperMove(**body['command']),body['request_id'])
        if path.startswith('/v1/requests/'):return s.get_request(path.rsplit('/',1)[-1])
        if path.startswith('/v1/jobs/'):return s.get_job(path.rsplit('/',1)[-1])
        if path=='/v1/stop':return s.stop()
        raise AssertionError(path)


def settings():
    return {'backend':'piper_mujoco','runtime':{'right_interface':None,'robot_model':'PiperX',
            'interface':'mujoco','trajectory_hz':30.,'max_decisions':3},'motion':{},
        'calibration':{'link6_from_sdk_eef':np.eye(4).tolist(),'link6_from_tcp':[[1,0,0,0],[0,1,0,0],[0,0,1,.1425],[0,0,0,1]],
                        'link6_from_camera':{},'base_from_camera':{}},'vision':{},
        'piper_simulation':{'url':'http://127.0.0.1:8808','model_token_file':'unused','operator_token_file':'unused'}}


@pytest.fixture
def robot(tmp_path):
    service=RobotService(MujocoBackend(seed=200),Settings(backend='mujoco',data_dir=tmp_path,tcp_offset_m=[0.,0.,.1425]))
    service.connect()
    time.sleep(.2)
    transport=LocalTransport(service)
    robot=PiperRobot(settings=settings(),transport=transport)
    yield robot,service,transport
    robot.close();service.close()


def test_radian_contract_and_real_mujoco_timed_motion(robot):
    r,s,t=robot
    start=r.state()
    target=np.array(start['tcp_xyzrpy'])
    target[2]+=.003
    from gpt_policy.geometry.poses import rpy_to_quaternion
    requested={'pose_xyzquat':[*target[:3].tolist(),*rpy_to_quaternion(target[3:])]}
    plan=r.plan_eef_trajectory([requested],'vertical baseline')
    assert plan['result']['timing']=='ruckig_scalar_path_parameterization'
    assert plan['result']['joint_waypoints']>=3
    assert plan['result']['endpoint_hold_waypoints']>=2
    samples=np.array(plan['result']['_trace']['tcp_samples_xyzrpy'])
    assert np.max(np.abs(samples[:,:2]-target[:2]))<1e-10
    assert np.max(np.abs(samples[:,3:]-target[3:]))<1e-8
    expected=np.rad2deg(plan['joint_positions_rad'])
    job=r.send_eef_trajectory(plan)
    np.testing.assert_allclose(job['timed_trajectory']['joints_deg'],expected,atol=0.,rtol=0.)
    np.testing.assert_allclose(job['execution_timing']['planned_relative_times_s'],plan['relative_times_s'],atol=0.,rtol=0.)
    assert job['execution_timing']['samples_sent']==len(expected)
    assert job['settle']['settled']
    assert not job['physical_validation']
    assert np.max(np.abs(np.array(r.state()['joint_positions_rad'])-plan['joint_positions_rad'][-1]))<.03
    assert s.backend.profile_speed_percent is None
    assert len([call for call in t.calls if call[0]=='POST' and call[1]=='/v1/simulation/trajectory'])==1


def test_check_path_performs_no_actuator_writes(robot):
    r,s,t=robot
    before=s.backend.tx_frames
    target=r.state()['tcp_xyzquat']
    result=r.execute('check_path',{'poses':[{'pose_xyzquat':target}],'note':'hold path'})
    assert result['path_check']['accepted'] and not result['path_check']['executed']
    assert s.backend.tx_frames==before


def test_unknown_submission_is_never_replayed(robot):
    r,s,t=robot
    original=t.call
    submitted=[]
    def failed(method,path,body=None):
        if method=='POST' and path=='/v1/move':
            submitted.append(body)
            raise BridgeError({'error':{'code':'transport_unknown'}})
        if path.startswith('/v1/requests/'):
            raise BridgeError({'error':{'code':'not_found'}})
        return original(method,path,body)
    t.call=failed
    from gpt_policy.hardware.motion_control import MotionFault
    with pytest.raises(MotionFault):r.execute('set_gripper',{'gripper':1.,'note':'open'})
    assert len(submitted)==1
    assert r.motion.fault['reason']=='submission_outcome_unknown'


def test_real_camera_rgb_only_and_calibration(tmp_path):
    from fastapi.testclient import TestClient
    from piperx_middleware.http_api import create_app
    s=RobotService(MujocoBackend(seed=200),Settings(backend='mujoco',data_dir=tmp_path));s.connect()
    app=create_app(s,'m'*40,'o'*40)
    class HttpTransport:
        def __init__(self,client):self.client=client
        def call(self,method,path,body=None):return checked(self.client.request(method,path,json=body).json())
    with TestClient(app,headers={'Authorization':'Bearer '+'m'*40}) as client:
        config=settings()
        raw=client.get('/v1/simulation/rgb').json()
        assert 'depth_npy_b64' not in raw
        assert not any(k in raw for k in ('objects','truth','evaluate','success'))
        cameras=SimulationCameraSet(config,transport=HttpTransport(client))
        before=time.time();image=cameras.capture()['top']
        assert image.captured_at>=before
        assert image.source_clock=='host_monotonic'
        assert len(image.rgb_data)==640*480*3
        assert config['vision']['camera_intrinsics']['top']==raw['camera_info']['intrinsics']
        assert config['calibration']['base_from_camera']['top']['left']==raw['camera_info']['base_from_camera']
        from gpt_policy.vision.perception import PixelLocalizer
        result=PixelLocalizer(config).locate({'camera':'top','pixel_xy':[320.,240.],
            'reference_step':0,'reference_pixel_xy':[330.,240.]},{}, {0:{}})
        assert not result['metric_position_available']
        assert 'triangulation_candidate_base_xyz' not in result


def test_upstream_runtime_fake_agent_records_full_run(robot,tmp_path):
    from gpt_policy.runtime.runner import run_loop
    from gpt_policy.settings import runtime_config
    from gpt_policy.input import resolve_run_input
    from gpt_policy.tools import ToolExecutor,load_tool_catalog
    from gpt_policy.vision.perception import PixelLocalizer
    from gpt_policy.recording.trace import RunRecorder
    from gpt_policy.hardware.camera import CapturedImage
    from gpt_policy.runtime.console import RunConsole
    from PIL import Image
    r,s,t=robot
    config=settings()
    buffer=io.BytesIO();Image.new('RGB',(640,480),(40,50,60)).save(buffer,format='JPEG')
    class Video:
        def snapshot(self,after=None):return {'top':CapturedImage('top',buffer.getvalue(),'image/jpeg',640,480,time.time(),None,time.monotonic(),'host_monotonic')}
    class Cameras:
        def describe(self,images):return [{'name':'top','width':640,'height':480,'captured_at':str(images['top'].captured_at),'depth_available':False}]
    class Agent:
        def __init__(self):self.calls=0
        def decide(self,turn):
            self.calls+=1
            assert set(turn.images)=={'top'}
            if self.calls==1:
                pose=r.state()['tcp_xyzquat'];pose[2]+=.001
                return {'name':'move_to','arguments':{'target':{'pose_xyzquat':pose},'note':'small calibrated motion'}}
            if self.calls==2:return {'name':'set_gripper','arguments':{'gripper':1.,'note':'open'}}
            return {'name':'done','arguments':{'reason':'fake-agent transport smoke only'}}
    directory=tmp_path/'run'
    recorder=RunRecorder(directory,{'instruction':'fake agent smoke; no task-success assertion'})
    agent=Agent()
    status=run_loop(runtime_config(config),resolve_run_input('fake-agent transport smoke',None,'offline-fake'),
        r,Cameras(),Video(),agent,ToolExecutor(load_tool_catalog(),('left',),r,PixelLocalizer(config)),
        recorder,display=RunConsole())
    output=recorder.close(status,None)
    assert status=='completed'
    assert agent.calls==3
    assert Path(output).is_dir()
    jsonl=list(Path(output).glob('*.jsonl'))
    assert jsonl
    events='\n'.join(p.read_text(encoding='utf-8') for p in jsonl)
    assert 'execution_result' in events and 'return_home' in events
