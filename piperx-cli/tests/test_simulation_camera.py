import asyncio
import pytest
from piperx_middleware.mcp_server import create_mcp
from piperx_middleware.simulation_camera import capture


class NoIO:
    def call(self, *args):
        raise AssertionError('Invalid destinations must not contact executor')


@pytest.mark.parametrize('output', ['.', '../escape', 'existing'])
def test_capture_rejects_unsafe_or_existing_destination(tmp_path, output):
    (tmp_path/'existing').mkdir()
    with pytest.raises(Exception, match='workspace|already exists'):
        capture(NoIO(), tmp_path, output)


def test_robot_camera_opt_in(tmp_path):
    async def run():
        plain = {t.name for t in await create_mcp(NoIO()).list_tools()}
        camera = {t.name for t in await create_mcp(NoIO(), workspace=tmp_path).list_tools()}
        assert 'simulation_observe' not in plain
        assert camera - plain == {'simulation_observe'}
        assert 'video_compile' not in camera
    asyncio.run(run())


def sensor_fixture():
    import base64,io
    import numpy as np
    from PIL import Image
    jpg=io.BytesIO();Image.new('RGB',(40,40),'gray').save(jpg,format='JPEG')
    depth=io.BytesIO();np.save(depth,np.full((40,40),.5,np.float32))
    return {'provenance':'mujoco_sensor','instance_id':'sim-test','connection_epoch':1,'epoch':0,'source_stamp_s':10.,
        'rgb_jpeg_b64':base64.b64encode(jpg.getvalue()).decode(),'depth_npy_b64':base64.b64encode(depth.getvalue()).decode(),
        'camera_info':{'intrinsics':[[50,0,20],[0,50,20],[0,0,1]],'base_from_camera':np.eye(4).tolist()},
        'robot_alignment':{'offset_s':0.,'request_latency_s':0.,'state':{
            'robot':{'q_deg':[0]*6,'velocity_deg_s':[0]*6,'feedback_age_s':0.},
            'observation_identity':{'instance_id':'sim-test','connection_epoch':1}}}}


@pytest.mark.asyncio
async def test_sim_observation_normalizes_without_overwriting_operator_config(tmp_path):
    import json
    from piperx_middleware.simulation_camera import SimulationCamera,capture
    from piperx_middleware.scene_bridge import SceneBridge
    from piperx_middleware.task_plan import TaskPlanStore
    operator=tmp_path/'camera.json';operator.write_text('{"serial":"physical-camera"}')
    original=operator.read_bytes();data=sensor_fixture()
    async def rest(*args):return data
    camera=SimulationCamera(tmp_path,rest)
    plans=TaskPlanStore(tmp_path);plans.create('Measure fixture')
    one=await camera.observe(roi=[10,10,30,30]);frame=one['frames'][0]
    result=await SceneBridge(tmp_path,plans).query({'operation':'surface_point',
        'observation_path':frame['observation_path'],'pixel':[20,20]})
    assert result['frame']=='base' and result['point_m']==[0,0,.5]
    assert result['source']['simulated'] is True and operator.read_bytes()==original
    assert len([p for p in camera.image_inputs(one) if p['type']=='image_url'])==2
    data['connection_epoch']=2;data['robot_alignment']['state']['observation_identity']['connection_epoch']=2
    await camera.observe()
    with pytest.raises(ValueError,match='changed'):
        await SceneBridge(tmp_path,plans).query({'operation':'surface_point','observation_path':frame['observation_path'],'pixel':[20,20]})
    await camera.close();assert not camera.image_inputs(one)


@pytest.mark.asyncio
async def test_sim_errors_do_not_fallback_or_publish_evidence(tmp_path):
    from piperx_middleware.simulation_camera import SimulationCamera
    data=sensor_fixture()
    async def rest(*args):return data
    c=SimulationCamera(tmp_path,rest)
    with pytest.raises(ValueError,match='serial'):await c.observe(serial='wrong')
    with pytest.raises(ValueError,match='ROI'):await c.observe(roi=[0,0,99,99])
    data['robot_alignment']['state']['observation_identity']['connection_epoch']=2
    with pytest.raises(ValueError,match='identities'):await c.observe()
    assert not (tmp_path/'simulation-camera.json').exists()
    assert not (tmp_path/'observations').exists()


@pytest.mark.asyncio
async def test_sim_wrist_and_dual_camera_observations(tmp_path):
    import numpy as np
    from piperx_middleware.kinematics import PiperKinematics
    from piperx_middleware.scene_bridge import SceneBridge
    from piperx_middleware.simulation_camera import SimulationCamera
    from piperx_middleware.task_plan import TaskPlanStore

    q_deg = [0.0, 20.0, -35.0, 0.0, 25.0, 0.0]
    flange_from_cam = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.9063078, 0.4226183, -0.068],
        [0.0, -0.4226183, 0.9063078, 0.055],
        [0.0, 0.0, 0.0, 1.0],
    ])
    base_from_cam = PiperKinematics(tcp_offset_m=None).matrix(q_deg) @ flange_from_cam

    async def rest(method, path):
        data = sensor_fixture()
        data['robot_alignment']['state']['robot']['q_deg'] = q_deg
        if 'mujoco-wrist' in path:
            data['camera_serial'] = 'mujoco-wrist'
            data['camera_role'] = 'wrist'
            data['camera_info'] = {
                'serial': 'mujoco-wrist',
                'role': 'wrist',
                'intrinsics': [[50, 0, 20], [0, 50, 20], [0, 0, 1]],
                'base_from_camera': base_from_cam.tolist(),
                'flange_from_camera': flange_from_cam.tolist(),
            }
        else:
            data['camera_serial'] = 'mujoco-overview'
            data['camera_role'] = 'fixed'
        return data

    camera = SimulationCamera(tmp_path, rest)
    plans = TaskPlanStore(tmp_path)
    plans.create('Dual camera check')
    bundle = await camera.observe(serial='all')
    assert len(bundle['frames']) == 2
    serials = [f['serial'] for f in bundle['frames']]
    assert serials == ['mujoco-overview', 'mujoco-wrist']
    wrist_frame = bundle['frames'][1]
    assert wrist_frame['role'] == 'wrist'
    res = await SceneBridge(tmp_path, plans).query({
        'operation': 'surface_point',
        'observation_path': wrist_frame['observation_path'],
        'pixel': [20, 20],
    })
    expected = (base_from_cam @ np.array([0.0, 0.0, 0.5, 1.0]))[:3]
    assert res['frame'] == 'base'
    assert np.allclose(res['point_m'], expected, atol=1e-4)
    await camera.close()
