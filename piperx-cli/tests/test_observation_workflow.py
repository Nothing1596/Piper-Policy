"""Vertical slice using saved synthetic evidence; no CAN/cameras required."""
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from piperx_middleware.console_interaction import InteractiveConsoleController
from piperx_middleware.scene_bridge import SceneBridge, annotate_capture, calibration
from piperx_middleware.task_plan import TaskPlanStore


def evidence(root, serial='cam', role='fixed', z=1000, state=True):
    root.mkdir(parents=True,exist_ok=True)
    cfile=root/(serial+'-calibration.json')
    cfile.write_text(json.dumps({'camera_serial':serial,'role':role,'units':'m',
        'base_from_camera' if role=='fixed' else 'flange_from_camera':np.eye(4).tolist()}))
    (root/'camera.json').write_text(json.dumps({'cameras':[{'serial':serial,'role':role,'calibration_file':str(cfile)}]}))
    folder=root/'observations'/('view'+str(z));folder.mkdir(parents=True,exist_ok=True)
    np.save(folder/'depth.npy',np.full((40,40),z,np.uint16))
    metadata={'camera_serial':serial,'frame':'camera_color_optical','captured_at':'2026-10-09T00:00:00Z',
        'depth_path':str(folder/'depth.npy'),'depth_scale_m':.001,
        'intrinsics':{'width':40,'height':40,'fx':50.,'fy':50.,'ppx':20.,'ppy':20.,'coeffs':[0.]*5}}
    if state:
        metadata['robot_alignment']={'offset_s':.01,'request_latency_s':.02,'state':{
            'observation_identity':{'instance_id':'test','connection_epoch':1},
            'robot':{'q_deg':[0.]*6,'velocity_deg_s':[0.]*6,'feedback_age_s':.01}}}
    path=folder/'observation.json';path.write_text(json.dumps(metadata))
    annotate_capture(root,{'frames':[metadata]})
    return str(path)


def controller(root):
    class Bridge:
        tools=[]
    return InteractiveConsoleController(Bridge(),root,emit=lambda _:None)


@pytest.mark.asyncio
async def test_plan_geometry_writeback_and_model_cannot_supply_transform(tmp_path):
    c=controller(tmp_path)
    plan=await c.invoke_tool('task_plan',{'operation':'create','goal':'Measure the cup handle'})
    assert plan.get('task_id'),plan
    update=await c.invoke_tool('task_plan',{'operation':'update','task_id':plan['task_id'],
        'expected_version':plan['version'],'changes':{'questions':[{'id':'surface','question':'Where is the visible surface?',
            'affects_action':True,'source':'geometry','query':{'operation':'surface_point','pixel':[20,20]}}]}})
    assert not update.get('error'),update
    path=evidence(tmp_path)
    request={'operation':'surface_point','observation_path':path,'pixel':[20,20],'question_id':'surface','expected_version':update['version']}
    result=await c.invoke_tool('scene_query',request)
    assert result.get('status')=='ok',result
    assert np.allclose(result['point_m'],[0,0,1])
    saved=c.plans().get()
    assert saved['questions'][0]['state']=='answered'
    bad=await c.invoke_tool('scene_query',dict(request,base_from_camera=np.eye(4).tolist()))
    assert 'error' in bad
    assert (await c.invoke_tool('task_plan',{'operation':'guide'}))['skill']=='observe-and-plan'
    resumed=await c.invoke_tool('task_plan',{'operation':'resume','task_id':plan['task_id']})
    assert resumed['questions'][0]['state']=='stale'


@pytest.mark.asyncio
async def test_geometry_no_calibration_optical_and_changed_or_moving_rejected(tmp_path):
    plans=TaskPlanStore(tmp_path);plans.create('Measure')
    scene=SceneBridge(tmp_path,plans)
    path=evidence(tmp_path)
    args={'operation':'surface_point','observation_path':path,'pixel':[20,20]}
    cfg=tmp_path/'camera.json';saved=cfg.read_bytes();cfg.unlink()
    assert (await scene.query(args))['frame']=='camera_color_optical'
    cfg.write_bytes(saved)
    frame=json.loads(Path(path).read_text());frame['robot_alignment']['state']['robot']['velocity_deg_s']=[2.]*6
    Path(path).write_text(json.dumps(frame))
    with pytest.raises(ValueError,match='stationary'):await scene.query(args)
    frame['robot_alignment']['state']['robot']['velocity_deg_s']=[0.]*6
    frame['calibration_id']='obsolete';Path(path).write_text(json.dumps(frame))
    with pytest.raises(ValueError,match='changed'):await scene.query(args)
    with pytest.raises(ValueError):await scene.query(dict(args,observation_path='/etc/passwd'))


@pytest.mark.asyncio
async def test_late_geometry_cannot_overwrite_edited_question(tmp_path,monkeypatch):
    plans=TaskPlanStore(tmp_path);p=plans.create('Measure')
    p=plans.update(p['task_id'],p['version'],{'questions':[{'id':'q','question':'Depth?',
        'affects_action':True,'source':'geometry'}]})
    path=evidence(tmp_path);scene=SceneBridge(tmp_path,plans)
    create=asyncio.create_subprocess_exec
    async def edited(*args,**kwargs):
        plans.update(p['task_id'],p['version'],{'current_stage':'changed'})
        return await create(*args,**kwargs)
    monkeypatch.setattr(asyncio,'create_subprocess_exec',edited)
    result=await scene.query({'operation':'surface_point','observation_path':path,'pixel':[20,20],'question_id':'q','expected_version':p['version']})
    assert result['plan_writeback']['status']=='conflict'
    assert plans.get()['current_stage']=='changed'


@pytest.mark.asyncio
async def test_worker_cancel_is_reaped(tmp_path,monkeypatch):
    import sys
    plans=TaskPlanStore(tmp_path);plans.create('Measure');scene=SceneBridge(tmp_path,plans)
    create=asyncio.create_subprocess_exec
    launched=[]
    async def slow(*args,**kwargs):
        p=await create(sys.executable,'-c','import time;time.sleep(60)',**kwargs);launched.append(p);return p
    monkeypatch.setattr(asyncio,'create_subprocess_exec',slow)
    task=asyncio.create_task(scene.query({'operation':'fit_plane','target_id':'x','batch_id':'b'}))
    while not launched:await asyncio.sleep(.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):await task
    assert launched[0].returncode is not None and scene.proc is None


@pytest.mark.asyncio
async def test_stored_measurement_revalidates_calibration_and_wrist_fk(tmp_path):
    plans=TaskPlanStore(tmp_path);plans.create('Measure')
    path=evidence(tmp_path,role='wrist')
    scene=SceneBridge(tmp_path,plans)
    point=await scene.query({'operation':'surface_point','observation_path':path,'pixel':[20,20]})
    from piperx_middleware.kinematics import PiperKinematics
    expected=(PiperKinematics().matrix([0.]*6) @ [0,0,1,1])[:3]
    assert np.allclose(point['point_m'],expected)
    file=tmp_path/'cam-calibration.json'
    data=json.loads(file.read_text());data['flange_from_camera'][0][3]=.1
    file.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='changed'):
        await scene.query({'operation':'point_distance','point_id':point['result_id'],'other_point_id':point['result_id']})


@pytest.mark.asyncio
async def test_observe_question_uses_original_plan_version(tmp_path):
    c=controller(tmp_path);p=c.plans().create('Measure')
    c.plans().update(p['task_id'],p['version'],{'questions':[{'id':'q','question':'Depth?',
        'affects_action':True,'source':'geometry','query':{'operation':'surface_point','pixel':[20,20]}}]})
    path=evidence(tmp_path)
    class Camera:
        async def observe(self,**kwargs):
            p=c.plans().get();c.plans().update(p['task_id'],p['version'],{'current_stage':'changed'})
            return {'frames':[json.loads(Path(path).read_text())]}
    c.camera=lambda:Camera()
    with pytest.raises(ValueError,match='changed during capture'):await c.observe(question_id='q')
    assert c.plans().get()['questions'][0]['state']=='pending'


@pytest.mark.asyncio
async def test_operator_camera_wizard_and_invalid_config_preserve_previous(tmp_path):
    c=controller(tmp_path);answers=iter(['1','123','wrist',''])
    async def prompt(*args,**kwargs):return next(answers)
    c.prompt=prompt
    await c._cmd_config('cameras')
    path=tmp_path/'camera.json';before=path.read_bytes()
    assert json.loads(before)['cameras']==[{'serial':'123','role':'wrist','calibration_file':None}]
    await c._cmd_config('cameras [{"serial":"123","role":"invalid"}]')
    assert path.read_bytes()==before


def test_unavailable_reanswer_clears_old_measurement(tmp_path):
    store=TaskPlanStore(tmp_path);p=store.create('Measure')
    p=store.update(p['task_id'],p['version'],{'questions':[{'id':'q','question':'Depth?', 'affects_action':True,'source':'geometry'}]})
    p=store.record_result(p['task_id'],p['version'],'q',{'status':'answered','value':.4,'evidence':'old.json'})
    p=store.resume(p['task_id'])
    p=store.record_result(p['task_id'],p['version'],'q',{'status':'unavailable','value':{'reason':'no depth'}})
    q=p['questions'][0]
    assert q['state']=='unavailable' and q['result']=={'reason':'no depth'}
    assert 'answered_at' not in q and 'evidence' not in q


@pytest.mark.asyncio
async def test_historical_action_does_not_stale_plan_but_completed_pending_action_does(tmp_path,monkeypatch):
    from piperx_middleware.console import ConsoleController
    import time
    c=controller(tmp_path);c.available_tools=lambda:[{'name':'move','inputSchema':{'required':['request_id']}}]
    p=c.plans().create('Measure')
    reply={'job_id':'old','status':'succeeded','accepted_at':time.time()-60}
    async def invoke(*args):return dict(reply)
    monkeypatch.setattr(ConsoleController,'invoke_tool',invoke)
    await c.invoke_tool('move',{'request_id':'old'})
    assert c.plans().get()['version']==p['version']
    reply={'job_id':'new','status':'awaiting_approval','accepted_at':time.time()}
    await c.invoke_tool('move',{'request_id':'new'})
    v=c.plans().get()['version']
    reply={'job_id':'new','status':'succeeded'}
    await c.invoke_tool('robot_status',{'job_id':'new'})
    assert c.plans().get()['version']==v+1 and not c._plan_jobs


def test_goal_change_invalidates_answers(tmp_path):
    store=TaskPlanStore(tmp_path);p=store.create('Cup A')
    p=store.update(p['task_id'],p['version'],{'questions':[{'id':'q','question':'Depth?', 'affects_action':True,'source':'geometry'}]})
    p=store.record_result(p['task_id'],p['version'],'q',{'value':.4})
    p=store.update(p['task_id'],p['version'],{'goal':'Cup B'})
    assert p['questions'][0]['state']=='stale'


def test_registered_root_may_use_system_symlink_but_inner_link_rejected(tmp_path):
    from piperx_middleware.scene_bridge import _safe_observation
    profile=tmp_path/'actual';profile.mkdir()
    alias=tmp_path/'profile-alias';alias.symlink_to(profile,target_is_directory=True)
    folder=profile/'observations';folder.mkdir();data=folder/'observation.json';data.write_text('{}')
    assert _safe_observation(alias,str(alias/'observations/observation.json'))==data
    link=folder/'link.json';link.symlink_to(data)
    with pytest.raises(ValueError,match='symlink'):_safe_observation(alias,str(link))


@pytest.mark.asyncio
async def test_external_question_requires_observed_plan_version(tmp_path):
    c=controller(tmp_path);p=c.plans().create('Measure')
    p=c.plans().update(p['task_id'],p['version'],{'questions':[{'id':'q','question':'Depth?', 'affects_action':True,'source':'geometry'}]})
    path=evidence(tmp_path)
    request={'operation':'surface_point','question_id':'q','observation_path':path,'pixel':[20,20]}
    assert 'error' in await c.invoke_tool('scene_query',request)
    c.plans().update(p['task_id'],p['version'],{'current_stage':'changed'})
    assert 'error' in await c.invoke_tool('scene_query',dict(request,expected_version=p['version']))
    assert c.plans().get()['questions'][0]['state']=='pending'


def test_other_task_requires_explicit_resume_for_writeback(tmp_path):
    from piperx_middleware.models import DomainError
    store=TaskPlanStore(tmp_path);a=store.create('A');b=store.create('B')
    with pytest.raises(DomainError,match='Resume'):
        store.update(a['task_id'],a['version'],{'current_stage':'changed'})
    a=store.resume(a['task_id'])
    assert store.get()['task_id']==a['task_id']
    store.update(a['task_id'],a['version'],{'current_stage':'changed'})
    assert store.get(b['task_id'])['goal']=='B'


@pytest.mark.asyncio
async def test_distance_has_durable_evidence_and_batch_query_does_not_bind_new_frame(tmp_path):
    c=controller(tmp_path);p=c.plans().create('Measure')
    path=evidence(tmp_path);point=await c.scene().query({'operation':'surface_point','observation_path':path,'pixel':[20,20]})
    result=await c.scene().query({'operation':'point_distance','point_id':point['result_id'],'other_point_id':point['result_id']})
    assert (tmp_path/result['result_path']).is_file() and result['distance_m']==0.
    p=c.plans().update(p['task_id'],p['version'],{'questions':[{'id':'plane','question':'Plane?', 'affects_action':True,'source':'geometry',
        'query':{'operation':'fit_plane','target_id':'cup','batch_id':'b'}}]})
    class Camera:
        async def observe(self,**kwargs):return {'frames':[json.loads(Path(path).read_text())]}
    c.camera=lambda:Camera()
    response=await c.observe(question_id='plane')
    assert 'plan_hint' in response and 'geometry' not in response


@pytest.mark.asyncio
async def test_plan_read_expires_answers_after_in_place_calibration_edit(tmp_path):
    c = controller(tmp_path)
    plan = c.plans().create('Measure')
    plan = c.plans().update(plan['task_id'], plan['version'], {'questions': [{
        'id': 'q', 'question': 'Surface?', 'affects_action': True, 'source': 'geometry'}]})
    path = evidence(tmp_path)
    result = await c.invoke_tool('scene_query', {
        'operation': 'surface_point', 'pixel': [20, 20], 'observation_path': path,
        'question_id': 'q', 'expected_version': plan['version']})
    assert result['status'] == 'ok'
    answered = await c.invoke_tool('task_plan', {'operation': 'get'})
    assert answered['questions'][0]['state'] == 'answered'
    file = tmp_path / 'cam-calibration.json'
    data = json.loads(file.read_text())
    data['base_from_camera'][0][3] = .1
    file.write_text(json.dumps(data))
    stale = await c.invoke_tool('task_plan', {'operation': 'get', 'task_id': plan['task_id']})
    assert stale['questions'][0]['state'] == 'stale'
    assert stale['version'] == answered['version'] + 1
    assert c.current_plan()['version'] == stale['version']
    assert c.plans().get()['questions'][0]['state'] == 'stale'


def test_missing_geometry_provenance_requests_resample(tmp_path):
    scene = SceneBridge(tmp_path, TaskPlanStore(tmp_path))
    with pytest.raises(ValueError, match='provenance; resample'):
        scene._revalidate({'calibration_provenance': [{'calibration_id': 'old'}]})
