"""Feedback causality and failure boundaries; test doubles never claim physics."""
import base64
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
sys.path.insert(0,str(Path(__file__).parents[1]))
import run_longvil_online as online
from piperx_middleware.task_plan import TASK_PLAN_TOOL


def tool(name,schema):
    value={'name':name,'description':name,'inputSchema':schema}
    return SimpleNamespace(model_dump=lambda **kwargs:value)


class Session:
    def __init__(self,unknown=False):
        self.time=0.;self.calls=[];self.moves=[];self.unknown=unknown
        self.plan={'task_id':'aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa','version':1,'goal':'generic','interpretation':''}
    async def list_tools(self):
        return SimpleNamespace(tools=[tool('move_to',{'type':'object','required':['request_id','xyz_m'],'properties':{
            'request_id':{'type':'string'},'xyz_m':{'type':'array','items':{'type':'number'},'minItems':3,'maxItems':3}},'additionalProperties':False}),
            tool('camera_observe',{'type':'object'}),tool('task_plan',TASK_PLAN_TOOL['inputSchema']),
            tool('robot_stop',{'type':'object','properties':{'request_id':{'type':'string'}}}),
            tool('scene_query',{'type':'object'})])
    async def call_tool(self,name,args):
        self.calls.append((name,copy.deepcopy(args)));images=[]
        if name=='task_plan':
            if args['operation']=='update':self.plan.update(args['changes']);self.plan['version']+=1
            value=copy.deepcopy(self.plan)
        elif name=='camera_observe':
            self.time+=1.;frame={'source_stamp_s':self.time,'sim_time_s':self.time,'instance_id':'sim','connection_epoch':1,
                'observation_path':f'/private/profile/observations/{self.time}/observation.json',
                'robot_alignment':{'state':{'backend':'mujoco','robot':{'q_deg':[len(self.moves)]*6}}}}
            value={'simulated':True,'frames':[frame]}
            images=[SimpleNamespace(type='image',mimeType='image/jpeg',data=base64.b64encode(str(self.time).encode()).decode())]
        elif name=='move_to':
            self.time+=.1;self.moves.append(copy.deepcopy(args))
            value={'status':'outcome_unknown' if self.unknown else 'succeeded','command':args,'request_id':args['request_id']}
        elif name=='scene_query':value={'status':'ok','point_m':[.3,.1,.2]}
        elif name=='robot_stop':self.time+=.1;value={'status':'stopped','request_id':args['request_id']}
        else:raise AssertionError(name)
        return SimpleNamespace(isError=False,structuredContent=value,content=images)


def fixture(tmp_path):
    image=tmp_path/'demo.jpg';image.write_bytes(b'demonstration pixels')
    packet={'source':{'sha256':'video'},'frames':[{'frame_id':'f0','timestamp_s':0.,'image_path':str(image)}],
            'evaluation':{'scope':'recorded CV'}}
    args=SimpleNamespace(session_id='session',object_list=['red cylinder','green cuboid'],max_rounds=6,max_seconds=60,
        completion_budget=65536,effort='high',max_tools=10,profile='private-profile',plan='on',skill='on')
    evidence={'source_sha256':'video','initial':packet,'supplement':packet}
    return args,evidence


def plan_update(version=1):
    return {'operation':'update','task_id':'aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa','expected_version':version,
            'changes':{'current_stage':'execute','interpretation':'Model-owned evidence and next action'}}


@pytest.mark.asyncio
async def test_model_decision_receives_actual_previous_result_and_new_pixels(tmp_path,monkeypatch):
    args,evidence=fixture(tmp_path);session=Session();payloads=[];events=[]
    async def paid(args,client,payload,ident):
        payloads.append(copy.deepcopy(payload));turn=len(payloads)-1
        if turn in (0,4):name='task_plan';body={'operation':'get','task_id':session.plan['task_id']}
        elif turn==1:name='task_plan';body=plan_update()
        elif turn<4:
            name='move_to';body={'request_id':'motion-'+str(turn-2),'xyz_m':[.3,.1+(turn-2)*.1,.2],
                                'observation_id':'live-'+str(turn-1).zfill(4)}
        else:name='finish_task';body={'summary':'observed','uncertainties':[],'observation_id':'live-0003'}
        return {'model':'deepseek-flash','usage':{'completion_tokens':100},'choices':[{'finish_reason':'tool_calls',
            'message':{'role':'assistant','content':'','reasoning_content':'provider continuation',
                       'tool_calls':[{'id':'c'+str(turn),'function':{'name':name,'arguments':json.dumps(body)}}]}}]}
    async def remote(args,operation,root):
        return {'sim_time_s':session.time,'bilateral_contact_objects':[],
                'objects':{'GT_PRIVATE_OBJECT':{'kind':'dynamic','position':[9,9,9]}},'secret_gt':'GT_PRIVATE_ANSWER'}
    monkeypatch.setattr(online,'paid',paid);monkeypatch.setattr(online,'remote',remote)
    report={};await online.online_loop(session,args,evidence,events.append,tmp_path,report,None)
    assert report['status']=='finished' and report['motions']==2 and report['observations']==3
    assert session.moves==[{'request_id':'motion-0','xyz_m':[.3,.1,.2]},
                           {'request_id':'motion-1','xyz_m':[.3,.2,.2]}]
    second=json.dumps(payloads[3]['messages']);assert 'live-0002' in second and 'succeeded' in second
    assert 'GT_PRIVATE_ANSWER' not in second and '/private/profile' not in second
    assert any('reasoning_content' in m for m in payloads[3]['messages'] if m['role']=='assistant')
    assert 'CURRENT execution_progress' in second and 'Model-owned evidence' in second
    assert not any(t['function']['name']=='submit_program' for t in payloads[0]['tools'])
    chain=json.loads((tmp_path/'action-chain.json').read_text())
    assert [r['result_observation_id'] for r in chain]==['live-0002','live-0003']


@pytest.mark.asyncio
async def test_unknown_motion_is_recorded_before_any_camera_or_api_retry(tmp_path,monkeypatch):
    args,evidence=fixture(tmp_path);args.plan='off';session=Session(unknown=True);calls=[]
    async def paid(*values):
        calls.append(1);name='move_to'
        body={'request_id':'unknown','xyz_m':[.3,0,.2],'observation_id':'live-0001'}
        return {'model':'deepseek-flash','usage':{'completion_tokens':50},'choices':[{
          'message':{'role':'assistant','content':'','tool_calls':[{'id':str(len(calls)),'function':{'name':name,
            'arguments':json.dumps(body)}}]}}]}
    async def remote(*args):return {'sim_time_s':session.time,'objects':{}}
    monkeypatch.setattr(online,'paid',paid);monkeypatch.setattr(online,'remote',remote)
    with pytest.raises(online.ModelCallAnomaly,match='unknown'):
        await online.online_loop(session,args,evidence,lambda row:None,tmp_path,{},None)
    assert len(calls)==1 and len(session.moves)==1
    assert len([c for c in session.calls if c[0]=='camera_observe'])==1
    assert json.loads((tmp_path/'action-chain.json').read_text())[0]['actual_result']['status']=='outcome_unknown'


def test_old_frame_duplicate_motion_and_early_finish_do_not_execute():
    gate=online.OnlineGate();gate.observation('new');gate.plan_required=False
    with pytest.raises(ValueError,match='latest'):gate.before('move_to',{'observation_id':'old','request_id':'a'},0)
    with pytest.raises(ValueError,match='actual'):gate.before('finish_task',{'observation_id':'new'},0)
    gate.before('move_to',{'observation_id':'new','request_id':'a'},0);gate.motions=1
    with pytest.raises(ValueError,match='Duplicate'):gate.before('move_to',{'observation_id':'new','request_id':'a'},1)
    with pytest.raises(ValueError):gate.before('finish_task',{'observation_id':'new'},0)


def test_plan_checkpoints_are_public_progress_not_motion_permission():
    gate=online.OnlineGate();gate.observation('new')
    gate.plan={'task_id':'real','version':1}
    body={'observation_id':'new','request_id':'a'}
    # Soft policy: motion and finish never require a plan read or write.
    gate.before('move_to',body,0);gate.motions=1
    gate.before('finish_task',{'observation_id':'new'},1)
    # A foreign plan result never replaces the real store state.
    gate.accept_plan({'task_id':'foreign','version':2,'interpretation':'x'},'update',0)
    assert gate.plan['task_id']=='real' and not gate.plan_required
    # Store version binding stays: a stale expected_version asks for a re-read.
    with pytest.raises(ValueError,match='version'):gate.before('task_plan',{'operation':'update','expected_version':2},2)
    assert gate.plan_required
    gate.before('task_plan',{'operation':'update','expected_version':1},2)
    # Observed close/open is a public checkpoint, not task success.
    before={'robot':{'gripper_width_m':.04,'gripper_contact':{'supported':True,'finger1_force_n':10,'finger2_force_n':10}}}
    gate.released(before,{'robot':{'gripper_width_m':.07}})
    assert gate.release_checkpoints==1
    gate.accept_plan({'task_id':'real','version':2,'interpretation':'Released, relation still uncertain'},'update',2)
    assert not gate.plan_required and gate.plan['version']==2
    gate.before('move_to',{'observation_id':'new','request_id':'b'},3)
    assert 'task_success' not in json.dumps(online.progress_context(gate,[],[]))


@pytest.mark.asyncio
async def test_exact_read_cache_is_frame_scoped(tmp_path,monkeypatch):
    args,evidence=fixture(tmp_path);args.max_rounds=4;session=Session();count=0
    async def paid(*values):
        nonlocal count
        count+=1
        name='camera_observe' if count==3 else 'scene_query'
        body={} if count==3 else {'operation':'surface_point','pixel':[10,20],'observation_path':'live-0001' if count<3 else 'live-0002'}
        return {'model':'deepseek-flash','usage':{'completion_tokens':50},'choices':[{'message':{
            'role':'assistant','content':'','tool_calls':[{'id':str(count),'function':{'name':name,'arguments':json.dumps(body)}}]}}]}
    monkeypatch.setattr(online,'paid',paid);report={};events=[]
    await online.online_loop(session,args,evidence,events.append,tmp_path,report,None)
    assert report['read_cache_hits']==1
    assert len([c for c in session.calls if c[0]=='scene_query'])==2
    assert any(r['kind']=='tool_result' and r['result'].get('cache_hit') for r in events)


def test_camera_axes_are_sensor_calibration_not_guessed_pixel_mapping():
    m=[[0,1,0,.3],[1,0,0,0],[0,0,-1,.9],[0,0,0,1]]
    frame={'camera_info':{'base_from_camera':m},'calibration_id':'known'}
    axes=online.camera_axes(frame)
    assert axes['image_right_base']==[0,1,0] and axes['image_down_base']==[1,0,0]
    assert online.camera_axes({})['status']=='unavailable'


def test_accumulation_schema_catches_observed_model_nesting_error():
    native=tool('scene_query',{'type':'object'}).model_dump()
    schema=online.select_tools([native])[0]['inputSchema']
    with pytest.raises(online.ValidationError,match='observations'):
        online.validate({'operation':'accumulate_region','target_id':'red','batch_id':'v1',
                         'observation_path':'live-0001','roi':[1,2,3,4]},schema)
    online.validate({'operation':'accumulate_region','target_id':'red','batch_id':'v1',
                     'observations':[{'observation_path':'live-0001','roi':[1,2,3,4]}]},schema)
    online.validate({'operation':'surface_point','observation_path':'live-0001','pixel':[1,2]},schema)


@pytest.mark.parametrize('relation,expected',[
    ('left',[.20,-.266]),('right',[.20,-.034]),('front',[.28,-.15]),('behind',[.12,-.15])])
def test_placement_uses_directional_extents_not_grasp_minor_width(relation,expected):
    source={'observed_bounds_m':{'lower':[.28,.10,-.20],'upper':[.32,.18,-.10]},
            'tcp_above_table_m':.0375,'poses':{'grasp':{'rpy_deg':[180,0,0],'xyz_m':[.3,.14,-.1625]}},'observed_width_m':.04}
    summary={'observed_bounds_m':{'lower':[.17,-.204,-.20],'upper':[.23,-.096,-.146]}}
    plane={'normal':[0,0,1],'point_on_plane_m':[0,0,-.20]}
    camera={'base_from_camera':[[0,1,0,.3],[1,0,0,0],[0,0,-1,.9],[0,0,0,1]]}
    target=online.placement_geometry(source,summary,plane,camera,relation,0,.022 if relation in ('left','right') else .03)
    assert target['xyz_m'][:2]==pytest.approx(expected)
    assert target['xyz_m'][2]==pytest.approx(-.1625)
    if relation=='left':
        assert target['source_half_extent_along_direction_m']==pytest.approx(.04)
        assert target['reference_half_extent_along_direction_m']==pytest.approx(.054)


def test_top_placement_uses_observed_support_and_recorded_grasp_offset():
    source={'tcp_above_table_m':.0125,'observed_bounds_m':{'lower':[.38,0,-.2],'upper':[.42,.04,-.15]},
            'poses':{'grasp':{'rpy_deg':[180,0,0],'xyz_m':[.4,.02,-.1875]}}}
    summary={'observed_bounds_m':{'lower':[.17,-.24,-.20],'upper':[.23,-.18,-.10]}}
    target=online.placement_geometry(source,summary,{'normal':[0,0,1],'point_on_plane_m':[0,0,-.2]}, {},'top',0,.02)
    assert target['xyz_m']==pytest.approx([.20,-.21,-.0675])
    assert target['support_height_m']==pytest.approx(-.10)
    assert 'Slip' in target['scope']


def test_historical_source_rotation_changes_projected_envelope():
    source={'observed_bounds_m':{'lower':[0,0,-.20],'upper':[.04,.12,-.10]},
            'tcp_above_table_m':.0375,'poses':{'grasp':{'rpy_deg':[180,0,0],'xyz_m':[.02,.06,-.1625]}}}
    summary={'observed_bounds_m':{'lower':[.17,-.20,-.20],'upper':[.23,-.10,-.146]}}
    plane={'normal':[0,0,1],'point_on_plane_m':[0,0,-.20]}
    camera={'base_from_camera':[[0,1,0,.3],[1,0,0,0],[0,0,-1,.9],[0,0,0,1]]}
    target=online.placement_geometry(source,summary,plane,camera,'left',90,.02)
    assert target['source_half_extent_along_direction_m']==pytest.approx(.02)
    assert target['xyz_m'][1]==pytest.approx(-.24)


def test_reference_roi_with_robot_occlusion_is_not_a_placement_target():
    summary={'observed_bounds_m':{'lower':[.17,-.20,-.20],'upper':[.23,-.10,.15]}}
    with pytest.raises(ValueError,match='occluding'):
        online.placement_geometry({},summary,{'normal':[0,0,1],'point_on_plane_m':[0,0,-.20]}, {},'top',0)


def test_finger_envelope_and_grasp_eccentricity_affect_actual_tcp_target():
    source={'observed_bounds_m':{'lower':[.28,.10,-.20],'upper':[.32,.18,-.10]},
            'tcp_above_table_m':.0375,'poses':{'grasp':{'rpy_deg':[180,0,0],'xyz_m':[.29,.14,-.1625]}}}
    summary={'observed_bounds_m':{'lower':[.17,-.204,-.20],'upper':[.23,-.096,-.146]}}
    plane={'normal':[0,0,1],'point_on_plane_m':[0,0,-.20]}
    camera={'base_from_camera':[[0,1,0,.3],[1,0,0,0],[0,0,-1,.9],[0,0,0,1]]}
    target=online.placement_geometry(source,summary,plane,camera,'front',0,.02,.06)
    assert target['finger_half_extent_along_direction_m']==pytest.approx(.056)
    assert target['assumed_source_offset_xy_from_tcp_m']==pytest.approx([.01,0])
    assert target['xyz_m'][0]==pytest.approx(.306)
    with pytest.raises(ValueError,match='Unknown'):
        online.placement_geometry(source,summary,plane,camera,'unknown',0)


@pytest.mark.asyncio
async def test_connection_failure_receipt_survives_usage_pause_without_retry(tmp_path,monkeypatch):
    import httpx
    from longvil_budget import Ledger,BudgetPause
    monkeypatch.setenv('DEEPSEEK_API_KEY','test-secret')
    class Client:
        attempts=0
        async def post(self,*args,**kwargs):
            self.attempts+=1;raise httpx.ConnectError('test connection failure test-secret')
    args=SimpleNamespace(output=tmp_path,ledger=Ledger(tmp_path/'ledger.jsonl',limit=None,stop_at=None))
    client=Client()
    with pytest.raises(BudgetPause):
        await online.paid(args,client,{'model':'deepseek-flash','max_tokens':4096},'attempt-01')
    receipt=json.loads((tmp_path/'api-receipts/attempt-01.json').read_text())
    assert receipt['error_type']=='ConnectError' and receipt['http_status'] is None
    assert receipt['error_message']=='test connection failure [REDACTED]' and not receipt['usage_available']
    assert client.attempts==1 and args.ledger.total==3


def test_uncontrolled_drop_is_not_accepted_as_a_successful_placement(monkeypatch):
    monkeypatch.setattr(online,'relation_status',lambda *args:True)
    gt=[('pick_place','source','reference','left')]
    transfer={'source':'source','lift_height_m':.08,'state_after':{},'controlled_release':False}
    assert not online.private_grade(gt,[transfer])['success']
    transfer['controlled_release']=True
    # A claimed release/lift cannot replace measured grasp and final-state
    # evidence; this was the old scorer's false-positive shortcut.
    assert not online.private_grade(gt,[transfer])['success']


@pytest.mark.asyncio
async def test_provider_error_receipt_precedes_missing_usage_pause(tmp_path):
    import httpx
    from longvil_budget import Ledger,BudgetPause
    class Client:
        async def post(self,*args,**kwargs):
            return httpx.Response(503,json={'error':{'code':'overload','message':'provider unavailable'}})
    args=SimpleNamespace(output=tmp_path,ledger=Ledger(tmp_path/'ledger.jsonl',limit=None,stop_at=None))
    with pytest.raises(BudgetPause):
        await online.paid(args,Client(),{'model':'deepseek-flash','max_tokens':4096},'attempt-02')
    receipt=json.loads((tmp_path/'api-receipts/attempt-02.json').read_text())
    assert receipt['http_status']==503 and receipt['error_type']=='HTTP_503'
    assert receipt['provider_error']['code']=='overload' and not receipt['usage_available']


@pytest.mark.asyncio
async def test_stop_returns_new_feedback_even_before_initial_plan(tmp_path,monkeypatch):
    args,evidence=fixture(tmp_path);args.max_rounds=1;session=Session()
    async def paid(*values):
        return {'model':'deepseek-flash','usage':{'completion_tokens':50},'choices':[{'message':{
            'role':'assistant','content':'','tool_calls':[{'id':'stop-call','function':{'name':'robot_stop',
                'arguments':json.dumps({'request_id':'stop-0001'})}}]}}]}
    async def remote(*values):return {'sim_time_s':session.time,'objects':{}}
    monkeypatch.setattr(online,'paid',paid);monkeypatch.setattr(online,'remote',remote)
    report={};await online.online_loop(session,args,evidence,lambda row:None,tmp_path,report,None)
    assert report['observations']==2 and report['motions']==0
    chain=json.loads((tmp_path/'action-chain.json').read_text())
    assert chain[0]['tool']=='robot_stop' and chain[0]['result_observation_id']=='live-0002'


def test_multi_motion_batch_is_rejected_before_any_action():
    gate=online.OnlineGate();calls=[{'id':str(i),'function':{'name':'move_to'}} for i in range(2)]
    assert gate.batch(calls,0) and gate.motions==0 and not gate.motion_ids


def test_projections_keep_real_errors_and_exclude_gt_and_files():
    value={'error':{'code':'ik_no_solution','message':'reject'},'code':['pick()'],'positions':{'a':[0,0,0]},
           'observation_path':'/private/frame.json','rgb_path':'/private/image.jpg','point_m':[.3,.1,.2]}
    projected=online.strip_paths(value,{'/private/frame.json':'live-0001'})
    assert projected=={'error':{'code':'ik_no_solution','message':'reject'},'observation_path':'live-0001','point_m':[.3,.1,.2]}
