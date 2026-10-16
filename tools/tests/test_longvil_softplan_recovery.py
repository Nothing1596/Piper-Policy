"""Soft plan policy, confirmed-blocked feedback and carry recovery boundaries.

All doubles replay recorded outcomes; they never claim physics, model quality or
task success. Private grading is a capturing double because the real
private_grade signature is owned by a separate module author.
"""
import copy
import itertools
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

sys.path.insert(0,str(Path(__file__).parents[1]))
import run_longvil_online as online
import run_longvil_online_pilot as pilot
from piperx_middleware.task_plan import TASK_PLAN_TOOL
from test_longvil_online import Session,fixture,plan_update,tool


SET_GRIPPER={'type':'object','additionalProperties':False,'required':['request_id','width_m'],
             'properties':{'request_id':{'type':'string'},'width_m':{'type':'number'}}}


class ContactWorld:
    """Measured contact/height bookkeeping double; not a physics engine.

    ``flight_z``/``settle_after`` let one scripted drop report an in-flight
    height on the first post-loss read and a settled table height afterwards.
    """
    def __init__(self):
        self.contact=False;self.z=0.;self.width=.07;self.flight_z=None;self.settle_after=0
    def read(self,sim_time):
        if self.flight_z is not None:
            self.z=self.flight_z
            if self.settle_after<=0:self.flight_z=None;self.z=0.
            else:self.settle_after-=1
        return {'sim_time_s':sim_time,'objects':{'obj':{'kind':'dynamic','position':[.3,.1,self.z]}},
                'bilateral_contact_objects':['obj'] if self.contact else [],
                'robot':{'gripper_width_m':self.width}}


def grasp(width=.02):
    def apply(world,body):world.contact=True;world.width=width
    return apply

def open_hand(width=.07):
    def apply(world,body):world.contact=False;world.width=width
    return apply

def lift(height,contact=True):
    def apply(world,body):world.z=height;world.contact=contact
    return apply

def drop(height):
    def apply(world,body):world.contact=False;world.z=height;world.flight_z=height;world.settle_after=1
    return apply


class ScriptedSession(Session):
    """Session double whose motions follow a script over a ContactWorld."""
    def __init__(self,world,script):
        super().__init__();self.world=world;self.script=list(script);self.index=0
    async def list_tools(self):
        base=await super().list_tools()
        return SimpleNamespace(tools=[*base.tools,tool('set_gripper',SET_GRIPPER)])
    async def call_tool(self,name,args):
        if name in ('move_to','set_gripper') and self.index<len(self.script):
            spec=self.script[self.index];self.index+=1
            args=copy.deepcopy(args);self.calls.append((name,copy.deepcopy(args)))
            self.time+=.1
            spec.get('apply',lambda world,body:None)(self.world,args)
            self.moves.append(copy.deepcopy(args))
            value=dict(spec.get('result',{'status':'succeeded'}))
            value.setdefault('request_id',args.get('request_id'));value['command']=copy.deepcopy(args)
            value['measured_after']={'robot':{'gripper_width_m':self.world.width,'velocity_deg_s':[0.]*6}}
            return SimpleNamespace(isError=False,structuredContent=value,content=[])
        return await super().call_tool(name,args)


def decision(name,body,ident):
    return {'model':'deepseek-flash','usage':{'completion_tokens':100},'choices':[{'finish_reason':'tool_calls',
        'message':{'role':'assistant','content':'','tool_calls':[
            {'id':ident,'function':{'name':name,'arguments':json.dumps(body)}}]}}]}


def driver(responses,payloads):
    state={'index':0}
    async def paid(args,client,payload,ident):
        payloads.append(copy.deepcopy(payload))
        index=state['index'];state['index']+=1
        assert index<len(responses),'unexpected extra model request'
        return responses[index]
    return paid


def motion(name,request_id,observation_id,index=1,**extra):
    body={'request_id':request_id,'xyz_m':[.3,.1+index*.1,.2],'observation_id':observation_id,**extra}
    return decision(name,body,request_id)


def gripper(request_id,width,observation_id):
    return decision('set_gripper',{'request_id':request_id,'width_m':width,'observation_id':observation_id},request_id)


def finish(observation_id,ident='finish-call'):
    return decision('finish_task',{'summary':'Observed result','uncertainties':[],'observation_id':observation_id},ident)


def progress_of(payload):
    text=payload['messages'][-1]['content']
    return json.loads(text.split('CURRENT execution_progress: ',1)[1])


def run_loop(tmp_path,monkeypatch,args,evidence,session,responses,payloads):
    monkeypatch.setattr(online,'paid',driver(responses,payloads))
    async def state(*values):
        return (session.world.read(session.time) if hasattr(session,'world')
                else {'sim_time_s':session.time,'objects':{},'bilateral_contact_objects':[]})
    monkeypatch.setattr(online,'remote',state)
    report={};events=[]
    async def go():
        return await online.online_loop(session,args,evidence,events.append,tmp_path,report,None)
    return go,report,events


# --- requirement 1: soft plan policy ------------------------------------------

@pytest.mark.asyncio
async def test_plan_on_motion_and_finish_need_no_plan_write(tmp_path,monkeypatch):
    args,evidence=fixture(tmp_path);session=Session();payloads=[]
    responses=[motion('move_to','motion-0001','live-0001'),finish('live-0002')]
    go,report,events=run_loop(tmp_path,monkeypatch,args,evidence,session,responses,payloads)
    await go()
    assert report['status']=='finished' and report['motions']==1
    assert report['plan_reads']==0 and report['plan_writes']==0
    assert report['plan_sync_reads']==3 and report['api_calls']==2 and report['tool_calls']==2
    assert report['plan_policy']=='soft' and report['plan_protocol_complete'] is None
    assert len([c for c in session.calls if c[0]=='task_plan'])==4   # create + one sync per request + post-action sync
    assert [progress_of(p)['current_plan']['version'] for p in payloads]==[1,1]
    assert all(progress_of(p)['plan_policy'].startswith('soft') for p in payloads)


@pytest.mark.asyncio
async def test_plan_sync_refreshes_real_version_without_extra_api_or_tool_rounds(tmp_path,monkeypatch):
    args,evidence=fixture(tmp_path);session=Session();payloads=[]
    responses=[decision('task_plan',plan_update(1),'plan-update'),motion('move_to','motion-0001','live-0001'),finish('live-0002')]
    go,report,events=run_loop(tmp_path,monkeypatch,args,evidence,session,responses,payloads)
    await go()
    assert report['plan_writes']==1 and report['plan_reads']==0
    assert report['plan_sync_reads']==4 and report['api_calls']==3 and report['tool_calls']==3
    assert [progress_of(p)['current_plan']['version'] for p in payloads]==[1,2,2]
    assert session.plan['version']==2
    assert len([c for c in session.calls if c[0]=='task_plan'])==6   # create + 3 decision syncs + post-action sync + model update


@pytest.mark.asyncio
@pytest.mark.parametrize('bad_sync',[
    None,
    {'task_id':'foreign','version':2},
    {'task_id':'aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa','version':0},
    {'task_id':'aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa','version':True},
    {'error':{'message':'read unavailable'}},
])
async def test_invalid_plan_sync_stops_before_paid_request_or_motion(tmp_path,monkeypatch,bad_sync):
    args,evidence=fixture(tmp_path);session=Session();payloads=[]
    original=session.call_tool
    async def broken_read(name,body):
        if name=='task_plan' and body['operation']=='get':
            return SimpleNamespace(isError=False,structuredContent=bad_sync,content=[])
        return await original(name,body)
    monkeypatch.setattr(session,'call_tool',broken_read)
    go,report,_=run_loop(tmp_path,monkeypatch,args,evidence,session,[],payloads)
    with pytest.raises(online.ModelCallAnomaly,match='sync failed'):
        await go()
    assert payloads==[] and session.moves==[]
    assert report['plan_sync_reads']==0


@pytest.mark.asyncio
async def test_plan_off_exposes_no_state_or_tools(tmp_path,monkeypatch):
    args,evidence=fixture(tmp_path);args.plan='off';session=Session();payloads=[]
    responses=[motion('move_to','motion-0001','live-0001'),finish('live-0002')]
    go,report,events=run_loop(tmp_path,monkeypatch,args,evidence,session,responses,payloads)
    await go()
    assert report['status']=='finished' and report['motions']==1
    assert report['plan_sync_reads']==0 and report['plan_policy'] is None and report['plan_protocol_complete'] is None
    for payload in payloads:
        assert 'task_plan' not in {t['function']['name'] for t in payload['tools']}
        assert 'task_plan' not in json.dumps(payload['messages'])
        assert 'current_plan' not in json.dumps(payload['messages'])
        assert 'plan_policy' not in json.dumps(payload['messages'])
    context=json.loads((tmp_path/'initial-model-context.json').read_text())
    assert context['plan'] is None and context['plan_enabled'] is False
    assert len([c for c in session.calls if c[0]=='task_plan'])==1   # private empty container only


@pytest.mark.asyncio
async def test_budget_is_checked_after_plan_sync_and_before_paid_dispatch(tmp_path,monkeypatch):
    import asyncio
    args,evidence=fixture(tmp_path);session=Session();dispatched=[]
    args.max_seconds=.5
    slow=session.call_tool
    async def call_tool(name,body):
        if name=='task_plan' and body.get('operation')=='get':await asyncio.sleep(.6)
        return await slow(name,body)
    monkeypatch.setattr(session,'call_tool',call_tool)
    async def paid(*values):dispatched.append(1);raise AssertionError('paid dispatch after the budget was exceeded')
    monkeypatch.setattr(online,'paid',paid)
    monkeypatch.setattr(online,'remote',lambda *values:{'sim_time_s':session.time,'objects':{}})
    report={}
    await online.online_loop(session,args,evidence,lambda row:None,tmp_path,report,None)
    assert report['status']=='time_budget' and report['api_calls']==0 and not dispatched
    assert report['plan_sync_reads']==1 and report['tool_calls']==0


@pytest.mark.asyncio
async def test_tool_budget_is_checked_before_any_model_tool_starts(tmp_path,monkeypatch):
    args,evidence=fixture(tmp_path);args.plan='off';args.max_tools=1;session=Session();payloads=[]
    reads=[{'id':'c1','function':{'name':'scene_query','arguments':json.dumps(
                {'operation':'surface_point','observation_path':'live-0001','pixel':[10,20]})}},
           {'id':'c2','function':{'name':'scene_query','arguments':json.dumps(
                {'operation':'surface_point','observation_path':'live-0001','pixel':[30,40]})}}]
    responses=[{'model':'deepseek-flash','usage':{'completion_tokens':100},'choices':[{'finish_reason':'tool_calls',
        'message':{'role':'assistant','content':'','tool_calls':reads}}]}]
    go,report,events=run_loop(tmp_path,monkeypatch,args,evidence,session,responses,payloads)
    await go()
    assert report['status']=='tool_budget' and report['tool_calls']==1 and report['api_calls']==1
    assert len([c for c in session.calls if c[0]=='scene_query'])==1
    assert [r['kind'] for r in events].count('tool_result')==1


@pytest.mark.asyncio
async def test_capture_label_exposes_only_safe_depth_filter_metadata(tmp_path,monkeypatch):
    args,evidence=fixture(tmp_path);session=Session();payloads=[]
    original=session.call_tool
    async def call_tool(name,body):
        result=await original(name,body)
        if name=='camera_observe':
            result.structuredContent['frames'][0]['depth_filter']={
                'applied':True,'method':'capture_aligned_geom_segmentation_robot_subtree','reason':'ok',
                'filtered_pixels':12,'image_pixels':640*480,'robot_body_count':4,'private_geom_path':'/private/x.stl'}
        return result
    monkeypatch.setattr(session,'call_tool',call_tool)
    responses=[motion('move_to','motion-0001','live-0001'),finish('live-0002')]
    go,report,events=run_loop(tmp_path,monkeypatch,args,evidence,session,responses,payloads)
    await go()
    labels=[e for e in events if e['kind']=='observation']
    assert len(labels)==2
    assert labels[0]['depth_filter']=={'applied':True,'method':'capture_aligned_geom_segmentation_robot_subtree',
        'reason':'ok','filtered_pixels':12,'image_pixels':640*480,'robot_body_count':4}
    assert 'private_geom_path' not in json.dumps([*labels,*payloads])


@pytest.mark.asyncio
async def test_geometry_unavailable_message_reports_support_and_assumptions(tmp_path,monkeypatch):
    args,evidence=fixture(tmp_path);session=Session();payloads=[];args.max_rounds=1
    original=session.call_tool
    async def call_tool(name,body):
        if name=='scene_query':
            return SimpleNamespace(isError=False,structuredContent={
                'status':'unavailable','reason':'No observed points above the selected support plane'},content=[])
        return await original(name,body)
    monkeypatch.setattr(session,'call_tool',call_tool)
    responses=[decision('grasp_target',{'observation_id':'live-0001','object_roi':[100,100,200,200],
                                        'table_roi':[0,0,639,479]},'geometry-call')]
    go,report,events=run_loop(tmp_path,monkeypatch,args,evidence,session,responses,payloads)
    await go()
    result=[e for e in events if e['kind']=='tool_result'][-1]['result']
    message=result['error']['message']
    assert result['error']['code']=='geometry_unavailable'
    assert 'Visible support' in message and 'masked' in message and 'assumption' in message
    assert 'tighter' not in message and 'No observed points' in message


# --- requirement 2: confirmed blocked vs unconfirmed/unknown ------------------

def blocked_result(*,confirmed=True,with_after=True):
    value={'status':'failed','error_code':'motion_blocked','recoverable':True,'simulation':True,
           'original_error_code':'motion_stalled','error':'Measured motion stopped progressing',
           'stop_result':{'status':'hold_requested','confirmed_stopped':confirmed,
                          'confirmation':{'samples':3,'settled_span_s':.3}}}
    if with_after:
        value['after']={'backend':'mujoco','robot':{'q_deg':[0.]*6,'gripper_width_m':.07,'velocity_deg_s':[0.]*6}}
    return value


@pytest.mark.asyncio
async def test_confirmed_block_gets_new_state_and_pixels_before_the_next_request(tmp_path,monkeypatch):
    args,evidence=fixture(tmp_path);world=ContactWorld()
    session=ScriptedSession(world,[{'result':blocked_result()},{'apply':lift(.03)}])
    payloads=[]
    responses=[motion('move_to','motion-0001','live-0001'),motion('move_to','motion-0002','live-0002',2),finish('live-0003')]
    go,report,events=run_loop(tmp_path,monkeypatch,args,evidence,session,responses,payloads)
    await go()
    assert report['status']=='finished' and report['motions']==1
    assert report['confirmed_blocked_primitives']==1 and report['unknown_outcome'] is False
    chain=json.loads((tmp_path/'action-chain.json').read_text())
    assert chain[0]['actual_result']['status']=='failed' and chain[0]['actual_result']['confirmed_blocked'] is True
    assert chain[0]['result_observation_id']=='live-0002' and chain[1]['result_observation_id']=='live-0003'
    second=json.dumps(payloads[1]['messages'])
    assert 'live-0002' in second
    blocked=[json.loads(m['content']) for m in payloads[1]['messages'] if m['role']=='tool']
    blocked=[m for m in blocked if m.get('error_code')=='motion_blocked']
    assert len(blocked)==1
    assert blocked[0]['confirmed_blocked'] is True and blocked[0]['recoverable'] is True
    assert blocked[0]['simulation'] is True and blocked[0]['stop_result']['confirmed_stopped'] is True
    assert blocked[0]['measured_after']['backend']=='mujoco'
    assert [m['request_id'] for m in session.moves]==['motion-0001','motion-0002']


@pytest.mark.asyncio
async def test_unconfirmed_motion_block_terminates_without_replay(tmp_path,monkeypatch):
    args,evidence=fixture(tmp_path);world=ContactWorld()
    session=ScriptedSession(world,[{'result':blocked_result(confirmed=False,with_after=False)}])
    payloads=[]
    go,report,events=run_loop(tmp_path,monkeypatch,args,evidence,session,[motion('move_to','motion-0001','live-0001')],payloads)
    with pytest.raises(online.ModelCallAnomaly,match='Unconfirmed'):
        await go()
    assert len(session.moves)==1 and len(payloads)==1
    assert len([c for c in session.calls if c[0]=='camera_observe'])==1
    assert len([c for c in session.calls if c[0]=='move_to'])==1
    chain=json.loads((tmp_path/'action-chain.json').read_text())
    assert chain[0]['actual_result']['error_code']=='motion_blocked'
    assert report['unknown_outcome'] is True


def test_confirmed_block_classification_requires_the_runtime_evidence_set():
    raw=blocked_result()
    assert online.confirmed_blocked(raw)
    projected=online.action_result(raw)
    assert projected['confirmed_blocked'] is True and projected['recoverable'] is True
    for missing in ('stop_result','after','simulation','recoverable'):
        broken=blocked_result();broken.pop(missing)
        assert not online.confirmed_blocked(broken),missing
    broken=blocked_result();broken['stop_result']['confirmed_stopped']=False
    assert not online.confirmed_blocked(broken)
    assert not online.confirmed_blocked({'status':'outcome_unknown','error_code':'motion_blocked'})


# --- requirement 3: carry bookkeeping -----------------------------------------

@pytest.mark.asyncio
async def test_grip_loss_attempt_and_later_regrasp_stay_separate(tmp_path,monkeypatch):
    args,evidence=fixture(tmp_path);world=ContactWorld()
    session=ScriptedSession(world,[{'apply':grasp()},{'apply':lift(.04)},{'apply':drop(.20)},
                                   {'apply':grasp()},{'apply':lift(.06)},{'apply':open_hand()}])
    payloads=[];sleeps=[]
    async def fake_sleep(seconds):sleeps.append(seconds)
    monkeypatch.setattr(online.asyncio,'sleep',fake_sleep)
    responses=[gripper('motion-0001',.02,'live-0001'),motion('move_to','motion-0002','live-0002',2),
               motion('move_to','motion-0003','live-0003',3),gripper('motion-0004',.02,'live-0004'),
               motion('move_to','motion-0005','live-0005',5),gripper('motion-0006',.07,'live-0006'),
               finish('live-0007')]
    args.max_rounds=len(responses)
    go,report,events=run_loop(tmp_path,monkeypatch,args,evidence,session,responses,payloads)
    transfers=await go()
    assert report['status']=='finished' and report['actual_transfer_count']==2
    assert [t['attempt'] for t in transfers]==[1,2] and [t['source'] for t in transfers]==['obj','obj']
    lost,released=transfers
    assert lost['end_reason']=='grip_lost' and lost['controlled_release'] is False and lost['incomplete'] is False
    assert lost['lift_height_m']==pytest.approx(.04)          # in-flight height after the loss is not lift
    assert lost['state_after']['bilateral_contact_objects']==[]
    assert released['end_reason']=='controlled_release' and released['controlled_release'] is True
    assert released['incomplete'] is False and released['lift_height_m']==pytest.approx(.06)
    assert released['state_before_release']['robot']['gripper_width_m']==pytest.approx(.02)
    assert released['state_after']['bilateral_contact_objects']==[]
    assert sleeps==[.6]                                        # .6 s wait before storing the released state
    saved=json.loads((tmp_path/'transfers.private.json').read_text())
    assert [t['attempt'] for t in saved]==[1,2]


@pytest.mark.asyncio
async def test_closing_induced_loss_is_a_failed_attempt_not_a_release(tmp_path,monkeypatch):
    args,evidence=fixture(tmp_path);world=ContactWorld()
    session=ScriptedSession(world,[{'apply':grasp()},{'apply':lift(.05)},{'apply':open_hand(width=.01)}])
    payloads=[]
    responses=[gripper('motion-0001',.02,'live-0001'),motion('move_to','motion-0002','live-0002',2),
               gripper('motion-0003',.01,'live-0003')]
    args.max_rounds=len(responses)
    go,report,events=run_loop(tmp_path,monkeypatch,args,evidence,session,responses,payloads)
    transfers=await go()
    assert report['actual_transfer_count']==1
    attempt=transfers[0]
    assert attempt['end_reason']=='grip_lost' and attempt['controlled_release'] is False
    assert 'state_before_release' not in attempt and attempt['incomplete'] is False


@pytest.mark.asyncio
async def test_carry_still_held_at_budget_end_is_retained_incomplete(tmp_path,monkeypatch):
    args,evidence=fixture(tmp_path);world=ContactWorld()
    session=ScriptedSession(world,[{'apply':grasp()},{'apply':lift(.04)}])
    payloads=[]
    responses=[gripper('motion-0001',.02,'live-0001'),motion('move_to','motion-0002','live-0002',2)]
    args.max_rounds=len(responses)
    go,report,events=run_loop(tmp_path,monkeypatch,args,evidence,session,responses,payloads)
    transfers=await go()
    assert report['status']=='round_budget' and report['api_calls']==len(responses)
    assert len(transfers)==1 and transfers[0]['incomplete'] is True
    assert transfers[0]['end_reason']=='incomplete' and transfers[0]['controlled_release'] is False
    assert transfers[0]['lift_height_m']==pytest.approx(.04)
    assert json.loads((tmp_path/'transfers.private.json').read_text())[0]['incomplete'] is True


@pytest.mark.asyncio
async def test_interrupted_carry_is_still_on_disk_for_the_private_scorer(tmp_path,monkeypatch):
    args,evidence=fixture(tmp_path);world=ContactWorld()
    session=ScriptedSession(world,[{'apply':grasp()},{'apply':lift(.04)},
                                   {'result':{'status':'outcome_unknown','request_id':'motion-0003'}}])
    payloads=[]
    responses=[gripper('motion-0001',.02,'live-0001'),motion('move_to','motion-0002','live-0002',2),
               motion('move_to','motion-0003','live-0003',3)]
    go,report,events=run_loop(tmp_path,monkeypatch,args,evidence,session,responses,payloads)
    with pytest.raises(online.ModelCallAnomaly,match='unknown'):
        await go()
    saved=json.loads((tmp_path/'transfers.private.json').read_text())
    assert len(saved)==1 and saved[0]['incomplete'] is True and saved[0]['lift_height_m']==pytest.approx(.04)


def test_load_transfers_recovers_partial_artifact_without_dropping_records(tmp_path):
    assert online.load_transfers(tmp_path,[{'attempt':9}])==[{'attempt':9}]
    (tmp_path/'transfers.private.json').write_text(json.dumps([{'attempt':1,'incomplete':True},{'attempt':2}]))
    assert online.load_transfers(tmp_path,[])==[{'attempt':1,'incomplete':True},{'attempt':2}]
    (tmp_path/'transfers.private.json').write_text(json.dumps([{'attempt':1}]))
    assert online.load_transfers(tmp_path,[{'attempt':1},{'attempt':2}])==[{'attempt':1},{'attempt':2}]


# --- requirement 4: final/partial state and grading after close ---------------

class MainSession(ScriptedSession):
    async def initialize(self):return SimpleNamespace(serverInfo=SimpleNamespace(version='test-cli'))
    async def call_tool(self,name,args):
        if name=='robot_connect':return SimpleNamespace(isError=False,structuredContent={'status':'connected'},content=[])
        if name=='robot_status':return SimpleNamespace(isError=False,structuredContent={'backend':'mujoco','ready':True},content=[])
        return await super().call_tool(name,args)


class Manager:
    def __init__(self,session):self.session=session;self.closed=False
    async def __aenter__(self):return self.session
    async def __aexit__(self,*exc):self.closed=True;return False


class FakeStdio:
    def __init__(self,*args,**kwargs):pass
    async def __aenter__(self):return (object(),object())
    async def __aexit__(self,*exc):return False


class FakeHttp:
    def __init__(self,*args,**kwargs):pass
    async def __aenter__(self):return self
    async def __aexit__(self,*exc):return False


def main_scene(tmp_path,monkeypatch,name,script,responses,grade):
    """Shared main() harness: fake MCP/API transport, real loop and artifacts."""
    _,evidence=fixture(tmp_path);world=ContactWorld()
    out=tmp_path/name;session=MainSession(world,script);manager=Manager(session)
    monkeypatch.setattr(online,'ClientSession',lambda *a,**k:manager)
    monkeypatch.setattr(online,'stdio_client',FakeStdio)
    monkeypatch.setattr(online.httpx,'AsyncClient',FakeHttp)
    monkeypatch.setenv('DEEPSEEK_API_KEY','test-secret')
    payloads=[]
    monkeypatch.setattr(online,'paid',driver(responses,payloads))
    score_closed=[]
    async def remote(args,operation,root):
        if operation=='score':
            score_closed.append(manager.closed)
            return {**world.read(session.time),'private_final_marker':True}
        if operation=='check':return {'runtime_remaining':False,'owned_process_alive':False,'recording_error':None}
        raise AssertionError(operation)
    monkeypatch.setattr(online,'remote',remote)
    monkeypatch.setattr(online,'private_grade',grade)
    annotation=tmp_path/(name+'-reference.json')
    annotation.write_text(json.dumps({'code':["pos = getpos('obj')","moveto(pos)","pick()",
                                              "ref = getpos('ref','left')","moveto(ref)","place()"]}))
    evidence_file=tmp_path/(name+'-evidence.json')
    evidence_file.write_text(json.dumps({'task':evidence}))
    main_args=SimpleNamespace(output=out,annotation=annotation,evidence=evidence_file,socket=tmp_path/'socket',
        host='host',remote_root='/remote',remote_python='python',scene='scene',task='task',
        object_list=json.dumps(['obj','ref']),port=54979,effort='high',plan='on',skill='on',
        decision_policy='standard',call_completion_cap=65536,interface_smoke=False,contact_smoke=False,
        native_probe=False,max_rounds=4,max_tools=10,max_seconds=60,completion_budget=65536)
    return main_args,manager,session,payloads,score_closed


INTERRUPTED_SCRIPT=[{'apply':grasp()},{'apply':lift(.04)},
                    {'result':{'status':'outcome_unknown','request_id':'motion-0003'}}]
INTERRUPTED_RESPONSES=[gripper('motion-0001',.02,'live-0001'),motion('move_to','motion-0002','live-0002',2),
                       motion('move_to','motion-0003','live-0003',3)]


@pytest.mark.asyncio
async def test_interrupted_loop_grades_partial_attempt_after_session_close(tmp_path,monkeypatch):
    seen={};box={}
    def grade(gt,transfers,final_state=None,unknown_outcome=None,actions=None):
        seen.update(gt=gt,transfers=copy.deepcopy(transfers),final_state=final_state,
                    unknown_outcome=unknown_outcome,closed=box['manager'].closed)
        return {'success':False,'goal_success':False,'ordered_success':False,'steps':[]}
    main_args,manager,_,_,score_closed=main_scene(tmp_path,monkeypatch,'run',INTERRUPTED_SCRIPT,INTERRUPTED_RESPONSES,grade)
    box['manager']=manager
    out=main_args.output
    await online.main(main_args)
    assert manager.closed is True
    # GT/scoring happens only after the MCP session closed; final state was captured live.
    assert seen['closed'] is True and score_closed[-1] is False
    assert seen['gt']==[('pick_place','obj','ref','left')]
    assert seen['final_state']['private_final_marker'] is True
    assert seen['unknown_outcome'] is True
    assert len(seen['transfers'])==1 and seen['transfers'][0]['incomplete'] is True
    assert seen['transfers'][0]['lift_height_m']==pytest.approx(.04)
    report=json.loads((out/'report.json').read_text())
    assert report['status']=='paused' and report['final_state_captured'] is True
    assert report['graded'] is True and report['physical_success'] is False
    assert report['goal_coverage_success'] is False and report['ordered_success'] is False
    assert report['unknown_outcome'] is True
    assert report['exceptions'][0]['type']=='ExecutionOutcomeUnknown' and 'unknown' in report['exceptions'][0]['message']
    assert (out/'transfers.private.json').exists() and (out/'final-state.private.json').exists()
    assert (out/'physics-score.private.json').exists()


@pytest.mark.asyncio
async def test_closeout_failure_never_masks_the_original_halt_cause(tmp_path,monkeypatch):
    def grade(gt,transfers,final_state=None,unknown_outcome=None,actions=None):
        raise TypeError("private_grade() got an unexpected keyword argument 'final_state'")
    main_args,manager,_,_,_=main_scene(tmp_path,monkeypatch,'run2',INTERRUPTED_SCRIPT,INTERRUPTED_RESPONSES,grade)
    out=main_args.output
    await online.main(main_args)
    report=json.loads((out/'report.json').read_text())
    assert report['status']=='paused' and report['graded'] is False
    assert report['exceptions'][0]['type']=='ExecutionOutcomeUnknown' and 'unknown' in report['exceptions'][0]['message']
    assert report['closeout_error'][0]['type']=='TypeError'
    assert (out/'transfers.private.json').exists() and (out/'final-state.private.json').exists()


@pytest.mark.asyncio
async def test_native_probe_never_reads_gt_or_invokes_the_scorer(tmp_path,monkeypatch):
    def forbidden_grade(*args,**kwargs):
        raise AssertionError('A camera/interface probe must not load or score GT')
    args,manager,_,payloads,score_calls=main_scene(
        tmp_path,monkeypatch,'probe',[],[],forbidden_grade)
    args.native_probe=True
    args.annotation=tmp_path/'does-not-exist-reference.json'
    await online.main(args)
    report=json.loads((args.output/'report.json').read_text())
    assert report['status']=='native_probe_pass'
    assert manager.closed and payloads==[] and score_calls==[]
    assert 'closeout_error' not in report and 'graded' not in report
    assert not (args.output/'physics-score.private.json').exists()


@pytest.mark.asyncio
async def test_changed_post_action_camera_identity_marks_unknown_and_prevents_next_decision(tmp_path,monkeypatch):
    def grade(gt,transfers,final_state=None,unknown_outcome=None,actions=None):
        assert unknown_outcome is True
        return {'success':False,'goal_success':False,'ordered_success':False,'steps':[]}
    args,manager,session,payloads,_=main_scene(tmp_path,monkeypatch,'changed-camera',
        [{}],[motion('move_to','motion-0001','live-0001')],grade)
    original=session.call_tool
    captures=0
    async def changed_camera(name,body):
        nonlocal captures
        reply=await original(name,body)
        if name=='camera_observe':
            captures+=1
            if captures==2:reply.structuredContent['frames'][0]['instance_id']='replacement-simulator'
        return reply
    monkeypatch.setattr(session,'call_tool',changed_camera)
    await online.main(args)
    report=json.loads((args.output/'report.json').read_text())
    assert manager.closed and report['status']=='paused' and report['unknown_outcome'] is True
    assert len(session.moves)==1 and len(payloads)==1 and captures==2
    assert report['physical_success'] is False


# --- requirement 5: shared contract across the eight conditions ---------------

def test_eight_conditions_share_prompt_tools_and_budget(tmp_path):
    native=[tool('move_to',{'type':'object','properties':{}}).model_dump(),tool('camera_observe',{'type':'object'}).model_dump(),
            tool('task_plan',TASK_PLAN_TOOL['inputSchema']).model_dump(),tool('scene_query',{'type':'object'}).model_dump()]
    common=set();skill=set();plan=set();shared=set();plan_only=set()
    for p,s,e in itertools.product(('off','on'),('off','on'),('low','high')):
        _,recipe=online.prompt_recipe(p=='on',s=='on')
        common.add(recipe['common_sha256']);skill.add(recipe['skill_sha256']);plan.add(recipe['plan_sha256'])
        listed=online.select_tools(native,p=='on')
        shared.add(json.dumps([t for t in listed if t['name']!='task_plan'],sort_keys=True))
        plan_tools=[t for t in listed if t['name']=='task_plan']
        assert bool(plan_tools)==(p=='on')
        if p=='on':plan_only.add(json.dumps(plan_tools,sort_keys=True))
    assert len(common)==1 and len(shared)==1 and len(plan_only)==1
    assert skill=={None,online.prompt_recipe(False,True)[1]['skill_sha256']}
    assert plan=={None,online.prompt_recipe(True,False)[1]['plan_sha256']}
    assert pilot.BUDGET=={'completion_budget':65536,'max_rounds':80,'max_tools':120,'max_seconds':1800}
    cases=pilot.schedule()
    assert len(cases)==24 and all('budget' not in c for c in cases)
    job=tmp_path/'job';job.mkdir()
    (job/'tasks.json').write_text(json.dumps({task:{'annotation':'reference.json','objects':['obj','ref'],
        'scene':'scene.xml'} for task in {case['task'] for case in cases}}))
    config={'python':'python','socket':'/tmp/socket','host':'host','port':54979,'node':'node',
        'key_launcher':'key-launcher','remote_root':'/remote','remote_python':'python','budget':pilot.BUDGET}
    budgets=set()
    for case in cases:
        cmd=pilot.command(job,config,case,str(tmp_path/'run'))
        flags={cmd[i]:cmd[i+1] for i in range(len(cmd)-1) if cmd[i].startswith('--max-') or cmd[i]=='--completion-budget'}
        budgets.add(json.dumps(flags,sort_keys=True))
    assert len(budgets)==1
    assert json.loads(budgets.pop())=={'--completion-budget':'65536','--max-rounds':'80','--max-tools':'120','--max-seconds':'1800'}
