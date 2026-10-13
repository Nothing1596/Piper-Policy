"""Physical runner evidence parity, protocol and abnormal-call boundaries."""
import copy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0,str(Path(__file__).parents[1]))
import run_longvil_physics as runner
from longvil_budget import Ledger,BudgetPause
from longvil_physics_execute import Player,PhysicsExecutionError
from test_run_plan_enforcement_pilot import Session,fixture
from test_plan_protocol import CODE,interpretation


@pytest.mark.asyncio
@pytest.mark.parametrize('mode',['R0','R1','R2'])
@pytest.mark.parametrize('effort',['low','high'])
async def test_shared_five_stages_and_recorded_sim_reference(tmp_path,monkeypatch,mode,effort):
    args,item,evidence,_=fixture(tmp_path);item.update(condition=mode,effort=effort)
    session=Session();calls=[]
    picture=tmp_path/'sim.jpg';picture.write_bytes(b'fixed initial simulation')
    reference={'image_path':str(picture),'sha256':hashlib.sha256(picture.read_bytes()).hexdigest(),
               'scene_sha256':'scene','sim_time_s':1.}
    report={'session_id':'session','api_attempts':0,'api_calls':0,'api_time_s':0.,'usage':{},
            'reasoning_tokens':0,'tool_attempts':0,'tool_calls':0,'interface_errors':0,'errors':0,'status':'running'}
    async def paid(args,client,payload,rid,phase):
        turn=len(calls);calls.append(copy.deepcopy(payload));msg={'role':'assistant','content':'brief evidence notes'}
        if mode!='R0' and turn<4:
            op=['get','update','update','get'][turn];body={'operation':op,'task_id':session.plan['task_id']}
            if op=='update':body.update(expected_version=session.plan['version'],changes={'interpretation':interpretation()})
            msg['tool_calls']=[{'id':'call-'+str(turn),'function':{'name':'task_plan','arguments':json.dumps(body)}}]
        if turn==4:msg['tool_calls']=[{'id':'submit','function':{'name':'submit_program','arguments':json.dumps({'code':CODE,'summary':'evidence','uncertainties':[]})}}]
        return {'model':'deepseek-flash','choices':[{'message':msg,'finish_reason':'stop'}],
                'usage':{'completion_tokens':50,'prompt_tokens':10,'total_tokens':60,'prompt_cache_hit_tokens':0,'prompt_cache_miss_tokens':10}}
    monkeypatch.setattr(runner,'paid_call',paid)
    candidate,accepted=await runner.generate(args,item,None,session,args.output,['cube','box'],
        evidence[item['task']],reference,lambda row:None,report)
    assert accepted and candidate['code']==CODE and len(calls)==5
    assert [runner.count_message_images(p['messages']) for p in calls]==[3,3,5,5,5]
    assert all(p['reasoning_effort']==effort and p['thinking']=={'type':'enabled'} for p in calls)
    visible='\n'.join(part['text'] for m in calls[0]['messages'] if isinstance(m['content'],list)
                      for part in m['content'] if part['type']=='text')
    assert 'simulation_initial_reference' in visible and '"is_live": false' in visible
    assert str(tmp_path) not in visible and 'private-level3-task' not in visible
    assert '"frame_id": "f2"' not in visible
    if mode=='R2':assert report['adherent'] and report['phase_success']==[True]*4


def test_disabling_cost_gate_keeps_unknown_usage_and_duplicate_detection(tmp_path):
    from datetime import datetime,timezone
    ledger=Ledger(tmp_path/'usage.jsonl',limit=None,stop_at=None)
    stamp=datetime.now(timezone.utc)
    usage={'prompt_cache_hit_tokens':0,'prompt_cache_miss_tokens':10_000_000,'completion_tokens':10_000_000}
    ledger.record(request_id='expensive',phase='test',usage=usage,start=stamp,end=stamp)
    ledger.check()
    with pytest.raises(ValueError,match='Duplicate'):ledger.record(request_id='expensive',phase='test',usage=usage,start=stamp,end=stamp)
    with pytest.raises(BudgetPause):ledger.record(request_id='missing',phase='test',usage=None,start=stamp,end=stamp)
    with pytest.raises(BudgetPause):ledger.check()


@pytest.mark.asyncio
async def test_unknown_sdk_result_is_never_reissued():
    class MCP:
        def __init__(self):self.calls=[]
        async def call_tool(self,name,args):
            self.calls.append((name,args));return SimpleNamespace(isError=False,structuredContent={'status':'outcome_unknown'})
    session=MCP();p=Player(session,None,None,lambda row:None,'unique',{'base_from_dataset':__import__('numpy').eye(4).tolist()})
    with pytest.raises(PhysicsExecutionError):await p.call('move_to',{'xyz_m':[.3,0,.1]})
    assert len(session.calls)==1 and session.calls[0][1]['request_id']=='unique-0001'


@pytest.mark.asyncio
async def test_empty_gripper_return_uses_measured_route_and_stops_at_unknown():
    class MCP:
        def __init__(self):self.calls=[]
        async def call_tool(self,name,args):
            self.calls.append((name,args))
            return SimpleNamespace(isError=False,structuredContent={'status':'succeeded' if len(self.calls)==1 else 'outcome_unknown'})
    session=MCP();p=Player(session,None,None,lambda row:None,'return',{'base_from_dataset':__import__('numpy').eye(4).tolist()})
    route=[[0.,30.,-30.,0.,0.,0.],[20.,90.,-50.,40.,0.,-70.],[40.,80.,-40.,50.,0.,-50.]]
    with pytest.raises(PhysicsExecutionError):await p.return_route(route)
    assert [c[1]['joints_deg'] for c in session.calls]==list(reversed(route))[0:2]
    assert all(c[0]=='robot_move_joints' for c in session.calls)


def test_reference_image_hash_is_checked_and_never_claimed_live(tmp_path):
    p=tmp_path/'sim.jpg';p.write_bytes(b'old')
    row={'image_path':str(p),'sha256':hashlib.sha256(b'old').hexdigest(),'scene_sha256':'scene','sim_time_s':2.}
    _,label=runner.simulation_reference(row);assert label['is_live'] is False
    p.write_bytes(b'new')
    with pytest.raises(ValueError,match='changed'):runner.simulation_reference(row)


@pytest.mark.parametrize('change',[{'model':'other-family'},
    {'usage':{'prompt_tokens':10,'completion_tokens':2,'total_tokens':99}},
    {'usage':{'prompt_tokens':10,'completion_tokens':2000,'total_tokens':2010}},
    {'choices':[]}])
def test_abnormal_model_response_is_refused_before_next_request(change):
    data={'model':'deepseek-flash','usage':{'prompt_tokens':10,'completion_tokens':2,'total_tokens':12},
          'choices':[{'message':{'role':'assistant','content':'ok'}}]}
    data.update(change)
    with pytest.raises(runner.ModelCallAnomaly):runner.check_response(data,100)


def test_explicit_resume_does_not_implicitly_repurchase_an_incomplete_session(tmp_path):
    p=tmp_path/'session';p.mkdir();(p/'attempt.json').write_text(json.dumps({'episode_key':'paid-incomplete'}))
    with pytest.raises(BudgetPause,match='retry-incomplete'):runner.check_resume(tmp_path,set())
    runner.check_resume(tmp_path,{'paid-incomplete'})
    runner.check_resume(tmp_path,set(),True)
