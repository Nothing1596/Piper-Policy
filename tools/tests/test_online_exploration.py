"""Exploration opt-in: real feedback changes and concurrent tool dispatch."""
import asyncio
import copy
import json
from types import SimpleNamespace
import pytest
from test_longvil_online import Session, fixture, tool
import run_longvil_online as online
from agy_online_decider import AgyDecider
from session_context import SessionContext


def signal(held):
    return {'q_deg':[0]*6,'gripper_width_m':.05 if held else 0.,
            'diagnostics':{'gripper_contact':{'supported':held,'finger1_force_n':15. if held else 0.,'finger2_force_n':15. if held else 0.}}}


class FeedbackSession(Session):
    def __init__(self):
        super().__init__();self.held=True;self.inflight=0;self.overlap=0
    async def list_tools(self):
        result=await super().list_tools()
        result.tools += [tool('robot_status',{'type':'object'}),tool('robot_diagnostics',{'type':'object'}),tool('robot_connect',{'type':'object'})]
        return result
    async def call_tool(self,name,args):
        if name in ('robot_status','robot_diagnostics'):
            self.calls.append((name,copy.deepcopy(args)))
            self.inflight+=1;self.overlap=max(self.overlap,self.inflight)
            await asyncio.sleep(.001)
            self.inflight-=1
            return SimpleNamespace(structuredContent={'backend':'mujoco','robot':signal(self.held)},content=[],isError=False)
        res=await super().call_tool(name,args)
        if name=='camera_observe':res.structuredContent['frames'][0]['robot_alignment']['state']['robot']=signal(self.held)
        return res


def response(calls,plan='Keep observed progress; choose next action'):
    return {'model':'test','plan_update':plan,'usage':{'completion_tokens':70000},'choices':[{'finish_reason':'tool_calls','message':{
        'role':'assistant','content':'observed','tool_calls':[{'id':str(i),'function':{'name':n,'arguments':json.dumps(a)}} for i,(n,a) in enumerate(calls)]}}]}


@pytest.mark.asyncio
async def test_contact_lost_during_decision_refreshes_without_stale_action_or_plan(tmp_path):
    args,evidence=fixture(tmp_path);args.exploration=True;args.completion_budget=None;args.max_rounds=2;args.max_seconds=None
    sess=FeedbackSession();events=[];payloads=[];cached=[]
    cache_root=tmp_path/'public';cache_root.mkdir()
    context=SessionContext(cache_root)
    async def decide(a,c,p,i):
        payloads.append(p)
        context.archive(len(payloads)-1,[],[],[],{})
        if len(payloads)==1:
            sess.held=False
            return response([('move_to',{'request_id':'stale-move','xyz_m':[.3,0,.2],'observation_id':'live-0001'})],'OLD still holding')
        assert 'PUBLIC GRIP FEEDBACK CHANGED' in json.dumps(p)
        assert 'live-0002' in json.dumps(p)
        return response([('robot_status',{})],'Dropped object: reobserve and regrasp')
    def cache_result(turn,row):
        cached.append(row)
        context.record_tool_result(turn,row)
    decide.record_tool_result=cache_result
    args.model_decider=decide
    report={};await online.online_loop(sess,args,evidence,events.append,tmp_path,report,None)
    assert not sess.moves
    assert report['stale_decisions_returned']==1 and report['grip_feedback_refreshes']==1
    assert report['model_authored_plan_updates']==1
    assert sess.plan['interpretation']=='Dropped object: reobserve and regrasp'
    assert report['completion_tokens']==140000
    cancelled=[r for r in cached if r['turn']==0]
    assert len(cancelled)==1 and cancelled[0]['name']=='move_to'
    assert cancelled[0]['result']['error']['code']=='grip_state_changed'
    assert cancelled[0]['result']['error']['executed'] is False
    assert report['tool_calls']==2 and report['model_tool_counts']['move_to']==1
    restored,_=context.query({'operation':'read','turn':0})
    assert restored['actual_tool_results']==cancelled
    assert (context.root/'turns/turn-000-results.json').exists()


@pytest.mark.asyncio
async def test_independent_native_calls_overlap_and_all_native_tools_exposed(tmp_path):
    args,evidence=fixture(tmp_path);args.exploration=True;args.max_rounds=1;args.completion_budget=None
    sess=FeedbackSession();events=[]
    async def decide(a,c,p,i):
        assert 'robot_connect' in {t['function']['name'] for t in p['tools']}
        assert 'task_plan' in {t['function']['name'] for t in p['tools']}
        assert 'Each executed motion refreshes observation_id' in p['messages'][0]['content']
        assert 'never silently rebound' in p['messages'][0]['content']
        return response([('robot_status',{}),('robot_diagnostics',{})])
    args.model_decider=decide
    report={};await online.online_loop(sess,args,evidence,events.append,tmp_path,report,None)
    assert sess.overlap==2
    assert report['tool_calls']==2
    assert len([e for e in events if e['kind']=='tool_result'])==2


@pytest.mark.asyncio
@pytest.mark.parametrize('external',[False,True])
async def test_unknown_usage_requires_explicit_external_assistant(tmp_path,external):
    args,evidence=fixture(tmp_path);args.exploration=True;args.max_rounds=1;args.completion_budget=None
    args.external_decision=external
    async def decide(a,c,p,i):
        data=response([('robot_status',{})]);data['usage']['completion_tokens']=None
        return data
    args.model_decider=decide
    report={};sess=FeedbackSession()
    if not external:
        with pytest.raises(online.ModelCallAnomaly,match='completion usage'):
            await online.online_loop(sess,args,evidence,lambda x:None,tmp_path,report,None)
    else:
        await online.online_loop(sess,args,evidence,lambda x:None,tmp_path,report,None)
        assert report['completion_tokens'] is None and report['reasoning_tokens'] is None
        assert report['tool_calls']==1


@pytest.mark.asyncio
async def test_agy_exploration_preserves_usage_but_does_not_reject_over_old_cap(tmp_path,monkeypatch):
    d=AgyDecider(tmp_path/'d',exploration=True);d.conversation='fresh'
    async def req(prompt):
        assert 'Remaining total output budget' not in prompt
        return dict(status='SUCCESS',conversation_id='fresh',num_turns=1,
            usage=dict(input_tokens=100,output_tokens=70000,thinking_tokens=65000,cache_read_tokens=0,total_tokens=70100),
            structured_output={'content':'check both','plan':'check','tool_calls':[{'name':'robot_status','arguments':{}},{'name':'robot_diagnostics','arguments':{}}]})
    monkeypatch.setattr(d,'request',req)
    result=await d(None,None,{'messages':[],'max_tokens':10,'tools':[{'function':{'name':n}} for n in ('robot_status','robot_diagnostics')]},'r')
    assert result['usage']['completion_tokens']==70000
    assert len(result['choices'][0]['message']['tool_calls'])==2


def test_feedback_signal_never_uses_private_object_labels():
    assert online.grip_signal({'objects':{'answer':{}}}) is None
    assert online.grip_changed(online.grip_signal(signal(True)),online.grip_signal(signal(False)))
    assert not online.grip_changed(None,online.grip_signal(signal(False)))


def test_exploration_keeps_native_limits_but_removes_experiment_constants():
    native=[{'name':'move_to','description':'move','inputSchema':{'type':'object','properties':{'speed_percent':{'type':'integer','minimum':1,'maximum':100}},'required':[]}}]
    spec=online.select_tools(native,exploration=True)[0]
    assert spec['inputSchema']['properties']['speed_percent']=={'type':'integer','minimum':1,'maximum':100}
    g=online.OnlineGate(concurrent_tools=True)
    calls=[{'id':str(i),'function':{'name':n}} for i,n in enumerate(['move_to','robot_status'])]
    assert g.batch(calls,0) is None
