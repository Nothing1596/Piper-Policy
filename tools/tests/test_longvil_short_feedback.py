"""Short decisions use the same execution gate; doubles are not physics evidence."""
import copy
import json
from types import SimpleNamespace

import pytest

from test_longvil_online import Session,fixture
import run_longvil_online as online
import run_longvil_online_pilot as pilot


def reply(name=None,body=None,*,tokens=100,finish='tool_calls',ident='c'):
    calls=[] if name is None else [{'id':ident,'function':{'name':name,'arguments':json.dumps(body)}}]
    return {'model':'deepseek-flash','usage':{'completion_tokens':tokens},
            'choices':[{'finish_reason':finish,'message':{'role':'assistant','content':'','tool_calls':calls}}]}


async def measured(session,*args):
    return {'sim_time_s':session.time,'objects':{},'bilateral_contact_objects':[]}


@pytest.mark.asyncio
async def test_no_tool_cap_truncation_is_charged_and_gets_one_fresh_decision(tmp_path,monkeypatch):
    args,evidence=fixture(tmp_path);args.plan='off';args.effort='low';args.decision_policy='short-feedback'
    args.call_completion_cap=8192;args.max_rounds=3
    session=Session();payloads=[];events=[]
    responses=[reply(tokens=8192,finish='length'),
               reply('move_to',{'request_id':'motion-0001','xyz_m':[.3,.1,.2],'observation_id':'live-0002'}),
               reply('finish_task',{'summary':'Observed','uncertainties':[],'observation_id':'live-0003'},ident='finish')]
    async def paid(*v):payloads.append(copy.deepcopy(v[2]));return responses[len(payloads)-1]
    monkeypatch.setattr(online,'paid',paid);monkeypatch.setattr(online,'remote',lambda *a:measured(session))
    report={};await online.online_loop(session,args,evidence,events.append,tmp_path,report,None)
    assert report['status']=='finished' and report['motions']==1 and report['observations']==3
    assert report['completion_tokens']==8392 and report['truncation_continuations']==1
    assert all(p['max_tokens']==8192 and p['reasoning_effort']=='low' for p in payloads)
    assert not any(m['role']=='assistant' for m in payloads[1]['messages'])
    assert len(session.moves)==1 and len([e for e in events if e['kind']=='execution_result'])==1


@pytest.mark.parametrize('policy,calls,total',[
    ('standard',False,65536),('short-feedback',True,65536),('short-feedback',False,8192)])
@pytest.mark.asyncio
async def test_truncated_tools_and_total_budget_never_continue_or_dispatch(policy,calls,total,tmp_path,monkeypatch):
    args,evidence=fixture(tmp_path);args.plan='off';args.effort='low';args.decision_policy=policy
    args.call_completion_cap=8192;args.completion_budget=total
    session=Session();api=[]
    async def paid(*v):
        api.append(v[2]);return reply('move_to' if calls else None,{'request_id':'motion-0001'},tokens=8192,finish='length')
    monkeypatch.setattr(online,'paid',paid);monkeypatch.setattr(online,'remote',lambda *a:measured(session))
    report={};await online.online_loop(session,args,evidence,lambda e:None,tmp_path,report,None)
    assert report['status']=='truncated' and report['completion_tokens']==8192
    assert len(api)==1 and not session.moves and report['truncation_continuations']==0


@pytest.mark.asyncio
async def test_consecutive_truncation_stops_instead_of_looping(tmp_path,monkeypatch):
    args,evidence=fixture(tmp_path);args.plan='off';args.effort='low';args.decision_policy='short-feedback';args.call_completion_cap=8192
    session=Session();api=[]
    async def paid(*v):api.append(v[2]);return reply(tokens=8192,finish='length')
    monkeypatch.setattr(online,'paid',paid);monkeypatch.setattr(online,'remote',lambda *a:measured(session))
    report={};await online.online_loop(session,args,evidence,lambda e:None,tmp_path,report,None)
    assert len(api)==2 and report['completion_tokens']==16384 and report['status']=='truncated'
    assert report['truncation_continuations']==1 and not session.moves


@pytest.mark.asyncio
async def test_short_policy_never_replays_unknown_motion(tmp_path,monkeypatch):
    args,evidence=fixture(tmp_path);args.plan='off';args.effort='low';args.decision_policy='short-feedback';args.call_completion_cap=8192
    session=Session(unknown=True);api=[]
    async def paid(*v):
        api.append(v[2]);return reply('move_to',{'request_id':'motion-0001','xyz_m':[.3,.1,.2],'observation_id':'live-0001'})
    monkeypatch.setattr(online,'paid',paid);monkeypatch.setattr(online,'remote',lambda *a:measured(session))
    with pytest.raises(online.ModelCallAnomaly,match='unknown'):
        await online.online_loop(session,args,evidence,lambda e:None,tmp_path,{},None)
    assert len(api)==len(session.moves)==1


def test_shared_gripper_guidance_does_not_rewrite_parameters():
    schema={'type':'object','properties':{'width_m':{'type':'number'},'completion':{'enum':['width','bilateral_contact'],'default':'width'}}}
    native=[{'name':'set_gripper','description':'gripper','inputSchema':schema}]
    before=copy.deepcopy(native)
    t=online.select_tools(native)[0]
    assert 'completion="bilateral_contact"' in t['description'] and native==before
    assert t['inputSchema']['properties']['completion']==schema['properties']['completion']


def test_policy_hint_is_separate_from_skill_and_pairs_hold_other_budgets():
    _,a=online.prompt_recipe(True,True);_,b=online.prompt_recipe(True,True,'short-feedback')
    assert a['skill_text']==b['skill_text'] and a['plan_text']==b['plan_text']
    assert not a['decision_hint'] and b['decision_hint']
    specs=pilot.feedback_conditions();cases=pilot.schedule(specs)
    assert len(cases)==6 and len({c['id'] for c in cases})==6
    assert all(c['effort']=='low' and c['plan']==c['skill']=='on' for c in cases)
    assert {c['call_completion_cap'] for c in specs}=={8192,65536}
    rows=[dict(c,physical_success=c['decision_policy']=='short-feedback',api_calls=2,motions=1,
               completion_tokens=100,reasoning_tokens=None,elapsed_s=2) for c in cases if c['phase']=='formal']
    result=pilot.feedback_effects(rows)
    assert result['complete_task_blocks']==2
    assert all(v['physical_success']==1 and 'reasoning_tokens' not in v for v in result['paired_differences'].values())
