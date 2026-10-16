import copy
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pytest
from agy_online_decider import AgyDecider, interrupted_generation


def failure():
    return dict(status='ERROR', conversation_id='broken', num_turns=2,
                error='API error: read: connection reset by peer', response='partial output',
                structured_output={'tool_calls':[{'name':'move_to','arguments':{'request_id':'discard'}}]},
                usage=dict(input_tokens=100,output_tokens=50,thinking_tokens=40,
                           cache_read_tokens=0,total_tokens=150))


@pytest.mark.asyncio
async def test_partial_candidate_is_saved_charged_but_never_returned(tmp_path,monkeypatch):
    d=AgyDecider(tmp_path/'decision',exploration=True)
    async def request(prompt):
        d.conversation='broken'
        return failure()
    monkeypatch.setattr(d,'request',request)
    data=await d(None,None,{'messages':[],'tools':[]},'request')
    assert data['usage']['completion_tokens']==50
    assert data['choices'][0]['message']['tool_calls']==[]
    assert 'plan_update' not in data and data['provider_interruption']['can_retry']
    assert json.loads((d.root/'turn-000-result.json').read_text())['response']=='partial output'
    assert json.loads((d.root/'turn-000-receipt.json').read_text())['usage_delta']['thinking_tokens']==40
    assert d.prior_message_count==0 and d.turn==1 and not d.seen_images


def test_only_transport_generation_errors_are_eligible():
    assert interrupted_generation(failure())
    assert interrupted_generation({**failure(),'error':'API error (attempt 1): UNAVAILABLE (code 503): The service is currently unavailable.'})
    for message in ['Eligibility check failed: code 503','authentication denied','invalid tool schema']:
        assert not interrupted_generation({**failure(),'error':message})


@pytest.mark.asyncio
async def test_credit_error_is_charged_without_retry_or_actions(tmp_path,monkeypatch):
    d=AgyDecider(tmp_path/'decision',exploration=True)
    async def request(prompt):
        d.conversation='broken'
        return {**failure(),'error':'Your AI credits balance is too low to continue.'}
    monkeypatch.setattr(d,'request',request)
    data=await d(None,None,{'messages':[],'tools':[]},'request')
    assert data['usage']['completion_tokens']==50
    assert data['provider_failure']['action_dispatched'] is False
    assert 'provider_interruption' not in data
    assert not data['choices'][0]['message']['tool_calls']


@pytest.mark.asyncio
async def test_unknown_usage_preserved_and_not_fabricated(tmp_path,monkeypatch):
    d=AgyDecider(tmp_path/'decision',exploration=True)
    async def request(prompt):
        d.conversation='broken'
        return {**failure(),'usage':None}
    monkeypatch.setattr(d,'request',request)
    with pytest.raises(ValueError,match='usage unavailable'):
        await d(None,None,{'messages':[],'tools':[]},'request')
    assert json.loads((d.root/'turn-000-result.json').read_text())['usage'] is None
    assert not list(d.root.glob('*receipt.json'))


@pytest.mark.asyncio
async def test_recovery_cap_does_not_erase_fourth_attempt_cost(tmp_path,monkeypatch):
    d=AgyDecider(tmp_path/'decision',exploration=True)
    d.generation_interruptions=3
    async def request(prompt):
        d.conversation='broken'
        return failure()
    monkeypatch.setattr(d,'request',request)
    data=await d(None,None,{'messages':[],'tools':[]},'request')
    assert not data['provider_interruption']['can_retry']
    assert data['usage']['completion_tokens']==50


@pytest.mark.asyncio
async def test_runner_refreshes_after_interruption_no_motion_replay(tmp_path,monkeypatch):
    from test_longvil_online import fixture,Session
    import run_longvil_online as online
    args,evidence=fixture(tmp_path);args.max_rounds=2
    session=Session();payloads=[];events=[]
    async def decide(args,client,payload,ident):
        payloads.append(copy.deepcopy(payload))
        if len(payloads)==1:
            return {'model':'test','usage':{'completion_tokens':50,
                       'completion_tokens_details':{'reasoning_tokens':40}},
                    'provider_interruption':{'can_retry':True,'action_dispatched':False},
                    'choices':[{'finish_reason':'provider_interrupted','message':{
                        'role':'assistant','content':'discarded','tool_calls':[]}}]}
        return {'model':'test','usage':{'completion_tokens':20},'choices':[{'message':{
            'role':'assistant','content':'read latest state','tool_calls':[{
                'id':'fresh','function':{'name':'robot_status','arguments':'{}'}}]}}]}
    async def sleep(seconds):pass
    monkeypatch.setattr(online,'paid',decide)
    monkeypatch.setattr(online.asyncio,'sleep',sleep)
    report={}
    await online.online_loop(session,args,evidence,events.append,tmp_path,report,None)
    assert len(payloads)==2 and not session.moves
    assert report['completion_tokens']==70 and report['generation_interruptions']==1
    assert report['observations']==2 and report['tool_calls']==1
    assert 'live-0002' in json.dumps(payloads[1])
    assert any(e['kind']=='generation_interrupted' for e in events)
