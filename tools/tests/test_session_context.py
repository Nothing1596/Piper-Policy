import base64
import hashlib
import io
import json
from pathlib import Path
import sys

import pytest
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from agy_online_decider import AgyDecider
from agy_observation_guard import permitted
from session_context import CONTEXT_TOOL


def pixels(color='red'):
    out=io.BytesIO();Image.new('RGB',(16,12),color).save(out,format='PNG')
    return {'type':'image_url','image_url':{'url':'data:image/png;base64,'+base64.b64encode(out.getvalue()).decode()}}


def initial():
    return [{'role':'system','content':'TASK CONTRACT, no private GT'},
            {'role':'user','content':[{'type':'text','text':'Public goal/object list'},
                                     {'type':'text','text':'Historical demonstration f0'},pixels()]},
            {'role':'user','content':[{'type':'text','text':'CURRENT simulation evidence live-0001'},pixels('blue')]},
            {'role':'user','content':'CURRENT execution_progress: {"current_plan":"carry", "latest_observation_id":"live-0001"}'}]


def next_turn(d,messages):
    d.prior_message_count=d.next_message_count;d.turn+=1
    d.seen_images.update(d.context.images[v]['path'] for v in d.context.images)
    messages.pop()  # transient progress
    # Simulate runner trimming all previous image bytes, without deleting archived pixels.
    for m in messages:
        if isinstance(m.get('content'),list):m['content']=[x for x in m['content'] if x['type']!='image_url']
    messages += [{'role':'tool','content':'ACTUAL FEEDBACK after motion'},
                 {'role':'user','content':[{'type':'text','text':'CURRENT simulation evidence live-0002'},pixels('green')]},
                 {'role':'user','content':'CURRENT execution_progress: {"current_plan":"release", "latest_observation_id":"live-0002"}'}]


def test_historical_image_survives_pruning_and_fresh_segment(tmp_path):
    d=AgyDecider(tmp_path/'d');messages=initial()
    tools=[{'function':{'name':'move_to','parameters':{'required':['xyz_m','request_id']}}}]
    _,required=d.prepare({'messages':messages,'tools':tools})
    old=next(m for m in d.context.images.values() if 'live-0001' in m['label'])
    d.context.record_decision(0,{'content':'chosen','plan':'carry','tool_calls':[]})
    d.record_tool_result(0,{'result':{'status':'succeeded'}})
    next_turn(d,messages)
    prompt,required=d.prepare({'messages':messages,'tools':tools})
    assert 'TASK CONTRACT' in prompt and 'Public goal/object list' in prompt
    assert 'ACTUAL FEEDBACK' in prompt and 'release' in prompt and 'xyz_m' in prompt
    assert old['path'] not in required
    allow=json.loads(d.state_path.read_text())['images']
    assert permitted({'name':'view_file','args':{'AbsolutePath':old['path']}},allow)[0]
    for path in d.cache_paths().values():assert permitted({'name':'view_file','args':{'AbsolutePath':path}},allow)[0]
    archive,_=d.context_call({'operation':'read','turn':0})
    assert archive['actual_tool_results'][0]['result']['status']=='succeeded'
    assert archive['decision']['model_candidate']['plan']=='carry'
    assert archive['scope'].startswith('HISTORICAL')
    # Text reads are exact, too; replacement cannot bypass the guard.
    Path(d.context.contract_path).write_text('tampered')
    assert not permitted({'name':'view_file','args':{'AbsolutePath':str(d.context.contract_path)}},allow)[0]


def test_crop_historical_pixels_preserves_origin_without_live_reclassification(tmp_path):
    d=AgyDecider(tmp_path/'d');d.prepare({'messages':initial(),'tools':[]})
    ident=next(iter(d.context.images));item=d.context.images[ident]
    receipt,parts=d.context_call({'operation':'crop','image_id':ident,'roi':[2,3,9,10]})
    assert receipt['source_sha256']==item['sha256']
    assert 'HISTORICAL' in parts[0]['text'] and 'CURRENT simulation' not in parts[0]['text']
    raw=base64.b64decode(parts[1]['image_url']['url'].split(',')[1])
    assert Image.open(io.BytesIO(raw)).size==(7,7)
    assert 'observation_id' not in receipt
    for body in [{'operation':'crop','image_id':'../../private.png','roi':[0,0,2,2]},
                 {'operation':'crop','image_id':ident,'roi':[0,0,99,99]},
                 {'operation':'read','turn':99}]:
        with pytest.raises(ValueError):d.context_call(body)
    Path(item['path']).write_bytes(b'changed')
    with pytest.raises(ValueError,match='changed'):d.context_call({'operation':'crop','image_id':ident,'roi':[0,0,2,2]})


def test_text_whitelist_does_not_grant_other_sessions_or_private_files(tmp_path):
    d=AgyDecider(tmp_path/'d');d.prepare({'messages':initial(),'tools':[]})
    private=d.workspace/'answer.json';private.write_text('PRIVATE')
    other=tmp_path/'other.json';other.write_text('PRIVATE')
    for path in (private,other):
        assert not permitted({'name':'view_file','args':{'AbsolutePath':str(path)}},d.context.files)[0]
    with pytest.raises(ValueError):d.prepare({'messages':initial(),'tools':[]})


def test_compacted_working_context_does_not_accumulate_all_prior_turns(tmp_path):
    d=AgyDecider(tmp_path/'d');messages=initial()
    for turn in range(6):
        prompt,required=d.prepare({'messages':messages,'tools':[]})
        d.context.record_decision(turn,{'content':f'NOTE-{turn}','plan':'plan','tool_calls':[]})
        next_turn(d,messages)
    assert 'NOTE-3' in prompt and 'NOTE-4' in prompt and 'NOTE-0' not in prompt
    assert len(d.context.turns)==6
    assert d.context_call({'operation':'read','turn':0})[0]['decision']['model_candidate']['content']=='NOTE-0'
    assert len(d.context_call({'operation':'list','start':1,'limit':2})[0]['turns'])==2


@pytest.mark.asyncio
async def test_fresh_process_sessions_keep_usage_and_cleanup(tmp_path):
    # Real subprocess protocol, no paid model or simulator. Each CLI starts at num_turns=1.
    fake=tmp_path/'fake-agy';fake.write_text('''#!/usr/bin/env python3
import json,sys,uuid
cid=str(uuid.uuid4())
print(json.dumps({'event':'init','conversation_id':cid}),flush=True)
for line in sys.stdin:
 print(json.dumps({'event':'result','result':{'status':'SUCCESS','conversation_id':cid,'num_turns':1,'usage':{'input_tokens':10,'output_tokens':20,'thinking_tokens':15,'cache_read_tokens':0,'total_tokens':30},'structured_output':{'content':'Observed','plan':'Continue','tool_calls':[{'name':'robot_status','arguments':{}}]}}}),flush=True)
''');fake.chmod(0o755)
    d=AgyDecider(tmp_path/'d',executable=str(fake));sums=[]
    for turn in range(3):
        result=await d(None,None,{'messages':[], 'max_tokens':100,'tools':[{'function':{'name':'robot_status'}}]},f'r{turn}')
        sums.append(result['usage']['completion_tokens']);assert d.proc is None
    assert sums==[20,20,20] and len(set(d.conversations))==3
    receipts=[json.loads(p.read_text()) for p in d.root.glob('*-receipt.json')]
    assert sum(r['usage_delta']['output_tokens'] for r in receipts)==60
    assert len({r['session_id'] for r in receipts})==1
    assert len({r['conversation_id'] for r in receipts})==3

@pytest.mark.asyncio
async def test_runner_history_processing_never_refreshes_motion_authority(tmp_path,monkeypatch):
    from test_longvil_online import fixture,Session
    import run_longvil_online as online
    args,evidence=fixture(tmp_path);args.max_rounds=4
    sess=Session();events=[]
    class Decider:
        context_tool=CONTEXT_TOOL
        def __init__(self):self.n=0;self.feedback=[]
        def context_call(self,body):
            assert body=={'operation':'crop','image_id':'old.png','roi':[0,0,2,2]}
            return {'scope':'HISTORICAL','source_image_id':'old.png'},[
                {'type':'text','text':'HISTORICAL CACHE CROP: old live-0001'},pixels()]
        def record_tool_result(self,turn,record):self.feedback.append(record)
        async def __call__(self,a,c,p,ident):
            n=self.n;self.n+=1
            assert 'session_context' in [t['function']['name'] for t in p['tools']]
            if n==1:name='session_context';body={'operation':'crop','image_id':'old.png','roi':[0,0,2,2]}
            else:
                name='move_to';body={'xyz_m':[.3,0,.2],'request_id':'move-'+str(n),'observation_id':'live-0002' if n==3 else 'live-0001'}
            if n==2:
                assert 'HISTORICAL CACHE CROP' in json.dumps(p['messages'])
                assert '"latest_observation_id": "live-0002"' in p['messages'][-1]['content']
            return {'model':'test','usage':{'completion_tokens':10},'choices':[{'message':{'role':'assistant','content':'test',
                'tool_calls':[{'id':str(n),'function':{'name':name,'arguments':json.dumps(body)}}]}}]}
    d=Decider();args.model_decider=d
    async def remote(*unused):return {'sim_time_s':sess.time,'objects':{}}
    monkeypatch.setattr(online,'remote',remote)
    report={};await online.online_loop(sess,args,evidence,events.append,tmp_path,report,None)
    assert len(sess.moves)==2 and report['observations']==3
    assert [m['request_id'] for m in sess.moves]==['move-0','move-3']
    assert d.feedback[2]['result']['error']['executed'] is False
    assert 'error' not in d.feedback[-1]['result']
    assert not any(name=='session_context' for name,body in sess.calls) # Never a hardware request.


@pytest.mark.asyncio
async def test_all_schema_errors_return_together_without_guessing_motion(tmp_path):
    from test_longvil_online import fixture,Session
    import run_longvil_online as online
    args,evidence=fixture(tmp_path);args.max_rounds=1;session=Session();events=[]
    async def decide(*unused):
        return {'model':'test','usage':{'completion_tokens':10},'choices':[{'message':{'role':'assistant','content':'test',
             'tool_calls':[{'id':'bad','function':{'name':'move_to','arguments':json.dumps({'xyz':[.3,0,.2], 'rpy':[180,0,0], 'observation_id':'live-0001'})}}]}}]}
    args.model_decider=decide
    await online.online_loop(session,args,evidence,events.append,tmp_path,{},None)
    error=next(e for e in events if e['kind']=='tool_result')['result']['error']
    assert error['executed'] is False
    assert {'xyz_m','request_id','observation_id'} <= set(error['expected_input_schema']['required'])
    assert any('xyz_m' in e['message'] for e in error['issues'])
    assert any('request_id' in e['message'] for e in error['issues'])
    assert any('Additional properties' in e['message'] for e in error['issues'])
    assert not session.moves


def test_cache_constructor_never_overwrites_an_existing_session(tmp_path):
    from session_context import SessionContext
    workspace=tmp_path/'public';workspace.mkdir()
    first=SessionContext(workspace)
    first.publish(first.index_path,{'session_id':first.session_id})
    with pytest.raises(FileExistsError):SessionContext(workspace)
    assert json.loads(first.index_path.read_text())['session_id']==first.session_id


def test_guard_main_uses_documented_allow_and_deny_values(tmp_path):
    import subprocess
    d=AgyDecider(tmp_path/'d');d.prepare({'messages':initial(),'tools':[]})
    hook=Path(__file__).resolve().parents[1]/'agy_observation_guard.py'
    for path,expected in [(str(d.context.contract_path),'allow'),(str(tmp_path/'private.json'),'deny')]:
        body={'conversationId':'test','toolCall':{'name':'view_file','args':{'AbsolutePath':path}}}
        result=subprocess.run([sys.executable,str(hook),'pre',str(d.state_path)],input=json.dumps(body),text=True,capture_output=True,check=True)
        assert json.loads(result.stdout)['decision']==expected
    # Recovery path always points at public files, not transcripts or private reference data.
    reason=json.loads(result.stdout)['reason']
    assert str(d.context.contract_path) in reason and 'transcript_full' not in reason


@pytest.mark.asyncio
async def test_unknown_outcome_is_cached_before_abort(tmp_path,monkeypatch):
    from test_longvil_online import fixture,Session
    import run_longvil_online as online
    args,evidence=fixture(tmp_path);sess=Session(unknown=True)
    class Decider:
        def __init__(self):self.records=[]
        def record_tool_result(self,turn,value):self.records.append(value)
        async def __call__(self,*unused):
            return {'model':'test','usage':{'completion_tokens':10},'choices':[{'message':{'role':'assistant','content':'test','tool_calls':[
                {'id':'one','function':{'name':'move_to','arguments':json.dumps({'xyz_m':[.3,0,.2],'request_id':'unknown','observation_id':'live-0001'})}}]}}]}
    d=Decider();args.model_decider=d
    async def remote(*unused):return {'sim_time_s':sess.time,'objects':{}}
    monkeypatch.setattr(online,'remote',remote)
    with pytest.raises(online.ModelCallAnomaly,match='unknown'):
        await online.online_loop(sess,args,evidence,lambda e:None,tmp_path,{},None)
    assert len(d.records)==1 and d.records[0]['result']['status']=='outcome_unknown'
    assert len(sess.moves)==1

@pytest.mark.asyncio
async def test_predispatch_grip_change_is_counted_and_cached(tmp_path):
    from test_longvil_online import fixture
    from test_online_exploration import FeedbackSession,response
    import run_longvil_online as online
    args,evidence=fixture(tmp_path);args.exploration=True;args.max_rounds=1
    class Session(FeedbackSession):
        def __init__(self):super().__init__();self.checks=0
        async def call_tool(self,name,body):
            if name=='robot_status':
                self.checks+=1
                if self.checks==3:self.held=False
            return await super().call_tool(name,body)
    class Decider:
        def __init__(self):self.records=[]
        def record_tool_result(self,turn,record):self.records.append(record)
        async def __call__(self,*unused):
            return response([('move_to',{'xyz_m':[.3,0,.2],'request_id':'no-dispatch','observation_id':'live-0001'})])
    d=Decider();args.model_decider=d;sess=Session();report={}
    await online.online_loop(sess,args,evidence,lambda e:None,tmp_path,report,None)
    assert not sess.moves
    assert d.records[0]['result']['error']['code']=='grip_state_changed'
    assert d.records[0]['result']['error']['executed'] is False
    assert report['tool_calls']==1 and report['model_tool_counts']['move_to']==1
