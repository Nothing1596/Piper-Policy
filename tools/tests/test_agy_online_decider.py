import base64
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pytest
from agy_observation_guard import permitted
from agy_online_decider import AgyDecider, usage_delta


def test_native_reader_is_not_a_structured_harness_action(tmp_path):
    from jsonschema import validate,ValidationError
    d=AgyDecider(tmp_path/'d')
    prompt,_=d.prepare({'messages':[],'tools':[{'function':{'name':'select_region'}}]})
    schema=json.loads((d.root/'response-schema.json').read_text())
    value={'content':'seen','plan':'select','tool_calls':[{'name':'select_region','arguments':{'roi':[1,1,3,3]}}]}
    validate(value,schema)
    value['tool_calls'][0]['name']='view_file'
    with pytest.raises(ValidationError):validate(value,schema)
    assert 'NATIVE evidence-reading call' in prompt
    assert 'never put it in the returned tool_calls' in prompt
    d.turn=1
    d.prepare({'messages':[],'tools':[{'function':{'name':'robot_status'}}]})
    schema=json.loads((d.root/'response-schema.json').read_text())
    assert schema['properties']['tool_calls']['items']['properties']['name']['enum']==['robot_status']
    value['tool_calls'][0]['name']='select_region'
    with pytest.raises(ValidationError):validate(value,schema)


def test_image_allowlist_denies_private_shell_and_symlink(tmp_path):
    image=tmp_path/'image.jpg'; image.write_bytes(b'pixels')
    private=tmp_path/'reference.json';private.write_text('{"code":"SECRET"}')
    allow={str(image):hashlib.sha256(image.read_bytes()).hexdigest()}
    assert permitted({'name':'view_file','args':{'AbsolutePath':str(image)}},allow)[0]
    for tool,path in [('view_file',private),('run_command',image),('search_web',image),('invoke_subagent',image)]:
        assert not permitted({'name':tool,'args':{'AbsolutePath':str(path)}},allow)[0]
    link=tmp_path/'link.jpg';link.symlink_to(image)
    assert not permitted({'name':'view_file','args':{'AbsolutePath':str(link)}},allow)[0]
    image.write_bytes(b'changed')
    assert not permitted({'name':'view_file','args':{'AbsolutePath':str(image)}},allow)[0]


def test_cumulative_usage_is_not_recounted_or_fabricated():
    a=dict(input_tokens=100,output_tokens=50,thinking_tokens=40,cache_read_tokens=20,total_tokens=150)
    b={k:v*2 for k,v in a.items()}
    assert usage_delta(b,a)==a
    with pytest.raises(ValueError):usage_delta({},a)
    with pytest.raises(ValueError):usage_delta(a,b)


def test_internal_structured_finish_is_output_not_a_robot_dispatch():
    assert permitted({'name':'finish','args':{'content':'Observed',
        'tool_calls':[{'name':'move_to','arguments':{}}]}},{}) == (True,None)
    assert not permitted({'name':'run_command','args':{'command':'run script'}},{})[0]


def test_real_demo_and_live_bytes_export_and_incremental_context(tmp_path):
    d=AgyDecider(tmp_path/'decision')
    image={'type':'image_url','image_url':{'url':'data:image/jpeg;base64,'+base64.b64encode(b'demo pixels').decode()}}
    msgs=[{'role':'system','content':'Only public task'}, {'role':'user','content':[
        {'type':'text','text':'Historical demonstration f0001 @ 1.25s'},image]},
        {'role':'user','content':'CURRENT execution_progress: {}'}]
    prompt,required=d.prepare({'messages':msgs,'tools':[]})
    assert len(required)==1 and Path(next(iter(required))).read_bytes()==b'demo pixels'
    assert 'f0001 @ 1.25s' in prompt and '<base64_image_omitted>' not in prompt
    d.seen_images.update(required);d.turn=1;d.prior_message_count=d.next_message_count
    msgs.insert(-1,{'role':'tool','content':'actual rejected action'})
    prompt,required=d.prepare({'messages':msgs,'tools':[]})
    assert 'actual rejected action' in prompt and not required


@pytest.mark.asyncio
async def test_missing_image_view_blocks_motion_even_with_valid_json(tmp_path,monkeypatch):
    d=AgyDecider(tmp_path/'decision');d.conversation='new-session'
    async def request(prompt):
        return dict(status='SUCCESS',conversation_id='new-session',num_turns=1,
                    usage=dict(input_tokens=100,output_tokens=50,thinking_tokens=40,cache_read_tokens=0,total_tokens=150),
                    structured_output={'content':'next','tool_calls':[{'name':'move_to','arguments':{}}]})
    monkeypatch.setattr(d,'request',request)
    payload={'max_tokens':1000,'tools':[{'function':{'name':'move_to'}}], 'messages':[
        {'role':'user','content':[{'type':'text','text':'CURRENT simulation evidence latest'},
         {'type':'image_url','image_url':{'url':'data:image/jpeg;base64,cGl4ZWxz'}}]},
        {'role':'user','content':'CURRENT execution_progress: {}'}]}
    with pytest.raises(ValueError,match='Image delivery'):
        await d(None,None,payload,'request')
    assert d.turn==0

@pytest.mark.asyncio
async def test_single_camera_blocks_explicit_wrist_request(tmp_path):
    from test_longvil_online import fixture,Session
    import run_longvil_online as online
    args,evidence=fixture(tmp_path);args.max_rounds=1;args.camera_condition='mujoco-overview'
    session=Session();events=[]
    async def decide(*values):
        return {'model':'test','usage':{'completion_tokens':10},'choices':[{'finish_reason':'tool_calls','message':{
            'role':'assistant','content':'','tool_calls':[{'id':'one','function':{'name':'camera_observe',
            'arguments':json.dumps({'serial':'all'})}}]}}]}
    args.model_decider=decide
    await online.online_loop(session,args,evidence,events.append,tmp_path,{},None)
    captures=[body for name,body in session.calls if name=='camera_observe']
    assert captures==[{'max_frames':1,'serial':'mujoco-overview'}]
    assert any('does not expose the wrist' in json.dumps(e) for e in events)

@pytest.mark.asyncio
async def test_scripted_multi_action_queue_is_rejected(tmp_path,monkeypatch):
    from agy_online_decider import DecisionRejected
    d=AgyDecider(tmp_path/'decision');d.conversation='fresh'
    async def request(prompt):
        return dict(status='SUCCESS',conversation_id='fresh',num_turns=1,
          usage=dict(input_tokens=100,output_tokens=50,thinking_tokens=20,cache_read_tokens=0,total_tokens=150),
          structured_output={'content':'queue','plan':'next','tool_calls':[{'name':'move_to','arguments':{}}]*2})
    monkeypatch.setattr(d,'request',request)
    with pytest.raises(DecisionRejected):
        await d(None,None,{'max_tokens':100,'messages':[],'tools':[{'function':{'name':'move_to'}}]},'one')
    assert d.turn==0


def test_prepare_places_required_image_views_before_session_cache_and_discourages_redundant_json_reads(tmp_path):
    d=AgyDecider(tmp_path/'decision')
    prompt,_=d.prepare({'messages':[],'tools':[{'function':{'name':'move_linear'}}]})
    assert prompt.index('REQUIRED_IMAGE_VIEWS') < prompt.index('WORKING_CONTEXT') < prompt.index('SESSION_CACHE')
    assert 'do NOT call view_file on SESSION_CACHE JSON files' in prompt
    assert 'multiple_clusters_detected' in prompt
    assert 'move_linear' in prompt
