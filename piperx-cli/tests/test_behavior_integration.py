"""Public frontend/model boundaries; no provider, hardware or grading oracle."""
import json

import httpx
import pytest

from piperx_middleware.agent_console import dispatch
from piperx_middleware.console_interaction import InteractiveConsoleController
from piperx_middleware.model_agent import ModelConfig, run_turn
from test_console_interaction import FakeBridge, FakeManaged, FakeOperator


def controller(root):
    return InteractiveConsoleController(bridge=FakeBridge(),root=root,emit=lambda _:None,
        managed=FakeManaged(),operator_call=FakeOperator().call)


@pytest.mark.asyncio
async def test_model_refreshes_plan_and_task_scoped_review_across_user_turns(tmp_path,monkeypatch):
    c=controller(tmp_path);p=c.plans().create('Inspect the visible surface')
    raw=c._invoke_tool
    async def invoke(name,body):
        if name=='scene_query':return {'status':'ok','point_m':[.1,.2,.3]}
        return await raw(name,body)
    monkeypatch.setattr(c,'_invoke_tool',invoke)
    query={'operation':'surface_point','pixel':[1,2],'observation_path':'same-view'}
    calls=[{'id':f'c{i}','type':'function','function':{'name':'scene_query',
        'arguments':json.dumps(query)}} for i in range(3)]
    responses=iter([{'choices':[{'finish_reason':'tool_calls','message':{
        'role':'assistant','tool_calls':calls}}]},
        {'choices':[{'finish_reason':'stop','message':{'content':'Review needed'}}]},
        {'choices':[{'finish_reason':'stop','message':{'content':'Still reviewing'}}]}])
    actual=httpx.AsyncClient;payloads=[]
    def transport(req):
        payloads.append(json.loads(req.content));return httpx.Response(200,json=next(responses))
    monkeypatch.setattr(httpx,'AsyncClient',lambda **kw:actual(transport=httpx.MockTransport(transport),**kw))
    config=ModelConfig(endpoint='http://127.0.0.1:1234/v1',model='fixture')
    messages=[{'role':'user','content':'Inspect the scene'}]
    await run_turn(config,messages,c.available_tools(),c.invoke_tool,lambda _:None,
                   refresh_context=c.refresh_harness_context)
    c.plans().update(p['task_id'],p['version'],{'current_stage':'verify'})
    messages.append({'role':'user','content':'Continue'})
    await run_turn(config,messages,c.available_tools(),c.invoke_tool,lambda _:None,
                   refresh_context=c.refresh_harness_context)
    def context(payload):
        rows=[m for m in payload['messages'] if isinstance(m.get('content'),str)
              and m['content'].startswith('CURRENT Piper harness context: ')]
        assert len(rows)==1
        return json.loads(rows[0]['content'].split(': ',1)[1])
    assert context(payloads[0])['current_plan']['version']==1
    assert context(payloads[1])['harness_review']['state']=='needs_review'
    latest=context(payloads[2])
    assert latest['current_plan']['version']==2
    assert latest['harness_review']['state']=='needs_review'
    assert len(list((tmp_path/'harness').glob('*.jsonl')))==1


@pytest.mark.asyncio
async def test_jsonl_preserves_execution_result_when_diagnostic_or_audit_fails(tmp_path,monkeypatch):
    c=controller(tmp_path);calls=[]
    async def raw(name,body):
        calls.append(name)
        return {'status':'succeeded','request_id':body['request_id'],'job_id':'job-1'}
    monkeypatch.setattr(c,'_invoke_tool',raw)
    def fail(*args,**kw):raise RuntimeError('diagnostic fixture failure')
    monkeypatch.setattr(c._behavior,'record_result',fail)
    result=await dispatch(c,{'id':'test','command':'call','tool':'move_to',
                            'arguments':{'request_id':'known-action'}})
    assert result['status']=='succeeded' and result['request_id']=='known-action'
    assert result['job_id']=='job-1' and result['harness_review']['state']=='unavailable'
    assert calls==['move_to']

    # A read-only audit path must not turn a known execution into a transport error.
    c=controller(tmp_path/'other')
    c.root.mkdir();(c.root/'harness').write_text('not a directory')
    monkeypatch.setattr(c,'_invoke_tool',raw)
    result=await c.invoke_tool('move_to',{'request_id':'known-action-2'})
    assert result['status']=='succeeded' and result['request_id']=='known-action-2'
    assert result['harness_review']['audit_error']


@pytest.mark.asyncio
async def test_scope_changes_reset_diagnostics_before_next_model_request(tmp_path,monkeypatch):
    c=controller(tmp_path/'simulation')
    p=c.plans().create('Task A');c._behavior.record_plan(p)
    for _ in range(3):c._behavior.record_result('scene_query',{'pixel':[1,2]}, {'status':'ok','point_m':[1,2,3]})
    assert c._behavior.snapshot()['state']=='needs_review'
    old_session=c._behavior_session
    c.root=tmp_path/'real'
    messages=[];await c.refresh_harness_context(messages)
    assert c._behavior.task_id is None and c._behavior.snapshot()['state']=='working'
    assert c._behavior_session!=old_session
    p=c.plans().create('Task B');await c.refresh_harness_context(messages)
    assert c._behavior.task_id==p['task_id']
    assert len(messages)==1

    # Operator /plan new also switches scope before any model request occurs.
    await c.handle_line('/plan new Operator task C')
    assert c._behavior.task_id==c.current_plan()['task_id']!=p['task_id']


@pytest.mark.asyncio
async def test_failed_completion_citation_is_visible_without_changing_plan(tmp_path,monkeypatch):
    c=controller(tmp_path);p=await c.invoke_tool('task_plan',{'operation':'create','goal':'A transfer'})
    raw=c._invoke_tool
    async def blocked(name,body):
        if name=='move_to':return {'status':'failed','error_code':'motion_blocked','request_id':body['request_id']}
        return await raw(name,body)
    monkeypatch.setattr(c,'_invoke_tool',blocked)
    await c.invoke_tool('move_to',{'request_id':'blocked-1'})
    ledger={'steps':[{'id':'s1','status':'completed','execution_refs':['blocked-1']}]}
    updated=await c.invoke_tool('task_plan',{'operation':'update','task_id':p['task_id'],
        'expected_version':p['version'],'changes':{'interpretation':json.dumps(ledger)}})
    assert any(i['code']=='unsupported_completion' for i in updated['harness_review']['issues'])
    assert json.loads(updated['interpretation'])==ledger
    # The advisory never hides the stop tool or rewrites the model's hypothesis.
    stopped=await c.invoke_tool('robot_stop',{})
    assert stopped['status']=='stopped'
