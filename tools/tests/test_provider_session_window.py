"""Bounded inference reuse; robot/executor ownership is outside this adapter."""
import json
import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]))
from agy_online_decider import AgyDecider,DecisionRejected


def fake_cli(root,bad_name=False):
    p=root/'fake-cli'
    p.write_text('#!'+sys.executable+'\n'+'''import sys,json,uuid
schema=json.load(open(sys.argv[sys.argv.index('--json-schema')+1]))
name=schema['properties']['tool_calls']['items']['properties']['name']['enum'][0]
cid=str(uuid.uuid4()); n=0
for line in sys.stdin:
 n+=1
 if n==1: print(json.dumps({'event':'init','conversation_id':cid}),flush=True)
 print(json.dumps({'event':'result','result':{'status':'SUCCESS','conversation_id':cid,'num_turns':n,
  'usage':{'input_tokens':10*n,'output_tokens':20*n,'thinking_tokens':15*n,'cache_read_tokens':0,'total_tokens':30*n},
  'structured_output':{'content':'Seen','plan':'Next','tool_calls':[{'name':NAME,'arguments':{}}]}}}),flush=True)
'''.replace('NAME',"'unavailable_tool'" if bad_name else 'name'))
    p.chmod(0o755);return str(p)


def payload(name='robot_status'):
    return {'messages':[],'tools':[{'function':{'name':name}}],'max_tokens':100}


@pytest.mark.asyncio
async def test_two_turn_window_reuses_then_refreshes_and_keeps_usage(tmp_path):
    d=AgyDecider(tmp_path/'d',executable=fake_cli(tmp_path),provider_turn_window=2)
    results=[];ids=[]
    try:
        for n in range(3):
            r=await d(None,None,payload(),str(n));results.append(r['usage']['completion_tokens']);ids.append(d.conversation)
            assert (d.proc is None)==(n==1)
        assert results==[20,20,20] and ids[0]==ids[1] and ids[1]!=ids[2]
        assert len(d.conversations)==2
        assert [json.loads((d.root/f'turn-{i:03d}-receipt.json').read_text())['segment_start_turn'] for i in range(3)]==[0,0,2]
        assert d.context.query({'operation':'read','turn':0})[0]['decision']['model_candidate']['plan']=='Next'
    finally:await d.close()
    assert d.proc is None


@pytest.mark.asyncio
async def test_tool_catalog_change_refreshes_before_loading_new_schema(tmp_path):
    d=AgyDecider(tmp_path/'d',executable=fake_cli(tmp_path),provider_turn_window=2)
    try:
        await d(None,None,payload(), '0');first=d.proc;old=d.conversation
        result=await d(None,None,payload('robot_diagnostics'),'1')
        assert d.conversation!=old and first.returncode is not None
        assert result['choices'][0]['message']['tool_calls'][0]['function']['name']=='robot_diagnostics'
        assert result['usage']['completion_tokens']==20
    finally:await d.close()


@pytest.mark.asyncio
async def test_invalid_result_closes_reused_process(tmp_path):
    d=AgyDecider(tmp_path/'d',executable=fake_cli(tmp_path,bad_name=True),provider_turn_window=2)
    with pytest.raises(DecisionRejected):await d(None,None,payload(),'0')
    assert d.proc is None and d.stderr is None and d.turn==0


@pytest.mark.parametrize('value',[0,9,True,1.5,None])
def test_invalid_window_rejected_before_creating_workspace(tmp_path,value):
    with pytest.raises(ValueError):AgyDecider(tmp_path/'d',provider_turn_window=value)
    assert not (tmp_path/'d').exists()


@pytest.mark.asyncio
async def test_repeated_initialization_is_rejected_even_default_window(tmp_path):
    cli=Path(fake_cli(tmp_path));source=cli.read_text()
    line=" if n==1: print(json.dumps({'event':'init','conversation_id':cid}),flush=True)"
    cli.write_text(source.replace(line,line+'\n'+line))
    d=AgyDecider(tmp_path/'d',executable=str(cli))
    with pytest.raises(ValueError,match='repeated initialization'):
        await d(None,None,payload(),'0')
    assert d.proc is None and d.turn==0


@pytest.mark.asyncio
async def test_known_exited_process_refreshes_without_replaying_previous_request(tmp_path):
    cli=Path(fake_cli(tmp_path));cli.write_text(cli.read_text()+'\n break\n')
    d=AgyDecider(tmp_path/'d',executable=str(cli),provider_turn_window=2)
    try:
        await d(None,None,payload(),'0');old=d.conversation
        await d.proc.wait()  # Known exit before the next request, not a retry.
        result=await d(None,None,payload(),'1')
        assert d.conversation!=old and d.turn==2
        assert result['usage']['completion_tokens']==20
        receipts=[json.loads(p.read_text()) for p in sorted(d.root.glob('turn-*-receipt.json'))]
        assert [r['request_id'] for r in receipts]==['0','1']
        assert [r['segment_start_turn'] for r in receipts]==[0,1]
    finally:await d.close()


@pytest.mark.asyncio
async def test_stalled_usage_in_reused_session_is_not_a_free_decision(tmp_path):
    cli=Path(fake_cli(tmp_path));source=cli.read_text()
    for expr in ('10*n','20*n','15*n','30*n'):source=source.replace(expr,expr.split('*')[0])
    cli.write_text(source)
    d=AgyDecider(tmp_path/'d',executable=str(cli),provider_turn_window=2)
    await d(None,None,payload(),'0')
    with pytest.raises(ValueError,match='usage did not advance'):
        await d(None,None,payload(),'1')
    assert d.proc is None and d.turn==1
    assert (d.root/'turn-001-result.json').exists()
    assert not (d.root/'turn-001-receipt.json').exists()


@pytest.mark.asyncio
async def test_new_process_must_not_reuse_closed_conversation_id(tmp_path):
    cli=Path(fake_cli(tmp_path));cli.write_text(cli.read_text().replace('str(uuid.uuid4())',"'same-id'"))
    d=AgyDecider(tmp_path/'d',executable=str(cli))
    await d(None,None,payload(),'0')
    with pytest.raises(ValueError,match='prior decision session'):
        await d(None,None,payload(),'1')
    assert d.proc is None and d.turn==1
