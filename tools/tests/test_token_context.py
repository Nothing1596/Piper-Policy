import base64
import copy
import io
import json

from PIL import Image
import pytest

from token_context import TokenContext, ContextBudgetError


def image(label, color='red'):
    out=io.BytesIO();Image.new('RGB',(32,24),color).save(out,format='PNG')
    return [{'type':'text','text':label}, {'type':'image_url','image_url':{
        'url':'data:image/png;base64,'+base64.b64encode(out.getvalue()).decode()}}]


def payload(history=0):
    parts=[]
    for n in range(12):parts+=image(f'RECORDED demo f{n:06d} timestamp_s={n}')
    messages=[{'role':'system','content':'fixed rules'}, {'role':'user','content':parts}]
    for n in range(history):
        messages += [{'role':'assistant','tool_calls':[{'id':f't{n}','function':{
            'name':'robot_status','arguments':'{}'}}]},
            {'role':'tool','tool_call_id':f't{n}','content':json.dumps({'request_id':f'action{n}','status':'succeeded'})}]
    messages += [{'role':'user','content':image('CURRENT simulation evidence live-0007','blue')},
                 {'role':'user','content':'CURRENT execution_progress: {"current_plan":{"version":7}}'}]
    return {'messages':messages,'tools':[{'type':'function','function':{'name':'robot_status'}}]}


async def counter(messages,tools):
    # Deliberately byte-costed fixture, not a claim about native model tokens.
    return len(json.dumps({'messages':messages,'tools':tools}))//4


@pytest.mark.asyncio
async def test_budget_retains_current_state_and_archives_all_original_images(tmp_path):
    c=TokenContext(tmp_path/'public',context_tokens=6800,output_tokens=1000,safety_tokens=300,image_tokens=800)
    p=payload(10);before=copy.deepcopy(p)
    packed,a=await c.pack(p,counter)
    assert p==before and a['prompt_token_bound']<=5500
    assert len(a['delivered_image_ids'])<13 and len(a['cached_image_ids'])==13
    assert any('live-0007' in str(m) and isinstance(m.get('content'),list)
               and any(x.get('type')=='image_url' for x in m['content']) for m in packed)
    progress=next(m['content'] for m in packed if isinstance(m.get('content'),str)
                  and m['content'].startswith('CURRENT execution_progress: '))
    assert json.loads(progress.split(': ',1)[1])['current_plan']['version']==7
    assert not any('base64,' in json.dumps(c.cache.read_public(row['path'])) for row in c.cache.turns.values())
    # No orphaned tool-result message after history eviction.
    ids={x['id'] for m in packed for x in m.get('tool_calls',[])}
    assert all(m['tool_call_id'] in ids for m in packed if m['role']=='tool')
    removed=next(i for i in c.cache.images if i not in a['delivered_image_ids'])
    receipt,pixels=c.query({'operation':'crop','image_id':removed,'roi':[0,0,32,24]})
    assert receipt['scope'].startswith('HISTORICAL') and pixels[1]['type']=='image_url'
    binary=base64.b64decode(pixels[1]['image_url']['url'].split(',',1)[1])
    assert Image.open(io.BytesIO(binary)).size==(32,24)


@pytest.mark.asyncio
async def test_requested_historical_image_survives_next_request_and_stays_historical(tmp_path):
    c=TokenContext(tmp_path/'public',context_tokens=12000,output_tokens=1000,safety_tokens=300,image_tokens=800)
    await c.pack(payload(),counter)
    c.record_response({'choices':[{'message':{'content':'inspect earlier image','reasoning_content':'private'}}]})
    c.record_tool_result(0,{'result':{'status':'ok','request_id':'known-result'}})
    ident=next(iter(c.cache.images));receipt,pixels=c.query({'operation':'crop','image_id':ident,'roi':[0,0,32,24]})
    p=payload(1);p['messages'].insert(-1,{'role':'user','content':pixels})
    packed,a=await c.pack(p,counter)
    assert any('HISTORICAL CACHE CROP' in str(m) and any(x.get('type')=='image_url' for x in m.get('content',[]) if isinstance(x,dict)) for m in packed)
    old,_=c.query({'operation':'read','turn':0})
    assert 'known-result' in json.dumps(old)
    assert 'private' not in json.dumps(old)
    assert old['next_start'] is not None
    assert a['turn']==1


@pytest.mark.asyncio
async def test_mandatory_overflow_is_archived_without_changing_latest_receipt(tmp_path):
    c=TokenContext(tmp_path/'public',context_tokens=1500,output_tokens=1000,safety_tokens=300)
    with pytest.raises(ContextBudgetError,match='Required'):
        await c.pack(payload(1),counter)
    assert 0 in c.cache.turns and c.turn==0


@pytest.mark.asyncio
async def test_same_policy_has_identical_projection_and_does_not_mutate_tools(tmp_path):
    p=payload(8);a=TokenContext(tmp_path/'a',context_tokens=12000,output_tokens=1000,image_tokens=800)
    b=TokenContext(tmp_path/'b',context_tokens=12000,output_tokens=1000,image_tokens=800)
    x,xa=await a.pack(p,counter);y,ya=await b.pack(p,counter)
    assert x==y and xa==ya
    assert a.tool==b.tool and p['tools']==[{'type':'function','function':{'name':'robot_status'}}]


@pytest.mark.asyncio
async def test_changed_cache_cannot_be_read_back(tmp_path):
    c=TokenContext(tmp_path/'public');await c.pack(payload(),counter)
    item=next(iter(c.cache.images.values()))
    from pathlib import Path
    Path(item['path']).write_bytes(b'changed')
    with pytest.raises(ValueError,match='changed'):
        c.query({'operation':'crop','image_id':item['image_id'],'roi':[0,0,32,24]})


@pytest.mark.asyncio
async def test_long_old_result_is_paged_losslessly_instead_of_refilling_context(tmp_path):
    c=TokenContext(tmp_path/'public');await c.pack(payload(),counter)
    c.record_response({'choices':[{'message':{'content':'candidate'}}]})
    value={'result':{'large':'x'*12000,'request_id':'already-executed'}}
    c.record_tool_result(0,value)
    start=0;chunks=[]
    while start is not None:
        page,_=c.query({'operation':'read','turn':0,'start':start,'limit':20})
        assert len(page['records'])<=2
        chunks += [x for x in page['records'] if x['kind']=='actual_tool_result']
        start=page['next_start']
    assert json.loads(''.join(x['json_text'] for x in chunks))==value


@pytest.mark.asyncio
async def test_cv_metadata_follows_selected_pixels_while_full_original_stays_cached(tmp_path):
    p=payload();frames=[{'frame_id':f'f{n:06d}','timestamp_s':n,'cv':{'raw':'cv'+str(n)*500}}
                       for n in range(12)]
    p['messages'][1]['content'].insert(0,{'type':'text','text':json.dumps({
        'goal':'reproduce','demo':{'frames':frames},'current_plan':{'version':1}})})
    c=TokenContext(tmp_path/'public',context_tokens=12000,output_tokens=1000,image_tokens=800)
    packed,a=await c.pack(p,counter)
    original=c.cache.read_public(c.cache.turns[0]['path'])
    assert json.loads(original['messages'][1]['content'][0]['text'])['demo']['frames']==frames
    projected=json.loads(packed[1]['content'][0]['text'])
    assert all(set(x)=={'frame_id','timestamp_s'} for x in projected['demo']['frames'])
    pictures=packed[-1]['content']
    assert any('Exact frame CV:' in x.get('text','') for x in pictures)
    assert 'current_plan' not in projected


@pytest.mark.asyncio
async def test_old_initial_capture_is_not_pinned_as_current_state(tmp_path):
    p=payload(8)
    p['messages'].insert(2,{'role':'user','content':image('CURRENT simulation evidence live-0001','green')})
    c=TokenContext(tmp_path/'public');packed,a=await c.pack(p,counter)
    assert 2 not in a['kept_message_indices']
    assert any('live-0007' in x.get('text','') for x in packed[-1]['content'])
    for m in packed:
        if isinstance(m.get('content'),list):
            for x in m['content']:
                if 'live-0001' in x.get('text',''):
                    assert x['text'].startswith('HISTORICAL')


@pytest.mark.asyncio
async def test_accumulating_measurement_history_cannot_displace_authoritative_plan(tmp_path):
    p=payload(1);progress={'current_plan':{'version':7,'interpretation':'Keep the full sequence'},
                         'video_plan_review':{'measurement_index':[{'value':'x'*900} for _ in range(30)]}}
    p['messages'][-1]['content']='CURRENT execution_progress: '+json.dumps(progress)
    c=TokenContext(tmp_path/'public',context_tokens=6800,output_tokens=1000,safety_tokens=300,image_tokens=800)
    packed,a=await c.pack(p,counter)
    assert a['omitted_historical_metadata_rows']>0 and a['prompt_token_bound']<=5500
    current=next(m['content'] for m in packed if isinstance(m.get('content'),str)
                 and m['content'].startswith('CURRENT execution_progress: '))
    assert json.loads(current.split(': ',1)[1])['current_plan']==progress['current_plan']
    old=c.cache.read_public(c.cache.turns[0]['path'])
    assert len(json.loads(old['messages'][-1]['content'].split(': ',1)[1])['video_plan_review']['measurement_index'])==30


@pytest.mark.asyncio
async def test_compacted_history_preserves_failed_motion_and_actual_followup_without_success_claim(tmp_path):
    c=TokenContext(tmp_path/'public',context_tokens=6800,output_tokens=1000,safety_tokens=300,image_tokens=800)
    await c.pack(payload(),counter)
    c.record_response({'choices':[{'message':{'content':'try a different approach'}}]})
    failed={'name':'move_to','arguments':{'request_id':'blocked-1','xyz_m':[.2,.1,.3]},
            'result':{'status':'failed','request_id':'blocked-1','error_code':'motion_blocked',
                      'confirmed_blocked':True,'stop_result':{'confirmed_stopped':True},
                      'measured_after':{'tcp':{'xyz_m':[.2,.1,.31]},'robot':{'gripper_width_m':.04}}}}
    c.record_tool_result(0,failed)
    await c.pack(payload(1),counter)
    c.record_response({'choices':[{'message':{'content':'feedback received'}}]})
    c.record_tool_result(1,{'name':'move_by','arguments':{'request_id':'new-2','delta_m':[0,0,.02]},
                          'result':{'status':'succeeded','request_id':'new-2'}})
    p=payload(16)
    for i,m in enumerate(p['messages']):
        if m['role']=='tool':m['content']=json.dumps({'obsolete_history':str(i)+'x'*2500})
    packed,a=await c.pack(p,counter)
    assert len(a['kept_message_indices'])<len(p['messages'])
    memory=json.loads(next(m['content'].split(': ',1)[1] for m in packed
                      if isinstance(m.get('content'),str) and m['content'].startswith('COMPRESSED PUBLIC MEMORY: ')))
    assert memory['motion_receipts'][0]['request_id']=='blocked-1'
    assert memory['motion_receipts'][0]['measured_after']['robot']['gripper_width_m']==.04
    assert memory['motion_receipts'][0]['confirmed_stopped'] is True
    assert memory['blocked_followups'][0]['next_request']=='new-2'
    assert 'does not certify recovery' in memory['blocked_followups'][0]['scope']
    assert 'succeeded primitives do not prove' in memory['scope']
    page,_=c.query({'operation':'read','turn':0})
    assert json.loads(next(x['json_text'] for x in page['records'] if x['kind']=='actual_tool_result'))==failed
    current=next(m['content'] for m in packed if isinstance(m.get('content'),str)
                 and m['content'].startswith('CURRENT execution_progress: '))
    assert json.loads(current.split(': ',1)[1])['current_plan']=={'version':7}


@pytest.mark.asyncio
async def test_measurement_compression_deduplicates_identical_evidence_but_not_changed_results(tmp_path):
    c=TokenContext(tmp_path/'public');await c.pack(payload(),counter)
    value={'name':'grasp_target','arguments':{'observation_id':'obs-1','object_roi':[1,2,3,4]},
           'result':{'status':'ok','source_measurement_id':'measure-1','observed_width_m':.04}}
    c.record_tool_result(0,value)
    c.record_response({'choices':[{'message':{'content':'inspect'}}]})
    await c.pack(payload(1),counter);c.record_tool_result(1,value)
    changed=copy.deepcopy(value);changed['result']['observed_width_m']=.07
    c.record_tool_result(1,changed)
    memory=c.memory_snapshot();rows=memory['measurements']
    assert len(rows)==2
    assert rows[0]['occurrences']==2 and rows[0]['first_source_turn']==0 and rows[0]['source_turn']==1
    assert rows[0]['observation_id']=='obs-1'
    assert [r['evidence']['observed_width_m'] for r in rows]==[.04,.07]
    assert c.cache.read_public(c.cache.root/'memory.json')==memory
    # Malformed arguments are retained as public error facts, never execution.
    c.record_tool_result(1,{'name':'move_to','arguments':'{','result':{
        'status':'error','error':{'code':'invalid_json','executed':False}}})
    assert c.memory_snapshot()['motion_receipts'][-1]['executed'] is False


@pytest.mark.asyncio
async def test_memory_quota_trims_old_rows_but_preserves_current_plan_and_raw_records(tmp_path):
    c=TokenContext(tmp_path/'public',context_tokens=4300,output_tokens=1000,safety_tokens=300,image_tokens=800)
    await c.pack(payload(),counter)
    for n in range(8):
        c.record_tool_result(0,{'name':'move_to','arguments':{'request_id':str(n)},'result':{
            'status':'failed','request_id':str(n),'error':{'message':str(n)+'x'*239},
            'measured_after':{'robot':{'q_deg':[n]*6},'tcp':{'xyz_m':[.2,.1,n/100]}}}})
    c.record_response({'choices':[{'message':{'content':'reconsider'}}]})
    packed,a=await c.pack(payload(1),counter)
    assert a['prompt_token_bound']<=3000 and a['compressed_memory_rows_omitted']>0
    assert all(row['status']=='failed' for row in c.memory_snapshot()['motion_receipts'])
    assert 'completed_steps' not in c.memory_snapshot()
    current=next(m['content'] for m in packed if isinstance(m.get('content'),str)
                 and m['content'].startswith('CURRENT execution_progress: '))
    assert json.loads(current.split(': ',1)[1])['current_plan']['version']==7
    assert len(c.cache.read_public(c.cache.turns[0]['results_path']))==8
