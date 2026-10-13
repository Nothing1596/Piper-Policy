"""Mocked five-stage runs validate input parity, rejection diagnostics and cleanup."""
import asyncio,copy,hashlib,json,sys
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]))
import run_plan_enforcement_pilot as runner
from plan_protocol import strict_program_actions
from longvil_budget import BudgetPause,Ledger
from test_plan_protocol import CODE,interpretation

class Session:
    def __init__(self):
        self.plan={'task_id':'11111111-1111-4111-8111-111111111111','version':1,'interpretation':'','questions':[]}
    async def initialize(self):return SimpleNamespace(serverInfo=SimpleNamespace(version='fixture'))
    async def call_tool(self,name,args):
        if args['operation']=='update':
            self.plan.update(args['changes']);self.plan['version']+=1
        return SimpleNamespace(isError=False,structuredContent=copy.deepcopy(self.plan))

def fixture(tmp_path):
    annotation=tmp_path/'gt.json';annotation.write_text(json.dumps({'object_list':['cube','box'],'code':CODE}))
    pictures=[]
    for i in range(4):
        p=tmp_path/f'{i}.jpg';p.write_bytes(b'fixture'+bytes([i]))
        pictures.append({'frame_id':f'f{i}','timestamp_s':float(i),'image_path':str(p),
                         'image_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'quality':{},'detections':[]})
    pack=lambda frames:dict(source={'sha256':'source'},frames=frames,evaluation={'scope':'selected_historical_frames','marker':'only_visible'})
    evidence={'private-level3-task':{'initial':pack(pictures[:2]),'supplement':pack(pictures[2:])}}
    output=tmp_path/'output';output.mkdir()
    args=SimpleNamespace(output=output,repo=tmp_path,cli='fixture',evaluator='fixture',
          ledger=SimpleNamespace(rows=[],total=0.,check=lambda:None),phase='development')
    item={'task':'private-level3-task','variant':'complex','condition':'R2','effort':'high','repeat':0}
    return args,item,evidence,{'annotation':str(annotation)}

def install(monkeypatch,mode,session=None,inconsistent=False,fail_at=None):
    session=session or Session();requests=[]
    @asynccontextmanager
    async def stdio(params,**kwargs):
        assert 'DEEPSEEK_API_KEY' not in params.env
        yield None,None
    @asynccontextmanager
    async def client(*args,**kwargs):yield session
    async def paid(args,cl,payload,rid,phase):
        turn=len(requests);requests.append(copy.deepcopy(payload))
        if turn==fail_at:raise BudgetPause('unknown usage fixture')
        message={'role':'assistant','content':'recorded review','reasoning_content':'fixture-not-exported'}
        if mode!='R0' and turn<4:
            op=['get','update','update','get'][turn];body={'operation':op,'task_id':session.plan['task_id']}
            if op=='update':body.update(expected_version=session.plan['version'],changes={'interpretation':interpretation(),'questions':[]})
            message['tool_calls']=[{'id':str(turn),'function':{'name':'task_plan','arguments':json.dumps(body)}}]
        if turn==4:
            code=[s.replace("'into'","'left'") for s in CODE] if inconsistent else CODE
            message['tool_calls']=[{'id':'final','function':{'name':'submit_program','arguments':json.dumps({'code':code,'summary':'recorded','uncertainties':[]})}}]
        return {'model':'deepseek-flash','choices':[{'finish_reason':'stop','message':message}],
                'usage':{'completion_tokens':50,'prompt_tokens':10,'total_tokens':60,'prompt_cache_hit_tokens':0,'prompt_cache_miss_tokens':10,'completion_tokens_details':{'reasoning_tokens':10}}}
    async def official(*args):return {'EMA':1,'FSA':1,'SMS':1.,'score_status':'ok'}
    monkeypatch.setattr(runner,'stdio_client',stdio);monkeypatch.setattr(runner,'ClientSession',client)
    monkeypatch.setattr(runner,'paid_call',paid);monkeypatch.setattr(runner,'official_score',official)
    return requests

@pytest.mark.parametrize('mode',['R0','R1','R2'])
@pytest.mark.asyncio
async def test_five_calls_same_visible_packets_no_private_path(monkeypatch,tmp_path,mode):
    args,item,ev,info=fixture(tmp_path);item['condition']=mode
    requests=install(monkeypatch,mode)
    r=await runner.episode(args,item,None,ev,info)
    assert r['api_calls']==5 and r['semantic_exact'] and r['runtime_remaining'] is False
    assert [runner.count_message_images(p['messages']) for p in requests]==[2,2,4,4,4]
    first=json.dumps(requests[0]['messages'])
    assert 'private-level3-task' not in first and str(tmp_path) not in first
    assert '"frame_id": "f2"' not in first and 'only_visible' in first
    assert requests[0]['max_tokens']==65536-4*2048
    if mode=='R2':assert r['phase_success']==[True]*4 and r['adherent']

@pytest.mark.asyncio
async def test_rejected_candidate_keeps_raw_score(monkeypatch,tmp_path):
    args,item,ev,info=fixture(tmp_path)
    requests=install(monkeypatch,'R2',inconsistent=True)
    r=await runner.episode(args,item,None,ev,info)
    assert len(requests)==5 and not r['harness_accepted'] and r['raw_FSA']==1
    assert r['raw_semantic_metrics']['spatial_errors']==1
    assert (args.output/r['session_id']/'Result.json').exists()

@pytest.mark.asyncio
async def test_provider_failure_is_unscored_not_a_model_failure(monkeypatch,tmp_path):
    args,item,ev,info=fixture(tmp_path);requests=install(monkeypatch,'R2',fail_at=2)
    r=await runner.episode(args,item,None,ev,info)
    assert r['status']=='budget_pause' and 'semantic_exact' not in r and len(requests)==3
    assert r['runtime_remaining'] is False

def test_schedule_and_terminal_resume_behavior(tmp_path):
    tasks=[{'id':str(i)} for i in range(9)];s=runner.build_schedule(tasks)
    assert len(s)==54 and len({runner.episode_key(r) for r in s})==54
    assert all(r['variant']=='complex' for r in s) and s==runner.build_schedule(tasks)
    failure={'episode_key':'failure','semantic_exact':False,'runtime_remaining':False}
    assert runner.completed_attempts([failure,{'episode_key':'infra','status':'infrastructure_error'}])==[failure]
    with pytest.raises(ValueError):runner.completed_attempts([failure,failure])

def test_comparisons_keep_task_unit_and_no_significance_claim(tmp_path):
    schedule=runner.build_schedule([{'id':str(i)} for i in range(9)])
    rows=[dict(item,episode_key=runner.episode_key(item),session_id=str(i),semantic_exact=item['condition']=='R2',raw_semantic_exact=True,runtime_remaining=False) for i,item in enumerate(schedule)]
    result=runner.summarize(tmp_path,rows,schedule)
    assert result['complete'] and result['comparisons'][0]['task_clusters']==9
    assert result['comparisons'][0]['mean_difference']==1
    assert 'cluster_bootstrap_97_5_interval' not in result['comparisons'][0]

def test_projection_gate_does_not_use_semantic_accuracy():
    source=Path(runner.__file__).read_text().split('if args.development:')[-1]
    assert "all(r['adherent']" in source and "54*statistics.mean" in source
