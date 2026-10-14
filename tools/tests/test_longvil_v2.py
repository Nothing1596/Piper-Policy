"""Semantic scoring, strict evidence delivery, and spending guardrails."""
from datetime import datetime,timezone
import importlib.util
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]))
from longvil_metrics import semantic_actions,compare_actions
from longvil_budget import Ledger,BudgetPause,usage_cost
from piperx_middleware.observation_context import label_demo_images


def program(var='p',reference='box',relation='into'):
    return [f"{var}=getpos('cube')",f'moveto({var})','pick()',f"q=getpos('{reference}','{relation}')",'moveto(q)','place()']


def test_semantics_ignore_variables_not_relations_or_order():
    reference=semantic_actions(program()+['open()','close()'])
    assert reference==semantic_actions(program('renamed')+['open()','close()'])
    reused=program()+["p=getpos('cube')",'moveto(p)','pick()','moveto(q)','place()']
    assert len(semantic_actions(reused))==2
    wrong=semantic_actions(program(relation='left')+['close()','open()'])
    result=compare_actions(reference,wrong)
    assert not result['semantic_exact'] and result['spatial_errors']==1
    assert compare_actions(reference,list(reversed(reference)))['order_mismatch']
    assert compare_actions(reference,reference[:-1])['missing_actions']==1
    assert compare_actions(reference,reference+[('open',)])['extra_actions']==1


@pytest.mark.parametrize('bad',[['print(1)'],['import os'],['open(1)'],["p=getpos('cube')",'moveto(p)','pick()'],program()+['while True: pass']])
def test_generated_program_never_executes_or_silently_discards_statements(bad):
    with pytest.raises(ValueError):semantic_actions(bad)


def test_picture_labels_include_source_time_and_recorded_role():
    result={'source':{'sha256':'source'},'frames':[{'frame_id':'f0','timestamp_s':.2},{'frame_id':'f1','timestamp_s':1.3}]}
    images=[{'type':'image_url','image_url':{'url':'data:image/jpeg;base64,AA=='}}]*2
    parts,manifest=label_demo_images(result,images,initial=True)
    assert 'f0 at 0.2s' in parts[0]['text'] and 'NOT live' in parts[0]['text']
    assert manifest[0]['role']=='recorded_initial' and manifest[1]['role']=='demo'
    with pytest.raises(ValueError,match='count mismatch'):label_demo_images(result,images[:1])


def test_peak_crossing_missing_usage_and_duplicate_billing(tmp_path):
    start=datetime(2026,10,9,0,59,tzinfo=timezone.utc);end=datetime(2026,10,9,1,1,tzinfo=timezone.utc)
    usage={'prompt_cache_hit_tokens':1000000,'prompt_cache_miss_tokens':1000000,'completion_tokens':1000000}
    assert usage_cost(usage,start,end)==10.04
    night=datetime(2026,10,9,12,tzinfo=timezone.utc)
    assert usage_cost(usage,night,night)==5.02
    ledger=Ledger(tmp_path/'fees.jsonl',limit=75,stop_at=72)
    ledger.record(request_id='one',phase='dev',usage=usage,start=night,end=night)
    with pytest.raises(ValueError,match='Duplicate'):ledger.record(request_id='one',phase='dev',usage=usage,start=night,end=night)
    with pytest.raises(BudgetPause):ledger.record(request_id='lost',phase='dev',usage=None,start=night,end=night)
    assert ledger.total==8.02
    with pytest.raises(BudgetPause):Ledger(tmp_path/'fees.jsonl').check()


def test_batched_queries_keep_two_groups_and_initial_pixels():
    from piperx_middleware.observation_context import append_demo_group
    pixels=[{'type':'image_url','image_url':{'url':'initial'}}]*12
    messages=[{'role':'system','content':'rules'},{'role':'user','content':pixels}];recent=[]
    for i in range(5):append_demo_group(messages,recent,[{'type':'image_url','image_url':{'url':str(i)}}]*12)
    assert len(recent)==2
    assert messages[1]['content']==pixels
    assert sum(sum(p.get('type')=='image_url' for p in m['content']) for m in messages if isinstance(m['content'],list))==36


def test_formal_schedule_and_private_review_gate(tmp_path):
    spec=importlib.util.spec_from_file_location('v2',Path(__file__).parents[1]/'run_longvil_v2.py')
    v2=importlib.util.module_from_spec(spec);spec.loader.exec_module(v2)
    tasks=[{'id':f't-{l}-{cat}-{i}','level':str(l),'category':cat} for l in range(1,4) for cat in ('blockmanipulation','tabletopcleanup','vegetablesorting') for i in range(3)]
    schedule=v2.build_schedule(tasks)
    assert len(schedule)==378 and len({v2.episode_key(s) for s in schedule})==378
    assert sum(s['pipeline']=='cv' for s in schedule)==324
    assert sum(s['pipeline']=='initial_vlm' for s in schedule)==54
    manifest=tmp_path/'pending.json';manifest.write_text('{"tasks":[],"approval":"pending"}')
    with pytest.raises(ValueError,match='approved'):v2.load_manifest(manifest,False)


def test_visual_citations_sort_only_known_refs_never_stage_order():
    from piperlab.video.watch import canonicalize_visual_references
    from piperlab.demonstration.compiler import _validate_semantic_response
    frames=[{'frame_id':'a','timestamp_s':0},{'frame_id':'b','timestamp_s':1}]
    stage=lambda refs:dict(id='s',operation='observe',object_roles=[],preconditions=[],expected_effects=[],uncertainty='',evidence_refs=refs)
    raw=dict(stages=[stage(['b','a'])],goal_constraints=[],unknowns=[],outcome_verdict='unknown',summary='')
    value,changes=canonicalize_visual_references(raw,frames)
    assert value['stages'][0]['evidence_refs']==['a','b'] and changes
    assert raw['stages'][0]['evidence_refs']==['b','a']
    bad={**raw,'stages':[stage(['b']),stage(['a'])]}
    fixed,_=canonicalize_visual_references(bad,frames)
    assert _validate_semantic_response(fixed,['a','b'],{'a':0,'b':1})
    unknown={**raw,'stages':[stage(['unknown','a'])]}
    fixed,changes=canonicalize_visual_references(unknown,frames)
    assert not changes and _validate_semantic_response(fixed,['a','b'],{'a':0,'b':1})


def runner():
    spec=importlib.util.spec_from_file_location('v2test',Path(__file__).parents[1]/'run_longvil_v2.py')
    v2=importlib.util.module_from_spec(spec);spec.loader.exec_module(v2);return v2


def test_resume_preserves_scored_failure_and_rejects_duplicate_scores():
    v2=runner()
    failure={'episode_key':'scored-failure','semantic_exact':False,'runtime_remaining':False}
    incomplete={'episode_key':'infra','status':'infrastructure_error','runtime_remaining':False}
    dirty={'episode_key':'leaked','semantic_exact':True,'runtime_remaining':True}
    attempts=[failure,incomplete,dirty]
    assert v2.completed_attempts(attempts)==[failure] and len(attempts)==3
    with pytest.raises(ValueError,match='Duplicate'):v2.completed_attempts([failure,failure])


def test_shared_budget_lock_and_stop_threshold(tmp_path):
    from longvil_budget import ledger_lock
    path=tmp_path/'ledger.jsonl'
    with ledger_lock(path):
        with pytest.raises(BudgetPause,match='Another'):
            with ledger_lock(path):pass
    with ledger_lock(path):pass
    path.write_text('{"status":"accounted","estimated_cny":72}\n')
    with pytest.raises(BudgetPause,match='checkpoint'):Ledger(path).check()


def test_complete_cluster_statistics_and_fixed_first_repeat_vision_pairs(tmp_path):
    v2=runner()
    tasks=[{'id':f't-{l}-{cat}-{i}','level':str(l),'category':cat} for l in range(1,4) for cat in ('block','cleanup','veg') for i in range(3)]
    schedule=v2.build_schedule(tasks);rows=[]
    for i,item in enumerate(schedule):
        success=item['condition']!='baseline' if item['pipeline']=='cv' else item['condition']=='baseline'
        rows.append({**item,'episode_key':v2.episode_key(item),'session_id':str(i),'semantic_exact':success,'runtime_remaining':False,'elapsed_s':1,'api_calls':1,'tool_calls':1,'watch_calls':0,'reasoning_tokens':1})
    result=v2.summarize(tmp_path,rows,schedule)
    assert result['complete'] and len(result['comparisons'])==2
    assert result['comparisons'][0]['task_clusters']==27
    assert result['comparisons'][0]['cluster_bootstrap_97_5_interval']==[1.,1.]
    comparisons={x['condition']:x for x in result['initial_vision_pairs']}
    assert comparisons['baseline']['improved']==18 and comparisons['skill']['degraded']==18
    assert all(x['task_count']==9 for x in comparisons.values())


def test_builder_refuses_development_gate_before_any_mutation(tmp_path):
    from build_longvil_v2_job import preflight
    dev=tmp_path/'dev';dev.mkdir();(dev/'development-gates.json').write_text('{"complete":true,"syntax_gate_passed":true,"cost_gate_passed":true}')
    with pytest.raises(ValueError,match='vision_gate_passed'):preflight(tmp_path,dev)


def test_plan_verification_follows_issued_active_task_not_stale_initial_id():
    v2=runner();initial={'task_id':'initial','version':1,'interpretation':''}
    replacement={'task_id':'new-issued-id','version':2,'interpretation':'f1 supports relation'}
    assert not v2.verified_plan(initial,{'initial'})
    assert v2.verified_plan(replacement,{'initial','new-issued-id'})
    assert not v2.verified_plan(replacement,{'initial'})


def test_detached_builder_freezes_data_and_records_child_exit(tmp_path,monkeypatch):
    import json,time
    from types import SimpleNamespace
    import build_longvil_v2_job as builder
    repo=tmp_path/'repo';root=tmp_path/'audit';dev=root/'dev';dev.mkdir(parents=True)
    for part in ('piperx-cli/src','piper-lab/src','tools'):
        p=repo/part;p.mkdir(parents=True);(p/'module.py').write_text('# fixture\n')
    annotation=root/'gt.json';annotation.write_text('{"object_list":[]}')
    video=root/'video.mp4';video.write_bytes(b'fixture video')
    tasks=[{'id':'t','annotation':str(annotation),'videos':{'clean':{'path':str(video)}}}]
    (root/'formal-manifest.json').write_text(json.dumps({'tasks':tasks}))
    for name in ('codex-review-ledger.jsonl','user-checklist.json','ground-truth-selfcheck.json','prompt-revision.json'):(root/name).write_text('{}')
    evaluator=root/'evaluate.py';evaluator.write_text('# fixture')
    detector=root/'detector.onnx';detector.write_bytes(b'fixture')
    monkeypatch.setattr(builder,'preflight',lambda *a:({'complete':True},tasks))
    monkeypatch.setattr(builder,'build_schedule',lambda *a:[{'task':'t'}])
    args=SimpleNamespace(root=root,development=dev,repo=repo,job=tmp_path/'job',evaluator=evaluator,detector=detector,node=Path('/usr/bin/true'),key_launcher=Path('/unused'),cli=Path('/unused'),video_python=Path('/unused'),submit=True)
    result=builder.build(args)
    video.write_bytes(b'changed original')
    assert (args.job/'private-data/t/clean.mp4').read_bytes()==b'fixture video'
    assert result['state']=='submitted'
    deadline=time.monotonic()+5
    while not (args.job/'exit.json').exists() and time.monotonic()<deadline:time.sleep(.02)
    assert json.loads((args.job/'exit.json').read_text())['exit_code']==0
    assert json.loads((args.job/'job.json').read_text())['state']=='submitted'


def test_explicit_conservative_settlement_is_append_only_and_keeps_charge(tmp_path):
    ledger=Ledger(tmp_path/'ledger.jsonl');now=datetime(2026,10,10,tzinfo=timezone.utc)
    with pytest.raises(BudgetPause):ledger.record(request_id='unknown',phase='formal',usage=None,start=now,end=now)
    before=ledger.path.read_bytes()
    with pytest.raises(ValueError):ledger.settle_conservative('not-issued')
    ledger.settle_conservative('unknown')
    assert ledger.path.read_bytes().startswith(before) and ledger.rows[0]['usage'] is None
    assert ledger.total==3 and ledger.rows[0]['status']=='usage_unavailable'
    Ledger(ledger.path).check()
    with pytest.raises(ValueError,match='Already'):ledger.settle_conservative('unknown')
    with pytest.raises(BudgetPause):ledger.record(request_id='second-unknown',phase='formal',usage=None,start=now,end=now)
    with pytest.raises(BudgetPause):ledger.check()


def test_visual_source_citation_moves_to_audit_but_never_hides_bad_frames():
    from piperlab.video.watch import canonicalize_visual_references
    from piperlab.demonstration.compiler import _validate_semantic_response
    frames=[{'frame_id':'a','timestamp_s':0},{'frame_id':'b','timestamp_s':1}];sha='1'*64
    stage=lambda refs:dict(id='s',operation='observe',object_roles=[],preconditions=[],expected_effects=[],uncertainty='',evidence_refs=refs)
    value=dict(stages=[stage(['b',sha,'a'])],goal_constraints=[],unknowns=[],outcome_verdict='unknown',summary='state')
    fixed,audit=canonicalize_visual_references(value,frames,sha)
    assert fixed['stages'][0]['evidence_refs']==['a','b'] and audit[0]['source_refs']==[sha]
    assert value['stages'][0]['evidence_refs']==['b',sha,'a']
    assert not _validate_semantic_response(fixed,['a','b'],{'a':0,'b':1})
    for refs in ([sha],['a',sha,'invented'],['a','2'*64]):
        bad={**value,'stages':[stage(refs)]};fixed,audit=canonicalize_visual_references(bad,frames,sha)
        assert fixed==bad and not audit and _validate_semantic_response(fixed,['a','b'],{'a':0,'b':1})
