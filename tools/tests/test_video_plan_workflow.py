import json
from types import SimpleNamespace
import pytest
from video_plan_workflow import review_context,VIDEO_PLAN_GUIDANCE


def test_public_review_preserves_repeated_steps_and_uncertainty():
    steps=[{'id':'s1','action':['pick_place','a','b','top'],'status':'completed','evidence_refs':['f1'],'execution_refs':['motion-1']},
           {'id':'s2','action':['pick_place','a','c','left'],'status':'uncertain','evidence_refs':['f2'],'execution_refs':[]}]
    result=review_context({'interpretation':json.dumps({'steps':steps})},[],[],{'live-2'},0,['f1','f2'])
    assert result['ledger']['steps']==steps
    assert result['review_reasons']==[]


def test_reminder_requires_model_review_not_intended_endpoint_or_auto_completion():
    chain=[{'tool':'move_linear','model_arguments':{'request_id':'move-1'},'actual_result':{'status':'failed','error_code':'motion_blocked'},'result_observation_id':'live-2'},
           {'tool':'set_gripper','model_arguments':{'request_id':'open-1','width_m':.07},'actual_result':{'status':'succeeded'},'result_observation_id':'live-3'}]
    plan={'interpretation':'{"steps":[{"id":"s1","status":"pending","evidence_refs":["f1"]}]}'}
    pending=review_context(plan,chain,[],{'live-3'},0)
    assert len(pending['review_reasons'])==2 and pending['ledger']['steps'][0]['status']=='pending'
    assert pending['execution_checkpoints'][0]['observation_id']=='live-2'
    reviewed=review_context(plan,chain,[],{'live-3'},2)
    assert reviewed['review_reasons']==[]
    assert len(reviewed['execution_checkpoints'])==2


def test_measurement_index_reuses_history_without_recertifying_it():
    m=[{'tool':'scene_query','source_observation_id':'live-1','arguments':{'pixel':[1,2]},'evidence':{'point_m':[.1,.2,.3]}},
       {'tool':'grasp_target','source_observation_id':'live-2','arguments':{'roi':[1,2,3,4]},'evidence':{'source_measurement_id':'g-1','observed_width_m':.05}}]
    result=review_context({},[],m+m,{'live-2'},0)
    assert len(result['measurement_index'])==2
    assert [x['matches_current_view'] for x in result['measurement_index']]==[False,True]
    assert result['measurement_index'][0]['evidence']['point_m']==[.1,.2,.3]
    assert 'maximum 2000' in VIDEO_PLAN_GUIDANCE
    assert 'repeated moves' in VIDEO_PLAN_GUIDANCE


def test_ledger_evidence_must_be_delivered_and_no_completion_is_invented():
    plan={'interpretation':json.dumps({'steps':[{'id':'s1','status':'completed','evidence_refs':['invented']} ]})}
    r=review_context(plan,[],[],set(),0,['f1'])
    assert r['ledger_issues'] and r['ledger']['steps'][0]['status']=='completed'
    assert 'invented' not in r['delivered_demo_frames']


@pytest.mark.asyncio
async def test_workflow_saves_model_sequence_and_returns_it_with_fresh_version(tmp_path,monkeypatch):
    import run_longvil_online as online
    from test_longvil_online import Session,fixture
    from test_longvil_softplan_recovery import decision,driver
    args,e=fixture(tmp_path);args.video_plan_workflow=True;args.max_rounds=3
    sess=Session();payloads=[]
    ledger={'steps':[{'id':'s1','action':['pick_place','red cylinder','green cuboid','left'],
                     'evidence_refs':['f0'],'status':'pending','execution_refs':[]}]}
    calls=[decision('task_plan',{'operation':'update','task_id':sess.plan['task_id'],'expected_version':1,
                                 'changes':{'interpretation':json.dumps(ledger),'current_stage':'execute_review'}},'plan-1'),
           decision('move_to',{'request_id':'motion-1','xyz_m':[.3,0,.2],'observation_id':'live-0001'},'motion-1'),
           decision('finish_task',{'outcome':'aborted','summary':'diagnostic','uncertainties':[],'observation_id':'live-0002'},'end')]
    monkeypatch.setattr(online,'paid',driver(calls,payloads))
    async def remote(*a):return {'sim_time_s':sess.time,'objects':{},'bilateral_contact_objects':[]}
    monkeypatch.setattr(online,'remote',remote)
    report={};await online.online_loop(sess,args,e,lambda x:None,tmp_path,report,None)
    progress=json.loads(payloads[1]['messages'][-1]['content'].split('CURRENT execution_progress: ',1)[1])
    assert progress['video_plan_review']['ledger']==ledger
    assert progress['video_plan_review']['ledger_issues']==[]
    assert progress['current_plan']['version']==2
    assert report['plan_writes']==1 and report['motions']==1
    assert (tmp_path/'video-plan-reviews.jsonl').exists()
    assert 'FULL demonstrated action sequence' in payloads[0]['messages'][0]['content']


def test_completion_citation_is_a_receipt_check_not_semantic_success():
    step={'id':'s1','status':'completed','evidence_refs':['f1'],'execution_refs':['made-up']}
    plan={'interpretation':json.dumps({'steps':[step]})}
    chain=[{'tool':'move_to','model_arguments':{'request_id':'real-move'},'actual_result':{'status':'succeeded'},'result_observation_id':'live-2'}]
    r=review_context(plan,chain,[],{'live-2'},0,['f1'])
    assert any('Completed steps' in s for s in r['ledger_issues'])
    step['execution_refs']=['real-move','live-2'];plan['interpretation']=json.dumps({'steps':[step]})
    assert review_context(plan,chain,[],{'live-2'},0,['f1'])['ledger_issues']==[]
    assert 'task_success' not in r


def test_same_pixels_on_distinct_captures_are_not_deduplicated():
    m=[{'tool':'scene_query','source_observation_id':f'live-{i}','arguments':{'pixel':[1,2]},'evidence':{'point_m':[i,0,0]}} for i in [1,2]]
    result=review_context({},[],m,{'live-2'},0)
    assert len(result['measurement_index'])==2
    assert [x['matches_current_view'] for x in result['measurement_index']]==[False,True]


@pytest.mark.asyncio
async def test_runner_exposes_repeat_diagnostic_without_extra_robot_or_model_calls(tmp_path,monkeypatch):
    import run_longvil_online as online
    from test_longvil_online import Session,fixture
    from test_longvil_softplan_recovery import decision,driver
    args,e=fixture(tmp_path);args.video_plan_workflow=True;args.max_rounds=4
    sess=Session();payloads=[];events=[]
    query={'operation':'surface_point','pixel':[10,20],'observation_path':'live-0001'}
    calls=[decision('scene_query',query,f'q-{i}') for i in range(3)]
    calls.append(decision('finish_task',{'outcome':'aborted','summary':'diagnostic','uncertainties':[],
                                        'observation_id':'live-0001'},'end'))
    monkeypatch.setattr(online,'paid',driver(calls,payloads))
    report={};await online.online_loop(sess,args,e,events.append,tmp_path,report,None)
    progress=json.loads(payloads[-1]['messages'][-1]['content'].split('CURRENT execution_progress: ',1)[1])
    assert progress['harness_review']['state']=='needs_review'
    assert progress['harness_review']['issues'][0]['code']=='repeated_query'
    assert report['api_calls']==4 and report['tool_calls']==4 and not sess.moves
    assert report['native_tool_calls']['scene_query']==1  # Existing cache, not new probes.


@pytest.mark.asyncio
async def test_stage_only_edit_does_not_acknowledge_blocked_execution(tmp_path,monkeypatch):
    import run_longvil_online as online
    from test_longvil_online import fixture
    from test_longvil_softplan_recovery import ContactWorld,ScriptedSession,blocked_result,decision,driver
    args,e=fixture(tmp_path);args.video_plan_workflow=True;args.max_rounds=4
    world=ContactWorld();sess=ScriptedSession(world,[{'result':blocked_result()}]);payloads=[]
    ledger={'steps':[{'id':'s1','action':['pick_place','red cylinder','green cuboid','top'],
                     'evidence_refs':['f0'],'status':'pending','execution_refs':[]}]}
    calls=[decision('task_plan',{'operation':'update','task_id':sess.plan['task_id'],'expected_version':1,
                                'changes':{'interpretation':json.dumps(ledger)}},'p1'),
           decision('move_to',{'request_id':'blocked-1','xyz_m':[.3,0,.2],'observation_id':'live-0001'},'b1'),
           decision('task_plan',{'operation':'update','task_id':sess.plan['task_id'],'expected_version':2,
                                'changes':{'current_stage':'verify'}},'p2'),
           decision('finish_task',{'outcome':'aborted','summary':'diagnostic','uncertainties':[],
                                   'observation_id':'live-0002'},'end')]
    monkeypatch.setattr(online,'paid',driver(calls,payloads))
    async def remote(*a):return world.read(sess.time)
    monkeypatch.setattr(online,'remote',remote)
    report={};await online.online_loop(sess,args,e,lambda _:None,tmp_path,report,None)
    progress=json.loads(payloads[-1]['messages'][-1]['content'].split('CURRENT execution_progress: ',1)[1])
    assert 'Review partial/blocked motion and remaining steps.' in progress['video_plan_review']['review_reasons']
    assert progress['current_plan']['version']==3 and report['confirmed_blocked_primitives']==1


@pytest.mark.asyncio
async def test_monitor_failure_never_obscures_executed_action_in_runner(tmp_path,monkeypatch):
    import run_longvil_online as online
    from piperx_middleware.behavior_monitor import BehaviorMonitor
    from test_longvil_online import Session,fixture
    from test_longvil_softplan_recovery import decision,driver
    args,e=fixture(tmp_path);args.video_plan_workflow=True;args.max_rounds=2
    sess=Session();payloads=[];events=[]
    calls=[decision('move_to',{'request_id':'known-1','xyz_m':[.3,0,.2],'observation_id':'live-0001'},'m1'),
           decision('finish_task',{'outcome':'aborted','summary':'diagnostic','uncertainties':[],
                                   'observation_id':'live-0002'},'end')]
    monkeypatch.setattr(online,'paid',driver(calls,payloads))
    def fail(*a,**kw):raise RuntimeError('diagnostic fixture failure')
    monkeypatch.setattr(BehaviorMonitor,'record_result',fail)
    async def remote(*a):return {'sim_time_s':sess.time,'objects':{},'bilateral_contact_objects':[]}
    monkeypatch.setattr(online,'remote',remote)
    report={};await online.online_loop(sess,args,e,events.append,tmp_path,report,None)
    assert len(sess.moves)==1 and sess.moves[0]['request_id']=='known-1'
    chain=json.loads((tmp_path/'action-chain.json').read_text())
    assert chain[0]['actual_result']['status']=='succeeded' and report['unknown_outcome'] is False
    assert any(r['kind']=='harness_review' and r['state']=='unavailable' for r in events)
