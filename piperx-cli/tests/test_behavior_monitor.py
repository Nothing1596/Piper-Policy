"""Diagnostics from public feedback; no robot/model or private scoring."""
import json
import pytest
from piperx_middleware.behavior_monitor import BehaviorMonitor, completion_issues, public_receipt, reviewed_prefix


def plan(version=1, steps=None, **extra):
    ledger={'steps':steps if steps is not None else [step('s1')],**extra}
    return {'task_id':'task-1','version':version,'interpretation':json.dumps(ledger),'current_stage':'execute'}


def step(ident, status='pending', refs=None):
    return {'id':ident,'action':['pick_place','a','b','top'],'status':status,'evidence_refs':['f1'],'execution_refs':refs or []}


def query(m, pixel=(1,2), view='live-1', result=None):
    m.record_result('scene_query',{'operation':'surface_point','observation_path':view,'pixel':list(pixel)},
                    result or {'status':'ok','point_m':[.1,.2,.3]})


def test_repeat_requires_same_view_arguments_and_answer():
    m=BehaviorMonitor();m.record_plan(plan())
    for i in range(3):query(m,result={'status':'ok','point_m':[.1,.2,.3],'elapsed_s':i})
    assert m.snapshot()['state']=='needs_review'
    assert m.snapshot()['issues'][0]['code']=='repeated_query'
    m=BehaviorMonitor()
    for i in range(4):query(m,view=f'live-{i}')
    assert not m.snapshot()['issues']
    for i in range(4):query(m,pixel=(i,3))
    assert not m.snapshot()['issues']


def test_partial_motion_followed_by_fresh_continuation_is_not_a_loop():
    m=BehaviorMonitor()
    m.record_result('move_linear',{'request_id':'first','xyz_m':[.3,0,.1],'observation_id':'live-1'},
                    {'status':'failed','error_code':'motion_blocked'},result_observation_id='live-2')
    m.record_result('move_linear',{'request_id':'next','xyz_m':[.3,0,.1],'observation_id':'live-2','timeout_s':60},
                    {'status':'succeeded'},result_observation_id='live-3')
    assert m.snapshot()['state']=='working'


def test_same_rejection_with_new_request_ids_is_detected():
    m=BehaviorMonitor()
    for i in range(3):m.record_result('move_to',{'request_id':f'move-{i}','xyz_m':[.3,0,.1],'observation_id':'live-1'},
                                      {'error':{'code':'joint_limits','message':'out of range'}})
    assert m.snapshot()['issues'][0]['code']=='ignored_feedback'


def test_unchanged_ab_query_cycle_but_not_changed_evidence():
    m=BehaviorMonitor()
    for point in [(1,2),(3,4),(1,2),(3,4)]:query(m,pixel=point)
    assert any(i['code']=='query_cycle' for i in m.snapshot()['issues'])
    m=BehaviorMonitor()
    for i,point in enumerate([(1,2),(3,4),(1,2),(3,4)]):query(m,pixel=point,result={'status':'ok','point_m':[i,0,0]})
    assert not m.snapshot()['issues']


def test_stage_change_and_same_plan_do_not_clear_review():
    m=BehaviorMonitor();p=plan();m.record_plan(p)
    for i in range(3):query(m)
    p.update(version=2,current_stage='verify');m.record_plan(p)
    assert m.snapshot()['state']=='needs_review'
    m.record_plan(plan(3,notes='Try a different visible edge'))
    assert m.snapshot()['state']=='verifying'
    query(m)
    assert m.snapshot()['state']=='needs_independent_review'
    # No supervisor is called by the monitor.
    m.record_plan(plan(4,notes='Changed selection based on live evidence'))
    query(m,pixel=(8,9))
    assert m.snapshot()['state']=='working'


def test_plan_shrink_stays_visible_until_steps_are_restored():
    m=BehaviorMonitor();m.record_plan(plan(steps=[step('s1'),step('s2')]))
    m.record_plan(plan(2,[step('s2')]))
    assert m.snapshot()['issues'][0]['code']=='plan_shrink'
    m.record_plan(plan(3,[step('s2')],notes='reviewed'))
    assert any(i['code']=='plan_shrink' for i in m.snapshot()['issues'])
    m.record_plan(plan(4,[step('s1'),step('s2')]))
    assert not m.snapshot()['issues']


def test_wording_revision_and_restoration_do_not_invent_removed_steps():
    m=BehaviorMonitor();original=step('s1');original['action'][-1]='move_beside_orange_cube'
    m.record_plan(plan(steps=[original,step('s2')]))
    changed=dict(original,action=[*original['action'][:-1],'beside_orange_cube'])
    m.record_plan(plan(2,[changed,step('s2')]))
    m.record_plan(plan(3,[original,step('s2')]))
    assert not m.snapshot()['issues']
    m.record_plan(plan(4,[step('s2')]))
    assert m.snapshot()['issues'][0]['code']=='plan_shrink'
    m.record_plan(plan(5,[changed,step('s2')]))
    assert not m.snapshot()['issues']  # Restoration does not require identical prose.


def test_failed_only_reference_cannot_support_completed_step_but_recovery_can():
    receipts={'bad':public_receipt({'status':'failed','error_code':'motion_blocked'}),
              'good':public_receipt({'status':'succeeded'}),
              'unknown':public_receipt({'status':'outcome_unknown'})}
    assert completion_issues({'steps':[step('s1','completed',['bad'])]},receipts)
    assert completion_issues({'steps':[step('s1','completed',['unknown'])]},receipts)
    assert not completion_issues({'steps':[step('s1','completed',['bad','good'])]},receipts)
    # Successful primitive evidence is not promoted to a semantic success verdict.


def test_checkpoint_requires_specific_execution_reference_not_stage_edit():
    chain=[{'tool':'move_linear','model_arguments':{'request_id':'blocked-1'},
            'actual_result':{'status':'failed','error_code':'motion_blocked'},'result_observation_id':'live-2'}]
    assert reviewed_prefix(plan(),chain)==0
    p=plan();p['current_stage']='verify';assert reviewed_prefix(p,chain)==0
    assert reviewed_prefix(plan(2,[step('s1','uncertain',['blocked-1'])]),chain)==1


def test_polling_and_idempotent_result_do_not_record_another_execution():
    m=BehaviorMonitor();body={'request_id':'motion-1','xyz_m':[.3,0,.1]}
    m.record_result('move_to',body,{'status':'accepted','job_id':'j1'})
    m.record_result('robot_status',{'job_id':'j1'},{'status':'succeeded','job_id':'j1'})
    count=len(m.rows)
    for _ in range(5):m.record_result('robot_status',{'job_id':'j1'},{'status':'succeeded','job_id':'j1'})
    m.record_result('move_to',body,{'status':'succeeded','job_id':'j1'})
    assert len(m.rows)==count and m.receipts['motion-1']['status']=='succeeded'
    assert not m.snapshot()['issues']


def test_unknown_outcome_not_suggested_as_retry_and_new_task_isolated():
    m=BehaviorMonitor();m.record_plan(plan())
    m.record_result('move_to',{'request_id':'unknown-1'},{'status':'outcome_unknown'})
    assert m.receipts['unknown-1']['unknown']
    for _ in range(3):query(m)
    m.record_result('task_plan',{'operation':'get'},{'task_id':'old-task','version':1,'interpretation':''})
    assert m.task_id=='task-1' and m.snapshot()['issues']
    m.record_result('task_plan',{'operation':'create'},{'task_id':'new-task','version':1,'interpretation':''})
    assert m.task_id=='new-task' and not m.snapshot()['issues'] and not m.receipts


def test_same_plan_read_is_not_another_failed_review():
    m=BehaviorMonitor();p=plan(steps=[step('s1','completed',['absent'])]);m.record_plan(p)
    before=m.snapshot()
    for _ in range(4):m.record_plan(p)
    assert m.snapshot()==before


def test_unknown_query_can_reconcile_receipt_without_counting_another_action():
    m=BehaviorMonitor()
    m.record_result('move_to',{'request_id':'m1'}, {'status':'outcome_unknown','job_id':'j1'})
    count=len(m.rows)
    m.record_result('robot_status',{'job_id':'j1'},{'status':'succeeded','job_id':'j1'})
    assert len(m.rows)==count and m.receipts['m1']['status']=='succeeded'
    assert not m.receipts['m1']['unknown']


def test_same_failed_request_id_still_gets_repetition_feedback_but_polls_do_not():
    m=BehaviorMonitor();body={'request_id':'bad-1','xyz_m':[1,2,3]}
    value={'status':'failed','error_code':'joint_limits','job_id':'j1'}
    m.record_result('move_to',body,value)
    for _ in range(4):m.record_result('robot_status',{'job_id':'j1'},value)
    assert not m.snapshot()['issues']
    for _ in range(2):m.record_result('move_to',body,value)
    assert m.snapshot()['issues'][0]['code']=='ignored_feedback'


def test_live_captures_use_immutable_observation_identity_without_simulator_clock():
    m=BehaviorMonitor()
    for i in range(4):
        m.record_result('camera_observe',{}, {'frames':[{'camera_serial':'D435','sequence':i,
                         'captured_at':f'time-{i}'}],'observation_path':f'/frames/{i}/observation.json'})
        m.record_result('scene_query',{'operation':'surface_point','pixel':[1,2]}, {'status':'ok','point_m':[1,2,3]})
    assert not m.snapshot()['issues']


def test_resumed_plan_missing_local_receipts_is_not_called_an_execution_failure():
    m=BehaviorMonitor();m.record_plan(plan(steps=[step('s1','completed',['previous-session-motion'])]))
    issue=m.snapshot()['issues'][0]
    assert 'unavailable in this monitor session' in issue['message']
    assert 'do not replay' in issue['message']
