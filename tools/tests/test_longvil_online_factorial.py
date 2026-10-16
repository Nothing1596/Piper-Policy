import copy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).parents[1]))
import run_longvil_online as online
import run_longvil_online_pilot as pilot
from test_longvil_online import Session,fixture,plan_update


@pytest.mark.parametrize('plan,skill,effort',[(p,s,e) for p in ('off','on') for s in ('off','on') for e in ('low','high')])
@pytest.mark.asyncio
async def test_eight_conditions_share_feedback_and_calculators(plan,skill,effort,tmp_path,monkeypatch):
    args,evidence=fixture(tmp_path);args.plan=plan;args.skill=skill;args.effort=effort
    session=Session();payloads=[];sequence=[]
    if plan=='on':sequence=[('task_plan',{'operation':'get','task_id':session.plan['task_id']}),('task_plan',plan_update())]
    sequence+=[('move_to',{'request_id':'motion-0001','xyz_m':[.3,.1,.2],'observation_id':'live-0001'})]
    if plan=='on':sequence.append(('task_plan',{'operation':'get','task_id':session.plan['task_id']}))
    sequence.append(('finish_task',{'summary':'Observed result','uncertainties':[],'observation_id':'live-0002'}))
    args.max_rounds=len(sequence)
    async def paid(args,client,payload,ident):
        payloads.append(copy.deepcopy(payload));i=len(payloads)-1;name,body=sequence[i]
        return {'model':'deepseek-flash','usage':{'completion_tokens':100},'choices':[{'message':{'role':'assistant','content':'',
            'tool_calls':[{'id':'call-'+str(i),'function':{'name':name,'arguments':json.dumps(body)}}]}}]}
    async def remote(*values):return {'sim_time_s':session.time,'objects':{},'secret_gt':'PRIVATE_GT_ANSWER'}
    monkeypatch.setattr(online,'paid',paid);monkeypatch.setattr(online,'remote',remote);report={}
    await online.online_loop(session,args,evidence,lambda row:None,tmp_path,report,None)
    assert report['status']=='finished' and report['motions']==1 and report['observations']==2
    context=json.loads((tmp_path/'initial-model-context.json').read_text())
    assert bool(context['recipe']['skill_text'])==(skill=='on')
    assert bool(context['recipe']['plan_text'])==(plan=='on')
    # Legacy hard-cadence field stays None under the soft plan policy.
    assert report['plan_protocol_complete'] is None
    assert report['plan_policy']==('soft' if plan=='on' else None)
    assert report['api_calls']==len(sequence) and report['tool_calls']==len(sequence)
    for payload in payloads:
        assert payload['reasoning_effort']==effort and payload['thinking']['type']=='enabled'
        names={t['function']['name'] for t in payload['tools']}
        assert ('task_plan' in names)==(plan=='on')
        assert {'scene_query','grasp_target','placement_target','preview_move_to'}<=names
        assert 'PRIVATE_GT_ANSWER' not in json.dumps(payload)
        if plan=='off':assert 'current_plan' not in json.dumps(payload['messages'])
    syncs=len(sequence)+1 if plan=='on' else 0  # Includes post-action version read.
    assert report['plan_sync_reads']==syncs
    assert len([c for c in session.calls if c[0]=='task_plan'])==(1+syncs+3 if plan=='on' else 1)


def test_skill_is_identical_across_plan_and_effort_and_guide_cannot_leak():
    _,a=online.prompt_recipe(False,True);_,b=online.prompt_recipe(True,True)
    assert a['skill_text']==b['skill_text'] and a['skill_sha256']==b['skill_sha256']
    assert 'camera_observe(question_id' not in a['skill_text']
    from piperx_middleware.task_plan import TASK_PLAN_TOOL
    schema=online.select_tools([copy.deepcopy(TASK_PLAN_TOOL)])[0]['inputSchema']
    with pytest.raises(online.ValidationError):online.validate({'operation':'guide'},schema)


def test_plan_update_keeps_version_binding_without_read_cadence():
    g=online.OnlineGate();g.plan={'task_id':'real','version':1}
    # Soft policy: an update does not need a preceding get in an earlier decision.
    g.before('task_plan',{'operation':'update','expected_version':1},0)
    # Store version binding still applies: a stale expected_version is refused.
    with pytest.raises(ValueError,match='version'):g.before('task_plan',{'operation':'update','expected_version':2},1)
    assert g.plan_required
    g.accept_plan({'task_id':'real','version':2},'get',1)
    assert g.plan['version']==2 and not g.plan_required
    g.before('task_plan',{'operation':'update','expected_version':2},2)
    # A foreign update result never replaces the real task state.
    g.accept_plan({'task_id':'foreign','version':9},'update',2)
    assert g.plan['task_id']=='real' and g.plan['version']==2


def test_plan_off_release_does_not_trigger_hidden_plan_gate():
    g=online.OnlineGate(False);g.observation('latest');g.motions=1;g.last_motion_turn=0
    g.released({'robot':{'gripper_width_m':.03,'gripper_contact':{'supported':True,'finger1_force_n':5,'finger2_force_n':5}}},
               {'robot':{'gripper_width_m':.07}})
    assert not g.plan_required and 'current_plan' not in online.progress_context(g,[],[])
    g.before('finish_task',{'observation_id':'latest'},1)


def test_schedule_is_complete_fixed_and_development_excluded():
    cases=pilot.schedule();assert cases==pilot.schedule() and len(cases)==24
    assert len({c['id'] for c in cases})==24
    for phase,task in [('development',pilot.DEVELOPMENT_TASK),*[('formal',t) for t in pilot.FORMAL_TASKS]]:
        group=[c for c in cases if c['phase']==phase and c['task']==task]
        assert len(group)==8 and len({c['condition'] for c in group})==8


def test_development_requires_contact_evidence_but_no_plan_cadence():
    rows=[dict(c,interface_smoke_pass=True,contact_smoke=True,unknown_outcome=False,
        tool_calls=10,interface_errors=0,plan_reads=0,plan_writes=0,plan_protocol_complete=None,
        cleanup={'runtime_remaining':False,'owned_process_alive':False,'recording_error':None})
        for c in pilot.conditions()]
    assert all(pilot.development_gate(rows).values())
    assert not pilot.development_gate(rows[:7])['complete']
    rows[0]['contact_smoke']=False  # old empty-width interface smoke is insufficient
    assert not pilot.development_gate(rows)['contact']
    rows[0]['contact_smoke']=True;rows[0]['unknown_outcome']=True
    assert not pilot.development_gate(rows)['contact']


def test_paired_effects_use_only_complete_task_blocks():
    rows=[dict(c,task='one',physical_success=c['plan']=='on') for c in pilot.conditions()]
    rows.append({'task':'partial','plan':'on','skill':'on','effort':'high','physical_success':True})
    e=pilot.effects(rows);assert e['complete_task_blocks']==1
    assert e['contrasts']['plan']['mean_difference']==1
    assert e['contrasts']['skill']['mean_difference']==0 and e['contrasts']['effort']['mean_difference']==0


def test_execution_failure_is_not_infrastructure_and_bad_cleanup_is():
    r={'status':'paused','cleanup':{'runtime_remaining':False,'owned_process_alive':False,'recording_error':None},
       'exceptions':[{'type':'ModelCallAnomaly','message':'Action outcome unknown; no retry or subsequent motion'}]}
    assert pilot.classification(r)=='execution_failure'
    r['exceptions']=[{'type':'BudgetPause','message':'Usage unavailable'}]
    assert pilot.classification(r)=='infrastructure'
    r['status']='finished';r['cleanup']['owned_process_alive']=True
    assert pilot.classification(r)=='infrastructure'


def test_no_completed_failure_is_retried_by_schedule_logic():
    cases=pilot.schedule();done={c['id'] for c in cases[:4]}
    pending=[c for c in cases if c['id'] not in done]
    assert len(pending)==20 and not done.intersection(c['id'] for c in pending)
