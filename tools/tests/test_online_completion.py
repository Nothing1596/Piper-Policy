import copy
import pytest
from online_completion import retreat_evidence,stable_after_retreat


def chain():
    return [{'tool':'set_gripper','actual_result':{'status':'succeeded'},
      'state_before':{'robot':{'gripper_width_m':.03}},
      'state_after':{'robot':{'gripper_width_m':.07,'tcp_pos':[0,0,0]}}},
      {'tool':'move_linear','actual_result':{'status':'succeeded'},
       'state_after':{'robot':{'gripper_width_m':.07,'tcp_pos':[0,0,.04]}}}]


def samples():
    return [{'sim_time_s':t,'epoch':0,'bilateral_contact_objects':[], 'contacts':[],
       'objects':{'object':{'kind':'dynamic','position':[0,0,0],'quaternion':[1,0,0,0],'speed_m_s':0}}}
       for t in (10,11.1,12.2)]


def test_release_without_retreat_and_failed_retreat_are_not_complete():
    c=chain();assert not retreat_evidence(c[:1])['ok'];c[1]['actual_result']['status']='failed'
    assert not retreat_evidence(c)['ok']
    c=chain();c[1]['state_after']['robot']['tcp_pos'][2]=.029
    assert not retreat_evidence(c)['ok']


def test_clear_stable_retreat_passes():
    assert stable_after_retreat(samples(),chain(),['object'])['ok']


@pytest.mark.parametrize('case',['single_contact','drift','rotation','speed','nonadvancing','epoch','missing'])
def test_final_contact_or_unstable_or_missing_evidence_fails(case):
    s=samples()
    if case=='single_contact':s[1]['contacts']=[{'object1':'gripper_link1_geom','object2':'object','normal_force_n':.001}]
    if case=='drift':s[-1]['objects']['object']['position'][0]=.003
    if case=='rotation':s[-1]['objects']['object']['quaternion']=[.999,0,0,.0447]
    if case=='speed':s[1]['objects']['object']['speed_m_s']=.006
    if case=='nonadvancing':s[1]['sim_time_s']=s[0]['sim_time_s']
    if case=='epoch':s[1]['epoch']=1
    if case=='missing':del s[1]['contacts']
    assert not stable_after_retreat(s,chain(),['object'])['ok']


@pytest.mark.asyncio
async def test_model_plan_saved_and_read_back_with_real_version(tmp_path):
    from test_longvil_online import fixture,Session
    import run_longvil_online as online
    args,evidence=fixture(tmp_path);args.max_rounds=2;session=Session();events=[];payloads=[]
    async def decide(args,client,payload,ident):
        payloads.append(copy.deepcopy(payload))
        return {'model':'test','usage':{'completion_tokens':10},'plan_update':'Steps: pick red; place left. Next: observe; uncertain contact.',
          'choices':[{'finish_reason':'tool_calls','message':{'role':'assistant','content':'current',
          'tool_calls':[{'id':ident,'function':{'name':'camera_observe','arguments':'{}'}}]}}]}
    args.model_decider=decide
    report={};await online.online_loop(session,args,evidence,events.append,tmp_path,report,None)
    updates=[b for n,b in session.calls if n=='task_plan' and b['operation']=='update']
    assert [b['expected_version'] for b in updates]==[1,2]
    assert report['model_authored_plan_updates']==2
    assert 'Steps: pick red' in payloads[1]['messages'][-1]['content']
    assert '"version": 2' in payloads[1]['messages'][-1]['content']


def test_exact_two_second_boundary_and_empty_chain():
    s=samples();s[-1]['sim_time_s']=12.
    assert stable_after_retreat(s,chain(),['object'])['ok']
    assert not retreat_evidence([])['ok']


def test_old_single_result_does_not_pass_new_finish_rule():
    # A fixture approximating the observed single-finger support after opening.
    s=samples();s[-1]['contacts']=[{'object1':'object','object2':'gripper_link1_geom','normal_force_n':.395}]
    assert not stable_after_retreat(s,chain()[:1],['object'])['ok']


@pytest.mark.asyncio
async def test_operator_goal_is_identical_in_plan_and_initial_model_context(tmp_path):
    import json
    from test_longvil_online import fixture,Session
    import run_longvil_online as online
    args,evidence=fixture(tmp_path);args.max_rounds=0
    args.task_goal='Place the selected object on its support; earlier demonstrated step is already complete.'
    session=Session()
    await online.online_loop(session,args,evidence,lambda row:None,tmp_path,{},None)
    create=next(body for name,body in session.calls if name=='task_plan' and body['operation']=='create')
    initial=json.loads((tmp_path/'initial-model-context.json').read_text())
    assert create['goal']==initial['goal']==args.task_goal


@pytest.mark.asyncio
async def test_contact_pilot_plan_does_not_request_entire_demonstration(tmp_path):
    from test_longvil_online import fixture,Session
    import run_longvil_online as online
    args,evidence=fixture(tmp_path);args.max_rounds=0;args.contact_smoke=True
    session=Session()
    await online.online_loop(session,args,evidence,lambda row:None,tmp_path,{},None)
    create=next(body for name,body in session.calls if name=='task_plan' and body['operation']=='create')
    assert create['goal']==online.CONTACT_GOAL
