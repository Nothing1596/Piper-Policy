"""Contact pilot must not accept tracker claims or a stable fall off the table."""
import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]))
from run_longvil_online import verify_transfer,contact_episode_completion
from online_completion import tabletop_return
from test_online_completion import chain


def state(t,z=0,held=False,table=True):
    return {'sim_time_s':t,'epoch':1,'base_from_dataset':[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],
        'bilateral_contact_objects':['object'] if held else [],
        'contacts':[{'object1':'object','object2':'table_top','normal_force_n':1.}] if table else [],
        'objects':{'object':{'kind':'dynamic','position':[0,0,z+.02],
            'aabb':{'lower':[-.01,-.01,z],'upper':[.01,.01,z+.04]},'speed_m_s':0,'quaternion':[1,0,0,0]}}}


def transfer():
    return {'source':'object','lift_height_m':.05,'controlled_release':True,'incomplete':False,
        'state_before':state(0),'state_held':state(1,held=True),
        'state_lifted':state(2,.05,held=True,table=False),
        'state_before_release':state(3,held=True),'state_after':state(4)}


def samples():return [state(t) for t in (10,11.1,12.2)]


def test_actual_controlled_lift_and_tabletop_return_pass():
    t=transfer();t['lift_height_m']=999 # reported height is not the grading source
    r=contact_episode_completion([t],samples(),chain())
    assert r['complete'] and r['transfers'][0]['lift_height_m']==pytest.approx(.05)


@pytest.mark.parametrize('case',['missing','forged_height','lost','incomplete','held_after','no_release_contact','invalid_position'])
def test_claims_or_broken_carry_do_not_pass(case):
    t=transfer()
    if case=='missing':t={'source':'object','lift_height_m':.05,'controlled_release':True}
    if case=='forged_height':t['state_lifted']=state(2,.004,held=True)
    if case=='lost':t['grip_lost']=True
    if case=='incomplete':t['incomplete']=True
    if case=='held_after':t['state_after']=state(4,held=True)
    if case=='no_release_contact':t['state_before_release']=state(3,held=False)
    if case=='invalid_position':t['state_lifted']['objects']['object']['position'][2]=float('nan')
    assert not contact_episode_completion([t],samples(),chain())['complete']


@pytest.mark.parametrize('case',['floor','side_contact','other_object','one_missing_support','initial_support_unknown','initial_bottom_missing'])
def test_stable_but_not_verified_tabletop_is_rejected(case):
    t=transfer();s=samples()
    if case=='floor':s=[state(x,-.30,table=False) for x in (10,11.1,12.2)]
    if case=='side_contact':s=[state(x,-.30,table=True) for x in (10,11.1,12.2)]
    if case=='other_object':
        for v in s:v['contacts'][0]['object2']='other_object'
    if case=='one_missing_support':s[1]['contacts']=[]
    if case=='initial_support_unknown':t['state_before']['contacts']=[]
    if case=='initial_bottom_missing':del t['state_before']['objects']['object']['aabb']
    assert not contact_episode_completion([t],s,chain())['complete']


def test_contact_completion_still_requires_withdrawal_and_settling():
    assert not contact_episode_completion([transfer()],samples(),chain()[:1])['complete']
    s=samples();s[-1]['objects']['object']['position'][0]=.01
    assert not contact_episode_completion([transfer()],s,chain())['complete']
    assert not contact_episode_completion([],samples(),chain())['complete']
