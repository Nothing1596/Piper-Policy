"""Actual-state grading invariants; no model API or simulated robot required."""
import copy
from pathlib import Path
import sys
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).parents[1]))
from longvil_physics_score import relation_status,score_episode


def obj(position,kind='dynamic',half=.02):
    p=np.asarray(position)
    return {'position':list(position),'aabb':{'lower':(p-half).tolist(),'upper':(p+half).tolist()},
            'kind':kind,'speed_m_s':0.}


def state():
    return {'objects':{'source':obj([-.15,0,.02]),'reference':obj([0,0,.02])},
            'contacts':[],'base_from_dataset':np.eye(4).tolist()}


def step(action,s,**grip):
    before=copy.deepcopy(s);held=copy.deepcopy(s);lifted=copy.deepcopy(s);after=copy.deepcopy(s)
    held['bilateral_contact_objects']=['source'] if grip.get('bilateral_contact',True) else []
    lifted['bilateral_contact_objects']=held['bilateral_contact_objects']
    lifted['objects']['source']['position'][2]+=grip.get('lift_height_m',.08)
    after['bilateral_contact_objects']=[] if grip.get('released',True) else ['source']
    return {'action':action,'status':'executed','state_after':s,
            'state_before':before,'state_held':held,'state_lifted':lifted,
            'state_after':after,
            'grasp_evidence':{'bilateral_contact':True,'lift_height_m':.08,'released':True,**grip}}


def test_lateral_uses_dataset_frame_and_dominant_direction():
    s=state();assert relation_status(s,'source','reference','left')
    transform=np.array([[0,-1,0,.4],[1,0,0,0],[0,0,1,0],[0,0,0,1.]])
    for o in s['objects'].values():
        o['position']=(transform[:3,:3]@o['position']+transform[:3,3]).tolist()
        p=np.asarray(o['position']);o['aabb']={'lower':(p-.02).tolist(),'upper':(p+.02).tolist()}
    s['base_from_dataset']=transform.tolist()
    assert relation_status(s,'source','reference','left')
    s=state();s['objects']['source']=obj([-.015,.3,.02])
    assert not relation_status(s,'source','reference','left')


def test_accepted_target_without_actual_lift_fails():
    s=state();action=['pick_place','source','reference','left']
    r=score_episode([action],[step(action,s,lift_height_m=0.)],s,s)
    assert r['sequence_success'] and not r['success']
    row=step(action,s,lift_height_m=0.,released=False)
    row['grasp_evidence'].update(lift_height_m=.1,released=True)
    assert not score_episode([action],[row],s,s)['success']


def test_top_requires_physical_support_and_stability():
    s=state();s['objects']['source']=obj([0,0,.06])
    assert not relation_status(s,'source','reference','top')
    s['contacts']=[{'object1':'source','object2':'reference','normal_force_n':.5}]
    assert relation_status(s,'source','reference','top')
    s['objects']['source']['speed_m_s']=.1
    assert not relation_status(s,'source','reference','top')


def test_into_requires_hollow_interior_release_and_floor_contact():
    s=state();s['objects']['reference']=obj([0,0,.06],'container',half=.10)
    s['objects']['reference']['container_bounds']={'lower':[-.07,-.07,0.],'upper':[.07,.07,.12]}
    s['objects']['source']=obj([0,0,.02])
    s['contacts']=[{'object1':'source','object2':'reference','normal_force_n':.5}]
    assert relation_status(s,'source','reference','into')
    a=['pick_place','source','reference','into']
    assert not score_episode([a],[step(a,s,released=False)],s,s)['success']
    s['objects']['source']=obj([0,0,.16])
    assert not relation_status(s,'source','reference','into')
    del s['objects']['reference']['container_bounds']
    with pytest.raises(ValueError,match='interior'):relation_status(s,'source','reference','into')


def test_contained_contents_can_support_each_other_but_not_through_outside_objects():
    s=state();s['objects']['reference']=obj([0,0,.06],'container',half=.10)
    s['objects']['reference']['container_bounds']={'lower':[-.07,-.07,0.],'upper':[.07,.07,.12]}
    s['objects']['source']=obj([0,0,.06]);s['objects']['support']=obj([0,0,.02])
    s['contacts']=[{'object1':'source','object2':'support','normal_force_n':.5},
                   {'object1':'support','object2':'reference','normal_force_n':.5}]
    assert relation_status(s,'source','reference','into')
    s['objects']['support']=obj([.2,0,.02])
    assert not relation_status(s,'source','reference','into')
    s['objects']['support']=obj([0,0,.02]);s['contacts'].pop()
    assert not relation_status(s,'source','reference','into')
    s['contacts'].append({'object1':'support','object2':'reference','normal_force_n':.5})
    s['objects']['source']=obj([0,0,.13])
    assert not relation_status(s,'source','reference','into')


def test_order_omissions_and_drawer_measurement():
    s=state();s['objects']['drawer']=obj([.3,0,0],'drawer');s['objects']['drawer'].update(qpos_m=.17,travel_m=.17)
    a=['pick_place','source','reference','left']
    opened={'action':['open'],'status':'executed','state_after':s}
    assert score_episode([['open'],a],[opened,step(a,s)],s,s)['success']
    assert not score_episode([['open'],a],[step(a,s),opened],s,s)['success']
    assert not score_episode([['open'],a],[opened],s,s)['success']
    s['objects']['drawer']['qpos_m']=0.
    assert not score_episode([['open']],[opened],s,s)['success']


def test_historical_placement_is_not_erased_by_later_reference_motion():
    before=state();placed=copy.deepcopy(before);after=copy.deepcopy(before)
    after['objects']['reference']=obj([-.3,0,.02])
    a=['pick_place','source','reference','left']
    r=score_episode([a],[step(a,placed)],before,after)
    assert r['success'] and r['final_state']['relations']['left source reference'] is False


def test_nonfinite_measurement_or_nonrigid_frame_rejected():
    s=state();s['objects']['source']['speed_m_s']=float('nan')
    with pytest.raises(ValueError):relation_status(s,'source','reference','left')
    s=state();s['base_from_dataset'][0][0]=2.
    with pytest.raises(ValueError,match='rigid'):relation_status(s,'source','reference','left')


def test_circular_interior_uses_actual_footprint_not_inscribed_rectangle():
    s=state();s['objects']['reference']=obj([0,0,.06],'container',half=.10)
    ref=s['objects']['reference'];ref['container_bounds']={'lower':[-.04,-.04,0.],'upper':[.04,.04,.12]}
    ref['inner_circle']={'center_xy_m':[0,0],'radius_m':.07}
    src=s['objects']['source']=obj([0,0,.02]);src['aabb']['lower'][:2]=[-.065,-.065];src['aabb']['upper'][:2]=[.065,.065]
    src['grasp_frame']={'footprint_xy_m':[[-.065,0],[0,-.065],[.065,0],[0,.065]]}
    s['contacts']=[{'object1':'source','object2':'reference','normal_force_n':.5}]
    assert relation_status(s,'source','reference','into')
    src['grasp_frame']['footprint_xy_m'][0]=[-.09,0]
    assert not relation_status(s,'source','reference','into')
    del src['grasp_frame']
    with pytest.raises(ValueError,match='footprint'):relation_status(s,'source','reference','into')
