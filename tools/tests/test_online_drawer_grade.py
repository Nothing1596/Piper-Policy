import copy
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import run_longvil_online as online


def state(q,t=0):
    return {'sim_time_s':t,'base_from_dataset':[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],
            'bilateral_contact_objects':[], 'contacts':[],
            'objects':{'drawer':{'kind':'drawer','qpos_m':q,'travel_m':.17,
                'position':[0,0,0],'aabb':{'lower':[0,0,0],'upper':[1,1,1]},'speed_m_s':0}}}


def action(before,after):
    return {'tool':'move_linear','actual_result':{'status':'succeeded'},
            'state_before':before,'state_after':after}


def test_open_close_use_existing_measured_thresholds():
    start=state(0);opened=state(.15,1);closed=state(0,2)
    chain=[action(start,opened),action(opened,closed)]
    r=online.private_grade([('open',),('close',)],[],final_state=closed,actions=chain)
    assert r['success'] and r['ordered_success']
    assert all(s['success'] for s in r['steps'])
    r=online.private_grade([('close',),('open',)],[],final_state=opened,actions=chain)
    assert r['success'] and not r['ordered_success']


def test_partial_or_unexecuted_drawer_motion_does_not_pass():
    start=state(0);partial=state(.09,1);opened=state(.15,1)
    assert not online.private_grade([('open',)],[],final_state=partial,
                                    actions=[action(start,partial)])['success']
    assert not online.private_grade([('open',)],[],final_state=opened,actions=[])['success']
    assert not online.private_grade([('open',)],[],final_state=opened,
                                    actions=[action(opened,opened)])['success']
    bad=action(start,opened);bad['actual_result']['status']='failed'
    assert not online.private_grade([('open',)],[],final_state=opened,actions=[bad])['success']


def test_existing_grasp_drop_verdict_is_unchanged():
    r=online.private_grade([('pick_place','source','reference','top')],[],final_state=state(0),actions=[])
    assert not r['success']
