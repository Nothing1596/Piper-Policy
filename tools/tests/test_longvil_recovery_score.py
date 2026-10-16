"""Recovery scoring: snapshot-verified transfers, distinct goal matching, order and final state scored separately."""
from pathlib import Path
import sys

ROOT=Path(__file__).parents[2]
for extra in ('tools','piperx-cli/src','piper-lab/src'):sys.path.insert(0,str(ROOT/extra))
from run_longvil_online import private_grade

KINDS={'marker':'static','crate':'static'}
IDENTITY=[[1.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,1.,0.],[0.,0.,0.,1.]]
P0={'block':(.45,-.05,.04),'marker':(.30,.10,.04),'crate':(.50,-.10,.04),'cube':(.10,-.15,.04)}
P1=dict(P0,block=(.18,.10,.04))            # block left of marker
PC=dict(P0,cube=(.62,-.10,.04))            # cube right of crate
P2=dict(PC,block=(.18,.10,.04))            # both goals satisfied
GOAL_BLOCK_LEFT=('pick_place','block','marker','left')
GOAL_CUBE_RIGHT=('pick_place','cube','crate','right')


def snap(positions,bilateral=(),speeds=None):
    speeds=speeds or {}
    return {'objects':{name:{'position':[x,y,z],'aabb':{'lower':[x-.02,y-.02,z-.02],'upper':[x+.02,y+.02,z+.02]},
                              'speed_m_s':speeds.get(name,0.),'kind':KINDS.get(name,'dynamic')}
                       for name,(x,y,z) in positions.items()},
            'contacts':[],'bilateral_contact_objects':list(bilateral),'base_from_dataset':IDENTITY}


def carry(source,before,after,lift=.05,held_contact=True,lifted_contact=True,release_contact=True,
          released=True,controlled=True,grip_lost=False,claim=None):
    """Minimal measured carry; before/after are full scene position maps."""
    lifted={n:(x,y,z+(lift if n==source else 0.)) for n,(x,y,z) in before.items()}
    return {'source':source,'state_before':snap(before),'state_held':snap(before,[source] if held_contact else []),
            'state_lifted':snap(lifted,[source] if lifted_contact else []),
            'state_before_release':snap(after,[source] if release_contact else []),
            'state_after':snap(after,[] if released else [source]),
            'controlled_release':controlled,'grip_lost':grip_lost,
            'lift_height_m':claim if claim is not None else lift}


def test_failed_grasp_then_successful_placement_passes():
    bad=carry('block',P0,P1,lift=.01,claim=.05)
    good=carry('block',P0,P1)
    result=private_grade([GOAL_BLOCK_LEFT],[bad,good],final_state=snap(P1))
    assert result['success'] and result['goal_success'] and result['ordered_success']
    assert [a['valid'] for a in result['attempts']]==[False,True]
    assert result['extra_attempts']==1
    assert result['steps'][0]['success'] and result['steps'][0]['transfer_index']==1
    assert abs(result['steps'][0]['lift_height_m']-.05)<1e-9


def test_reordered_independent_goals_pass_goal_but_fail_order():
    first=carry('cube',P0,PC)
    second=carry('block',PC,P2)
    result=private_grade([GOAL_BLOCK_LEFT,GOAL_CUBE_RIGHT],[first,second],final_state=snap(P2))
    assert result['goal_success'] and not result['ordered_success']
    assert result['success']


def test_extra_corrections_retained_and_counted():
    good=carry('block',P0,P1)
    correction=carry('block',P1,P1,lift=0.,claim=.03)
    transfers=[good,correction]
    result=private_grade([GOAL_BLOCK_LEFT],transfers,final_state=snap(P1))
    assert result['success']
    assert len(transfers)==2 and len(result['attempts'])==2
    assert result['extra_attempts']==1 and result['attempts'][1]['valid'] is False
    assert result['attempts'][1]['reasons']


def test_repeated_action_cannot_reuse_candidate():
    one=[carry('block',P0,P1)]
    result=private_grade([GOAL_BLOCK_LEFT,GOAL_BLOCK_LEFT],one,final_state=snap(P1))
    assert not result['goal_success'] and not result['success']
    assert [s['success'] for s in result['steps']]==[True,False]
    two=[carry('block',P0,P1),carry('block',P0,P1)]
    result=private_grade([GOAL_BLOCK_LEFT,GOAL_BLOCK_LEFT],two,final_state=snap(P1))
    assert result['goal_success'] and result['success']


def test_initial_relation_without_transfer_fails():
    result=private_grade([GOAL_BLOCK_LEFT],[],final_state=snap(P1))
    assert not result['success'] and not result['goal_success'] and not result['ordered_success']
    assert result['steps'][0]['transfer_index'] is None


def test_final_destruction_fails():
    good=carry('block',P0,P1)
    knocked=snap(dict(P1,block=(.60,.30,.04)))
    result=private_grade([GOAL_BLOCK_LEFT],[good],final_state=knocked)
    assert result['goal_success'] and not result['success'] and result['final_state']['ok'] is False
    sliding=private_grade([GOAL_BLOCK_LEFT],[good],final_state=snap(P1,speeds={'block':.5}))
    assert sliding['goal_success'] and not sliding['success']


def test_wrong_contact_or_fabricated_lift_fails():
    cases={'held_contact_missing':dict(held_contact=False),'lifted_contact_missing':dict(lifted_contact=False),
           'before_release_contact_missing':dict(release_contact=False),'not_released_after':dict(released=False),
           'uncontrolled_release':dict(controlled=False),'grip_lost':dict(grip_lost=True),
           'fabricated_lift':dict(lift=.01,claim=.06)}
    for label,kw in cases.items():
        result=private_grade([GOAL_BLOCK_LEFT],[carry('block',P0,P1,**kw)],final_state=snap(P1))
        assert not result['success'] and not result['goal_success'],label
        assert result['attempts'][0]['valid'] is False and result['attempts'][0]['reasons'],label
    missing=carry('block',P0,P1);del missing['state_before_release']
    malformed=carry('block',P0,P1);malformed['state_held']={'broken':True}
    for label,t in (('missing_snapshot',missing),('malformed_snapshot',malformed)):
        result=private_grade([GOAL_BLOCK_LEFT],[t],final_state=snap(P1))
        assert not result['goal_success'] and result['attempts'][0]['valid'] is False,label


def test_unknown_outcome_always_incomplete():
    good=carry('block',P0,P1)
    result=private_grade([GOAL_BLOCK_LEFT],[good],final_state=snap(P1),unknown_outcome=True)
    assert not result['success'] and result['goal_success'] and result['unknown_outcome']


def test_missing_final_state_partial_not_full():
    good=carry('block',P0,P1)
    result=private_grade([GOAL_BLOCK_LEFT],[good])
    assert not result['success'] and result['goal_success'] and result['ordered_success']
    assert result['final_state'] is None


def test_repeated_source_newest_relation_governs_final():
    goals=[('pick_place','block','marker','left'),('pick_place','block','crate','right')]
    Q2=dict(P1,block=(.62,-.10,.04))       # block moved on: right of crate, no longer left of marker
    transfers=[carry('block',P0,P1),carry('block',P1,Q2)]
    result=private_grade(goals,transfers,final_state=snap(Q2))
    assert result['success'] and result['goal_success'] and result['ordered_success']
    assert list(result['final_state']['sources'])==['block']
    assert result['final_state']['sources']['block']['relation']=='right'
    stale=private_grade(goals,transfers,final_state=snap(P1))
    assert stale['goal_success'] and not stale['success']


def test_moving_reference_invalidates_old_final_relation_not_historical_step():
    # block left of cube, then cube moves right of marker. The former
    # relationship is no longer an asserted terminal requirement.
    start=dict(P0,cube=(.30,.10,.04))
    first=dict(start,block=(.18,.10,.04))
    end=dict(first,cube=(.62,.10,.04),block=(.75,.10,.04))
    goals=[('pick_place','block','cube','left'),('pick_place','cube','marker','right')]
    transfers=[carry('block',start,first),carry('cube',first,end)]
    result=private_grade(goals,transfers,final_state=snap(end))
    assert result['goal_success'] and result['ordered_success'] and result['success']
    assert set(result['final_state']['sources'])=={'cube'}
    assert result['steps'][0]['success']
    assert result['final_state']['invalidated_relations'][0]['moved_object']=='cube'
    # Invalidating the relation cannot excuse a missing historical action.
    missing=private_grade(goals,transfers[1:],final_state=snap(end))
    assert not missing['goal_success'] and not missing['success']
    held=private_grade(goals,transfers,final_state=snap(end,['block']))
    assert not held['success'] and not held['final_state']['all_sources_released']


def test_unrelated_move_does_not_invalidate_final_relation():
    transfers=[carry('block',P0,P1),carry('cube',P1,P2)]
    result=private_grade([GOAL_BLOCK_LEFT,GOAL_CUBE_RIGHT],transfers,final_state=snap(P2))
    assert set(result['final_state']['sources'])=={'block','cube'}
    assert result['final_state']['invalidated_relations']==[]
    broken=snap(dict(P2,block=(.80,.10,.04)))
    assert not private_grade([GOAL_BLOCK_LEFT,GOAL_CUBE_RIGHT],transfers,final_state=broken)['success']


def test_reference_move_then_reassert_relation_restores_final_check():
    first=dict(P0,cube=(.30,.10,.04),block=(.18,.10,.04))
    second=dict(first,cube=(.62,.10,.04))
    third=dict(second,block=(.50,.10,.04))
    goals=[('pick_place','block','cube','left'),('pick_place','cube','marker','right'),
           ('pick_place','block','cube','left')]
    transfers=[carry('block',P0,first),carry('cube',first,second),carry('block',second,third)]
    result=private_grade(goals,transfers,final_state=snap(third))
    assert result['success'] and result['ordered_success']
    assert set(result['final_state']['sources'])=={'cube','block'}
