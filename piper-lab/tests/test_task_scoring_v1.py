import math
from piperlab.policy.task_scoring import placed_block


TRAY={'bounds_xy':[.28,.42,.13,.27],'z_surface':.006}


def block(x=.35,y=.20,z=.021,quaternion=(1,0,0,0)):
    return {'position':[x,y,z],'quaternion':list(quaternion),'in_tray':True,'grasped':False}


def test_center_in_tray_does_not_prove_full_block_inside():
    assert placed_block(block(),TRAY)['success']
    result=placed_block(block(x=.414),TRAY)
    assert not result['success'] and not result['entire_footprint_inside']


def test_orientation_and_height_are_part_of_placement_score():
    yaw45=(math.cos(math.pi/8),0,0,math.sin(math.pi/8))
    assert placed_block(block(x=.40),TRAY)['success']
    assert not placed_block(block(x=.40,quaternion=yaw45),TRAY)['success']
    assert not placed_block(block(z=.08),TRAY)['success']
    assert not placed_block({**block(),'grasped':True},TRAY)['success']
    assert not placed_block(block(quaternion=(0,0,0,0)),TRAY)['success']
