"""Behavior tests for prediction syntax and real-result protocol identity."""
import json
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]))
from plan_protocol import Protocol,strict_program_actions

CODE=["p=getpos('cube')","moveto(p)","pick()","q=getpos('box','into')","moveto(q)","place()"]
def interpretation(ref='f0',relation='into'):
    return json.dumps({'actions':[['pick_place','cube','box',relation]],'evidence_refs':[[ref]],'notes':'recorded evidence'})

@pytest.mark.parametrize('code',[["p=getpos('cube'); moveto(p)","pick()"],["p=getpos(\n'cube')"],['import os'],['print(1)'],["p=getpos('cube')","moveto(p)","pick()"],["p=getpos('cube')","moveto(p)","pick()","place()"]])
def test_invalid_prediction(code):
    with pytest.raises((ValueError,SyntaxError)):strict_program_actions(code,['cube','box'])

def test_variable_renaming_preserves_semantics_but_relation_and_order_do_not():
    assert strict_program_actions(CODE)==strict_program_actions([s.replace('p','v') if s=='moveto(p)' else s.replace('p=','v=') for s in CODE])
    assert strict_program_actions(CODE)!=strict_program_actions([s.replace("'into'","'left'") for s in CODE])
    assert strict_program_actions(['open()','close()'])!=strict_program_actions(['close()','open()'])

def flow(mode='R2'):
    p=Protocol(mode,'task',1,['cube','box']);state={'task_id':'task','version':1,'interpretation':''}
    for phase,op in enumerate(['get','update','update','get']):
        p.begin_phase(phase,['f0']);body={'operation':op,'task_id':'task'}
        if op=='update':
            body.update(expected_version=state['version'],changes={'interpretation':interpretation(),'questions':[]})
            new=dict(state,version=state['version']+1,interpretation=interpretation())
        else:new=dict(state)
        p.before_call('task_plan',body);p.after_call('task_plan',body,new);p.finish_phase();state=new
    p.begin_phase(4,['f0']);return p,state

def test_complete_sequence_accepts_consistent_code():
    p,state=flow();assert p.accept_submission({'code':CODE},state)
    assert p.phase_success==[True]*4 and p.adherent

def test_absent_flow_is_optional_only_in_soft_condition():
    for mode,accepted in [('R0',True),('R1',True),('R2',False)]:
        p=Protocol(mode,'task',1,['cube','box'])
        for phase in range(4):p.begin_phase(phase,['f0']);p.finish_phase()
        p.begin_phase(4,['f0']);assert p.accept_submission({'code':CODE})==accepted

def test_raw_program_can_be_correct_while_plan_is_inconsistent():
    p,state=flow();wrong=[s.replace("'into'","'left'") for s in CODE]
    assert not p.accept_submission({'code':wrong},state) and p.consistent is False
    p,state=flow('R1');assert p.accept_submission({'code':wrong},state) and p.consistent is False

def test_same_turn_read_submit_and_update_after_read_are_rejected():
    p,state=flow();p.begin_phase(4,['f0'])
    with pytest.raises(ValueError):p.before_call('task_plan',{'operation':'get'})
    assert not p.accept_submission({'code':CODE},state)
    p,state=flow();p.begin_phase(4,['f0'])
    with pytest.raises(ValueError):p.before_call('task_plan',{'operation':'update','task_id':'task','expected_version':3,'changes':{'interpretation':interpretation()}})

def test_stale_identity_and_future_frame_are_rejected():
    p=Protocol('R2','task',1,['cube','box']);p.begin_phase(1,['f0'])
    for body in [
      {'operation':'update','task_id':'other','expected_version':1,'changes':{'interpretation':interpretation()}},
      {'operation':'update','task_id':'task','expected_version':2,'changes':{'interpretation':interpretation()}},
      {'operation':'update','task_id':'task','expected_version':1,'changes':{'interpretation':interpretation('future')}}]:
        with pytest.raises(ValueError):p.before_call('task_plan',body)
    p=Protocol('R2','task',1,['cube','box']);p.begin_phase(0,['f0'])
    p.after_call('task_plan',{'operation':'get'},{'task_id':'task','version':2});p.finish_phase()
    assert not p.phase_success[0]

def test_backend_fields_and_non_string_interpretation_not_allowed():
    p=Protocol('R1','task',1,['cube','box']);p.begin_phase(1,['f0'])
    with pytest.raises(ValueError):p.before_call('task_plan',{'operation':'update','task_id':'task','expected_version':1,'changes':{'interpretation':json.loads(interpretation())}})
    with pytest.raises(ValueError):p.before_call('task_plan',{'operation':'update','task_id':'task','expected_version':1,'changes':{'interpretation':interpretation(),'questions':[{'id':'q','source':'demo','state':'answered'}]}})
