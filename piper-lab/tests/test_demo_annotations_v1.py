import copy
import pytest
from piperlab.demonstration.annotations import evaluate_annotations


def examples():
    demo={'demo_id':'synthetic','source_hash':'a'*64,'outcome_verdict':'supported',
        'frames':[{'frame_id':'f1','timestamp_s':1.},{'frame_id':'f2','timestamp_s':3.}],
        'stages':[{'id':'grasp','operation':'pick','object_roles':['red_block'],'evidence_refs':['f1']},
                  {'id':'release','operation':'place','object_roles':['red_block','tray'],'evidence_refs':['f2']}]}
    annotations={'source_hash':'a'*64,'provenance':'synthetic_fixture','scope':'exhaustive_steps',
        'steps':[{'id':'hand_grasps','operations':['pick','grasp'],'object_roles':['red block'],'time_range_s':[.8,1.2]},
                 {'id':'hand_releases','operations':['place'],'object_roles':['red_block'],'time_range_s':[2.8,3.2]}]}
    return demo,annotations


def test_independent_annotations_count_coverage_and_order():
    demo,annotations=examples()
    demo['stages'][0]['object_roles']=['red cube']
    annotations['role_aliases']={'red_block':['red cube']}
    result=evaluate_annotations(demo,annotations)
    assert result['recall']==1 and result['precision']==1 and result['order_verdict']=='supported'
    demo['stages'].reverse()
    assert evaluate_annotations(demo,annotations)['order_verdict']=='refuted'


def test_one_stage_cannot_satisfy_duplicate_required_steps():
    demo,annotations=examples()
    annotations['steps']=[annotations['steps'][0],copy.deepcopy(annotations['steps'][0])]
    annotations['steps'][1]['id']='another_grasp'
    result=evaluate_annotations(demo,annotations)
    assert result['matched_steps']==1 and result['recall']==.5 and result['order_verdict']=='unknown'


def test_missing_or_unseen_evidence_cannot_count_as_matching():
    demo,annotations=examples()
    demo['stages'][0]['evidence_refs']=['invented']
    result=evaluate_annotations(demo,annotations)
    assert result['recall']==.5 and result['invalid_predictions']
    annotations['source_hash']='b'*64
    with pytest.raises(ValueError,match='source_hash'):evaluate_annotations(demo,annotations)


def test_partial_annotation_scope_does_not_report_false_precision():
    demo,annotations=examples()
    annotations['scope']='required_steps';annotations['steps']=annotations['steps'][:1]
    result=evaluate_annotations(demo,annotations)
    assert result['recall']==1 and result['precision'] is None
    assert result['unmatched_predictions']==['release']
    annotations['provenance']='model_generated'
    with pytest.raises(ValueError,match='provenance'):evaluate_annotations(demo,annotations)
