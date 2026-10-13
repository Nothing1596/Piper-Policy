"""Checks experimental isolation and scoring, not model capability."""
import importlib.util
from pathlib import Path
import copy

spec=importlib.util.spec_from_file_location('ablation',Path(__file__).parents[1]/'run_plan_ablation.py')
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)


def test_factor_tool_filters_do_not_mutate_original():
    tools=[{'name':n,'description':'x','inputSchema':{'type':'object','properties':{'question_id':{'type':'string'}}}} for n in ['task_plan','scene_query','camera_observe','move_to','robot_connect','demo_watch']]
    original=copy.deepcopy(tools)
    for condition in a.CONDITIONS:
        selected=a.conditions_tools(tools,condition);names={t['name'] for t in selected}
        assert ('task_plan' in names)==condition.startswith('plan_')
        assert ('scene_query' in names)==(not condition.endswith('no_geometry'))
        assert {'camera_observe','move_to'}<=names
        assert 'demo_watch' not in names and 'robot_connect' not in names
    assert tools==original


def truth(tcp,red=(.3,-.1,.015),blue=(.35,.1,.015)):
    return {'truth':{'robot':{'tcp_pos':tcp},'objects':{'cube_red':{'position':red},'cube_blue':{'position':blue}}}}


def test_sequence_requires_order_and_realized_position():
    initial=truth([0,0,.4]);red=truth([.3,-.1,.15]);blue=truth([.35,.1,.15])
    assert a.score('blue_then_red',initial,[blue,red],red)['success']
    assert not a.score('blue_then_red',initial,[red,blue],red)['success']
    assert not a.score('hover_red',initial,[],red)['success']
    assert not a.score('hover_red',initial,[red],truth([.33,-.1,.15]))['success']
    assert not a.score('hover_red',initial,[red],truth([.3,-.1,.15],red=(.31,-.1,.015)))['success']


def test_analysis_keeps_infrastructure_errors_out_of_success_denominator(tmp_path):
    import json
    spec=importlib.util.spec_from_file_location('analysis',Path(__file__).parents[1]/'analyze_plan_ablation.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    item={'task':'hover_red','seed':401,'condition':'plan_geometry'}
    (tmp_path/'protocol.json').write_text(json.dumps({'schedule':[item,dict(item,condition='no_plan_geometry')]}))
    row=dict(item,session_id='unique',label='one',status='infrastructure_error',usage={},cleanup={'runtime_remaining':False})
    (tmp_path/'episodes.jsonl').write_text(json.dumps(row)+'\n')
    result=module.analyze(tmp_path)
    assert result['complete'] is False
    assert result['groups'][0]['completed']==1 and result['groups'][0]['scored']==0
    assert result['groups'][0]['mean_elapsed_s'] is None
    (tmp_path/'episodes.jsonl').write_text((json.dumps(row)+'\n')*2)
    import pytest
    with pytest.raises(ValueError,match='Duplicate episode'):module.analyze(tmp_path)


def test_posthoc_threshold_does_not_change_primary_score():
    spec=importlib.util.spec_from_file_location('analysis',Path(__file__).parents[1]/'analyze_plan_ablation.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    initial=truth([0,0,.4]);near=truth([.3,-.1,.160015])
    assert not a.score('hover_red',initial,[near],near)['success']
    assert not module.sensitivity_success('hover_red',initial,[near],near,.010)
    assert module.sensitivity_success('hover_red',initial,[near],near,.011)
    assert not module.sensitivity_success('blue_then_red',initial,[near],near,.011)
