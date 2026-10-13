"""Experiment integrity checks; no model-performance claims."""
import importlib.util
import json
from pathlib import Path

spec = importlib.util.spec_from_file_location('longvil_ablation', Path(__file__).parents[1] / 'run_longvil_ablation.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_reference_answers_and_coordinates_never_enter_public_task():
    annotation = {'object_list': ['cube', 'box'], 'code': ['secret()'],
                  'action_sequences': ['secret'], 'positions': {'cube': [1, 2, 3]},
                  'task_horizon': 12, 'pre_action_sequences': 'private', 'future_field': 'private'}
    assert module.public_task(annotation) == {'object_list': ['cube', 'box']}


def test_incomplete_and_infrastructure_episodes_are_not_successes(tmp_path):
    rows = [dict(task='one', variant='clean', condition='plan', session_id='a',
                 status='infrastructure_error', runtime_remaining=False)]
    module.summarize(tmp_path, rows, 12)
    result = json.loads((tmp_path / 'summary.json').read_text())
    assert not result['complete']
    assert result['groups'][1]['scored'] == 0
    assert result['groups'][1]['FSA_successes'] == 0
    assert result['groups'][1]['mean_SMS'] is None
    module.summarize(tmp_path, rows, 1)
    result = json.loads((tmp_path / 'summary.json').read_text())
    assert result['schedule_ended'] and not result['complete']
