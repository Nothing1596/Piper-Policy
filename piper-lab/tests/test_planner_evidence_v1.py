import json
import pytest
from piperlab.policy.planner import VisionPlanner, DECISION_SCHEMA


class Model:
    def __init__(self, refs):
        self.refs = refs
        self.calls = []

    def infer(self, prompt, images, schema):
        self.calls.append((prompt, images, schema))
        return {'action': 'observe', 'evidence_refs': self.refs}


def test_evidence_enum_matches_attached_frames_without_mutating_global_schema():
    frames = [{'frame_id': f'f{i}', 'image_path': f'{i}.jpg', 'timestamp_s': i} for i in range(9)]
    model = Model(['live'])
    VisionPlanner(model).decide(task='test', observation_path='live.jpg', robot_state={},
        demo={'frames': frames, 'stages': [{'id': 's', 'evidence_refs': ['f1']} ]})
    prompt, images, schema = model.calls[0]
    context = json.loads(prompt[prompt.index('{'):])
    allowed = context['allowed_current_evidence_refs']
    assert set(allowed) == {'live', 'f0', 'f3', 'f6', 'f8'}
    assert schema['properties']['evidence_refs']['items']['enum'] == allowed
    assert len(images) == len(allowed) <= 6
    assert 'enum' not in DECISION_SCHEMA['properties']['evidence_refs']['items']
    assert context['historical']['stages'][0]['evidence_refs'] == ['f1']
    other = Model(['live'])
    VisionPlanner(other).decide(task='test', observation_path='live.jpg', robot_state={})
    assert other.calls[0][2]['properties']['evidence_refs']['items']['enum'] == ['live']


def test_unattached_archived_reference_still_fails_after_one_correction():
    model = Model(['live', 'archived-only'])
    with pytest.raises(ValueError, match='invalid_decision_evidence'):
        VisionPlanner(model).decide(task='test', observation_path='live.jpg', robot_state={},
            demo={'frames': [], 'stages': [{'id': 's', 'evidence_refs': ['archived-only']}]})
    assert len(model.calls) == 2
