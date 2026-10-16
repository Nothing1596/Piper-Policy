import hashlib
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pytest
from video_experience import ensure_disjoint, validate_lessons, render_library


def test_no_cross_split_task_or_video():
    a = [{'task_id': 'a', 'source_sha256': 'one'}]
    ensure_disjoint(a, [{'task_id': 'b', 'source_sha256': 'two'}])
    for row in ({'task_id': 'a', 'source_sha256': 'two'},
                {'task_id': 'b', 'source_sha256': 'one'}, {'task_id': 'b'}):
        with pytest.raises(ValueError):
            ensure_disjoint(a, [row])


def test_uncertainty_and_real_references():
    lesson = dict(when='occlusion', guidance='inspect another view',
                  verification='visible relative motion', uncertainty='force unknown',
                  evidence_refs=['v1/f2'])
    validate_lessons({'lessons': [lesson]}, ['v1/f2'])
    with pytest.raises(ValueError):
        validate_lessons({'lessons': [lesson]}, ['v1/f1'])
    validate_lessons({'lessons': []}, [])


def test_frozen_packet_no_claim_of_physical_validation(tmp_path):
    p = tmp_path/'library.json'
    p.write_text(json.dumps({'lessons': []}))
    h = hashlib.sha256(p.read_bytes()).hexdigest()
    assert 'not physically validated' in render_library(p, h)
    p.write_text(json.dumps({'lessons': ['changed']}))
    with pytest.raises(ValueError):
        render_library(p, h)


@pytest.mark.asyncio
async def test_video_pass_real_view_session_and_usage(tmp_path, monkeypatch):
    import video_experience as v
    image = tmp_path/'frame.jpg'
    image.write_bytes(b'fixture-pixels')
    frame = dict(evidence_id='v/f1', image_path=str(image), timestamp_s=1.25,
                 image_sha256=hashlib.sha256(image.read_bytes()).hexdigest())
    async def request(self, prompt):
        self.conversation = 'fresh'
        state = json.loads(self.state_path.read_text())
        path = next(iter(state['images']))
        self.audit_path.write_text(json.dumps(dict(phase='post', conversation_id='fresh',
            allowed=True, tool='view_file', path=path, error=None, model='test'))+'\n')
        return dict(status='SUCCESS', conversation_id='fresh', structured_output={'lessons': []},
                    usage=dict(input_tokens=100, output_tokens=20, thinking_tokens=10,
                               cache_read_tokens=0, total_tokens=120))
    monkeypatch.setattr(v.AgyDecider, 'request', request)
    assert await v.video_pass(tmp_path/'pass', [frame], model='test') == {'lessons': []}
    receipt = json.loads((tmp_path/'pass/receipt.json').read_text())
    assert receipt['usage']['output_tokens'] == 20
    with pytest.raises(FileExistsError):
        await v.video_pass(tmp_path/'pass', [frame], model='test')
    with pytest.raises(ValueError):
        await v.video_pass(tmp_path/'empty', [], model='test')


@pytest.mark.asyncio
async def test_missing_view_does_not_produce_lesson(tmp_path, monkeypatch):
    import video_experience as v
    image = tmp_path/'frame.jpg'; image.write_bytes(b'pixels')
    frame = dict(evidence_id='v/f1', image_path=str(image), timestamp_s=1,
                 image_sha256=hashlib.sha256(image.read_bytes()).hexdigest())
    async def request(self, prompt):
        self.conversation = 'fresh'
        return dict(status='SUCCESS', conversation_id='fresh', structured_output={'lessons': []},
                    usage=dict(input_tokens=100, output_tokens=20, thinking_tokens=10,
                               cache_read_tokens=0, total_tokens=120))
    monkeypatch.setattr(v.AgyDecider, 'request', request)
    with pytest.raises(ValueError, match='image delivery'):
        await v.video_pass(tmp_path/'pass', [frame], model='test')


@pytest.mark.asyncio
async def test_video_interruption_keeps_cost_and_rereads_pixels(tmp_path,monkeypatch):
    import video_experience as v
    image=tmp_path/'frame.jpg';image.write_bytes(b'pixels')
    frame=dict(evidence_id='v/f1',image_path=str(image),timestamp_s=1,
               image_sha256=hashlib.sha256(image.read_bytes()).hexdigest())
    calls=[]
    async def request(self,prompt):
        calls.append(self.turn);self.conversation=f'session-{self.turn}'
        state=json.loads(self.state_path.read_text());path=next(iter(state['images']))
        with self.audit_path.open('a') as out:
            for phase,tool in [('invocation',None),('post','view_file')]:
                out.write(json.dumps(dict(phase=phase,conversation_id=self.conversation,
                    allowed=True,tool=tool,path=path,error=None,model='test'))+'\n')
        return dict(status='ERROR' if self.turn==0 else 'SUCCESS',
            error='connection reset by peer' if self.turn==0 else None,
            conversation_id=self.conversation,structured_output={'lessons':[]},
            usage=dict(input_tokens=100,output_tokens=20,thinking_tokens=10,
                       cache_read_tokens=0,total_tokens=120))
    async def sleep(seconds):pass
    monkeypatch.setattr(v.AgyDecider,'request',request)
    monkeypatch.setattr(v.asyncio,'sleep',sleep)
    assert await v.video_pass(tmp_path/'pass',[frame],model='test')=={'lessons':[]}
    receipt=json.loads((tmp_path/'pass/receipt.json').read_text())
    assert calls==[0,1] and receipt['usage']['output_tokens']==40
    assert receipt['generation_interruptions']==1 and receipt['model_invocations']==2
    assert (tmp_path/'pass/attempt-00-result.json').exists()
