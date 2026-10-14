import json
import pytest
from pathlib import Path
from piperlab.demonstration import compile_demo,DemoStore
from piperlab.policy.context import active_demo,segment_count
from test_demonstration_v1 import FakeModel,make_fake_builder


def keep_all(ids,index):
    return {'selections':[{'frame_id':i,'keep':True,'reason':'synthetic required step'} for i in ids]}


def test_long_demo_retains_all_segments_and_archives_once(tmp_path):
    video=tmp_path/'input.mp4';video.write_bytes(b'synthetic fixture')
    model=FakeModel(select_fn=keep_all)
    bundle=tmp_path/'bundle'
    demo=compile_demo(str(video),str(bundle),'sequence',model,candidate_builder=make_fake_builder(70),cache_dir=str(tmp_path/'cache'))
    assert segment_count(demo)==4
    assert len(demo['frames'])==70
    assert all(len(s['frames'])<=24 for s in demo['segments'])
    assert len({s['id'] for s in demo['stages']})==len(demo['stages'])
    assert len(list(bundle.rglob('source.mp4')))==1
    assert all(Path(f['image_path']).is_file() for f in demo['frames'])
    assert all(Path(s['source_path']).is_file() for s in demo['segments'])
    assert all(len(c['images'])<=6 for c in model.calls)
    current=active_demo(demo,2)
    assert 'segments' not in current and current['segment_context']['index']==2
    assert current['frames'][0]['timestamp_s']==20
    with DemoStore(tmp_path/'store') as store:
        store.register(bundle);stored=store.get(demo['demo_id'])
        assert store.register(bundle)==demo['demo_id']
        for segment in stored['segments']:
            assert Path(segment['source_path']).is_relative_to(store.root)
            assert all(Path(f['image_path']).is_file() and Path(f['image_path']).is_relative_to(store.root) for f in segment['frames'])
    cached=FakeModel(select_fn=keep_all)
    second=compile_demo(str(video),str(tmp_path/'second'),'sequence',cached,candidate_builder=make_fake_builder(70),cache_dir=str(tmp_path/'cache'))
    assert not cached.calls
    assert second['demo_id']==demo['demo_id']
    assert all(s['cache']['status']=='hit' for s in second['segments'])
    import shutil
    from piperlab.video_cli import load_bundle
    moved=tmp_path/'relocated';shutil.copytree(bundle,moved)
    relocated=load_bundle(moved)
    assert all(Path(f['image_path']).is_relative_to(moved) for s in relocated['segments'] for f in s['frames'])
    assert Path(relocated['source_path']).is_relative_to(moved)


def test_failed_segment_prevents_claiming_complete_demo(tmp_path):
    video=tmp_path/'input.mp4';video.write_bytes(b'synthetic fixture')
    def semantic(ids,index):
        return {} if index==1 else FakeModel._default_semantic(ids,index)
    demo=compile_demo(str(video),str(tmp_path/'bundle'),'sequence',FakeModel(semantic_fn=semantic),
        candidate_builder=make_fake_builder(65),requery_budget=0)
    assert len(demo['segments'])==4
    assert demo['outcome_verdict']=='unresolved'
    assert demo['segments'][1]['outcome_verdict']=='unresolved'
    assert demo['segments'][-1]['frames'][-1]['frame_id']=='f000064'


@pytest.mark.parametrize('completion_verdict',['supported','refuted'])
def test_segment_done_advances_without_claiming_whole_task(monkeypatch,tmp_path,completion_verdict):
    import time
    import numpy as np
    from types import SimpleNamespace
    import piperlab.policy.runner as module
    class Camera:
        def __init__(self,*args):self.thread=SimpleNamespace(ident=None)
        def start(self):pass
        def latest(self):
            return {'rgb':np.zeros((32,32,3),dtype=np.uint8),'state_version':0,'epoch':0,
                    'source_stamp_s':time.monotonic(),'candidates':[]}
    class Executor:
        def request(self,*args):return {'backend':'mujoco'}
        def state(self):return {'active_job_id':None}
        def stop(self):pass
    class Model:
        identity={'name':'test-double'}
        prompts=[]
        def infer(self,prompt,images,schema):
            self.prompts.append(prompt)
            return dict(action='done',stage_id='',object_point=[0,0],destination_point=[0,0],
                object_candidate_id='',destination_candidate_id='',object_label='',expected_effect='',
                reason='Synthetic segment completion proposal',evidence_refs=['live'],query='')
    monkeypatch.setattr(module,'ObservationStream',Camera)
    monkeypatch.setattr(module,'recheck_latest_effects',lambda observation,history: {'verdict':completion_verdict,'checks':[]})
    segments=[{'demo_id':str(i),'stages':[],'frames':[],'summary':f'ACTIVE_SEGMENT_{i}'} for i in range(2)]
    demo={'demo_id':'parent','segments':segments,'stages':[],'outcome_verdict':'supported'}
    model=Model()
    report=module.run_policy(Executor(),model,task='multi-stage synthetic task',output_dir=tmp_path/'run',demo=demo,max_decisions=3)
    if completion_verdict=='refuted':
        assert report['decisions']==1
        assert report['status']=='completion_recheck_failed'
        assert report['history'][0]['outcome']=='completion_not_supported_by_current_observation'
        assert report['segment_progress']['active_index']==0
        return
    assert report['decisions']==2
    assert report['history'][0]['outcome']=='segment_done_pending_task_evaluation'
    assert report['status']=='model_declared_done_pending_task_evaluation'
    assert report['segment_progress']=={'active_index':1,'count':2}
    assert 'ACTIVE_SEGMENT_0' in model.prompts[0] and 'ACTIVE_SEGMENT_1' not in model.prompts[0]
    assert 'ACTIVE_SEGMENT_1' in model.prompts[1] and 'ACTIVE_SEGMENT_0' not in model.prompts[1]
