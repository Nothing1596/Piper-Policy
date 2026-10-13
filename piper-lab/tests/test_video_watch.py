import json
import pytest
from piperlab.video.watch import watch
from test_video_v1 import _write_synthetic_video,needs_av,needs_cv2


@needs_av
@needs_cv2
def test_direct_point_and_range_preserve_real_pts_and_no_evaluation(tmp_path):
    source=_write_synthetic_video(tmp_path/'source.mp4')
    one=watch(str(source),tmp_path/'one',at_s=.55)
    assert len(one['frames'])==1 and .55<=one['frames'][0]['timestamp_s']<.7 and one['evaluation'] is None
    interval=watch(str(source),tmp_path/'range',start_s=.5,end_s=1.5,max_frames=3)
    assert len(interval['frames'])==3
    assert all(.5<=f['timestamp_s']<=1.5 for f in interval['frames'])


@needs_av
@needs_cv2
def test_pipeline_returns_middleware_evaluation_and_bounded_vision(tmp_path):
    source=_write_synthetic_video(tmp_path/'source.mp4')
    class Model:
        identity={'model':'fixture'}
        def infer(self,prompt,images,schema):
            assert len(images)<=3 and 'Detector labels may be wrong' in prompt
            return {'summary':'fixture','stages':[],'goal_constraints':[],
                    'unknowns':['synthetic'],'outcome_verdict':'unknown'}
    result=watch(str(source),tmp_path/'pipeline',mode='pipeline',start_s=.5,end_s=1.5,max_frames=6,model=Model())
    assert result['evaluation']['status']=='evaluated'
    assert result['evaluation']['vision']['summary']=='fixture'
    assert result['evaluation']['task_success']=='unverified'
    assert len(result['evaluation']['vision']['frames_supplied'])<=3
    assert result['evaluation']['quality'] and result['evaluation']['full_video_context']['scope']=='full_source_context_not_interval_verdict'


@needs_av
@needs_cv2
def test_watch_rejects_bad_times_and_fabricated_vision_refs(tmp_path):
    source=_write_synthetic_video(tmp_path/'source.mp4')
    for args in ({'at_s':1.,'start_s':0.,'end_s':2.},{'start_s':2.,'end_s':1.},{'at_s':99.}):
        with pytest.raises(ValueError):watch(str(source),tmp_path/'bad',**args)
    class Bad:
        identity={}
        def infer(self,*args):return {'summary':'invented','stages':[{'id':'s','operation':'x',
            'object_roles':[],'preconditions':[],'expected_effects':[],'evidence_refs':['invented-frame'],'uncertainty':'x'}],
            'goal_constraints':[],'unknowns':[],'outcome_verdict':'supported'}
    with pytest.raises(Exception,match='validation'):watch(str(source),tmp_path/'bad-refs',mode='pipeline',model=Bad())


@needs_av
@needs_cv2
def test_full_video_cv_cache_reused_across_intervals(tmp_path,monkeypatch):
    source=_write_synthetic_video(tmp_path/'source.mp4')
    cache=tmp_path/'cache'
    a=watch(str(source),tmp_path/'a',mode='pipeline',start_s=0.,end_s=1.,analysis_cache=cache)
    assert not a['evaluation']['full_video_context']['cache_hit']
    def unexpected(*args,**kwargs):raise AssertionError('CV analysis repeated')
    monkeypatch.setattr('piperlab.video.candidates.build_candidates',unexpected)
    b=watch(str(source),tmp_path/'b',mode='pipeline',start_s=1.,end_s=2.,analysis_cache=cache)
    assert b['evaluation']['full_video_context']['cache_hit']
    assert b['evaluation']['full_video_context']['manifest_path']==a['evaluation']['full_video_context']['manifest_path']
    assert b['frames'][0]['timestamp_s']>=1.
