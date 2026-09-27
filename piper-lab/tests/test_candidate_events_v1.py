import numpy as np
from piperlab.video.candidates import _scene_cut_score,_track_changes,_plan_selection


def test_histogram_scene_change_keeps_adjacent_evidence_without_motion_peak():
    a=np.zeros((32,32),np.uint8);b=np.full_like(a,240)
    score=_scene_cut_score(a,b)
    assert score>0 and _scene_cut_score(a,a)==0
    samples=[{'sequence':i,'timestamp_s':i/10,'motion':0.,'scene_cut_score':score if i==13 else 0.}
             for i in range(30)]
    selected,dropped=_plan_selection(samples,window_s=10,max_per_window=24,baseline_hz=1)
    assert 'scene_cut_before' in selected[12] and 'scene_cut_after' in selected[13]
    assert selected[0] and selected[29]


def summary(box,track=1,epoch=0):
    return {'detector_report':{'tracker_epoch':epoch},'regions':[] if box is None else [
        {'bbox_xyxy':box,'track_id':track,'track_epoch':epoch,'label':'block'}]}


def test_track_changes_are_detector_measurements_not_contact_claims():
    old=summary([0,0,10,10]);new=summary([20,0,30,10])
    events=_track_changes(old,new,100,100)
    assert events[0]['kind']=='track_measurement_change'
    assert _track_changes(new,summary(None),100,100)[0]['meaning']=='observation_gap'
    assert _track_changes(summary(None),new,100,100)[0]['kind']=='track_newly_observed'
    reset=_track_changes(old,summary([70,70,80,80],epoch=1),100,100)
    assert [e['kind'] for e in reset]==['tracker_epoch_reset']
    assert _track_changes(None,new,100,100)==[]
    assert _track_changes(summary(None),summary(None),100,100)==[]
    samples=[{'sequence':i,'timestamp_s':i/10,'motion':0.,'track_events':events if i==13 else []} for i in range(30)]
    chosen,_=_plan_selection(samples,window_s=10,max_per_window=24,baseline_hz=1)
    assert 'track_change' in chosen[13]


def test_dense_scene_events_cannot_overrun_candidate_window():
    samples=[{'sequence':i,'timestamp_s':i/10,'motion':0.,'scene_cut_score':1. if i else 0.} for i in range(100)]
    chosen,dropped=_plan_selection(samples,window_s=10,max_per_window=13,baseline_hz=1)
    assert len(chosen)<=13 and 0 in chosen and 99 in chosen
    rejected=[d for d in dropped if d['kind']=='event' and d['reason']=='window_budget']
    assert rejected and all('scene_cut_candidate' in d['event_kinds'] for d in rejected)
