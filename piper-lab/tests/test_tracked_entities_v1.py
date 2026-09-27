from piperlab.perception.detect import Detection
from piperlab.perception.tracked_entities import TrackedEntities


def detection(sequence,track=1,source='video',label='block'):
    return Detection(source_id=source,sequence=sequence,source_s=float(sequence),scorer='synthetic',
        label=label,score=.9,bbox_xyxy=(10.,10.,30.,30.),centroid_xy=(20.,20.),area_px=400.,
        detail={'track_id':track})


def test_occlusion_is_gap_and_epoch_does_not_alias():
    adapter=TrackedEntities()
    first,_=adapter.update([detection(0)],source_id='video',epoch=0,source_s=0)
    key=first[0].detail['entity_key']
    second,_=adapter.update([detection(1)],source_id='video',epoch=0,source_s=1)
    assert second[0].detail['entity_key']==key and second[0].detail['entity_state']=='tracked'
    _,report=adapter.update([],source_id='video',epoch=0,source_s=2)
    assert report['events'][0]['meaning']=='observation_gap'
    changed,report=adapter.update([detection(3)],source_id='video',epoch=1,source_s=3)
    assert changed[0].detail['entity_key']!=key and report['scope_reset']
    other,_=adapter.update([detection(4,source='other')],source_id='other',epoch=1,source_s=4)
    assert other[0].detail['entity_key']!=changed[0].detail['entity_key']


def test_different_track_or_label_does_not_inherit_entity():
    adapter=TrackedEntities()
    first,_=adapter.update([detection(0)],source_id='video',epoch=0,source_s=0)
    second,_=adapter.update([detection(1,track=2)],source_id='video',epoch=0,source_s=1)
    assert first[0].detail['entity_key']!=second[0].detail['entity_key']
    third,_=adapter.update([detection(2,track=2,label='cup')],source_id='video',epoch=0,source_s=2)
    assert second[0].detail['entity_key']!=third[0].detail['entity_key']
