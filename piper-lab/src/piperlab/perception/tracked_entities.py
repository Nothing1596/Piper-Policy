"""Keep ByteTrack IDs local while reusing the perception entity/gap lifecycle."""
from dataclasses import replace
from .tracks import EntityTracker
from .budget import Budget


class _LocalEntities(EntityTracker):
    def _spawn(self,detection,*,source_s):
        entity=super()._spawn(detection,source_s=source_s)
        entity.meta['byte_track_id']=detection.detail.get('track_id')
        return entity

    def _match(self,entity,items,used):
        excluded=set(used)
        for i,detection in enumerate(items):
            if (detection.detail.get('track_id')!=entity.meta.get('byte_track_id') or
                    detection.source_id!=entity.track.source_id or detection.label!=entity.label):
                excluded.add(i)
        return super()._match(entity,items,excluded)


class TrackedEntities:
    def __init__(self):
        self.scope=None;self.tracker=None;self.states={}

    def update(self,detections,*,source_id,epoch,source_s):
        scope=(source_id,epoch)
        reset=self.scope is not None and self.scope!=scope
        if self.scope!=scope:
            self.scope=scope;self.states={}
            self.tracker=_LocalEntities(source_id=source_id,tracker_epoch=epoch,
                budget=Budget(max_entities_per_frame=128,max_entities_total=512))
        touched=self.tracker.update(detections,source_s=source_s)
        events=[]
        for entity in self.tracker.entities:
            if entity.state!=self.states.get(entity.entity_id):
                events.append({'entity_key':f'{source_id}:{epoch}:{entity.entity_id}',
                    'state':entity.state,'source_s':source_s,
                    'meaning':'observation_gap' if entity.gap else 'local_visual_track'})
            self.states[entity.entity_id]=entity.state
        values=[]
        for detection in detections:
            matching=[e for e in touched if e.last_sequence==detection.sequence and
                e.bbox_xyxy==detection.bbox_xyxy and e.meta.get('byte_track_id')==detection.detail.get('track_id')]
            detail=dict(detection.detail)
            if len(matching)==1:
                entity=matching[0]
                detail.update(entity_key=f'{source_id}:{epoch}:{entity.entity_id}',entity_state=entity.state,
                    entity_scope='source_and_epoch_local; no cross-epoch identity claim')
            values.append(replace(detection,detail=detail))
        return values,{'scope_reset':reset,'events':events,'entity_count':len(self.tracker.entities),
            'visible_count':self.tracker.visible_count(),'contact_inference':False}
