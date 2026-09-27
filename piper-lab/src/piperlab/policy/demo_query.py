"""Bounded re-reading of a historical clip; never produces robot coordinates."""
import hashlib
import json
from pathlib import Path
from PIL import Image
from ..video.source import VideoSource

SCHEMA={'type':'object','properties':{
    'answer':{'type':'string'},'verdict':{'type':'string','enum':['supported','refuted','unknown']},
    'evidence_refs':{'type':'array','items':{'type':'string'}}},
    'required':['answer','verdict','evidence_refs'],'additionalProperties':False}


def query_demo(demo,query,stage_id,model,output_dir,round_number):
    if not demo or not demo.get('frames'):
        raise ValueError('no_demonstration_to_query')
    if not 0<=round_number<3:
        raise ValueError('demonstration_query_budget_exhausted')
    source=VideoSource(demo['source_path'])
    if source.metadata()['sha256']!=demo['source_hash']:
        raise ValueError('demonstration_source_changed')
    stage=next((s for s in demo.get('stages',[]) if s['id']==stage_id),None)
    if stage_id and stage is None:raise ValueError('unknown_demo_query_stage')
    anchors=[f for f in demo['frames'] if not stage or f['frame_id'] in stage['evidence_refs']]
    if not anchors:raise ValueError('demonstration_query_has_no_anchor')
    center=anchors[len(anchors)//2]['timestamp_s']
    bounds=source.frame_index()
    start=max(bounds['first_timestamp_s'],center-1)
    end=min(bounds['last_timestamp_s'],center+1)
    output=Path(output_dir)/f'demo-query-{round_number}'
    output.mkdir()
    frames=[]
    for i in range(6):
        target=start+(end-start)*i/5
        frame=source.at(target)
        if any(f['pts']==frame.pts for f in frames):continue
        path=output/f'{frame.pts}.jpg';Image.fromarray(frame.image).save(path,quality=90)
        frames.append({'frame_id':f'query{round_number}-pts{frame.pts}','pts':frame.pts,
                       'timestamp_s':frame.timestamp_s,'image_path':str(path)})
    prompt='Read this historical demonstration clip to answer the procedural question. Image order and IDs are listed below. Do not infer robot or live-camera coordinates. If contact or causality is not visible return unknown. Cite only supplied frame IDs.\n'+json.dumps({'query':query,'frames':frames},ensure_ascii=False)
    result=model.infer(prompt,[Path(f['image_path']) for f in frames],SCHEMA)
    if not result['evidence_refs'] or not set(result['evidence_refs'])<={f['frame_id'] for f in frames}:
        raise ValueError('invalid_demo_query_evidence')
    result.update(frames=frames,source_hash=demo['source_hash'],scope='historical_demo_only')
    (output/'answer.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    return result
