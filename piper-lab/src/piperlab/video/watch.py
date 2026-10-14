"""Repeatable timestamped viewing and bounded evaluation through the existing compiler."""
import hashlib
import json
import math
import time
from pathlib import Path
from urllib.parse import unquote, urlparse

import numpy as np
from PIL import Image

from .source import VideoSource, VideoError


def vision_evaluation_prompt(frames, object_list=()):
    """A bounded historical visual hypothesis, never an executable plan or GT."""
    return ('Give a SHORT visual state hint from these HISTORICAL images. '
            'Inspect the FIRST and LAST supplied image for visible object configurations and spatial uncertainty. '
            'Do not reconstruct motion, action order, hidden transitions or a complete video plan: a later model does that. '
            'Do not generate robot code. Only two observation stages at most: initial visible state and final visible state. '
            'Detector labels may be wrong. Never infer current coordinates or task success. '
            'For one image, temporal transitions are unknown. Use only supplied frame IDs. '
            'Return ONE compact JSON object. stages is an array of objects. Each has id and operation (strings); '
            'object_roles, preconditions, expected_effects, evidence_refs (arrays of strings, NEVER dictionaries or single strings); '
            'uncertainty (string). goal_constraints and unknowns are arrays of strings. '
            'Set outcome_verdict="unknown"; summary is a string <=40 words. '
            'Use empty preconditions/expected_effects when a still frame cannot establish them. At most 3 unknowns. '
            'No exhaustive per-object inventory or long narrative is needed.\n'
            +json.dumps({'frames':[{'frame_id':f['frame_id'],'timestamp_s':f['timestamp_s'],
                                   'detections':f.get('detections',[])} for f in frames],
                        'object_list':list(object_list),'scope':'selected_historical_frames_only'},ensure_ascii=False))


def canonicalize_visual_references(value, frames, source_sha256=None):
    """Separate a known source citation and sort frames; never accept unknown IDs."""
    import copy
    result=copy.deepcopy(value);changes=[]
    times={f['frame_id']:f['timestamp_s'] for f in frames}
    for index,stage in enumerate(result.get('stages',[]) if isinstance(result,dict) else []):
        if not isinstance(stage,dict):continue
        refs=stage.get('evidence_refs')
        if not isinstance(refs,list) or not all(isinstance(r,str) for r in refs):continue
        frame_refs=[r for r in refs if r!=source_sha256]
        if not frame_refs or not all(r in times for r in frame_refs):continue
        ordered=sorted(frame_refs,key=lambda r:(times[r],r))
        if refs!=ordered:
            changes.append({'stage_index':index,'before':refs,'after':ordered,
                            'source_refs':[source_sha256] if source_sha256 in refs else []})
            stage['evidence_refs']=ordered
    return result,changes


def local_source(address, cache):
    if not isinstance(address,str) or not address.strip():
        raise ValueError('address is required')
    parsed=urlparse(address)
    if parsed.scheme in ('http','https'):
        if parsed.username or parsed.password or parsed.fragment:
            raise ValueError('Video URL must not contain credentials or fragment')
        import httpx
        cache=Path(cache);cache.mkdir(parents=True,exist_ok=True)
        key=hashlib.sha256(address.encode()).hexdigest()
        saved=cache/(key+'.video')
        if not saved.exists():
            temp=saved.with_suffix('.download')
            try:
                with httpx.stream('GET',address,follow_redirects=True,timeout=30,trust_env=False) as response:
                    response.raise_for_status()
                    if 'text/html' in response.headers.get('content-type',''):
                        raise ValueError('Provide a downloadable video URL, not a webpage')
                    size=0
                    with temp.open('wb') as stream:
                        for chunk in response.iter_bytes():
                            size+=len(chunk)
                            if size>512*1024*1024:raise ValueError('Video download exceeds 512 MiB')
                            stream.write(chunk)
                temp.replace(saved)
            finally:temp.unlink(missing_ok=True)
        return saved
    if parsed.scheme=='file':
        address=unquote(parsed.path)
    elif parsed.scheme and not (len(parsed.scheme)==1 and len(address)>2 and address[1]==':'):
        raise ValueError('Supported addresses: local path, file URL, downloadable HTTP(S) video')
    path=Path(address).expanduser().resolve(strict=True)
    if not path.is_file():raise ValueError('Video address must be a file')
    return path


def watch(address, output, *, mode='direct', at_s=None, start_s=None, end_s=None, max_frames=6, detector=None, model=None, analysis_cache=None):
    if mode not in ('direct','pipeline'):raise ValueError('mode must be direct or pipeline')
    if type(max_frames) is not int or not 1<=max_frames<=12:raise ValueError('max_frames must be 1..12')
    for value in (at_s,start_s,end_s):
        if value is not None and (type(value) not in (int,float) or not math.isfinite(value) or value<0):
            raise ValueError('Times must be finite nonnegative seconds')
    if at_s is not None and (start_s is not None or end_s is not None):raise ValueError('Choose at_s or start_s/end_s')
    if (start_s is None)!=(end_s is None) or (start_s is not None and start_s>=end_s):raise ValueError('A range requires start_s < end_s')
    output=Path(output).absolute()
    if output.exists():raise FileExistsError('Watch output exists; choose a fresh destination')
    output.parent.mkdir(parents=True,exist_ok=True)
    begun=time.monotonic();timings={}
    progress_file=output.with_suffix('.progress.json')
    def progress(phase):
        progress_file.write_text(json.dumps({'phase':phase,'elapsed_s':time.monotonic()-begun}),encoding='utf-8')
    progress('source_index')
    source=VideoSource(local_source(address,output.parent/'watch-downloads'))
    rows=source.frame_index()['frames']
    timings['source_index_s']=time.monotonic()-begun
    lo,hi=rows[0]['timestamp_s'],rows[-1]['timestamp_s']
    if at_s is not None:
        times=[float(at_s)]
        if not lo<=at_s<=hi:raise ValueError('Requested point outside decoded video time')
    else:
        start,end=(lo,hi) if start_s is None else (start_s,end_s)
        if not lo<=start<=end<=hi:raise ValueError('Requested range outside decoded video time')
        available=[r['timestamp_s'] for r in rows if start<=r['timestamp_s']<=end]
        if not available:raise ValueError('No decoded frame in requested range')
        if len(available)>1 and max_frames<2:raise ValueError('A time range requires max_frames >=2 to retain endpoints')
        times=list(np.linspace(available[0],available[-1],min(max_frames,len(available))))
    # at() chooses the first real PTS >= request. Report that offset, never
    # relabel an approximate seek result as an exact requested timestamp.
    decoded={}
    for requested in times:
        frame=source.at(float(requested))
        decoded.setdefault(frame.sequence,(frame,float(requested)))
    selected=sorted(decoded.values(),key=lambda pair:pair[0].timestamp_s)
    timings['frame_seek_s']=time.monotonic()-begun-timings['source_index_s']
    if mode=='direct':
        output.mkdir(parents=True)
        frames=[]
        for frame,requested in selected:
            path=output/f'f{frame.sequence:06d}.jpg';Image.fromarray(frame.image).save(path,quality=90)
            frames.append({'frame_id':f'f{frame.sequence:06d}','timestamp_s':frame.timestamp_s,
                'requested_s':requested,'seek_offset_s':frame.timestamp_s-requested,'pts':frame.pts,
                'time_base':frame.time_base,'image_path':str(path)})
        evaluation=None
    else:
        from .candidates import build_candidates
        progress('cv_analysis');phase=time.monotonic()
        analysis_path = output/'analysis'
        cache_hit = False
        # Cache only the deterministic full-video CV pass, never a VLM verdict.
        # Unknown/custom detectors have no stable identity and bypass caching.
        if analysis_cache and (detector is None or getattr(detector, 'sha256', None)):
            identity = {'version': 1, 'source': source.sha256, 'algorithm': hashlib.sha256(
                Path(__file__).with_name('candidates.py').read_bytes()).hexdigest(),
                'detector': None if detector is None else {k: getattr(detector,k,None)
                    for k in ('name','sha256','threshold','nms_iou','max_candidates')}}
            key = hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
            analysis_path = Path(analysis_cache)/key
        manifest_file = analysis_path/'manifest.json'
        if manifest_file.is_file():
            try:
                manifest = json.loads(manifest_file.read_text(encoding='utf-8'))
                cache_hit = manifest.get('kind') == 'piperlab.video.candidates'
            except (ValueError, OSError):
                cache_hit = False
        if not cache_hit:
            if analysis_path.exists():
                # Leave corrupt/partial caches intact for diagnosis; build a fresh result.
                analysis_path = output/'analysis'
            try:
                manifest=build_candidates(str(source.path),str(analysis_path),detector=detector)
            except FileExistsError:
                # Another reader may have completed the same immutable cache entry.
                manifest=json.loads((analysis_path/'manifest.json').read_text(encoding='utf-8'))
                if manifest.get('kind') != 'piperlab.video.candidates':
                    raise ValueError('Invalid concurrent analysis cache')
                cache_hit=True
        timings['cv_analysis_s']=time.monotonic()-phase
        progress('selected_frame_detection');phase=time.monotonic()
        output.mkdir(parents=True,exist_ok=True)
        frames=[]
        # Evaluate exactly the requested decoded frames. Detector sampling in
        # full-source analysis is separate and its context scope is explicit.
        import cv2
        previous=None
        for frame,requested in selected:
            path=output/f'f{frame.sequence:06d}.jpg';Image.fromarray(frame.image).save(path,quality=90)
            gray=cv2.resize(cv2.cvtColor(frame.image,cv2.COLOR_RGB2GRAY),(160,90))
            details={'clarity_laplacian_variance':float(cv2.Laplacian(gray,cv2.CV_64F).var()),
                'sampled_visual_change':None if previous is None else float(np.abs(gray.astype(float)-previous.astype(float)).mean()/255),
                'camera_compensation':'not applied to these interval samples'}
            detections=[]
            if detector is not None:
                from dataclasses import asdict
                detections=[asdict(d) for d in detector.score(frame.image,source_id=source.sha256,sequence=frame.sequence,source_s=frame.timestamp_s)]
            frames.append({'frame_id':f'f{frame.sequence:06d}','timestamp_s':frame.timestamp_s,'requested_s':requested,
                'seek_offset_s':frame.timestamp_s-requested,'pts':frame.pts,'time_base':frame.time_base,
                'image_path':str(path),'quality':details,'detections':detections})
            previous=gray
        evaluation={'scope':'selected_historical_frames','status':'evaluated',
            'middleware':['project video source / PTS validation','OpenCV clarity / visual change',
                          getattr(detector,'name','object detector unavailable')],
            'frame_count':len(frames),'quality':[{'frame_id':f['frame_id'],**f['quality']} for f in frames],
            'object_detection':{'available':detector is not None,'frames':[{'frame_id':f['frame_id'],'detections':f['detections']} for f in frames]},
            'full_video_context':{'scope':'full_source_context_not_interval_verdict','analysis':manifest.get('analysis'),
                                  'manifest_path':str(analysis_path/'manifest.json'), 'cache_hit':cache_hit},
            'events':[{'frame_id':c['frame_id'],'timestamp_s':c['timestamp_s'],'events':c.get('track_events',[]),
                       'scene_cut_score':c.get('scores',{}).get('scene_cut_score')} for c in manifest['candidates']
                      if (at_s is None and (start_s is None or start_s<=c['timestamp_s']<=end_s))],
            'unknowns':['Grasp/contact/task success are not determined by detector or visual change alone'],
            'task_success':'unverified','semantic_interpretation':'execution model interprets these middleware results and images'}
        if detector is None:evaluation['unknowns'].append('No configured detector weights; object detection unavailable')
        timings['selected_detection_s']=time.monotonic()-phase
        if model is not None:
            progress('vision_evaluation');phase=time.monotonic()
            from ..demonstration.compiler import SEMANTIC_SCHEMA,_validate_semantic_response
            # A cheap perception layer proposes frames; the fast vision API
            # only reviews boundaries and the strongest change/clarity sample.
            chosen=[frames[0]]
            if len(frames)>2:
                chosen.append(max(frames[1:-1],key=lambda f:(f['quality']['sampled_visual_change'] or 0,f['quality']['clarity_laplacian_variance'])))
            if len(frames)>1:chosen.append(frames[-1])
            ids=[f['frame_id'] for f in chosen];times={f['frame_id']:f['timestamp_s'] for f in chosen}
            prompt=vision_evaluation_prompt(chosen)
            try:
                semantic=model.infer(prompt,[Path(f['image_path']) for f in chosen],SEMANTIC_SCHEMA)
            except Exception:
                call=getattr(model,'last_call',{})
                (output/'evaluation-error.json').write_text(json.dumps({k:call.get(k) for k in
                    ('status','error','elapsed_s','finish_reason','response_model','usage')},indent=2),encoding='utf-8')
                raise
            raw_semantic=semantic
            semantic,normalization=canonicalize_visual_references(semantic,chosen,source.sha256)
            errors=_validate_semantic_response(semantic,ids,times)
            if errors:raise VideoError('Vision evaluation failed validation: '+'; '.join(errors))
            evaluation['vision']={**semantic,'model_identity':model.identity,'frames_supplied':ids,'scope':'selected_historical_frames_only'}
            evaluation['vision']['reference_normalization']=normalization
            (output/'evaluation-raw.json').write_text(json.dumps(raw_semantic,ensure_ascii=False,indent=2),encoding='utf-8')
            evaluation['vision']['call_metadata']={k:model.last_call.get(k) for k in ('elapsed_s','response_model','usage')} if hasattr(model,'last_call') else {}
            evaluation['hierarchy']='CV / detector -> bounded fast vision evaluation -> execution agent'
            timings['vision_evaluation_s']=time.monotonic()-phase
        else:
            evaluation['vision']={'status':'not_configured','unknowns':['No fast visual evaluator configured']}
    result={'schema_version':1,'kind':'piperlab.video.watch','mode':mode,'address':address,
        'source':{'path':str(source.path),'sha256':source.sha256},
        'selection':{'at_s':at_s,'start_s':start_s,'end_s':end_s,'actual_times_s':[f['timestamp_s'] for f in frames]},
        'frames':frames,'evaluation':evaluation,'timings':timings,'scope':'historical_demonstration_only'}
    (output/'watch.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    progress('completed')
    return result
