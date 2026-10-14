"""Prepare reviewed complex-only evidence; no model calls or robot connection."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


def select_tasks(source_job):
    source_job = Path(source_job)
    manifest = json.loads((source_job/'manifest.json').read_text())
    schedule = json.loads((source_job/'schedule.json').read_text())
    ids = {r['task'] for r in schedule if r['pipeline']=='initial_vlm'
           and r['variant']=='complex' and r['condition']=='baseline'}
    tasks = [copy.deepcopy(t) for t in manifest['tasks'] if t['id'] in ids]
    if len(tasks)!=9 or len({(t['level'],t['category']) for t in tasks})!=9:
        raise ValueError('Expected the previous fixed nine-stratum sample')
    dev = copy.deepcopy(next(t for t in manifest['tasks']
                             if t['id']=='level3_16_blockmanipulation_2'))
    if dev['id'] in ids:
        raise ValueError('Development task overlaps pilot')
    for task in [*tasks,dev]:
        if task.get('review_status')!='approved' or task['videos']['complex'].get('review_status')!='approved':
            raise ValueError('Task/video was not approved')
        if digest(task['annotation'])!=task['sha256'] or digest(task['videos']['complex']['path'])!=task['videos']['complex']['sha256']:
            raise ValueError('Reviewed input bytes changed')
        task['videos'] = {'complex':task['videos']['complex']}
    return sorted(tasks,key=lambda t:t['id']),dev,manifest['dataset_revision']


def packet(source_sha, frames):
    """Only selected-frame evidence; never full-source events or inferred actions."""
    return {'source':{'sha256':source_sha},'frames':frames,
        'evaluation':{'scope':'selected_historical_frames','status':'evaluated',
            'middleware':['PTS validation','OpenCV clarity','frozen ONNX detector'],
            'quality':[{'frame_id':f['frame_id'],**f.get('quality',{})} for f in frames],
            'object_detection':{'available':True,'frames':[{'frame_id':f['frame_id'],'detections':f.get('detections',[])} for f in frames]},
            'task_success':'unverified','vision':{'status':'not_configured'},
            'unknowns':['These recorded frames do not establish current robot state or grasp/contact success.']}}


def prepare(source_job, root, detector_path):
    from piperlab.video.source import VideoSource
    from piperlab.video.watch import watch
    from piperlab.perception.detector_onnx import OnnxDetector
    tasks,dev,revision = select_tasks(source_job)
    root=Path(root); data=root/'prepared'; data.mkdir(exist_ok=False)
    detector=OnnxDetector(str(detector_path)); evidence={}
    for task in [dev,*tasks]:
        dest=data/task['id'];dest.mkdir()
        annotation=dest/'reference.json';shutil.copyfile(task['annotation'],annotation)
        task['annotation']=str(annotation)
        video=dest/'complex.mp4';shutil.copyfile(task['videos']['complex']['path'],video)
        task['videos']['complex']['path']=str(video)
        source=VideoSource(video); index=source.frame_index()['frames']
        lo,hi=index[0]['timestamp_s'],index[-1]['timestamp_s']
        initial=watch(str(video),dest/'initial',mode='pipeline',max_frames=12,
                      detector=detector,analysis_cache=data/'cv-cache')
        supplement=[]
        for i in range(12):
            requested=lo+(i+.5)*(hi-lo)/12
            result=watch(str(video),dest/f'supplement-{i:02d}',mode='pipeline',at_s=requested,
                         max_frames=1,detector=detector,analysis_cache=data/'cv-cache')
            supplement.extend(result['frames'])
        # Keep actual decoded identities; no fabricated timestamp or duplicated picture.
        supplement=list({f['frame_id']:f for f in supplement}.values())
        supplement.sort(key=lambda f:f['timestamp_s'])
        packs=[]
        for frames in (initial['frames'],supplement):
            if not frames or len(frames)>12:raise ValueError('Invalid frozen evidence size')
            for f in frames:f['image_sha256']=digest(f['image_path'])
            packs.append(packet(source.sha256,frames))
        evidence[task['id']]={'source_sha256':source.sha256,'initial':packs[0],'supplement':packs[1]}
        save(root/'evidence.partial.json',evidence)
        print(json.dumps({'prepared':task['id'],'initial':len(packs[0]['frames']),
                          'supplement':len(packs[1]['frames'])}),flush=True)
    manifest={'tasks':tasks,'development_task':dev,'dataset_revision':revision,
              'approval':'approved','variant':'complex','scope':'known-video pilot, not an independent formal test',
              'source_job':str(source_job),'detector_sha256':digest(detector_path)}
    save(root/'manifest.json',manifest);save(root/'evidence.json',evidence)
    save(root/'preparation-checks.json',{'tasks':9,'development_tasks':1,'strata':9,
        'complex_only':True,'paid_calls':0,'full_source_context_projected_out':True})
    return manifest,evidence


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-job',type=Path,required=True)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--detector',type=Path,required=True)
    args=parser.parse_args();prepare(args.source_job,args.root,args.detector)
