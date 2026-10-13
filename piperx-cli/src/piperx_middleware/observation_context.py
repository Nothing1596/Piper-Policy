"""Shared historical-image labels; evidence is separate from trusted workflow."""
import math


def demo_caption(result, frame, *, initial=False):
    ident=frame.get('frame_id');stamp=frame.get('timestamp_s')
    if not isinstance(ident,str) or type(stamp) not in (int,float) or not math.isfinite(stamp):
        raise ValueError('Demonstration frame needs an ID and real timestamp')
    source=(result.get('source') or {}).get('sha256') or result.get('demo_id') or 'selected demo'
    role='RECORDED INITIAL SCENE + demonstration start; NOT live' if initial else 'Historical demonstration; NOT live'
    return f'{role}: {source} / {ident} at {stamp}s'


def label_demo_images(result, images, *, initial=False):
    """Attach one real frame label to every transmitted image; fail on mismatch."""
    frames=result.get('frames',[])
    if not frames or len(frames)!=len(images) or len(frames)>12:
        raise ValueError('Demonstration metadata/image count mismatch')
    parts=[];manifest=[]
    for index,(frame,picture) in enumerate(zip(frames,images)):
        if picture.get('type')!='image_url':raise ValueError('Expected demonstration pixels')
        parts.extend([{'type':'text','text':demo_caption(result,frame,initial=initial and index==0)},picture])
        manifest.append({'source':(result.get('source') or {}).get('sha256'),
                         'frame_id':frame['frame_id'],'timestamp_s':frame['timestamp_s'],
                         'role':'recorded_initial' if initial and index==0 else 'demo',
                         'evaluation_scope':(result.get('evaluation') or {}).get('scope')})
    return parts,manifest


def append_demo_group(messages,recent,parts):
    """Keep the pinned initial message plus two individual tool-image groups."""
    if sum(p.get('type')=='image_url' for p in parts)>12:
        raise ValueError('Historical group exceeds twelve images')
    while len(recent)>=2:
        index=recent.pop(0)
        messages[index]['content']=[p for p in messages[index]['content'] if p.get('type')!='image_url']
    recent.append(len(messages))
    messages.append({'role':'user','content':[{'type':'text','text':'Additional historical evidence; not live:'},*parts]})
