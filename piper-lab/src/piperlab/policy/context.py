"""Bound long demonstrations by selecting one recorded semantic segment at a time."""
def segment_count(demo):
    return len((demo or {}).get('segments',[])) or 1


def active_demo(demo,index):
    if not demo or not demo.get('segments'):return demo
    segments=demo['segments']
    if not 0<=index<len(segments):raise ValueError('demonstration_segment_out_of_range')
    value=dict(segments[index])
    value['segment_context']={'index':index,'count':len(segments),
        'instruction':'Follow only this current video segment. done advances to the next segment; it does not complete the whole task until the final segment.'}
    return value
