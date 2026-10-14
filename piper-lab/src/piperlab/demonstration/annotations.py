"""Score compiled historical steps against separately supplied human annotations."""
from __future__ import annotations
import json
import math
import re
from html import escape
from pathlib import Path


def evaluate_annotations(demo,annotation):
    """Deterministic one-to-one matching, without a model grading its own output."""
    digest=annotation.get('source_hash','')
    if not re.fullmatch('[0-9a-f]{64}',digest) or digest!=demo.get('source_hash'):
        raise ValueError('annotation_source_hash_mismatch')
    provenance=annotation.get('provenance')
    if provenance not in ('human_review','synthetic_fixture'):raise ValueError('annotation_provenance_required')
    scope=annotation.get('scope','required_steps')
    if scope not in ('required_steps','exhaustive_steps'):raise ValueError('unsupported_annotation_scope')
    tolerance=annotation.get('time_tolerance_s',.25)
    threshold=annotation.get('minimum_time_iou',.1)
    if isinstance(tolerance,bool) or not isinstance(tolerance,(float,int)) or not math.isfinite(tolerance) or not 0<=tolerance<=2:
        raise ValueError('invalid_annotation_time_tolerance')
    if isinstance(threshold,bool) or not isinstance(threshold,(float,int)) or not math.isfinite(threshold) or not 0<threshold<=1:
        raise ValueError('invalid_annotation_time_iou')
    required=annotation.get('steps')
    if not isinstance(required,list) or not required:raise ValueError('annotation_steps_required')
    if len(required)>512 or len(demo.get('stages',[]))>4096:raise ValueError('annotation_matching_budget_exceeded')
    normal=lambda text:re.sub(r'\s+','_',text.strip().lower())
    aliases={}
    for canonical,alternatives in annotation.get('role_aliases',{}).items():
        if not isinstance(canonical,str) or not isinstance(alternatives,list) or not all(isinstance(v,str) and v.strip() for v in alternatives):
            raise ValueError('invalid_annotation_role_aliases')
        for name in [canonical,*alternatives]:
            name=normal(name)
            if name in aliases and aliases[name]!=normal(canonical):raise ValueError('ambiguous_annotation_role_alias')
            aliases[name]=normal(canonical)
    role=lambda value:aliases.get(normal(value),normal(value))
    ids=set();last_start=-1.
    for item in required:
        if not isinstance(item,dict) or not isinstance(item.get('id'),str) or not item['id'] or item['id'] in ids:
            raise ValueError('invalid_annotation_step_id')
        ids.add(item['id'])
        interval=item.get('time_range_s',[])
        if (not isinstance(interval,(list,tuple)) or len(interval)!=2 or any(isinstance(v,bool) or not isinstance(v,(float,int)) or not math.isfinite(v) for v in interval)
                or not 0<=interval[0]<=interval[1] or interval[0]<last_start):
            raise ValueError('invalid_annotation_step_time')
        last_start=interval[0]
        for field in ('operations','object_roles'):
            values=item.get(field)
            if not isinstance(values,list) or not values or not all(isinstance(v,str) and v.strip() for v in values):
                raise ValueError('annotation_operations_and_roles_required')
    frames={f['frame_id']:f['timestamp_s'] for f in demo.get('frames',[])}
    predicted=[];invalid=[]
    for index,stage in enumerate(demo.get('stages',[])):
        refs=stage.get('evidence_refs',[])
        if not refs or any(ref not in frames for ref in refs):
            invalid.append({'stage_id':stage.get('id'),'reason':'missing_or_unknown_evidence'});continue
        stamps=[frames[ref] for ref in refs]
        if not all(isinstance(t,(float,int)) and not isinstance(t,bool) and math.isfinite(t) for t in stamps):
            invalid.append({'stage_id':stage.get('id'),'reason':'invalid_evidence_timestamp'});continue
        predicted.append({'stage_id':stage['id'],'index':index,'operation':normal(stage['operation']),
            'roles':{role(r) for r in stage['object_roles']},'start':min(stamps),'end':max(stamps)})
    edges=[]
    for item in required:
        start,end=item['time_range_s'];start=max(0,start-tolerance);end+=tolerance
        candidates=[]
        for j,stage in enumerate(predicted):
            if stage['operation'] not in {normal(v) for v in item['operations']}:continue
            if not {role(v) for v in item['object_roles']}<=stage['roles']:continue
            # A single evidence frame is a point interval. Expand both sides
            # by one millisecond solely to make the overlap score defined.
            pstart,pend=stage['start'],max(stage['end'],stage['start']+.001)
            union=max(end,pend)-min(start,pstart)
            overlap=max(0.,min(end,pend)-max(start,pstart))
            if stage['start']==stage['end'] and start<=stage['start']<=end:iou=1.
            else:iou=overlap/union if union>0 else 0.
            if iou>=threshold:candidates.append((j,iou))
        edges.append(sorted(candidates,key=lambda pair:(-pair[1],predicted[pair[0]]['index'])))
    owners={}
    def assign(i,seen):
        for j,_ in edges[i]:
            if j in seen:continue
            seen.add(j)
            if j not in owners or assign(owners[j],seen):owners[j]=i;return True
        return False
    for i in range(len(required)):assign(i,set())
    by_required={i:j for j,i in owners.items()}
    matches=[{'annotation_id':required[i]['id'],'stage_id':predicted[j]['stage_id'],
              'time_iou':dict(edges[i])[j],'stage_index':predicted[j]['index']} for i,j in sorted(by_required.items())]
    ambiguous=[required[i]['id'] for i,e in enumerate(edges) if len(e)>1]
    indices=[m['stage_index'] for m in matches]
    order='unknown' if len(matches)!=len(required) or ambiguous else ('supported' if indices==sorted(indices) else 'refuted')
    return {'scope':'historical_demo_annotation_only','demo_id':demo.get('demo_id'),'source_hash':digest,
        'annotation_provenance':provenance,'annotation_scope':scope,
        'provenance_verification':'Caller declaration; this tool does not certify who produced the video or annotations.',
        'method':'Maximum-cardinality one-to-one operation/role/time matching with predeclared lexical aliases; no semantic model calls',
        'time_tolerance_s':tolerance,'minimum_time_iou':threshold,
        'required_steps':len(required),'predicted_steps':len(predicted)+len(invalid),'matched_steps':len(matches),
        'recall':len(matches)/len(required),
        'precision':len(matches)/(len(predicted)+len(invalid)) if scope=='exhaustive_steps' and (predicted or invalid) else None,
        'order_verdict':order,'matches':matches,'ambiguous_steps':ambiguous,
        'missing_steps':[item['id'] for i,item in enumerate(required) if i not in by_required],
        'unmatched_predictions':[s['stage_id'] for j,s in enumerate(predicted) if j not in owners],
        'invalid_predictions':invalid,'compiler_outcome':demo.get('outcome_verdict'),
        'note':'Historical step coverage does not establish robot execution success or learning improvement.'}


def write_annotation_report(result,annotation,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    (output/'annotations.json').write_text(json.dumps(annotation,ensure_ascii=False,indent=2),encoding='utf-8')
    (output/'report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    (output/'report.html').write_text('<!doctype html><meta charset="utf-8"><title>Demonstration annotation evaluation</title>'
        '<h1>Historical demonstration step evaluation</h1><p>Independent supplied annotations; no model-based grading.</p><pre>'
        +escape(json.dumps(result,ensure_ascii=False,indent=2))+'</pre>',encoding='utf-8')
