"""Paired-seed experiments. Simulator truth stays exclusively in this module."""
import copy
import json
import random
from pathlib import Path
from .runner import run_policy

CONDITIONS=('no_demo','correct_demo','shuffled_demo')


def score_episode(spec,report,before,after):
    """Use explicit task annotations and physical scoring, not VLM done claims."""
    expected=spec.get('expected_order',[])
    actual=[]
    for entry in report.get('history',[]):
        if entry.get('verification',{}).get('verdict')=='supported':
            label=entry['decision']['object_label'].lower()
            color=next((c for c in ('red','blue') if c in label),None)
            if color and (not actual or actual[-1]!=color):actual.append(color)
    objects=after.get('objects',{})
    placement_checks={}
    if spec['kind'] in ('transfer','ordered_place'):
        required=spec.get('objects',expected or ['red'])
        def placed(color):
            obj=objects.get('cube_'+color,{})
            from .task_scoring import placed_block
            placement_checks[color]=placed_block(obj,after['tray'],spec.get('object_size_m',.03))
            return placement_checks[color]['success']
        positions_ok=all([placed(c) for c in required])
    elif spec['kind']=='push':
        import numpy as np
        color=spec.get('objects',['red'])[0];name='cube_'+color
        start=np.asarray(before['objects'][name]['position'])
        finish=np.asarray(after['objects'][name]['position'])
        if spec.get('direction')=='image_right_on_table':
            transform=np.asarray(before['camera_info']['base_from_camera'])
            rotation=transform[:3,:3]
            ray=start-transform[:3,3]
            tangent=rotation[:,0]-ray*(rotation[2,0]/ray[2])
            direction=tangent[:2]
        else:direction=np.asarray(spec['direction_xy'],dtype=float)
        direction=direction/np.linalg.norm(direction)
        delta=finish[:2]-start[:2]
        projection=float(delta@direction)
        lateral=float(np.linalg.norm(delta-projection*direction))
        positions_ok=projection>=spec.get('minimum_distance_m',.05) and lateral<=.04 and abs(finish[2]-start[2])<.02
    else:raise ValueError('unknown_eval_task_kind')
    order_ok=not expected or actual[:len(expected)]==expected
    return {'success':bool(positions_ok and order_ok and report['status'] not in ('error','execution_failed')),
            'scoring_version':'whole_block_v2','placement_checks':placement_checks,
            'physical_goal_reached':bool(positions_ok),'order_correct':order_ok,'observed_order':actual,
            'expected_order':expected,'human_interventions':0,'policy_status':report['status']}


def run_suite(executor,model_factory,suite,output_dir,load_demo,*,seeds=range(10),conditions=CONDITIONS):
    output=Path(output_dir).resolve();output.mkdir(parents=True,exist_ok=False)
    if not suite.get('tasks'):raise ValueError('suite_has_no_tasks')
    rows=[]
    for spec in suite['tasks']:
        original=load_demo(spec['demo'])
        for seed in seeds:
            for condition in conditions:
                if condition not in CONDITIONS:raise ValueError('unknown_experiment_condition')
                episode=output/f"{spec['id']}-{seed}-{condition}"
                executor.stop()
                executor.request('POST',f'/operator/simulation/reset?seed={seed}',{},operator=True)
                before=executor.evaluate()
                before['camera_info']=executor.capture()['camera_info']
                demo=None if condition=='no_demo' else copy.deepcopy(original)
                if condition=='shuffled_demo':
                    demo['stages']=list(reversed(demo['stages']))
                    demo['frames']=list(reversed(demo['frames']))
                    demo['demo_id']+='-shuffled-control'
                    demo['summary']='Follow the stages in their listed order.'
                    # Do not leak the original sequence through redundant prose.
                    demo['goal_constraints']=[]
                    for stage in demo['stages']:
                        stage['preconditions']=[]
                model=model_factory(episode.with_name(episode.name+'-model-calls'))
                report=run_policy(executor,model,task=spec['task'],output_dir=episode,demo=demo,
                                  max_decisions=suite.get('max_decisions',8),skill_config=suite.get('skill_config'))
                after=executor.evaluate()
                result=score_episode(spec,report,before,after)
                row={'task_id':spec['id'],'seed':seed,'condition':condition,**result,
                     'report':str(episode/'report.html'),'decisions':report['decisions'],
                     'elapsed_s':report['elapsed_s'],'model_identity':model.identity,
                     'failure_reason':report.get('error') or next((h.get('execution',{}).get('error') for h in report['history'] if h.get('execution',{}).get('error')),None),
                     'discarded_responses':sum(h.get('outcome','').startswith('discarded') for h in report['history'])}
                (episode/'operator-evaluation.json').write_text(json.dumps({'result':row,'before':before,'after':after},indent=2),encoding='utf-8')
                rows.append(row)
                with (output/'episodes.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row)+'\n')
    aggregates=[]
    for task in suite['tasks']:
        for condition in conditions:
            group=[r for r in rows if r['task_id']==task['id'] and r['condition']==condition]
            aggregates.append({'task_id':task['id'],'condition':condition,'n':len(group),'successes':sum(r['success'] for r in group),
                               'success_rate':sum(r['success'] for r in group)/len(group) if group else None})
    result={'suite':suite.get('name'),'input_provenance':suite.get('provenance','unspecified'),
            'human_demo_verified':suite.get('human_demo_verified',False),'aggregates':aggregates,'episodes':rows,
            'interpretation':'Report observed rates only; this experiment does not establish general learning ability.'}
    (output/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    from html import escape
    links=''.join('<li><a href="'+escape(str(Path(r['report']).relative_to(output)).replace('\\','/'))+'">'+escape(f"{r['task_id']} seed={r['seed']} {r['condition']} success={r['success']}")+'</a></li>' for r in rows)
    (output/'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>Paired simulation evaluation</h1><pre>'+escape(json.dumps(aggregates,indent=2))+'</pre><ul>'+links+'</ul>',encoding='utf-8')
    return result
