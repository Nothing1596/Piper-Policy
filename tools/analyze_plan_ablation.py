"""Read-only descriptive analysis of frozen paired CLI experiments."""
import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path
import statistics


def sensitivity_success(task, before, samples, after, z_tolerance):
    """Post-hoc threshold check; never overwrites the frozen primary score."""
    sequence = ['cube_red'] if task == 'hover_red' else ['cube_blue', 'cube_red']

    def reached(state, name):
        tcp = state['truth']['robot']['tcp_pos']
        target = before['truth']['objects'][name]['position']
        return (math.hypot(tcp[0] - target[0], tcp[1] - target[1]) <= .015
                and abs(tcp[2] - target[2] - .015 - .12) <= z_tolerance)

    progress = 0
    for sample in samples:
        if reached(sample, sequence[progress]):
            progress += 1
            if progress == len(sequence):
                break
    displacement = max(math.dist(before['truth']['objects'][name]['position'],
                                 after['truth']['objects'][name]['position'])
                       for name in ('cube_red', 'cube_blue'))
    return progress == len(sequence) and reached(after, 'cube_red') and displacement <= .005


def analyze(root):
    protocol=json.loads((root/'protocol.json').read_text())
    rows=[json.loads(l) for l in (root/'episodes.jsonl').read_text().splitlines()]
    keys=[(r['task'],r['seed'],r['condition']) for r in rows]
    if len(keys)!=len(set(keys)):raise ValueError('Duplicate episode key')
    if len({r['session_id'] for r in rows})!=len(rows):raise ValueError('Session reuse')
    for r in rows:
        path=root/r['label']/'events.jsonl'
        events=[json.loads(l) for l in path.read_text().splitlines()] if path.exists() else []
        calls=[e for e in events if e['kind']=='tool']
        r['reasoning_tokens']=sum(e.get('usage',{}).get('completion_tokens_details',{}).get('reasoning_tokens',0) for e in events if e['kind']=='api')
        r['attempted_tool_calls']=len(calls)
        r['tool_names']=dict(Counter(e['name'] for e in calls))
        r['error_codes']=dict(Counter(e['result']['error'].get('code','unknown') for e in calls if isinstance(e.get('result',{}).get('error'),dict)))
        r['plan_version_conflicts']=sum('Plan changed during capture' in str(e.get('result',{}).get('error','')) for e in calls)
        r['all_version_conflicts']=sum('Plan changed during capture' in str(e.get('result',{}).get('error','')) or
            (isinstance(e.get('result',{}).get('error'),dict) and e['result']['error'].get('code')=='version_conflict') for e in calls)
        folder=root/r['label']
        if 'success' in r and all((folder/name).exists() for name in ('private-before.json','private-after.json')):
            before=json.loads((folder/'private-before.json').read_text())
            after=json.loads((folder/'private-after.json').read_text())
            sample_path=folder/'private-motion-samples.json'
            samples=json.loads(sample_path.read_text()) if sample_path.exists() else []
            r['posthoc_z_sensitivity']={str(mm):sensitivity_success(r['task'],before,samples,after,mm/1000) for mm in (5,10,11,15,20)}
            if r['posthoc_z_sensitivity']['10']!=r['success']:
                raise ValueError('Independent primary score mismatch')
    expected={(r['task'],r['seed'],r['condition']) for r in protocol['schedule']}
    if not set(keys)<=expected:raise ValueError('Unexpected condition')
    groups=[]
    for c in dict.fromkeys(r['condition'] for r in protocol['schedule']):
        rs=[r for r in rows if r['condition']==c];valid=[r for r in rs if 'success' in r]
        good=[r for r in valid if r['success']]
        def mean(k,data=valid):return statistics.mean(r[k] for r in data) if data else None
        groups.append({'condition':c,'completed':len(rs),'scored':len(valid),'successes':len(good),
          'success_and_reobserved':sum(r['success'] and r['post_motion_observation'] for r in valid),
          'success_reobserved_and_model_finished':sum(r['success'] and r['post_motion_observation'] and r['status']=='model_finished' for r in valid),
          'mean_elapsed_s':mean('elapsed_s'),'mean_success_elapsed_s':mean('elapsed_s',good),
          'mean_api_calls':mean('api_calls'),'mean_tool_calls':mean('tool_calls'),'mean_observations':mean('observations'),
          'mean_attempted_tool_calls':mean('attempted_tool_calls'),
          'mean_api_time_s':mean('api_time_s'),'mean_tool_time_s':mean('tool_time_s'),
          'mean_images_sent':mean('images_sent'),
          'total_tokens':sum(r['usage'].get('total_tokens',0) for r in rs),
          'completion_tokens':sum(r['usage'].get('completion_tokens',0) for r in rs),
          'reasoning_tokens':sum(r['reasoning_tokens'] for r in rs),
          'cached_prompt_tokens':sum(r['usage'].get('prompt_cache_hit_tokens',0) for r in rs),
          'plan_version_conflicts':sum(r['plan_version_conflicts'] for r in rs),
          'all_version_conflicts':sum(r['all_version_conflicts'] for r in rs),
          'posthoc_z_sensitivity_successes':{str(mm):sum(r.get('posthoc_z_sensitivity',{}).get(str(mm),False) for r in valid) for mm in (5,10,11,15,20)},
          'statuses':dict(Counter(r['status'] for r in rs))})
    pairs=[]
    index={k:r for k,r in zip(keys,rows)}
    for factor in ('plan','geometry'):
        for other in (False,True):
            if factor=='plan':a,b=('plan_','no_plan_');suffix='geometry' if other else 'no_geometry';a+=suffix;b+=suffix
            else:a=('plan_' if other else 'no_plan_')+'geometry';b=('plan_' if other else 'no_plan_')+'no_geometry'
            counts=Counter();deltas=[]
            for task,seed in dict.fromkeys((r['task'],r['seed']) for r in rows):
                ra,rb=index.get((task,seed,a)),index.get((task,seed,b))
                if not ra or not rb or 'success' not in ra or 'success' not in rb:continue
                counts['pairs']+=1
                counts['both_success' if ra['success'] and rb['success'] else 'treatment_only' if ra['success'] else 'control_only' if rb['success'] else 'both_failed']+=1
                deltas.append(ra['elapsed_s']-rb['elapsed_s'])
            pairs.append({'factor':factor,'treatment':a,'control':b,**dict(counts),'mean_time_difference_s':statistics.mean(deltas) if deltas else None})
    initial=[]
    for task,seed in dict.fromkeys((r['task'],r['seed']) for r in rows):
        rs=[r for r in rows if r['task']==task and r['seed']==seed and (root/r['label']/'private-before.json').exists()]
        states=[json.loads((root/r['label']/'private-before.json').read_text()) for r in rs]
        maximum=0.
        if states:
            for s in states[1:]:
                for name in ('cube_red','cube_blue'):
                    maximum=max(maximum,math.dist(s['truth']['objects'][name]['position'],states[0]['truth']['objects'][name]['position']))
        initial.append({'task':task,'seed':seed,'conditions':len(states),'max_initial_object_delta_m':maximum})
    return {'complete':set(keys)==expected,'episodes':len(rows),'expected':len(expected),
            'all_sessions_unique':True,'all_runtimes_released':all(r.get('cleanup')=={'runtime_remaining':False} for r in rows),
            'groups':groups,'paired':pairs,'initial_matching':initial,'rows':rows}


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=analyze(a.run);a.output.mkdir(parents=True,exist_ok=False)
    (a.output/'summary.json').write_text(json.dumps(result,indent=2,ensure_ascii=False))
    keys=['task','seed','condition','status','success','post_motion_observation','api_calls','tool_calls','attempted_tool_calls','observations','motions','errors','plan_version_conflicts','elapsed_s','api_time_s','tool_time_s','session_id']
    with (a.output/'episodes.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows({k:r.get(k) for k in keys} for r in result['rows'])
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
