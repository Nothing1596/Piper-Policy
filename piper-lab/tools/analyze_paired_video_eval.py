"""Read-only re-scoring of saved operator evidence; never modifies original episodes."""
import argparse
import json
import math
import os
import statistics
from html import escape
from pathlib import Path
from piperlab.policy.evaluate import score_episode


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',required=True);parser.add_argument('--suite',required=True)
    parser.add_argument('--output',required=True);parser.add_argument('--expected-episodes',type=int,default=90)
    args=parser.parse_args()
    root=Path(args.run).resolve();out=Path(args.output).resolve()
    if out.exists():raise FileExistsError(out)
    suite=json.loads(Path(args.suite).read_text(encoding='utf-8'))
    tasks={t['id']:t for t in suite['tasks']}
    rows=[]
    for line in (root/'episodes.jsonl').read_text(encoding='utf-8').splitlines():
        original=json.loads(line)
        episode=root/Path(original['report']).parent.name
        if not (episode/'operator-evaluation.json').is_file():
            raise FileNotFoundError(f'Saved episode evidence missing under supplied run: {episode}')
        report=json.loads((episode/'report.json').read_text(encoding='utf-8'))
        truth=json.loads((episode/'operator-evaluation.json').read_text(encoding='utf-8'))
        score=score_episode(tasks[original['task_id']],report,truth['before'],truth['after'])
        calls=[]
        for path in sorted(episode.with_name(episode.name+'-model-calls').glob('*.json')):
            record=json.loads(path.read_text(encoding='utf-8'))
            prompt=record.get('prompt','')
            calls.append({'elapsed_s':record['elapsed_s'],'status':record['status'],
                'corrective_prompt':('\nFORMAT CORRECTION:' in prompt or '\nCORRECTION:' in prompt),
                'reported_input_tokens':record.get('usage',{}).get('input_tokens'),
                'reported_output_tokens':record.get('usage',{}).get('total_output_tokens')})
        reason=original.get('failure_reason')
        if not score['success'] and not reason:
            reason=report['status'] if report['status']!='model_declared_done_pending_task_evaluation' else (
                'physical_goal_not_reached' if not score['physical_goal_reached'] else 'wrong_step_order')
        rows.append({**original,**score,'frozen_original_success':original['success'],
            'model_declared_done':report['status']=='model_declared_done_pending_task_evaluation',
            'completion_recheck_verdict':next((h['completion_recheck']['verdict'] for h in reversed(report.get('history',[])) if 'completion_recheck' in h),None),
            'report':str(episode/'report.html'),
            'failure_reason':reason,'inference_calls':len(calls),'corrective_requests':sum(c['corrective_prompt'] for c in calls),
            'inference_errors':sum(c['status']!='ok' for c in calls),'inference_elapsed_s':sum(c['elapsed_s'] for c in calls),
            'call_latencies_s':[c['elapsed_s'] for c in calls],'reported_tokens':calls})
    groups=[]
    conditions=('no_demo','correct_demo','shuffled_demo')
    keys=[(r['task_id'],r['seed'],r['condition']) for r in rows]
    if len(keys)!=len(set(keys)):raise ValueError('duplicate_episode_key')
    if any(task not in tasks or condition not in conditions for task,seed,condition in keys):
        raise ValueError('unexpected_task_or_condition')
    if args.expected_episodes<=0 or args.expected_episodes%(len(tasks)*len(conditions)):
        raise ValueError('expected_episodes_must_describe_equal_paired_groups')
    expected_seeds=args.expected_episodes//(len(tasks)*len(conditions))
    paired=[];complete=len(rows)==args.expected_episodes
    for task in tasks:
        indexed={condition:{r['seed']:r for r in rows if r['task_id']==task and r['condition']==condition} for condition in conditions}
        seed_sets=[set(indexed[c]) for c in conditions]
        complete=complete and all(s==seed_sets[0] and len(s)==expected_seeds for s in seed_sets)
        for control in ('no_demo','shuffled_demo'):
            seeds=sorted(set(indexed['correct_demo']) & set(indexed[control]))
            outcomes={'both_success':0,'correct_demo_only':0,'control_only':0,'both_failed':0}
            for seed in seeds:
                a=indexed['correct_demo'][seed]['success'];b=indexed[control][seed]['success']
                key='both_success' if a and b else 'correct_demo_only' if a else 'control_only' if b else 'both_failed'
                outcomes[key]+=1
            paired.append({'task_id':task,'control':control,'paired_seeds':seeds,'n':len(seeds),**outcomes,
                'observed_success_rate_difference':(outcomes['correct_demo_only']-outcomes['control_only'])/len(seeds) if seeds else None})
        for condition in conditions:
            group=[r for r in rows if r['task_id']==task and r['condition']==condition]
            failures={}
            for row in group:
                if not row['success']:failures[row['failure_reason']]=failures.get(row['failure_reason'],0)+1
            latencies=sorted(v for r in group for v in r['call_latencies_s'])
            groups.append({'task_id':task,'condition':condition,'n':len(group),'successes':sum(r['success'] for r in group),
                'physical_goal_reached':sum(r['physical_goal_reached'] for r in group),
                'order_correct':sum(r['order_correct'] for r in group),
                'model_declared_done':sum(r['model_declared_done'] for r in group),
                'done_but_physical_goal_failed':sum(r['model_declared_done'] and not r['physical_goal_reached'] for r in group),
                'original_successes':sum(r['frozen_original_success'] for r in group),
                'success_rate':sum(r['success'] for r in group)/len(group) if group else None,
                'median_episode_s':statistics.median(r['elapsed_s'] for r in group) if group else None,
                'median_call_s':statistics.median(latencies) if latencies else None,
                'p95_call_s':latencies[math.ceil(.95*len(latencies))-1] if latencies else None,
                'corrective_requests':sum(r['corrective_requests'] for r in group),'failures':failures})
    result={'complete':bool(complete),'completed_episodes':len(rows),
        'expected_episodes':args.expected_episodes,'scoring_version':'whole_block_v2',
        'input_provenance':suite.get('provenance'),'human_demo_verified':suite.get('human_demo_verified',False),
        'notes':['Original execution logs and original scores remain unchanged.',
            'Placement now checks full oriented block bounds and support height, not center containment alone.',
            'Push scoring checks final displacement/height; it does not certify continuous surface contact throughout motion.',
            'Model completion, physical goal, and observed step order are separate measurements. Missing completion recheck means not recorded, not passed.',
            'Shuffled control reverses stage/frame lists but retains original frame timestamps and visual content; it is not a consistent counterfactual video.',
            'Inference latencies are measured API-call times; p95 uses the nearest-rank method.',
            'Few trials and synthetic inputs do not establish generalized human-video learning.'],
        'aggregates':groups,'paired_comparisons':paired,'episodes':rows}
    out.mkdir(parents=True)
    (out/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    table=''.join('<tr>'+''.join('<td>'+escape(str(g[k]))+'</td>' for k in (
        'task_id','condition','n','successes','physical_goal_reached','order_correct','model_declared_done','done_but_physical_goal_failed'))+'</tr>' for g in groups)
    links=''.join('<li><a href="'+escape(os.path.relpath(r['report'],out).replace('\\','/'),quote=True)+'">'
        +escape(f"{r['task_id']} / {r['seed']} / {r['condition']}: success={r['success']}, reason={r['failure_reason']}")+'</a></li>' for r in rows)
    html='''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Paired simulation analysis</title><style>body{font:16px/1.5 system-ui,sans-serif;max-width:1200px;margin:32px auto;padding:0 20px;color:#172331}table{border-collapse:collapse;width:100%}td,th{padding:8px;border:1px solid #ccd3db;text-align:left}th{background:#edf2f6}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f4f6f8;padding:16px}.state{font-weight:bold;color:#8b4100}a{color:#075ca7}</style>
<h1>配对仿真评测</h1>'''
    html+='<p class="state">'+('完整运行' if result['complete'] else '部分运行：不能作为最终结果')+f" · {len(rows)}/{args.expected_episodes}</p>"
    html+='<p>所有通过数均为回合计数。步骤顺序列对没有顺序约束的任务不代表额外能力；模型宣布完成不等于物理目标达成。</p>'
    html+='<table><thead><tr>'+''.join('<th>'+x+'</th>' for x in ('任务','条件','回合','综合成功','物理达标','顺序符合','模型宣布完成','宣布完成但物理失败'))+'</tr></thead><tbody>'+table+'</tbody></table>'
    html+='<h2>口径与限制</h2><ul>'+''.join('<li>'+escape(n)+'</li>' for n in result['notes'])+'</ul><h2>逐回合证据</h2><ul>'+links+'</ul>'
    html+='<details><summary>相同种子的成对结果</summary><pre>'+escape(json.dumps(paired,ensure_ascii=False,indent=2))+'</pre></details>'
    html+='<details><summary>延迟、纠错与失败统计</summary><pre>'+escape(json.dumps(groups,ensure_ascii=False,indent=2))+'</pre></details></html>'
    (out/'index.html').write_text(html,encoding='utf-8')
    print(json.dumps({'complete':result['complete'],'episodes':len(rows),'successes':sum(r['success'] for r in rows),
                      'score_changes':sum(r['success']!=r['frozen_original_success'] for r in rows)}))


if __name__=='__main__':main()
