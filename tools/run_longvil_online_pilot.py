"""Small eight-condition job over frozen tools and native contact physics."""
import argparse
from datetime import datetime,timezone
import hashlib
import itertools
import json
import os
from pathlib import Path
import random
import shlex
import shutil
import subprocess
import sys
import tarfile

FORMAL_TASKS=['level1_4_vegetablesorting_4','level2_8_blockmanipulation_8']
DEVELOPMENT_TASK='level1_4_blockmanipulation_3'
BUDGET={'completion_budget':65536,'max_rounds':80,'max_tools':120,'max_seconds':1800}
SEED=20261011

def save(path,value):
    path=Path(path);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False));tmp.replace(path)
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def ssh(config):return ['ssh','-T','-S',config['socket'],'-p',str(config['port']),'-o','BatchMode=yes','-o','StrictHostKeyChecking=yes',config['host']]
def conditions():
    return [{'plan':p,'skill':s,'effort':e,'condition':f'P{int(p=="on")}S{int(s=="on")}-{e}'}
            for p,s,e in itertools.product(('off','on'),('off','on'),('low','high'))]
def feedback_conditions():
    return [dict(plan='on',skill='on',effort='low',decision_policy=p,call_completion_cap=cap,condition=p+'-low')
            for p,cap in [('standard',65536),('short-feedback',8192)]]
def schedule(specs=None):
    rng=random.Random(SEED);result=[]
    for phase,tasks in [('development',[DEVELOPMENT_TASK]),('formal',FORMAL_TASKS)]:
        for task in tasks:
            cases=[dict(c) for c in (specs if specs is not None else conditions())];rng.shuffle(cases)
            for item in cases:result.append(dict(item,phase=phase,task=task,id=phase+'--'+task+'--'+item['condition']))
    return result

def prepare(args):
    from run_longvil_online import prompt_recipe,PROMPT,PLAN_CONTRACT
    study=getattr(args,'study','factorial')
    specs=feedback_conditions() if study=='short-feedback' else conditions()
    job=args.job.resolve();job.mkdir(parents=True,exist_ok=False);source=job/'source'
    for part in ('piperx-cli/src','piper-lab/src','piperx-cli/assets/mujoco'):
        shutil.copytree(args.repo/part,source/part,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    (source/'tools').mkdir()
    for path in (args.repo/'tools').glob('*.py'):shutil.copy2(path,source/'tools'/path.name)
    original=json.loads((args.reference_job/'manifest.json').read_text());prior=json.loads((args.reference_job/'launch-config.json').read_text())
    selected=[next(t for t in original['tasks'] if t['id']==ident) for ident in [DEVELOPMENT_TASK,*FORMAL_TASKS]]
    passed={r['task'] for r in json.loads((args.reference_job/'preflight-summary.json').read_text())['rows'] if r['success'] is True}
    evidence=json.loads((args.reference_job/'evidence.json').read_text());inputs={};tasks={};frozen_evidence={}
    for task in selected:
        ident=task['id'];video=task['videos']['complex']
        if ident not in passed or task.get('review_status')!='approved' or video.get('review_status')!='approved':
            raise ValueError('Unreviewed or physically infeasible task: '+ident)
        folder=job/'private-data'/ident;folder.mkdir(parents=True)
        if digest(task['annotation'])!=task['sha256'] or digest(video['path'])!=video['sha256']:raise ValueError('Dataset source changed')
        shutil.copy2(task['annotation'],folder/'reference.json');annotation=json.loads((folder/'reference.json').read_text())
        packet=evidence[ident]
        for kind in ('initial','supplement'):
            for i,frame in enumerate(packet[kind]['frames']):
                if digest(frame['image_path'])!=frame['image_sha256']:raise ValueError('Evidence frame changed')
                target=folder/f'{kind}-{i:02d}.jpg';shutil.copy2(frame['image_path'],target);frame['image_path']=str(target)
        frozen_evidence[ident]=packet
        scene=args.reference_job/'scene-metadata'/(ident+'.json');shutil.copy2(scene,folder/'scene-metadata.json')
        tasks[ident]={'annotation':str(folder/'reference.json'),'objects':annotation['object_list'],
                     'scene':prior['remote_root']+'/scenes/'+ident+'.xml','scene_sha256':json.loads(scene.read_text())['scene_sha256'],
                     'source_video_sha256':video['sha256'],'review_status':'approved','reference_physics_pass':True}
    save(job/'tasks.json',tasks);save(job/'evidence.json',frozen_evidence);save(job/'schedule.json',schedule(specs))
    prompts={}
    for c in specs:
        text,recipe=prompt_recipe(c['plan']=='on',c['skill']=='on',c.get('decision_policy','standard'))
        prompts[c['condition']]={'system':text,'recipe':recipe,'system_sha256':hashlib.sha256(text.encode()).hexdigest()}
    save(job/'prompts.json',prompts)
    config={'host':args.host,'port':args.port,'socket':str(args.socket),'remote_root':args.remote_root,
        'remote_python':prior['remote_python'],'python':sys.executable,'node':str(args.node),'key_launcher':str(args.key_launcher),
        'model':'deepseek-flash','thinking':'enabled','budget':BUDGET,'seed':SEED,
        'expected_development':len(specs),'expected_formal':2*len(specs),'retry_on_anomaly':False,'study':study,'condition_specs':specs,
        'scope':'Two reviewed complex tasks, native contact physics; pilot only, not hardware or significance',
        'clock':'Existing real-time simulation; effort includes latency/physical waiting effects',
        'primary_outcome':'Measured goal coverage and final physical relations; demonstrated order reported separately',
        'plan_policy':'soft','score_revision':'goal-recovery-v2','contact_development':True,
        'price_snapshot':'2026-10-09 ledger snapshot; estimates only'}
    save(job/'launch-config.json',config)
    save(job/'frozen-hashes.json',{str(p.relative_to(job)):digest(p) for p in job.rglob('*') if p.is_file()})
    save(job/'state.json',{'status':'prepared','at':datetime.now(timezone.utc).isoformat()})
    return config

def verify(job):
    for rel,expected in json.loads((job/'frozen-hashes.json').read_text()).items():
        if digest(job/rel)!=expected:raise RuntimeError('Frozen input changed: '+rel)

def deploy(job,config,log):
    bundle=job/'source.tar.gz'
    with tarfile.open(bundle,'w:gz') as archive:archive.add(job/'source',arcname='source')
    remote=config['remote_root'];python=config['remote_python']
    subprocess.run([*ssh(config),shlex.join([python,'-c','from pathlib import Path;Path('+repr(remote)+').mkdir(parents=True,exist_ok=False)'])],check=True,stdout=log,stderr=log,timeout=60)
    subprocess.run(['scp','-P',str(config['port']),'-o','ControlPath='+config['socket'],'-o','BatchMode=yes','-o','StrictHostKeyChecking=yes',str(bundle),config['host']+':'+remote+'/source.tar.gz'],check=True,stdout=log,stderr=log,timeout=180)
    code='from pathlib import Path;import tarfile,hashlib;r=Path('+repr(remote)+');p=r/"source.tar.gz";assert hashlib.sha256(p.read_bytes()).hexdigest()=='+repr(digest(bundle))+';tarfile.open(p).extractall(r,filter="data")'
    subprocess.run([*ssh(config),shlex.join([python,'-c',code])],check=True,stdout=log,stderr=log,timeout=90)

def command(job,config,item,folder,*,probe=False):
    task=json.loads((job/'tasks.json').read_text())[item['task']]
    cmd=[config['python'],str(job/'source/tools/run_longvil_online.py'),'--output',str(folder),
        '--annotation',task['annotation'],'--evidence',str(job/'evidence.json'),'--socket',config['socket'],
        '--host',config['host'],'--port',str(config['port']),'--remote-root',config['remote_root'],
        '--remote-python',config['remote_python'],'--scene',task['scene'],'--task',item['task'],
        '--object-list',json.dumps(task['objects']),'--plan',item['plan'],'--skill',item['skill'],'--effort',item['effort']]
    for name,value in config['budget'].items():cmd.extend(['--'+name.replace('_','-'),str(value)])
    for name in ('decision_policy','call_completion_cap'):
        if name in item:cmd.extend(['--'+name.replace('_','-'),str(item[name])])
    if probe:cmd.append('--native-probe')
    elif item['phase']=='development':cmd.append('--contact-smoke')
    return cmd if probe else [config['node'],config['key_launcher'],*cmd]

def clean(report):
    r=report.get('cleanup',{})
    return r.get('runtime_remaining') is False and r.get('owned_process_alive') is False and r.get('recording_error') is None

def classification(report):
    if not clean(report):return 'infrastructure'
    if report['status']!='paused':return 'scored'
    errors=report.get('exceptions',[])
    if errors and all(e.get('type')=='ModelCallAnomaly' and e.get('message')=='Action outcome unknown; no retry or subsequent motion' for e in errors):
        return 'execution_failure'
    return 'infrastructure'

def development_gate(rows,expected=8):
    return {'complete':len(rows)==expected,'contact':all(r.get('interface_smoke_pass') is True and
        r.get('contact_smoke') is True and r.get('unknown_outcome') is not True for r in rows),
        'interface':sum(r.get('interface_errors',0) for r in rows)/max(1,sum(r.get('tool_calls',0) for r in rows))<=.10,
        'resources':all(clean(r) for r in rows)}

def effects(rows,metric='physical_success'):
    blocks={}
    for r in rows:blocks.setdefault(r['task'],{})[(r['plan'],r['skill'],r['effort'])]=int(r.get(metric) is True)
    blocks={k:v for k,v in blocks.items() if len(v)==8}
    result={'complete_task_blocks':len(blocks),'scope':'Paired within semantic tasks; descriptive only, no significance or precise compute claim','contrasts':{}}
    for index,name in enumerate(('plan','skill','effort')):
        levels=('low','high') if index==2 else ('off','on');per_task={}
        for task,block in blocks.items():
            other=sorted({tuple(k[i] for i in range(3) if i!=index) for k in block})
            deltas=[]
            for pair in other:
                keys=[]
                for level in levels:
                    key=list(pair);key.insert(index,level);keys.append(tuple(key))
                deltas.append(block[keys[1]]-block[keys[0]])
            per_task[task]=sum(deltas)/len(deltas)
        result['contrasts'][name]={'per_task':per_task,'mean_difference':sum(per_task.values())/len(per_task) if per_task else None}
    return result

def feedback_effects(rows):
    blocks={}
    for r in rows:blocks.setdefault(r['task'],{})[r['decision_policy']]=r
    paired={k:v for k,v in blocks.items() if set(v)=={'standard','short-feedback'}}
    per_task={}
    for task,block in paired.items():
        a,b=block['standard'],block['short-feedback']
        per_task[task]={k:(int(b.get(k) is True)-int(a.get(k) is True) if k=='physical_success' else b.get(k,0)-a.get(k,0))
                        for k in ('physical_success','api_calls','motions','completion_tokens','reasoning_tokens','elapsed_s')
                        if type(a.get(k)) in (int,float,bool) and type(b.get(k)) in (int,float,bool)}
    return {'complete_task_blocks':len(paired),'scope':'Short-feedback minus standard low; plan and skill on, same total budgets. Two paired tasks, exploratory only.',
            'paired_differences':per_task}

def summarize(job):
    rows=json.loads((job/'results.json').read_text()) if (job/'results.json').exists() else []
    config=json.loads((job/'launch-config.json').read_text());feedback=config.get('study')=='short-feedback'
    formal=[r for r in rows if r['phase']=='formal'];summary=feedback_effects(formal) if feedback else effects(formal)
    if not feedback:summary['ordered_effects']=effects(formal,metric='ordered_success')
    summary.update(score_revision=config.get('score_revision'),plan_policy=config.get('plan_policy'))
    summary.update(completed_development=sum(r['phase']=='development' for r in rows),completed_formal=len(formal))
    accounted=reserved=0.
    for file in (job/'runs').glob('*/attempt-*/usage-ledger.jsonl'):
        for r in map(json.loads,file.read_text().splitlines()):
            if r['status']=='accounted':accounted+=r['estimated_cny']
            elif r['status']=='usage_unavailable':reserved+=r['estimated_cny']
    summary.update(accounted_estimated_cny=accounted,unknown_usage_reserve_cny=reserved);save(job/'summary.json',summary)
    text=['# 短调用反馈策略物理仿真' if feedback else '# 八组在线物理仿真小实验','',summary['scope'],'',
          '| 场景 | Plan | Skill | Effort | 目标成功 | 顺序覆盖 | API尝试 | 工具尝试 | token | 秒 | 分类 |',
          '| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |']
    for r in formal:text.append(f"| {r['task']} ({r.get('decision_policy','standard')}) | {r['plan']} | {r['skill']} | {r['effort']} | {int(r.get('physical_success') is True)} | {int(r.get('ordered_success') is True)} | {r.get('api_calls',0)} | {r.get('tool_calls',0)} | {r.get('completion_tokens',0)} | {r.get('elapsed_s',0):.1f} | {r['classification']} |")
    text+=['',f"完整配对任务块：{summary['complete_task_blocks']}。"+('只有两个策略齐全的任务块进入差值。' if feedback else '只有8组齐全的任务块进入差值。'),'',
          '本轮为short-feedback与standard的low配对；固定plan/skill开启，总预算不变，不把共同夹爪说明的效果归为策略效果。' if feedback else 'Plan/Skill差值为on−off，effort为high−low；先在任务内平均其余两个变量的四个配对，再平均任务。',
           '推理设置会改变实际用量及等待；模拟器按既有实时规则推进，结果属于端到端系统效果。',
           'Plan 为可修订的软持久化建议，不以读写次数限制动作。失败尝试保留；目标成功要求实测转移覆盖、释放及终态关系，示范顺序另报。',
           '评分与共同执行层已修正，旧冻结结果不改写；新旧物理成功率不能作为同一规则下的因果对照。',
           '只包含两个独立语义任务，不做显著性、真机或泛化结论。基础设施异常暂停，不计作模型失败。',
           '',f"已记账用量估算：{accounted:.4f}元；未知用量保守预留：{reserved:.2f}元（不是实际消费）。"]
    (job/'report.zh.md').write_text('\n'.join(text)+'\n');return summary

def run(job,*,resume=False,retry_incomplete=False):
    job=job.resolve();verify(job);config=json.loads((job/'launch-config.json').read_text());awake=None
    cases=json.loads((job/'schedule.json').read_text());rows=json.loads((job/'results.json').read_text()) if (job/'results.json').exists() else []
    done={r['id'] for r in rows}
    if len(done)!=len(rows):raise RuntimeError('Duplicate scored case IDs')
    if (job/'submission.json').exists() and not resume:raise RuntimeError('Explicit --resume is required')
    def status(name,**details):save(job/'state.json',{'status':name,'at':datetime.now(timezone.utc).isoformat(),**details})
    env={k:v for k,v in os.environ.items() if k!='DEEPSEEK_API_KEY'}
    env['PYTHONPATH']=str(job/'source/piperx-cli/src')+':'+str(job/'source/piper-lab/src')
    save(job/'submission.json',{'pid':os.getpid(),'at':datetime.now(timezone.utc).isoformat(),'resume':resume})
    if sys.platform=='darwin':
        (job/'power-at-start.txt').write_text(subprocess.check_output(['pmset','-g','batt'],text=True))
        awake=subprocess.Popen(['caffeinate','-i','-s','-w',str(os.getpid())],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    try:
        with (job/'run.log').open('a') as log:
            if not (job/'native-probe/report.json').exists():
                status('deploying');deploy(job,config,log)
                probe=dict(cases[0],task=DEVELOPMENT_TASK)
                subprocess.run(command(job,config,probe,job/'native-probe',probe=True),env=env,stdout=log,stderr=log,check=True,timeout=180)
            report=json.loads((job/'native-probe/report.json').read_text())
            if report['status']!='native_probe_pass' or not clean(report):raise RuntimeError('Native CLI/camera/resources probe did not pass')
            acceptance=job/'native-feedback-acceptance'
            if not (acceptance/'report.json').exists():
                status('native_feedback_acceptance')
                subprocess.run([config['python'],str(job/'source/tools/validate_longvil_feedback_runtime.py'),
                    '--job',str(job),'--output',str(acceptance)],env=env,stdout=log,stderr=log,check=True,timeout=180)
            verified=json.loads((acceptance/'report.json').read_text())
            if verified.get('passed') is not True or not clean(verified):
                raise RuntimeError('Measured stop/recovery and capture mask acceptance did not pass')
            tasks=json.loads((job/'tasks.json').read_text())
            paths={k:t['scene'] for k,t in tasks.items()}
            code='import json,hashlib;from pathlib import Path;p='+repr(paths)+';print(json.dumps({k:hashlib.sha256(Path(v).read_bytes()).hexdigest() for k,v in p.items()}))'
            result=subprocess.run([*ssh(config),shlex.join([config['remote_python'],'-c',code])],check=True,capture_output=True,text=True,timeout=60)
            actual=json.loads(result.stdout)
            if any(actual[k]!=tasks[k]['scene_sha256'] for k in tasks):raise RuntimeError('Frozen physical scene changed')
            save(job/'scene-contract-audit.json',{'identical':True,'scene_sha256s':actual})
            from run_longvil_online import select_tools
            native=json.loads((job/'native-probe/native-tools.json').read_text());base={}
            for c in config.get('condition_specs',conditions()):
                tools=[t for t in select_tools(native,c['plan']=='on') if t['name']!='task_plan']
                base[c['condition']]=hashlib.sha256(json.dumps(tools,sort_keys=True).encode()).hexdigest()
            if len(set(base.values()))!=1:raise RuntimeError('Common tool schemas differ between conditions')
            save(job/'tool-contract-audit.json',{'base_schema_hashes':base,'identical':True})
            for item in cases:
                if item['id'] in done:continue
                verify(job)
                if item['phase']=='formal' and not (job/'formal-freeze.json').exists():
                    gate=development_gate([r for r in rows if r['phase']=='development'],config['expected_development']);save(job/'development-gates.json',gate)
                    if not all(gate.values()):status('development_failed',gates=gate);return
                    save(job/'formal-freeze.json',{'at':datetime.now(timezone.utc).isoformat(),'frozen_manifest_sha256':digest(job/'frozen-hashes.json'),
                        'development_gates':gate,'common_tool_schema_sha256':next(iter(base.values()))})
                parent=job/'runs'/item['id'];prior=sorted(parent.glob('attempt-*'))
                if prior and not retry_incomplete:status('paused_incomplete',case=item['id'],reason='Explicit --retry-incomplete required; no request started');return
                folder=parent/f'attempt-{len(prior)+1:02d}';parent.mkdir(parents=True,exist_ok=True)
                status(item['phase'],case=item['id'],completed=len(done),expected=len(cases))
                subprocess.run(command(job,config,item,folder),env=env,stdout=log,stderr=log,check=True)
                report=json.loads((folder/'report.json').read_text());kind=classification(report)
                if kind=='infrastructure':status('paused_infrastructure',case=item['id'],report=str(folder/'report.json'));return
                report.update(item,classification=kind,attempt=str(folder));rows.append(report);done.add(item['id']);save(job/'results.json',rows);summarize(job)
            status('complete',completed=len(done));summarize(job)
    except Exception as exc:status('paused_infrastructure',error_type=type(exc).__name__,error=str(exc)[:600]);raise
    finally:
        summarize(job)
        if awake is not None:awake.terminate();awake.wait(timeout=10)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='mode',required=True)
    prep=sub.add_parser('prepare')
    for name in ('job','repo','reference-job','socket','node','key-launcher'):prep.add_argument('--'+name,type=Path,required=True)
    prep.add_argument('--host',required=True);prep.add_argument('--port',type=int,default=54979);prep.add_argument('--remote-root',required=True)
    prep.add_argument('--study',choices=['factorial','short-feedback'],default='factorial')
    launch=sub.add_parser('run');launch.add_argument('--job',type=Path,required=True);launch.add_argument('--resume',action='store_true');launch.add_argument('--retry-incomplete',action='store_true')
    args=p.parse_args()
    if args.mode=='prepare':print(json.dumps(prepare(args),ensure_ascii=False))
    else:
        from longvil_budget import ledger_lock
        with ledger_lock(args.job/'.supervisor-owner'):
            run(args.job,resume=args.resume,retry_incomplete=args.retry_incomplete)
