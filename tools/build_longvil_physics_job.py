"""Freeze and submit one serial native-physics job; never retry anomalies.

The supervisor runs task feasibility, then six development sessions, then the
54 comparisons. It is a job process, not a recurring monitoring service.
"""
import argparse
import copy
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tarfile

TOOLS=['build_longvil_physics_job.py','run_longvil_physics.py','longvil_physics_host.py',
       'longvil_physics_scene.py','longvil_physics_execute.py','longvil_physics_score.py',
       'validate_longvil_physics.py','run_plan_enforcement_pilot.py','plan_protocol.py',
       'run_longvil_v2.py','longvil_metrics.py','longvil_budget.py']


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,value):
    path=Path(path);temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False));temp.replace(path)


def ssh(config):
    return ['ssh','-T','-S',config['socket'],'-p',str(config['port']),
            '-o','BatchMode=yes','-o','StrictHostKeyChecking=yes',config['host']]


def prepare(args):
    job=args.job.resolve();job.mkdir(parents=True,exist_ok=False);source=job/'source'
    for part in ('piperx-cli/src','piper-lab/src','piperx-cli/assets/mujoco'):
        shutil.copytree(args.repo/part,source/part,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    (source/'tools').mkdir()
    for name in TOOLS:shutil.copyfile(args.repo/'tools'/name,source/'tools'/name)
    manifest=copy.deepcopy(json.loads((args.data/'manifest.json').read_text()))
    evidence=copy.deepcopy(json.loads((args.data/'evidence.json').read_text()));translations={}
    tasks=[*manifest['tasks'],manifest['development_task']]
    if len(manifest['tasks'])!=9 or len({t['id'] for t in tasks})!=10:
        raise ValueError('Exactly nine fixed tasks and one distinct development task are required')
    for task in tasks:
        ident=task['id'];dest=job/'private-data'/ident;dest.mkdir(parents=True)
        annotation=Path(task['annotation']);target=dest/'reference.json'
        if digest(annotation)!=task['sha256']:raise ValueError('Annotation changed: '+ident)
        shutil.copyfile(annotation,target);task['annotation']=str(target)
        video=task['videos']['complex']
        if digest(video['path'])!=video['sha256']:raise ValueError('Recorded source video changed: '+ident)
        for packet in ('initial','supplement'):
            for i,frame in enumerate(evidence[ident][packet]['frames']):
                target=dest/f'{packet}-{i:02d}.jpg'
                if digest(frame['image_path'])!=frame['image_sha256']:raise ValueError('Evidence image changed')
                shutil.copyfile(frame['image_path'],target);frame['image_path']=str(target)
        meta=json.loads((args.scene_metadata/(ident+'.json')).read_text())
        rotation=[row[:3] for row in meta['base_from_dataset'][:3]]
        if rotation!=[[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]]:
            raise ValueError('Fixed coordinate rotation changed; only translation is authorized')
        translations[ident]=[row[3] for row in meta['base_from_dataset'][:3]]
    save(job/'manifest.json',manifest);save(job/'evidence.json',evidence)
    save(job/'translations.json',translations);shutil.copyfile(args.evaluator,job/'evaluate.py')
    bundle=job/'remote-source.tar.gz'
    with tarfile.open(bundle,'w:gz') as archive:
        archive.add(source,arcname='source');archive.add(job/'translations.json',arcname='translations.json')
        for task in tasks:
            archive.add(task['annotation'],arcname='annotations/'+task['id']+'.json')
    config={'job':str(job),'host':args.host,'port':args.port,'socket':str(args.socket),
        'remote_base':args.remote_base,'remote_root':args.remote_root,
        'remote_python':args.remote_base+'/.venv/bin/python','python':sys.executable,
        'node':str(args.node),'key_launcher':str(args.key_launcher),
        'pythonpath':os.pathsep.join([str(source/'piperx-cli/src'),str(source/'piper-lab/src')]),
        'source_bundle_sha256':digest(bundle),'created_at':datetime.now(timezone.utc).isoformat(),
        'model':'deepseek-flash','reasoning_efforts':['low','high'],'expected_development':6,
        'expected_formal':54,'fees_capped':False,'retry_on_anomaly':False,
        'physical_execution_budget':{'sdk_calls':2000,'new_call_start_s':2700,'speed_percent':10},
        'scope':'Known complex videos; translated original task assets; shared pose oracle; contact physics, not hardware'}
    save(job/'launch-config.json',config)
    hashes={str(p.relative_to(job)):digest(p) for p in job.rglob('*') if p.is_file()}
    save(job/'frozen-hashes.json',hashes)
    save(job/'state.json',{'status':'prepared','at':datetime.now(timezone.utc).isoformat()})
    if args.submit:
        with (job/'supervisor.log').open('a') as log:
            proc=subprocess.Popen([sys.executable,str(source/'tools'/TOOLS[0]),'supervise','--job',str(job)],
                cwd=job,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
                env={k:v for k,v in os.environ.items() if k!='DEEPSEEK_API_KEY'})
        save(job/'submission.json',{'supervisor_pid':proc.pid,'submitted_at':datetime.now(timezone.utc).isoformat()})
    return config


def deploy(config,job,log):
    base=config['remote_base'];remote=config['remote_root'];python=config['remote_python']
    # Never overwrite a prior remote checkout or a user's existing environment.
    subprocess.run([*ssh(config),shlex.join([python,'-c',
        'from pathlib import Path;Path('+repr(remote)+').mkdir(parents=True,exist_ok=False)'])],
        stdout=log,stderr=log,check=True,timeout=60)
    subprocess.run(['scp','-P',str(config['port']),'-o','ControlPath='+config['socket'],
        '-o','BatchMode=yes','-o','StrictHostKeyChecking=yes',str(job/'remote-source.tar.gz'),
        config['host']+':'+remote+'/source.tar.gz'],stdout=log,stderr=log,check=True,timeout=180)
    script='''from pathlib import Path
import json,tarfile,shutil,sys,hashlib
root=Path(REMOTE);bundle=root/'source.tar.gz'
if hashlib.sha256(bundle.read_bytes()).hexdigest()!=EXPECTED:raise RuntimeError('Remote source transfer hash mismatch')
with tarfile.open(bundle) as archive:archive.extractall(root,filter='data')
shutil.copytree(Path(BASE)/'longvil-assets',root/'longvil-assets')
sys.path[:0]=[str(root/'source/tools'),str(root/'source/piperx-cli/src'),str(root/'source/piper-lab/src')]
from longvil_physics_scene import build_scene,complete_scene_metadata
translations=json.loads((root/'translations.json').read_text())
for ident,translation in translations.items():
 annotation=json.loads((root/'annotations'/(ident+'.json')).read_text())
 output=root/'scenes'/(ident+'.xml')
 build_scene(annotation,root/'source/piperx-cli/assets/mujoco/scene.xml',root/'longvil-assets',output,translation_m=translation)
 meta=complete_scene_metadata(output)
 print(json.dumps({'compiled':ident,'scene_sha256':meta['scene_sha256']}),flush=True)
with tarfile.open(root/'scene-metadata.tar.gz','w:gz') as archive:
 for path in (root/'scenes').glob('*.json'):archive.add(path,arcname=path.name)
'''.replace('REMOTE',repr(remote)).replace('EXPECTED',repr(config['source_bundle_sha256'])).replace('BASE',repr(base))
    subprocess.run([*ssh(config),shlex.join([python,'-'])],input=script,text=True,
                   stdout=log,stderr=log,check=True,timeout=300)
    subprocess.run(['scp','-P',str(config['port']),'-o','ControlPath='+config['socket'],
        '-o','BatchMode=yes','-o','StrictHostKeyChecking=yes',config['host']+':'+remote+'/scene-metadata.tar.gz',
        str(job/'scene-metadata.tar.gz')],stdout=log,stderr=log,check=True,timeout=120)
    with tarfile.open(job/'scene-metadata.tar.gz') as archive:archive.extractall(job/'scene-metadata',filter='data')


def supervise(job):
    job=job.resolve();awake=None;state='error';details={}
    def status(value,**kwargs):
        save(job/'state.json',{'status':value,'at':datetime.now(timezone.utc).isoformat(),**kwargs})
    try:
        config=json.loads((job/'launch-config.json').read_text())
        for name,expected in json.loads((job/'frozen-hashes.json').read_text()).items():
            if digest(job/name)!=expected:raise RuntimeError('Frozen job input changed: '+name)
        if sys.platform=='darwin':
            (job/'power-at-start.txt').write_text(subprocess.check_output(['/usr/bin/pmset','-g','batt'],text=True))
            awake=subprocess.Popen(['/usr/bin/caffeinate','-i','-s','-w',str(os.getpid())],stdin=subprocess.DEVNULL,
                                   stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        source=job/'source';env={k:v for k,v in os.environ.items() if k!='DEEPSEEK_API_KEY'}
        env['PYTHONPATH']=config['pythonpath'];python=config['python'];remote=config['remote_root']
        common=['--host',config['host'],'--port',str(config['port']),'--socket',config['socket']]
        with (job/'run.log').open('a') as log:
            status('deploying');deploy(config,job,log)
            manifest=json.loads((job/'manifest.json').read_text());tasks=[*manifest['tasks'],manifest['development_task']]
            preflights=[];references={}
            for task in tasks:
                ident=task['id'];folder=job/'preflight'/ident
                status('physics_preflight',task=ident,completed=len(preflights),expected=10)
                command=[python,str(source/'tools/validate_longvil_physics.py'),*common,
                    '--python',config['remote_python'],'--helper',remote+'/source/tools/longvil_physics_host.py',
                    '--profile',remote+'/sessions/preflight-'+ident,'--scene',remote+'/scenes/'+ident+'.xml',
                    '--scene-metadata',str(job/'scene-metadata'/(ident+'.json')),
                    '--annotation',task['annotation'],'--output',str(folder)]
                # PYTHONPATH is also supplied to remote commands by the validator.
                subprocess.run(command,env=env,stdout=log,stderr=log,check=True,timeout=3000)
                report=json.loads((folder/'report.json').read_text())
                row={'task':ident,'success':report.get('physics_score',{}).get('success') is True,
                     'sdk_calls':report.get('sdk_calls'),'cleanup':report.get('cleanup'),'report':str(folder/'report.json')}
                preflights.append(row);save(job/'preflight-summary.json',{'rows':preflights,'complete':len(preflights)==10})
                if report.get('cleanup',{}).get('runtime_remaining') is not False or report.get('cleanup',{}).get('recording_error'):
                    raise RuntimeError('Preflight resources not clean; no next task or paid model request')
                frame=json.loads((folder/'frame-initial-metadata.json').read_text())['frames'][0]
                image=folder/'frame-initial.jpg';meta=json.loads((job/'scene-metadata'/(ident+'.json')).read_text())
                references[ident]={'image_path':str(image),'sha256':digest(image),'scene_sha256':meta['scene_sha256'],
                    'sim_time_s':frame['sim_time_s'],'source_timestamp_s':frame['source_stamp_s'],'is_live':False}
            if not all(row['success'] for row in preflights):
                state='physics_preflight_failed';details={'passed':sum(r['success'] for r in preflights),'expected':10}
                return
            save(job/'sim-references.json',references)
            save(job/'scene-evidence-hashes.json',{str(p.relative_to(job)):digest(p)
                for folder in ('scene-metadata','preflight') for p in (job/folder).rglob('*') if p.is_file()})
            base=[config['node'],config['key_launcher'],python,str(source/'tools/run_longvil_physics.py'),
                *common,'--manifest',str(job/'manifest.json'),'--evidence',str(job/'evidence.json'),
                '--sim-references',str(job/'sim-references.json'),'--scene-metadata',str(job/'scene-metadata'),
                '--repo',str(source),'--evaluator',str(job/'evaluate.py'),'--budget-ledger',str(job/'usage-ledger.jsonl'),
                '--remote-root',remote,'--remote-python',config['remote_python'],
                '--remote-helper',remote+'/source/tools/longvil_physics_host.py']
            for phase in ('development','formal'):
                status('model_'+phase,expected=6 if phase=='development' else 54)
                command=[*base,'--output',str(job/phase)]
                if phase=='development':command.append('--development')
                subprocess.run(command,env=env,cwd=source,stdout=log,stderr=log,check=True)
                summary=json.loads((job/phase/'summary.json').read_text())
                if not summary['complete']:
                    state='model_'+phase+'_paused';return
                if phase=='development':
                    gates=json.loads((job/phase/'development-gates.json').read_text())
                    if not all(gates.get(k) is True for k in ('complete','plan_gate_passed','interface_gate_passed','resource_gate_passed')):
                        state='development_gate_failed';details={'gates':gates};return
            state='complete'
    except Exception as exc:details={'error_type':type(exc).__name__,'error':str(exc)[:1500]}
    finally:
        if awake:awake.terminate();awake.wait(timeout=10)
        status(state,**details)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='operation',required=True)
    prep=sub.add_parser('prepare')
    for name in ('repo','data','job','scene-metadata','evaluator','node','key-launcher','socket'):
        prep.add_argument('--'+name,type=Path,required=True)
    for name in ('host','remote-base','remote-root'):prep.add_argument('--'+name,required=True)
    prep.add_argument('--port',type=int,default=54979);prep.add_argument('--submit',action='store_true')
    run=sub.add_parser('supervise');run.add_argument('--job',type=Path,required=True)
    args=parser.parse_args()
    if args.operation=='prepare':print(json.dumps(prepare(args),ensure_ascii=False,indent=2))
    else:supervise(args.job)
