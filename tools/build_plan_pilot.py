"""Freeze the complex-only pilot and submit only after protocol and cost gates."""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime, timezone

from longvil_budget import Ledger


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path=Path(path);temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False));temp.replace(path)


def preflight(root, development, forecast_authorization=None):
    ledger=Ledger(Path(root)/'usage-ledger.jsonl',limit=15,stop_at=12);ledger.check()
    if development is None:
        if forecast_authorization:raise ValueError('Forecast exception requires a completed development job')
        return None
    gate_path=Path(development)/'results/development-gates.json'
    gates=json.loads(gate_path.read_text())
    for field in ['complete','plan_gate_passed','interface_gate_passed','resource_gate_passed']:
        if gates.get(field) is not True:raise ValueError('Development gate failed: '+field)
    ceiling=12
    if forecast_authorization:
        authorization=json.loads(Path(forecast_authorization).read_text())
        expected={'approved':True,'scope':'forecast-submission-only','projection_ceiling_cny':12.5,
                  'actual_stop_at_cny':12,'limit_cny':15,
                  'development_job':str(Path(development).resolve()),'development_gates_sha256':digest(gate_path)}
        if any(authorization.get(k)!=v for k,v in expected.items()):
            raise ValueError('Forecast authorization does not match the scoped exception or development snapshot')
        ceiling=12.5
        gates=dict(gates,forecast_authorization=authorization)
    elif gates.get('cost_gate_passed') is not True:
        raise ValueError('Development gate failed: cost_gate_passed')
    if not math.isfinite(gates['projected_total_cny']) or gates['projected_total_cny']>ceiling:
        raise ValueError(f'Projection exceeds {ceiling} CNY checkpoint or is unavailable')
    gates=dict(gates,forecast_submission_ceiling_cny=ceiling,forecast_gate_passed=True)
    summary_path=Path(development)/'results/summary.json'
    if summary_path.exists():
        summary=json.loads(summary_path.read_text());rows=summary.get('rows',[])
        if len(rows)!=6 or not summary.get('complete'):raise ValueError('Development records incomplete')
        attempts=errors=0
        for row in rows:
            path=Path(development)/'results'/row['session_id']/'events.jsonl'
            for line in path.read_text().splitlines():
                event=json.loads(line)
                if event.get('kind')!='tool':continue
                attempts+=1;error=event.get('result',{}).get('error')
                if error and not str(error.get('message','')).startswith(('Wrong-phase','Early submission')):
                    errors+=1
        rate=errors/max(1,attempts)
        if rate>.1:raise ValueError('Independently audited interface error rate exceeds 10%')
        gates=dict(gates,verified_interface_error_rate=rate,verified_tool_attempts=attempts)
    return gates


def build(args):
    authorization=getattr(args,'forecast_authorization',None)
    gates=preflight(args.root,None if args.development else args.development_job,authorization)
    if not args.development and args.development_job is None:
        raise ValueError('Pilot requires a completed development job')
    if gates:
        old=json.loads((args.development_job/'frozen-hashes.json').read_text())
        for name,h in old.items():
            if digest(args.development_job/name)!=h:raise ValueError('Development snapshot changed')
        # No unreviewed source/prompt edits between acceptance and pilot.
        for name in ['plan_protocol.py','run_plan_enforcement_pilot.py','run_longvil_v2.py','longvil_metrics.py','longvil_budget.py']:
            if digest(args.repo/'tools'/name)!=digest(args.development_job/'source/tools'/name):
                raise ValueError('Development-accepted source changed: '+name)
        for package in ['piperx-cli/src','piper-lab/src']:
            live={str(p.relative_to(args.repo)):digest(p) for p in (args.repo/package).rglob('*')
                  if p.is_file() and p.suffix in ('.py','.md')}
            accepted={str(p.relative_to(args.development_job/'source')):digest(p)
                      for p in (args.development_job/'source'/package).rglob('*')
                      if p.is_file() and p.suffix in ('.py','.md')}
            if live!=accepted:raise ValueError('Development-accepted package changed: '+package)
    args.job.mkdir(parents=True,exist_ok=False)
    source=args.job/'source';source.mkdir()
    for part in ['piperx-cli/src','piper-lab/src']:
        shutil.copytree(args.repo/part,source/part,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    (source/'tools').mkdir()
    for name in ['plan_protocol.py','run_plan_enforcement_pilot.py','run_longvil_v2.py','longvil_metrics.py','longvil_budget.py']:
        shutil.copyfile(args.repo/'tools'/name,source/'tools'/name)
    manifest=copy.deepcopy(json.loads((args.root/'manifest.json').read_text()))
    evidence=copy.deepcopy(json.loads((args.root/'evidence.json').read_text()))
    for task in [manifest['development_task'],*manifest['tasks']]:
        dest=args.job/'private-data'/task['id'];dest.mkdir(parents=True)
        annotation=dest/'reference.json';shutil.copyfile(task['annotation'],annotation)
        if digest(annotation)!=task['sha256']:raise ValueError('Annotation changed')
        task['annotation']=str(annotation)
        video=dest/'complex.mp4';shutil.copyfile(task['videos']['complex']['path'],video)
        if digest(video)!=task['videos']['complex']['sha256']:raise ValueError('Video changed')
        task['videos']['complex']['path']=str(video)
        entry=evidence[task['id']]
        for group in ['initial','supplement']:
            for i,frame in enumerate(entry[group]['frames']):
                picture=dest/f'{group}-{i:02d}.jpg';shutil.copyfile(frame['image_path'],picture)
                if digest(picture)!=frame['image_sha256']:raise ValueError('Evidence picture changed')
                frame['image_path']=str(picture)
    save(args.job/'manifest.json',manifest);save(args.job/'evidence.json',evidence)
    shutil.copyfile(args.evaluator,args.job/'evaluate.py')
    if gates:save(args.job/'development-gates.json',gates)
    if gates and 'forecast_authorization' in gates:
        save(args.job/'forecast-authorization.json',gates['forecast_authorization'])
    if args.readme:shutil.copyfile(args.readme,args.job/'README.zh.md')
    command=[str(args.node),str(args.key_launcher),sys.executable,str(source/'tools/run_plan_enforcement_pilot.py'),
        '--manifest',str(args.job/'manifest.json'),'--evidence',str(args.job/'evidence.json'),
        '--output',str(args.job/'results'),'--repo',str(source),'--cli',str(args.cli),
        '--evaluator',str(args.job/'evaluate.py'),'--budget-ledger',str(args.root/'usage-ledger.jsonl')]
    if args.development:command.append('--development')
    config={'command':command,'cwd':str(source),'pythonpath':os.pathsep.join([str(source/'piperx-cli/src'),str(source/'piper-lab/src')]),
        'created_at':datetime.now(timezone.utc).isoformat(),'mode':'development' if args.development else 'pilot',
        'expected_episodes':6 if args.development else 54,'state':'prepared','development_job':str(args.development_job) if gates else None,
        'budget':{'limit_cny':15,'stop_at_cny':12,'forecast_submission_ceiling_cny':gates['forecast_submission_ceiling_cny'] if gates else 12},
        'scope':'complex-only offline protocol pilot; no automatic retries or monitoring'}
    save(args.job/'job.json',config)
    save(args.job/'launch-config.json',config)
    hashes={str(p.relative_to(args.job)):digest(p) for p in args.job.rglob('*') if p.is_file() and p.name!='job.json'}
    save(args.job/'frozen-hashes.json',hashes)
    if args.submit:
        supervisor=args.job/'supervisor.py'
        supervisor.write_text('''import hashlib,json,os,subprocess,sys,time,re
from pathlib import Path
from datetime import datetime,timezone
root=Path(sys.argv[1]);result={};awake=None
def now():return datetime.now(timezone.utc).isoformat()
def state(status,**fields):
 temp=root/'state.tmp';temp.write_text(json.dumps(dict(status=status,at=now(),**fields)));temp.replace(root/'state.json')
try:
 config=json.loads((root/'launch-config.json').read_text())
 for name,h in json.loads((root/'frozen-hashes.json').read_text()).items():
  if hashlib.sha256((root/name).read_bytes()).hexdigest()!=h:raise RuntimeError('Frozen file changed: '+name)
 # Operator explicitly requested direct launch; record power without a wait gate.
 if sys.platform=='darwin':
  (root/'power-at-start.txt').write_text(subprocess.check_output(['/usr/bin/pmset','-g','batt'],text=True))
 env={k:v for k,v in os.environ.items() if k!='DEEPSEEK_API_KEY'};env['PYTHONPATH']=config['pythonpath']
 with (root/'run.log').open('a') as log:
  command=list(config['command'])
  if '--resume' in sys.argv[2:]:command.append('--resume')
  child=subprocess.Popen(command,cwd=config['cwd'],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
  if sys.platform=='darwin':awake=subprocess.Popen(['/usr/bin/caffeinate','-i','-s','-w',str(child.pid)],stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
  state('running',child_pid=child.pid,started_at=now())
  result['exit_code']=child.wait()
 if (root/'results/summary.json').exists():result['complete']=json.loads((root/'results/summary.json').read_text()).get('complete',False)
except Exception as exc:result.update(exit_code=1,error=str(exc))
finally:
 if awake:
  try:awake.wait(timeout=5)
  except subprocess.TimeoutExpired:awake.terminate();awake.wait(timeout=5)
 result['ended_at']=now();(root/'exit.json').write_text(json.dumps(result));state('complete' if result.get('complete') else 'stopped',**result)
''')
        with (args.job/'supervisor.log').open('a') as log:
            child=subprocess.Popen([sys.executable,str(supervisor),str(args.job)],cwd=args.job,
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        config.update(state='submitted',supervisor_pid=child.pid,submitted_at=datetime.now(timezone.utc).isoformat())
        save(args.job/'job.json',config)
    return config


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['root','repo','job','evaluator','node','key-launcher','cli']:
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--development-job',type=Path);p.add_argument('--readme',type=Path)
    p.add_argument('--forecast-authorization',type=Path)
    p.add_argument('--development',action='store_true');p.add_argument('--submit',action='store_true')
    print(json.dumps(build(p.parse_args()),ensure_ascii=False,indent=2))
