"""Freeze a reviewed LongVIL v2 job; submit only after development gates pass."""
import argparse
import json
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime,timezone
from longvil_budget import Ledger
from run_longvil_v2 import load_manifest,build_schedule


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_job(path,data):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(data,indent=2));temp.replace(path)


def preflight(root,development):
    gates=json.loads((development/'development-gates.json').read_text())
    supplement=development/'verification-gates.json'
    if supplement.exists():
        verification=json.loads(supplement.read_text())
        checks=verification.get('checks',[])
        if len(checks)!=6 or not all(c.get('verified') for c in checks):raise ValueError('Plan recheck evidence incomplete')
        gates.update({k:verification[k] for k in ('plan_gate_passed','projected_total_cny','cost_gate_passed')})
        gates['verification_path']=str(supplement)
    required=('complete','syntax_gate_passed','vision_gate_passed','plan_gate_passed','cost_gate_passed')
    failed=[k for k in required if gates.get(k) is not True]
    if failed:raise ValueError('Development gates not passed: '+', '.join(failed))
    ledger=Ledger(root/'usage-ledger.jsonl');ledger.check()
    if gates['projected_total_cny']>72:raise ValueError('Projection does not fit the spending checkpoint')
    checklist=json.loads((root/'user-checklist.json').read_text())
    if checklist.get('confirmed') is not True:raise ValueError('User checklist not confirmed')
    tasks=load_manifest(root/'formal-manifest.json',False)
    if not {(c['id'],c['variant']) for c in checklist['items']} <= {(t['id'],v) for t in tasks for v in ('clean','complex')}:
        raise ValueError('Checklist task changed')
    gates.update(formal_submission_allowed=True,remaining_gate=None)
    return gates,tasks


def build(args):
    gates,tasks=preflight(args.root,args.development)
    args.job.mkdir(parents=True,exist_ok=False)
    source=args.job/'source';source.mkdir()
    for part in ('piperx-cli/src','piper-lab/src','tools'):
        shutil.copytree(args.repo/part,source/part,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    # Videos/GT use independent copies so editing the source cannot alter a running job.
    for task in tasks:
        dest=args.job/'private-data'/task['id'];dest.mkdir(parents=True)
        annotation=dest/'reference.json';shutil.copyfile(task['annotation'],annotation);task['annotation']=str(annotation)
        for variant,video in task['videos'].items():
            path=dest/(variant+'.mp4');shutil.copyfile(video['path'],path);video['path']=str(path)
    manifest=json.loads((args.root/'formal-manifest.json').read_text());manifest['tasks']=tasks
    manifest['formal_submission_allowed']=True
    (args.job/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    evaluator=args.job/'evaluate.py';shutil.copyfile(args.evaluator,evaluator)
    detector=args.job/'detector.onnx';shutil.copyfile(args.detector,detector)
    schedule=build_schedule(tasks);(args.job/'schedule.json').write_text(json.dumps(schedule,indent=2))
    (args.job/'development-gates.json').write_text(json.dumps(gates,indent=2))
    for name in ('codex-review-ledger.jsonl','user-checklist.json','ground-truth-selfcheck.json','prompt-revision.json'):
        shutil.copyfile(args.root/name,args.job/name)
    if (args.development/'verification-gates.json').exists():shutil.copyfile(args.development/'verification-gates.json',args.job/'verification-gates.json')
    guide=args.repo/'docs/implementation/longvil-video-v2-20261010.md'
    if guide.exists():shutil.copyfile(guide,args.job/'README.zh.md')
    hashes={str(p.relative_to(args.job)):digest(p) for p in args.job.rglob('*') if p.is_file()}
    (args.job/'frozen-hashes.json').write_text(json.dumps(hashes,indent=2))
    # Credentials stay in the existing private launcher and only the API child environment.
    command=[str(args.node),str(args.key_launcher),sys.executable,str(source/'tools/run_longvil_v2.py'),
             '--manifest',str(args.job/'manifest.json'),'--output',str(args.job/'results'),
             '--repo',str(source),'--cli',str(args.cli),'--evaluator',str(evaluator),
             '--video-python',str(args.video_python),'--detector',str(detector),
             '--budget-ledger',str(args.root/'usage-ledger.jsonl'),'--vision-cache',str(args.root/'vision-cache')]
    config={'created_at':datetime.now(timezone.utc).isoformat(),'command':command,'cwd':str(source),
            'pythonpath':os.pathsep.join([str(source/'piperx-cli/src'),str(source/'piper-lab/src')]),
            'schedule_count':len(schedule),'development_path':str(args.development),
            'state':'prepared','scope':'offline only; no robot tools, automatic retries, or monitoring'}
    save_job(args.job/'job.json',config)
    if args.submit:
        # Supervisor waits once for the child and writes an exit record; it never retries.
        supervisor=args.job/'supervisor.py'
        supervisor.write_text("""import json,os,subprocess,sys,hashlib
from pathlib import Path
from datetime import datetime,timezone
root=Path(sys.argv[1]);result={}
try:
 d=json.loads((root/'job.json').read_text())
 for name,expected in json.loads((root/'frozen-hashes.json').read_text()).items():
  if hashlib.sha256((root/name).read_bytes()).hexdigest()!=expected:raise RuntimeError('Frozen job changed: '+name)
 env={k:v for k,v in os.environ.items() if k!='DEEPSEEK_API_KEY'};env['PYTHONPATH']=d['pythonpath']
 with (root/'run.log').open('a') as log:
  p=subprocess.Popen(d['command'],cwd=d['cwd'],env=env,stdout=log,stderr=subprocess.STDOUT)
  code=p.wait()
 result['exit_code']=code
 if (root/'results/summary.json').exists():result['complete']=json.loads((root/'results/summary.json').read_text())['complete']
except Exception as error:result.update(exit_code=1,startup_error=str(error))
result['ended_at']=datetime.now(timezone.utc).isoformat()
(root/'exit.json').write_text(json.dumps(result,indent=2))
""")
        with (args.job/'supervisor.log').open('a') as log:
            proc=subprocess.Popen([sys.executable,str(supervisor),str(args.job)],cwd=args.job,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        config.update(state='submitted',supervisor_pid=proc.pid,submitted_at=datetime.now(timezone.utc).isoformat())
        save_job(args.job/'job.json',config)
    return config


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','development','repo','job','evaluator','detector','node','key-launcher','cli','video-python'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--submit',action='store_true');args=p.parse_args()
    print(json.dumps(build(args),indent=2))
