"""Nine full native tasks, baseline then frozen video-only experience.

Uses existing physics, CLI and evaluator unchanged. Private annotations never
enter the provider workspace. Infrastructure failures pause; scored failures stay.
"""
import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tarfile
from types import SimpleNamespace


def save(path, value):
    path = Path(path)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temp.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def transport(c):
    if c['host'] == 'local':
        return ['bash', '-lc']
    return ['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes',
            '-o', 'ConnectTimeout=15', '-p', str(c['port']), c['host']]


def prepare(args):
    job = args.job.resolve()
    source = job/'source'
    if source.exists():
        raise ValueError('Never overwrite frozen source')
    job.mkdir(exist_ok=True, parents=True)
    repo = args.repo.resolve()
    reference = args.reference.resolve()
    for part in ('piperx-cli/src', 'piper-lab/src', 'tools'):
        shutil.copytree(repo/part, source/part,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    sys.path.insert(0, str(source/'tools'))
    from video_experience import ensure_disjoint
    original = json.loads((reference/'manifest.json').read_text())
    packets = json.loads((reference/'evidence.json').read_text())
    tasks = []
    evidence = {}
    old = {r['task']: r['success'] for r in
           json.loads((reference/'preflight-summary.json').read_text())['rows']}
    for t in original['tasks']:
        if t['review_status'] != 'approved' or t['videos']['complex']['review_status'] != 'approved':
            raise ValueError('Unreviewed task')
        ident = t['id']
        private = job/'private'/ident
        private.mkdir(parents=True)
        if digest(t['annotation']) != t['sha256']:
            raise ValueError('Changed reference')
        shutil.copy2(t['annotation'], private/'reference.json')
        packet = packets[ident]
        for kind in ('initial', 'supplement'):
            for i, frame in enumerate(packet[kind]['frames']):
                if digest(frame['image_path']) != frame['image_sha256']:
                    raise ValueError('Changed demonstration')
                dest = private/f'{kind}-{i:02d}.jpg'
                shutil.copy2(frame['image_path'], dest)
                frame['image_path'] = str(dest)
        evidence[ident] = packet
        meta = json.loads((reference/'scene-metadata'/f'{ident}.json').read_text())
        tasks.append({'task_id': ident, 'source_sha256': packet['source_sha256'],
                      'annotation': str(private/'reference.json'),
                      'objects': json.loads((private/'reference.json').read_text())['object_list'],
                      'scene_sha256': meta['scene_sha256'], 'old_reference_pass': old[ident]})
    if len(tasks) != 9:
        raise ValueError('Expected nine fixed test tasks')
    # Preselect three disjoint training videos without using any execution result.
    pool = json.loads((args.video_pool/'manifest.json').read_text())['tasks']
    test_ids = {t['task_id'] for t in tasks}
    selected = []
    for level, category in [('1', 'blockmanipulation'), ('2', 'vegetablesorting'), ('3', 'tabletopcleanup')]:
        eligible = sorted([t for t in pool if t['id'] not in test_ids and
                           str(t['level']) == level and t['category'] == category and
                           t['review_status'] == 'approved' and
                           t['videos']['complex']['review_status'] == 'approved'], key=lambda t: t['id'])
        if not eligible:
            raise ValueError('No disjoint training video in stratum')
        selected.append(eligible[0])
    import av
    import numpy as np
    training = []
    for i, t in enumerate(selected):
        video = t['videos']['complex']
        if digest(video['path']) != video['sha256']:
            raise ValueError('Changed learning video')
        with av.open(video['path']) as stream:
            times = [float(f.time) for f in stream.decode(video=0)]
        if len(times) < 2:
            raise ValueError('Insufficient decoded frames')
        indices = {'initial': np.linspace(0, len(times)-1, min(12, len(times))).round().astype(int).tolist(),
                   'supplement': [int((j+.5)*len(times)/12) for j in range(12)]}
        wanted = set(indices['initial'] + indices['supplement'])
        folder = job/'training-frames'/f'video-{i:02d}'
        folder.mkdir(parents=True)
        frames = {}
        with av.open(video['path']) as stream:
            for n, f in enumerate(stream.decode(video=0)):
                if n in wanted:
                    dest = folder/f'f{n:06d}.jpg'
                    f.to_image().save(dest)
                    frames[n] = {'evidence_id': f'v{i:02d}/f{n:06d}', 'timestamp_s': float(f.time),
                                 'image_path': str(dest), 'image_sha256': digest(dest)}
        training.append({'task_id': t['id'], 'source_sha256': video['sha256'],
                         **{k: [frames[n] for n in ns] for k, ns in indices.items()}})
    ensure_disjoint(training, tasks)
    config = {'model': 'gemini-4-argon-high', 'host': args.host, 'port': args.port,
              'remote_root': '/home/featurize/.local/share/'+job.name,
              'remote_scenes': '/home/featurize/.local/share/piper-physics-sim-20261011/physics-job-01/scenes',
              'remote_python': '/home/featurize/.local/share/piper-physics-sim-20261011/.venv/bin/python',
              'python': sys.executable, 'max_rounds': 80, 'max_tools': 120, 'max_seconds': None,
              'budget_revision': 'unbounded-task-time-v2',
              'camera': 'mujoco-overview', 'provider_turn_window': 2,
              'task_scope': 'Full nine complex tasks, original initial state; no partial-placement fixture',
              'design': 'baseline then video experience, one run per task/condition; order confounded with time',
              'training': 'Three disjoint videos, two evidence passes, no simulation or hardware calls',
              'limitations': 'Known task suite; not pristine unseen-task benchmark. Four old reference rollouts failed.',
              'retry_on_anomaly': False}
    save(job/'config.json', config)
    save(job/'tasks.json', tasks)
    save(job/'training.json', training)
    save(job/'evidence.json', evidence)
    save(job/'schedule.json', [{'condition': c, 'task': t['task_id']}
                              for c in ('baseline', 'video-experience') for t in tasks])
    with tarfile.open(job/'source.tar.gz', 'w:gz') as archive:
        archive.add(source, arcname='source')
    (job/'workspace.patch').write_bytes(subprocess.check_output(['git', '-C', str(repo), 'diff']))
    save(job/'frozen-hashes.json', {str(p.relative_to(job)): digest(p)
         for p in [*source.rglob('*'), *list((job/'private').rglob('*')),
                   *list((job/'training-frames').rglob('*')),
                   *[job/n for n in ('config.json', 'tasks.json', 'training.json', 'evidence.json',
                                     'schedule.json', 'source.tar.gz')]] if p.is_file()})
    save(job/'state.json', {'status': 'prepared', 'expected_runs': 18})


def verify(job):
    for name, expected in json.loads((job/'frozen-hashes.json').read_text()).items():
        if digest(job/name) != expected:
            raise ValueError('Frozen artifact changed: '+name)


async def run_case(job, condition, task_id, probe=False):
    c = json.loads((job/'config.json').read_text())
    task = next(t for t in json.loads((job/'tasks.json').read_text()) if t['task_id'] == task_id)
    source = job/'source'
    for part in ('tools', 'piperx-cli/src', 'piper-lab/src'):
        sys.path.insert(0, str(source/part))
    from agy_online_decider import AgyDecider
    import run_longvil_online as online
    from video_experience import render_library
    root = job/('probes' if probe else condition)/task_id
    root.mkdir(parents=True, exist_ok=False)
    d = None
    if not probe:
        d = AgyDecider(root/'decision', model=c['model'], exploration=True,
                       provider_turn_window=c['provider_turn_window'], executable=c.get('agy_executable', 'agy'))
        if condition == 'video-experience':
            packet = render_library(job/'learning/library.json',
                                    (job/'learning/library.sha256').read_text().strip())
            original_prepare = d.prepare
            def with_experience(payload):
                import copy
                enriched = copy.deepcopy(payload)
                enriched['messages'].insert(0, {'role': 'system', 'content': packet})
                return original_prepare(enriched)
            d.prepare = with_experience
    kw = dict(output=root/'execution', annotation=Path(task['annotation']), evidence=job/'evidence.json',
              socket=job/'ssh-control', host=c['host'], port=c['port'], remote_root=c['remote_root'],
              remote_python=c['remote_python'], scene=c['remote_scenes']+'/'+task_id+'.xml',
              task=task_id, object_list=json.dumps(task['objects']), effort='high', plan='on', skill='on',
              decision_policy='standard', call_completion_cap=65536, interface_smoke=False,
              contact_smoke=False, native_probe=probe, max_rounds=c['max_rounds'], max_tools=c['max_tools'],
              max_seconds=c['max_seconds'], completion_budget=None, exploration=True,
              camera_condition=c['camera'], requested_model=c['model'], strict_completion=True,
              plan_envelope=True, task_goal=None, physics_revision=c.get('physics_revision', 'original-impratio1-v1'),
              scene_sha256=task['scene_sha256'],
              budget_revision=c.get('budget_revision', 'time1800-v1'))
    if d:
        kw['model_decider'] = d
    try:
        await online.main(SimpleNamespace(**kw))
    finally:
        if d:
            await d.close()


def healthy(report, probe=False):
    clean = report.get('cleanup', {})
    return (clean.get('runtime_remaining') is False and clean.get('owned_process_alive') is False
            and not report.get('cleanup_error') and not clean.get('recording_error')
            and (report.get('status') == 'native_probe_pass' if probe else
                 report.get('status') != 'paused' and report.get('graded') is True))


def supervise(job):
    verify(job)
    c = json.loads((job/'config.json').read_text())
    tasks = json.loads((job/'tasks.json').read_text())
    schedule = json.loads((job/'schedule.json').read_text())
    runner = job/'source/tools/run_argon_video_study.py'
    awake = subprocess.Popen(['/usr/bin/caffeinate', '-i', '-s', '-w', str(os.getpid())]) if sys.platform == 'darwin' else None
    try:
        with (job/'supervisor.log').open('a') as log:
            def launch(argv):
                subprocess.run([c['python'], str(runner), *argv, '--job', str(job)],
                               stdin=subprocess.DEVNULL, stdout=log, stderr=log, check=True)
            if c['host'] == 'local' and not (job/'deployed.json').exists():
                raise RuntimeError('Stage local source and deployed.json before starting this job')
            if not (job/'deployed.json').exists():
                remote = c['remote_root']
                code = 'from pathlib import Path;Path('+repr(remote)+').mkdir(exist_ok=False)'
                subprocess.run([*transport(c), shlex.join([c['remote_python'], '-c', code])], check=True, stdout=log, stderr=log)
                subprocess.run(['scp', '-P', str(c['port']), '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes',
                                str(job/'source.tar.gz'), c['host']+':'+remote+'/source.tar.gz'], check=True, stdout=log, stderr=log)
                script = ('import tarfile,hashlib;from pathlib import Path;r=Path('+repr(remote)+');p=r/"source.tar.gz";'
                          'assert hashlib.sha256(p.read_bytes()).hexdigest()=='+repr(digest(job/'source.tar.gz'))+';'
                          'tarfile.open(p).extractall(r,filter="data")')
                subprocess.run([*transport(c), shlex.join([c['remote_python'], '-c', script])], check=True, stdout=log, stderr=log)
                save(job/'deployed.json', {'remote_root': remote})
            # Validate frozen shared scene XML and every declared asset before execution.
            checks = [(c['remote_scenes']+'/'+t['task_id']+'.xml', t['scene_sha256']) for t in tasks]
            script = ('import json,hashlib;from pathlib import Path;checks='+repr(checks)+'\n'
                      'for p,h in checks:\n'
                      ' assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p\n'
                      ' meta=json.loads(Path(p).with_suffix(".json").read_text())\n'
                      ' for asset,expected in meta["asset_sha256s"].items():\n'
                      '  assert hashlib.sha256(Path(asset).read_bytes()).hexdigest()==expected,asset\n')
            subprocess.run([*transport(c), shlex.join([c['remote_python'], '-c', script])], check=True, stdout=log, stderr=log)
            for t in tasks:
                p = job/'probes'/t['task_id']/'execution/report.json'
                if not p.exists():
                    save(job/'state.json', {'status': 'native_preflight', 'task': t['task_id']})
                    launch(['case', '--condition', 'baseline', '--task', t['task_id'], '--probe'])
                if not healthy(json.loads(p.read_text()), True):
                    raise RuntimeError('Native preflight failed: '+t['task_id'])
            rows = []
            for item in schedule:
                condition, task = item['condition'], item['task']
                if condition == 'video-experience' and not (job/'learning/library.json').exists():
                    save(job/'state.json', {'status': 'video_only_learning', 'completed': len(rows)})
                    launch(['learn'])
                p = job/condition/task/'execution/report.json'
                if not p.exists():
                    save(job/'state.json', {'status': 'running', **item, 'completed': len(rows), 'expected': len(schedule)})
                    launch(['case', '--condition', condition, '--task', task])
                report = json.loads(p.read_text())
                rows.append({**item, 'report': str(p), **report})
                save(job/'results.json', rows)
                if not healthy(report):
                    raise RuntimeError('Infrastructure/scoring/cleanup requires review: '+condition+'/'+task)
            save(job/'state.json', {'status': 'complete', 'completed': len(rows), 'expected': len(schedule)})
    except Exception as exc:
        save(job/'state.json', {'status': 'paused', 'reason': str(exc), 'automatic_retry': False})
        raise
    finally:
        if awake:
            awake.terminate()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('operation', choices=['prepare', 'supervise', 'case', 'learn'])
    p.add_argument('--job', type=Path, required=True)
    p.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument('--reference', type=Path)
    p.add_argument('--video-pool', type=Path)
    p.add_argument('--host', default='featurize@wsl-ros-jazzy')
    p.add_argument('--port', type=int, default=2222)
    p.add_argument('--condition', choices=['baseline', 'video-experience'])
    p.add_argument('--task')
    p.add_argument('--probe', action='store_true')
    a = p.parse_args()
    if a.operation == 'prepare':
        prepare(a)
    elif a.operation == 'supervise':
        supervise(a.job)
    elif a.operation == 'case':
        verify(a.job)
        asyncio.run(run_case(a.job, a.condition, a.task, a.probe))
    else:
        verify(a.job)
        from video_experience import learn
        c = json.loads((a.job/'config.json').read_text())
        asyncio.run(learn(a.job/'learning', json.loads((a.job/'training.json').read_text()),
                          json.loads((a.job/'tasks.json').read_text()), c['model']))


if __name__ == '__main__':
    main()
