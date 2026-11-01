#!/usr/bin/env python3
"""Install, check and launch the existing simulation batch from a downloaded kit."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import urllib.request
import uuid
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parent
TASKS = [(63, 315, 'GX010391.MP4'), (64, 320, 'GX010392.MP4'), (71, 355, 'GX010399.MP4')]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def fetch(path, source):
    """Reuse only verified files; failed downloads never become cache entries."""
    if path.exists():
        if digest(path) != source['sha256']:
            raise ValueError('Cached file hash mismatch; retained for inspection: ' + str(path))
        return
    import certifi
    import ssl
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + '.' + uuid.uuid4().hex + '.partial')
    try:
        with urllib.request.urlopen(source['url'], timeout=120,
                context=ssl.create_default_context(cafile=certifi.where())) as response, partial.open('xb') as out:
            shutil.copyfileobj(response, out)
        if digest(partial) != source['sha256']:
            raise ValueError('Downloaded file hash mismatch: ' + path.name)
        try:
            os.link(partial, path)  # Atomic, and never overwrite another preparation.
        except FileExistsError:
            if digest(path) != source['sha256']:
                raise ValueError('Concurrent cache file differs: ' + path.name)
    finally:
        partial.unlink(missing_ok=True)


def prepare(directory):
    from piperx_middleware.mujoco_backend import DEFAULT_ASSET_PATH
    from rvicl_ig.piper_evaluate import validate_scenes
    cache = ROOT / 'var/resources'
    templates = ROOT / 'examples/watchact'
    if directory.exists():
        saved = json.loads((directory / 'PREPARED.json').read_text())
        if saved['root'] != str(ROOT) or any(digest(directory / n) != h for n, h in saved['files'].items()):
            raise ValueError('Prepared files changed or moved; use a fresh --prepared directory')
        resources = json.loads((templates / 'resources.json').read_text())
        for n, source in resources.items():
            if digest(cache / n) != source['sha256']:
                raise ValueError('Prepared resource changed: ' + n)
        return directory / 'manifest.json'
    for name, source in json.loads((templates / 'resources.json').read_text()).items():
        print('Checking resource:', name, flush=True)
        fetch(cache / name, source)
    directory.mkdir(parents=True)
    rows = []
    for number, ident, video in TASKS:
        scene = directory / f'scene-{number}.xml'
        tree = ET.parse(templates / f'{number}.xml')
        for element in tree.iter():
            for key, value in element.attrib.items():
                element.set(key, value.replace('__ROBOT_MESHES__', str(DEFAULT_ASSET_PATH.parent / 'meshes')).replace('__RESOURCES__', str(cache)))
        tree.write(scene, encoding='unicode')
        evaluation = directory / f'evaluation-{number}.json'
        shutil.copyfile(templates / evaluation.name, evaluation)
        rows.append({'job_id': f'imitation_data_{ident}', 'kind': 'model', 'arm_count': 1, 'seed': 0,
            'video': str(cache / 'videos' / video),
            'language_instruction': 'Start executing the instruction from the initial scene of the video: move objects by imitating the sequence of actions shown in the video.',
            'camera_perspective': 'front', 'spatial_reference': 'camera',
            'simulation_asset': str(scene), 'evaluation_file': str(evaluation)})
    (directory / 'manifest.json').write_text(json.dumps(rows, indent=2) + '\n')
    validate_scenes([{'kind': 'model', 'config': {'simulation_asset': row['simulation_asset'], 'seed': 0},
                     'evaluation': row['evaluation_file']} for row in rows])
    (directory / 'PREPARED.json').write_text(json.dumps({'root': str(ROOT), 'files': {
        f.name: digest(f) for f in directory.iterdir() if f.is_file()}}, indent=2) + '\n')
    return directory / 'manifest.json'


def setup(rpent):
    uv = shutil.which('uv')
    if not uv:
        raise ValueError('Install uv first: https://docs.astral.sh/uv/getting-started/installation/')
    for name in ('ffmpeg', 'ffprobe'):
        if not shutil.which(name):
            raise ValueError('Missing ' + name + '; install ffmpeg (macOS: brew install ffmpeg; Ubuntu: sudo apt install ffmpeg)')
    python = ROOT / '.venv/bin/python'
    wheels = list((ROOT / 'wheels').glob('piperx_middleware-0.7.2-*.whl'))
    source = ROOT.parents[1] / 'piperx-cli'
    if len(wheels) == 1:
        package = str(wheels[0]) + '[simulation]'
    elif (source / 'pyproject.toml').exists():
        package = str(source) + '[simulation]'
    else:
        raise ValueError('Missing bundled 0.7.2 wheel; download the runnable release ZIP, not GitHub Source code ZIP')
    if not python.exists():
        subprocess.run([uv, 'venv', str(ROOT / '.venv'), '--python', '3.12'], check=True)
    subprocess.run([uv, 'pip', 'install', '--python', str(python), '-r',
                    str(ROOT / 'requirements.lock.txt'), package], check=True)
    if not rpent.exists():
        commit = json.loads((ROOT / 'BASELINE.lock.json').read_text())['rpent']['commit']
        subprocess.run(['git', 'clone', '--filter=blob:none', 'https://github.com/RLinf/RPent.git', str(rpent)], check=True)
        subprocess.run(['git', '-C', str(rpent), 'checkout', '--detach', commit], check=True)
    env = environment(rpent)
    subprocess.run([uv, 'pip', 'check', '--python', str(python)], check=True, env=env)
    subprocess.run([str(python), '-c', 'from rvicl_ig.piper_cli import verify_sources; import os; verify_sources(os.environ["RPENT_ROOT"]); print("Installed; RPent source verified.")'], check=True, env=env)


def environment(rpent):
    env = dict(os.environ, RPENT_ROOT=str(rpent), PYTHONPATH=str(ROOT) + os.pathsep + str(rpent))
    env['PATH'] = str(ROOT / '.venv/bin') + os.pathsep + env.get('PATH', '')
    if sys.platform == 'linux':
        env.setdefault('MUJOCO_GL', 'egl')
    return env


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['setup', 'smoke', 'prepare', 'run'])
    p.add_argument('--rpent-root', type=Path, default=ROOT / 'deps/RPent')
    p.add_argument('--prepared', type=Path, default=ROOT / 'var/watchact-v5')
    p.add_argument('--output', type=Path)
    p.add_argument('--env-file', type=Path, default=ROOT / '.env.model')
    p.add_argument('--config', type=Path, default=ROOT / 'examples/model-config.json')
    p.add_argument('--dry-run', action='store_true', help='Check prepared scene/scorer bindings; no model requests')
    args = p.parse_args(argv)
    if sys.platform not in ('linux', 'darwin'):
        p.error('Use Linux, macOS or Windows WSL2; native Windows batch locking is not supported')
    rpent = args.rpent_root.resolve()
    if args.command == 'setup':
        setup(rpent)
        return 0
    python = ROOT / '.venv/bin/python'
    if not python.exists():
        p.error('Run python3 quickstart.py setup first')
    # Re-enter this same thin launcher with the isolated installed interpreter.
    if Path(sys.prefix).resolve() != (ROOT / '.venv').resolve():
        return subprocess.call([str(python), str(Path(__file__).resolve()), *(argv or sys.argv[1:])], env=environment(rpent))
    from rvicl_ig.piper_cli import verify_sources
    verify_sources(rpent)
    if args.command == 'prepare':
        print('Prepared:', prepare(args.prepared.resolve()))
        return 0
    output = (args.output or ROOT / ('runs/smoke' if args.command == 'smoke' else 'runs/watchact')).resolve()
    command = [str(python), '-m', 'rvicl_ig.piper_evaluate', '--rpent-root', str(rpent),
               '--config', str(args.config.resolve()), '--output-root', str(output)]
    if args.command == 'smoke':
        command.append('--smoke')
    else:
        manifest = args.prepared.resolve() / 'manifest.json'
        if not manifest.exists():
            p.error('Run python3 quickstart.py prepare first')
        # Verify retained evidence again without downloading or changing it.
        prepare(args.prepared.resolve())
        if not args.dry_run and not args.env_file.is_file():
            p.error('Copy examples/model.env.example to .env.model and configure your model provider')
        command += ['--manifest', str(manifest)]
        if args.env_file.is_file():
            command += ['--env-file', str(args.env_file.resolve())]
    if args.dry_run:
        command.append('--dry-run')
    return subprocess.call(command, env=environment(rpent))


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        raise SystemExit(str(error))
