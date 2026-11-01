"""Build the independent simulation kit from verified public source and pinned dependencies."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

WHEEL_SHA256 = 'c801f6d312eb6eee41feb9d5ab3d0f3306857f6960d4933e9a401091a18c6e41'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--wheel', type=Path, required=True)
    p.add_argument('--rpent-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--source-commit', required=True)
    args = p.parse_args()
    if len(args.source_commit) != 40 or any(c not in '0123456789abcdef' for c in args.source_commit):
        p.error('source-commit must be the exact Git commit')
    if sha(args.wheel) != WHEEL_SHA256:
        p.error('Expected the published, unchanged Piper middleware 0.7.2 wheel')
    source = Path(__file__).resolve().parents[1] / 'baselines/rvicl_piperx'
    manifest = json.loads((source / 'ADAPTER.sha256.json').read_text())
    lock = json.loads((source / 'BASELINE.lock.json').read_text())['rpent']
    for name, digest in manifest['files'].items():
        if sha(source / name) != digest:
            raise ValueError('Adapter file differs from manifest: ' + name)
    for name, digest in lock['files'].items():
        if sha(args.rpent_root / name) != digest:
            raise ValueError('RPent file differs from lock: ' + name)
    root = args.output / 'rpent-piperx-1.6'
    root.mkdir(parents=True, exist_ok=False)
    for name in list(manifest['files']) + ['ADAPTER.sha256.json']:
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / name, target)
    shutil.copyfile(source / 'README.zh.md', root / 'README.md')
    for name in list(lock['files']) + ['LICENSE']:
        target = root / 'deps/RPent' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(args.rpent_root / name, target)
    (root / 'wheels').mkdir()
    shutil.copyfile(args.wheel, root / 'wheels' / args.wheel.name)
    (root / 'BUILD-INFO.json').write_text(json.dumps({
        'source_commit': args.source_commit, 'adapter_revision': manifest['revision'],
        'rpent_commit': lock['commit'], 'rpent_python_files': len(lock['files']),
        'wheel_sha256': WHEEL_SHA256,
        'installation': 'Linux/macOS/WSL2; downloads Python dependencies and example assets; not an offline kit',
        'task_success_claim': False, 'hardware_acceptance': False,
    }, indent=2) + '\n')
    files = sorted(f for f in root.rglob('*') if f.is_file())
    (root / 'PACKAGE.sha256.json').write_text(json.dumps({
        str(f.relative_to(root)): sha(f) for f in files}, indent=2) + '\n')
    archive = args.output / (root.name + '.zip')
    if archive.exists():
        raise ValueError('Existing archive retained; use a new output directory')
    with zipfile.ZipFile(archive, 'x', zipfile.ZIP_DEFLATED) as z:
        for f in sorted(root.rglob('*')):
            if f.is_file():
                z.write(f, str(f.relative_to(args.output)))
    checksum = args.output / 'SHA256SUMS.txt'
    checksum.write_text(sha(archive) + '  ' + archive.name + '\n')
    print(json.dumps({'archive': str(archive), 'sha256': sha(archive), 'files': len(files) + 1}))


if __name__ == '__main__':
    main()
