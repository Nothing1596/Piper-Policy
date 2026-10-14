"""Seal an allowlisted distribution; exclude installed venv and runtime data."""
import hashlib
import json
from pathlib import Path
import zipfile
import argparse

base = Path('artifacts/human-video-v1')
parser=argparse.ArgumentParser()
parser.add_argument('--root',default=str(base/'joint-cli-0.2.1-win64-py312'))
parser.add_argument('--archive',default=str(base/'piper-combined-cli-0.2.1-win64-py312.zip'))
parser.add_argument('--manifest-only',action='store_true')
args=parser.parse_args()
root = Path(args.root)
top_files = ('piper.py', 'piper.cmd', 'install.ps1', 'check_python.py', 'README.md', 'README.en.md', 'START-HERE.md',
             'THIRD-PARTY.md', 'requirements-win-py312.lock', 'model-files.json',
             'PACKAGE-VALIDATION.json')
paths = [root/n for n in top_files]
for dirname in ('wheels', 'examples', 'models', 'licenses', 'profiles', 'docs'):
    paths.extend(p for p in (root/dirname).rglob('*') if p.is_file())
assert all(p.is_file() for p in paths)
assert len(list((root/'wheels').glob('*.whl'))) == 101
rows = [{'path':p.relative_to(root).as_posix(),'bytes':p.stat().st_size,
         'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(paths)]
(root/'SHA256.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
paths.append(root/'SHA256.json')
if args.manifest_only:
    print(json.dumps({'manifest':str(root/'SHA256.json'),'files':len(paths)}))
    raise SystemExit(0)
archive = Path(args.archive)
with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=3) as z:
    for p in paths:
        z.write(p, 'piper-cli/'+p.relative_to(root).as_posix())
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    for row in rows:
        assert hashlib.sha256(z.read('piper-cli/'+row['path'])).hexdigest()==row['sha256']
digest=hashlib.sha256(archive.read_bytes()).hexdigest()
archive.with_suffix('.zip.sha256').write_text(digest+'  '+archive.name+'\n',encoding='ascii')
print(json.dumps({'zip':str(archive),'bytes':archive.stat().st_size,
                  'sha256':digest,'files':len(paths),'archive_integrity':'passed'},indent=2))
