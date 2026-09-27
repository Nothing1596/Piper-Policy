"""Build independent Windows/Python 3.12 offline kits from a local wheel depot.

Requires freshly built project wheels. Does not download dependencies or copy credentials.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from urllib.parse import unquote, urlparse
from urllib.request import url2pathname


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--depot', type=Path, required=True)
    p.add_argument('--project-wheels', type=Path, required=True)
    p.add_argument('--template', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    repo = Path(__file__).resolve().parents[1]
    a.output.mkdir(parents=True, exist_ok=True)
    for kind, version, wheel, extras, module in [
        ('video', '0.3.0', 'piper_lab-0.3.0-py3-none-any.whl', 'video,detector,model,mcp', 'piperlab.video_entry'),
        ('robot', '0.6.0', 'piperx_middleware-0.6.0-py3-none-any.whl', 'simulation,hardware', 'piperx_middleware.standalone_cli'),
    ]:
        root = a.output / f'piper-{kind}-{version}-win64-py312'
        root.mkdir(exist_ok=False)
        (root/'wheels').mkdir()
        report = a.output/f'{kind}-resolution.json'
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--dry-run', '--ignore-installed', '--no-index',
                        '--find-links', str(a.depot), '--report', str(report),
                        str(a.project_wheels/wheel)+f'[{extras}]'], check=True)
        pins = []
        for item in json.loads(report.read_text(encoding='utf-8'))['install']:
            source = Path(url2pathname(unquote(urlparse(item['download_info']['url']).path)))
            target = root/'wheels'/source.name
            shutil.copy2(source, target)
            pins.append(f"{item['metadata']['name']}=={item['metadata']['version']} --hash=sha256:{sha(target)}")
        (root/'requirements-win-py312.lock').write_text('\n'.join(sorted(pins))+'\n')
        for name in ['check_python.py', 'THIRD-PARTY.md']:
            shutil.copy2(a.template/name, root/name)
        shutil.copytree(a.template/'licenses', root/'licenses')
        if kind == 'video':
            for name in ['examples', 'profiles', 'models']:
                shutil.copytree(a.template/name, root/name)
            shutil.copy2(repo/'docs/DETECTOR-SOURCE.md', root/'DETECTOR-SOURCE.md')
        (root/'work').mkdir()
        reference = (repo/f'docs/{kind.upper()}-CLI-REFERENCE.md').read_text(encoding='utf-8')
        reference = reference.replace('(../README.md)', '(https://github.com/Nothing1596/Piper-Policy/blob/main/README.md)')
        reference = reference.replace('(../README.en.md)', '(https://github.com/Nothing1596/Piper-Policy/blob/main/README.en.md)')
        reference = reference.replace(f'({kind.upper()}-GUIDE.zh.md)', '(README.md)').replace(f'({kind.upper()}-GUIDE.en.md)', '(README.en.md)')
        reference = reference.replace('(SPLIT-VALIDATION.md)', '(https://github.com/Nothing1596/Piper-Policy/blob/main/docs/SPLIT-VALIDATION.md)')
        (root/f'{kind.upper()}-CLI-REFERENCE.md').write_text(reference, encoding='utf-8')
        for language, suffix in [('zh', ''), ('en', '.en')]:
            guide = (repo/f'docs/{kind.upper()}-GUIDE.{language}.md').read_text(encoding='utf-8')
            guide = guide.replace(f'{kind.upper()}-GUIDE.en.md', 'README.en.md').replace(f'{kind.upper()}-GUIDE.zh.md', 'README.md')
            (root/f'README{suffix}.md').write_text(guide, encoding='utf-8')
        installer = (a.template/'install.ps1').read_text()
        start = installer.index('& $python -m pip install --no-index --no-deps')
        end = installer.index('& $python -m pip check', start)
        installer = installer[:start]+installer[end:]
        start = installer.index('& $python piper.py doctor')
        installer = installer[:start]+f'& $python -m {module} --help\nif ($LASTEXITCODE -ne 0) {{ throw "CLI check failed" }}\nWrite-Output "Installed. Run .\\piper-{kind}.cmd --help."\n'
        (root/'install.ps1').write_text(installer)
        (root/f'piper-{kind}.cmd').write_text(f'@echo off\r\n"%~dp0.venv\\Scripts\\python.exe" -m {module} %*\r\nexit /b %errorlevel%\r\n')
        manifest = [{'path': f.relative_to(root).as_posix(), 'sha256': sha(f)}
                    for f in sorted(root.rglob('*')) if f.is_file()]
        (root/'SHA256.json').write_text(json.dumps(manifest, indent=2))
        print(json.dumps({'package': str(root), 'wheels': len(pins)}))


if __name__ == '__main__':
    main()
