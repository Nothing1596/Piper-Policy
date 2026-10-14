"""Seal tested kits using their original allowlisted manifest, excluding runtime files."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    a = p.parse_args()
    repo = Path(__file__).resolve().parents[1]
    for kind, version, namespace, project in [('video','0.3.0','piperlab','piper-lab'),
                                              ('robot','0.6.0','piperx_middleware','piperx-cli')]:
        kit = a.root/f'piper-{kind}-{version}-win64-py312'
        manifest = json.loads((kit/'SHA256.json').read_text(encoding='utf-8'))
        names = {entry['path'] for entry in manifest}
        for entry in manifest:
            assert sha((kit/entry['path']).read_bytes()) == entry['sha256'], entry['path']
        for language, suffix in [('zh',''),('en','.en')]:
            guide = (repo/f'docs/{kind.upper()}-GUIDE.{language}.md').read_text(encoding='utf-8')
            guide = guide.replace(f'{kind.upper()}-GUIDE.en.md','README.en.md').replace(f'{kind.upper()}-GUIDE.zh.md','README.md')
            (kit/f'README{suffix}.md').write_text(guide,encoding='utf-8')
        validation = json.loads((a.root/f'{kind}-validation/validation.json').read_text(encoding='utf-8'))
        validation.pop('python')
        sources = {}
        wheel = next((kit/'wheels').glob('piper_*.whl' if kind=='video' else 'piperx_*.whl'))
        with zipfile.ZipFile(wheel) as archive:
            for path in sorted((repo/project/'src'/namespace).rglob('*.py')):
                relative = path.relative_to(repo/project/'src').as_posix()
                source = path.read_bytes()
                assert archive.read(relative) == source, relative
                installed = kit/'.venv/Lib/site-packages'/relative
                assert installed.read_bytes() == source, relative
                sources[relative] = sha(source)
        validation.update({'source_wheel_installed_match':True,'source_sha256':sources,
                           'model_api_inference_tested':False,'physical_hardware_tested':False})
        (kit/'PACKAGE-VALIDATION.json').write_text(json.dumps(validation,indent=2),encoding='utf-8')
        names.add('PACKAGE-VALIDATION.json')
        manifest = [{'path':name,'sha256':sha((kit/name).read_bytes())} for name in sorted(names)]
        (kit/'SHA256.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
        names.add('SHA256.json')
        output = a.root/(kit.name+'.zip')
        assert not output.exists()
        with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
            for name in sorted(names):
                assert not any(part in ('.venv','work') for part in Path(name).parts)
                assert not name.endswith(('.token','.key'))
                archive.write(kit/name,kit.name+'/'+name)
        with zipfile.ZipFile(output) as archive:
            assert archive.testzip() is None
            for entry in manifest:
                assert sha(archive.read(kit.name+'/'+entry['path'])) == entry['sha256']
        digest = sha(output.read_bytes())
        (a.root/f'{kind}-SHA256SUMS.txt').write_text(f'{digest}  {output.name}\n')
        print(json.dumps({'zip':output.name,'size':output.stat().st_size,'sha256':digest,'source_modules':len(sources)}))


if __name__ == '__main__':
    main()
