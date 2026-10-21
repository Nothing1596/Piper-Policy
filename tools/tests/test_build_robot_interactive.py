"""Package layout/provenance checks; fixture wheels are not installed or executed."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest


def fixture_repo(tmp_path):
    repo=tmp_path/'repo'
    (repo/'tools').mkdir(parents=True)
    script=Path(__file__).resolve().parents[1]/'build_robot_interactive.py'
    (repo/'tools'/script.name).write_bytes(script.read_bytes())
    for name,version in [('piperx-cli','0.7.2'),('piper-lab','0.3.1')]:
        (repo/name/'src').mkdir(parents=True)
        (repo/name/'pyproject.toml').write_text(f'[project]\nname="fixture"\nversion="{version}"\n')
        (repo/name/'src'/'example.py').write_text('VALUE=1\n')
    (repo/'piperx-cli'/'private.token').write_text('fixture-token-do-not-package')
    (repo/'docs'/'implementation').mkdir(parents=True)
    (repo/'docs'/'versions').mkdir()
    (repo/'docs'/'versions'/'README.zh.md').write_text('version navigation\n')
    for doc in ('ONE-TASK-DEMO.zh.md','ROBOT-GUIDE.zh.md','ROBOT-GUIDE.en.md',
                'ROBOT-INTERACTIVE.zh.md','ROBOT-CLI-REFERENCE.md','ROBOT-RELEASE-0.7.2.md'):
        (repo/'docs'/doc).write_text(doc+'\n')
    (repo/'gpt.md').write_text('fixture conventions\n')
    (repo/'tools'/'parallels_mcp.py').write_text('# fixture\n')
    return repo


def wheel(tmp_path,name,version):
    path=tmp_path/f'{name}-{version}-py3-none-any.whl'
    with zipfile.ZipFile(path,'w') as z:
        z.writestr(f'{name}-{version}.dist-info/METADATA',f'Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n')
    return path


def test_kit_preserves_version_docs_sources_and_exact_wheel_identity(tmp_path):
    repo=fixture_repo(tmp_path)
    robot=wheel(tmp_path,'piperx_middleware','0.7.2')
    video=wheel(tmp_path,'piper_lab','0.3.1')
    output=tmp_path/'output';output.mkdir()
    result=subprocess.run([sys.executable,str(repo/'tools/build_robot_interactive.py'),
        '--wheel',str(robot),'--video-wheel',str(video),'--output',str(output),
        '--source-commit','a'*40],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    kit=output/'piper-robot-0.7.2-interactive'
    assert (kit/'docs/versions/README.zh.md').read_text()=='version navigation\n'
    assert (kit/'README.zh.md').read_text()=='ROBOT-GUIDE.zh.md\n'
    assert (kit/'source/piper-lab/src/example.py').exists()
    assert not (kit/'source/piperx-cli/private.token').exists()
    info=json.loads((kit/'BUILD-INFO.json').read_text())
    assert info['source_commit']=='a'*40 and info['video_version']=='0.3.1'
    assert info['wheels'][robot.name]==hashlib.sha256(robot.read_bytes()).hexdigest()
    for item in json.loads((kit/'SHA256.json').read_text()):
        assert hashlib.sha256((kit/item['path']).read_bytes()).hexdigest()==item['sha256']
    assert 'Python 3.12+' in (kit/'INSTALL.txt').read_text()
    with zipfile.ZipFile(output/'piper-robot-0.7.2-interactive.zip') as archive:
        assert 'piper-robot-0.7.2-interactive/docs/versions/README.zh.md' in archive.namelist()


@pytest.mark.parametrize('robot_version,video_version,message',[
    ('0.7.1','0.3.1','Robot wheel and source versions differ'),
    ('0.7.2','0.3.0','Video wheel and source versions differ')])
def test_version_mismatch_is_rejected_before_creating_kit(tmp_path,robot_version,video_version,message):
    repo=fixture_repo(tmp_path)
    output=tmp_path/'output';output.mkdir()
    result=subprocess.run([sys.executable,str(repo/'tools/build_robot_interactive.py'),
        '--wheel',str(wheel(tmp_path,'piperx_middleware',robot_version)),
        '--video-wheel',str(wheel(tmp_path,'piper_lab',video_version)),
        '--output',str(output)],capture_output=True,text=True)
    assert result.returncode!=0 and message in result.stderr
    assert list(output.iterdir())==[]
