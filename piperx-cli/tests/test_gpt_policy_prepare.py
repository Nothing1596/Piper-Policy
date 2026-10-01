"""Pinned-checkout preparation and offline config checks, without inference."""
from pathlib import Path
import json
import subprocess
import sys
import pytest
from gpt_policy_piper.prepare import prepare,verify_prepared,write_config,MANIFEST
from piperx_middleware.cli import initialize


def test_generated_config_uses_upstream_provider_directory(tmp_path):
    upstream=tmp_path/'upstream';(upstream/'configs').mkdir(parents=True)
    runtime=tmp_path/'runtime'
    initialize(runtime,'mujoco',port=8808,tcp_offset_m=[0.,0.,.1425])
    path=write_config(upstream,runtime)
    value=json.loads(path.read_text())
    assert value['agent_config_dir']==str(upstream/'configs')
    assert value['calibration']['link6_from_tcp'][2][3]==.1425
    assert 'token' not in value['piper_simulation']
    with pytest.raises(FileExistsError):write_config(upstream,runtime)


def test_preparation_pin_idempotency_and_manifest_tamper(tmp_path):
    upstream_module=pytest.importorskip('gpt_policy')
    source=Path(upstream_module.__file__).resolve().parents[2]
    if not (source/'.git').exists():pytest.skip('Requires a supplied Git checkout')
    target=tmp_path/'evaluation-checkout'
    subprocess.run(['git','-C',str(source),'worktree','add','--detach',str(target),'HEAD'],check=True,capture_output=True)
    try:
        assert prepare(target)['status']=='prepared'
        assert prepare(target)['status']=='already_prepared'
        manifest=json.loads((target/MANIFEST).read_text())
        changed=target/'src/gpt_policy/main.py'
        changed.write_text(changed.read_text()+'\n# unapproved edit\n')
        import hashlib
        manifest['files']['src/gpt_policy/main.py']=hashlib.sha256(changed.read_bytes()).hexdigest()
        (target/MANIFEST).write_text(json.dumps(manifest))
        with pytest.raises(ValueError,match='changed'):verify_prepared(target)
        assert changed.read_text().endswith('# unapproved edit\n')
    finally:
        subprocess.run(['git','-C',str(source),'worktree','remove','--force',str(target)],check=True,capture_output=True)


def test_import_upstream_main_without_linux_fcntl():
    pytest.importorskip('gpt_policy')
    code='''import importlib.abc, sys
class NoFcntl(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "fcntl": raise ModuleNotFoundError("no fcntl on Windows")
sys.modules.pop("fcntl",None)
sys.meta_path.insert(0,NoFcntl())
import gpt_policy.main
assert gpt_policy.main.CameraSet is not None
'''
    result=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
