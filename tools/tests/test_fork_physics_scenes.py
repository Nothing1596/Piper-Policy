import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fork_physics_scenes import fork_scene


def scene(tmp_path, directory='/absolute/assets'):
    p = tmp_path/'old.xml'
    p.write_text(f'<mujoco><compiler meshdir="{directory}"/><option solver="Newton" cone="elliptic"/><asset/><worldbody><geom friction="1.5 .01 .001"/></worldbody></mujoco>')
    p.with_suffix('.json').write_text(json.dumps({'scene_sha256':hashlib.sha256(p.read_bytes()).hexdigest(), 'asset_sha256s':{}, 'initial':'preserved'}))
    return p


def test_fork_is_only_ratio_and_never_overwrites(tmp_path):
    old=scene(tmp_path); before=old.read_bytes(); target=tmp_path/'new/s.xml'
    change=fork_scene(old,target)
    assert old.read_bytes()==before
    new=ET.parse(target).getroot(); new.find('option').attrib.pop('impratio')
    assert ET.tostring(new)==ET.tostring(ET.fromstring(before))
    assert change['change']=={'impratio':{'old':1,'new':100}}
    assert json.loads(target.with_suffix('.json').read_text())['initial']=='preserved'
    with pytest.raises(FileExistsError):fork_scene(old,target)


def test_reject_modified_input(tmp_path):
    old=scene(tmp_path);old.write_text(old.read_text()+' ')
    with pytest.raises(AssertionError, match='Source scene changed'):fork_scene(old,tmp_path/'new.xml')


def test_reject_relative_assets(tmp_path):
    old=scene(tmp_path,'meshes')
    with pytest.raises(AssertionError, match='Relative asset directory'):fork_scene(old,tmp_path/'new.xml')
