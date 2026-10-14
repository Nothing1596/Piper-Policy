"""Scene construction boundaries, including upstream malformed comment tails."""
import copy
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as E

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).parents[1]))
from longvil_physics_scene import BASE_FROM_DATASET,build_scene,parse_urdf

TEMPLATE=Path(__file__).parents[2]/'piperx-cli/assets/mujoco/scene.xml'


def fixture(tmp_path):
    folder=tmp_path/'assets/buildings';folder.mkdir(parents=True)
    (folder/'cube.urdf').write_text('''<robot name="cube"><link name="root">
    <visual><geometry><box size=".04 .04 .04"/></geometry></visual>
    <collision><geometry><box size=".04 .04 .04"/></geometry></collision>
    <inertial><mass value=".1"/></inertial></link></robot>''')
    return {'object_list':['orange cube'],'positions':{'orange cube':{'x':.1,'y':-.5,'z':.03}},
            'pre_action_sequences':None,'code':['DO NOT READ'],'action_sequences':['DO NOT READ']}


def test_scene_uses_initial_fields_without_changing_robot_or_adding_welds(tmp_path):
    data=fixture(tmp_path);first=tmp_path/'first.xml'
    meta=build_scene(data,TEMPLATE,tmp_path/'assets',first)
    modified=copy.deepcopy(data);modified['code']=['arbitrary different answer'];modified['action_sequences']=[]
    second=tmp_path/'second.xml';build_scene(modified,TEMPLATE,tmp_path/'assets',second)
    assert first.read_bytes()==second.read_bytes()
    original=E.parse(TEMPLATE).getroot();rebuilt=E.parse(first).getroot()
    assert E.tostring(original.find('actuator'))==E.tostring(rebuilt.find('actuator'))
    assert E.tostring(original.find("worldbody/body[@name='base_link']"))==E.tostring(rebuilt.find("worldbody/body[@name='base_link']"))
    assert not rebuilt.findall('.//weld')
    assert rebuilt.find("worldbody/body[@name='cube_red']") is None
    assert meta['object_count']==1
    body=rebuilt.find("worldbody/body[@name='object__orange_cube']")
    assert body.find('freejoint') is not None
    assert np.allclose(np.fromstring(body.get('pos'),sep=' '),[.38,.1,.03])


def test_missing_authoritative_asset_is_not_silently_substituted(tmp_path):
    data=fixture(tmp_path);(tmp_path/'assets/buildings/cube.urdf').unlink()
    with pytest.raises(ValueError,match='authoritative'):build_scene(data,TEMPLATE,tmp_path/'assets',tmp_path/'scene.xml')


def test_trailing_nested_comments_do_not_drop_active_geometry(tmp_path):
    p=tmp_path/'asset.urdf'
    p.write_text('<robot name="fruit"><link name="active"/></robot>\n<!-- inactive <!-- nested --> -->')
    root=parse_urdf(p)
    assert root.find('link').get('name')=='active'
    p.write_text('<robot><link broken="true"></robot>')
    with pytest.raises(E.ParseError):parse_urdf(p)


def test_rigid_retarget_preserves_scale_and_dataset_directions():
    R=BASE_FROM_DATASET[:3,:3];p=np.array([.2,-.5,.03]);delta=np.array([-.1,0,0])
    assert np.isclose(np.linalg.norm(R@delta),np.linalg.norm(delta))
    assert np.allclose(R.T@((R@(p+delta))-(R@p)),delta)


def test_translation_moves_table_and_objects_without_scaling_geometry(tmp_path):
    data=fixture(tmp_path);first=tmp_path/'original.xml';second=tmp_path/'translated.xml'
    build_scene(data,TEMPLATE,tmp_path/'assets',first)
    meta=build_scene(data,TEMPLATE,tmp_path/'assets',second,translation_m=[-.3,-.1,-.2])
    a,b=E.parse(first).getroot(),E.parse(second).getroot()
    body=b.find("worldbody/body[@name='object__orange_cube']")
    assert np.allclose(np.fromstring(body.get('pos'),sep=' '),[.2,0,-.17])
    assert meta['surface_z_m']==-.2
    assert a.find('.//geom[@name="object__orange_cube__root__c0"]').get('size')==body.find('geom[@name="object__orange_cube__root__c0"]').get('size')
    assert E.tostring(a.find('actuator'))==E.tostring(b.find('actuator'))
    assert np.allclose(np.asarray(meta['base_from_dataset'])[:3,:3],BASE_FROM_DATASET[:3,:3])
