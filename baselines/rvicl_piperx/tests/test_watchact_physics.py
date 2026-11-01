from pathlib import Path
import hashlib
import xml.etree.ElementTree as ET
import pytest
from scripts.fix_watchact_scene_physics import validate_and_patch

SCENE = '''<mujoco><worldbody>
<geom name="floor" type="plane" friction="1 .005 .0001"/>
<body name="table" pos=".32 0 -.2"><geom name="table_top" type="box" size=".44 .40 .20" friction=".8 .005 .0001"/></body>
<body name="robot"><joint name="j"/><geom name="finger" friction="2 .01 .001"/></body>
<body name="cube_red"><freejoint/><geom name="can" type="cylinder" mass=".15" size=".034 .0418"/></body>
<body name="tray"><geom name="tray_floor" friction=".95 .3 .1"/></body>
</worldbody><actuator><position joint="j" kp="100"/></actuator></mujoco>'''


def test_only_support_attributes_change_and_source_is_retained(tmp_path):
    source, output = tmp_path/'old.xml', tmp_path/'new.xml'
    source.write_text(SCENE); original = source.read_bytes()
    result = validate_and_patch(source, output)
    before, after = ET.fromstring(original), ET.parse(output).getroot()
    assert source.read_bytes() == original
    assert result['source_sha256'] == hashlib.sha256(original).hexdigest()
    assert result['output_sha256'] == hashlib.sha256(output.read_bytes()).hexdigest()
    changes=[]
    for old,new in zip(before.iter(),after.iter(),strict=True):
        assert old.tag == new.tag
        for key in old.attrib.keys() | new.attrib.keys():
            if old.get(key) != new.get(key):changes.append((new.get('name'),key))
    assert sorted(changes) == [('floor','condim'),('floor','friction'),('floor','pos'),('table_top','condim'),('table_top','friction'),('tray_floor','condim'),('tray_floor','friction')]
    assert after.find("worldbody/geom[@name='floor']").get('pos') == '0.0 0.0 -0.4'
    for name in ['floor','table_top','tray_floor']:
        a=after.find(f".//geom[@name='{name}']");b=before.find(f".//geom[@name='{name}']")
        assert list(map(float,a.get('friction').split()))[:2] == list(map(float,b.get('friction').split()))[:2]
        assert float(a.get('friction').split()[2]) == .002
        assert a.get('condim') == '6'
    output_bytes=output.read_bytes()
    with pytest.raises(FileExistsError):validate_and_patch(source,output)
    assert output.read_bytes() == output_bytes


@pytest.mark.parametrize('replace,new',[
    ('type="plane"','type="box"'),
    ('type="plane"','type="plane" quat="1 0 0 0"'),
    ('name="table"','name="table" euler="0 0 0"'),
    ('name="table_top"','name="table_top" axisangle="0 0 1 0"'),
    ('name="table_top"','name="table_top" fromto="0 0 0 1 0 0"'),
    ('size=".44 .40 .20"','size="nan .4 .2"'),
    ('size=".44 .40 .20"','size="inf .4 .2"'),
    ('size=".44 .40 .20"','size=".44 0 .2"'),
    ('pos=".32 0 -.2"','pos=".32 nan -.2"'),
    ('name="floor"','name="missing"'),
    ('name="table_top"','name="missing"'),
    ('<body name="table" pos=".32 0 -.2">','<body name="table" pos=".32 0 -.2"><freejoint/>'),
    ('<body name="table" pos=".32 0 -.2">','<body name="table" pos=".32 0 -.2"><joint/>'),
    ('<worldbody>','<worldbody><geom name="floor" type="plane"/>'),
])
def test_unsupported_or_corrupt_scene_is_rejected_before_creation(tmp_path,replace,new):
    source, output = tmp_path/'old.xml', tmp_path/'new.xml'
    source.write_text(SCENE.replace(replace,new))
    with pytest.raises(ValueError):validate_and_patch(source,output)
    assert not output.exists()


def test_table_geom_offset_and_old_floor_xy_are_respected(tmp_path):
    source, output = tmp_path/'old.xml', tmp_path/'new.xml'
    source.write_text(SCENE.replace('type="plane"','type="plane" pos="1 2 0"')
        .replace('name="table_top"','name="table_top" pos="0 0 -.05"'))
    validate_and_patch(source,output)
    pos=ET.parse(output).find("worldbody/geom[@name='floor']").get('pos')
    assert list(map(float,pos.split())) == [1,2,-.45]
