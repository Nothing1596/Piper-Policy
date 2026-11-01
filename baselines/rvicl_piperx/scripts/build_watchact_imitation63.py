#!/usr/bin/env python3
"""Build one explicitly adapted WatchAct scene, without changing robot/solver physics."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation
from piperx_middleware.mujoco_backend import DEFAULT_ASSET_PATH


def build(output, object_assets, basket_source):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    assets = output / 'assets'
    shutil.copytree(object_assets, assets)
    shutil.copytree(basket_source, assets / 'basket', ignore=shutil.ignore_patterns('*.blend', '*.stl'))
    scene = ET.parse(DEFAULT_ASSET_PATH)
    root = scene.getroot()
    compiler = root.find('compiler')
    compiler.set('meshdir', str(DEFAULT_ASSET_PATH.parent / 'meshes'))
    asset = root.find('asset')
    world = root.find('worldbody')
    for name in ('cube_red', 'cube_blue', 'tray'):
        world.remove(world.find(f"body[@name='{name}']"))
    world.find("body[@name='table']/geom").set('size', '.44 .40 .20')
    world.find("body[@name='table']/geom").set('rgba', '.42 .27 .14 1')
    world.find("body[@name='table']/geom").attrib.pop('material', None)
    cam = world.find("camera[@name='overview_camera']")
    pos, target = np.array([.75, -.55, .9]), np.array([.32, .03, .015])
    back = pos - target; back /= np.linalg.norm(back)
    right = np.cross([0., 0., 1.], back); right /= np.linalg.norm(right)
    up = np.cross(back, right)
    cam.set('pos', ' '.join(map(str, pos)))
    cam.set('quat', ' '.join(map(str, Rotation.from_matrix(np.column_stack([right, up, back])).as_quat(scalar_first=True))))
    cam.set('fovy', '48')
    for obj, alias, xyz, half, mass in (
        ('alphabet_soup', 'cube_red', [.27, -.20, .043], [.034, .0418], '.15'),
        ('ketchup', 'cube_blue', [.27, -.05, .075], [.0323, .0217, .0744], '.12')):
        ET.SubElement(asset, 'texture', name=obj+'_texture', type='2d', file=str(assets/obj/'texture_map.png'))
        ET.SubElement(asset, 'material', name=obj+'_material', texture=obj+'_texture', texuniform='false')
        ET.SubElement(asset, 'mesh', name=obj+'_mesh', file=str(assets/obj/'visual/textured_vis.msh'), scale='.01 .01 .01')
        body=ET.SubElement(world, 'body', name=alias, pos=' '.join(map(str, xyz)))
        # Deliberately different free-joint names preserve asset initial positions;
        # the legacy backend only randomizes its two default cube_*_joint names.
        ET.SubElement(body, 'freejoint', name=obj+'_free')
        ET.SubElement(body, 'geom', name=alias+'_visual', type='mesh', mesh=obj+'_mesh',
            quat='.7071067812 .7071067812 0 0', material=obj+'_material', contype='0', conaffinity='0', mass='0', group='2')
        ET.SubElement(body, 'geom', name=alias+'_geom', type='cylinder' if obj=='alphabet_soup' else 'box',
            size=' '.join(map(str,half)), mass=mass, rgba='0 0 0 0', group='3', friction='1.2 .005 .0001',
            solref='.004 1', solimp='.95 .99 .001')
    source = ET.parse(assets/'basket/model_tier1.xml').getroot()
    for item in source.find('asset'):
        item = copy.deepcopy(item)
        if item.get('file'): item.set('file', str(assets/'basket'/item.get('file')))
        asset.append(item)
    basket = ET.SubElement(world, 'body', name='tray', pos='.40 .20 0')
    for index, item in enumerate(source.findall('worldbody/body/body/geom')):
        if item.get('contype') == '0' and item.get('conaffinity') == '0':
            item=copy.deepcopy(item);item.set('name','tray_visual_'+str(index));basket.append(item)
    # The source convex pieces bridge the cavity in this importer. Use an
    # explicit open floor and four walls at the published contain-region bounds.
    for name, pos, size in (
        ('floor', [0,0,.001], [.09516,.07526,.001]),
        ('east', [.09016,0,.0645], [.005,.07526,.0625]),
        ('west', [-.09016,0,.0645], [.005,.07526,.0625]),
        ('north', [0,.07026,.0645], [.08516,.005,.0625]),
        ('south', [0,-.07026,.0645], [.08516,.005,.0625])):
        ET.SubElement(basket,'geom',name='tray_'+name,type='box',
            pos=' '.join(map(str,pos)),size=' '.join(map(str,size)),rgba='0 0 0 0',group='3',
            friction='.95 .3 .1',solimp='.998 .998 .001',solref='.001 1')
    ET.SubElement(basket, 'site', name='tray_center', pos='0 0 .06985', size='.001', rgba='0 0 0 0')
    scene.write(output/'scene.xml', encoding='unicode')
    provenance = {'task':'imitation_data_315','original_id':'following_sequence_example_63',
        'adaptation':'Original visual assets, cylinder/box collision proxies for can/bottle; open basket floor/four walls replace cavity-bridging convex pieces. Fixed basket, unchanged robot control/contact parameters.',
        'body_aliases':{'cube_red':'alphabet_soup_1','cube_blue':'ketchup_1','tray':'basket_1'},
        'basket_floor_top_m': .002,
        'layout_transform':{'source':'official BDDL plane centres','xy_translation_m':[.34,-.15], 'object_scale':1},
        'base_scene_sha256':hashlib.sha256(DEFAULT_ASSET_PATH.read_bytes()).hexdigest(),
        'files':{str(p.relative_to(output)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(output.rglob('*')) if p.is_file()}}
    (output/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    return output/'scene.xml'


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--object-assets',type=Path,required=True)
    p.add_argument('--basket-source',type=Path,required=True)
    a=p.parse_args();print(build(a.output,a.object_assets,a.basket_source))
