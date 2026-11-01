#!/usr/bin/env python3
"""Patch WatchAct ground height and support rolling resistance into a new XML."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

# Engineering material assumption (length units), frozen before policy runs.
# It is validated with free-body fixtures, not calibrated to task success.
ROLLING_FRICTION_M = .002


def vector(text, *, positive=False):
    values = [float(v) for v in text.split()]
    if len(values) != 3 or not all(math.isfinite(v) and (not positive or v > 0) for v in values):
        raise ValueError('Expected three finite' + (' positive' if positive else '') + ' numbers')
    return values


def single(parent, query):
    matches = parent.findall(query)
    if len(matches) != 1:
        raise ValueError('Expected exactly one ' + query)
    return matches[0]


def validate_and_patch(source_path: Path, output_path: Path) -> dict:
    original = source_path.read_bytes()
    root = ET.fromstring(original)
    if root.tag != 'mujoco':
        raise ValueError('Expected mujoco root')
    world = single(root, 'worldbody')
    floor = single(world, "geom[@name='floor']")
    table = single(world, "body[@name='table']")
    top = single(table, "geom[@name='table_top']")
    if floor.get('type') != 'plane' or top.get('type') != 'box':
        raise ValueError('Expected plane floor and box table_top')
    for element in (floor, table, top):
        if {'quat', 'euler', 'axisangle', 'xyaxes', 'zaxis', 'fromto'} & element.attrib.keys():
            raise ValueError('Only explicitly axis-aligned geometry is supported')
    if table.find('joint') is not None or table.find('freejoint') is not None:
        raise ValueError('Expected static table')
    floor_pos = vector(floor.get('pos', '0 0 0'))
    table_pos = vector(table.get('pos', '0 0 0'))
    geom_pos = vector(top.get('pos', '0 0 0'))
    size = vector(top.get('size', ''), positive=True)
    floor_pos[2] = table_pos[2] + geom_pos[2] - size[2]
    supports = [floor, top]
    basket_floors = world.findall("body[@name='tray']/geom[@name='tray_floor']")
    if len(basket_floors) > 1:
        raise ValueError('Duplicate basket floor')
    supports.extend(basket_floors)
    updates = [(floor, 'pos', ' '.join(map(str, floor_pos)))]
    for support in supports:
        friction = vector(support.get('friction', '1 .005 .0001'))
        if any(v < 0 for v in friction):
            raise ValueError('Negative support friction')
        friction[2] = ROLLING_FRICTION_M
        updates.extend([(support, 'condim', '6'),
                        (support, 'friction', ' '.join(map(str, friction)))])
    changes = []
    for element, key, value in updates:
        old = element.get(key)
        element.set(key, value)
        changes.append({'element': element.get('name'), 'attribute': key, 'old': old, 'new': value})
    with output_path.open('xb') as stream:
        ET.ElementTree(root).write(stream, encoding='utf-8', xml_declaration=True)
    return {'source_sha256': hashlib.sha256(original).hexdigest(),
            'output_sha256': hashlib.sha256(output_path.read_bytes()).hexdigest(), 'changes': changes}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(validate_and_patch(args.source, args.output), indent=2))
