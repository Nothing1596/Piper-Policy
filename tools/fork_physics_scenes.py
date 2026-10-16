"""Fork frozen MuJoCo scenes; change only impratio, never overwrite inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

REVISION = 'contact-impratio100-v2'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fork_scene(source, destination):
    source, destination = Path(source), Path(destination)
    if destination.exists() or destination.with_suffix('.json').exists():
        raise FileExistsError(destination)
    original = source.read_bytes()
    metadata = json.loads(source.with_suffix('.json').read_text())
    assert metadata['scene_sha256'] == digest(source), 'Source scene changed'
    for asset, expected in metadata['asset_sha256s'].items():
        assert digest(asset) == expected, 'Source asset changed: '+asset
    root = ET.fromstring(original)
    compiler = root.find('compiler')
    for attr in ('meshdir', 'texturedir'):
        if compiler is not None and compiler.get(attr):
            assert Path(compiler.get(attr)).is_absolute(), 'Relative asset directory'
    for asset in root.findall('./asset/*'):
        if asset.get('file') and not Path(asset.get('file')).is_absolute():
            key = 'meshdir' if asset.tag == 'mesh' else 'texturedir'
            assert compiler is not None and compiler.get(key) and Path(compiler.get(key)).is_absolute(), 'Relative asset file'
    option = root.find('option')
    assert option is not None
    previous = option.get('impratio')
    old_ratio = float(previous or '1')
    assert old_ratio == 1, 'Expected original profile; do not stack revisions'
    option.set('impratio', '100')
    new_bytes = ET.tostring(root, encoding='utf-8')
    option.attrib.pop('impratio')
    if previous is not None:
        option.set('impratio', previous)
    assert ET.tostring(root) == ET.tostring(ET.fromstring(original))
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('xb') as stream:
        stream.write(new_bytes)
    change = {'physics_revision': REVISION, 'source_scene': str(source),
              'source_scene_sha256': hashlib.sha256(original).hexdigest(),
              'scene': str(destination), 'scene_sha256': digest(destination),
              'old_options': ET.fromstring(original).find('option').attrib,
              'new_options': ET.fromstring(new_bytes).find('option').attrib,
              'change': {'impratio': {'old': old_ratio, 'new': 100}},
              'unchanged': 'All other XML elements and asset hashes; no weld or attachment'}
    metadata.update(scene_sha256=digest(destination), physics_revision=REVISION, physics_change=change)
    with destination.with_suffix('.json').open('x') as stream:
        json.dump(metadata, stream, indent=2)
    assert source.read_bytes() == original
    return change


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--destination', required=True, type=Path)
    parser.add_argument('--tasks', nargs='+', required=True)
    args = parser.parse_args()
    args.destination.mkdir(parents=True, exist_ok=False)
    changes = [fork_scene(args.source/(task+'.xml'), args.destination/(task+'.xml')) for task in args.tasks]
    (args.destination/'physics-manifest.json').write_text(json.dumps(changes, indent=2))


if __name__ == '__main__':
    main()
