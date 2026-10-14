"""Small local RGB-D CLI adapter; camera worker has no CAN ownership."""
import asyncio
import json
from pathlib import Path
import sys
import uuid


async def observe(root, serial=None):
    root = Path(root)
    config_path = root / 'camera.json'
    config = json.loads(config_path.read_text(encoding='utf-8')) if config_path.exists() else {}
    output = Path(config.get('output_root', root / 'observations')) / uuid.uuid4().hex
    args = ['--output', str(output)]
    if serial or config.get('serial'):
        args += ['--serial', serial or config['serial']]
    result = await worker(config, args)
    if 'error' not in result:
        (root / 'last_observation.json').write_text(json.dumps({'path': str(output / 'observation.json')}), encoding='utf-8')
    return result


async def pixel(root, u, v):
    root = Path(root)
    config_path = root / 'camera.json'
    config = json.loads(config_path.read_text(encoding='utf-8')) if config_path.exists() else {}
    last = json.loads((root / 'last_observation.json').read_text(encoding='utf-8'))
    return await worker(config, ['--observation', last['path'], '--pixel', str(u), str(v)])


async def worker(config, args):
    proc = await asyncio.create_subprocess_exec(
        config.get('python', sys.executable), str(Path(__file__).with_name('camera_capture.py')), *args,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=60)
    except BaseException:
        if proc.returncode is None:
            proc.kill()
        await proc.communicate()
        raise
    try:
        result = json.loads(stdout.decode('utf-8'))
    except (ValueError, UnicodeError):
        raise RuntimeError('Camera worker failed: ' + stderr.decode('utf-8', errors='replace')[-1200:])
    if proc.returncode and 'error' not in result:
        raise RuntimeError(f'Camera worker exited {proc.returncode}')
    return result
