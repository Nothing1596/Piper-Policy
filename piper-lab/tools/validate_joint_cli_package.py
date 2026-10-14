"""Exercise a fresh offline installation without model/GPU inference or hardware."""
import hashlib
import importlib.metadata as md
import json
from pathlib import Path
import subprocess
import sys
import asyncio
import tempfile
import zipfile

root = Path(sys.argv[1]).resolve()
assert Path(sys.prefix).resolve() == root / '.venv'
import piperlab
import piperx_middleware
assert Path(piperlab.__file__).is_relative_to(root / '.venv')
assert Path(piperx_middleware.__file__).is_relative_to(root / '.venv')
result = {'fresh_offline_install': True, 'shared_development_dependencies': False,
          'python': sys.version, 'cli_checks': [], 'model_inference_tested': False,
          'hardware_tested': False}
for args in ([], ['demo', 'compile', '--help'], ['video', 'inspect', '--help'],
             ['policy', 'run', '--help'], ['sim', 'start', '--help'],
             ['eval', 'run', '--help'], ['device', '--help'], ['lab', '--help'],
             ['models','--help'], ['mcp','--help'], ['models','show','--model-config','profiles/lmstudio.json'],
             ['doctor'], ['unknown-command']):
    completed = subprocess.run([str(root/'piper.cmd'), *args], cwd=root,
                               capture_output=True, text=True, timeout=60)
    expected = 2 if args == ['unknown-command'] else 0
    assert completed.returncode == expected, (args, completed.stderr)
    result['cli_checks'].append({'args': args, 'exit_code': completed.returncode})
import av
with av.open(str(root/'examples/transfer.mp4')) as container:
    frame = next(container.decode(video=0))
    rgb = frame.to_ndarray(format='rgb24')
    assert frame.pts is not None
    result['sample_decode'] = {'shape': list(rgb.shape), 'pts': frame.pts}
from piperlab.perception.detector_onnx import TrackedDetector
detector = TrackedDetector(root/'models/yolo11n.onnx')
detections = detector.score(rgb, source_id='package-test', sequence=0, source_s=0.0)
result['cpu_detector_and_tracker'] = {'ran': True, 'detection_count': len(detections)}
from piperx_middleware.mujoco_backend import MujocoBackend
backend = MujocoBackend(seed=200)
try:
    backend.connect()
    state = backend.snapshot()
    result['mujoco'] = {'loaded_packaged_assets': True, 'snapshot_type': type(state).__name__,
                        'nq': backend.model.nq, 'simulation_time': float(backend.data.time)}
    assert backend.model.nq > 0 and backend.data.time > 0
finally:
    backend.close()
check = subprocess.run([sys.executable, '-m', 'pip', 'check'], capture_output=True, text=True)
assert check.returncode == 0, check.stdout
result['pip_check'] = check.stdout.strip()
result['versions'] = {n: md.version(n) for n in ('piper-lab', 'piperx-middleware', 'av',
                                               'onnxruntime', 'mujoco', 'trackers')}
result['wheel_sha256'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in (root/'wheels').glob('piper*.whl')}
verified_modules = {}
site = Path(piperlab.__file__).parent.parent
with zipfile.ZipFile(next((root/'wheels').glob('piper_lab-*.whl'))) as wheel:
    for name in wheel.namelist():
        if name.startswith('piperlab/') and name.endswith('.py'):
            digest = hashlib.sha256(wheel.read(name)).hexdigest()
            assert hashlib.sha256((site/name).read_bytes()).hexdigest() == digest
            verified_modules[name] = digest
result['installed_upper_module_hashes'] = verified_modules

async def verify_mcp(workspace):
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    from PIL import Image
    Image.fromarray(rgb).save(workspace/'test.png')
    params = StdioServerParameters(command=sys.executable,args=[str(root/'piper.py'),'mcp','--workspace',str(workspace)])
    async with stdio_client(params) as (reader, writer):
        async with ClientSession(reader, writer) as session:
            await session.initialize()
            names = [t.name for t in (await session.list_tools()).tools]
            assert 'video_compile' in names and not any(n.startswith('robot_') for n in names)
            response = await session.call_tool('video_frame',{'image':'test.png'})
            assert not response.isError and response.content[0].type == 'image'
            return {'transport':'stdio','tools':names,'image_transfer':True,'robot_connected':False}
with tempfile.TemporaryDirectory(prefix='piper-package-mcp-') as temporary:
    result['mcp'] = asyncio.run(verify_mcp(Path(temporary)))
(root/'PACKAGE-VALIDATION.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps(result, indent=2))
