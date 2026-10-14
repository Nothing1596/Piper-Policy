"""Run under each independently installed interpreter; no model API calls or hardware."""
import argparse
import asyncio
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import time

from PIL import Image
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def protocol(kind, kit, work, port):
    if kind == 'video':
        args = ['-m', 'piperlab.video_entry', 'mcp', '--workspace', str(kit)]
    else:
        args = ['-m', 'piperx_middleware.standalone_cli', 'mcp', '--root', str(work/'sim'),
                '--url', f'http://127.0.0.1:{port}', '--workspace', str(work)]
    params = StdioServerParameters(command=sys.executable, args=args)
    async with stdio_client(params) as (reader, writer):
        async with ClientSession(reader, writer) as session:
            await session.initialize()
            names = {t.name for t in (await session.list_tools()).tools}
            if kind == 'video':
                assert 'robot_status' not in names and 'simulation_observe' not in names
                result = await session.call_tool('video_candidates', {'video':'examples/transfer.mp4','output':'work/mcp-candidates'})
                assert not result.isError, result
                result = await session.call_tool('video_candidate_page', {'manifest':'work/mcp-candidates/manifest.json'})
                assert not result.isError and 'frame_id' in result.content[0].text
                result = await session.call_tool('video_frame', {'image':'examples/probe.jpg'})
            else:
                assert 'video_compile' not in names
                result = await session.call_tool('robot_status', {})
                assert not result.isError, result
                result = await session.call_tool('simulation_observe', {'output':'mcp-frame'})
            assert not result.isError, result
            import base64
            pixels = next(c for c in result.content if c.type == 'image')
            with Image.open(io.BytesIO(base64.b64decode(pixels.data))) as img:
                size = list(img.size)
            if kind == 'robot':
                result = await session.call_tool('robot_stop', {})
                assert not result.isError, result
            return {'mcp_tools':sorted(names), 'image_dimensions':size, 'mcp_passed':True}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('kind', choices=['video','robot'])
    p.add_argument('--kit', type=Path, required=True)
    p.add_argument('--work', type=Path, required=True)
    p.add_argument('--port', type=int, default=8819)
    a = p.parse_args()
    a.kit = a.kit.resolve(); a.work = a.work.resolve()
    a.work.mkdir(parents=True, exist_ok=False)
    opposite = 'piperx_middleware' if a.kind == 'video' else 'piperlab'
    assert importlib.util.find_spec(opposite) is None, 'Opposite distribution unexpectedly installed'
    module = 'piperlab.video_entry' if a.kind == 'video' else 'piperx_middleware.standalone_cli'
    result = {'kind':a.kind,'opposite_distribution_absent':True,'python':sys.executable}
    process = None
    try:
        if a.kind == 'video':
            subprocess.run([sys.executable,'-m',module,'candidates','--video',str(a.kit/'examples/transfer.mp4'),
                            '--output',str(a.work/'cli-candidates')],check=True,stdout=subprocess.DEVNULL)
            from piperlab.perception.detector_onnx import TrackedDetector
            detector = TrackedDetector(a.kit/'models/yolo11n.onnx')
            result['onnx_loaded'] = True
        else:
            log = (a.work/'server.log').open('w')
            process = subprocess.Popen([sys.executable,'-m',module,'sim','start','--root',str(a.work/'sim'),
                                        '--port',str(a.port),'--seed','200'],stdout=log,stderr=log,
                                       creationflags=subprocess.CREATE_NO_WINDOW)
            import httpx
            from piperx_middleware.client import RobotClient
            deadline = time.monotonic()+40
            while time.monotonic()<deadline:
                if process.poll() is not None:
                    raise RuntimeError('Simulator exited; see server.log')
                try:
                    response = httpx.get(f'http://127.0.0.1:{a.port}/health',timeout=1)
                    if response.status_code < 500: break
                except httpx.HTTPError: pass
                time.sleep(.25)
            else: raise TimeoutError('Simulator did not listen')
            subprocess.run([sys.executable,'-m',module,'--root',str(a.work/'sim'),'--url',f'http://127.0.0.1:{a.port}','connect'],
                           check=True,stdout=subprocess.DEVNULL)
            subprocess.run([sys.executable,'-m',module,'observe','--root',str(a.work/'sim'),'--url',f'http://127.0.0.1:{a.port}',
                            '--workspace',str(a.work),'--output','cli-frame'],check=True,stdout=subprocess.DEVNULL)
            import numpy as np
            result['depth_shape'] = list(np.load(a.work/'cli-frame/depth.npy',allow_pickle=False).shape)
            result['cli_image_dimensions'] = list(Image.open(a.work/'cli-frame/rgb.jpg').size)
        result.update(asyncio.run(asyncio.wait_for(protocol(a.kind,a.kit,a.work,a.port),timeout=75)))
        (a.work/'validation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        print(json.dumps(result))
    finally:
        if process is not None:
            subprocess.run([sys.executable,'-m',module,'--root',str(a.work/'sim'),'--url',f'http://127.0.0.1:{a.port}','shutdown'],
                           stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=10)
            try: process.wait(timeout=10)
            except subprocess.TimeoutExpired: process.terminate()


if __name__ == '__main__':
    main()
