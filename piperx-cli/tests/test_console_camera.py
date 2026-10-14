import asyncio
import json
from types import SimpleNamespace
import numpy as np
import pytest
from piperx_middleware import camera_capture, console_camera
from piperx_middleware.console_interaction import InteractiveConsoleController


def test_pixel_invalid_depth_and_bounds(tmp_path):
    depth = tmp_path / 'depth.npy'
    np.save(depth, np.array([[0, 1000], [2000, 3000]], dtype=np.uint16))
    obs = tmp_path / 'observation.json'
    obs.write_text(json.dumps({'intrinsics': {'width': 2, 'height': 2, 'fx': 2, 'fy': 2,
        'ppx': 0, 'ppy': 0, 'coeffs': [0]*5}, 'depth_path': str(depth), 'depth_scale_m': .001,
        'captured_at': 'test', 'camera_serial': 'test'}))
    assert camera_capture.pixel(obs, 1, 0)['xyz_m'] == [.5, 0., 1.]
    with pytest.raises(ValueError, match='No valid depth'):
        camera_capture.pixel(obs, 0, 0)
    with pytest.raises(ValueError, match='outside'):
        camera_capture.pixel(obs, -1, 0)


@pytest.mark.asyncio
async def test_failed_capture_keeps_previous_observation(tmp_path, monkeypatch):
    last = tmp_path / 'last_observation.json'
    last.write_text('{"path":"previous"}')
    async def failed(*args): return {'error': 'camera busy'}
    monkeypatch.setattr(console_camera, 'worker', failed)
    assert 'error' in await console_camera.observe(tmp_path)
    assert json.loads(last.read_text())['path'] == 'previous'


@pytest.mark.asyncio
async def test_cancel_reaps_camera_worker(monkeypatch):
    class Proc:
        returncode = None
        killed = False
        async def communicate(self):
            if not self.killed: raise asyncio.CancelledError()
            self.returncode = -1
            return b'', b''
        def kill(self): self.killed = True
    proc = Proc()
    async def create(*args, **kwargs): return proc
    monkeypatch.setattr(asyncio, 'create_subprocess_exec', create)
    with pytest.raises(asyncio.CancelledError): await console_camera.worker({}, [])
    assert proc.killed and proc.returncode == -1


@pytest.mark.asyncio
async def test_cli_blocks_remote_camera_substitution():
    messages=[]
    controller=object.__new__(InteractiveConsoleController)
    controller._draining=False
    controller.mode='real'; controller.target='remote';controller.emit=messages.append
    assert await controller.handle_line('/observe')
    assert 'no remote/local camera substitution' in messages[-1]
