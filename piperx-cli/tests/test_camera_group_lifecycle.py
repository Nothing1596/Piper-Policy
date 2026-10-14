"""Additional integration review cases against real camera-only child processes."""
import asyncio
import json
import sys

import pytest

from piperx_middleware.live_camera import CameraGroup
from test_live_video import SYNTHETIC_MULTI_WORKER


def setup(tmp_path,monkeypatch,slow=False):
    (tmp_path/'camera.json').write_text(json.dumps({'cameras':[
        {'serial':'a','role':'fixed'},{'serial':'b','role':'wrist'}]}))
    original=asyncio.create_subprocess_exec
    launched=[]
    async def create(*args,**kwargs):
        proc=await original(sys.executable,'-c', 'import time;time.sleep(60)' if slow else SYNTHETIC_MULTI_WORKER,
            *map(str,args[2:]),**kwargs)
        launched.append((proc,args))
        return proc
    monkeypatch.setattr(asyncio,'create_subprocess_exec',create)
    return CameraGroup(tmp_path),launched


@pytest.mark.asyncio
async def test_sequential_start_budget_and_serializable_status(tmp_path,monkeypatch):
    group,launched=setup(tmp_path,monkeypatch)
    try:
        json.dumps(await group.start('a'),allow_nan=False)
        json.dumps(await group.start('b'),allow_nan=False)
        await group.start()
        assert len(launched)==2
        assert sum(int(args[args.index('--max-bytes')+1]) for _,args in launched)<=128*1024*1024
        frame=await group.observe(max_frames=1,roi=[5,5,30,30])
        json.dumps(frame,allow_nan=False)
        labels=[json.loads(p['text']) for p in group.image_inputs(frame) if p['type']=='text']
        assert {(p['camera_serial'],p['view']) for p in labels} >= {
            ('a','full RGB'),('b','full RGB'),('a','ROI crop'),('b','ROI crop')}
    finally:await group.close()
    assert all(p.returncode is not None for p,_ in launched)


@pytest.mark.asyncio
async def test_cancel_start_reaps_both_children(tmp_path,monkeypatch):
    group,launched=setup(tmp_path,monkeypatch,slow=True)
    task=asyncio.create_task(group.start())
    while len(launched)<2:await asyncio.sleep(.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):await task
    await group.close()
    assert all(p.returncode is not None for p,_ in launched)


@pytest.mark.asyncio
async def test_cancel_observe_reaps_all(tmp_path,monkeypatch):
    group,launched=setup(tmp_path,monkeypatch)
    await group.start()
    calls=[]
    for worker in group._workers.values():
        original=worker._request
        async def slow(request,original=original):
            if request.get('command')=='observe':
                calls.append(1);await asyncio.sleep(60)
            return await original(request)
        worker._request=slow
    task=asyncio.create_task(group.observe())
    while len(calls)<2:await asyncio.sleep(.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):await task
    await group.close()
    assert all(p.returncode is not None for p,_ in launched)


@pytest.mark.asyncio
@pytest.mark.parametrize('config',[
    {'cameras':[]}, {'cameras':[{'serial':'a'},{'serial':'a'}]},
    {'cameras':[{'serial':'a'},{'role':'wrist'}]},
    {'cameras':[{'serial':'../outside'}]}, {'max_bytes':129*1024*1024}])
async def test_invalid_group_configuration_starts_nothing(tmp_path,config):
    (tmp_path/'camera.json').write_text(json.dumps(config))
    group=CameraGroup(tmp_path)
    with pytest.raises(ValueError):await group.start()
    assert group.proc is None
