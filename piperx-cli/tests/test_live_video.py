"""Live observation tests use an explicitly injected synthetic source, never CAN."""
import asyncio
import io
import json
from pathlib import Path
import sys
import time

import numpy as np
import pytest

from piperx_middleware.live_frames import Frame, FrameRing, select
from piperx_middleware.live_camera import CameraGroup, LiveCamera, compact_state


def frame(sequence, stamp, gray=0):
    return Frame(sequence, stamp, {}, b'jpeg', np.zeros((8,8),np.uint16), np.full((8,8),gray,np.uint8), 1.)


def test_ring_budgets_and_truncated_coverage():
    ring = FrameRing(seconds=2,max_frames=3,max_bytes=600)
    for i in range(10):ring.add(frame(i,float(i)))
    assert len(ring.frames)<=3 and ring.bytes<=600 and ring.dropped>0
    _, coverage = select(list(ring.frames),since_mono_s=0,now_mono_s=9)
    assert coverage['history_truncated']


def test_change_selection_and_static_dedup():
    frames = [frame(i,float(i),0 if i<3 else 200) for i in range(7)]
    chosen, _ = select(frames,since_mono_s=0,now_mono_s=6,max_frames=3)
    assert [(f.sequence,role) for f,role,_ in chosen]==[(0,'before'),(3,'visual_change'),(6,'after')]
    chosen,_=select([frame(i,float(i)) for i in range(7)],since_mono_s=0,now_mono_s=6)
    assert len(chosen)==2
    assert not any(role=='contact' for _,role,_ in chosen)


def test_stale_invalid_region_and_latest():
    frames=[frame(0,0),frame(1,1)]
    with pytest.raises(ValueError,match='camera_stale'):select(frames,since_mono_s=0,now_mono_s=10)
    with pytest.raises(ValueError,match='roi outside'):select(frames,since_mono_s=0,now_mono_s=1,roi=[0,0,9,9])
    picked,_=select(frames,since_mono_s=0,now_mono_s=1,max_frames=1)
    assert picked[0][0].sequence==1


SYNTHETIC_WORKER = '''
import time,numpy as np
from piperx_middleware.camera_stream import Stream,serve
def source(stream):
 i=0
 while not stream.stop_event.wait(.05):
  stamp=time.monotonic()
  stream.add(np.full((48,64,3),i%255,np.uint8),np.full((48,64),1000,np.uint16),
   {'received_mono_s':stamp,'captured_at':str(stamp),'camera_serial':'synthetic-test',
    'frame':'camera_color_optical','calibration_applied':False,'depth_scale_m':.001,
    'intrinsics':{'width':64,'height':48,'fx':50,'fy':50,'ppx':32,'ppy':24,'coeffs':[0]*5}})
  i+=10
serve(Stream(source=source))
'''


@pytest.mark.asyncio
async def test_actual_worker_reuse_eof_cleanup_alignment_and_pixels(tmp_path,monkeypatch):
    pytest.importorskip('cv2')
    create=asyncio.create_subprocess_exec
    launches=[]
    async def fake(*args,**kwargs):
        proc=await create(sys.executable,'-c',SYNTHETIC_WORKER,**kwargs)
        launches.append(proc)
        return proc
    monkeypatch.setattr(asyncio,'create_subprocess_exec',fake)
    async def state():return {'robot':{'q_deg':[1]*6,'feedback_age_s':.01,'diagnostics':{'huge':1}}}
    camera=LiveCamera(tmp_path,state)
    try:
        first=await camera.start()
        await asyncio.sleep(.4)
        a=await camera.observe(roi=[0,0,32,24])
        b=await camera.observe(max_frames=1)
        assert a['stream_id']==b['stream_id']==first['stream_id']
        assert len(launches)==1 and len(a['frames'])<=3
        assert Path(a['storyboard_path']).is_file() and Path(a['frames'][0]['crop_path']).is_file()
        assert b['frames'][0]['robot_alignment']['state']['robot']['q_deg']==[1]*6
        assert 'diagnostics' not in b['frames'][0]['robot_alignment']['state']['robot']
        from piperx_middleware.camera_capture import pixel
        assert pixel(b['observation_path'],32,24)['xyz_m']==[0.,0.,1.]
        assert camera.image_inputs(b)[1]['image_url']['url'].startswith('data:image/jpeg;base64,')
        assert camera.image_inputs({'schema':'piper.observation.v1','frames':[{'rgb_path':'/etc/passwd'}]})==[]
    finally:await camera.close()
    assert launches[0].returncode==0 and camera._state_task is None


@pytest.mark.asyncio
async def test_camera_timeout_reaps_and_cannot_cross_reply_boundary(tmp_path,monkeypatch):
    create=asyncio.create_subprocess_exec
    async def fake(*args,**kwargs):
        return await create(sys.executable,'-c','import time;time.sleep(60)',**kwargs)
    monkeypatch.setattr(asyncio,'create_subprocess_exec',fake)
    camera=LiveCamera(tmp_path)
    read=camera._read
    async def short(timeout=10):return await read(.05)
    monkeypatch.setattr(camera,'_read',short)
    with pytest.raises(asyncio.TimeoutError):await camera.start()
    assert camera.proc is None


def test_state_error_is_preserved():
    assert compact_state({'error':{'code':'transport_unknown'}})['error']['code']=='transport_unknown'


SYNTHETIC_MULTI_WORKER = '''
import argparse, sys, time, numpy as np
from piperx_middleware.camera_stream import Stream, serve
parser = argparse.ArgumentParser()
parser.add_argument('--serial', default='test-cam')
parser.add_argument('--max-bytes', type=int, default=128 * 1024 * 1024)
parser.add_argument('--sample-hz', type=float, default=3)
parser.add_argument('--buffer-s', type=float, default=20)
args, _ = parser.parse_known_args()
serial = args.serial
if serial == 'fail-cam':
    raise RuntimeError('Camera fail-cam unavailable')
def source(stream):
    i = 0
    while not stream.stop_event.wait(.05):
        stamp = time.monotonic()
        stream.add(np.full((48,64,3), i%255, np.uint8), np.full((48,64), 1000, np.uint16),
            {'received_mono_s': stamp, 'captured_at': str(stamp), 'camera_serial': serial,
             'frame': 'camera_color_optical', 'calibration_applied': False, 'depth_scale_m': .001,
             'intrinsics': {'width': 64, 'height': 48, 'fx': 50, 'fy': 50, 'ppx': 32, 'ppy': 24, 'coeffs': [0]*5}})
        i += 10
serve(Stream(serial=serial, source=source, max_bytes=args.max_bytes, sample_hz=args.sample_hz, buffer_s=args.buffer_s))
'''


@pytest.mark.asyncio
async def test_camera_group_two_serials_reuse_budget_merge_and_cleanup(tmp_path, monkeypatch):
    pytest.importorskip('cv2')
    create = asyncio.create_subprocess_exec
    launches = []

    async def fake(*args, **kwargs):
        cli_args = [str(a) for a in args[2:]]
        proc = await create(sys.executable, '-c', SYNTHETIC_MULTI_WORKER, *cli_args, **kwargs)
        launches.append((proc, cli_args))
        return proc

    monkeypatch.setattr(asyncio, 'create_subprocess_exec', fake)

    camera_cfg = {
        'cameras': [
            {'serial': 'cam-fixed', 'role': 'fixed'},
            {'serial': 'cam-wrist', 'role': 'wrist'}
        ]
    }
    (tmp_path / 'camera.json').write_text(json.dumps(camera_cfg), encoding='utf-8')

    async def state():
        return {
            'observation_identity': {'instance_id': 'inst-1', 'connection_epoch': 42},
            'backend': 'connected',
            'robot': {'q_deg': [2] * 6, 'feedback_age_s': 0.02, 'diagnostics': {'huge': 1}}
        }

    group = CameraGroup(tmp_path, state)
    assert group.proc is None

    try:
        start_res = await group.start()
        assert group.proc is not None
        assert start_res['status'] == 'streaming'

        # Global budget: 128 MiB divided across 2 workers -> 64 MiB each
        assert len(launches) == 2
        for _, cli_args in launches:
            assert '--max-bytes' in cli_args
            idx = cli_args.index('--max-bytes')
            assert int(cli_args[idx + 1]) == 64 * 1024 * 1024

        await asyncio.sleep(0.4)

        # First observe with ROI crop
        obs1 = await group.observe(roi=[0, 0, 32, 24])
        # Second observe
        obs2 = await group.observe(max_frames=1)

        # Process reuse: still exactly 2 workers launched (at most 1 process per serial)
        assert len(launches) == 2

        # Bundle format and frame merging
        assert obs1['schema'] == obs2['schema'] == 'piper.observation.v1'
        assert obs1['group_id'] in obs1['bundle_path']
        assert Path(obs1['bundle_path']).is_file()

        # Both cameras present in merged bundle
        serials1 = {f['camera_serial'] for f in obs1['frames']}
        roles1 = {f['camera_role'] for f in obs1['frames']}
        assert serials1 == {'cam-fixed', 'cam-wrist'}
        assert roles1 == {'fixed', 'wrist'}

        # Annotation of role and serial
        for f in obs1['frames']:
            assert f['role'] in ('fixed', 'wrist')
            assert f['source'] in ('cam-fixed', 'cam-wrist')
            assert Path(f['rgb_path']).is_file()
            assert Path(f['depth_path']).is_file()
            assert Path(f['crop_path']).is_file()
            # Actual robot snapshot identity attached, no fabricated extrinsics
            ident = f['robot_alignment']['state']['observation_identity']
            assert ident == {'instance_id': 'inst-1', 'connection_epoch': 42}
            assert f['robot_alignment']['state']['robot']['q_deg'] == [2] * 6
            assert 'diagnostics' not in f['robot_alignment']['state']['robot']

        # Most recent observation path
        assert Path(obs2['observation_path']).is_file()
        from piperx_middleware.camera_capture import pixel
        pix = pixel(obs2['observation_path'], 32, 24)
        assert pix['xyz_m'] == [0., 0., 1.]

        # Coverage preserved per camera
        assert 'cam-fixed' in obs1['coverage']['cameras']
        assert 'cam-wrist' in obs1['coverage']['cameras']

        # Image inputs handles issued paths, full image plus ROI crop (not crop replacing full image)
        imgs = group.image_inputs(obs1)
        assert 1 <= len(imgs) // 2 <= 6
        img_urls = [it['image_url']['url'] for it in imgs if 'image_url' in it]
        assert all(u.startswith('data:image/jpeg;base64,') for u in img_urls)
        # Unknown paths rejected
        assert group.image_inputs({'schema': 'piper.observation.v1', 'frames': [{'rgb_path': '/etc/passwd'}]}) == []

        # Explicit routing to one camera
        obs_fixed = await group.observe(serial='cam-fixed')
        assert len(launches) == 2  # Reused
        assert all(f['camera_serial'] == 'cam-fixed' for f in obs_fixed['frames'])

        # Status check
        st = await group.status()
        assert st['status'] == 'streaming'
        assert st['cameras']['cam-fixed']['status'] == 'streaming'
        assert st['cameras']['cam-wrist']['status'] == 'streaming'

    finally:
        await group.close()

    # Cleanup: EOF mode switch closes all, no orphan processes
    assert group.proc is None
    assert group._state_task is None
    assert launches[0][0].returncode == 0
    assert launches[1][0].returncode == 0


@pytest.mark.asyncio
async def test_camera_group_partial_failure_and_unavailable_no_fallback(tmp_path, monkeypatch):
    create = asyncio.create_subprocess_exec
    launches = []

    async def fake(*args, **kwargs):
        cli_args = [str(a) for a in args[2:]]
        proc = await create(sys.executable, '-c', SYNTHETIC_MULTI_WORKER, *cli_args, **kwargs)
        launches.append(proc)
        return proc

    monkeypatch.setattr(asyncio, 'create_subprocess_exec', fake)

    camera_cfg = {
        'cameras': [
            {'serial': 'cam-ok', 'role': 'fixed'},
            {'serial': 'fail-cam', 'role': 'wrist'}
        ]
    }
    (tmp_path / 'camera.json').write_text(json.dumps(camera_cfg), encoding='utf-8')

    group = CameraGroup(tmp_path)
    with pytest.raises(RuntimeError, match='fail-cam unavailable'):
        await group.start()

    st = await group.status()
    assert st['status'] == 'failed'
    assert st['cameras']['fail-cam']['status'] == 'failed'
    assert 'fail-cam unavailable' in st['cameras']['fail-cam']['error']

    st_fail = await group.status(serial='fail-cam')
    assert st_fail['status'] == 'failed'

    with pytest.raises(RuntimeError, match='fail-cam unavailable; no fallback to other camera'):
        await group.observe()

    with pytest.raises(RuntimeError, match='fail-cam unavailable; no fallback to other camera'):
        await group.observe(serial='fail-cam')

    with pytest.raises(ValueError, match='Camera unknown-cam unavailable'):
        await group.observe(serial='unknown-cam')

    await group.close()
    assert group.proc is None


@pytest.mark.asyncio
async def test_camera_group_config_validation_and_legacy_behavior(tmp_path, monkeypatch):
    create = asyncio.create_subprocess_exec
    launches = []

    async def fake(*args, **kwargs):
        cli_args = [str(a) for a in args[2:]]
        proc = await create(sys.executable, '-c', SYNTHETIC_MULTI_WORKER, *cli_args, **kwargs)
        launches.append((proc, cli_args))
        return proc

    monkeypatch.setattr(asyncio, 'create_subprocess_exec', fake)

    group = CameraGroup(tmp_path)
    res = await group.start()
    assert res['status'] == 'streaming'
    assert len(launches) == 1
    idx = launches[0][1].index('--max-bytes')
    assert int(launches[0][1][idx + 1]) == 128 * 1024 * 1024
    await group.close()

    (tmp_path / 'camera.json').write_text(json.dumps({
        'cameras': [
            {'serial': 'c1', 'role': 'fixed'},
            {'serial': 'c2', 'role': 'wrist'},
            {'serial': 'c3', 'role': 'fixed'}
        ]
    }), encoding='utf-8')
    group2 = CameraGroup(tmp_path)
    with pytest.raises(ValueError, match='exceeds maximum of 2'):
        await group2.start()

    (tmp_path / 'camera.json').write_text(json.dumps({
        'cameras': [{'serial': 'c1', 'role': 'overhead'}]
    }), encoding='utf-8')
    group3 = CameraGroup(tmp_path)
    with pytest.raises(ValueError, match="must be 'fixed' or 'wrist'"):
        await group3.start()


@pytest.mark.asyncio
async def test_camera_group_single_camera_json_serialization_not_circular(tmp_path, monkeypatch):
    create = asyncio.create_subprocess_exec
    async def fake(*args, **kwargs):
        cli_args = [str(a) for a in args[2:]]
        return await create(sys.executable, '-c', SYNTHETIC_MULTI_WORKER, *cli_args, **kwargs)
    monkeypatch.setattr(asyncio, 'create_subprocess_exec', fake)

    group = CameraGroup(tmp_path)
    try:
        res = await group.start()
        # Verify JSON serializability - must not raise ValueError: Circular reference detected
        serialized = json.dumps(res)
        parsed = json.loads(serialized)
        assert parsed['status'] == 'streaming'
        assert 'cameras' in parsed
        assert parsed['cameras'][parsed['camera_serial']]['status'] == 'streaming'
    finally:
        await group.close()


@pytest.mark.asyncio
async def test_camera_group_sequential_serial_budgets_holds_128mib(tmp_path, monkeypatch):
    create = asyncio.create_subprocess_exec
    launches = []
    async def fake(*args, **kwargs):
        cli_args = [str(a) for a in args[2:]]
        proc = await create(sys.executable, '-c', SYNTHETIC_MULTI_WORKER, *cli_args, **kwargs)
        launches.append((proc, cli_args))
        return proc
    monkeypatch.setattr(asyncio, 'create_subprocess_exec', fake)

    camera_cfg = {
        'cameras': [
            {'serial': 'cam-A', 'role': 'fixed'},
            {'serial': 'cam-B', 'role': 'wrist'}
        ]
    }
    (tmp_path / 'camera.json').write_text(json.dumps(camera_cfg), encoding='utf-8')

    group = CameraGroup(tmp_path)
    try:
        # Start A explicitly first
        res_a = await group.start(serial='cam-A')
        assert res_a['status'] == 'streaming'
        assert len(launches) == 1
        idx_a = launches[0][1].index('--max-bytes')
        assert int(launches[0][1][idx_a + 1]) == 64 * 1024 * 1024

        # Sequentially start B explicitly
        res_b = await group.start(serial='cam-B')
        assert res_b['status'] == 'streaming'
        assert len(launches) == 2
        idx_b = launches[1][1].index('--max-bytes')
        assert int(launches[1][1][idx_b + 1]) == 64 * 1024 * 1024

        total_bytes = int(launches[0][1][idx_a + 1]) + int(launches[1][1][idx_b + 1])
        assert total_bytes == 128 * 1024 * 1024

        st = await group.status()
        assert st['status'] == 'streaming'
        assert st['cameras']['cam-A']['status'] == 'streaming'
        assert st['cameras']['cam-B']['status'] == 'streaming'
    finally:
        await group.close()


@pytest.mark.asyncio
async def test_camera_group_cancellation_reaps_inflight_and_no_orphan_sibling(tmp_path, monkeypatch):
    create = asyncio.create_subprocess_exec
    procs = []

    SLOW_SYNTHETIC = '''
import argparse, time
parser = argparse.ArgumentParser()
parser.add_argument('--serial', default='test')
args, _ = parser.parse_known_args()
if args.serial == 'slow-cam':
    time.sleep(10)
from piperx_middleware.camera_stream import Stream, serve
serve(Stream(serial=args.serial, source=lambda s: None))
'''
    async def fake(*args, **kwargs):
        cli_args = [str(a) for a in args[2:]]
        proc = await create(sys.executable, '-c', SLOW_SYNTHETIC, *cli_args, **kwargs)
        procs.append(proc)
        return proc
    monkeypatch.setattr(asyncio, 'create_subprocess_exec', fake)

    camera_cfg = {
        'cameras': [
            {'serial': 'fast-cam', 'role': 'fixed'},
            {'serial': 'slow-cam', 'role': 'wrist'}
        ]
    }
    (tmp_path / 'camera.json').write_text(json.dumps(camera_cfg), encoding='utf-8')

    group = CameraGroup(tmp_path)
    start_task = asyncio.create_task(group.start())
    await asyncio.sleep(0.3)
    start_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await start_task

    await asyncio.sleep(0.2)
    assert len(procs) == 2
    assert procs[0].returncode is not None
    assert procs[1].returncode is not None
    await group.close()


@pytest.mark.asyncio
async def test_camera_group_observe_cancellation_reaps_workers(tmp_path, monkeypatch):
    create = asyncio.create_subprocess_exec
    procs = []
    async def fake(*args, **kwargs):
        cli_args = [str(a) for a in args[2:]]
        proc = await create(sys.executable, '-c', SYNTHETIC_MULTI_WORKER, *cli_args, **kwargs)
        procs.append(proc)
        return proc
    monkeypatch.setattr(asyncio, 'create_subprocess_exec', fake)

    camera_cfg = {
        'cameras': [
            {'serial': 'cam-1', 'role': 'fixed'},
            {'serial': 'cam-2', 'role': 'wrist'}
        ]
    }
    (tmp_path / 'camera.json').write_text(json.dumps(camera_cfg), encoding='utf-8')
    group = CameraGroup(tmp_path)
    try:
        await group.start()
        assert len(procs) == 2
        orig_req = group._workers['cam-2']._request
        async def slow_req(req):
            if req.get('command') == 'observe':
                await asyncio.sleep(10)
            return await orig_req(req)
        monkeypatch.setattr(group._workers['cam-2'], '_request', slow_req)

        obs_task = asyncio.create_task(group.observe(window_s=10))
        await asyncio.sleep(0.05)
        obs_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await obs_task
        await asyncio.sleep(0.2)
        assert procs[0].returncode is not None
        assert procs[1].returncode is not None
    finally:
        await group.close()


@pytest.mark.asyncio
async def test_camera_group_validation_rejects_duplicates_unsafe_paths_and_empty_list(tmp_path):
    (tmp_path / 'camera.json').write_text(json.dumps({'cameras': []}), encoding='utf-8')
    group = CameraGroup(tmp_path)
    with pytest.raises(ValueError, match='cameras list cannot be empty'):
        await group.start()

    (tmp_path / 'camera.json').write_text(json.dumps({
        'cameras': [{'serial': 'cam-1', 'role': 'fixed'}, {'serial': 'cam-1', 'role': 'wrist'}]
    }), encoding='utf-8')
    with pytest.raises(ValueError, match='Duplicate camera serial'):
        await group.start()

    for unsafe_serial in ('../evil', '/etc/passwd', 'foo/bar', '..', '.hidden'):
        (tmp_path / 'camera.json').write_text(json.dumps({
            'cameras': [{'serial': unsafe_serial, 'role': 'fixed'}]
        }), encoding='utf-8')
        with pytest.raises(ValueError, match='Unsafe camera serial path component'):
            await group.start()
        with pytest.raises(ValueError, match='Unsafe camera serial path component'):
            await group.start(serial=unsafe_serial)


@pytest.mark.asyncio
async def test_camera_group_start_explicit_then_no_serial_does_not_duplicate_worker(tmp_path, monkeypatch):
    create = asyncio.create_subprocess_exec
    launches = []
    async def fake(*args, **kwargs):
        cli_args = [str(a) for a in args[2:]]
        proc = await create(sys.executable, '-c', SYNTHETIC_MULTI_WORKER, *cli_args, **kwargs)
        launches.append(proc)
        return proc
    monkeypatch.setattr(asyncio, 'create_subprocess_exec', fake)

    (tmp_path / 'camera.json').write_text(json.dumps({}), encoding='utf-8')
    group = CameraGroup(tmp_path)
    try:
        res_a = await group.start(serial='cam-A')
        assert res_a['status'] == 'streaming'
        assert len(launches) == 1
        assert 'cam-A' in group._workers

        res_auto = await group.start()
        assert res_auto['status'] == 'streaming'
        assert len(launches) == 1
        assert None not in group._workers
        assert len(group._workers) == 1
    finally:
        await group.close()
