"""Frontend-owned camera lifetime and best-effort robot-state alignment."""
import asyncio
import base64
from collections import deque
import json
from pathlib import Path
import sys
import time
import uuid


def compact_state(state):
    """Keep useful control evidence; omit megabytes of counters and credentials."""
    if 'error' in state:
        return state
    result = {k: state[k] for k in ('backend', 'connection', 'ready', 'active_job_id', 'fault', 'tcp',
                                    'observation_identity', 'instance_id', 'connection_epoch') if k in state}
    robot = state.get('robot')
    if isinstance(robot, dict):
        result['robot'] = {k: v for k, v in robot.items() if k not in ('diagnostics', 'motor_telemetry', 'received_frames', 'tx_frames')}
    return result


def _validate_serial(serial):
    if serial is None:
        return
    if not isinstance(serial, str) or not serial:
        raise ValueError(f"Invalid camera serial: {serial!r}")
    if '/' in serial or '\\' in serial or '..' in serial or serial.startswith('.'):
        raise ValueError(f"Unsafe camera serial path component: {serial!r}")
    import re
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', serial):
        raise ValueError('Unsafe camera serial path component')
    p = Path(serial)
    if p.is_absolute() or p.name != serial:
        raise ValueError(f"Unsafe camera serial path component: {serial!r}")


def _align_frame(frame, states):
    stamp = frame['received_mono_s']
    match = min(states, key=lambda s: abs(s[0] - stamp), default=None)
    frame['robot_alignment'] = {
        'method': 'nearest host HTTP receipt; not hardware synchronized',
        'offset_s': match[0] - stamp if match else None,
        'request_latency_s': match[2] if match else None,
        'state': match[1] if match and abs(match[0] - stamp) + match[2] <= .5 else None
    }


class LiveCamera:
    def __init__(self, root, state_reader=None, *, _config=None):
        self.root = Path(root)
        self.state_reader = state_reader
        self._config = _config
        self.role = _config.get('role', 'fixed') if _config else 'fixed'
        self.calibration_file = _config.get('calibration_file') if _config else None
        self.proc = None
        self._lock = asyncio.Lock()
        self._states = deque(maxlen=240)
        self._state_task = None
        self._stderr_task = None
        self._stderr_tail = ''
        self._serial = _config.get('serial') if _config else None
        self._image_paths = deque(maxlen=300)
        self.error = None

    def config(self):
        if self._config is not None:
            return dict(self._config)
        path = self.root / 'camera.json'
        return json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}

    async def start(self, serial=None):
        _validate_serial(serial)
        async with self._lock:
            config = self.config()
            serial = serial or config.get('serial') or self._serial
            _validate_serial(serial)
            if self.proc and self.proc.returncode is None:
                if serial and serial != self._serial:
                    raise ValueError('Stop /video before changing camera')
                return await self._request({'command': 'status'})
            await self._close()
            self.proc = await asyncio.create_subprocess_exec(
                config.get('python', sys.executable),
                str(Path(__file__).with_name('camera_stream.py')),
                *(['--serial', serial] if serial else []),
                '--sample-hz', str(config.get('sample_hz', 3)),
                '--buffer-s', str(config.get('buffer_s', 20)),
                '--max-bytes', str(config.get('max_bytes', 128 * 1024 * 1024)),
                stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            self._stderr_tail = ''
            self._stderr_task = asyncio.create_task(self._read_stderr(self.proc))
            try:
                result = await self._read(20)
                if result.get('error'):
                    msg = result['error'].get('message') if isinstance(result['error'], dict) else str(result['error'])
                    raise RuntimeError(msg)
                self._serial = result.get('camera_serial') or serial
                self.error = None
                self._states.clear()
                if self.state_reader:
                    self._state_task = asyncio.create_task(self._poll_states())
                if hasattr(self, 'role') and self.role:
                    result['role'] = self.role
                return result
            except BaseException as exc:
                self.error = str(exc)
                await self._close()
                raise

    async def _read(self, timeout=10):
        line = await asyncio.wait_for(self.proc.stdout.readline(), timeout)
        if not line:
            if self._stderr_task:
                try:
                    await asyncio.wait_for(asyncio.shield(self._stderr_task), .2)
                except asyncio.TimeoutError:
                    pass
            raise RuntimeError('Camera worker exited: ' + self._stderr_tail[-1200:])
        return json.loads(line)

    async def _request(self, request):
        if not self.proc or self.proc.returncode is not None:
            raise RuntimeError('Start /video or /observe first')
        try:
            self.proc.stdin.write((json.dumps(request, allow_nan=False) + '\n').encode('utf-8'))
            await self.proc.stdin.drain()
            return await self._read()
        except BaseException:
            # A timed-out response must never become the next request's reply.
            await self._close()
            raise

    async def status(self):
        async with self._lock:
            if not self.proc or self.proc.returncode is not None:
                if self.error:
                    return {'status': 'failed', 'error': self.error, 'camera_serial': self._serial, 'role': getattr(self, 'role', 'fixed')}
                return {'status': 'stopped', 'camera_serial': self._serial, 'role': getattr(self, 'role', 'fixed')}
            st = await self._request({'command': 'status'})
            if hasattr(self, 'role') and self.role:
                st['role'] = self.role
            return st

    async def observe(self, *, window_s=2, max_frames=3, roi=None, since_mono_s=None, serial=None):
        await self.start(serial)
        async with self._lock:
            config = self.config()
            output = Path(config.get('output_root', self.root / 'observations')) / uuid.uuid4().hex
            result = await self._request({
                'command': 'observe', 'output': str(output),
                'window_s': window_s, 'max_frames': max_frames, 'roi': roi, 'since_mono_s': since_mono_s
            })
            if 'error' not in result:
                for frame in result['frames']:
                    self._image_paths.append(frame['rgb_path'])
                    if frame.get('crop_path'):
                        self._image_paths.append(frame['crop_path'])
                    _align_frame(frame, self._states)
                (self.root / 'last_observation.json').write_text(json.dumps({'path': result['observation_path']}), encoding='utf-8')
                Path(result['bundle_path']).write_text(json.dumps(result, indent=2), encoding='utf-8')
            return result

    def image_inputs(self, result):
        bundles = [result] if result.get('schema') == 'piper.observation.v1' else []
        evidence = result.get('visual_evidence') or {}
        bundles += [evidence[k] for k in ('before_request', 'after') if isinstance(evidence.get(k), dict)]
        frames = [frame for bundle in bundles for frame in bundle.get('frames', [])]
        if len(frames) > 3:
            frames = [frames[0], frames[len(frames) // 2], frames[-1]]
        content = []
        budget = 3
        for frame in frames:
            if len(content) // 2 >= budget:
                break
            rgb = frame.get('rgb_path')
            if rgb and rgb in self._image_paths:
                f = Path(rgb)
                if f.is_file() and f.stat().st_size <= 2 * 1024 * 1024:
                    content.append({'type': 'text', 'text': json.dumps({
                        'role': frame.get('role'), 'sequence': frame.get('sequence'), 'captured_at': frame.get('captured_at'),
                        'view': 'full RGB'})})
                    content.append({'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,' + base64.b64encode(f.read_bytes()).decode('ascii')}})
            crop = frame.get('crop_path')
            if crop and crop in self._image_paths and len(content) // 2 < budget:
                f = Path(crop)
                if f.is_file() and f.stat().st_size <= 2 * 1024 * 1024:
                    content.append({'type': 'text', 'text': json.dumps({
                        'role': frame.get('role'), 'sequence': frame.get('sequence'), 'captured_at': frame.get('captured_at'),
                        'view': 'ROI crop'})})
                    content.append({'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,' + base64.b64encode(f.read_bytes()).decode('ascii')}})
        return content

    async def _poll_states(self):
        while True:
            try:
                requested = time.monotonic()
                state = await asyncio.wait_for(self.state_reader(), 1.)
                received = time.monotonic()
                if isinstance(state, dict) and 'error' not in state:
                    self._states.append((received, compact_state(state), received - requested))
            except asyncio.CancelledError:
                raise
            except Exception:
                pass  # Missing state is represented as null, never substituted.
            await asyncio.sleep(.25)

    async def _read_stderr(self, proc):
        while True:
            chunk = await proc.stderr.read(1024)
            if not chunk:
                return
            self._stderr_tail = (self._stderr_tail + chunk.decode('utf-8', errors='replace'))[-1200:]

    async def _close(self):
        if self._state_task:
            self._state_task.cancel()
            await asyncio.gather(self._state_task, return_exceptions=True)
            self._state_task = None
        proc, self.proc = self.proc, None
        self._states.clear()
        if proc:
            if proc.returncode is None:
                proc.stdin.close()  # EOF asks the SDK owner to stop and release.
                try:
                    await asyncio.wait_for(proc.wait(), 5)
                except asyncio.TimeoutError:
                    if proc.returncode is None:
                        proc.kill()
                    await proc.wait()
            else:
                await proc.wait()
        if self._stderr_task:
            try:
                await asyncio.wait_for(asyncio.shield(self._stderr_task), 1)
            except asyncio.TimeoutError:
                self._stderr_task.cancel()
            await asyncio.gather(self._stderr_task, return_exceptions=True)
            self._stderr_task = None

    async def close(self):
        async with self._lock:
            await self._close()


class CameraGroup:
    def __init__(self, root, state_reader=None):
        self.root = Path(root)
        self.state_reader = state_reader
        self.group_id = uuid.uuid4().hex
        self._lock = asyncio.Lock()
        self._workers = {}  # serial -> LiveCamera
        self._states = deque(maxlen=240)
        self._state_task = None
        self._image_paths = deque(maxlen=600)

    @property
    def proc(self):
        for w in self._workers.values():
            if w.proc and w.proc.returncode is None:
                return w.proc
        return None

    def config(self):
        path = self.root / 'camera.json'
        return json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}

    async def _poll_states(self):
        return await LiveCamera._poll_states(self)

    def _parse_camera_configs(self, cfg):
        cameras = cfg.get('cameras')
        if cameras is not None:
            if not isinstance(cameras, list):
                raise ValueError("cameras must be a list")
            if len(cameras) == 0:
                raise ValueError("cameras list cannot be empty")
            if len(cameras) > 2:
                raise ValueError("cameras list exceeds maximum of 2")
            result, seen = [], set()
            for cam in cameras:
                if not isinstance(cam, dict):
                    raise ValueError("camera entry must be a dictionary")
                serial = cam.get('serial')
                if serial is None:
                    raise ValueError('Camera list entries require explicit serials')
                _validate_serial(serial)
                if serial is not None:
                    if serial in seen:
                        raise ValueError(f"Duplicate camera serial: {serial!r}")
                    seen.add(serial)
                role = cam.get('role', 'fixed')
                if role not in ('fixed', 'wrist'):
                    raise ValueError(f"role must be 'fixed' or 'wrist', got {role!r}")
                result.append({'serial': serial, 'role': role, 'calibration_file': cam.get('calibration_file')})
            if len(cameras) == 2 and cameras[0].get('serial') is None and cameras[1].get('serial') is None:
                raise ValueError("Duplicate camera serial: ambiguous specification")
            return result
        serial = cfg.get('serial')
        _validate_serial(serial)
        return [{'serial': serial, 'role': 'fixed', 'calibration_file': None}]

    async def start(self, serial=None):
        _validate_serial(serial)
        async with self._lock:
            cfg = self.config()
            configured = self._parse_camera_configs(cfg)
            capacity = max(1, len(configured))
            total_budget = int(cfg.get('max_bytes', 128 * 1024 * 1024))
            if not 0 < total_budget <= 128 * 1024 * 1024:
                raise ValueError('Camera group budget must be within 128 MiB')

            if serial is not None:
                matched = [c for c in configured if c['serial'] == serial]
                if not matched:
                    if any(c['serial'] is not None for c in configured):
                        raise ValueError(f"Camera {serial} unavailable; no fallback to other camera")
                    matched = [{'serial': serial, 'role': 'fixed', 'calibration_file': None}]
                target_configs = matched
            else:
                target_configs = configured

            claimed = {c['serial'] for c in target_configs if c['serial'] is not None}
            workers_to_run = []
            for c in target_configs:
                s = c['serial']
                worker = self._workers.get(s)
                if worker is None and s is None:
                    for exist_s, exist_w in self._workers.items():
                        if exist_s not in claimed:
                            worker, s = exist_w, exist_s
                            break
                workers_to_run.append((worker, s, c))

            running_allocated = sum(
                getattr(w, '_allocated_bytes', 0)
                for w in self._workers.values()
                if w.proc and w.proc.returncode is None
            )

            run_list = []
            for worker, s, c in workers_to_run:
                if worker is not None and worker.proc and worker.proc.returncode is None:
                    run_list.append(worker)
                    continue
                rem_budget = max(0, total_budget - running_allocated)
                worker_budget = min(total_budget // capacity, rem_budget)
                if worker_budget <= 0:
                    raise ValueError("Camera memory budget exhausted")
                cam_cfg = {**cfg, 'serial': s, 'role': c['role'], 'calibration_file': c.get('calibration_file'), 'max_bytes': worker_budget}
                if worker is None:
                    worker = LiveCamera(self.root, state_reader=None, _config=cam_cfg)
                    self._workers[s] = worker
                else:
                    worker._config = cam_cfg
                    worker.role = c['role']
                    worker.calibration_file = c.get('calibration_file')
                worker._allocated_bytes = worker_budget
                running_allocated += worker_budget
                run_list.append(worker)

            try:
                tasks = [w.start(serial=w._serial or c['serial']) for w, c in zip(run_list, target_configs)]
                results = await asyncio.gather(*tasks, return_exceptions=True)
            except BaseException:
                await asyncio.gather(*(w.close() for w in run_list), return_exceptions=True)
                raise

            failures = [(w, r) for w, r in zip(run_list, results) if isinstance(r, Exception)]
            if failures:
                await asyncio.gather(*(w.close() for w in run_list), return_exceptions=True)
                failed_w, exc = failures[0]
                raise RuntimeError(f"Camera {failed_w._serial} unavailable; no fallback to other camera: {exc}")

            for w in run_list:
                if None in self._workers and w._serial is not None:
                    del self._workers[None]
                    self._workers[w._serial] = w
                elif w._serial:
                    self._workers[w._serial] = w

            if self.state_reader and self._state_task is None:
                self._states.clear()
                self._state_task = asyncio.create_task(self._poll_states())

            if len(run_list) == 1:
                cam_res = dict(results[0])
                cam_res['role'] = run_list[0].role
                res = dict(cam_res)
                res['cameras'] = {run_list[0]._serial: dict(cam_res)}
                return res
            cam_statuses = {w._serial: res for w, res in zip(run_list, results)}
            return {'stream_id': self.group_id, 'status': 'streaming',
                    'cameras': cam_statuses, 'camera_serial': [w._serial for w in run_list]}

    async def status(self, serial=None):
        _validate_serial(serial)
        async with self._lock:
            if not self._workers:
                return {'status': 'stopped'}
            if serial is not None:
                worker = self._workers.get(serial)
                return await worker.status() if worker else {'status': 'failed', 'error': f'Camera {serial} unavailable', 'camera_serial': serial}

            workers = list(self._workers.values())
            statuses = await asyncio.gather(*(w.status() for w in workers), return_exceptions=True)
            cam_statuses = {
                w._serial: ({'status': 'failed', 'error': str(st), 'camera_serial': w._serial, 'role': w.role}
                            if isinstance(st, Exception) else st)
                for w, st in zip(workers, statuses)
            }
            if len(workers) == 1:
                st = dict(list(cam_statuses.values())[0])
                st['cameras'] = cam_statuses
                return st
            all_streaming = len(cam_statuses) > 0 and all(s.get('status') == 'streaming' for s in cam_statuses.values())
            any_error = next((s.get('error') for s in cam_statuses.values() if s.get('error')), None)
            return {'status': 'streaming' if all_streaming else 'failed', 'stream_id': self.group_id,
                    'error': any_error, 'cameras': cam_statuses, 'camera_serial': list(cam_statuses.keys())}

    async def close(self):
        async with self._lock:
            if self._state_task:
                self._state_task.cancel()
                await asyncio.gather(self._state_task, return_exceptions=True)
                self._state_task = None
            self._states.clear()
            await asyncio.gather(*(w.close() for w in self._workers.values()), return_exceptions=True)
            self._workers.clear()

    async def observe(self, *, window_s=2, max_frames=3, roi=None, since_mono_s=None, serial=None):
        _validate_serial(serial)
        await self.start(serial)
        async with self._lock:
            cfg = self.config()
            if serial is not None:
                worker = self._workers.get(serial)
                if not worker or worker.error or not worker.proc or worker.proc.returncode is not None:
                    raise RuntimeError(f"Camera {serial} unavailable; no fallback to other camera")
                target_workers = [worker]
            else:
                target_workers = list(self._workers.values())
                if not target_workers:
                    raise RuntimeError("No camera workers running")
                for w in target_workers:
                    if w.error or not w.proc or w.proc.returncode is not None:
                        raise RuntimeError(f"Camera {w._serial} unavailable; no fallback to other camera")

            group_id = uuid.uuid4().hex
            group_dir = Path(cfg.get('output_root', self.root / 'observations')) / group_id
            group_dir.mkdir(parents=True, exist_ok=False)

            try:
                tasks = [
                    w._request({
                        'command': 'observe', 'output': str(group_dir / str(w._serial or 'camera')),
                        'window_s': window_s, 'max_frames': max_frames, 'roi': roi, 'since_mono_s': since_mono_s
                    })
                    for w in target_workers
                ]
                results = await asyncio.gather(*tasks)
            except BaseException:
                await asyncio.gather(*(w.close() for w in target_workers), return_exceptions=True)
                raise

            for w, r in zip(target_workers, results):
                if isinstance(r, dict) and r.get('error'):
                    raise RuntimeError(f"Camera {w._serial} capture failed: {r['error']}; no fallback to other camera")

            all_frames, per_cam_coverage = [], {}
            for w, r in zip(target_workers, results):
                per_cam_coverage[w._serial] = r.get('coverage', {})
                for frame in r.get('frames', []):
                    self._image_paths.append(frame['rgb_path'])
                    if frame.get('crop_path'):
                        self._image_paths.append(frame['crop_path'])
                    _align_frame(frame, self._states)
                    frame['camera_role'] = w.role
                    frame['camera_serial'] = w._serial
                    frame['serial'] = w._serial
                    frame['source'] = w._serial
                    frame['selection_role'] = frame.get('role')
                    if w.role:
                        frame['role'] = w.role
                    obs_path = Path(frame['rgb_path']).parent / 'observation.json'
                    if obs_path.exists():
                        obs_path.write_text(json.dumps(frame, indent=2), encoding='utf-8')
                    all_frames.append(frame)

            if not all_frames:
                raise RuntimeError("No frames captured")
            all_frames.sort(key=lambda f: f.get('received_mono_s', 0.))

            covs = list(per_cam_coverage.values())
            coverage = dict(results[0].get('coverage', {})) if len(target_workers) == 1 else {
                'cameras': per_cam_coverage,
                'history_truncated': any(c.get('history_truncated') for c in covs),
                'latest_age_s': max((c.get('latest_age_s', 0.) for c in covs), default=0.),
                'first_mono_s': min((c.get('first_mono_s', float('inf')) for c in covs), default=0.),
                'last_mono_s': max((c.get('last_mono_s', 0.) for c in covs), default=0.),
                **per_cam_coverage
            }
            if len(target_workers) == 1:
                coverage['cameras'] = per_cam_coverage

            storyboard_path = self._build_storyboard(group_dir, [r.get('storyboard_path') for r in results if r.get('storyboard_path')])
            most_recent_frame = all_frames[-1]
            most_recent_obs_path = str(Path(most_recent_frame['rgb_path']).parent / 'observation.json')
            source = [w._serial for w in target_workers] if len(target_workers) > 1 else target_workers[0]._serial

            bundle = {
                'schema': 'piper.observation.v1', 'stream_id': self.group_id, 'group_id': group_id,
                'source': source, 'frames': all_frames, 'coverage': coverage,
                'storyboard_path': storyboard_path,
                'selection': 'endpoints + spaced visual changes; camera motion is not compensated',
                'contact_verified': False, 'calibration_applied': False, 'robot_motion': False,
                'observation_path': most_recent_obs_path, 'bundle_path': str(group_dir / 'bundle.json')
            }
            (self.root / 'last_observation.json').write_text(json.dumps({'path': most_recent_obs_path}), encoding='utf-8')
            Path(bundle['bundle_path']).write_text(json.dumps(bundle, indent=2), encoding='utf-8')
            return bundle

    def _build_storyboard(self, group_dir, storyboard_paths):
        valid_paths = [Path(p) for p in storyboard_paths if p and Path(p).is_file()]
        if not valid_paths:
            return None
        if len(valid_paths) == 1:
            return str(valid_paths[0])
        try:
            import cv2, numpy as np
            imgs = [im for im in (cv2.imread(str(p)) for p in valid_paths) if im is not None]
            if imgs:
                max_w = max(im.shape[1] for im in imgs)
                resized = [cv2.resize(im, (max_w, int(im.shape[0] * max_w / im.shape[1]))) for im in imgs]
                out_sb = group_dir / 'storyboard.jpg'
                cv2.imwrite(str(out_sb), np.concatenate(resized, axis=0))
                return str(out_sb)
        except Exception:
            pass
        return str(valid_paths[0])

    def image_inputs(self, result):
        bundles = [result] if result.get('schema') == 'piper.observation.v1' else []
        evidence = result.get('visual_evidence') or {}
        bundles += [evidence[k] for k in ('before_request', 'after') if isinstance(evidence.get(k), dict)]
        frames = [frame for bundle in bundles for frame in bundle.get('frames', [])]
        if not frames:
            return []

        by_camera = {}
        for f in frames:
            src = f.get('camera_serial') or f.get('serial') or f.get('source') or 'default'
            by_camera.setdefault(src, []).append(f)

        sorted_sources = sorted(by_camera.keys())
        max_images = 6
        selected_items = []

        def frame_items(f):
            items = []
            for is_crop, key in ((False, 'rgb_path'), (True, 'crop_path')):
                p = f.get(key)
                if p and p in self._image_paths:
                    file = Path(p)
                    if file.is_file() and file.stat().st_size <= 2 * 1024 * 1024:
                        items.append((f, is_crop, p))
            return items

        latest_per_cam = {}
        for src in sorted_sources:
            cam_frames = sorted(by_camera[src], key=lambda x: x.get('received_mono_s', 0.))
            latest = cam_frames[-1]
            latest_per_cam[src] = latest
            selected_items.extend(frame_items(latest))

        if len(selected_items) < max_images:
            for src in sorted_sources:
                cam_frames = sorted(by_camera[src], key=lambda x: x.get('received_mono_s', 0.))
                earlier = [f for f in cam_frames if f is not latest_per_cam[src]]
                if len(earlier) > 2:
                    earlier = [earlier[0], earlier[-1]]
                for f in earlier:
                    for it in frame_items(f):
                        if len(selected_items) < max_images:
                            selected_items.append(it)

        selected_items = selected_items[:max_images]
        selected_items.sort(key=lambda it: (
            str(it[0].get('camera_serial') or it[0].get('serial') or it[0].get('source')),
            it[0].get('received_mono_s', 0.),
            it[0].get('sequence', 0),
            1 if it[1] else 0
        ))

        content = []
        for frame, is_crop, path in selected_items:
            file = Path(path)
            content.append({'type': 'text', 'text': json.dumps({
                'role': frame.get('role'),
                'camera_role': frame.get('camera_role'),
                'camera_serial': frame.get('camera_serial') or frame.get('serial'),
                'sequence': frame.get('sequence'),
                'captured_at': frame.get('captured_at'),
                'view': 'ROI crop' if is_crop else 'full RGB'
            })})
            content.append({'type': 'image_url', 'image_url': {
                'url': 'data:image/jpeg;base64,' + base64.b64encode(file.read_bytes()).decode('ascii')
            }})
        return content
