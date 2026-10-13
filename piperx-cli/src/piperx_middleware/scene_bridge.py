"""Front-end geometry adapter: trusted capture poses in, bounded CPU evidence out.

The model selects pixels/regions, never extrinsics. No SDK or motion connection.
"""
import asyncio
import hashlib
import json
import os
import time
from pathlib import Path
import sys

import numpy as np


OPS = ['surface_point', 'accumulate_region', 'fit_plane', 'point_distance', 'point_plane_distance']
_ID = {'type': 'string', 'pattern': '^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$'}
_PIXEL = {'type': 'array', 'items': {'type': 'integer', 'minimum': 0}, 'minItems': 2, 'maxItems': 2}
_REGION = {'observation_path': {'type': 'string'},
           'roi': {'type': 'array', 'items': {'type': 'integer', 'minimum': 0}, 'minItems': 4, 'maxItems': 4},
           'polygon': {'type': 'array', 'items': _PIXEL, 'minItems': 3, 'maxItems': 64}}
SCENE_TOOL = {'name': 'scene_query', 'description': 'Compute surface points, local accumulated point clouds, planes or distances from saved RGB-D evidence. CPU only; never moves robot. Extrinsics come from operator calibration. Residuals are not accuracy; unknown space is not free space.',
 'inputSchema': {'type': 'object', 'additionalProperties': False, 'required': ['operation'], 'dependentRequired': {'question_id':['expected_version']}, 'properties': {
    'operation': {'enum': OPS}, 'task_id': _ID, 'question_id': _ID,
    'expected_version': {'type':'integer','minimum':1},
    **_REGION, 'pixel': _PIXEL, 'target_id': _ID, 'batch_id': _ID,
    'observations': {'type': 'array', 'minItems': 1, 'maxItems': 16, 'items': {
        'type': 'object', 'additionalProperties': False, 'required': ['observation_path'], 'properties': _REGION}},
    'point_id': _ID, 'other_point_id': _ID, 'plane_id': _ID,
    'voxel_size_m': {'type': 'number', 'minimum': .001, 'maximum': .05},
    'stride': {'type': 'integer', 'minimum': 1, 'maximum': 16},
    'patch_half_size': {'type': 'integer', 'minimum': 1, 'maximum': 8},
    'reference_normal': {'type': 'array', 'items': {'type': 'number'}, 'minItems': 3, 'maxItems': 3}}}}


def camera_entries(root):
    path = Path(root)/'camera.json'
    config = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
    entries = config.get('cameras', [config] if config.get('serial') else [])
    simulated = Path(root)/'simulation-camera.json'
    if simulated.exists():
        sim_cfg = json.loads(simulated.read_text(encoding='utf-8'))
        sim_entries = sim_cfg.get('cameras', [sim_cfg] if sim_cfg.get('serial') else [])
        for entry in sim_entries:
            if any(e.get('serial') == entry.get('serial') for e in entries):
                raise ValueError('Simulation camera serial conflicts with operator camera configuration')
            entries = [*entries, entry]
    return entries


def calibration(entry):
    path = entry.get('calibration_file')
    if not path:
        raise ValueError('No registered calibration for this camera')
    raw = Path(path).expanduser().read_bytes()
    if len(raw) > 65536:
        raise ValueError('Calibration exceeds 64 KiB')
    data = json.loads(raw)
    if data.get('camera_serial') != entry['serial'] or data.get('role') != entry.get('role'):
        raise ValueError('Calibration camera serial/role mismatch')
    if data.get('units') != 'm':
        raise ValueError('Calibration must declare units m')
    key = 'base_from_camera' if entry['role'] == 'fixed' else 'flange_from_camera'
    matrix = np.asarray(data.get(key), dtype=float)
    if (matrix.shape != (4,4) or not np.isfinite(matrix).all()
            or not np.allclose(matrix[3], [0,0,0,1])
            or not np.allclose(matrix[:3,:3].T @ matrix[:3,:3], np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(matrix[:3,:3]), 1, atol=1e-5)):
        raise ValueError('Calibration requires a proper rigid 4x4 transform')
    return matrix, hashlib.sha256(raw).hexdigest()


def validate_cameras(entries):
    if not isinstance(entries, list) or not 1 <= len(entries) <= 2:
        raise ValueError('Configure one or two cameras')
    serials = set()
    for entry in entries:
        if not isinstance(entry,dict) or set(entry)-{'serial','role','calibration_file'}:
            raise ValueError('Camera fields: serial, role, calibration_file')
        if not isinstance(entry.get('serial'),str) or not entry['serial'].strip() or entry['serial'] in serials:
            raise ValueError('Camera serials must be explicit and unique')
        from .live_camera import _validate_serial
        _validate_serial(entry['serial'])
        serials.add(entry['serial'])
        if entry.get('role') not in ('fixed','wrist'):
            raise ValueError('Camera role must be fixed or wrist')
        if entry.get('calibration_file'):
            calibration(entry)
    return entries


def _safe_observation(root, value):
    root = Path(root).resolve()
    raw = Path(value)
    path = (raw if raw.is_absolute() else root/raw)
    if '..' in path.parts:
        raise ValueError('Observation traversal is not allowed')
    cursor = path
    while cursor.resolve() != root:
        if cursor.is_symlink():
            raise ValueError('Observation symlinks inside the profile are not allowed')
        if cursor == cursor.parent:
            raise ValueError('Observation is outside this profile')
        cursor = cursor.parent
    path = path.resolve(strict=True)
    if not any(path.is_relative_to(root/name) for name in ('observations','tasks')) or path.stat().st_size > 2*1024*1024:
        raise ValueError('Observation must be a bounded file in this profile')
    return path


def annotate_capture(root, bundle):
    """Save same-host alignment and calibration fingerprint alongside each frame."""
    entries = {e.get('serial'): e for e in camera_entries(root)}
    for frame in bundle.get('frames', []):
        try:
            path = _safe_observation(root, str(Path(frame['depth_path']).with_name('observation.json')))
            entry = entries.get(frame.get('camera_serial'))
            if entry:
                _, fingerprint = calibration(entry)
                frame['calibration_id'] = fingerprint
                frame['camera_role'] = entry['role']
            frame['observation_path'] = str(path)
            path.write_text(json.dumps(frame, allow_nan=False), encoding='utf-8')
        except (ValueError, KeyError, OSError) as exc:
            frame['geometry_unavailable'] = str(exc)
    if bundle.get('bundle_path'):
        path = _safe_observation(root, bundle['bundle_path'])
        path.write_text(json.dumps(bundle, allow_nan=False), encoding='utf-8')
    return bundle


class SceneBridge:
    def __init__(self, root, plans):
        self.root, self.plans = Path(root), plans
        self._lock = asyncio.Lock()
        self.proc = None

    def _observation(self, item, *, needs_base):
        item = dict(item)
        path = _safe_observation(self.root, item['observation_path'])
        data = json.loads(path.read_text(encoding='utf-8'))
        if 'frames' in data:
            raise ValueError('Choose a frame observation_path, not a multi-frame bundle')
        entry = next((e for e in camera_entries(self.root) if e.get('serial') == data.get('camera_serial')), None)
        if not entry or not entry.get('calibration_file'):
            if needs_base:
                raise ValueError('Base-frame query requires registered calibration')
            return item
        matrix, fingerprint = calibration(entry)
        if data.get('calibration_id') != fingerprint:
            raise ValueError('Capture calibration missing or changed; collect a new observation')
        align = data.get('robot_alignment') or {}
        state = align.get('state') or {}
        robot = state.get('robot') or {}
        identity = state.get('observation_identity') or {}
        offset, latency = align.get('offset_s'), align.get('request_latency_s')
        age = robot.get('feedback_age_s')
        velocity = robot.get('velocity_deg_s')
        if (not identity.get('instance_id') or identity.get('connection_epoch') is None
                or offset is None or latency is None or not np.isfinite([offset,latency]).all()
                or abs(offset)+latency > .5 or latency < 0
                or age is None or not np.isfinite(age) or not 0 <= age <= .5
                or not isinstance(velocity,list) or len(velocity)!=6
                or not np.isfinite(velocity).all() or max(abs(v) for v in velocity) > .5):
            raise ValueError('Metric geometry requires stationary, fresh robot feedback aligned to capture')
        if entry['role'] == 'wrist':
            q = robot.get('q_deg')
            if not isinstance(q,list) or len(q)!=6 or not np.isfinite(q).all():
                raise ValueError('Wrist observation lacks measured joint angles')
            from .kinematics import PiperKinematics
            matrix = PiperKinematics().matrix(q) @ matrix
        item.update(observation_path=str(path), base_from_camera=matrix.tolist(), calibration_id=fingerprint)
        return item

    def _revalidate(self, result):
        entries = result.get('calibration_provenance', [])
        if not entries:
            entries = [dict(o.get('source', {}), calibration_id=o.get('calibration_id'))
                       for o in result.get('observations', [])]
        identities = set()
        for entry in entries:
            if not entry.get('calibration_id'):
                continue
            if not entry.get('observation'):
                raise ValueError('Stored geometry lacks observation provenance; resample')
            item = self._observation({'observation_path':entry['observation']}, needs_base=True)
            if item.get('calibration_id') != entry['calibration_id']:
                raise ValueError('Stored geometry calibration changed; resample')
            data = json.loads(Path(item['observation_path']).read_text())
            identity = data['robot_alignment']['state']['observation_identity']
            identities.add((identity['instance_id'], identity['connection_epoch']))
        if len(identities)>1:
            raise ValueError('Geometry crosses robot connection/instance identity; start a new batch')

    def refresh_plan(self, plan):
        """Recheck saved calibrated answers when the current plan is read."""
        stale = []
        for question in plan.get('questions', []):
            if question.get('state') != 'answered' or not isinstance(question.get('result'), dict):
                continue
            try:
                self._revalidate(question['result'])
            except (ValueError, KeyError, TypeError, OSError):
                stale.append(question['id'])
        if stale:
            return self.plans.stale_questions(plan['task_id'], plan['version'], stale)
        return plan

    async def query(self, request, *, expected_version=None):
        from jsonschema import validate
        validate(request, SCENE_TOOL['inputSchema'])
        async with self._lock:
            plan = self.plans.get(request.get('task_id'))
            if not plan.get('task_id'):
                raise ValueError('Create a task_plan first')
            if self.plans.get()['task_id'] != plan['task_id']:
                raise ValueError('Resume this task explicitly before geometry work')
            plan = self.refresh_plan(plan)
            expected_version = request.get('expected_version', expected_version)
            if expected_version is not None and plan['version'] != expected_version:
                raise ValueError('Plan changed during capture; re-read question before querying')
            question_id = request.get('question_id')
            if question_id and not any(q['id']==question_id for q in plan['questions']):
                raise ValueError('Unknown question_id')
            body = {k:v for k,v in request.items() if k not in ('question_id','expected_version')}
            body['task_id'] = plan['task_id']
            if 'observations' in body:
                body['observations'] = [self._observation(o, needs_base=True) for o in body['observations']]
            if 'observation_path' in body:
                body.update(self._observation({k:body[k] for k in _REGION if k in body},
                                             needs_base=body['operation']!='surface_point'))
            started = time.monotonic()
            env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1')
            self.proc = await asyncio.create_subprocess_exec(sys.executable, '-m', 'piperx_middleware.scene_worker', str(self.root),
                stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, env=env)
            proc = self.proc
            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(json.dumps(body,allow_nan=False).encode()), 30)
                if proc.returncode:
                    raise ValueError('Geometry worker failed: '+stderr.decode(errors='replace')[-800:])
                if len(stdout)>2*1024*1024:
                    raise ValueError('Geometry result exceeds budget')
                result = json.loads(stdout)
            finally:
                if proc.returncode is None:
                    proc.kill()
                await proc.communicate()
                self.proc = None
            result['elapsed_s'] = round(time.monotonic()-started,4)
            result['compute_host'] = 'frontend_local_cpu'
            self._revalidate(result)
            result['evidence_scope'] = 'saved_observations; reobserve after object or robot changes'
            if question_id:
                try:
                    updated = self.plans.record_result(plan['task_id'],plan['version'],question_id, {
                        'status':'answered' if result.get('status')=='ok' else 'unavailable',
                        'value':{k:v for k,v in result.items() if k not in ('observations','sources')},
                        'evidence':result.get('result_path') or result.get('metadata_json')})
                    result['plan_version'] = updated['version']
                except Exception as exc:
                    result['plan_writeback'] = {'status':'conflict','message':str(exc)}
            return result

    async def close(self):
        if self.proc and self.proc.returncode is None:
            self.proc.kill()
            await self.proc.wait()
