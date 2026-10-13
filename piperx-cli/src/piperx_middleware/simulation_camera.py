"""Current simulation evidence shared by the independent CLI and MCP."""
import base64
import json
from pathlib import Path
from mcp.server.fastmcp.exceptions import ToolError


_SIM_SERIALS = {
    None: ['mujoco-overview'],
    'mujoco-overview': ['mujoco-overview'],
    'overview': ['mujoco-overview'],
    'overview_camera': ['mujoco-overview'],
    'fixed': ['mujoco-overview'],
    'mujoco-wrist': ['mujoco-wrist'],
    'wrist': ['mujoco-wrist'],
    'wrist_camera': ['mujoco-wrist'],
    'all': ['mujoco-overview', 'mujoco-wrist'],
    'both': ['mujoco-overview', 'mujoco-wrist'],
}


def capture(client, workspace, output, serial=None):
    if serial not in _SIM_SERIALS or len(_SIM_SERIALS[serial]) != 1:
        raise ToolError('Simulator camera serial must be mujoco-overview or mujoco-wrist')
    target_serial = _SIM_SERIALS[serial][0]
    root = Path(workspace).resolve(strict=True)
    destination = (root / output).resolve()
    if not root.is_dir() or destination == root or not destination.is_relative_to(root):
        raise ToolError('Path must be inside the configured workspace')
    if destination.exists():
        raise ToolError('Output already exists; choose a new path')
    capabilities = client.call('GET', '/v1/capabilities')
    if 'error' in capabilities:
        raise ToolError('Cannot read simulator capabilities: '+str(capabilities['error']))
    if capabilities.get('backend') != 'mujoco':
        raise ToolError('This camera tool requires the MuJoCo simulator')
    path = '/v1/simulation/observation' if target_serial == 'mujoco-overview' else f'/v1/simulation/observation?serial={target_serial}'
    observation = dict(client.call('GET', path))
    if 'error' in observation:
        raise ToolError('Simulation observation unavailable')
    return {'evidence': str(destination), **save_observation(root, destination, observation)}


def save_observation(root, destination, observation):
    """Adapt authenticated simulator evidence to the same recorded RGB-D schema."""
    import io
    import numpy as np
    from .scene_bridge import calibration

    root, destination = Path(root).resolve(), Path(destination).resolve()
    if destination == root or not destination.is_relative_to(root) or destination.exists():
        raise ValueError('Choose a new observation directory inside this profile')
    if observation.get('provenance') != 'mujoco_sensor' or not observation.get('instance_id'):
        raise ValueError('Expected identified MuJoCo sensor evidence')
    alignment = observation.get('robot_alignment')
    if not isinstance(alignment, dict):
        raise ValueError('Simulator lacks capture-aligned state; update the executor')
    identity = (alignment.get('state') or {}).get('observation_identity') or {}
    if any(identity.get(k) != observation.get(k) for k in ('instance_id','connection_epoch')):
        raise ValueError('Capture and robot state identities differ')
    rgb = base64.b64decode(observation['rgb_jpeg_b64'], validate=True)
    depth = base64.b64decode(observation['depth_npy_b64'], validate=True)
    values = np.load(io.BytesIO(depth), allow_pickle=False)
    if values.ndim != 2 or values.dtype.kind != 'f':
        raise ValueError('Expected 2-D simulated depth in metres')
    height, width = values.shape
    camera = observation['camera_info']
    k = np.asarray(camera['intrinsics'], dtype=float)
    if k.shape != (3,3) or not np.isfinite(k).all() or k[0,0] <= 0 or k[1,1] <= 0:
        raise ValueError('Invalid simulator intrinsics')
    role = camera.get('role') if camera.get('role') in ('fixed', 'wrist') else 'fixed'
    serial = camera.get('camera_serial') or camera.get('serial') or ('mujoco-wrist' if role == 'wrist' else 'mujoco-overview')
    from .scene_geometry import _validate_transform
    if role == 'wrist':
        flange_from_camera = camera.get('flange_from_camera')
        _validate_transform(flange_from_camera, 'simulator wrist calibration')
        cal = {'camera_serial':serial,'role':'wrist','units':'m',
               'flange_from_camera':flange_from_camera,'provenance':'mujoco_sensor',
               'instance_id':observation['instance_id'],'connection_epoch':observation['connection_epoch'],
               'scene_epoch':observation['epoch']}
    else:
        _validate_transform(camera['base_from_camera'], 'simulator calibration')
        cal = {'camera_serial':serial,'role':'fixed','units':'m',
               'base_from_camera':camera['base_from_camera'],'provenance':'mujoco_sensor',
               'instance_id':observation['instance_id'],'connection_epoch':observation['connection_epoch'],
               'scene_epoch':observation['epoch']}
    operator = root/'camera.json'
    if operator.exists():
        config = json.loads(operator.read_text(encoding='utf-8'))
        entries = config.get('cameras', [config])
        if any(e.get('serial') == serial for e in entries):
            raise ValueError('Simulation camera serial conflicts with operator camera configuration')
    # This simulator-only registration never overwrites operator camera.json.
    cal_path = root/('simulation-wrist-calibration.json' if role == 'wrist' else 'simulation-calibration.json')
    def write(path, value):
        temp = path.with_suffix('.tmp')
        temp.write_text(json.dumps(value,allow_nan=False),encoding='utf-8')
        temp.replace(path)
    root.mkdir(parents=True,exist_ok=True)
    write(cal_path,cal)
    entry = {'serial':serial,'role':role,'calibration_file':str(cal_path)}
    _, fingerprint = calibration(entry)
    sim_cam_path = root/'simulation-camera.json'
    existing_sim = []
    if sim_cam_path.exists():
        try:
            raw_sim = json.loads(sim_cam_path.read_text(encoding='utf-8'))
            existing_sim = raw_sim.get('cameras', [raw_sim] if raw_sim.get('serial') else [])
        except Exception:
            existing_sim = []
    merged = [e for e in existing_sim if isinstance(e, dict) and e.get('serial') and e.get('serial') != serial]
    merged.append(entry)
    write(sim_cam_path, entry if len(merged) == 1 else {'cameras': merged})
    destination.mkdir(parents=True,exist_ok=False)
    (destination/'rgb.jpg').write_bytes(rgb)
    (destination/'depth.npy').write_bytes(depth)
    frame = {k:v for k,v in observation.items() if k not in ('rgb_jpeg_b64','depth_npy_b64')}
    frame.update(serial=serial,camera_serial=serial,role=role,camera_role=role,simulated=True,
                 frame='camera_color_optical',captured_at=str(observation['source_stamp_s']),
                 rgb_path=str(destination/'rgb.jpg'),depth_path=str(destination/'depth.npy'),depth_scale_m=1.,
                 observation_path=str(destination/'observation.json'),calibration_id=fingerprint,
                 intrinsics={'width':width,'height':height,'fx':float(k[0,0]),'fy':float(k[1,1]),
                             'ppx':float(k[0,2]),'ppy':float(k[1,2]),'coeffs':[0.]*5})
    write(destination/'observation.json',frame)
    return frame


class SimulationCamera:
    """One fresh simulator snapshot per request; no camera process or fake history."""
    proc = None

    def __init__(self, root, rest):
        import asyncio
        from collections import deque
        self.root, self.rest = Path(root), rest
        self._lock = asyncio.Lock()
        self._image_paths = deque(maxlen=300)

    async def observe(self, *, window_s=2, max_frames=3, roi=None, since_mono_s=None, serial=None):
        import uuid
        import math
        if serial not in _SIM_SERIALS:
            raise ValueError('Simulator camera serial must be mujoco-overview or mujoco-wrist')
        if not math.isfinite(window_s) or not 0 < window_s <= 120 or type(max_frames) is not int or not 1 <= max_frames <= 6:
            raise ValueError('Invalid observation budget')
        target_serials = _SIM_SERIALS[serial]
        async with self._lock:
            from PIL import Image
            import io
            group_folder = self.root/'observations'/uuid.uuid4().hex
            frames = []
            for idx, target_serial in enumerate(target_serials):
                path = '/v1/simulation/observation' if target_serial == 'mujoco-overview' else f'/v1/simulation/observation?serial={target_serial}'
                data = await self.rest('GET', path)
                if 'error' in data:
                    raise ValueError(data['error'].get('message',str(data['error'])))
                image = Image.open(io.BytesIO(base64.b64decode(data['rgb_jpeg_b64'],validate=True)))
                if roi is not None:
                    if (not isinstance(roi,list) or len(roi)!=4 or any(type(x) is not int for x in roi)
                            or not 0 <= roi[0] < roi[2] <= image.width
                            or not 0 <= roi[1] < roi[3] <= image.height):
                        raise ValueError('ROI must fit inside the image')
                folder = group_folder if len(target_serials) == 1 else group_folder/target_serial
                frame = save_observation(self.root,folder,data)
                frame['received_mono_s'] = float(frame.get('source_stamp_s', idx))
                self._image_paths.append(frame['rgb_path'])
                if roi is not None:
                    crop = folder/'roi.jpg'; image.crop(roi).save(crop,format='JPEG')
                    frame.update(crop_path=str(crop),roi=roi)
                    self._image_paths.append(str(crop))
                (folder/'observation.json').write_text(json.dumps(frame),encoding='utf-8')
                frames.append(frame)
            result = {'schema':'piper.observation.v1','provenance':'mujoco_sensor','simulated':True,
                      'capture_mode':'on_demand_snapshot','requested_window_s':window_s,'window_available_s':0,
                      'note':'One fresh snapshot; no continuous history. Geometry and images do not prove task success.',
                      'frames':frames,'observation_path':frames[-1]['observation_path'],
                      'bundle_path':str(group_folder/'bundle.json')}
            (group_folder/'bundle.json').write_text(json.dumps(result),encoding='utf-8')
            (self.root/'last_observation.json').write_text(json.dumps({'path':frames[-1]['observation_path']}),encoding='utf-8')
            return result

    def image_inputs(self, result):
        from .live_camera import CameraGroup
        return CameraGroup.image_inputs(self,result)

    async def close(self):
        self._image_paths.clear()
