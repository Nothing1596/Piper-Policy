"""Apply small, auditable integration edits to the user's separate local checkout.

No upstream code is vendored, copied to Piper-Policy, or published by this tool.
The supplied upstream license still controls use of that checkout.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from . import UPSTREAM_COMMIT

MANIFEST = '.piper-gpt-adapter.json'
BASE_HASHES = {
 'src/gpt_policy/main.py':'98ed1737bb1198e28a29f80252e3c2b0b515a2ab5bb7ac8aca5fe67936d52166',
 'src/gpt_policy/hardware/camera.py':'83b5c426ce8f15f66fd2067e43358dd353baec945c556f172efdcd880a54bc86',
 'src/gpt_policy/input/video_cache.py':'cf6f7ff372cecba54c42ad844a2154f5cd1d250ab7941dd3d562919a82719ad7',
 'src/gpt_policy/harness/process.py':'273d863d031abb0b8b306e16d2c1627d7b0c05b23f53a4ffeaebb7f104210fa3',
 'src/gpt_policy/harness/codex.py':'dfce8843c0708a3505725cf7b9918c08582ef82f822bf463380e126e2b616e45',
 'src/gpt_policy/harness/protocol.py':'8d5f7151418bfc177fed4786b832e630eb8a0e82b8e62afc4d99e5fc88216ce1',
 'src/gpt_policy/preflight.py':'7b14c1e9240b73fb92f102ece8f3d19320c1748871d2834688dc70ef016ff173',
 'pyproject.toml':'daa0d3c2b1a153429b00ce10310153405306dc07e26e1869a3e707476de934f4',
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def revision(root):
    return subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError('Pinned upstream integration location changed: '+repr(old[:80]))
    return source.replace(old,new,1)


def transformed(path, source):
    if path.endswith('/main.py'):
        source = replace_once(source,'from .hardware.camera import CameraSet',
            'from gpt_policy_piper.robot import PiperRobot\nfrom gpt_policy_piper.camera import SimulationCameraSet\nfrom .hardware.camera import CameraSet')
        source = replace_once(source,'        if settings.get("camera_backend", "v4l2") == "realsense":',
            '        if settings.get("backend") == "piper_mujoco":\n            cameras = SimulationCameraSet(settings, runtime.camera_width, runtime.camera_height)\n        elif settings.get("camera_backend", "v4l2") == "realsense":')
        source = replace_once(source,'        if backend == "yam":',
            '        if backend == "piper_mujoco":\n            if runtime.right_interface:\n                raise ValueError("Piper baseline is single-arm simulation only")\n            robot = PiperRobot(runtime.robot_model, runtime.interface, runtime.gripper_open_readout, runtime.trajectory_hz, settings)\n            arms = ("left",)\n            interface_text = runtime.interface\n        elif backend == "yam":')
    elif path.endswith('/camera.py'):
        source = replace_once(source,'import fcntl','try:\n    import fcntl\nexcept ImportError:\n    fcntl = None  # CapturedImage remains usable; V4L2 is explicitly unavailable')
        source = replace_once(source,'        self.path = path','        if fcntl is None:\n            raise RuntimeError("V4L2 capture requires Linux; use the Piper simulation camera")\n        self.path = path')
    elif path.endswith('/video_cache.py'):
        source = replace_once(source,'import fcntl','from gpt_policy_piper.platform import acquire_cache_lock')
        source = source.replace('fcntl.flock(lock.fileno(), fcntl.LOCK_EX)','acquire_cache_lock(lock.fileno())')
    elif path.endswith('/process.py'):
        source = replace_once(source,'        os.set_blocking(self.process.stdin.fileno(), False)',
            '        if os.name != "nt":\n            os.set_blocking(self.process.stdin.fileno(), False)')
        source = replace_once(source,'        descriptor = self.process.stdin.fileno()',
            '        if os.name == "nt":\n            from gpt_policy_piper.platform import write_windows_pipe\n            try:\n                write_windows_pipe(self.process, data.tobytes().decode("utf-8"), deadline)\n            except TimeoutError as exc:\n                raise AgentTimeoutError("Agent input write timed out") from exc\n            except (OSError, ValueError) as exc:\n                raise AgentError("Agent input pipe failed") from exc\n            return\n        descriptor = self.process.stdin.fileno()')
        source = replace_once(source,'        self._closed = True\n        self._signal(signal.SIGTERM)',
            '        self._closed = True\n        if os.name == "nt":\n            from gpt_policy_piper.platform import close_windows_json_process\n            close_windows_json_process(self)\n            return\n        self._signal(signal.SIGTERM)')
    elif path.endswith('/codex.py'):
        source = replace_once(source,'        if process.poll() is None:\n            try:\n                process_group = process.pid',
            '        if os.name == "nt":\n            from gpt_policy_piper.platform import close_windows_process\n            close_windows_process(process)\n            return\n        if process.poll() is None:\n            try:\n                process_group = process.pid')
    elif path.endswith('/protocol.py'):
        source = replace_once(source,'\n    camera_lines = [',
            '\n    if (settings or {}).get("backend") == "piper_mujoco":\n        from gpt_policy_piper.platform import calibration_notes\n        return calibration_notes(arms, settings)\n    camera_lines = [')
        source = replace_once(source,'    robot_label = "YAM" if',
            '    robot_label = "PiperX simulation" if (settings or {}).get("backend") == "piper_mujoco" else "YAM" if')
    elif path.endswith('/preflight.py'):
        source = replace_once(source,'if backend not in {"arx", "yam"}:','if backend not in {"arx", "yam", "piper_mujoco"}:')
        source = replace_once(source,'    if runtime.right_interface == runtime.interface:',
            '    if backend == "piper_mujoco" and runtime.right_interface:\n        raise ValueError("Piper baseline requires one simulated arm")\n    if runtime.right_interface == runtime.interface:')
    elif path == 'pyproject.toml':
        source += '\n[tool.hatch.metadata]\nallow-direct-references = true\n'
    return source


def verify_prepared(root):
    root = Path(root).resolve()
    if revision(root) != UPSTREAM_COMMIT:
        raise ValueError('Upstream HEAD must equal '+UPSTREAM_COMMIT)
    manifest = json.loads((root/MANIFEST).read_text(encoding='utf-8'))
    if manifest.get('upstream_commit') != UPSTREAM_COMMIT or set(manifest['files']) != set(BASE_HASHES):
        raise ValueError('Invalid adapter preparation manifest')
    for path, expected in manifest['files'].items():
        original = subprocess.check_output(['git','-C',str(root),'show','HEAD:'+path])
        if digest(original) != BASE_HASHES[path]:
            raise ValueError('Pinned upstream source does not match: '+path)
        approved = digest(transformed(path,original.decode('utf-8')).encode('utf-8'))
        if expected != approved or digest((root/path).read_bytes()) != approved:
            raise ValueError('Prepared upstream file changed: '+path)
    return root


def prepare(root):
    root = Path(root).resolve()
    if (root/MANIFEST).exists():
        verify_prepared(root)
        return {'status':'already_prepared','upstream_commit':UPSTREAM_COMMIT}
    if revision(root) != UPSTREAM_COMMIT:
        raise ValueError('Check out exact upstream commit '+UPSTREAM_COMMIT)
    outputs = {}
    for path,expected in BASE_HASHES.items():
        raw = (root/path).read_bytes()
        if digest(raw) != expected:
            raise ValueError('Refusing to modify an unexpected or edited upstream file: '+path)
        outputs[path] = transformed(path,raw.decode('utf-8')).encode('utf-8')
    # Validate every transformation before changing any file.
    written = []
    try:
        for path,raw in outputs.items():
            (root/path).write_bytes(raw)
            written.append(path)
        manifest = {'upstream_commit':UPSTREAM_COMMIT,'adapter':'piper-mujoco-v1',
                    'files':{p:digest(raw) for p,raw in outputs.items()}}
        (root/MANIFEST).write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    except Exception:
        for path in written:
            original = subprocess.check_output(['git','-C',str(root),'show','HEAD:'+path])
            (root/path).write_bytes(original)
        raise
    return {'status':'prepared','upstream_commit':UPSTREAM_COMMIT,'modified_files':list(outputs)}


def write_config(root, runtime_root, url='http://127.0.0.1:8808'):
    from piperx_middleware.models import Settings
    from piperx_middleware.kinematics import PiperKinematics
    settings = Settings.model_validate_json((runtime_root/'config.json').read_text(encoding='utf-8'))
    if settings.backend != 'mujoco' or settings.control_profile != 'direct':
        raise ValueError('Select an existing direct-profile MuJoCo runtime')
    matrix = PiperKinematics(settings.tcp_offset_m,settings.tcp_offset_rpy_deg)._tcp_offset.tolist()
    config = {'machine':'piper-mujoco-baseline','backend':'piper_mujoco','agent':'codex',
        'agent_config_dir':str(root/'configs'),'tool_catalog':'configs/tools.json',
        'runtime':{'robot_model':'PiperX','interface':'mujoco','right_interface':None,
                   'trajectory_hz':30.,'camera_width':640,'camera_height':480,'task_name':'piper-simulation',
                   'convert_camera_images_to_jpeg':True,'camera_jpeg_quality':85,'max_decisions':100},
        'robot':{'gripper_width_m':settings.gripper_max_m},'motion':{},
        'cameras':{},'vision':{'camera_intrinsics':{},'distortion_coefficients':{}},
        'calibration':{'link6_from_sdk_eef':[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],
                       'link6_from_tcp':matrix,'link6_from_camera':{},'base_from_camera':{}},
        'recording':{'fps':10,'jpeg_quality':75,'state_hz':20},
        'scene':{'safety_notes':['Single-arm RGB-only MuJoCo baseline; no depth or wrist view is available.']},
        'piper_simulation':{'url':url,'model_token_file':str(runtime_root/'model.token'),
                            'operator_token_file':str(runtime_root/'operator.token')}}
    path = root/'configs/piper-mujoco.local.json'
    if path.exists():
        raise FileExistsError('Configuration exists; review it before choosing a new runtime: '+str(path))
    path.write_text(json.dumps(config,indent=2)+'\n',encoding='utf-8')
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream',type=Path,required=True)
    parser.add_argument('--runtime-root',type=Path)
    parser.add_argument('--url',default='http://127.0.0.1:8808')
    args = parser.parse_args()
    root = args.upstream.resolve()
    result = prepare(root)
    if args.runtime_root:
        result['config'] = str(write_config(root,args.runtime_root.resolve(),args.url))
    print(json.dumps(result,indent=2))

if __name__ == '__main__':
    main()
