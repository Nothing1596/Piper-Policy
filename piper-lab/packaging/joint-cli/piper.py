"""One CLI for the video pipeline and PiperX executor; never auto-start hardware."""
import importlib
import importlib.metadata as metadata
import json
from pathlib import Path
import subprocess
import sys

HELP = """Piper combined CLI

  piper demo ...       Compile, inspect, find and evaluate demonstrations
  piper video ...      Alias for demo
  piper policy ...     Run the visual policy with explicit backend/token
  piper sim ...        Start or evaluate MuJoCo simulation
  piper eval ...       Run paired demonstration evaluations
  piper device ...     All lower-controller CLI commands (piperx)
  piper lab ...        Other upper-controller commands (piper-lab)
  piper models ...     Show provider profiles or probe an image API
  piper mcp ...        MCP stdio tools for Codex, Claude Code and other clients
  piper doctor         Read-only installed dependency/asset checks

Examples:
  piper demo compile --help
  piper device --help
  piper device --root .runtime status

Model inference uses local LM Studio by default, or an explicit --model-config API profile.
No command is run when this help is shown; installation does not connect hardware.
"""


def main():
    args = sys.argv[1:]
    if not args or args[0] in ('-h', '--help'):
        print(HELP)
        return 0
    group, *rest = args
    if group == 'doctor':
        if rest:
            print('doctor takes no arguments', file=sys.stderr)
            return 2
        checks = {}
        for name in ('piper-lab', 'piperx-middleware', 'av', 'opencv-python',
                     'onnxruntime', 'mujoco', 'trackers', 'supervision', 'python-can'):
            try:
                checks[name] = metadata.version(name)
            except metadata.PackageNotFoundError:
                checks[name] = None
        for module in ('av', 'cv2', 'onnxruntime', 'mujoco', 'trackers', 'supervision'):
            try:
                importlib.import_module(module)
                checks['import:' + module] = True
            except Exception as exc:
                checks['import:' + module] = str(exc)
        from piperx_middleware.mujoco_backend import DEFAULT_ASSET_PATH
        checks['simulation_asset_exists'] = Path(DEFAULT_ASSET_PATH).is_file()
        ok = all(v is not None and v is not False for v in checks.values())
        ok = ok and all(v is True for k, v in checks.items() if k.startswith('import:'))
        print(json.dumps({'ok': ok, 'python': sys.version, 'checks': checks,
                          'model_inference_tested': False, 'hardware_connected': False}, indent=2))
        return 0 if ok else 1
    if group == 'device':
        command = [sys.executable, '-m', 'piperx_middleware.cli', *rest]
    elif group == 'lab':
        command = [sys.executable, '-m', 'piperlab.cli', *rest]
    elif group in ('demo', 'video', 'policy', 'sim', 'eval', 'models', 'mcp'):
        command = [sys.executable, '-m', 'piperlab.cli',
                   'demo' if group == 'video' else group, *rest]
    else:
        print('Unknown command: ' + group, file=sys.stderr)
        return 2
    return subprocess.call(command)


if __name__ == '__main__':
    raise SystemExit(main())
