"""Standalone robot CLI, including simulator and MCP entry points."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] == 'host':
        from .ssh_runtime import host_main
        parser = argparse.ArgumentParser(prog='piper-robot host')
        parser.add_argument('--stdio', action='store_true', required=True)
        parser.parse_args(args[1:])
        return host_main()
    if args and args[0] == 'mcp':
        from .mcp_server import main as serve
        return serve(args[1:])
    if args and args[0] == 'sim':
        parser = argparse.ArgumentParser(prog='piper-robot sim')
        parser.add_argument('operation', choices=['start'])
        parser.add_argument('--root', required=True)
        parser.add_argument('--port', type=int, default=8798)
        parser.add_argument('--seed', type=int, default=200)
        a = parser.parse_args(args[1:])
        from .cli import initialize
        root = Path(a.root).resolve()
        if not (root / 'config.json').exists():
            initialize(root, 'mujoco', port=a.port, simulation_seed=a.seed,
                       tcp_offset_m=[0., 0., .1425], tcp_offset_rpy_deg=[0., 0., 0.])
        else:
            settings = json.loads((root / 'config.json').read_text())
            if (settings['backend'], settings['port'], settings.get('simulation_seed')) != ('mujoco', a.port, a.seed):
                raise ValueError('Existing simulation configuration differs; choose a new --root')
        return subprocess.call([sys.executable, '-m', 'piperx_middleware.cli', '--root', str(root), 'serve'])
    if args and args[0] == 'observe':
        parser = argparse.ArgumentParser(prog='piper-robot observe')
        parser.add_argument('--root', type=Path, required=True)
        parser.add_argument('--url')
        parser.add_argument('--workspace', type=Path, required=True)
        parser.add_argument('--output', required=True)
        parser.add_argument('--serial', help='Simulator camera serial: mujoco-overview (default) or mujoco-wrist')
        a = parser.parse_args(args[1:])
        from .client import RobotClient
        from .simulation_camera import capture
        from .cli import configured_url
        from .managed_runtime import read_runtime_json, validate_runtime_url
        explicit = a.url or os.environ.get('PIPERX_URL')
        runtime = read_runtime_json(a.root/'runtime.json') if not explicit else None
        url = explicit or (validate_runtime_url(runtime['url']) if runtime else configured_url(a.root))
        client = RobotClient.from_env(root=a.root, url=url)
        try:
            if runtime:
                actual = client.call('GET','/v1/capabilities')
                if actual.get('instance_id') != runtime['instance_id'] or actual.get('backend') != runtime['backend']:
                    raise ValueError('Executor identity differs from this profile runtime; reconnect the frontend')
            print(json.dumps(capture(client, a.workspace, a.output, serial=a.serial), indent=2))
        finally:
            client.close()
        return 0
    if args and args[0] in ('--help', '-h'):
        print('piper-robot: sim start | observe | mcp | [--root DIR] <executor command>\n'
              'Use sim --help, observe --help, mcp --help, or the executor help below.')
        args = ['--help']
    return subprocess.call([sys.executable, '-m', 'piperx_middleware.cli', *args])


if __name__ == '__main__':
    raise SystemExit(main())
