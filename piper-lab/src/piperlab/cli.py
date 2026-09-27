"""Local operations. All learning commands default to offline, read-only hardware."""
import argparse
import json
from pathlib import Path


def main():
    import sys
    if len(sys.argv)>1 and sys.argv[1]=='mcp':
        from .harness import main as mcp_main
        raise SystemExit(mcp_main(sys.argv[2:]))
    if len(sys.argv)>1 and sys.argv[1] in ('demo','policy','sim','eval','models'):
        from .video_cli import main as video_main
        raise SystemExit(video_main(sys.argv[1:]))
    parser = argparse.ArgumentParser(prog='piper-lab')
    sub = parser.add_subparsers(dest='command', required=True)
    for name,description in [('demo','Compile, inspect and find video demonstrations'),('policy','Run the visual simulation policy'),
                             ('sim','Start and inspect physical simulation'),('eval','Run paired demonstration evaluations'),
                             ('models','Show provider profiles or probe a vision API'),('mcp','Expose pipeline tools over MCP stdio')]:
        sub.add_parser(name,help=description)
    p = sub.add_parser('mock'); p.add_argument('output'); p.add_argument('--episodes', type=int, default=5); p.add_argument('--frames', type=int, default=40)
    p = sub.add_parser('mock-mcap'); p.add_argument('output'); p.add_argument('--episodes', type=int, default=5); p.add_argument('--frames', type=int, default=40)
    p = sub.add_parser('convert'); p.add_argument('source'); p.add_argument('output')
    p = sub.add_parser('export'); p.add_argument('source'); p.add_argument('output'); p.add_argument('--repo-id', default='local/piper_lab')
    for name in ('train', 'shadow'):
        p = sub.add_parser(name); p.add_argument('dataset'); p.add_argument('output'); p.add_argument('--gpu-uuid'); p.add_argument('--config', default='config/hardware.yaml')
        if name == 'train':
            p.add_argument('--steps', type=int, default=100); p.add_argument('--batch-size', type=int, default=4); p.add_argument('--resume-from')
        else:
            p.add_argument('--checkpoint', required=True); p.add_argument('--max-frames', type=int)
    p = sub.add_parser('check-config'); p.add_argument('--config', default='config/hardware.yaml')
    args = vars(parser.parse_args()); command = args.pop('command')
    if command in ('train', 'shadow'):
        from . import learning
        from .safety import load_config
        config = args.pop('config')
        args['gpu_uuid'] = args['gpu_uuid'] or load_config(config)['gpu_uuid']
        result = getattr(learning, command)(**args)
    elif command == 'mock-mcap':
        from .mock_mcap import generate
        result = generate(**args)
    elif command == 'check-config':
        from .safety import load_config, commissioning_errors
        config = load_config(args['config']); result = {'mode': config['mode'], 'real_hardware_missing': commissioning_errors({**config, 'mode':'real'})}
    else:
        from . import data
        result = getattr(data, {'mock':'generate_mock','convert':'convert_mcap','export':'export_lerobot'}[command])(**args)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
