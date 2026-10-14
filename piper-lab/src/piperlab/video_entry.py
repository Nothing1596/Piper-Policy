"""Video-only CLI: usable without the robot distribution."""
import argparse
import json
import sys


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(prog='piper-video', description='Independent video evidence pipeline')
    parser.add_argument('command', choices=['candidates', 'compile', 'inspect', 'find', 'evaluate', 'models', 'mcp'])
    if not args or args[0] in ('-h', '--help'):
        parser.print_help()
        return 0
    command = parser.parse_args(args[:1]).command
    if command == 'mcp':
        p = argparse.ArgumentParser(prog='piper-video mcp')
        p.add_argument('--workspace', required=True)
        p.add_argument('--model-config')
        p.add_argument('--detector-onnx')
        p.parse_args(args[1:])
        from .harness import main as serve
        return serve(args[1:])
    if command == 'candidates':
        p = argparse.ArgumentParser(prog='piper-video candidates')
        p.add_argument('--video', required=True)
        p.add_argument('--output', required=True)
        a = p.parse_args(args[1:])
        from .video import build_candidates
        print(json.dumps(build_candidates(a.video, a.output), ensure_ascii=False, indent=2))
        return 0
    from .video_cli import main as video_main
    return video_main(args if command == 'models' else ['demo', *args])


if __name__ == '__main__':
    raise SystemExit(main())
