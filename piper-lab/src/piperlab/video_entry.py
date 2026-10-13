"""Video-only CLI: usable without the robot distribution."""
import argparse
import json
import sys


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(prog='piper-video', description='Independent video evidence pipeline')
    parser.add_argument('command', choices=['watch', 'candidates', 'compile', 'inspect', 'find', 'evaluate', 'models', 'mcp'])
    if not args or args[0] in ('-h', '--help'):
        parser.print_help()
        return 0
    command = parser.parse_args(args[:1]).command
    if command == 'watch':
        p=argparse.ArgumentParser(prog='piper-video watch')
        p.add_argument('--address',required=True);p.add_argument('--output',required=True)
        p.add_argument('--mode',choices=['direct','pipeline'],default='direct')
        p.add_argument('--at-s',type=float);p.add_argument('--start-s',type=float);p.add_argument('--end-s',type=float)
        p.add_argument('--analysis-cache');p.add_argument('--max-frames',type=int,default=6);p.add_argument('--detector-onnx');p.add_argument('--model-config')
        a=p.parse_args(args[1:]);detector=None;model=None
        if a.mode=='pipeline' and a.detector_onnx:
            from .perception.detector_onnx import OnnxDetector
            detector=OnnxDetector(a.detector_onnx)
        if a.mode=='pipeline' and a.model_config:
            from .models.providers import ModelProfile,create_model
            model=create_model(ModelProfile.load(a.model_config));model.discover_identity()
        from .video.watch import watch
        result=watch(a.address,a.output,mode=a.mode,at_s=a.at_s,start_s=a.start_s,end_s=a.end_s,max_frames=a.max_frames,detector=detector,model=model,analysis_cache=a.analysis_cache)
        print(json.dumps(result,ensure_ascii=False,indent=2));return 0
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
        p.add_argument('--detector-onnx')
        a = p.parse_args(args[1:])
        from .video import build_candidates
        detector = None
        if a.detector_onnx:
            from .perception.detector_onnx import OnnxDetector
            detector = OnnxDetector(a.detector_onnx)
        print(json.dumps(build_candidates(a.video, a.output, detector=detector), ensure_ascii=False, indent=2))
        return 0
    from .video_cli import main as video_main
    return video_main(args if command == 'models' else ['demo', *args])


if __name__ == '__main__':
    raise SystemExit(main())
