"""CLI entry points for human-video learning and visual simulation."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import sys


def model_options(parser):
    parser.add_argument('--model-config', help='Explicit JSON provider profile; keys are read from named environment variables')
    parser.add_argument('--model')
    parser.add_argument('--model-url')
    parser.add_argument('--model-manifest')


def configured_model(args, log_dir, *, max_output_tokens=2048):
    from .models.providers import ModelProfile, create_model
    if args.model_config:
        if args.model or args.model_url or args.model_manifest:
            raise ValueError('Use --model-config or legacy --model/--model-url/--model-manifest, not both')
        profile = ModelProfile.load(args.model_config)
    else:
        profile = ModelProfile(model=args.model, base_url=args.model_url,
                               artifact_manifest=args.model_manifest, max_output_tokens=max_output_tokens)
    model = create_model(profile, log_dir=log_dir)
    model.discover_identity()
    return model


def load_bundle(path,store_path=None):
    if str(path).startswith('demo-'):
        if not store_path:raise ValueError('Use --store to resolve a demo ID')
        from .demonstration.store import DemoStore
        with DemoStore(store_path) as store:return store.get(str(path))
    path=Path(path).resolve()
    if path.is_dir():path=path/'demo.json'
    bundle=json.loads(path.read_text(encoding='utf-8'))
    from .demonstration.paths import resolve_bundle
    bundle=resolve_bundle(bundle,path.parent)
    for frame in bundle.get('frames',[]):
        image=Path(frame['image_path'])
        frame['image_path']=str(image if image.is_absolute() else path.parent/image)
    return bundle


def main(argv=None):
    parser=argparse.ArgumentParser(prog='piper-lab')
    sub=parser.add_subparsers(dest='group',required=True)
    demo=sub.add_parser('demo').add_subparsers(dest='operation',required=True)
    p=demo.add_parser('compile');p.add_argument('--video',required=True);p.add_argument('--task',required=True)
    p.add_argument('--output',required=True);model_options(p)
    p.add_argument('--max-keyframes',type=int,default=24)
    p.add_argument('--store');p.add_argument('--cache')
    p.add_argument('--detector-onnx')
    p=demo.add_parser('inspect');selector=p.add_mutually_exclusive_group(required=True)
    selector.add_argument('--bundle');selector.add_argument('--id');p.add_argument('--store')
    p=demo.add_parser('find');p.add_argument('--store',required=True);p.add_argument('--task',required=True)
    p=demo.add_parser('evaluate');p.add_argument('--bundle',required=True);p.add_argument('--store')
    p.add_argument('--annotations',required=True);p.add_argument('--output',required=True)
    policy=sub.add_parser('policy').add_subparsers(dest='operation',required=True)
    p=policy.add_parser('run');p.add_argument('--demo');p.add_argument('--task',required=True)
    p.add_argument('--output',required=True);p.add_argument('--backend',default='http://127.0.0.1:8798')
    p.add_argument('--token-file',required=True);model_options(p)
    p.add_argument('--max-decisions',type=int,default=20);p.add_argument('--skill-config')
    p.add_argument('--store')
    sim=sub.add_parser('sim').add_subparsers(dest='operation',required=True)
    p=sim.add_parser('start');p.add_argument('--root',default='.runtime-vision');p.add_argument('--port',type=int,default=8798)
    p.add_argument('--seed',type=int,default=0);p.add_argument('--scene',choices=['tabletop-piperx'],default='tabletop-piperx')
    p=sim.add_parser('evaluate');p.add_argument('--backend',default='http://127.0.0.1:8798')
    p.add_argument('--token-file',required=True);p.add_argument('--operator-token-file',required=True)
    evaluation=sub.add_parser('eval').add_subparsers(dest='operation',required=True)
    p=evaluation.add_parser('run');p.add_argument('--suite',required=True);p.add_argument('--output',required=True)
    p.add_argument('--backend',default='http://127.0.0.1:8798');p.add_argument('--token-file',required=True)
    p.add_argument('--operator-token-file',required=True);p.add_argument('--seeds',type=int,default=10)
    p.add_argument('--seed-start',type=int,default=0)
    model_options(p)
    models=sub.add_parser('models').add_subparsers(dest='operation',required=True)
    p=models.add_parser('show');p.add_argument('--model-config',required=True)
    p=models.add_parser('probe');p.add_argument('--model-config',required=True)
    p.add_argument('--image',required=True);p.add_argument('--output',required=True)
    args=parser.parse_args(argv)
    if args.group=='models':
        from .models.providers import ModelProfile,create_model
        from dataclasses import asdict
        import os
        profile=ModelProfile.load(args.model_config)
        if args.operation=='show':
            result={'profile':asdict(profile),'api_key_present':bool(os.environ.get(profile.api_key_env or '')),
                    'runtime_verified':False,'note':'Configuration only; no inference performed'}
        else:
            output=Path(args.output).resolve()
            output.mkdir(parents=True,exist_ok=False)
            model=create_model(profile,log_dir=output/'calls');model.discover_identity()
            schema={'type':'object','properties':{'description':{'type':'string'},'uncertainty':{'type':'string'}},
                    'required':['description','uncertainty'],'additionalProperties':False}
            result={'identity':model.identity,'result':model.infer('Describe the supplied image. State any uncertainty.',[Path(args.image)],schema),
                    'scope':'single-image transport probe, not task or robot acceptance'}
            (output/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    elif args.group=='demo' and args.operation=='compile':
        from .demonstration.compiler import compile_demo
        from .demonstration.store import DemoStore
        log_dir=Path(args.output).resolve().with_name(Path(args.output).name+'-model-calls')
        model=configured_model(args,log_dir,max_output_tokens=4096)
        detector=None
        if args.detector_onnx:
            from .perception.detector_onnx import TrackedDetector
            detector=TrackedDetector(args.detector_onnx)
        result=compile_demo(args.video,args.output,args.task,model,max_keyframes=args.max_keyframes,cache_dir=args.cache,detector=detector)
        if args.store:
            with DemoStore(args.store) as store:store.register(args.output)
    elif args.group=='demo' and args.operation=='inspect':
        result=load_bundle(args.bundle or args.id,args.store)
    elif args.group=='demo' and args.operation=='find':
        from .demonstration.store import DemoStore
        with DemoStore(args.store) as store:result=store.find(args.task)
    elif args.group=='demo' and args.operation=='evaluate':
        from .demonstration.annotations import evaluate_annotations,write_annotation_report
        annotation=json.loads(Path(args.annotations).read_text(encoding='utf-8'))
        result=evaluate_annotations(load_bundle(args.bundle,args.store),annotation)
        write_annotation_report(result,annotation,args.output)
    elif args.group=='policy':
        from .policy.executor import PiperExecutor
        from .policy.runner import run_policy
        output=Path(args.output).resolve()
        model=configured_model(args,output.with_name(output.name+'-model-calls'))
        executor=PiperExecutor(args.backend,args.token_file)
        config=json.loads(Path(args.skill_config).read_text()) if args.skill_config else None
        try:
            result=run_policy(executor,model,task=args.task,output_dir=output,
                demo=load_bundle(args.demo,args.store) if args.demo else None,max_decisions=args.max_decisions,skill_config=config)
        finally:executor.close()
    elif args.group=='eval':
        from .policy.executor import PiperExecutor
        from .policy.evaluate import run_suite
        if args.seeds<1:raise ValueError('seeds must be positive')
        suite=json.loads(Path(args.suite).read_text(encoding='utf-8'))
        executor=PiperExecutor(args.backend,args.token_file,operator_token_file=args.operator_token_file)
        try:
            def factory(log):
                return configured_model(args,log)
            result=run_suite(executor,factory,suite,args.output,load_bundle,seeds=range(args.seed_start,args.seed_start+args.seeds))
        finally:executor.close()
    elif args.operation=='start':
        from piperx_middleware.cli import initialize
        root=Path(args.root).resolve()
        if not (root/'config.json').exists():
            initialize(root,'mujoco',port=args.port,simulation_seed=args.seed,
                       tcp_offset_m=[0.,0.,.1425],tcp_offset_rpy_deg=[0.,0.,0.])
        else:
            settings=json.loads((root/'config.json').read_text())
            if settings['backend']!='mujoco' or settings['port']!=args.port or settings.get('simulation_seed')!=args.seed:
                raise ValueError('Existing simulation configuration differs; choose a new --root')
        return subprocess.call([sys.executable,'-m','piperx_middleware.cli','--root',str(root),'serve'])
    else:
        from .policy.executor import PiperExecutor
        executor=PiperExecutor(args.backend,args.token_file,operator_token_file=args.operator_token_file)
        try:result=executor.evaluate()
        finally:executor.close()
    print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))
    if isinstance(result,dict) and (result.get('status') in ('error','execution_failed','cancelled','needs_demo_evidence','completion_recheck_failed') or result.get('outcome_verdict')=='unresolved'):
        return 2
    return 0
