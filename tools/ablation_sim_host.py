"""Experiment-only Windows helper. No real backend; evaluator is never a model tool."""
import argparse
import json
from pathlib import Path
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument('operation', choices=['start', 'score', 'check'])
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--seed', type=int, default=401)
    a = p.parse_args()
    root = a.root.resolve()
    profile = root / 'profiles/simulation/local'
    if a.operation == 'start':
        root.mkdir(parents=True, exist_ok=False)
        from piperx_middleware.profiles import ensure_profile
        ensure_profile(root, 'simulation', simulation_backend='mujoco')
        path = profile / 'config.json'
        config = json.loads(path.read_text())
        config.update(simulation_seed=a.seed, tcp_offset_m=[0., 0., .1425], tcp_offset_rpy_deg=[0., 0., 0.])
        from piperx_middleware.models import Settings
        Settings.model_validate(config)
        path.write_text(json.dumps(config), encoding='utf-8')
        from piperx_middleware.cli import main as cli
        sys.argv = ['piper-robot', '--root', str(root), '--mode', 'simulation', '--simulation-backend', 'mujoco', '--agent-mcp']
        cli()
    elif a.operation == 'check':
        print(json.dumps({'runtime_remaining': (profile / 'runtime.json').exists()}))
    else:
        from piperx_middleware.client import RobotClient
        runtime = json.loads((profile / 'runtime.json').read_text())
        config = json.loads((profile / 'config.json').read_text())
        if runtime['backend'] != 'mujoco' or config['backend'] != 'mujoco':
            raise ValueError('Experiment requires MuJoCo')
        client = RobotClient(runtime['url'], profile / 'model.token')
        operator = RobotClient(runtime['url'], profile / 'operator.token')
        try:
            caps = client.call('GET', '/v1/capabilities')
            if caps.get('instance_id') != runtime['instance_id']:
                raise ValueError('Executor identity mismatch')
            truth = operator.call('GET', '/operator/simulation/evaluate')
            if 'error' in truth:
                raise ValueError('Operator evaluation failed')
            state = client.call('GET', '/v1/state')
            print(json.dumps({'truth': truth, 'tcp': state.get('tcp'), 'instance_id': runtime['instance_id']}))
        finally:
            client.close()
            operator.close()


if __name__ == '__main__':
    main()
