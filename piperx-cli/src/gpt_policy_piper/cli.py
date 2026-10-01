"""Launch the pinned external runtime without vendoring or monkey-patching it."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
from .prepare import verify_prepared


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--upstream',type=Path,required=True)
    parser.add_argument('--allow-model-calls',action='store_true')
    args,remaining = parser.parse_known_args()
    root = verify_prepared(args.upstream)
    if '--help' not in remaining and '--check' not in remaining and not args.allow_model_calls:
        raise SystemExit('Model calls may incur provider charges. Add --allow-model-calls only when you intend to run them; --check is offline.')
    sys.path.insert(0,str(root/'src'))
    if '--help' in remaining:
        sys.argv = ['piper-gpt',*remaining]
        from gpt_policy.main import main as upstream_main
        upstream_main()
        return
    from gpt_policy.settings import load_settings
    selected = argparse.ArgumentParser(add_help=False)
    selected.add_argument('--config',type=Path)
    config_args,_ = selected.parse_known_args(remaining)
    config = config_args.config or root/'configs/piper-mujoco.local.json'
    settings = load_settings(config)
    if settings.get('backend') != 'piper_mujoco' or settings.get('runtime',{}).get('right_interface'):
        raise SystemExit('This entrypoint accepts only the single-arm Piper MuJoCo profile')
    if not config_args.config:
        remaining = ['--config',str(config),*remaining]
    sys.argv = ['piper-gpt',*remaining]
    from gpt_policy.main import main as upstream_main
    upstream_main()

if __name__ == '__main__':
    main()
