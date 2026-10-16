import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_argon_video_study import healthy


def test_scored_failure_can_continue_infrastructure_cannot():
    r = dict(status='round_budget', graded=True, task_complete=False,
             cleanup=dict(runtime_remaining=False, owned_process_alive=False, recording_error=None))
    assert healthy(r)
    assert not healthy({**r, 'status': 'paused'})
    assert not healthy({**r, 'graded': False})
    assert not healthy({**r, 'cleanup': {'runtime_remaining': True}})
    assert not healthy(r, probe=True)
    assert healthy({**r, 'status': 'native_probe_pass'}, probe=True)


def test_no_time_limit_passes_to_runner_without_altering_other_limits(tmp_path, monkeypatch):
    import asyncio
    import json
    import agy_online_decider
    import run_longvil_online
    from run_argon_video_study import run_case
    config = dict(model='test-no-provider', provider_turn_window=2, host='unused', port=2222,
                  remote_root='/unused', remote_python='/unused/python', remote_scenes='/unused/scenes',
                  max_rounds=80, max_tools=120, max_seconds=None, camera='mujoco-overview',
                  budget_revision='unbounded-task-time-v2')
    task = dict(task_id='fixture', annotation='/unused/reference.json', objects=['fixture-object'],
                scene_sha256='unchanged-scene-hash')
    (tmp_path/'config.json').write_text(json.dumps(config))
    (tmp_path/'tasks.json').write_text(json.dumps([task]))
    class NoProvider:
        def __init__(self, *a, **kw): pass
        async def close(self): pass
    captured = []
    async def no_network(args): captured.append(args)
    monkeypatch.setattr(agy_online_decider, 'AgyDecider', NoProvider)
    monkeypatch.setattr(run_longvil_online, 'main', no_network)
    asyncio.run(run_case(tmp_path, 'baseline', 'fixture'))
    args = captured[0]
    assert args.max_seconds is None
    assert (args.max_rounds, args.max_tools, args.completion_budget) == (80, 120, None)
    assert args.budget_revision == 'unbounded-task-time-v2'
    assert args.scene_sha256 == 'unchanged-scene-hash'


def test_time_budget_failure_stays_recorded_without_retry():
    report = dict(status='time_budget', graded=True, task_complete=False,
                  cleanup=dict(runtime_remaining=False, owned_process_alive=False, recording_error=None))
    assert healthy(report)
