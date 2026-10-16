"""One separately labelled recovery, after the frozen primary job completes.

No old action dispatcher, provider conversation or observation ID is reused.
Historical claims are explicitly hypotheses; the normal runner supplies new RGB-D.
"""
import argparse
import asyncio
import copy
import json
from pathlib import Path
import time

from run_argon_video_study import run_case, verify, save, healthy


def recovery_payload(payload, public_context):
    value = copy.deepcopy(payload)
    value['messages'].insert(1, {'role': 'user', 'content':
        'APPROXIMATE RECOVERY, a new independent experiment segment. '
        'Recorded object/robot poses were restored, but velocities were set to zero, '
        'arm references set to measured hold; gripper initialized to measured hold or an '
        'audited previous closing target for a held-object snapshot; contacts cold-started and time restarted. '
        'This is NOT uninterrupted execution. Last accepted action below already succeeded; '
        'DO NOT replay it or restart completed transfers. Old observation IDs, geometry '
        'and candidate targets are stale. Use the NEW current RGB-D and measured robot '
        'state to decide each next tool call. The previous plan is an UNVERIFIED MODEL '
        'HYPOTHESIS, not a completion certificate; recheck its claims from current pixels.\n'
        + json.dumps(public_context, ensure_ascii=False)})
    return value


async def run(job, probe=False):
    verify(job)
    recovery = json.loads((job/'recovery.json').read_text())
    task = recovery['task']
    if not probe:
        import agy_online_decider
        original = agy_online_decider.AgyDecider
        class RecoveryDecider(original):
            def prepare(self, payload):
                return super().prepare(recovery_payload(payload, recovery['public_context']))
        agy_online_decider.AgyDecider = RecoveryDecider
    await run_case(job, 'recovery', task, probe=probe)
    if probe:
        report = json.loads((job/'probes'/task/'execution/report.json').read_text())
        if not healthy(report, True):
            raise RuntimeError('Recovery native preflight failed')
        return
    execution = job/'recovery'/task/'execution'
    report = json.loads((execution/'report.json').read_text())
    report.update(experiment_segment='approximate_recovery',
                  continuous_with_original=False, excluded_from_primary_comparison=True,
                  recovery_source=recovery['source'], old_actions_replayed=0)
    save(execution/'report.json', report)
    # A separate cross-segment coverage diagnostic, never overwrite the original
    # score or call pose restoration a measured transfer. Private inputs are read
    # only after all model decisions have finished.
    final_path = execution/'final-state.private.json'
    if report.get('graded') and final_path.exists():
        from run_longvil_online import private_grade
        from longvil_metrics import semantic_actions
        annotation = json.loads(Path(recovery['annotation']).read_text())
        old = json.loads(Path(recovery['prior_transfers']).read_text())
        new_path = execution/'transfers.private.json'
        new = json.loads(new_path.read_text()) if new_path.exists() else []
        grade = private_grade(semantic_actions(annotation['code'], annotation['object_list']),
                              old+new, final_state=json.loads(final_path.read_text()))
        save(execution/'cross-segment-coverage.private.json', {
            'scope': 'historical controlled transfers plus new recovery transfers; not continuous execution',
            'historical_transfer_count': len(old), 'new_transfer_count': len(new),
            'grade': grade, 'final_stability': report.get('stable_after_retreat'),
            'continuous_task_success': None, 'primary_benchmark_score': None})
    if not healthy(report):
        raise RuntimeError('Recovery infrastructure/scoring/cleanup requires review')
    save(job/'state.json', {'status': 'complete', 'segment': 'approximate_recovery',
                            'report': str(execution/'report.json')})


def queued(job):
    verify(job)
    recovery = json.loads((job/'recovery.json').read_text())
    parent = Path(recovery['wait_for_job'])
    save(job/'state.json', {'status': 'waiting_for_primary_job', 'job': str(parent)})
    waiting_on_pause = False
    while True:
        state = json.loads((parent/'state.json').read_text())
        if state['status'] == 'complete':
            break
        if state['status'] == 'paused':
            if not waiting_on_pause:
                save(job/'state.json', {'status': 'waiting_for_explicit_primary_resume',
                    'job': str(parent), 'primary_reason': state.get('reason'),
                    'model_calls': 0, 'actions_dispatched': 0})
            waiting_on_pause = True
            time.sleep(30)
            continue
        waiting_on_pause = False
        import subprocess
        if subprocess.run(['tmux', 'has-session', '-t', recovery['wait_for_session']],
                          capture_output=True).returncode:
            raise RuntimeError('Primary supervisor exited without completion')
        time.sleep(30)
    save(job/'state.json', {'status': 'running', 'segment': 'approximate_recovery'})
    asyncio.run(run(job))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=['probe', 'queued'])
    parser.add_argument('--job', type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.operation == 'probe':
            asyncio.run(run(args.job, probe=True))
        else:
            queued(args.job)
    except Exception as exc:
        save(args.job/'state.json', {'status': 'paused', 'reason': str(exc),
                                    'automatic_restart': False})
        raise


if __name__ == '__main__':
    main()
