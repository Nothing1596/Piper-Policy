"""Native MuJoCo acceptance through the normal CLI/MCP, with no model calls.

Uses a fresh instance of the already-frozen vegetable scene. A deliberately
offset corn approach checks a measured blocked result, then a NEW upward
request checks recovery. No reset, object teleport, CAN or physical hardware.
"""
import argparse
import asyncio
import base64
from datetime import timedelta
import json
import os
from pathlib import Path
import shlex
import uuid

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from piperx_middleware.model_agent import is_uncertain_result
from run_longvil_physics import remote, ssh_args


async def validate(job, output):
    from types import SimpleNamespace
    config = json.loads((job / 'launch-config.json').read_text())
    task = json.loads((job / 'tasks.json').read_text())['level1_4_vegetablesorting_4']
    args = SimpleNamespace(**{k: config[k] for k in ('host', 'port', 'socket', 'remote_root', 'remote_python')})
    args.profile = args.remote_root + '/acceptance/' + uuid.uuid4().hex
    args.remote_helper = args.remote_root + '/source/tools/longvil_physics_host.py'
    pythonpath = args.remote_root + '/source/piperx-cli/src:' + args.remote_root + '/source/piper-lab/src'
    command = shlex.join(['env', 'MUJOCO_GL=egl', 'PYOPENGL_PLATFORM=egl', 'PYTHONPATH=' + pythonpath,
                         args.remote_python, args.remote_helper, 'start', '--root', args.profile, '--scene', task['scene']])
    params = StdioServerParameters(command='ssh', args=[*ssh_args(args), command], env={'PATH': os.environ.get('PATH', '')})
    output.mkdir(parents=True, exist_ok=False)
    report = {'scope': 'Native CLI/MCP simulation acceptance; no model or hardware',
              'scene_sha256': task['scene_sha256'], 'remote_profile': args.profile, 'passed': False}
    rows = []

    def save():
        (output / 'events.json').write_text(json.dumps(rows, indent=2, allow_nan=False))
        (output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False))

    async def call(session, name, body):
        rows.append({'name': name, 'arguments': body, 'dispatched': True})
        save()
        reply = await session.call_tool(name, body)
        value = reply.structuredContent or {}
        rows[-1]['result'] = value
        save()
        if is_uncertain_result(value)[0]:
            raise RuntimeError('Unknown action result; acceptance stops without replay')
        return reply, value

    async def capture(session, label):
        reply, value = await call(session, 'camera_observe', {'max_frames': 1})
        frames = value.get('frames', [])
        images = [c for c in reply.content if c.type == 'image']
        if reply.isError or len(frames) != 1 or len(images) != 1 or not value.get('simulated'):
            raise RuntimeError('Native image/metadata mismatch')
        (output / (label + '.jpg')).write_bytes(base64.b64decode(images[0].data))
        (output / (label + '.json')).write_text(json.dumps(value, indent=2, allow_nan=False))
        return frames[0]

    try:
        with (output / 'stderr.log').open('w') as err:
            async with stdio_client(params, errlog=err) as (read, write):
                async with ClientSession(read, write, read_timeout_seconds=timedelta(seconds=120)) as session:
                    await session.initialize()
                    reply, connected = await call(session, 'robot_connect', {})
                    if reply.isError:
                        raise RuntimeError('Native connection failed')
                    _, status = await call(session, 'robot_status', {})
                    if status.get('backend') != 'mujoco':
                        raise RuntimeError('Acceptance refuses non-MuJoCo')
                    initial = await capture(session, 'initial')
                    report['depth_filter'] = initial.get('depth_filter', {})
                    if (report['depth_filter'].get('applied') is not True
                            or report['depth_filter'].get('filtered_pixels', 0) <= 0):
                        raise RuntimeError('Native capture did not apply the robot self-occlusion mask')
                    _, approach = await call(session, 'move_to', {
                        'xyz_m': [.12, .14, -.10], 'rpy_deg': [180., 0., 0.],
                        'speed_percent': 10, 'timeout_s': 60, 'request_id': 'accept-approach-0001'})
                    if approach.get('status') != 'succeeded':
                        raise RuntimeError('Clear approach did not complete')
                    before = await capture(session, 'before-blocked')
                    _, blocked = await call(session, 'move_linear', {
                        'xyz_m': [.12, .14, -.185], 'speed_percent': 10, 'timeout_s': 60,
                        'request_id': 'accept-contact-0002'})
                    if (blocked.get('status') != 'failed' or blocked.get('error_code') != 'motion_blocked'
                            or blocked.get('recoverable') is not True
                            or blocked.get('stop_result', {}).get('confirmed_stopped') is not True):
                        raise RuntimeError('Contact did not produce a measured recoverable blocked result')
                    after = await capture(session, 'after-blocked')
                    if (after['instance_id'], after['connection_epoch']) != (before['instance_id'], before['connection_epoch']):
                        raise RuntimeError('Simulator identity changed during recovery')
                    if after['sim_time_s'] <= before['sim_time_s']:
                        raise RuntimeError('No new physical feedback after blocked action')
                    measured = blocked['after']
                    q = measured.get('robot', measured)['q_deg']
                    from piperx_middleware.kinematics import PiperKinematics
                    actual = PiperKinematics([0., 0., .1425]).pose(q)['xyz_m']
                    _, recovered = await call(session, 'move_linear', {
                        'xyz_m': [actual[0], actual[1], actual[2] + .06],
                        'speed_percent': 10, 'timeout_s': 60, 'request_id': 'accept-retract-0003'})
                    if recovered.get('status') != 'succeeded':
                        raise RuntimeError('New measured upward recovery did not complete')
                    await capture(session, 'after-recovery')
                    report.update(passed=True, blocked_result=blocked, recovery_result=recovered,
                                  new_request_after_confirmed_stop=True, unknown_replays=0)
    except Exception as exc:
        report['error'] = {'type': type(exc).__name__, 'message': str(exc)[:600]}
    finally:
        try:
            report['cleanup'] = await remote(args, 'check', args.profile)
            if report['cleanup']['runtime_remaining']:
                report['fallback_stop'] = await remote(args, 'stop', args.profile)
                report['cleanup'] = await remote(args, 'check', args.profile)
            if (report['cleanup']['runtime_remaining'] or report['cleanup']['owned_process_alive']
                    or report['cleanup']['recording_error']):
                report['passed'] = False
        except Exception as exc:
            report.update(passed=False, cleanup_error=str(exc)[:400])
        save()
    print(json.dumps({'passed': report['passed'], 'error': report.get('error'), 'cleanup': report.get('cleanup')}))
    return report['passed']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    options = parser.parse_args()
    raise SystemExit(0 if asyncio.run(validate(options.job, options.output)) else 1)
