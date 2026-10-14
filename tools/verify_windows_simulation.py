"""Exercise the installed single-terminal controller against MuJoCo only.

Run with the installed environment's Python, away from the source checkout.
No real backend is selectable. Evidence excludes credentials and session IDs.
"""
import argparse
import asyncio
import base64
import json
import platform
import sys
import time
from pathlib import Path

from piperx_middleware.console_interaction import InteractiveConsoleController
from piperx_middleware.managed_console import ManagedConsole, checked


async def verify(output):
    output.mkdir(parents=True, exist_ok=False)
    report = {'platform': platform.platform(), 'python': sys.version,
              'backend': 'mujoco', 'physical_hardware': False, 'actions': []}
    messages = []
    m = ManagedConsole(output / 'isolated-profile', 'simulation', 'local', 'mujoco')
    c = None

    async def manual(text, request_id):
        await c.handle_line('/manual ' + text)
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            result = await m.bridge.rest('GET', '/v1/requests/' + request_id)
            if 'job_id' in result:
                break
            await asyncio.sleep(.05)
        else:
            raise AssertionError('Manual command did not create job: ' + repr(messages[-5:]))
        assert result['status'] == 'awaiting_approval', result
        await c.handle_line('/status')
        await c.handle_line('/approve ' + result['job_id'])
        if c.background_task:
            await asyncio.wait_for(asyncio.shield(c.background_task), 30)
        final = checked(await m.bridge.rest('GET', '/v1/requests/' + request_id))
        assert final['status'] == 'succeeded', final
        state = checked(await m.bridge.rest('GET', '/v1/state'))
        report['actions'].append({'request_id': request_id, 'job_id': final['job_id'],
                                  'status': final['status'], 'robot': state.get('robot')})
        print(request_id + ': succeeded (approval + original-ID lookup)', flush=True)

    async def image(name):
        observation = checked(await m.bridge.rest('GET', '/v1/simulation/observation'))
        rgb = base64.b64decode(observation.pop('rgb_jpeg_b64'))
        depth = base64.b64decode(observation.pop('depth_npy_b64'))
        (output / (name + '.jpg')).write_bytes(rgb)
        (output / (name + '-depth.npy')).write_bytes(depth)
        (output / (name + '.json')).write_text(json.dumps(observation, indent=2), encoding='utf-8')
        print(name + ': RGB-D captured', flush=True)

    try:
        bridge = await m.open()
        c = InteractiveConsoleController(bridge, m.connection.profile_root, emit=messages.append,
                                        operator_call=m.operator_call, managed=m)
        m.controller = c
        await c.start()
        await c.handle_line('/connect')
        state = checked(await bridge.rest('GET', '/v1/state'))
        assert state['ready'], state
        report['tools'] = [t['name'] for t in bridge.tools]
        report['initial_robot'] = state.get('robot')
        report['instance_id'] = m.connection.instance_id
        print('MuJoCo connected; MCP tools=' + str(len(report['tools'])), flush=True)
        try:
            await image('before')
        except Exception as exc:
            report['render_error'] = str(exc)
            print('RGB-D unavailable: ' + str(exc), flush=True)
        await c.handle_line('/approval always')
        code = next(iter(c._proposals))
        await c.handle_line('/confirm ' + code)
        await manual('robot_move_joints(joints_deg=[5,0,0,0,0,0], speed_percent=5, request_id="win-sim-joint-001")', 'win-sim-joint-001')
        await manual('robot_gripper(width_m=0.03, effort_protocol=0.5, request_id="win-sim-gripper-001")', 'win-sim-gripper-001')
        if 'render_error' not in report:
            await image('after')
        await manual('robot_move_joints(joints_deg=[0,0,0,0,0,0], speed_percent=5, request_id="win-sim-home-001")', 'win-sim-home-001')
        await c.handle_line('/quit')
        await asyncio.wait_for(c._drain_task, 30)
        assert c.exit_requested
        assert not (m.connection.profile_root / 'runtime.json').exists()
        report['exit'] = 'drained; runtime record removed'
        report['status'] = 'passed'
    except BaseException as exc:
        report['status'] = 'failed'
        report['error'] = str(exc)
        raise
    finally:
        try:
            await m.shutdown()
        finally:
            if c:
                await c.close()
            await m.close()
            (output / 'result.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
            (output / 'console.txt').write_text('\n'.join(messages), encoding='utf-8')
    print(json.dumps({'status': report['status'], 'actions': len(report['actions']), 'exit': report['exit']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    asyncio.run(verify(parser.parse_args().output))
