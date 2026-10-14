"""JSON-lines model frontend; uses the existing managed executor and tool schemas.

This interface exposes model tools, not operator policy/approval credentials.
There is no new controller, action retry loop, or independent CAN connection.
"""
import asyncio
import json
import sys
import threading

from .live_camera import compact_state


CAMERA_TOOLS = [
    {'name': 'camera_observe', 'description': 'Observe current RGB-D and optionally answer a saved plan question using measured geometry. Real/local selects live frames; MuJoCo returns one fresh simulated snapshot, without video history. Optional pixel ROI. Inspect geometry frame and provenance; visual changes are not contact proof.',
     'inputSchema': {'type': 'object', 'additionalProperties': False, 'properties': {
         'question_id': {'type': 'string'},
         'window_s': {'type': 'number', 'exclusiveMinimum': 0, 'maximum': 120, 'default': 2},
         'max_frames': {'type': 'integer', 'minimum': 1, 'maximum': 6, 'default': 3},
         'roi': {'type': 'array', 'minItems': 4, 'maxItems': 4, 'items': {'type': 'integer', 'minimum': 0}},
         'serial': {'type': 'string', 'minLength': 1}}}},
    {'name': 'camera_video', 'description': 'Start/reuse, inspect, or stop the frontend-owned live camera. No robot motion.',
     'inputSchema': {'type': 'object', 'additionalProperties': False, 'properties': {
         'operation': {'enum': ['start', 'status', 'stop'], 'default': 'status'},
         'serial': {'type': 'string', 'minLength': 1}}}}
]


def error(code, message):
    return {'error': {'code': code, 'message': message}}


async def dispatch(controller, request):
    if not isinstance(request, dict) or not isinstance(request.get('id'), (str, int)) or isinstance(request.get('id'), bool):
        raise ValueError('Every line requires a string/integer id and command')
    command = request.get('command')
    allowed = {'call': {'id','command','tool','arguments'}, 'status': {'id','command','job_id'},
               'tools': {'id','command'}, 'observe': {'id','command','arguments'},
               'video': {'id','command','arguments'}, 'quit': {'id','command'}}
    if command not in allowed or not set(request) <= allowed[command]:
        raise ValueError('Unknown command or fields; use call/status/tools/observe/video/quit; demonstrations use call demo_reference')
    if command == 'quit':
        return {'status': 'draining'}
    if command == 'tools':
        return {'tools': controller.available_tools()}
    if command == 'status':
        if request.get('job_id'):
            return await controller.invoke_tool('robot_status', {'job_id': request['job_id']})
        return compact_state(await controller.bridge.rest('GET', '/v1/state'))
    if command in ('observe', 'video'):
        return await controller.invoke_tool('camera_'+command, request.get('arguments', {}))
    name, arguments = request.get('tool'), request.get('arguments', {})
    if not isinstance(name, str) or not isinstance(arguments, dict):
        raise ValueError('call requires tool string and arguments object')
    if name == 'robot_stop':
        # Cancel preparation/polling immediately, then send stop without waiting
        # for slow camera subprocess cleanup. Backend retains action ownership.
        task = controller.background_task
        if task and not task.done():
            task.cancel()
        return await controller.invoke_tool(name, arguments)
    tool = next((t for t in controller.available_tools() if t['name'] == name), None)
    if tool and 'request_id' in tool['inputSchema'].get('required', []) and not arguments.get('request_id'):
        return error('request_id_required', 'Model actions require an explicit stable request_id. Query unknown outcomes; never resend with a new ID.')
    return await controller.invoke_tool(name, arguments)


async def run_agent_console(args, base):
    from .managed_console import ManagedConsole
    from .console_interaction import InteractiveConsoleController
    if args.mode is None:
        raise SystemExit('--agent requires explicit --mode simulation or real')
    if args.url or args.token_file:
        raise SystemExit('--agent uses saved profiles; select --target')
    managed = ManagedConsole(base, args.mode, args.target, args.simulation_backend)
    controller, tasks = None, set()

    def emit(ident, result, content=None):
        reply = {'id': ident, 'result': result}
        if content:
            reply['content'] = content
        print(json.dumps(reply, ensure_ascii=False, allow_nan=False), flush=True)

    try:
        bridge = await managed.open()
        controller = InteractiveConsoleController(bridge, managed.connection.profile_root,
            emit=lambda message: print(message, file=sys.stderr, flush=True),
            operator_call=managed.operator_call, managed=managed)
        controller.agent_mode = True
        # Resolve native dependencies before the background stdin reader starts.
        controller.available_tools()
        managed.controller = controller
        await controller.start()
        if not bridge.connected:
            raise RuntimeError('Executor MCP is not ready; no model session started. Restart the frontend after checking the executor.')
        emit(None, {'schema': 'piper.agent.v1', 'status': 'ready', 'mode': args.mode,
                    'target': args.target, 'approval': 'operator policy enforced; this interface cannot approve or configure',
                    'commands': ['tools', 'status', 'call', 'video', 'observe', 'quit']})
        queue, loop = asyncio.Queue(maxsize=64), asyncio.get_running_loop()
        def read():
            while True:
                line = sys.stdin.readline(65537)
                if len(line) > 65536 and not line.endswith('\n'):
                    while line and not line.endswith('\n'):
                        line = sys.stdin.readline(65537)
                    line = '{invalid oversized request}\n'
                try:
                    asyncio.run_coroutine_threadsafe(queue.put(line), loop).result()
                except Exception:
                    return
                if not line:
                    return
        threading.Thread(target=read, name='piper-agent-input', daemon=True).start()
        in_flight_ids = set()
        async def handle(request):
            ident = request.get('id') if isinstance(request, dict) else None
            try:
                result = await dispatch(controller, request)
                emit(ident, result, controller.image_inputs(result))
            except asyncio.CancelledError:
                emit(ident, error('outcome_unknown', 'Local request cancelled; query its stable request_id/job_id before retrying'))
                raise
            except Exception as exc:
                emit(ident, error('invalid_request', str(exc)))
            finally:
                in_flight_ids.discard(ident)
        while True:
            line = await queue.get()
            if not line:
                break
            try:
                request = json.loads(line, parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))
                ident = request.get('id') if isinstance(request, dict) else None
                if not isinstance(ident, (str, int)) or isinstance(ident, bool):
                    raise ValueError('Request id must be string/integer')
                if ident in in_flight_ids:
                    emit(ident, error('duplicate_id', 'This line id is already in flight'))
                    continue
                if request.get('command') == 'quit':
                    emit(ident, await dispatch(controller, request))
                    break
                if len(tasks) >= 16:
                    emit(ident, error('busy', 'Too many outstanding requests'))
                    continue
                tool = next((t for t in controller.available_tools() if t['name'] == request.get('tool')), {})
                action = request.get('command') == 'call' and 'request_id' in tool.get('inputSchema', {}).get('required', [])
                if action and controller.background_task and not controller.background_task.done():
                    emit(ident, error('busy', 'An action is already in flight; status and stop remain available'))
                    continue
                in_flight_ids.add(ident)
                task = asyncio.create_task(handle(request))
                if action:
                    controller.background_task = task
                tasks.add(task)
                task.add_done_callback(tasks.discard)
            except (ValueError, TypeError) as exc:
                emit(None, error('invalid_json', str(exc)))
    finally:
        try:
            await managed.shutdown()  # Drain accepted actions; release pending approvals.
            if tasks:
                await asyncio.wait(tasks, timeout=5)
        finally:
            for task in list(tasks):
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            if controller:
                await controller.close()
            await managed.close()
