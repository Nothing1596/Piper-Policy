"""Standard stdio MCP for the existing frontend's robot, video and demo tools."""
import asyncio
import json
import sys

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import CallToolResult, ImageContent, TextContent, Tool

from .agent_console import dispatch, error


def create_frontend_mcp(controller):
    from .task_plan import guide
    server = Server('piper-robot-model', instructions=(
        'Read task_plan(operation=guide,context=live or offline_demo) for the current evidence context. '
        'Operator policy applies; this server cannot configure or approve.\n'+guide()))

    @server.list_tools()
    async def tools():
        return [Tool(**tool) for tool in controller.available_tools()]

    @server.call_tool()
    async def call(name, arguments):
        tool = next((t for t in controller.available_tools() if t['name'] == name), {})
        action = 'request_id' in tool.get('inputSchema', {}).get('required', [])
        if action and controller.background_task and not controller.background_task.done():
            return error('busy', 'An action is already in flight; status/stop remain available')
        if action:
            controller.background_task = asyncio.current_task()
        try:
            result = await dispatch(controller, {'id':'mcp','command':'call','tool':name,'arguments':arguments})
        except asyncio.CancelledError:
            result = error('outcome_unknown','Local call cancelled; query original request_id before any retry')
        finally:
            if action and controller.background_task is asyncio.current_task():
                controller.background_task = None
        try:
            parts = controller.image_inputs(result)
        except (ValueError, OSError) as exc:
            result = error('image_delivery_error', str(exc))
            parts = []
        content = [TextContent(type='text',text=json.dumps(result,ensure_ascii=False))]
        for part in parts:
            if part['type'] == 'text':
                content.append(TextContent(type='text', text=part['text']))
            elif part['type'] == 'image_url':
                header, data = part['image_url']['url'].split(',',1)
                content.append(ImageContent(type='image', mimeType=header.split(';')[0][5:], data=data))
        return CallToolResult(content=content,structuredContent=result,isError=isinstance(result.get('error'),dict))

    return server


async def run_frontend_mcp(args, base):
    from .managed_console import ManagedConsole
    from .console_interaction import InteractiveConsoleController
    from . import __version__
    if args.mode is None:
        raise SystemExit('--agent-mcp requires explicit --mode simulation or real')
    if args.url or args.token_file:
        raise SystemExit('--agent-mcp uses managed profiles; select --target')
    managed = ManagedConsole(base,args.mode,args.target,args.simulation_backend)
    controller = None
    try:
        bridge = await managed.open()
        controller = InteractiveConsoleController(bridge,managed.connection.profile_root,
            emit=lambda line:print(line,file=sys.stderr,flush=True),operator_call=managed.operator_call,managed=managed)
        controller.agent_mode = True
        # Load native tool dependencies before stdio starts a blocking reader.
        # On Windows, first importing NumPy while that reader is active can
        # stall DLL initialization and tools/list until the input is closed.
        controller.available_tools()
        managed.controller = controller
        await controller.start()
        if not bridge.connected:
            raise RuntimeError('Executor MCP is not ready; no model session started. Restart the frontend after checking the executor.')
        server = create_frontend_mcp(controller)
        options = server.create_initialization_options()
        options.server_version = __version__
        async with stdio_server() as (read,write):
            await server.run(read,write,options)
    finally:
        try:
            await managed.shutdown()
        finally:
            if controller:
                await controller.close()
            await managed.close()
