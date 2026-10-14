"""One-terminal operator shell; only this layer holds the operator credential."""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import httpx

from .console_bridge import MCPBridge
from .managed_runtime import RuntimeManager
from .models import DomainError, RuntimeParameters
from .profiles import list_remotes, save_remote


def checked(result):
    if isinstance(result, dict) and 'error' in result:
        error = result['error']
        raise DomainError(error.get('code', 'request_failed'), error.get('message', 'Request failed'))
    return result


class ManagedConsole:
    def __init__(self, base, mode, target, simulation_backend='mujoco'):
        self.base, self.mode, self.target = Path(base), mode, target
        self.simulation_backend = simulation_backend
        self.manager = RuntimeManager(self.base, mode, target, simulation_backend=simulation_backend)
        self.connection = None
        self.bridge = None
        self.operator = None
        self.controller = None
        self.session_id = None
        self._lifecycle = asyncio.Lock()
        self._released = False

    async def open(self):
        self.connection = await self.manager.ensure()
        try:
            token = self.connection.operator_token_file.read_text(encoding='utf-8').strip()
            self.operator = httpx.AsyncClient(base_url=self.connection.url,
                headers={'Authorization': 'Bearer ' + token}, trust_env=False,
                follow_redirects=False, timeout=httpx.Timeout(10, connect=3))
            self.bridge = MCPBridge(self.connection.url, self.connection.model_token_file)
            await self._acquire()
        except BaseException:
            if self.operator is not None:
                await self.operator.aclose()
            if self.connection.owned:
                await self.manager.release(self.connection)
            raise
        self._released = False
        return self.bridge

    async def _acquire(self):
        result = checked(await self.operator_call('POST', '/operator/session', {
            'owner': f'piper-robot-{os.getpid()}', 'shutdown_on_loss': self.connection.owned}))
        self.session_id = result['session_id']
        self.bridge.session_id = self.session_id

    async def operator_call(self, method, path, body=None):
        if not path.startswith('/operator/') or path.startswith('//'):
            raise ValueError('Expected operator path')
        if self.operator is None:
            raise DomainError('not_connected', 'Executor is not connected.')
        try:
            response = await self.operator.request(method, path, json=body,
                headers={'X-Piper-Control-Session': self.session_id} if self.session_id else {})
            result = response.json()
            if not isinstance(result, dict) or response.is_redirect:
                raise ValueError('Expected JSON object')
            return result
        except (httpx.TransportError, ValueError):
            return {'error': {'code': 'transport_unknown',
                'message': 'Response lost; query state/request ID before another action. No automatic replay.'}}

    async def reconnect(self):
        """Explicit operator recovery. Never retry an action or resume a model turn."""
        async with self._lifecycle:
            if self.session_id:
                result = await self.operator_call('POST', '/operator/session/heartbeat', {'session_id': self.session_id})
                if 'error' not in result:
                    return result
                if result['error'].get('code') not in ('session_expired', 'missing_control_session', 'transport_unknown'):
                    checked(result)
            old_bridge, old_operator = self.bridge, self.operator
            if old_bridge:
                await old_bridge.close()
            if old_operator:
                await old_operator.aclose()
            self.session_id = None
            await self.open()
            await self.bridge.open()
            self._bind_controller()
            return {'status': 'reconnected', 'replayed_actions': 0}

    def _bind_controller(self):
        if self.controller is not None:
            from .model_agent import load_model_config
            self.controller.bridge = self.bridge
            self.controller.root = self.connection.profile_root
            self.controller.model_config = load_model_config(self.connection.profile_root)
            self.controller.messages = []
            self.controller.cached_status = None

    async def shutdown(self):
        async with self._lifecycle:
            if self._released or self.connection is None:
                return {'status': 'released'}
            if self.session_id:
                result = await self.operator_call('POST', '/operator/session/release', {'session_id': self.session_id})
                # Lost release response is not evidence of success. The session will expire;
                # managed runtime still verifies idleness and identity before shutdown.
                error = result.get('error', {}).get('code')
                if error and error not in ('session_expired', 'missing_control_session', 'transport_unknown'):
                    checked(result)
                self.session_id = None
            result = await self.manager.release(self.connection)
            if result.get('status') in ('busy', 'shutdown_unconfirmed'):
                raise DomainError('shutdown_unconfirmed', 'Executor has not confirmed a drained shutdown; inspect /status.')
            self._released = True
            return result

    async def close(self):
        if self.bridge:
            await self.bridge.close()
        if self.operator:
            await self.operator.aclose()

    async def switch(self, mode, target):
        if mode not in ('simulation', 'real'):
            raise ValueError('Invalid mode')
        await self.shutdown()
        await self.close()
        self.mode, self.target = mode, target
        self.manager = RuntimeManager(self.base, mode, target, simulation_backend=self.simulation_backend)
        self.connection = None
        self.session_id = None
        await self.open()
        await self.bridge.open()
        self._bind_controller()
        return {'mode': mode, 'target': target}

    async def remotes(self):
        return await asyncio.to_thread(list_remotes, self.base)

    async def save_remote(self, name, host):
        return await asyncio.to_thread(save_remote, self.base, name, host)

    async def configure(self, values):
        if not values:
            return {'profile': checked(await self.operator_call('GET', '/operator/settings')),
                    'runtime': checked(await self.bridge.rest('GET', '/v1/parameters'))}
        firmware = {"payload", "collision_rating", "joint_acc_rad_s2"}
        if not set(values) <= set(RuntimeParameters.model_fields) or not set(values) & firmware:
            return checked(await self.operator_call("PATCH", "/operator/settings", {"changes": values}))
        request = RuntimeParameters.model_validate(values)
        return checked(await self.operator_call('PATCH', '/operator/parameters', request.model_dump(exclude_none=True)))


async def run_managed_console(args, base):
    from .console import HAVE_PROMPT_TOOLKIT, ConsoleCompleter
    from .console_interaction import InteractiveConsoleController
    interactive = sys.stdin.isatty()
    if not interactive and args.mode is None:
        raise SystemExit('Non-interactive startup requires --mode simulation or --mode real.')
    if args.url or args.token_file:
        raise SystemExit('Use a saved /remote profile for the managed shell; --url/--token-file remain available for scripts.')
    if HAVE_PROMPT_TOOLKIT and interactive:
        from prompt_toolkit import PromptSession
        from prompt_toolkit.patch_stdout import patch_stdout
        from prompt_toolkit.history import InMemoryHistory
        selection = PromptSession()
        async def prompt(text, default=''):
            return await selection.prompt_async(text, default=default)
    else:
        async def prompt(text, default=''):
            if not interactive:
                return default
            return await asyncio.to_thread(input, text) or default
    mode, target = args.mode, args.target
    if mode is None:
        while mode not in ('simulation', 'real'):
            value = (await prompt('选择模式：1 仿真 / 2 真机 [1]: ')).strip().lower()
            mode = {'': 'simulation', '1': 'simulation', '2': 'real'}.get(value, value)
        remotes = list_remotes(base)
        print('目标：local' + ''.join(f", {r['name']} ({r['ssh_host']})" for r in remotes))
        target = (await prompt('选择目标 [local]: ')).strip() or 'local'
    managed = ManagedConsole(base, mode, target, args.simulation_backend)
    controller = None
    try:
        bridge = await managed.open()
        controller = InteractiveConsoleController(bridge, managed.connection.profile_root,
            operator_call=managed.operator_call, managed=managed, prompt=prompt if interactive else None)
        managed.controller = controller
        await controller.start()
        print(f'模式 {mode} | 目标 {target} | /connect 连接设备 | /help 查看命令')
        async def handle(line):
            try:
                return await controller.handle_line(line)
            except (DomainError, ValueError) as exc:
                print(str(exc)); return True
        if interactive and HAVE_PROMPT_TOOLKIT:
            def toolbar():
                status = controller.cached_status or {}
                return (f' {managed.mode} | {managed.target} | approval={controller.approval_mode}'
                        f" | CAN={(status.get('connection') or {}).get('status', 'disconnected')} | ready={status.get('ready', False)} ")
            session = PromptSession(history=InMemoryHistory(), completer=ConsoleCompleter(controller),
                                    bottom_toolbar=toolbar, refresh_interval=1)
            with patch_stdout():
                while not getattr(controller, 'exit_requested', False):
                    task = asyncio.create_task(session.prompt_async('piper-robot> '))
                    try:
                        while not task.done() and not getattr(controller, 'exit_requested', False):
                            await asyncio.wait({task}, timeout=.1)
                        if getattr(controller, 'exit_requested', False):
                            task.cancel(); await asyncio.gather(task, return_exceptions=True); break
                        if not await handle(task.result()):
                            break
                    except KeyboardInterrupt:
                        print('使用 /stop 停止动作，/quit 退出。')
                    except EOFError:
                        break
        else:
            # A daemon reader lets /quit exit even when the pipe writer keeps
            # stdin open. An executor-thread readline would block asyncio's
            # shutdown indefinitely, and would hide /stop during draining.
            import threading
            queue = asyncio.Queue()
            loop = asyncio.get_running_loop()
            def read_lines():
                while True:
                    line = sys.stdin.readline()
                    try:
                        loop.call_soon_threadsafe(queue.put_nowait, line)
                    except RuntimeError:
                        return  # Event loop already closed.
                    if not line:
                        return
            threading.Thread(target=read_lines, name='piper-stdin', daemon=True).start()
            while not getattr(controller, 'exit_requested', False):
                try:
                    line = await asyncio.wait_for(queue.get(), .1)
                except asyncio.TimeoutError:
                    continue
                if not line or not await handle(line.strip()):
                    break
    finally:
        # Release the admission session first so pending approvals do not hang exit.
        try:
            await managed.shutdown()
        finally:
            if controller:
                await controller.close()
            await managed.close()
