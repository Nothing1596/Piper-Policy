"""Actual local executor/MCP processes; simulation only, isolated profiles."""
import asyncio
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import httpx
import pytest

from piperx_middleware.managed_console import ManagedConsole, checked
from piperx_middleware.managed_runtime import RuntimeManager
from piperx_middleware.console_interaction import InteractiveConsoleController


@pytest.mark.asyncio
async def test_real_process_shell_approval_and_resource_release(tmp_path):
    m=ManagedConsole(tmp_path,'simulation','local','sim')
    c=None
    try:
        bridge=await m.open()
        c=InteractiveConsoleController(bridge,m.connection.profile_root,emit=lambda _:None,
                                      operator_call=m.operator_call,managed=m)
        m.controller=c
        await c.start()
        await c.handle_line('/connect')
        assert (await bridge.rest('GET','/v1/state'))['ready']
        await c.handle_line('/approval always')
        code=next(iter(c._proposals))
        await c.handle_line('/confirm '+code)
        result=checked(await bridge.call('robot_move_joints', {'joints_deg':[1.,0.,0.,0.,0.,0.], 'request_id':'process-mcp-job1'}))
        assert result['status']=='awaiting_approval'
        await c.handle_line('/status')
        await c.handle_line('/approve '+result['job_id'])
        for _ in range(100):
            j=await bridge.rest('GET','/v1/jobs/'+result['job_id'])
            if j['status'] not in ('accepted','running'): break
            await asyncio.sleep(.02)
        assert j['status']=='succeeded',j
        await c.handle_line('/quit')
        assert c._draining and not c.exit_requested
        await asyncio.wait_for(c._drain_task,10)
        assert c.exit_requested
        assert not (m.connection.profile_root/'runtime.json').exists()
    finally:
        await m.shutdown()
        if c: await c.close()
        await m.close()


@pytest.mark.asyncio
async def test_two_runtime_managers_reuse_one_process_and_shared_detach(tmp_path):
    first=RuntimeManager(tmp_path,'simulation',simulation_backend='sim')
    second=RuntimeManager(tmp_path,'simulation',simulation_backend='sim')
    a=await first.ensure()
    try:
        b=await second.ensure()
        assert a.instance_id==b.instance_id and a.url==b.url
        assert a.owned and not b.owned
        assert (await second.release(b))['status']=='detached'
        async with httpx.AsyncClient(trust_env=False) as client:
            assert (await client.get(a.url+'/health')).json()['instance_id']==a.instance_id
    finally: await first.release(a)


def test_no_tty_requires_explicit_mode(tmp_path):
    result=subprocess.run([sys.executable,'-m','piperx_middleware.standalone_cli','--root',str(tmp_path)],
                          input='',text=True,capture_output=True,timeout=10)
    assert result.returncode!=0 and 'requires --mode' in result.stderr
    assert not list(tmp_path.rglob('runtime.json'))


def test_frontend_crash_releases_owned_executor_after_session_loss(tmp_path):
    program='''import asyncio,json,sys
from pathlib import Path
from piperx_middleware.managed_console import ManagedConsole
async def main():
 m=ManagedConsole(Path(sys.argv[1]),'simulation','local','sim')
 b=await m.open(); await b.open(); await b.call('robot_connect',{})
 print(json.dumps({'url':m.connection.url,'root':str(m.connection.profile_root)}),flush=True)
 await asyncio.sleep(60)
asyncio.run(main())
'''
    process=subprocess.Popen([sys.executable,'-c',program,str(tmp_path)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    record=None
    try:
        # Child bootstrap is bounded by RuntimeManager startup timeout.
        line=process.stdout.readline()
        assert line,process.stderr.read()
        record=json.loads(line)
        process.terminate();process.wait(timeout=5)
        deadline=time.monotonic()+12
        while time.monotonic()<deadline and (Path(record['root'])/'runtime.json').exists():
            time.sleep(.1)
        assert not (Path(record['root'])/'runtime.json').exists()
        with pytest.raises(httpx.TransportError):
            httpx.get(record['url']+'/health',trust_env=False,timeout=1)
    finally:
        if process.poll() is None: process.terminate();process.wait(timeout=5)
        if record and (Path(record['root'])/'runtime.json').exists():
            # Only the test's isolated simulated instance, never a user executor.
            token=(Path(record['root'])/'model.token').read_text().strip()
            health=httpx.get(record['url']+'/health',trust_env=False).json()
            httpx.post(record['url']+'/v1/shutdown',headers={'Authorization':'Bearer '+token},
                       json={'expected_instance_id':health['instance_id']},trust_env=False)


def test_quit_exits_even_if_pipe_writer_keeps_stdin_open(tmp_path):
    p=subprocess.Popen([sys.executable,'-m','piperx_middleware.standalone_cli','--root',str(tmp_path),
                        '--mode','simulation','--simulation-backend','sim'],
                       stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    try:
        p.stdin.write('/quit\n');p.stdin.flush()
        assert p.wait(timeout=10)==0,p.stderr.read()
        assert not list(tmp_path.rglob('runtime.json'))
    finally:
        p.stdin.close()
        if p.poll() is None:p.terminate();p.wait(timeout=5)


@pytest.mark.asyncio
async def test_session_heartbeat_survives_slow_initial_mcp_handshake(tmp_path, monkeypatch):
    from piperx_middleware.console_bridge import MCPBridge
    original = MCPBridge.open

    async def delayed_open(bridge):
        # A slow network handshake must not starve the already-acquired 5s lease.
        await asyncio.sleep(5.2)
        await original(bridge)

    monkeypatch.setattr(MCPBridge, 'open', delayed_open)
    m = ManagedConsole(tmp_path, 'simulation', 'local', 'sim')
    c = None
    try:
        b = await m.open()
        c = InteractiveConsoleController(b, m.connection.profile_root, emit=lambda _: None,
                                        operator_call=m.operator_call, managed=m)
        m.controller = c
        original_session = m.session_id
        await c.start()
        assert m.session_id == original_session and not c._session_lost
        await c.handle_line('/connect')
        assert not c._session_lost
        assert (await b.rest('GET', '/v1/state'))['ready']
        assert (await m.operator_call('POST', '/operator/session/heartbeat',
                                     {'session_id': m.session_id})).get('error') is None
    finally:
        await m.shutdown()
        if c:
            await c.close()
        await m.close()


@pytest.mark.asyncio
async def test_lost_http_action_response_queries_original_request_without_replay(tmp_path):
    """A real proxy drops the reply AFTER the executor accepted a simulation action."""
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    import threading

    m = ManagedConsole(tmp_path, 'simulation', 'local', 'sim')
    c = None
    proxy = None
    thread = None
    accepted = []
    try:
        b = await m.open()
        c = InteractiveConsoleController(b, m.connection.profile_root, emit=lambda _: None,
                                        operator_call=m.operator_call, managed=m)
        m.controller = c
        await c.start()
        await c.handle_line('/connect')
        headers = {'Authorization': 'Bearer ' + m.connection.model_token_file.read_text().strip(),
                   'X-Piper-Control-Session': m.session_id}

        class DropReply(BaseHTTPRequestHandler):
            def do_POST(self):
                payload = self.rfile.read(int(self.headers['Content-Length']))
                reply = httpx.post(m.connection.url + self.path, content=payload,
                                   headers=headers | {'Content-Type': 'application/json'},
                                   trust_env=False, timeout=5)
                accepted.append(reply.json())
                self.close_connection = True  # No status line/body reaches the client.

            def log_message(self, *args):
                pass

        proxy = ThreadingHTTPServer(('127.0.0.1', 0), DropReply)
        thread = threading.Thread(target=proxy.serve_forever, daemon=True)
        thread.start()
        async with httpx.AsyncClient(trust_env=False, timeout=5) as client:
            with pytest.raises(httpx.TransportError):
                await client.post(f'http://127.0.0.1:{proxy.server_port}/v1/move', json={
                    'request_id': 'lost-reply-001',
                    'command': {'kind': 'joint', 'joints_deg': [1, 0, 0, 0, 0, 0], 'speed_percent': 5}})
            assert len(accepted) == 1 and accepted[0]['status'] in ('accepted', 'running')
            for _ in range(100):
                job = checked(await b.rest('GET', '/v1/requests/lost-reply-001'))
                if job['status'] not in ('accepted', 'running'):
                    break
                await asyncio.sleep(.02)
            assert job['status'] == 'succeeded'
            assert job['job_id'] == accepted[0]['job_id']
            # Only reads during recovery, one persisted action with that request ID.
            again = checked(await b.rest('GET', '/v1/requests/lost-reply-001'))
            assert again['job_id'] == job['job_id'] and len(accepted) == 1
    finally:
        if proxy:
            await asyncio.to_thread(proxy.shutdown)
            proxy.server_close()
        if thread:
            thread.join(timeout=2)
        await m.shutdown()
        if c:
            await c.close()
        await m.close()
