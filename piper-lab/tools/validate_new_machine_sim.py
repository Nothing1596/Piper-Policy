"""Run documented fresh-package simulation/MCP startup without a model or motion."""
import asyncio
import base64
import io
import json
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
from datetime import timedelta
from PIL import Image
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

root=Path(sys.argv[1]).resolve()
assert Path(sys.prefix).resolve()==root/'.venv'
port=8819
with socket.socket() as probe:
    probe.bind(('127.0.0.1',port))

async def exercise(work):
    params=StdioServerParameters(command=sys.executable,args=[str(root/'piper.py'),'mcp',
        '--workspace',str(work),'--robot-root',str(work/'sim'),'--robot-url',f'http://127.0.0.1:{port}'])
    async with stdio_client(params) as (reader,writer):
        async with ClientSession(reader,writer,read_timeout_seconds=timedelta(seconds=40)) as session:
            await session.initialize()
            names=[t.name for t in (await session.list_tools()).tools]
            assert 'simulation_observe' in names and 'robot_status' in names
            status=await session.call_tool('robot_status',{})
            assert not status.isError
            data=json.loads(next(c.text for c in status.content if c.type=='text'))
            assert data['backend']=='mujoco'
            connected=await session.call_tool('robot_connect',{})
            assert not connected.isError
            frame=await session.call_tool('simulation_observe',{'output':'observations/smoke'})
            assert not frame.isError
            pixels=base64.b64decode(next(c.data for c in frame.content if c.type=='image'))
            image=Image.open(io.BytesIO(pixels))
            assert image.size==(640,480)
            assert (work/'observations/smoke/depth.npy').is_file()
            assert not (await session.call_tool('robot_disconnect',{})).isError
            return {'fresh_root':True,'documented_start_connect_flow':True,
                    'mcp_simulation_observe':True,'rgb_size':list(image.size),'depth_saved':True,
                    'motion_sent':False,'model_called':False,'hardware_connected':False}

with tempfile.TemporaryDirectory(prefix='piper-new-machine-') as folder:
    work=Path(folder)
    with (work/'server.log').open('w',encoding='utf-8') as log:
        proc=subprocess.Popen([sys.executable,str(root/'piper.py'),'sim','start','--root',str(work/'sim'),
            '--port',str(port),'--seed','200'],cwd=root,stdout=log,stderr=log,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform=='win32' else 0)
        try:
            deadline=time.monotonic()+30
            while True:
                if proc.poll() is not None:raise RuntimeError('Simulator exited during startup')
                try:
                    with socket.create_connection(('127.0.0.1',port),timeout=.3):break
                except OSError:
                    if time.monotonic()>deadline:raise TimeoutError('Simulator did not listen')
                    time.sleep(.2)
            result=asyncio.run(exercise(work))
        finally:
            stopped=subprocess.run([sys.executable,str(root/'piper.py'),'device','--root',str(work/'sim'),
                '--url',f'http://127.0.0.1:{port}','shutdown'],capture_output=True,text=True,timeout=20)
            proc.wait(timeout=20)
            assert stopped.returncode==0, stopped.stderr
receipt=root/'PACKAGE-VALIDATION.json'
record=json.loads(receipt.read_text())
record['fresh_simulation_mcp']=result
receipt.write_text(json.dumps(record,indent=2),encoding='utf-8')
print(json.dumps(result))
