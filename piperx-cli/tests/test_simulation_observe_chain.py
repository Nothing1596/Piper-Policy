"""Cold installed-style MCP: real rendered evidence, plan writeback and motion."""
import asyncio
from datetime import timedelta
import json
import sys
from pathlib import Path
import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


@pytest.mark.asyncio
async def test_mujoco_model_observe_plan_execute_reobserve(tmp_path):
    pytest.importorskip('mujoco')
    pytest.importorskip('PIL')
    params=StdioServerParameters(command=sys.executable,args=['-m','piperx_middleware.standalone_cli',
        '--root',str(tmp_path),'--mode','simulation','--simulation-backend','mujoco','--agent-mcp'])
    async with stdio_client(params) as (read,write):
        async with ClientSession(read,write,read_timeout_seconds=timedelta(seconds=45)) as s:
            await s.initialize()
            names={t.name for t in (await s.list_tools()).tools}
            assert 'camera_observe' in names and 'camera_video' not in names
            async def call(name,arguments):
                r=await s.call_tool(name,arguments)
                assert not r.isError,r.structuredContent
                return r
            await call('robot_connect',{})
            first=await call('camera_observe',{'max_frames':1})
            assert any(c.type=='image' for c in first.content)
            frame=first.structuredContent['frames'][0]
            assert first.structuredContent['simulated'] is True
            q=frame['robot_alignment']['state']['robot']['q_deg']
            plan=(await call('task_plan',{'operation':'create','goal':'Measure the observed simulated surface'})).structuredContent
            profile=tmp_path/'profiles/simulation/local'
            legacy=await asyncio.create_subprocess_exec(sys.executable,'-m','piperx_middleware.standalone_cli',
                'observe','--root',str(profile),'--workspace',str(profile),'--output','observations/legacy',
                stdout=asyncio.subprocess.PIPE,stderr=asyncio.subprocess.PIPE)
            stdout,stderr=await asyncio.wait_for(legacy.communicate(),30)
            assert legacy.returncode==0,stderr.decode(errors='replace')
            saved=json.loads(stdout)
            compatible=(await call('scene_query',{'operation':'surface_point',
                'observation_path':saved['observation_path'],'pixel':[320,240]})).structuredContent
            assert compatible['frame']=='base'
            await call('task_plan',{'operation':'update','task_id':plan['task_id'],'expected_version':plan['version'],
                'changes':{'questions':[{'id':'surface','question':'Where is this visible surface?',
                    'affects_action':True,'source':'geometry','query':{'operation':'surface_point','pixel':[320,240]}}]}})
            measured=await call('camera_observe',{'question_id':'surface','max_frames':1,'roi':[300,220,340,260]})
            result=measured.structuredContent
            assert result['geometry']['frame']=='base'
            assert result['geometry']['source']['simulated'] is True
            assert len([c for c in measured.content if c.type=='image'])==2
            assert (await call('task_plan',{'operation':'get'})).structuredContent['questions'][0]['state']=='answered'
            goal=list(q);goal[0]+=2
            moved=(await call('robot_move_joints',{'joints_deg':goal,'request_id':'observe-chain-move-001','speed_percent':5})).structuredContent
            assert moved['status']=='succeeded'
            assert (await call('task_plan',{'operation':'get'})).structuredContent['questions'][0]['state']=='stale'
            after=(await call('camera_observe',{'question_id':'surface','max_frames':1})).structuredContent
            assert after['frames'][0]['source_stamp_s'] > frame['source_stamp_s']
            assert after['frames'][0]['observation_path'] != frame['observation_path']
            assert after['geometry']['status']=='ok'
            assert (await call('task_plan',{'operation':'get'})).structuredContent['questions'][0]['state']=='answered'
    for _ in range(30):
        if not list(tmp_path.rglob('runtime.json')):break
        await asyncio.sleep(.1)
    assert not list(tmp_path.rglob('runtime.json'))


@pytest.mark.asyncio
async def test_jsonl_transports_simulated_pixels_without_remote_file_access(tmp_path):
    pytest.importorskip('mujoco')
    pytest.importorskip('PIL')
    with (tmp_path/'stderr.log').open('w') as err:
        proc=await asyncio.create_subprocess_exec(sys.executable,'-m','piperx_middleware.standalone_cli',
            '--root',str(tmp_path),'--mode','simulation','--simulation-backend','mujoco','--agent',
            stdin=asyncio.subprocess.PIPE,stdout=asyncio.subprocess.PIPE,stderr=err,limit=4*1024*1024)
        async def read():return json.loads(await asyncio.wait_for(proc.stdout.readline(),30))
        async def send(value):
            proc.stdin.write((json.dumps(value)+'\n').encode());await proc.stdin.drain()
            return await read()
        try:
            assert (await read())['result']['status']=='ready'
            await send({'id':'connect','command':'call','tool':'robot_connect','arguments':{}})
            observed=await send({'id':'observe','command':'observe','arguments':{'max_frames':1}})
            assert observed['result']['simulated'] is True
            assert any(part.get('image_url',{}).get('url','').startswith('data:image/jpeg;base64,') for part in observed['content'])
            await send({'id':'quit','command':'quit'})
            assert await asyncio.wait_for(proc.wait(),20)==0
        finally:
            if proc.returncode is None:proc.kill();await proc.wait()
    assert not list(tmp_path.rglob('runtime.json'))
