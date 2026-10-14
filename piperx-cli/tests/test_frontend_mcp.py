import asyncio
from datetime import timedelta
import json
import os
from pathlib import Path
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import pytest


@pytest.mark.asyncio
async def test_actual_frontend_mcp_robot_and_demo_pixels_same_session(tmp_path):
    from PIL import Image
    demo=tmp_path/'demo';demo.mkdir()
    image=demo/'frame.png';Image.new('RGB',(32,32),'red').save(image)
    manifest=demo/'manifest.json'
    manifest.write_text(json.dumps({'kind':'piperlab.video.candidates','candidates':[
        {'frame_id':'f0','timestamp_s':0.,'image_path':str(image)}]}))
    profile=tmp_path/'profiles/simulation/local';profile.mkdir(parents=True)
    (profile/'demo-reference.json').write_text(json.dumps({'path':str(manifest)}))
    params=StdioServerParameters(command=sys.executable,args=['-m','piperx_middleware.standalone_cli',
        '--root',str(tmp_path),'--mode','simulation','--simulation-backend','sim','--agent-mcp'])
    async with stdio_client(params) as (read,write):
        async with ClientSession(read,write,read_timeout_seconds=timedelta(seconds=30)) as session:
            initialized=await session.initialize()
            names={t.name for t in (await session.list_tools()).tools}
            assert 'demo_reference' in names and 'robot_move_joints' in names
            assert 'operator_approve' not in names
            assert {'task_plan','scene_query'} <= names
            task=await session.call_tool('task_plan',{'operation':'create','goal':'Measure a synthetic surface'})
            assert not task.isError and task.structuredContent['task_id']
            from test_observation_workflow import evidence
            observation=evidence(profile)
            measured=await session.call_tool('scene_query',{'operation':'surface_point',
                'observation_path':observation,'pixel':[20,20]})
            assert not measured.isError and measured.structuredContent['point_m']==[0.,0.,1.]
            guide=await session.call_tool('task_plan',{'operation':'guide'})
            assert not guide.isError and 'observe-and-plan' in guide.structuredContent['content']
            ref=await session.call_tool('demo_reference',{})
            assert not ref.isError and any(c.type=='image' for c in ref.content)
            assert ref.structuredContent['scope']=='historical_demonstration_only'
            assert not (await session.call_tool('robot_connect',{})).isError
            move=await session.call_tool('robot_move_joints',{'joints_deg':[1.,0.,0.,0.,0.,0.],'request_id':'frontend-mcp-test1'})
            assert not move.isError and move.structuredContent['status']=='succeeded'
            missing=await session.call_tool('robot_move_joints',{'joints_deg':[2.,0.,0.,0.,0.,0.]})
            assert missing.isError
    for _ in range(30):
        if not list(tmp_path.rglob('runtime.json')):break
        await asyncio.sleep(.1)
    assert not list(tmp_path.rglob('runtime.json'))
