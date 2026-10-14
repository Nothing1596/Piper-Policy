import asyncio
import json
from pathlib import Path
import sys
import pytest
from PIL import Image
from piperlab.harness import create_server


def test_tools_and_workspace_boundaries(tmp_path):
    image=tmp_path/'frame.png';Image.new('RGB',(8,8)).save(image)
    async def run():
        server=create_server(tmp_path)
        names={t.name for t in await server.list_tools()}
        assert {'video_compile','video_frame','video_candidates','video_inspect'} <= names
        assert not any(n.startswith('robot_') for n in names)
        result=await server.call_tool('video_frame',{'image':'frame.png'})
        assert any(getattr(c,'type',None)=='image' for c in (result[0] if isinstance(result,tuple) else result))
        with pytest.raises(Exception,match='workspace'):
            await server.call_tool('video_frame',{'image':'../outside.png'})
        with pytest.raises(Exception,match='model-config'):
            await server.call_tool('video_compile',{'video':'x','task':'x','output':'out'})
    asyncio.run(run())


def test_combined_robot_tools_keep_existing_client(tmp_path):
    class Client:
        def call(self,*args,**kwargs):raise AssertionError('startup must not contact hardware')
    async def run():
        names={t.name for t in await create_server(tmp_path,robot_client=Client()).list_tools()}
        assert 'robot_status' in names and 'video_candidates' in names and 'simulation_observe' in names
    asyncio.run(run())


def test_simulation_observe_rejects_real_backend(tmp_path):
    class Client:
        def call(self,method,path,body=None):
            assert path=='/v1/capabilities'
            return {'backend':'piper'}
    async def run():
        with pytest.raises(Exception,match='MuJoCo'):
            await create_server(tmp_path,robot_client=Client()).call_tool('simulation_observe',{'output':'frame'})
        assert not (tmp_path/'frame').exists()
    asyncio.run(run())


def test_simulation_observe_returns_pixels_and_no_ground_truth(tmp_path):
    import io,base64,numpy as np
    rgb=io.BytesIO();Image.new('RGB',(8,8)).save(rgb,format='JPEG')
    depth=io.BytesIO();np.save(depth,np.ones((8,8),dtype=np.float32),allow_pickle=False)
    class Client:
        def call(self,method,path,body=None):
            if path=='/v1/capabilities':return {'backend':'mujoco'}
            assert path=='/v1/simulation/observation'
            return {'rgb_jpeg_b64':base64.b64encode(rgb.getvalue()).decode(),
                    'depth_npy_b64':base64.b64encode(depth.getvalue()).decode(),
                    'camera_info':{},'source_stamp_s':1,'epoch':0}
    async def run():
        server=create_server(tmp_path,robot_client=Client())
        result=await server.call_tool('simulation_observe',{'output':'frame'})
        content=result[0] if isinstance(result,tuple) else result
        assert any(c.type=='image' for c in content)
        meta=json.loads((tmp_path/'frame/observation.json').read_text())
        assert 'rgb_jpeg_b64' not in meta and 'objects' not in meta
        assert np.load(tmp_path/'frame/depth.npy',allow_pickle=False).shape==(8,8)
        with pytest.raises(Exception,match='already exists'):
            await server.call_tool('simulation_observe',{'output':'frame'})
    asyncio.run(run())


def test_real_stdio_protocol(tmp_path):
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    Image.new('RGB',(8,8)).save(tmp_path/'image.png')
    from test_video_v1 import _write_synthetic_video
    _write_synthetic_video(tmp_path/'video.mp4',frames=12)
    async def run():
        params=StdioServerParameters(command=sys.executable,args=['-m','piperlab.harness','--workspace',str(tmp_path)])
        async with stdio_client(params) as (reader,writer):
            async with ClientSession(reader,writer) as session:
                await session.initialize()
                tools=await session.list_tools()
                assert 'video_frame' in {t.name for t in tools.tools}
                result=await session.call_tool('video_frame',{'image':'image.png'})
                assert not result.isError and result.content[0].type=='image'
                error=await session.call_tool('video_frame',{'image':'../outside.png'})
                assert error.isError
                result=await session.call_tool('video_candidates',{'video':'video.mp4','output':'candidates'})
                assert not result.isError
                result=await session.call_tool('video_candidate_page',{'manifest':'candidates/manifest.json'})
                assert not result.isError and 'frame_id' in result.content[0].text
                result=await session.call_tool('video_candidates',{'video':'video.mp4','output':'candidates'})
                assert result.isError  # existing evidence cannot be overwritten
    asyncio.run(run())
