import asyncio
import json
from pathlib import Path

import pytest

from piperx_middleware.demo_bridge import DemoBridge
from piperx_middleware.console_interaction import InteractiveConsoleController


def artifact(tmp_path, compiled=False):
    folder=tmp_path/'demo';folder.mkdir()
    frames=[]
    for i in range(4):
        image=folder/f'{i}.jpg';image.write_bytes(b'image-fixture')
        frames.append({'frame_id':f'f{i}','timestamp_s':float(i),'image_path':str(image)})
    value={'kind':'piperlab.video.candidates','candidates':frames}
    if compiled:
        value={'created_by':'piperlab.demonstration.compiler.compile_demo','frames':frames,
            'stages':[{'id':'grasp','operation':'grasp_handle','evidence_refs':['f1','f2']}],
            'goal_constraints':['drawer open'],'unknowns':['pull distance'],'summary':'grasp, pull, release'}
    path=folder/'manifest.json';path.write_text(json.dumps(value));return path


def test_pipeline_pages_semantics_and_images(tmp_path):
    bridge=DemoBridge(tmp_path/'profile')
    result=bridge.use(artifact(tmp_path,True))
    assert result['kind']=='compiled_demonstration' and result['stages'][0]['operation']=='grasp_handle'
    assert result['unknowns']==['pull distance'] and result['scope']=='historical_demonstration_only'
    assert result['next_offset']==3 and len(bridge.image_inputs(result))==6
    assert bridge.reference(3)['frames'][0]['frame_id']=='f3'
    assert bridge.clear()['status']=='cleared' and bridge.reference()['status']=='not_selected'


def test_reject_bad_reference_keeps_previous_and_no_external_image(tmp_path):
    bridge=DemoBridge(tmp_path/'profile');path=artifact(tmp_path)
    assert bridge.use(path)['kind']=='pipeline_candidates'
    original=(bridge.root/'demo-reference.json').read_bytes()
    bad=tmp_path/'bad';bad.mkdir();external=tmp_path/'secret.jpg';external.write_bytes(b'secret')
    value={'kind':'piperlab.video.candidates','candidates':[{'frame_id':'f0','timestamp_s':0.,'image_path':str(external)}]}
    broken=bad/'manifest.json';broken.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='archived'):bridge.use(broken)
    assert (bridge.root/'demo-reference.json').read_bytes()==original


@pytest.mark.asyncio
async def test_historical_frames_reach_model_context(tmp_path):
    controller=object.__new__(InteractiveConsoleController)
    controller.root=tmp_path/'profile';controller._demo=None
    controller.demo().use(artifact(tmp_path,True))
    messages=[{'role':'user','content':'Open the drawer'}]
    await controller.prepare_model_context(messages)
    content=messages[-1]['content']
    assert 'historical_demonstration_only' in content[0]['text']
    assert any(part['type']=='image_url' for part in content)
    await controller.prepare_model_context(messages)
    assert len(messages)==3


@pytest.mark.asyncio
async def test_prepare_delegates_to_video_pipeline_without_implicit_model(tmp_path,monkeypatch):
    bridge=DemoBridge(tmp_path/'profile');bridge.root.mkdir()
    source=tmp_path/'source.mp4';source.write_bytes(b'fixture')
    called=[]
    async def create(*args,**kwargs):
        assert kwargs['stdin']==asyncio.subprocess.DEVNULL
        called.append(args)
        out=Path(args[args.index('--output')+1]);out.mkdir(parents=True)
        img=out/'frame.jpg';img.write_bytes(b'fixture')
        (out/'manifest.json').write_text(json.dumps({'kind':'piperlab.video.candidates',
            'candidates':[{'frame_id':'f0','timestamp_s':0.,'image_path':str(img)}]}))
        class Proc:
            returncode=0
            async def communicate(self):return b'',b''
        return Proc()
    monkeypatch.setattr(asyncio,'create_subprocess_exec',create)
    result=await bridge.prepare(source)
    assert called[0][1:4]==('-m','piperlab.video_entry','candidates')
    assert result['kind']=='pipeline_candidates' and result['stages']==[]
    with pytest.raises(ValueError,match='Configure /model'):await bridge.prepare(source,compile=True,task='drawer')


def test_overview_stage_pages_and_original_frames_not_hidden(tmp_path):
    bridge=DemoBridge(tmp_path/'profile');bridge.use(artifact(tmp_path,True))
    result=bridge.reference(overview=True)
    assert result['frames'][0]['frame_id']=='f0' and result['frames'][-1]['frame_id']=='f3'
    result['evaluation']={'vision':{'frames_supplied':['f0']}}
    assert len([i for i in bridge.image_inputs(result) if i['type']=='image_url'])==4
    stage=bridge.reference(stage_id='grasp',limit=1)
    assert stage['frames'][0]['frame_id']=='f1' and stage['next_offset']==1
    assert bridge.reference(stage_id='grasp',offset=1)['frames'][0]['frame_id']=='f2'
    with pytest.raises(ValueError,match='Unknown'):bridge.reference(stage_id='missing')


def test_twelve_frame_watch_delivers_last_frame_pixels(tmp_path):
    folder=tmp_path/'watch';folder.mkdir()
    frames=[]
    for i in range(12):
        path=folder/f'{i}.jpg';path.write_bytes(f'frame-{i}'.encode())
        frames.append({'frame_id':f'f{i}','timestamp_s':float(i),'image_path':str(path)})
    manifest=folder/'watch.json'
    manifest.write_text(json.dumps({'kind':'piperlab.video.watch','frames':frames,'mode':'pipeline','evaluation':{}}))
    bridge=DemoBridge(tmp_path/'profile')
    result=bridge._describe(manifest,0,12)
    content=bridge.image_inputs(result)
    pixels=[p for p in content if p['type']=='image_url']
    assert len(result['frames'])==len(pixels)==12
    import base64
    assert base64.b64decode(pixels[-1]['image_url']['url'].split(',')[1])==b'frame-11'
    assert '/ f11 at 11.0s' in content[-2]['text']
