"""Region selection must reach the existing worker without semantic invention."""
import copy
import json
import sys
from pathlib import Path

import pytest
from jsonschema import Draft7Validator

sys.path.insert(0, str(Path(__file__).parents[1]))
from run_longvil_online import GRASP_TARGET, PLACEMENT_TARGET, selected_region


@pytest.mark.parametrize('polygon', [[[1,1],[1,1],[1,1]], [[0,0],[10,10],[20,20]]])
def test_zero_area_polygon_rejected_before_worker(polygon):
    with pytest.raises(ValueError,match='nonzero pixel area'):
        selected_region({'object_polygon':polygon},'object')


@pytest.mark.parametrize('tool,prefix', [(GRASP_TARGET,'object'),(PLACEMENT_TARGET,'reference')])
@pytest.mark.parametrize('kind', ['roi','polygon'])
def test_selection_schema_compatibility_and_exclusivity(tool, prefix, kind):
    validator = Draft7Validator(tool['inputSchema'])
    roi = [0,0,40,40]
    polygon = [[0,0],[40,0],[20,40]]
    shape = roi if kind=='roi' else polygon
    body = {'observation_id':'o',prefix+'_'+kind:shape,'table_'+kind:shape}
    if prefix=='reference':body.update(source_measurement_id='m',relation='top')
    assert validator.is_valid(body)
    assert selected_region(body,prefix)=={kind:shape}
    for key in (prefix,'table'):
        both = dict(body, **{key+('_polygon' if kind=='roi' else '_roi'):polygon if kind=='roi' else roi})
        assert not validator.is_valid(both)
        with pytest.raises(ValueError):selected_region(both,key)
        missing = dict(body);missing.pop(key+'_'+kind)
        assert not validator.is_valid(missing)
        with pytest.raises(ValueError):selected_region(missing,key)
        if kind=='polygon':
            for invalid in (polygon[:2],polygon*22,[[0,0],[1.5,2],[3,4]],[[0,0],[-1,2],[3,4]]):
                bad=copy.deepcopy(body);bad[key+'_polygon']=invalid
                assert not validator.is_valid(bad)


@pytest.mark.asyncio
@pytest.mark.parametrize('kind',['roi','polygon'])
@pytest.mark.parametrize('echo_mismatch',[False,True])
async def test_grasp_then_placement_pass_same_selection_to_worker(tmp_path, monkeypatch, kind, echo_mismatch):
    from types import SimpleNamespace
    import numpy as np
    import run_longvil_online as online
    from test_longvil_online import Session, fixture
    from piperx_middleware.scene_geometry import _surface_summary
    from piperx_middleware.kinematics import PiperKinematics
    args,evidence=fixture(tmp_path);args.max_rounds=1 if echo_mismatch else 2
    points=np.array([[x,y,-.15] for x in np.linspace(.28,.32,12) for y in np.linspace(-.02,.02,12)])
    summary=_surface_summary(points)
    class GeometrySession(Session):
        async def call_tool(self,name,body):
            if name!='scene_query':return await super().call_tool(name,body)
            self.calls.append((name,copy.deepcopy(body)))
            if body['operation']=='fit_plane':
                value={'status':'ok','plane_id':'p','normal':[0,0,1],'point_on_plane_m':[0,0,-.2]}
            else:
                entry=body['observations'][0]
                value={'status':'ok','surface_summary':summary,'quality':{'voxels':144},
                       'observations':[{'region':{k:v for k,v in entry.items() if k in ('roi','polygon')}}]}
                if echo_mismatch:value['observations'][0]['region']={'roi':[0,0,5,5]}
            return SimpleNamespace(isError=False,structuredContent=value,content=[])
    monkeypatch.setattr(PiperKinematics,'pose',lambda *a:{'xyz_m':[.3,0,0],'rpy_deg':[180,0,0]})
    monkeypatch.setattr(PiperKinematics,'solve',lambda *a:{'q_deg':[0]*6})
    region=[10,10,30,30] if kind=='roi' else [[10,10],[30,10],[20,30]]
    events=[];session=GeometrySession();turn=0
    async def decide(*unused):
        nonlocal turn
        if turn==0:
            name='grasp_target';body={'observation_id':'live-0001','object_'+kind:region,'table_'+kind:region}
        else:
            source=next(e['result']['source_measurement_id'] for e in events if e['kind']=='tool_result' and e['name']=='grasp_target')
            name='placement_target';body={'observation_id':'live-0001','reference_'+kind:region,'table_'+kind:region,
                                          'relation':'top','source_measurement_id':source}
        turn+=1
        return {'model':'test','usage':{'completion_tokens':5},'choices':[{'finish_reason':'tool_calls','message':{'role':'assistant','content':'',
                'tool_calls':[{'id':str(turn),'function':{'name':name,'arguments':json.dumps(body)}}]}}]}
    args.model_decider=decide
    await online.online_loop(session,args,evidence,events.append,tmp_path,{},None)
    if echo_mismatch:
        rejected=next(e['result'] for e in events if e['kind']=='tool_result' and e['name']=='grasp_target')
        assert 'selection differs' in rejected['error']['message']
        assert 'poses' not in rejected and not session.moves
        return
    calls=[b for n,b in session.calls if n=='scene_query']
    assert len(calls)==3 and not session.moves  # Same-frame support plane reused.
    assert calls[0][kind]==region
    assert calls[2]['observations'][0][kind]==region
    receipt=next(e['result'] for e in events if e['kind']=='tool_result' and e['name']=='placement_target')
    assert receipt['status']=='ok'
    assert receipt['selection_evidence']['target_region']=={kind:region}
    assert receipt['selection_evidence']['single_object_verified'] is False
