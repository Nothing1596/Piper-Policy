"""Regressions from the paired complex task; fixtures do not claim physics."""
import copy
import json
from types import SimpleNamespace

import pytest
import run_longvil_online as online
from test_longvil_online import Session,fixture,plan_update
from test_longvil_softplan_recovery import decision,driver,progress_of


class ExplorationSession(Session):
    async def call_tool(self,name,args):
        if name=='robot_status':
            return SimpleNamespace(isError=False,content=[],structuredContent={'backend':'mujoco',
                'robot':{'gripper_width_m':.07,'gripper_contact':{'supported':False}}})
        return await super().call_tool(name,args)


def batch(*rows):
    result=copy.deepcopy(rows[0])
    result['choices'][0]['message']['tool_calls']=[c for row in rows for c in row['choices'][0]['message']['tool_calls']]
    return result


def end():
    return decision('finish_task',{'outcome':'aborted','summary':'fixture complete','uncertainties':[]},'end')


def setup(monkeypatch,sess,responses,payloads):
    monkeypatch.setattr(online,'paid',driver(responses,payloads))
    async def state(*a):return {'sim_time_s':sess.time,'objects':{},'bilateral_contact_objects':[]}
    monkeypatch.setattr(online,'remote',state)


@pytest.mark.asyncio
@pytest.mark.parametrize('bad', ['{"operation":"update","changes":{}', 'null', '[]', '"update"'])
async def test_bad_plan_candidate_is_tool_feedback_not_round_abort(tmp_path,monkeypatch,bad):
    args,e=fixture(tmp_path);args.exploration=True;args.max_rounds=3
    sess=ExplorationSession();payloads=[];events=[]
    broken=decision('task_plan',{},'bad-plan')
    broken['choices'][0]['message']['tool_calls'][0]['function']['arguments']=bad
    reads=[decision('scene_query',{'operation':'surface_point','observation_path':'live-0001','pixel':[i,20]},f'q{i}') for i in range(4)]
    responses=[batch(broken,*reads),decision('task_plan',plan_update(1),'corrected'),end()]
    setup(monkeypatch,sess,responses,payloads);report={}
    await online.online_loop(sess,args,e,events.append,tmp_path,report,None)
    failures=[r for r in events if r['kind']=='tool_result' and r['name']=='task_plan' and 'error' in r['result']]
    assert failures[0]['result']['error']['executed'] is False
    assert report['api_responses']==3 and report['completion_tokens']==300
    assert report['tool_calls']==7 and report['plan_writes']==1 and report['interface_errors']==1
    assert not sess.moves and report['status']=='aborted'
    assert len([c for c in sess.calls if c[0]=='task_plan' and c[1]['operation']=='update'])==1


@pytest.mark.asyncio
async def test_adapter_measurements_have_separate_counts_and_frame_scoped_plane_cache(tmp_path,monkeypatch):
    class GeometrySession(Session):
        async def call_tool(self,name,args):
            if name!='scene_query':return await super().call_tool(name,args)
            self.calls.append((name,copy.deepcopy(args)))
            value=({'status':'ok','plane_id':'p1','normal':[0,0,1],'point_on_plane_m':[0,0,0]}
                   if args['operation']=='fit_plane' else {'status':'unavailable','reason':'occluded'})
            # The frontend sees repeated internal plane fits, not model choices.
            value['harness_review']={'state':'needs_review','issues':[{'code':'repeated_query'}]}
            return SimpleNamespace(isError=False,structuredContent=value,content=[])
    args,e=fixture(tmp_path);args.video_plan_workflow=True;args.max_rounds=5
    sess=GeometrySession();payloads=[];events=[]
    def grasp(i,view='live-0001'):
        return decision('grasp_target',{'observation_id':view,'object_roi':[10+i,20,40+i,60],
                                       'table_roi':[100,100,120,120]},f'g{i}')
    responses=[grasp(0),grasp(1),grasp(2),decision('camera_observe',{},'cam'),grasp(3,'live-0002')]
    setup(monkeypatch,sess,responses,payloads);report={}
    await online.online_loop(sess,args,e,events.append,tmp_path,report,None)
    assert len([a for n,a in sess.calls if n=='scene_query' and a['operation']=='fit_plane'])==2
    assert report['internal_plane_cache_hits']==2 and report['tool_calls']==5
    assert report['native_calls_by_origin']['adapter_internal']['scene_query']==6
    assert report['native_tool_calls']['scene_query']==6
    assert not progress_of(payloads[-1])['harness_review']['issues']
    assert all('harness_review' not in r['result'] for r in events if r['kind']=='tool_result')
    assert any(r['kind']=='native_harness_review' and r['origin']=='adapter_internal' for r in events)


@pytest.mark.asyncio
async def test_action_version_returned_stale_write_rejected_then_next_decision_writes(tmp_path,monkeypatch):
    class VersionSession(ExplorationSession):
        async def call_tool(self,name,args):
            result=await super().call_tool(name,args)
            if name=='move_to':self.plan['version']+=1  # Runtime invalidates dependent geometry.
            return result
    args,e=fixture(tmp_path);args.exploration=True;args.max_rounds=3
    sess=VersionSession();payloads=[];events=[]
    move=decision('move_to',{'request_id':'move-once','observation_id':'live-0001','xyz_m':[.3,0,.2]},'move')
    responses=[batch(move,decision('task_plan',plan_update(1),'stale')),
               decision('task_plan',plan_update(2),'fresh'),end()]
    setup(monkeypatch,sess,responses,payloads);report={}
    await online.online_loop(sess,args,e,events.append,tmp_path,report,None)
    writes=[r['result'] for r in events if r['kind']=='tool_result' and r['name']=='task_plan']
    assert writes[0]['error']['executed'] is False and writes[0]['current_plan']['version']==2
    assert writes[1]['version']==3 and report['plan_writes']==1 and len(sess.moves)==1
    result=next(r['result'] for r in events if r['kind']=='tool_result' and r['name']=='move_to')
    assert result['plan_writeback']['version']==2
    assert progress_of(payloads[1])['current_plan']['version']==2
    assert report['native_calls_by_origin']['model_tool']['move_to']==1
    assert [a['expected_version'] for n,a in sess.calls if n=='task_plan' and a['operation']=='update']==[2]


@pytest.mark.asyncio
async def test_post_action_plan_store_failure_does_not_erase_known_result(tmp_path,monkeypatch):
    class BrokenSyncSession(Session):
        async def call_tool(self,name,args):
            if name=='task_plan' and args['operation']=='get' and self.moves:
                raise RuntimeError('store transport disconnected')
            return await super().call_tool(name,args)
    args,e=fixture(tmp_path);args.max_rounds=2;sess=BrokenSyncSession();payloads=[];events=[]
    responses=[decision('move_to',{'request_id':'known-once','observation_id':'live-0001','xyz_m':[.3,0,.2]},'known')]
    setup(monkeypatch,sess,responses,payloads);report={}
    with pytest.raises(RuntimeError,match='store transport disconnected'):
        await online.online_loop(sess,args,e,events.append,tmp_path,report,None)
    result=next(r['result'] for r in events if r['kind']=='tool_result' and r['name']=='move_to')
    assert result['status']=='succeeded' and result['plan_writeback_error']=='store transport disconnected'
    assert 'plan_writeback' not in result and not report.get('unknown_outcome',False)
    assert len(sess.moves)==1 and len(payloads)==1


@pytest.mark.asyncio
async def test_control_disables_only_monitor_feedback_not_video_plan_or_tools(tmp_path,monkeypatch):
    args,e=fixture(tmp_path);args.video_plan_workflow=True;args.behavior_monitor=False;args.max_rounds=4
    sess=Session();payloads=[];events=[]
    query={'operation':'surface_point','observation_path':'live-0001','pixel':[10,20]}
    responses=[decision('scene_query',query,f'q{i}') for i in range(3)]+[end()]
    setup(monkeypatch,sess,responses,payloads);report={}
    await online.online_loop(sess,args,e,events.append,tmp_path,report,None)
    assert report['behavior_monitor'] is None and report['api_responses']==4 and report['tool_calls']==4
    assert all('harness_review' not in progress_of(p) and 'video_plan_review' in progress_of(p) for p in payloads)
    assert report['plan_enabled'] and report['skill_enabled']
    assert report['native_tool_calls']['scene_query']==1 and report['read_cache_hits']==2
