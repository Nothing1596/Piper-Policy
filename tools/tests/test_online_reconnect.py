"""Explicit reconnect refreshes evidence; spontaneous identity changes still fail."""
import copy
from types import SimpleNamespace
import pytest
import run_longvil_online as online
from test_longvil_online import fixture,tool
from test_online_exploration import FeedbackSession
from test_longvil_softplan_recovery import decision,driver

class ReconnectSession(FeedbackSession):
    def __init__(self,bad=None):
        super().__init__();self.epoch=1;self.bad=bad;self.reconnected=False
    async def list_tools(self):
        return await super().list_tools()
    async def call_tool(self,name,args):
        if name=='robot_connect':
            self.calls.append((name,copy.deepcopy(args)));self.epoch=3;self.reconnected=True
            return SimpleNamespace(isError=False,content=[],structuredContent={
                'backend':'mujoco','instance_id':'sim','connection_epoch':3,'active_job_id':None,
                'observation_identity':{'instance_id':'sim','connection_epoch':3}})
        res=await super().call_tool(name,args)
        if name=='camera_observe':
            f=res.structuredContent['frames'][0];f['connection_epoch']=self.epoch
            if self.reconnected:
                if self.bad=='instance':f['instance_id']='other'
                if self.bad=='epoch':f['connection_epoch']=4
                if self.bad=='time':f['sim_time_s']=0
                if self.bad=='stamp':f['source_stamp_s']=0
        return res

@pytest.mark.asyncio
async def test_reconnect_refreshes_before_next_decision_and_rejects_old_view(tmp_path,monkeypatch):
    args,e=fixture(tmp_path);args.exploration=True;args.plan_envelope=False;args.max_rounds=3
    sess=ReconnectSession();payloads=[]
    calls=[decision('robot_connect',{'reconnect':True},'reconnect'),
           decision('move_to',{'request_id':'old-motion','xyz_m':[.3,0,.2],'observation_id':'live-0001'},'old'),
           decision('move_to',{'request_id':'new-motion','xyz_m':[.3,0,.2],'observation_id':'live-0002'},'new')]
    monkeypatch.setattr(online,'paid',driver(calls,payloads))
    async def remote(*a):return {'sim_time_s':sess.time,'objects':{},'bilateral_contact_objects':[]}
    monkeypatch.setattr(online,'remote',remote)
    report={};events=[];await online.online_loop(sess,args,e,events.append,tmp_path,report,None)
    assert report['reconnect_observation_refreshes']==1
    assert [m['request_id'] for m in sess.moves]==['new-motion']
    assert any(x['kind']=='observation' and x['observation_id']=='live-0002' for x in events)
    assert 'old_observations_invalidated' in str(payloads[1])
    assert report['plan_writes']==0

@pytest.mark.asyncio
@pytest.mark.parametrize('bad',['instance','epoch','time','stamp'])
async def test_reconnect_does_not_accept_unverified_continuity(tmp_path,monkeypatch,bad):
    args,e=fixture(tmp_path);args.exploration=True;args.plan_envelope=False
    sess=ReconnectSession(bad)
    monkeypatch.setattr(online,'paid',driver([decision('robot_connect',{'reconnect':True},'r')],[]))
    with pytest.raises(online.ModelCallAnomaly):await online.online_loop(sess,args,e,lambda x:None,tmp_path,{},None)
    assert not sess.moves

@pytest.mark.asyncio
async def test_unrequested_epoch_change_still_stops(tmp_path,monkeypatch):
    args,e=fixture(tmp_path);args.exploration=True;args.plan_envelope=False
    sess=ReconnectSession()
    async def decide(*a):
        sess.epoch=3
        return decision('camera_observe',{},'unexpected-change')
    monkeypatch.setattr(online,'paid',decide)
    with pytest.raises(online.ModelCallAnomaly,match='identity changed'):
        await online.online_loop(sess,args,e,lambda x:None,tmp_path,{},None)
    assert not sess.moves
