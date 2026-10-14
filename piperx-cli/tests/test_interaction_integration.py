"""Offline admission/transport tests. No CAN, camera or physical robot evidence."""
import json
import time
from contextlib import contextmanager

import pytest
from fastapi.testclient import TestClient

from piperx_middleware.backends import SimBackend
from piperx_middleware.http_api import create_app
from piperx_middleware.interaction_types import InteractionPolicy, ExecutionLimits
from piperx_middleware.models import Settings, JointMove
from piperx_middleware.service import RobotService

MODEL = {'Authorization': 'Bearer ' + 'm' * 48}
OPERATOR = {'Authorization': 'Bearer ' + 'o' * 48}
MOVE = {'kind': 'joint', 'joints_deg': [1., 0., 0., 0., 0., 0.], 'speed_percent': 5}


@contextmanager
def running(tmp_path, mode='always', limits=None):
    policy = InteractionPolicy(mode=mode, limits=limits or ExecutionLimits())
    s = RobotService(SimBackend(), Settings(data_dir=tmp_path, managed_control=True, interaction_policy=policy))
    with TestClient(create_app(s, 'm'*48, 'o'*48)) as c:
        assert c.post('/v1/connect', headers=MODEL).status_code == 200
        yield s, c


def acquire(c):
    response = c.post('/operator/session', headers=OPERATOR, json={'owner': 'test-console'})
    assert response.status_code == 200, response.text
    return {'X-Piper-Control-Session': response.json()['session_id']}


def move(c, sid, ident='integration-job-1', command=None):
    return c.post('/v1/move', headers=MODEL | sid, json={'request_id': ident, 'command': command or MOVE})


def approve(c, sid, j):
    return c.post('/operator/approvals/'+j['job_id'], headers=OPERATOR | sid, json={'approved': True}).json()


def wait(c, j):
    deadline=time.monotonic()+5
    while j['status'] in ('accepted', 'running'):
        assert time.monotonic()<deadline
        time.sleep(.01)
        j=c.get('/v1/jobs/'+j['job_id'], headers=MODEL).json()
    return j


def test_http_session_approval_and_dedup(tmp_path):
    with running(tmp_path) as (s,c):
        assert move(c, {}).json()['error']['code']=='missing_control_session'
        sid=acquire(c)
        j=move(c,sid).json()
        assert j['status']=='awaiting_approval' and not s.backend.commands
        assert move(c,sid).json()['job_id']==j['job_id']
        assert wait(c,approve(c,sid,j))['status']=='succeeded'
        count=len(s.backend.commands)
        assert move(c,sid).json()['job_id']==j['job_id']
        assert len(s.backend.commands)==count


@pytest.mark.parametrize('change',['pose','stop','policy','disconnect','stale'])
def test_pending_invalidated_without_command(tmp_path,change):
    with running(tmp_path) as (s,c):
        sid=acquire(c); j=move(c,sid).json()
        if change=='pose': s.backend.q[0]=.2
        elif change=='stop': c.post('/v1/stop',headers=MODEL)
        elif change=='disconnect': c.post('/v1/disconnect',headers=MODEL)
        elif change=='stale': s.backend.stale=True
        else:
            policy=InteractionPolicy(mode='auto').model_dump()
            assert c.put('/operator/interaction',headers=OPERATOR|sid,json={'policy':policy}).status_code==200
        result=approve(c,sid,j)
        assert result['status']=='cancelled'
        assert not s.backend.commands


@pytest.mark.parametrize('mode',['always','risk','auto'])
def test_hard_cap_cannot_be_approved(tmp_path,mode):
    with running(tmp_path,mode,ExecutionLimits(max_speed_percent=3)) as (s,c):
        sid=acquire(c)
        assert move(c,sid).json()['error']['code']=='speed_limit'
        assert not s.backend.commands
        assert c.get('/operator/interaction',headers=OPERATOR).json()['pending']==[]


def test_model_token_cannot_write_operator_policy(tmp_path):
    with running(tmp_path) as (s,c):
        sid=acquire(c)
        assert c.put('/operator/interaction',headers=MODEL|sid,json={'policy':{'mode':'auto'}}).status_code==401
        assert c.post('/operator/session',headers=MODEL,json={'owner':'model'}).status_code==401


def test_mcp_and_http_share_admission(tmp_path):
    with running(tmp_path) as (s,c):
        headers=MODEL | {'Accept':'application/json, text/event-stream','MCP-Protocol-Version':'2025-11-25'}
        def call(sid, ident):
            result=c.post('/mcp',headers=headers|sid,json={'jsonrpc':'2.0','id':1,'method':'tools/call',
                'params':{'name':'robot_move_joints','arguments':{'joints_deg':MOVE['joints_deg'],'request_id':ident}}})
            assert result.status_code==200,result.text
            return result.json()['result']
        assert 'missing_control_session' in json.dumps(call({},'mcp-no-session-1'))
        sid=acquire(c)
        result=call(sid,'mcp-approval-1')
        assert not result.get('isError'),result
        j=result.get('structuredContent') or json.loads(result['content'][0]['text'])
        assert j['status']=='awaiting_approval' and not s.backend.commands
        assert wait(c,approve(c,sid,j))['status']=='succeeded'


def test_status_poll_does_not_hide_session_expiry(tmp_path):
    with running(tmp_path) as (s,c):
        clock=[0.];s.sessions._now=lambda: clock[0]
        sid=acquire(c);j=move(c,sid).json()
        clock[0]=6.
        c.get('/v1/state',headers=MODEL)
        s._expire_control()
        assert s.get_job(j['job_id'])['status']=='cancelled'
        assert move(c,sid,ident='after-expiry-1').json()['error']['code']=='session_expired'
        assert not s.backend.commands
        new=acquire(c)
        assert new!=sid


def test_expiry_during_action_finishes_once(tmp_path):
    with running(tmp_path,'auto') as (s,c):
        clock=[0.];s.sessions._now=lambda: clock[0]
        sid=acquire(c)
        j=move(c,sid).json();clock[0]=6.;s._expire_control()
        assert wait(c,j)['status']=='succeeded'
        n=len(s.backend.commands)
        assert move(c,sid,ident='after-running-expiry').json()['error']['code']=='session_expired'
        assert move(c,sid).json()['job_id']==j['job_id']
        assert len(s.backend.commands)==n


def test_stop_cancels_pending_when_request_lock_temporarily_busy(tmp_path):
    import threading
    with running(tmp_path) as (s,c):
        sid=acquire(c);j=move(c,sid).json()
        held=threading.Event();release=threading.Event()
        def hold():
            with s.lock: held.set();release.wait(2)
        thread=threading.Thread(target=hold);thread.start();assert held.wait(1)
        try: s.stop()
        finally: release.set();thread.join()
        s._expire_control()
        assert s.get_job(j['job_id'])['status']=='cancelled'
        assert not s.backend.commands


def test_stale_policy_proposal_rejected(tmp_path):
    with running(tmp_path) as (s,c):
        sid=acquire(c);old=s.policy.model_dump()
        assert c.put('/operator/interaction',headers=OPERATOR|sid,json={'policy':old}).status_code==200
        response=c.put('/operator/interaction',headers=OPERATOR|sid,json={'policy':old})
        assert response.json()['error']['code']=='policy_changed'


def test_unclaimed_managed_runtime_exits_and_prevents_late_acquire(tmp_path):
    from piperx_middleware.models import DomainError
    with running(tmp_path) as (s,c):
        exits=[];s.on_idle_session_loss=lambda:exits.append(True)
        s._managed_started=time.monotonic()-31
        s._expire_control()
        assert exits==[True]
        with pytest.raises(DomainError) as exc:s.acquire_session('late-owner')
        assert exc.value.code=='shutting_down'


def test_operator_profile_configuration_is_persistent_and_disconnects(tmp_path):
    with running(tmp_path) as (s,c):
        sid=acquire(c);j=move(c,sid).json()
        changes={'allow_motion':False,'control_profile':'calibration','max_move_deg':2.}
        assert c.patch('/operator/settings',headers=MODEL|sid,json={'changes':changes}).status_code==401
        response=c.patch('/operator/settings',headers=OPERATOR|sid,json={'changes':changes})
        assert response.status_code==200,response.text
        assert response.json()['reconnect_required'] and not response.json()['motion_sent']
        saved=Settings.model_validate_json((tmp_path/'config.json').read_text())
        assert saved.allow_motion is False and saved.max_move_deg==2.
        assert not s.backend.snapshot().connected
        assert s.get_job(j['job_id'])['status']=='cancelled'
        assert not s.backend.commands
        forbidden=c.patch('/operator/settings',headers=OPERATOR|sid,json={'changes':{'backend':'agx'}})
        assert forbidden.json()['error']['code']=='invalid_settings'


def test_profile_rejects_invalid_value_without_mutation(tmp_path):
    with running(tmp_path) as (s,c):
        sid=acquire(c)
        response=c.patch('/operator/settings',headers=OPERATOR|sid,json={'changes':{'allow_motion':'false'}})
        assert response.status_code==422
        assert s.settings.allow_motion is True and s.backend.snapshot().connected
