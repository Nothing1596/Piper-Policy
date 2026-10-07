"""Recovery admission and approval tests; no hardware is opened."""
import pytest
from pydantic import ValidationError

from piperx_middleware.approval_policy import PolicyEngine
from piperx_middleware.interaction_types import InteractionPolicy, ExecutionLimits
from piperx_middleware.models import Settings, JOINT_LIMITS_DEG, DomainError
from test_interaction_integration import running, acquire, move, approve, wait, MODEL, OPERATOR


def enable(c, sid, mode='auto'):
    current = c.get('/operator/interaction', headers=OPERATOR).json()['policy']
    current.update(force=True, mode=mode)
    r = c.put('/operator/interaction', headers=OPERATOR | sid, json={'policy': current})
    assert r.status_code == 200, r.text
    return r.json()['policy']


@pytest.mark.parametrize('index', range(6))
@pytest.mark.parametrize('side', [0, 1])
def test_five_degree_boundary_and_direction(index, side):
    engine = PolicyEngine(InteractionPolicy(force=True, mode='auto'))
    lo, hi = JOINT_LIMITS_DEG[index]
    q = [0.] * 6; q[index] = lo - 5 if side == 0 else hi + 5
    state = {'q_deg': q}
    command = {'kind': 'joint', 'joints_deg': q.copy()}
    assert engine.evaluate(command, state, settings=Settings())['decision'] == 'ask'
    command['joints_deg'][index] += .001 if side else -.001
    with pytest.raises(DomainError, match='hold or move inward'):
        engine.evaluate(command, state, settings=Settings())
    command['joints_deg'][index] = lo if side == 0 else hi
    assert engine.evaluate(command, state, settings=Settings())['decision'] == 'ask'
    q[index] += .001 if side else -.001
    with pytest.raises(DomainError) as exc:
        engine.evaluate(command, state, settings=Settings())
    assert exc.value.code == 'recovery_limit'


def test_no_new_violation_and_site_bounds():
    policy = InteractionPolicy(force=True, limits=ExecutionLimits(joint_upper_deg=[10.,180.,0.,89.,89.,180.]))
    engine = PolicyEngine(policy)
    q = [12.,0.,0.,0.,0.,0.]
    result = engine.evaluate({'kind':'control_mode'}, {'q_deg':q}, settings=Settings())
    assert result['recovery_bounds_deg'][0][1] == 12.
    with pytest.raises(DomainError):
        engine.evaluate({'kind':'joint', 'joints_deg':[13.,0.,0.,0.,0.,0.]}, {'q_deg':q}, settings=Settings())
    with pytest.raises(DomainError):
        PolicyEngine(InteractionPolicy(force=True)).evaluate(
            {'kind':'joint', 'joints_deg':[151.,0.,0.,0.,0.,0.]}, {'q_deg':[0.]*6}, settings=Settings())


@pytest.mark.parametrize('mode', ['auto','risk','always'])
def test_switch_then_recover_each_requires_confirmation(tmp_path, mode):
    with running(tmp_path, mode) as (s,c):
        sid=acquire(c); enable(c,sid,mode)
        original=[0.,-1.168,1.056,-39.249,-89.516,58.842]
        s.backend.q=original.copy(); s.backend.ctrl_mode=2; s.backend.motion_mode=0
        j=c.post('/v1/control-mode', headers=MODEL|sid, json={'request_id':'force-mode-0001'}).json()
        assert j['status']=='awaiting_approval' and not s.backend.commands
        assert c.post('/operator/approvals/'+j['job_id'], headers=MODEL|sid, json={'approved':True}).status_code==401
        assert wait(c,approve(c,sid,j))['status']=='succeeded'
        assert s.backend.q==original
        target=[0.,0.,0.,-39.249,-89.,58.842]
        request={'kind':'joint','joints_deg':target}
        j=move(c,sid,'force-recover-0001',request).json()
        assert j['status']=='awaiting_approval' and len(s.backend.commands)==1
        assert wait(c,approve(c,sid,j))['status']=='succeeded'
        n=len(s.backend.commands)
        assert s.backend.q==target
        assert move(c,sid,'force-recover-0001',request).json()['job_id']==j['job_id']
        assert len(s.backend.commands)==n
        j=move(c,sid,'force-recover-0002',request).json()
        assert j['status']=='awaiting_approval'


@pytest.mark.parametrize('change', ['pose','policy','stop','stale','fault','session'])
def test_approval_invalidations_send_nothing(tmp_path, change):
    with running(tmp_path,'auto') as (s,c):
        sid=acquire(c); policy=enable(c,sid)
        s.backend.q=[0.,-1.,0.,0.,0.,0.]
        j=move(c,sid,'force-invalidate-01',{'kind':'joint','joints_deg':[0.]*6}).json()
        assert j['status']=='awaiting_approval'
        if change=='pose':s.backend.q[1]=-.5
        elif change=='policy':
            policy['force']=False
            c.put('/operator/interaction',headers=OPERATOR|sid,json={'policy':policy})
        elif change=='stop':c.post('/v1/stop',headers=MODEL)
        elif change=='stale':s.backend.stale=True
        elif change=='fault':s.backend.fault='driver fault'
        else:
            s.sessions._now=lambda:1e20
            s._expire_control()
        if change=='session': assert s.get_job(j['job_id'])['status']=='cancelled'
        else: assert approve(c,sid,j)['status']=='cancelled'
        assert not s.backend.commands


def test_force_defaults_off_strict_and_operator_only(tmp_path):
    assert InteractionPolicy().force is False
    with pytest.raises(ValidationError):InteractionPolicy(force='true')
    with running(tmp_path,'auto') as (s,c):
        sid=acquire(c)
        assert c.put('/operator/interaction',headers=MODEL|sid,json={'policy':{'force':True}}).status_code==401
        s.backend.q[1]=-1.
        r=c.post('/v1/control-mode',headers=MODEL|sid,json={'request_id':'no-force-mode-01'})
        assert r.json()['error']['code']=='joint_limits'
        assert not s.backend.commands


def test_controller_limits_not_relaxed(tmp_path):
    with running(tmp_path) as (s,c):
        sid=acquire(c); enable(c,sid)
        s.backend.q[1]=-1.
        s.measured_limits={'available':True,'connection_epoch':s.epoch,
            'joints':[{'available':True,'min_deg':lo,'max_deg':hi} for lo,hi in JOINT_LIMITS_DEG]}
        r=c.post('/v1/control-mode',headers=MODEL|sid,json={'request_id':'hardware-bounds-01'})
        assert r.json()['error']['code']=='measured_joint_limits'
        assert not s.backend.commands


def test_mcp_force_cannot_skip_approval(tmp_path):
    with running(tmp_path,'auto') as (s,c):
        sid=acquire(c);enable(c,sid)
        s.backend.q[1]=-1.
        headers=MODEL|sid|{'Accept':'application/json, text/event-stream','MCP-Protocol-Version':'2025-11-25'}
        r=c.post('/mcp',headers=headers,json={'jsonrpc':'2.0','id':1,'method':'tools/call',
            'params':{'name':'robot_set_control_mode','arguments':{'request_id':'force-mcp-001'}}})
        assert r.status_code==200
        assert 'awaiting_approval' in r.text and not s.backend.commands


def test_force_requires_session_even_unmanaged_sim(tmp_path):
    from piperx_middleware.backends import SimBackend
    from piperx_middleware.service import RobotService
    from piperx_middleware.models import ControlMode
    s=RobotService(SimBackend(),Settings(data_dir=tmp_path,interaction_policy=InteractionPolicy(force=True)))
    try:
        s.connect(); s.backend.q[1]=-1.
        with pytest.raises(DomainError) as exc:s.move(ControlMode(),'force-no-session-1')
        assert exc.value.code=='missing_control_session'
        assert not s.backend.commands
    finally:s.close()


def test_stop_between_approval_evaluation_and_admission_is_rejected(tmp_path, monkeypatch):
    """Reproduce the CLI reviewer's proposed race; the generation check blocks it."""
    with running(tmp_path,'auto') as (s,c):
        sid=acquire(c);enable(c,sid)
        s.backend.q[1]=-1.
        j=move(c,sid,'force-stop-race-01',{'kind':'joint','joints_deg':[0.]*6}).json()
        evaluate=s._evaluate_interaction
        def stop_after_evaluate(*args, **kwargs):
            result=evaluate(*args, **kwargs)
            s.stop()
            return result
        monkeypatch.setattr(s,'_evaluate_interaction',stop_after_evaluate)
        result=approve(c,sid,j)
        assert result['status']=='cancelled'
        assert result['error_code']=='approval_changed'
        assert not s.backend.commands
