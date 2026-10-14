"""Failed/aborted termination is not successful placement or a motion request."""
import json
import sys
from pathlib import Path
import pytest
from jsonschema import validate, ValidationError
sys.path.insert(0,str(Path(__file__).parents[1]))
import run_longvil_online as online
from test_longvil_online import fixture, Session
from test_longvil_softplan_recovery import driver, decision
from test_longvil_recovery_score import carry, P0, P1


@pytest.mark.parametrize('outcome', ['failed', 'aborted'])
@pytest.mark.asyncio
async def test_unsuccessful_closeout_needs_no_release_or_motion(tmp_path, monkeypatch, outcome):
    args, evidence = fixture(tmp_path)
    args.strict_completion = True
    body = {'outcome':outcome, 'summary':'Could not establish a grasp', 'uncertainties':['No movement attempted']}
    validate(body, online.FINISH['inputSchema'])
    session = Session()
    monkeypatch.setattr(online, 'paid', driver([decision('finish_task', body, 'finish')], []))
    report = {}
    await online.online_loop(session, args, evidence, lambda row:None, tmp_path, report, None)
    assert report['status'] == outcome and report['model_outcome'] == outcome
    assert report['motions'] == 0 and report['api_calls'] == 1
    assert not session.moves
    assert not any(n in online.EXECUTIONS for n, _ in session.calls)
    assert json.loads((tmp_path/'action-chain.json').read_text()) == []


@pytest.mark.asyncio
async def test_strict_success_still_rejects_no_release_then_failed_finishes(tmp_path, monkeypatch):
    args, evidence = fixture(tmp_path)
    args.strict_completion = True
    session = Session(); events = []
    responses = [decision('move_to', {'xyz_m':[.3,.1,.2], 'request_id':'motion-001', 'observation_id':'live-0001'}, 'move'),
                 decision('finish_task', {'summary':'done', 'uncertainties':[], 'observation_id':'live-0002'}, 'bad-finish'),
                 decision('finish_task', {'outcome':'failed', 'summary':'No controlled release', 'uncertainties':[]}, 'failed-finish')]
    monkeypatch.setattr(online, 'paid', driver(responses, []))
    async def remote(*args):return {'sim_time_s':session.time,'objects':{},'bilateral_contact_objects':[]}
    monkeypatch.setattr(online, 'remote', remote)
    report={}
    await online.online_loop(session,args,evidence,events.append,tmp_path,report,None)
    assert report['status']=='failed' and len(session.moves)==1
    assert any('Finish requires measured release' in json.dumps(e) for e in events)


def test_failure_does_not_clear_unknown_and_completion_still_requires_feedback():
    g=online.OnlineGate(False);g.unknown=True
    g.before('finish_task', {'outcome':'failed'},0)
    assert g.unknown
    with pytest.raises(ValueError):g.before('finish_task', {'outcome':'completed'},0)
    for body in ({'summary':'done','uncertainties':[]}, {'outcome':'success','summary':'done','uncertainties':[]}):
        with pytest.raises(ValidationError):validate(body,online.FINISH['inputSchema'])


@pytest.mark.parametrize('reason', ['grip_lost', 'grip_lost_between_decisions'])
def test_drop_reason_precedes_absent_normal_release_snapshot(reason):
    t=carry('block',P0,P1,controlled=False)
    t['end_reason']=reason;del t['state_before_release']
    result=online.verify_transfer(t)
    assert not result['valid'] and result['reason_code']=='grip_lost'
    assert reason in result['reasons'][0] and 'malformed' not in result['reasons'][0]


def test_actual_corrupt_or_missing_snapshot_names_the_phase():
    t=carry('block',P0,P1);del t['state_before_release']
    assert online.verify_transfer(t)['reasons']==['missing state_before_release']
    t=carry('block',P0,P1);t['state_held']={}
    assert 'invalid state_held' in online.verify_transfer(t)['reasons'][0]
    assert online.verify_transfer(carry('block',P0,P1))['valid']
