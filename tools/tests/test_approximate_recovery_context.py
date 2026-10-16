import copy
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_approximate_recovery import recovery_payload


def test_recovery_context_does_not_dispatch_or_reuse_history():
    payload={'messages':[{'role':'system','content':'original'},
                         {'role':'user','content':'current RGB-D'}], 'tools':[]}
    before=copy.deepcopy(payload)
    value=recovery_payload(payload, {'previous_plan_hypothesis':'banana in box',
        'last_reconciled_action':{'request_id':'old-action','status':'succeeded'}})
    assert payload==before
    assert value['messages'][-1]['content']=='current RGB-D'
    text=value['messages'][1]['content']
    for phrase in ('UNVERIFIED MODEL HYPOTHESIS','DO NOT replay','Old observation IDs',
                   'velocities were set to zero','old-action','banana in box'):
        assert phrase in text
    assert value['tools']==[]


def test_queue_paused_primary_waits_without_model_or_actions(tmp_path,monkeypatch):
    import json
    import run_approximate_recovery as recovery
    parent=tmp_path/'primary';parent.mkdir()
    (parent/'state.json').write_text(json.dumps({'status':'paused','reason':'quota'}))
    (tmp_path/'recovery.json').write_text(json.dumps({'wait_for_job':str(parent),'wait_for_session':'unused'}))
    calls=[]
    monkeypatch.setattr(recovery,'verify',lambda p:None)
    def wait(seconds):
        assert not calls
        state=json.loads((tmp_path/'state.json').read_text())
        assert state['status']=='waiting_for_explicit_primary_resume'
        assert state['model_calls']==0 and state['actions_dispatched']==0
        (parent/'state.json').write_text(json.dumps({'status':'complete'}))
    async def run(job):calls.append(job)
    monkeypatch.setattr(recovery.time,'sleep',wait)
    monkeypatch.setattr(recovery,'run',run)
    recovery.queued(tmp_path)
    assert calls==[tmp_path]
