import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pytest
from agy_online_decider import AgyDecider, retryable_admission


def rejection():
    return dict(status='ERROR', error='Eligibility check failed: UNAVAILABLE (code 503)',
                conversation_id='', response='', num_turns=0,
                usage=dict(input_tokens=0, output_tokens=0, thinking_tokens=0,
                           cache_read_tokens=0, total_tokens=0))


def test_only_explicit_zero_work_admission_is_retryable():
    r = rejection()
    assert retryable_admission(r)
    for changes in [{'status':'SUCCESS'}, {'num_turns':1}, {'conversation_id':'active'},
                    {'usage':None}, {'response':'partial'}, {'error':'connection reset'},
                    {'error':'Eligibility check failed: code 401'}]:
        assert not retryable_admission({**r, **changes})


@pytest.mark.asyncio
async def test_bounded_retry_and_saved_attempts(tmp_path, monkeypatch):
    import agy_online_decider as m
    d = AgyDecider(tmp_path/'d')
    calls=[]; delays=[]
    async def request(prompt):
        calls.append(prompt)
        return rejection()
    async def sleep(seconds): delays.append(seconds)
    monkeypatch.setattr(d, '_request_once', request)
    monkeypatch.setattr(m.asyncio, 'sleep', sleep)
    assert (await d.request('same prompt'))['status'] == 'ERROR'
    assert len(calls) == 4 and delays == [5,15,45]
    assert len(list((d.root/'admission-retries').glob('*.json'))) == 3


@pytest.mark.asyncio
async def test_success_and_unknown_outcome_not_retried(tmp_path, monkeypatch):
    d = AgyDecider(tmp_path/'d'); calls=[]
    async def request(prompt):
        calls.append(prompt)
        raise ConnectionError('unknown provider outcome')
    monkeypatch.setattr(d, '_request_once', request)
    with pytest.raises(ConnectionError): await d.request('prompt')
    assert len(calls)==1


@pytest.mark.asyncio
async def test_admission_recovery_returns_single_decision(tmp_path, monkeypatch):
    import agy_online_decider as m
    d = AgyDecider(tmp_path/'d'); calls=[]
    success = {'status':'SUCCESS', 'response':'one decision'}
    async def request(prompt):
        calls.append(prompt)
        return rejection() if len(calls)==1 else success
    async def sleep(seconds): pass
    monkeypatch.setattr(d, '_request_once', request)
    monkeypatch.setattr(m.asyncio, 'sleep', sleep)
    assert await d.request('unchanged') is success
    assert calls == ['unchanged','unchanged'] and d.turn == 0
