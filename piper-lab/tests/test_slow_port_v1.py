import threading
import time
from types import SimpleNamespace
import pytest
from piperlab.models.slow_port import VisionSlowModelPort
from piperlab.models.worker import Ticket


def test_nonfinite_deadlines_are_not_an_infinite_budget():
    for value in (float('nan'),float('inf'),-1,True):
        with pytest.raises(ValueError):Ticket('id',0,0,value)


def test_slow_port_preserves_request_identity_and_cancels():
    class Model:
        entered=threading.Event();released=threading.Event()
        def infer(self,*args):self.entered.set();self.released.wait(2);return {'ok':True}
        def cancel(self):self.released.set()
    model=Model();port=VisionSlowModelPort(model)
    request=SimpleNamespace(request_id='request',state_version=7,evidence_sequences=(1,3),
        detail={'clock_domain':'host_monotonic','acquisition_epoch':4,'deadline_host_monotonic_s':time.monotonic()+3,
                'prompt':'synthetic','images':[],'schema':{'type':'object'}})
    accepted=port.submit(request)
    assert accepted['request_id']=='request' and accepted['acquisition_epoch']==4 and accepted['evidence_ref']==[1,3]
    assert model.entered.wait(1) and not port.available()
    active=port.worker.active
    result=port.cancel()
    assert result['status'] in ('cancelled','cancellation_requested')
    assert active[1].result(timeout=1)=={'ok':True}
    assert port.poll(state_version=7,epoch=4)['status']=='discarded'
    assert not port.available()
    with pytest.raises(RuntimeError,match='closed'):port.submit(request)
    port.close()


def test_slow_port_refuses_unmapped_clock():
    port=VisionSlowModelPort(None)
    try:
        with pytest.raises(ValueError,match='mapped_clock'):
            port.submit(SimpleNamespace(detail={'clock_domain':'device'}))
    finally:port.close()
