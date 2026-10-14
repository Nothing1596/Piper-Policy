"""Nonblocking adapter for the existing perception SlowModelPort contract."""
import time
from .worker import ModelWorker,Ticket


class VisionSlowModelPort:
    name='local_vision_worker'

    def __init__(self,model):
        self.worker=ModelWorker(model)

    def available(self):
        return not self.worker.closed and self.worker.active is None

    def submit(self,request):
        detail=request.detail
        if detail.get('clock_domain')!='host_monotonic':
            raise ValueError('slow_model_requires_mapped_clock')
        ticket=Ticket(request.request_id,request.state_version,detail['acquisition_epoch'],
                      detail['deadline_host_monotonic_s'])
        self.worker.submit(ticket,detail['prompt'],detail['images'],detail['schema'])
        return {'status':'accepted','request_id':request.request_id,'state_version':request.state_version,
                'acquisition_epoch':ticket.epoch,'evidence_ref':list(request.evidence_sequences)}

    def poll(self,*,state_version,epoch):
        return self.worker.poll(state_version=state_version,epoch=epoch)

    def close(self):self.worker.close()

    def cancel(self):
        active=self.worker.active
        self.worker.close()
        return {'status':'cancellation_requested' if active and not active[1].done() else 'cancelled',
                'request_id':active[0].request_id if active else None}
