"""One bounded inference worker; capture/control threads never wait inside submit."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import time
import math


@dataclass(frozen=True)
class Ticket:
    request_id: str
    state_version: int
    epoch: int
    deadline: float

    def __post_init__(self):
        if not isinstance(self.request_id,str) or not self.request_id:raise ValueError('missing_request_id')
        for value in (self.state_version,self.epoch):
            if not isinstance(value,int) or isinstance(value,bool) or value<0:raise ValueError('invalid_request_version')
        if isinstance(self.deadline,bool) or not isinstance(self.deadline,(float,int)) or not math.isfinite(self.deadline) or self.deadline<=0:
            raise ValueError('invalid_request_deadline')


class ModelWorker:
    def __init__(self, model):
        self.model = model
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vision-model")
        self.active = None
        self.closed=False

    def submit(self, ticket, prompt, images, schema):
        self.submit_call(ticket,self.model.infer,prompt,images,schema)

    def submit_call(self,ticket,function,*args,**kwargs):
        if self.closed:raise RuntimeError('model_worker_closed')
        if self.active is not None:
            raise RuntimeError("model_busy: poll or cancel the previous request")
        if ticket.deadline <= time.monotonic():
            raise ValueError("request_expired")
        self.active = (ticket, self.pool.submit(function,*args,**kwargs))

    def poll(self, *, state_version, epoch):
        if self.closed:
            request_id=self.active[0].request_id if self.active else None
            return {'status':'discarded','reason':'worker_cancelled','request_id':request_id}
        if self.active is None:
            return {"status": "idle"}
        ticket, future = self.active
        if not future.done():
            return {"status": "expired_pending" if time.monotonic() > ticket.deadline else "pending"}
        self.active = None
        if time.monotonic() > ticket.deadline or ticket.state_version != state_version or ticket.epoch != epoch:
            return {"status": "discarded", "reason": "expired_or_scene_changed", "request_id": ticket.request_id}
        try:
            return {"status": "ok", "value": future.result(), "request_id": ticket.request_id}
        except Exception as exc:
            return {"status": "error", "reason": str(exc), "request_id": ticket.request_id}

    def close(self):
        if self.closed:return
        self.closed=True
        try:
            if self.active is not None and not self.active[1].done():
                self.active[1].cancel()
                if callable(getattr(self.model,'cancel',None)):self.model.cancel()
        finally:self.pool.shutdown(wait=False, cancel_futures=True)
