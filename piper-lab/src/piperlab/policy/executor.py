"""Loopback PiperX client; all motion still passes through the shared executor."""
from __future__ import annotations
import base64
import io
import time
import uuid
from pathlib import Path
from urllib.parse import urlparse
import httpx
import numpy as np
from PIL import Image


class ExecutionError(RuntimeError):
    pass


class PiperExecutor:
    def __init__(self, url, token_file, *, operator_token_file=None):
        if urlparse(url).hostname not in ("127.0.0.1","localhost","::1"):
            raise ValueError("This first-version sensor adapter requires same-host clocks")
        self.url=url.rstrip('/')
        token=Path(token_file).read_text().strip()
        self.client=httpx.Client(base_url=self.url,headers={"Authorization":"Bearer "+token},
                                 timeout=35,trust_env=False)
        self._operator_token=Path(operator_token_file).read_text().strip() if operator_token_file else None

    def request(self, method,path,payload=None, *, operator=False):
        headers={}
        if operator:
            if not self._operator_token:
                raise ExecutionError("operator_token_required")
            headers['Authorization']='Bearer '+self._operator_token
        response=self.client.request(method,path,json=payload,headers=headers)
        data=response.json()
        if response.is_error:
            raise ExecutionError(str(data.get('error',data)))
        return data

    def state(self):
        return self.request('GET','/v1/state')

    def capture(self):
        observation=self.request('GET','/v1/simulation/observation')
        if observation.get('source_clock')!='host_monotonic':
            raise ExecutionError('unmapped_sensor_clock')
        rgb=np.asarray(Image.open(io.BytesIO(base64.b64decode(observation.pop('rgb_jpeg_b64')))).convert('RGB'))
        depth=np.load(io.BytesIO(base64.b64decode(observation.pop('depth_npy_b64'))),allow_pickle=False)
        return {**observation,'rgb':rgb,'depth':depth}

    def primitive(self, command, *, monitor=None):
        return self._execute(command,'/v1/primitives',monitor)

    def joint_move(self,joints_deg,*,monitor=None):
        return self._execute({'kind':'joint','joints_deg':list(joints_deg),'speed_percent':15,'timeout_s':30},'/v1/move',monitor)

    def _execute(self,command,endpoint,monitor):
        request_id=str(uuid.uuid4())
        result=self.request('POST',endpoint,{'command':command,'request_id':request_id})
        deadline=time.monotonic()+command.get('timeout_s',30)+5
        while result['status'] in ('accepted','running'):
            if time.monotonic()>deadline:
                self.request('POST','/v1/stop',{})
                raise ExecutionError('job_deadline')
            if monitor is not None:
                try:
                    monitor()
                except BaseException:
                    self.request('POST','/v1/stop',{})
                    raise
            time.sleep(.1)
            result=self.request('GET','/v1/jobs/'+result['job_id'])
        if result['status']!='succeeded':
            raise ExecutionError(str(result))
        return result

    def stop(self):
        return self.request('POST','/v1/stop',{})

    def evaluate(self):
        # Never passed to VisionPlanner or scene grounding.
        return self.request('GET','/operator/simulation/evaluate',operator=True)

    def close(self):
        self.client.close()
