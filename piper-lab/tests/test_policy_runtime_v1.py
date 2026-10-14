import threading
import time
from pathlib import Path
import numpy as np
import pytest
from piperlab.models.worker import ModelWorker,Ticket
from piperlab.policy.observation import ObservationStream


def test_worker_nonblocking_rejects_old_scene_and_deadline():
    gate=threading.Event();worker=ModelWorker(None)
    ticket=Ticket('req',1,0,time.monotonic()+3)
    worker.submit_call(ticket,lambda:gate.wait(1))
    assert worker.poll(state_version=1,epoch=0)['status']=='pending'
    with pytest.raises(RuntimeError,match='busy'):worker.submit_call(ticket,lambda:None)
    gate.set()
    for _ in range(100):
        result=worker.poll(state_version=2,epoch=0)
        if result['status']!='pending':break
        time.sleep(.005)
    assert result['status']=='discarded'
    with pytest.raises(ValueError,match='expired'):
        worker.submit_call(Ticket('old',0,0,time.monotonic()-1),lambda:None)
    worker.close()


def test_camera_latest_only_records_while_planner_waits(tmp_path):
    class Camera:
        count=0
        def capture(self):
            self.count+=1
            return {'rgb':np.zeros((64,64,3),dtype=np.uint8),'source_stamp_s':time.monotonic(),'epoch':0}
    camera=Camera();stream=ObservationStream(camera,tmp_path,period_s=.04)
    stream.start()
    first=stream.latest()['source_stamp_s']
    time.sleep(.18)
    assert stream.latest()['source_stamp_s']>first
    assert camera.count>=3
    assert stream.latest()['state_version']==0
    stream.close()
    import av
    with av.open(str(tmp_path/'simulation.mp4')) as container:
        frames=list(container.decode(video=0))
    assert len(frames)>=3
    assert all(b.pts>b0.pts for b0,b in zip(frames,frames[1:]))
