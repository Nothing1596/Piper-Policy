"""Contact completion contract: arrival alone never substitutes for grasp evidence."""
import time
import pytest
from piperx_middleware.backends import SimBackend
from piperx_middleware.models import Settings,GripperMove,DomainError
from piperx_middleware.service import RobotService


class ContactPeer(SimBackend):
    name='mujoco'
    evidence=True
    contact_stale=False
    def snapshot(self):
        state=super().snapshot()
        stamp=time.monotonic()-(10 if self.contact_stale else 0)
        state.diagnostics['gripper_contact']={'supported':self.evidence,'source_timestamp_s':stamp,
                                              'finger1_force_n':1,'finger2_force_n':1 if self.evidence else 0}
        return state


def run(service,command):
    job=service.move(command,'contact-contract-01')
    deadline=time.monotonic()+3
    while job['status'] in ('accepted','running'):
        assert time.monotonic()<deadline
        time.sleep(.02);job=service.get_job(job['job_id'])
    return job


def test_unavailable_contact_sensor_rejected_before_command(tmp_path):
    service=RobotService(SimBackend(),Settings(data_dir=tmp_path));service.connect()
    try:
        with pytest.raises(DomainError,match='requires the MuJoCo'):
            service.move(GripperMove(width_m=.001,completion='bilateral_contact'),'contact-contract-01')
        assert not service.backend.commands
    finally:service.close()


@pytest.mark.parametrize('supported,stale,success',[(True,False,True),(False,False,False),(True,True,False)])
def test_contact_must_be_bilateral_fresh_and_repeated(tmp_path,supported,stale,success):
    backend=ContactPeer();backend.evidence=supported;backend.contact_stale=stale
    service=RobotService(backend,Settings(backend='mujoco',data_dir=tmp_path));service.connect()
    try:
        job=run(service,GripperMove(width_m=.001,completion='bilateral_contact',timeout_s=.7))
        assert (job['status']=='succeeded')==success
        if success:assert job['contact_evidence']['supported'] is True
        else:assert job['error_code']=='execution_timeout'
    finally:service.close()


def test_width_retry_of_older_journal_is_idempotent(tmp_path):
    service=RobotService(SimBackend(),Settings(data_dir=tmp_path));service.connect()
    try:
        command=GripperMove(width_m=.021)
        job=run(service,command)
        assert job['status']=='succeeded'
        job['command'].pop('completion')
        service.store.put(job)
        before=len(service.backend.commands)
        retry=service.move(command,'contact-contract-01')
        assert retry['job_id']==job['job_id']
        assert len(service.backend.commands)==before
    finally:service.close()
