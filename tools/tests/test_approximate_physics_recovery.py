"""Pose recovery is startup-only, hash-bound, and cannot claim exact continuity."""
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from approximate_physics_recovery import apply_recorded_pose


def fixture(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, 'mujoco', SimpleNamespace(
        mjtJoint=SimpleNamespace(mjJNT_FREE=0), mj_forward=lambda *a: None))
    model = SimpleNamespace(body_jntadr=np.array([0,0,0]), body_parentid=np.array([0,0,0]),
        body_jntnum=np.array([0,1,0]), jnt_type=np.array([0]), jnt_qposadr=np.array([8]))
    data = SimpleNamespace(qpos=np.zeros(15), qvel=np.ones(14), qacc_warmstart=np.ones(14),
        ctrl=np.zeros(8), time=10., xpos=np.array([[0,0,0],[0,0,0],[.4,0,0]]),
        xquat=np.array([[1,0,0,0]]*3))
    backend=SimpleNamespace(connected=False,worker_thread=None,model=model,data=data,
        object_ids={'cube':1,'box':2},arm_qpos_indices=list(range(6)),gripper_qpos_indices=[6,7],
        arm_act_indices=list(range(6)),gripper_act_indices=[6,7],reference_velocity=np.ones(6),active_target=True)
    state={'sim_time_s':100.,'bilateral_contact_objects':[],
        'robot':{'q_deg':[1,2,3,4,5,6],'gripper_width_m':.04},
        'objects':{'cube':{'kind':'dynamic','position':[.2,.1,.05],'quaternion':[1,0,0,0]},
                   'box':{'kind':'container','position':[.4,0,0],'quaternion':[1,0,0,0]}}}
    path=tmp_path/'snapshot.json'
    def write():
        path.write_text(json.dumps(state));return {'snapshot':str(path),'snapshot_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    return backend,state,write


def test_pose_only_restore_and_cold_hold(tmp_path,monkeypatch):
    backend,state,write=fixture(tmp_path,monkeypatch)
    receipt=apply_recorded_pose(backend,write())
    assert np.allclose(backend.data.qpos[:6],np.deg2rad([1,2,3,4,5,6]))
    assert np.allclose(backend.data.qpos[8:],[.2,.1,.05,1,0,0,0])
    assert np.allclose(backend.data.ctrl,backend.data.qpos[:8])
    assert not backend.data.qvel.any() and not backend.data.qacc_warmstart.any()
    assert backend.data.time==0 and receipt['source_sim_time_s']==100
    assert receipt['actions_replayed']==0 and 'not_continuous' in receipt['scope']


def test_hash_and_live_backend_rejected(tmp_path,monkeypatch):
    backend,state,write=fixture(tmp_path,monkeypatch);config=write()
    Path(config['snapshot']).write_text('{}')
    with pytest.raises(ValueError,match='changed'):apply_recorded_pose(backend,config)
    backend.connected=True
    with pytest.raises(ValueError,match='precede'):apply_recorded_pose(backend,config)


def test_held_pose_requires_audited_control_not_an_action_replay(tmp_path,monkeypatch):
    backend,state,write=fixture(tmp_path,monkeypatch)
    state['bilateral_contact_objects']=['cube']
    config=write()
    record=tmp_path/'control.json'
    record.write_text(json.dumps({'result':{'status':'succeeded','command':{'kind':'gripper','width_m':.01}}}))
    config['gripper_controller_record']={'path':str(record),'sha256':hashlib.sha256(record.read_bytes()).hexdigest()}
    receipt=apply_recorded_pose(backend,config)
    assert np.allclose(backend.data.qpos[6:8],[.02,.02])
    assert np.allclose(backend.data.ctrl[6:8],[.005,.005])
    assert receipt['actions_replayed']==0
    record.write_text('{}')
    with pytest.raises(ValueError,match='record changed'):apply_recorded_pose(backend,config)


@pytest.mark.parametrize('change', ['held','static','parent','identity','nan'])
def test_invalid_restore_does_not_mutate_pose(tmp_path,monkeypatch,change):
    backend,state,write=fixture(tmp_path,monkeypatch)
    if change=='held':state['bilateral_contact_objects']=['cube']
    if change=='static':state['objects']['box']['position']=[9,9,9]
    if change=='parent':backend.model.body_parentid[1]=2
    if change=='identity':del state['objects']['box']
    if change=='nan':state['objects']['cube']['position']=[float('nan'),0,0]
    before=backend.data.qpos.copy()
    with pytest.raises(ValueError):apply_recorded_pose(backend,write())
    assert np.array_equal(backend.data.qpos,before)
