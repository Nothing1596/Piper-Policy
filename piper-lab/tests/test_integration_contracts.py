"""Regressions at data/control boundaries; hardware tests remain separate."""
import json
from pathlib import Path
import numpy as np
import pytest
from piperlab import data,learning,mock_mcap
from piperlab.safety import SafetyGate,Rejected,load_config

def test_cdr_provenance_depth_and_source_timestamps(tmp_path):
    mock_mcap.generate(tmp_path/'source.mcap',episodes=2,frames=6)
    result=data.convert_mcap(str(tmp_path/'source.mcap'),str(tmp_path/'normalized'))
    assert result['synthetic'] is True and result['total_frames']==12
    manifest=data.load_manifest(tmp_path/'normalized'); ep=data.load_episode(tmp_path/'normalized',manifest['episodes'][0])
    assert ep['source_stamps_ns'].shape==(6,4)
    assert np.all(ep['source_stamps_ns']<=ep['stamps_ns'][:,None])
    assert ep['depth'].dtype==np.uint16 and ep['sample_valid'].all()

def test_convert_multiple_recordings_preserves_episode_split(tmp_path):
    src=tmp_path/'bags'; src.mkdir()
    for i in range(2): mock_mcap.generate(src/f'{i}.mcap',episodes=1,frames=4)
    result=data.convert_mcap(str(src),str(tmp_path/'converted'))
    assert result['episodes']==2 and result['splits']=={'train':[0],'val':[1]}
    assert result['synthetic'] is True

def test_status_events_do_not_require_episode_markers():
    stamp=1_000_000_000
    records=[(stamp,'/lab/events',{'data':json.dumps({'kind':'status','synthetic':True})}),
             (stamp,'/lab/observation',{'name':data.JOINT_NAMES,'position':[0.]*7})]
    episodes,_=data._collect_records(records,.001)
    assert len(episodes)==1

def test_gpu_uuid_is_not_replaced_with_index(monkeypatch):
    monkeypatch.setattr(learning,'_nvidia_smi_gpu_table',lambda:[('3','GPU-changed-order')])
    assert learning._resolve_cuda_visible_devices('GPU-changed-order')=='GPU-changed-order'

def test_split_overlap_refused(tmp_path):
    (tmp_path/'piperlab_export.json').write_text(json.dumps({'splits':{'train':[0,1],'val':[1]}}))
    with pytest.raises(learning.LearningError): learning.dataset_split(tmp_path,'train')

def test_operator_stop_cancels_policy_without_its_token():
    c=load_config(Path(__file__).resolve().parents[1]/'config/hardware.yaml')
    g=SafetyGate(c,clock=lambda:100.,wall_clock=lambda:100.)
    g.observe(g.names,[0,1,-1,0,0,0,.05],100.);g.observe_camera(100.)
    token=g.control('acquire','policy');g.control('arm','policy',token)
    g.control('stop','teleop','')
    assert not g.armed and g.owner is None and g.fault=='manual_stop'
    with pytest.raises(Rejected): g.control('arm','policy',token)
