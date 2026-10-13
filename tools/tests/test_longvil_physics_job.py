"""Job gates do not launch paid comparisons after invalid physics or interfaces."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0,str(Path(__file__).parents[1]))
import build_longvil_physics_job as jobs


@pytest.mark.parametrize('failure,expected_paid,status',[
    ('physics',0,'physics_preflight_failed'),
    ('development',1,'development_gate_failed'),
    (None,2,'complete')])
def test_serial_phases_are_gated_without_automatic_retries(tmp_path,monkeypatch,failure,expected_paid,status):
    job=tmp_path/'job';job.mkdir();tasks=[{'id':f'task{i}','annotation':f'/private/{i}.json'} for i in range(10)]
    jobs.save(job/'manifest.json',{'tasks':tasks[:9],'development_task':tasks[-1]})
    jobs.save(job/'launch-config.json',{'socket':'socket','port':123,'host':'host','python':'python','node':'node',
        'remote_root':'/isolated/job','remote_python':'python','pythonpath':'frozen/packages','key_launcher':'launcher'})
    jobs.save(job/'frozen-hashes.json',{})
    def deploy(config,job,log):
        (job/'scene-metadata').mkdir()
        for t in tasks:jobs.save(job/'scene-metadata'/(t['id']+'.json'),{'scene_sha256':'scene'})
    calls=[]
    def run(command,**kwargs):
        calls.append(command)
        folder=Path(command[command.index('--output')+1]);folder.mkdir(parents=True)
        if 'validate_longvil_physics.py' in command[1]:
            ident=folder.name
            jobs.save(folder/'report.json',{'physics_score':{'success':not(failure=='physics' and ident=='task0')},
                'sdk_calls':10,'cleanup':{'runtime_remaining':False,'recording_error':None}})
            jobs.save(folder/'frame-initial-metadata.json',{'frames':[{'sim_time_s':1.,'source_stamp_s':100.}]})
            (folder/'frame-initial.jpg').write_bytes(b'fixed simulated evidence')
        else:
            jobs.save(folder/'summary.json',{'complete':True})
            if '--development' in command:
                jobs.save(folder/'development-gates.json',{'complete':True,'plan_gate_passed':failure!='development',
                    'interface_gate_passed':True,'resource_gate_passed':True})
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(jobs,'deploy',deploy);monkeypatch.setattr(jobs.subprocess,'run',run)
    monkeypatch.setattr(jobs.sys,'platform','linux')
    jobs.supervise(job)
    assert json.loads((job/'state.json').read_text())['status']==status
    assert len([c for c in calls if c[0]=='node'])==expected_paid
    assert len([c for c in calls if 'validate_longvil_physics.py' in c[1]])==10
    assert len({c[c.index('--profile')+1] for c in calls if 'validate_longvil_physics.py' in c[1]})==10
