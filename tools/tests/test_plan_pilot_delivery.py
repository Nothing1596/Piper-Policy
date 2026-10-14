"""No model calls: checked evidence projection and frozen delivery boundaries."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest

sys.path.insert(0,str(Path(__file__).parents[1]))
from prepare_plan_pilot import digest,packet,select_tasks
from build_plan_pilot import build,preflight
from longvil_budget import BudgetPause


def test_packet_only_contains_delivered_frames():
    frame={'frame_id':'f1','timestamp_s':.3,'quality':{'clarity':1},'detections':[]}
    p=packet('source',[frame])
    assert p['frames']==[frame]
    assert 'full_video_context' not in p['evaluation'] and 'events' not in p['evaluation']
    assert p['evaluation']['object_detection']['frames'][0]['frame_id']=='f1'


def test_approved_fixed_sample_has_only_complex_and_disjoint_development():
    source=Path('/Users/nothing/DEV/piper-longvil-v2-20261010/formal-job')
    if not source.exists():pytest.skip('Local reviewed dataset not available')
    tasks,dev,_=select_tasks(source)
    assert len(tasks)==9 and all(list(t['videos'])==['complex'] for t in tasks)
    assert dev['id'] not in {t['id'] for t in tasks}


def test_gates_fail_before_any_job_directory_is_created(tmp_path):
    root=tmp_path/'root';root.mkdir();dev=tmp_path/'dev/results';dev.mkdir(parents=True)
    (dev/'development-gates.json').write_text(json.dumps({'complete':True,'plan_gate_passed':False}))
    with pytest.raises(ValueError,match='plan_gate_passed'):preflight(root,dev.parent)


def fixture(tmp_path):
    root=tmp_path/'root';root.mkdir();repo=tmp_path/'repo'
    for part in ['piperx-cli/src','piper-lab/src','tools']:(repo/part).mkdir(parents=True)
    for part in ['piperx-cli/src','piper-lab/src']:(repo/part/'module.py').write_text('# local source')
    for name in ['plan_protocol.py','run_plan_enforcement_pilot.py','run_longvil_v2.py','longvil_metrics.py','longvil_budget.py']:(repo/'tools'/name).write_text('# fixture')
    annotation=root/'reference.json';annotation.write_text('{"object_list":[]}')
    video=root/'complex.mp4';video.write_bytes(b'video');image=root/'image.jpg';image.write_bytes(b'pixels')
    task={'id':'t','annotation':str(annotation),'sha256':digest(annotation),
          'videos':{'complex':{'path':str(video),'sha256':digest(video)}}}
    (root/'manifest.json').write_text(json.dumps({'tasks':[task],'development_task':dict(task,id='dev')}))
    p=packet('source',[{'frame_id':'f1','timestamp_s':0,'image_path':str(image),'image_sha256':digest(image)}])
    (root/'evidence.json').write_text(json.dumps({i:{'initial':p,'supplement':p} for i in ['t','dev']}))
    evaluator=root/'evaluate.py';evaluator.write_text('# fixture')
    return SimpleNamespace(root=root,repo=repo,job=tmp_path/'job',development=True,development_job=None,
        evaluator=evaluator,node=Path('/unused/node'),key_launcher=Path('/unused/key-launcher'),
        cli=Path('/unused/cli'),readme=None,submit=False),video,image


def test_snapshot_is_independent_and_config_contains_no_credential(tmp_path,monkeypatch):
    args,video,image=fixture(tmp_path);monkeypatch.setenv('DEEPSEEK_API_KEY','sentinel-secret')
    build(args);video.write_bytes(b'changed');image.write_bytes(b'changed')
    frozen=args.job/'private-data/t'
    assert (frozen/'complex.mp4').read_bytes()==b'video'
    assert (frozen/'initial-00.jpg').read_bytes()==b'pixels'
    assert 'sentinel-secret' not in (args.job/'job.json').read_text()
    assert all(digest(args.job/n)==h for n,h in json.loads((args.job/'frozen-hashes.json').read_text()).items())


def test_changed_accepted_package_prevents_pilot(tmp_path):
    args,_,_=fixture(tmp_path);build(args)
    gates={'complete':True,'plan_gate_passed':True,'interface_gate_passed':True,
           'resource_gate_passed':True,'cost_gate_passed':True,'projected_total_cny':1}
    (args.job/'results').mkdir();(args.job/'results/development-gates.json').write_text(json.dumps(gates))
    (args.repo/'piperx-cli/src/module.py').write_text('# changed after acceptance')
    args.development_job=args.job;args.job=tmp_path/'pilot';args.development=False
    with pytest.raises(ValueError,match='accepted package'):build(args)
    assert not args.job.exists()


def forecast_fixture(tmp_path):
    args,_,_=fixture(tmp_path);build(args)
    gates={'complete':True,'plan_gate_passed':True,'interface_gate_passed':True,
           'resource_gate_passed':True,'cost_gate_passed':False,'projected_total_cny':12.26963198}
    gate_path=args.job/'results/development-gates.json';gate_path.parent.mkdir()
    gate_path.write_text(json.dumps(gates))
    authorization={'approved':True,'scope':'forecast-submission-only','projection_ceiling_cny':12.5,
                   'actual_stop_at_cny':12,'limit_cny':15,'development_job':str(args.job.resolve()),
                   'development_gates_sha256':digest(gate_path),'user_instruction':'允许，启动提交作业'}
    auth_path=args.root/'forecast-authorization.json';auth_path.write_text(json.dumps(authorization))
    args.development_job=args.job;args.job=tmp_path/'pilot';args.development=False
    args.forecast_authorization=auth_path
    return args,gate_path,auth_path


def test_projection_exception_requires_explicit_matching_authorization(tmp_path):
    args,gate_path,auth_path=forecast_fixture(tmp_path)
    with pytest.raises(ValueError,match='cost_gate_passed'):preflight(args.root,args.development_job)
    authorization=json.loads(auth_path.read_text());authorization['actual_stop_at_cny']=12.5
    auth_path.write_text(json.dumps(authorization))
    with pytest.raises(ValueError,match='authorization'):preflight(args.root,args.development_job,auth_path)
    authorization['actual_stop_at_cny']=12;auth_path.write_text(json.dumps(authorization))
    gate_path.write_text(gate_path.read_text()+'\n')
    with pytest.raises(ValueError,match='snapshot'):preflight(args.root,args.development_job,auth_path)


def test_authorized_forecast_is_frozen_without_rewriting_development_or_spending_caps(tmp_path):
    args,gate_path,auth_path=forecast_fixture(tmp_path);original=gate_path.read_bytes()
    config=build(args)
    assert gate_path.read_bytes()==original
    accepted=json.loads((args.job/'development-gates.json').read_text())
    assert accepted['cost_gate_passed'] is False and accepted['forecast_gate_passed'] is True
    assert config['budget']=={'limit_cny':15,'stop_at_cny':12,'forecast_submission_ceiling_cny':12.5}
    assert json.loads((args.job/'forecast-authorization.json').read_text())==json.loads(auth_path.read_text())
    hashes=json.loads((args.job/'frozen-hashes.json').read_text())
    assert hashes['forecast-authorization.json']==digest(args.job/'forecast-authorization.json')
    (args.root/'usage-ledger.jsonl').write_text(json.dumps({'status':'accounted','estimated_cny':12})+'\n')
    with pytest.raises(BudgetPause,match='checkpoint'):preflight(args.root,args.development_job,auth_path)


def test_authorization_does_not_bypass_forecast_ceiling_or_protocol_gate(tmp_path):
    args,gate_path,auth_path=forecast_fixture(tmp_path)
    gates=json.loads(gate_path.read_text());authorization=json.loads(auth_path.read_text())
    for field,value,expected in [('projected_total_cny',12.5001,'Projection'),('plan_gate_passed',False,'plan_gate_passed')]:
        gates[field]=value;gate_path.write_text(json.dumps(gates))
        authorization['development_gates_sha256']=digest(gate_path);auth_path.write_text(json.dumps(authorization))
        with pytest.raises(ValueError,match=expected):preflight(args.root,args.development_job,auth_path)
