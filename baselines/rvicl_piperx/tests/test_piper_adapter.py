import json
import os
from pathlib import Path
from types import SimpleNamespace
import threading

import numpy as np
import pytest

from rvicl_ig.agent import AgentTools, tool_specs
from rvicl_ig.perception import Perception
from rvicl_ig.piper_cli import verify_sources
from rvicl_ig.piper_prompt import piper_prompt
from rvicl_ig.piper_scene import PiperScene

RPENT = Path(os.environ.get("RPENT_ROOT", Path(__file__).resolve().parents[1] / "deps/RPent"))


def test_native_source_pin_and_recursive_protocol_are_preserved():
    verify_sources(RPENT)
    from rvicl_ig.agent import PROMPT
    prompt = piper_prompt()
    recursive = lambda text: text.split("RECURSIVE VIDEO PROTOCOL\n")[1].split("WHAT THE")[0]
    assert recursive(prompt).replace("PiperX grippers", "Panda grippers") == recursive(PROMPT)
    assert "task_plan" not in prompt and "harness_review" not in prompt
    assert "single-arm" not in PROMPT  # the original prompt is not rewritten in place


def test_default_inference_profile_matches_the_remote_reference():
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "piper_simulation.json").read_text())
    lock = json.loads((root / "BASELINE.lock.json").read_text())
    assert lock["rpent"]["commit"] == "c9026c2bf9c1342b88cdc682e60d578770f871d3"
    assert (config["model"], config["reasoning_effort"], config["primitive_image_history"]) == (
        "gpt-6.1-sol", "low", "latest")
    assert config["sam3_enabled"] is True
    assert (config["sam3_endpoint"], config["sam3_protocol"]) == ("http://127.0.0.1:8114", "rest")
    assert config["max_turns"] == 100 and config["max_episode_steps"] == 500


@pytest.mark.parametrize("profile,cameras", [
    ("piperx", ("head", "left_wrist")),
    ("panda", ("head", "left_wrist", "right_wrist")),
])
@pytest.mark.parametrize("task_instruction", [None, "Undo the demonstrated operation."])
def test_latest_history_prompt_uses_actual_robot_cameras(tmp_path, monkeypatch, profile, cameras, task_instruction):
    from rvicl_ig.planner import solve_native
    from rpent.planner.api_loop import ApiAgentLoop
    captured = {}

    def solve(self, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(error=None, finish_result=None, messages=[], stats={"turns_used": 0})

    monkeypatch.setattr(ApiAgentLoop, "solve", solve)
    scene = SimpleNamespace(robot_profile=profile, steps=0, truncated=False, cameras=cameras,
        observe=lambda: {"state": {"steps": 0}, "_images": []})
    client = SimpleNamespace(base="http://fixture.invalid/v1", key="fixture-not-a-secret",
        model="fixture", effort="low", timeout=1, calls=0)
    job = {"rpent_root": str(RPENT), "primitive_image_history": "latest", "max_turns": 2,
           "sam3_enabled": True}
    if task_instruction:
        job.update(language_instruction=task_instruction, camera_perspective="front", spatial_reference="camera")
    job["oracle_plan"] = "PRIVATE MUST NOT REACH POLICY"
    solve_native(client, scene, SimpleNamespace(page_count=1), job, tmp_path, prompt=piper_prompt())
    history = captured["system_prompt"].split("\n\nIMAGE HISTORY\n")[1]
    assert f"latest live {'/'.join(cameras)} camera batch" in history
    if profile == "piperx":
        assert "right_wrist" not in captured["system_prompt"]
    message = json.loads(captured["user_message"])
    assert message.get("language_instruction") == task_instruction
    assert "PRIVATE MUST NOT REACH POLICY" not in captured["user_message"]


def test_pin_verification_covers_tool_dispatch_source(tmp_path):
    import shutil
    from rvicl_ig.piper_cli import ROOT
    lock = json.loads((ROOT / "BASELINE.lock.json").read_text())
    # A different native dispatcher must fail before model or executor startup.
    for rel in lock["rpent"]["files"]:
        target = tmp_path / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(RPENT / rel, target)
    with (tmp_path / "rpent/tools/toolkit.py").open("a") as stream:
        stream.write("\n# changed dispatcher\n")
    with pytest.raises(ValueError, match="Frozen RPent source changed: rpent/tools/toolkit.py"):
        verify_sources(tmp_path)


def test_only_robot_capabilities_change_and_original_panda_profile_remains():
    names = lambda specs: {spec["function"]["name"]: spec["function"] for spec in specs}
    piper = names(tool_specs(RPENT, False, robot_profile="piperx"))
    panda = names(tool_specs(RPENT, False))
    assert "move_both" not in piper and "move_both" in panda
    assert piper["move_to"]["parameters"]["properties"]["arm"]["enum"] == ["left"]
    assert panda["move_to"]["parameters"]["properties"]["arm"]["enum"] == ["left", "right"]
    assert piper["back_project"]["parameters"]["properties"]["camera"]["enum"] == ["head", "left_wrist", "agentview", "wrist"]
    assert not {"task_plan", "guide", "harness_review", "move_both"} & piper.keys()


def test_right_arm_refused_before_any_io():
    scene = SimpleNamespace(robot_profile="piperx", truncated=False, check_cancelled=lambda: None)
    tools = AgentTools(scene, SimpleNamespace(ready=True), RPENT)
    for name, arguments in [("move_to", {"arm": "right", "xyz": [0, 0, 0]}),
                            ("move_both", {"left": None, "right": None})]:
        with pytest.raises(ValueError, match="Only arm"):
            tools.dispatch(name, arguments)
    assert tools._primitives is None


def test_two_camera_captures_refresh_at_same_control_step(tmp_path):
    perception = Perception(RPENT, tmp_path, {"camera_names": ["head", "left_wrist"], "sam3_enabled": False}, client=object())
    def views(depth):
        return {c: {"rgb": np.zeros((6, 8, 3), dtype=np.uint8), "depth_m": np.full((6, 8), depth),
                    "intrinsic_cv": [[2, 0, 4], [0, 2, 3], [0, 0, 1]],
                    "extrinsic_cv": np.eye(4), "source_stamp_s": float(depth)} for c in ("head", "left_wrist")}
    first = perception.record(views(1), 0, capture_id=1)
    second = perception.record(views(2), 0, capture_id=2)
    assert second == first + 1
    for camera in perception.states:
        old = perception.back_project(camera=camera, step=first, row=3, col=4, resolution="high")
        new = perception.back_project(camera=camera, step=second, row=3, col=4, resolution="high")
        assert old != new
    with pytest.raises(ValueError, match="configured camera"):
        perception.record({"head": views(1)["head"]}, 1, capture_id=3)


def bare_scene(tmp_path, client):
    scene = object.__new__(PiperScene)
    scene.client, scene.directory, scene.steps = client, tmp_path, 0
    scene.paused, scene.last_error = None, None
    scene._exit = threading.Event()
    scene._execution = None
    scene.job = {}
    scene._operator_trace = []
    scene.identity = ("original", 1)
    return scene


def test_lost_post_response_reconciles_same_request_without_replay(tmp_path, monkeypatch):
    calls = []
    def call(method, path, body=None):
        calls.append((method, path, body))
        if path == "/v1/state":
            return {"ready": True, "instance_id": "original", "connection_epoch": 1}
        if method == "POST":
            return {"error": {"code": "transport_unknown"}}
        return {"status": "succeeded"}
    scene = bare_scene(tmp_path, SimpleNamespace(call=call))
    assert scene._action({"kind": "joint", "timeout_s": 1.})
    assert sum(method == "POST" for method, _, _ in calls) == 1
    post = next(body for method, _, body in calls if method == "POST")
    assert calls[-1][1] == "/v1/requests/" + post["request_id"]
    assert len(list((tmp_path / "requests").glob("*.json"))) == 1


def test_unreconciled_outcome_blocks_next_motion(tmp_path):
    calls = []
    def call(method, path, body=None):
        calls.append((method, path))
        if path == "/v1/state":
            return {"ready": True, "instance_id": "original", "connection_epoch": 1}
        return {"error": {"code": "transport_unknown" if method == "POST" else "not_found"}}
    scene = bare_scene(tmp_path, SimpleNamespace(call=call))
    for _ in range(2):
        with pytest.raises(RuntimeError, match="no replay"):
            scene._action({"kind": "joint", "timeout_s": 1.})
    assert sum(method == "POST" for method, _ in calls) == 1


def test_replaced_executor_never_reconnects_or_executes(tmp_path):
    calls = []
    def call(method, path, body=None):
        calls.append((method, path))
        return {"ready": True, "instance_id": "replacement", "connection_epoch": 1}
    scene = bare_scene(tmp_path, SimpleNamespace(call=call))
    with pytest.raises(RuntimeError, match="identity changed"):
        scene._action({"kind": "joint", "timeout_s": 1.})
    assert calls == [("GET", "/v1/state")]


def test_executor_outcome_unknown_is_not_an_ordinary_failed_motion(tmp_path):
    def call(method, path, body=None):
        if path == "/v1/state":
            return {"ready": True, "instance_id": "original", "connection_epoch": 1}
        return {"status": "outcome_unknown"}
    scene = bare_scene(tmp_path, SimpleNamespace(call=call))
    with pytest.raises(RuntimeError, match="Uncertain action"):
        scene._action({"kind": "joint", "timeout_s": 1.})
    assert scene.paused and scene.steps == 0


def test_empty_closing_preserves_cancelled_receipt_after_verified_width(tmp_path, monkeypatch):
    stopped, stamp, posts = False, 0., []
    def call(method, path, body=None):
        nonlocal stopped, stamp
        if path == "/v1/state":
            stamp += 1
            return {"ready": True, "instance_id": "original", "connection_epoch": 1, "active_job_id": "one",
                    "robot": {"gripper_stamp_s": stamp, "gripper_age_s": 0., "gripper_width_m": .02}}
        if method == "POST":
            posts.append(path)
            if path == "/v1/stop":
                stopped = True
                return {"status": "cancellation_requested"}
        return {"status": "cancelled" if stopped else "running", "job_id": "one"}
    monkeypatch.setattr("rvicl_ig.piper_scene.time.sleep", lambda seconds: None)
    scene = bare_scene(tmp_path, SimpleNamespace(call=call))
    scene.feedback_timeout_s = .15
    assert scene._action({"kind": "gripper", "completion": "bilateral_contact", "width_m": .02, "timeout_s": 1.})
    assert posts == ["/v1/move", "/v1/stop"]
    receipt = json.loads(next((tmp_path / "receipts").glob("*.json")).read_text())
    assert receipt["status"] == "cancelled"
    assert receipt["adapter_completion"]["measured_width_confirmed"]
    assert receipt["adapter_completion"]["grasp_verified"] is False


def test_capture_caption_keeps_two_camera_timestamps():
    from rvicl_ig.image_history import describe_tool_images
    result = describe_tool_images({"_images": [{"path": "unused", "camera": "left_wrist", "camera_group": ["head", "left_wrist"],
        "source_stamp_s": 10.2, "sim_time_s": .5}]}, "view_env_state")
    caption = result["_images"][0]["caption"]
    assert "10.2" in caption and "not hardware synchronized" in caption
    assert "right_wrist" not in caption


def test_path_uses_existing_piper_resolver_without_an_extra_cartesian_planner(tmp_path):
    from scipy.spatial.transform import Rotation
    from piperx_middleware.kinematics import PiperKinematics
    from piperx_middleware.models import Settings, MoveTo
    from piperx_middleware.primitives import resolve_primitive
    from piperx_middleware.mujoco_backend import RECOMMENDED_TCP_OFFSET_M, RECOMMENDED_TCP_RPY_DEG
    settings = Settings(tcp_offset_m=RECOMMENDED_TCP_OFFSET_M, tcp_offset_rpy_deg=RECOMMENDED_TCP_RPY_DEG)
    kin = PiperKinematics(settings.tcp_offset_m, settings.tcp_offset_rpy_deg)
    q = [0., 30., -30., 0., 0., 0.]
    goal = kin.matrix([2., 32., -33., 1., 0., 0.])
    quat = Rotation.from_matrix(goal[:3, :3]).as_quat(scalar_first=True)
    scene = bare_scene(tmp_path, SimpleNamespace(call=lambda *args: {
        "ready": True, "instance_id": "original", "connection_epoch": 1, "robot": {"q_deg": q}}))
    scene.settings = settings
    result = scene.plan_arm_path([*goal[:3, 3], *quat])
    expected, _ = resolve_primitive(MoveTo(xyz_m=goal[:3, 3].tolist(),
        rpy_deg=Rotation.from_quat(quat, scalar_first=True).as_euler("xyz", degrees=True).tolist()), q, settings)
    assert result["status"] == "Success"
    np.testing.assert_allclose(np.rad2deg(result["position"][-1]), expected.joints_deg, atol=1e-8)
    path = np.vstack([q, np.rad2deg(result["position"])])
    assert np.max(np.abs(np.diff(path, axis=0))) <= 2.000001


def test_tool_axis_facts_match_the_frozen_mujoco_asset():
    import mujoco
    from piperx_middleware.mujoco_backend import MujocoBackend
    backend = MujocoBackend()
    try:
        mujoco.mj_forward(backend.model, backend.data)
        tcp = backend.data.site_xmat[backend.tcp_site_id].reshape(3, 3)
        for name in ("gripper_link1", "gripper_link2"):
            body = mujoco.mj_name2id(backend.model, mujoco.mjtObj.mjOBJ_BODY, name)
            slide_axis = backend.data.xmat[body].reshape(3, 3)[:, 2]
            assert abs(tcp[:, 0] @ slide_axis) > .999999
            assert abs(tcp[:, 2] @ slide_axis) < 1e-6
    finally:
        backend.close()


def test_execution_samples_have_call_identity_without_persisting_internal_world_maps(tmp_path):
    import base64
    import io
    from PIL import Image
    rgb = io.BytesIO()
    Image.new("RGB", (8, 6), "blue").save(rgb, format="JPEG")
    stamp = 0.
    def call(method, path, body=None):
        nonlocal stamp
        stamp += 1
        return {"instance_id": "original", "connection_epoch": 1, "source_stamp_s": stamp,
                "sim_time_s": stamp, "rgb_jpeg_b64": base64.b64encode(rgb.getvalue()).decode()}
    scene = bare_scene(tmp_path, SimpleNamespace(call=call))
    scene._last_stamps, scene._process_metadata, scene.records = {}, {}, []
    scene.observe = lambda: (_ for _ in ()).throw(AssertionError("Historical RGB must not call observe/world_map"))
    with scene.record_tool("move_to", {}, turn=7, call_id="fixture-call"):
        with scene.capture_execution("move_to", {}) as frames:
            scene.steps += 1
            scene._capture_process()
    assert len(frames) == 4  # two times, two cameras
    assert all(item["source_turn"] == 7 and item["source_call_id"] == "fixture-call" for item in frames)
    assert not (tmp_path / "observations").exists()


@pytest.mark.parametrize("stopped", [True, False])
def test_owned_executor_cleanup_is_confirmed_and_repeat_close_is_idempotent(tmp_path, stopped):
    scene = bare_scene(tmp_path, None)
    calls = []
    scene._heartbeat = scene.operator = None
    scene.connection = SimpleNamespace(owned=True)
    scene.manager = SimpleNamespace(release_sync=lambda connection: calls.append(connection) or {"stopped": stopped})
    if stopped:
        scene.close()
        scene.close()
        assert len(calls) == 1
        assert scene.connection is None
    else:
        with pytest.raises(RuntimeError, match="shutdown unconfirmed"):
            scene.close()
        assert scene.connection is not None
        assert json.loads((tmp_path / "cleanup.json").read_text())["stopped"] is False


def test_operator_trace_is_saved_separately_from_model_tool_record(tmp_path):
    sample={'sim_time_s':1., 'objects':{'cube_red':{'position':[.1,.2,.3]}}}
    scene=bare_scene(tmp_path, SimpleNamespace())
    scene.job={'operator_evaluation_trace':True}
    scene.records=[]
    scene.operator=SimpleNamespace(call=lambda *a: sample)
    with scene.record_tool('view_env_state', {}, turn=1, call_id='test') as record:
        record['observed']=True
    assert 'objects' not in record
    saved=json.loads((tmp_path/'operator_trace.json').read_text())
    assert saved[0]['objects']==sample['objects'] and saved[0]['call_id']=='test'
    assert 'objects' not in json.loads((tmp_path/'tools.json').read_text())[0]


def test_schema_text_compatibility_only_repeats_existing_schema(monkeypatch):
    from rvicl_ig.piper_cli import SchemaTextClient
    from rvicl_ig.io import Client
    captured={}
    def complete(self,messages,**kwargs):
        captured.update(messages=messages,kwargs=kwargs)
        return {'content':'{}'}
    monkeypatch.setattr(Client,'complete',complete)
    client=object.__new__(SchemaTextClient)
    schema={'type':'object','properties':{'summary':{'type':'string'}}}
    original=[{'role':'system','content':'ORIGINAL HIERARCHY'}, {'role':'user','content':[]}]
    client.complete(original,schema=schema,artifact='fixture.json')
    assert original[0]['content']=='ORIGINAL HIERARCHY'
    assert captured['messages'][1]==original[1]
    assert captured['kwargs']['schema'] is schema
    assert json.dumps(schema) in captured['messages'][0]['content']


@pytest.mark.parametrize('current,target,completion', [(.070000000001,1.,'width'),(.0705,1.,'width'),(.07,.5,'bilateral_contact')])
def test_near_equal_gripper_values_are_width_hold_not_contact_close(tmp_path,current,target,completion):
    scene=bare_scene(tmp_path,SimpleNamespace())
    scene.job={'max_episode_steps':500,'seed':0}
    scene.gripper_max_m=.07
    scene._state=lambda:{'robot':{'gripper_width_m':current}}
    calls=[]
    scene._action=lambda cmd:calls.append(cmd) or True
    result=scene.apply_updates([{'gripper':target}])
    assert result['executed_actions']==1 and calls[0]['completion']==completion


def test_explicit_contact_direction_rejection_is_not_unknown_or_replayed(tmp_path):
    calls=[]
    def call(method,path,body=None):
        calls.append((method,path))
        return {'error':{'code':'contact_completion_requires_closing','message':'known rejected before admission'}}
    scene=bare_scene(tmp_path,SimpleNamespace(call=call))
    scene._state=lambda:{}
    assert not scene._action({'kind':'gripper','width_m':.07,'timeout_s':1.})
    assert calls==[('POST','/v1/move')] and scene.paused is None
    assert len(list((tmp_path/'receipts').glob('*.json')))==1


def test_hardware_parameters_come_from_active_runtime_without_task_or_private_settings():
    from rvicl_ig.piper_prompt import hardware_parameters
    from piperx_middleware.models import Settings, JOINT_LIMITS_DEG
    from piperx_middleware.kinematics import PIPERX_MDH
    settings=Settings(gripper_max_m=.055,tcp_offset_m=[.01,.02,.13],data_dir=Path('/private/OPERATOR_ONLY'))
    facts=hardware_parameters(settings)
    assert facts['joint_limits_deg']==[list(p) for p in JOINT_LIMITS_DEG]
    assert facts['kinematics']['rows']==[list(p) for p in PIPERX_MDH]
    assert facts['gripper']['configured_width_range_m']==[0.,.055]
    assert facts['flange_to_tcp']['translation_m']==[.01,.02,.13]
    assert facts['ik']['orientation_search'] is False
    text=json.dumps(facts)
    assert 'OPERATOR_ONLY' not in text
    assert not {'target','oracle_plan','recommended_pose','candidate_pose','trajectory','preflight'} & facts.keys()


def test_state_axis_vectors_are_fk_of_current_measurements(tmp_path):
    from piperx_middleware.models import Settings
    from piperx_middleware.kinematics import PiperKinematics
    from rvicl_ig.piper_prompt import hardware_parameters
    q=[10.,30.,-30.,20.,-15.,5.]
    scene=bare_scene(tmp_path,None)
    scene.job={'max_episode_steps':500}
    scene.settings=Settings(tcp_offset_m=[0.,0.,.1425])
    scene.robot_parameters=hardware_parameters(scene.settings)
    scene.last_ik_feedback=None
    scene.kinematics=PiperKinematics(scene.settings.tcp_offset_m,scene.settings.tcp_offset_rpy_deg)
    scene._state=lambda:{'robot':{'q_deg':q,'gripper_width_m':.04},'observation_identity':{'instance_id':'test'}}
    state=scene.state()
    R=scene.kinematics.matrix(q)[:3,:3]
    axes=state['arms']['left']['tcp_axes_world']
    np.testing.assert_allclose(np.column_stack([axes[a] for a in ['x','y','z']]),R)
    assert state['arms']['left']['joints_deg']==q
    assert state['last_ik_feedback'] is None
    assert state['robot_parameters']['gripper']['configured_width_range_m']==[0.,.07]


def test_ik_failure_reports_only_requested_pose_and_actual_solver_error_then_clears(tmp_path,monkeypatch):
    from piperx_middleware.models import Settings,DomainError
    from scipy.spatial.transform import Rotation
    q=[0.,30.,-30.,0.,0.,0.]
    scene=bare_scene(tmp_path,None)
    scene.settings=Settings(tcp_offset_m=[0.,0.,.1425])
    scene._state=lambda:{'robot':{'q_deg':q}}
    target=[.21,-.12,.19,.9238795325,.3826834324,0.,0.]
    def fail(*args,**kwargs):raise DomainError('ik_no_solution','original finite solver failure',422)
    monkeypatch.setattr('rvicl_ig.piper_scene.resolve_primitive',fail)
    result=scene.plan_arm_path(target)
    assert result['position'] is None
    feedback=scene.last_ik_feedback
    assert feedback['requested_xyz_m']==target[:3] and feedback['seed_joints_deg']==q
    assert feedback['error_code']=='ik_no_solution' and feedback['executed_steps']==0
    assert not feedback['collision_checked'] and not feedback['global_unreachability_proven']
    R=Rotation.from_quat(target[3:],scalar_first=True).as_matrix()
    np.testing.assert_allclose(np.column_stack([feedback['requested_tcp_axes_world'][a] for a in ['x','y','z']]),R)
    assert not {'candidate','suggested_quaternion','closest_solution','residual','violating_joint'} & feedback.keys()
    monkeypatch.setattr('rvicl_ig.piper_scene.resolve_primitive',lambda *a:(SimpleNamespace(joints_deg=q),{}))
    assert scene.plan_arm_path(target)['status']=='Success'
    assert scene.last_ik_feedback['status']=='solution_found' and 'solver_message' not in scene.last_ik_feedback


@pytest.mark.parametrize('profile', ['piperx','panda'])
def test_piper_failure_corrects_only_public_diagnostic_not_native_primitive(profile):
    old={'completed':False,'stop_reason':'plan_failed','hint':'target may be unreachable or in collision'}
    feedback={'error_code':'ik_no_solution','collision_checked':False,'requested_xyz_m':[.1,.2,.3]}
    scene=SimpleNamespace(robot_profile=profile,truncated=False,check_cancelled=lambda:None,
                          last_ik_feedback=feedback,observe=lambda:{'state':{},'_images':[]})
    tools=AgentTools(scene,SimpleNamespace(ready=True),str(RPENT))
    tools._primitives=SimpleNamespace(move_to=lambda **a:old)
    result=tools.dispatch('move_to',{'arm':'left','xyz':[.1,.2,.3]})['execution']
    assert old=={'completed':False,'stop_reason':'plan_failed','hint':'target may be unreachable or in collision'}
    if profile=='piperx':
        assert result['ik_feedback']==feedback
        assert result['upstream_hint']==old['hint'] and 'Collision was not checked' in result['hint']
    else:assert result is old
