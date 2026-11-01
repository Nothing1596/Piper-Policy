import json
from pathlib import Path

import pytest

from rvicl_ig.piper_evaluate import build_jobs, evaluation_spec, score_metrics


def inputs(tmp_path):
    (tmp_path / "demo.mp4").write_bytes(b"unit-test video identity; not a decoded clip")
    (tmp_path / "scene.xml").write_text("<mujoco/>")
    spec = {"scope": "piper_scene_metrics", "predicates": [
        {"path": ["objects", "cube_blue", "in_tray"], "equals": True},
        {"path": ["objects", "cube_blue", "grasped"], "equals": False}]}
    (tmp_path / "goal.json").write_text(json.dumps(spec))
    return {"job_id": "one", "arm_count": 1, "video": "demo.mp4", "simulation_asset": "scene.xml",
            "evaluation_file": "goal.json", "language_instruction": "Undo what the person did.",
            "camera_perspective": "front", "spatial_reference": "camera",
            "oracle_plan": "PRIVATE ANSWER MUST NOT REACH POLICY"}


def test_single_arm_binding_preserves_public_instruction_and_separates_goals(tmp_path):
    row = inputs(tmp_path)
    job = build_jobs([row], {"model": "fixture", "oracle_plan": "private"}, relative_to=tmp_path)[0]
    assert not job["blocked"]
    assert job["config"]["language_instruction"] == row["language_instruction"]
    assert job["config"]["camera_perspective"] == "front"
    assert "PRIVATE" not in json.dumps(job["config"])
    assert not {"evaluation_file", "evaluation", "predicates", "oracle_plan"} & job["config"].keys()
    assert job["evaluation"] == str(tmp_path / "goal.json")
    assert set(job["input_identity"]) == {"video", "simulation_asset", "evaluation_file"}


def test_missing_scene_never_silently_uses_default_two_cube_world(tmp_path):
    row = inputs(tmp_path)
    for key in ("simulation_asset", "evaluation_file", "arm_count"):
        row.pop(key)
    job = build_jobs([row], {}, relative_to=tmp_path)[0]
    assert len(job["blocked"]) == 3
    assert "simulation_asset" not in job["config"]


def test_episode_ids_are_distinct_and_duplicate_manifest_ids_rejected(tmp_path):
    row = inputs(tmp_path)
    row.pop("job_id")
    row["env_id"] = "L0_Task-v1"
    first = dict(row, human_episode=0)
    second = dict(row, human_episode=1)
    jobs = build_jobs([first, second], {}, relative_to=tmp_path)
    assert jobs[0]["job_id"] != jobs[1]["job_id"]
    with pytest.raises(ValueError, match="unique"):
        build_jobs([dict(row, job_id="same"), dict(row, job_id="same")], {}, relative_to=tmp_path)


@pytest.mark.parametrize("name", ["../escape", "a/b", "a.b", ""])
def test_job_id_cannot_escape_output_directory(tmp_path, name):
    with pytest.raises(ValueError, match="Job IDs"):
        build_jobs([dict(inputs(tmp_path), job_id=name)], {}, relative_to=tmp_path)


def test_goal_scoring_reads_actual_metrics_and_missing_metric_is_not_failure(tmp_path):
    inputs(tmp_path)
    spec = evaluation_spec(tmp_path / "goal.json")
    assert score_metrics({"objects": {"cube_blue": {"in_tray": True, "grasped": False}}}, spec)["success"]
    assert not score_metrics({"objects": {"cube_blue": {"in_tray": True, "grasped": True}}}, spec)["success"]
    with pytest.raises(ValueError, match="unavailable"):
        score_metrics({}, spec)


def test_symbolic_watchact_goal_cannot_be_misrepresented_as_physical_score(tmp_path):
    (tmp_path / "goal.json").write_text(json.dumps({"scope": "watchact_symbolic", "oracle_plan": []}))
    with pytest.raises(ValueError, match="BDDL"):
        evaluation_spec(tmp_path / "goal.json")


def test_watchact_import_only_reads_public_rows(tmp_path):
    import importlib.util
    script = Path(__file__).resolve().parents[1] / "scripts/import_watchact.py"
    module_spec = importlib.util.spec_from_file_location("watchact_import_fixture", script)
    importer = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(importer)
    (tmp_path / "data").mkdir()
    row = {"id": "sample_1", "original_id": "original_1", "language_instruction": "Restore the cup.",
           "camera_perspective": "front", "video": "front/one.MP4", "spatial_reference": "human",
           "bddl_file": "Imitation/one.bddl", "oracle_plan": "PRIVATE"}
    (tmp_path / "data/Imitation.jsonl").write_text(json.dumps(row) + "\n")
    output = tmp_path / "manifest.json"
    importer.main(["--data-root", str(tmp_path), "--output", str(output)])
    jobs = json.loads(output.read_text())
    assert jobs[0]["language_instruction"] == "Restore the cup."
    assert jobs[0]["arm_count"] == 1
    assert "PRIVATE" not in output.read_text()
    assert "simulation_asset" not in jobs[0]  # LIBERO scenes are not silently converted.


def test_boolean_video_time_is_rejected(tmp_path):
    row = inputs(tmp_path)
    row['video'] = {'path': 'demo.mp4', 'start_s': False}
    with pytest.raises(ValueError, match='interval'):
        build_jobs([row], {}, relative_to=tmp_path)


def test_physical_preflight_rejects_invalid_asset_before_model_use(tmp_path):
    from rvicl_ig.piper_evaluate import validate_scenes
    row = inputs(tmp_path)
    jobs = build_jobs([row], {}, relative_to=tmp_path)
    with pytest.raises(Exception):
        validate_scenes(jobs)


def test_post_scoring_refreshes_summary_without_model_finish_becoming_success(tmp_path):
    from rvicl_ig.io import save
    from rvicl_ig.piper_evaluate import assess
    row = inputs(tmp_path)
    job = build_jobs([row], {}, relative_to=tmp_path)[0]
    save(tmp_path / 'one/rollout/result.json', {'model_declared_status': 'success',
        'operator_scene_metrics': {'objects': {'cube_blue': {'in_tray': False, 'grasped': False}}}})
    report = assess([job], tmp_path, {'results': [{'job_id': 'one', 'kind': 'model', 'status': 'completed'}]})
    assert report['results'][0]['task_success'] is False
    assert report['summary']['scored_tasks'] == 1
    assert report['summary']['task_success_rate'] == 0
