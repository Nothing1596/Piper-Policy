import json
import os
from pathlib import Path
import signal
import subprocess

import pytest

from rvicl_ig.io import save
from rvicl_ig.piper_batch import run_jobs

RPENT = Path(os.environ.get("RPENT_ROOT", "/Users/nothing/DEV/piper-rvicl-align-20261101-111802/RPent-reference"))


def _make_job(job_id, kind="model", **config_overrides):
    config = {
        "rpent_root": str(RPENT),
        "model": "gpt-6.1-sol",
        "trial_timeout": 30,
        "cleanup_grace_s": 0.2,
        **config_overrides,
    }
    video = {"path": "/tmp/demo.mp4", "start_s": 1.25, "end_s": 9.5} if kind == "model" else None
    evaluation = {"target_object": "cube", "secret_goal_xyz": [0.25, 0.0, 0.1]}
    return {
        "job_id": job_id,
        "kind": kind,
        "config": config,
        "video": video,
        "evaluation": evaluation,
    }


def _write_vlm_trace(directory, *, model="gpt-6.1-sol", status="response_received",
                     started_at="2026-11-01T00:00:00+00:00", usage=None):
    payload = {"status": status, "model": model, "started_at": started_at}
    if usage is not None:
        payload["usage"] = usage
    save(Path(directory) / "metadata.json", payload)


def _populate_rollout(rollout_dir, *, kind="model", declared_status="success", stopped=True,
                      write_cleanup=True, tools_count=3, requests_count=2,
                      hierarchy_usage=None, planner_usages=()):
    rollout_dir = Path(rollout_dir)
    sim_dir = rollout_dir / "simulation"
    save(sim_dir / "tools.json", [{"name": f"tool_{i}"} for i in range(tools_count)])
    for i in range(requests_count):
        save(sim_dir / "requests" / f"rvicl-{i:04d}.json", {"request_id": f"rvicl-{i:04d}"})
    if write_cleanup:
        save(sim_dir / "cleanup.json", {"stopped": stopped})
    if hierarchy_usage is not False:
        _write_vlm_trace(rollout_dir / "hierarchy" / "hierarchy_response", usage=hierarchy_usage)
    for idx, usage in enumerate(planner_usages, 1):
        _write_vlm_trace(rollout_dir / "model_calls" / f"{idx:04d}", usage=usage)
    if kind == "fixture":
        save(rollout_dir / "result.json", {
            "status": "completed",
            "kind": "scripted_provider_real_mujoco_protocol_smoke",
            "model_task_success": None,
            "independent_task_score": None,
        })
    else:
        save(rollout_dir / "result.json", {
            "status": "completed",
            "kind": "native_rpent_model_run",
            "model_declared_status": declared_status,
            "model_task_success": None,
            "operator_scene_metrics": {"distance_m": 0.01},
        })


class FakeProcess:
    def __init__(self, pid=4200, returncode=0, on_wait=None):
        self.pid = pid
        self.returncode = returncode
        self.on_wait = on_wait
        self.signals = []
        self.terminated = False
        self.killed = False
        self.wait_calls = []

    def wait(self, timeout=None):
        self.wait_calls.append(timeout)
        if self.on_wait is not None:
            return self.on_wait(self, timeout)
        return self.returncode

    def send_signal(self, sig):
        self.signals.append(sig)

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True


def _output_arg(cmd):
    return Path(cmd[cmd.index("--output") + 1])


def test_two_serial_workers_and_private_eval_isolation(tmp_path):
    jobs = [
        _make_job("job-alpha", "model", env_file="/tmp/fake.env"),
        _make_job("job-beta", "model"),
    ]
    events = []
    active = 0

    def launch(cmd, stdout=None, stderr=None, start_new_session=None):
        nonlocal active
        assert active == 0, "Workers must execute serially"
        active += 1
        rollout_dir = _output_arg(cmd)
        job_dir = rollout_dir.parent
        receipt_before = json.loads((job_dir / "receipt.json").read_text())
        assert receipt_before["status"] == "running"
        cfg = json.loads((job_dir / "config.json").read_text())
        assert "evaluation" not in cfg
        assert "secret_goal_xyz" not in (job_dir / "config.json").read_text()
        events.append((job_dir.name, list(cmd), start_new_session))
        if stdout is not None:
            stdout.write(f"worker {job_dir.name} ok\n")
            stdout.flush()

        def finish(proc, _timeout):
            nonlocal active
            _populate_rollout(
                rollout_dir,
                hierarchy_usage={"prompt_tokens": 100, "completion_tokens": 25},
                planner_usages=[{"prompt_tokens": 200, "completion_tokens": 50}],
                tools_count=4,
                requests_count=3,
            )
            active -= 1
            return 0

        return FakeProcess(pid=1000 + len(events), returncode=0, on_wait=finish)

    out = run_jobs(jobs, tmp_path, python="/usr/bin/python3", launch=launch)
    assert [item[0] for item in events] == ["job-alpha", "job-beta"]
    assert "--env-file" in events[0][1] and "--env-file" not in events[1][1]
    assert events[0][1][:5] == ["/usr/bin/python3", "-u", "-m", "rvicl_ig.piper_cli", "run"]
    assert all(item[2] == (os.name == "posix") for item in events)

    assert out["paused"] is False
    assert out["completed_task"] == 2
    assert out["errors"] == 0
    assert out["pending"] == 0
    assert out["fixtures"] == 0

    for idx, job_id in enumerate(("job-alpha", "job-beta"), 1):
        receipt = json.loads((tmp_path / job_id / "receipt.json").read_text())
        assert receipt["status"] == "completed"
        assert receipt["pid"] == 1000 + idx
        assert receipt["cleanup_confirmed"] is True
        assert "worker " + job_id in (tmp_path / job_id / "worker.log").read_text()

    rows = json.loads((tmp_path / "results.json").read_text())
    assert len(rows) == 2
    for row in rows:
        assert row["status"] == "completed"
        assert row["model_declared_status"] == "success"
        assert row["task_success"] is None
        assert row["input_tokens"] == 300
        assert row["output_tokens"] == 75
        assert row["model_api_calls"] == 2
        assert row["missing_usage"] == 0
        assert row["model_tool_attempts"] == 4
        assert row["executor_request_count"] == 3
        assert isinstance(row["elapsed_s"], float)
        assert row["operator_scene_metrics"] == {"distance_m": 0.01}
        assert row["child_result"].endswith("rollout/result.json")


def test_terminal_failure_pauses_then_skips_on_resume_without_repeating_failure(tmp_path):
    jobs = [_make_job("job-fail"), _make_job("job-after")]
    launched = []

    def launch_first(cmd, **kwargs):
        rollout_dir = _output_arg(cmd)
        job_id = rollout_dir.parent.name
        launched.append(job_id)
        save(rollout_dir / "simulation" / "cleanup.json", {"stopped": True})
        save(rollout_dir / "error.json", {"status": "paused", "error_type": "RuntimeError", "message": "boom"})
        return FakeProcess(pid=2001, returncode=1)

    first = run_jobs(jobs, tmp_path, launch=launch_first)
    assert launched == ["job-fail"]
    assert first["paused"] is True
    assert first["errors"] == 1
    assert first["pending"] == 1
    assert json.loads((tmp_path / "job-fail" / "receipt.json").read_text())["status"] == "error"

    def launch_resume(cmd, **kwargs):
        rollout_dir = _output_arg(cmd)
        job_id = rollout_dir.parent.name
        launched.append(job_id)
        _populate_rollout(
            rollout_dir,
            hierarchy_usage={"prompt_tokens": 50, "completion_tokens": 10},
            planner_usages=[{"prompt_tokens": 70, "completion_tokens": 20}],
        )
        return FakeProcess(pid=2002, returncode=0)

    second = run_jobs(jobs, tmp_path, launch=launch_resume)
    assert launched == ["job-fail", "job-after"]
    assert second["paused"] is False
    assert second["errors"] == 1
    assert second["completed_task"] == 1
    assert second["pending"] == 0
    statuses = {r["job_id"]: r["status"] for r in second["results"]}
    assert statuses == {"job-fail": "error", "job-after": "completed"}


def test_running_receipt_with_result_and_cleanup_reconciles_without_relaunch(tmp_path):
    jobs = [_make_job("job-reconcile"), _make_job("job-next")]
    job_dir = tmp_path / "job-reconcile"
    save(job_dir / "receipt.json", {
        "job_id": "job-reconcile",
        "kind": "model",
        "status": "running",
        "pid": 3333,
        "started_at": "2026-11-01T00:00:00+00:00",
        "elapsed_s": 12.5,
    })
    _populate_rollout(
        job_dir / "rollout",
        stopped=True,
        hierarchy_usage={"prompt_tokens": 80, "completion_tokens": 15},
        planner_usages=[{"prompt_tokens": 120, "completion_tokens": 25}],
    )

    launched = []

    def launch(cmd, **kwargs):
        rollout_dir = _output_arg(cmd)
        launched.append(rollout_dir.parent.name)
        _populate_rollout(
            rollout_dir,
            stopped=True,
            hierarchy_usage={"prompt_tokens": 40, "completion_tokens": 10},
            planner_usages=[],
        )
        return FakeProcess(pid=3334, returncode=0)

    out = run_jobs(jobs, tmp_path, launch=launch)
    assert launched == ["job-next"]
    assert out["paused"] is False
    assert out["completed_task"] == 2
    rec = json.loads((job_dir / "receipt.json").read_text())
    assert rec["status"] == "completed"
    assert rec["reconciled"] is True
    assert rec["cleanup_confirmed"] is True


@pytest.mark.parametrize("has_result,cleanup_stopped", [
    (False, True),
    (True, False),
    (False, False),
])
def test_unknown_running_receipt_pauses_with_no_relaunch(tmp_path, has_result, cleanup_stopped):
    jobs = [_make_job("job-unknown"), _make_job("job-later")]
    job_dir = tmp_path / "job-unknown"
    rollout_dir = job_dir / "rollout"
    save(job_dir / "receipt.json", {
        "job_id": "job-unknown",
        "kind": "model",
        "status": "running",
        "pid": 4444,
        "started_at": "2026-11-01T00:00:00+00:00",
    })
    if has_result:
        save(rollout_dir / "result.json", {"status": "completed", "kind": "native_rpent_model_run"})
    if cleanup_stopped is not False:
        save(rollout_dir / "simulation" / "cleanup.json", {"stopped": True})
    else:
        save(rollout_dir / "simulation" / "cleanup.json", {"stopped": False})

    launched = []
    out = run_jobs(jobs, tmp_path, launch=lambda cmd, **kw: launched.append(cmd))
    assert launched == []
    assert out["paused"] is True
    assert out["pending"] == 1
    assert out["errors"] == 1
    receipt = json.loads((job_dir / "receipt.json").read_text())
    assert receipt["status"] == "running"
    assert receipt["reconciled"] is False

    # Subsequent invocation must still refuse to replay an unreconciled running receipt.
    out_again = run_jobs(jobs, tmp_path, launch=lambda cmd, **kw: launched.append(cmd))
    assert launched == []
    assert out_again["paused"] is True


def test_missing_or_unconfirmed_cleanup_pauses_batch(tmp_path):
    jobs = [_make_job("job-dirty"), _make_job("job-skipped")]
    launched = []

    def launch(cmd, **kwargs):
        rollout_dir = _output_arg(cmd)
        launched.append(rollout_dir.parent.name)
        _populate_rollout(rollout_dir, write_cleanup=False)
        return FakeProcess(pid=5001, returncode=0)

    out = run_jobs(jobs, tmp_path, launch=launch)
    assert launched == ["job-dirty"]
    assert out["paused"] is True
    assert out["completed_task"] == 0
    assert out["errors"] == 1
    assert out["pending"] == 1
    receipt = json.loads((tmp_path / "job-dirty" / "receipt.json").read_text())
    assert receipt["status"] == "error"
    assert receipt["cleanup_confirmed"] is False
    assert receipt["error"] == "cleanup_unconfirmed"


def test_usage_aggregation_includes_hierarchy_and_missing_usage_remains_none(tmp_path):
    jobs = [_make_job("job-partial-usage"), _make_job("job-all-missing")]

    def launch(cmd, **kwargs):
        rollout_dir = _output_arg(cmd)
        name = rollout_dir.parent.name
        if name == "job-partial-usage":
            _populate_rollout(
                rollout_dir,
                hierarchy_usage={"prompt_tokens": 110, "completion_tokens": 15},
                planner_usages=[
                    {"prompt_tokens": 240, "completion_tokens": 35},
                    None,
                ],
            )
            # Non-VLMTrace metadata.json in model_calls must be ignored.
            save(rollout_dir / "model_calls" / "unrelated" / "metadata.json", {"unrelated": True})
        else:
            _populate_rollout(
                rollout_dir,
                hierarchy_usage=None,
                planner_usages=[None],
            )
        return FakeProcess(pid=6001, returncode=0)

    out = run_jobs(jobs, tmp_path, launch=launch)
    rows = {r["job_id"]: r for r in out["results"]}

    partial = rows["job-partial-usage"]
    assert partial["model_api_calls"] == 3
    assert partial["hierarchy_api_calls"] == 1
    assert partial["planner_api_calls"] == 2
    assert partial["missing_usage"] == 1
    assert partial["input_tokens"] == 350
    assert partial["output_tokens"] == 50

    missing = rows["job-all-missing"]
    assert missing["model_api_calls"] == 2
    assert missing["missing_usage"] == 2
    assert missing["input_tokens"] is None
    assert missing["output_tokens"] is None


def test_fixture_is_marked_nonbenchmark_synthetic_and_never_task_success(tmp_path):
    jobs = [_make_job("job-fixture", kind="fixture")]
    seen_cmd = []

    def launch(cmd, **kwargs):
        seen_cmd.extend(cmd)
        rollout_dir = _output_arg(cmd)
        _populate_rollout(
            rollout_dir,
            kind="fixture",
            planner_usages=[{"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}] * 3,
            tools_count=10,
            requests_count=5,
        )
        return FakeProcess(pid=7001, returncode=0)

    out = run_jobs(jobs, tmp_path, launch=launch)
    assert "smoke" in seen_cmd and "--video" not in seen_cmd
    row = out["results"][0]
    assert row["kind"] == "fixture"
    assert row["status"] == "completed"
    assert row["nonbenchmark"] is True
    assert row["task_success"] is None
    assert row["usage_synthetic"] is True
    assert row["usage_mode"] == "synthetic"
    assert row["paid_requests"] == 0
    assert row["model_api_calls"] == 0
    assert row["input_tokens"] is None
    assert row["output_tokens"] is None

    summary = out["summary"]
    assert summary["fixtures"] == 1
    assert summary["completed_fixtures"] == 1
    assert summary["completed_task"] == 0
    assert summary["completed_task_successes"] == 0
    assert summary["task_successes"] == 0
    assert summary["task_success_rate"] is None


def test_timeout_requests_sigint_then_terminates_owned_child_and_pauses(tmp_path):
    jobs = [_make_job("job-timeout", trial_timeout=0.05, cleanup_grace_s=0.05), _make_job("job-after-timeout")]
    procs = []

    def on_wait(proc, timeout):
        if len(proc.wait_calls) <= 2:
            raise subprocess.TimeoutExpired(cmd="piper_cli", timeout=timeout)
        proc.returncode = -15
        return -15

    def launch(cmd, **kwargs):
        proc = FakeProcess(pid=8001, returncode=None, on_wait=on_wait)
        procs.append(proc)
        return proc

    out = run_jobs(jobs, tmp_path, launch=launch)
    assert len(procs) == 1
    proc = procs[0]
    assert proc.signals == [signal.SIGINT]
    assert proc.terminated is True
    assert out["paused"] is True
    assert out["errors"] == 1
    assert out["pending"] == 1
    receipt = json.loads((tmp_path / "job-timeout" / "receipt.json").read_text())
    assert receipt["status"] == "error"
    assert receipt["error"] == "trial_timeout"
    assert receipt["cleanup_confirmed"] is False


def test_unconfirmed_cleanup_still_blocks_next_job_on_resume(tmp_path):
    jobs = [_make_job('dirty'), _make_job('next')]
    save(tmp_path / 'dirty/receipt.json', {'status': 'error', 'error': 'cleanup_unconfirmed'})
    report = run_jobs(jobs, tmp_path, launch=lambda *a, **k: pytest.fail('must not launch'))
    assert report['paused'] and report['results'][1]['status'] == 'pending'


def test_duplicate_batch_owner_never_launches(tmp_path):
    import fcntl
    with (tmp_path / 'batch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match='active owner'):
            run_jobs([_make_job('one')], tmp_path,
                     launch=lambda *a, **k: pytest.fail('must not launch'))


@pytest.mark.parametrize('contents', ['{bad json', '{}', '{"status":"mystery"}'])
def test_damaged_receipt_is_never_replayed(tmp_path, contents):
    (tmp_path / 'one').mkdir()
    (tmp_path / 'one/receipt.json').write_text(contents)
    with pytest.raises(RuntimeError, match='Invalid receipt'):
        run_jobs([_make_job('one')], tmp_path,
                 launch=lambda *a, **k: pytest.fail('must not launch'))
