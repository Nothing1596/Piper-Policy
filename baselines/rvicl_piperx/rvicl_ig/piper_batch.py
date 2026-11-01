"""Narrow batch-worker lifecycle manager for the single-arm RPent/PiperX simulation adapter."""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from .io import save

TERMINAL_RECEIPT_STATUSES = {"completed", "error", "interrupted"}


def _now_iso():
    return datetime.now(timezone.utc).isoformat()


def _read_json(path):
    try:
        if Path(path).is_file():
            return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return None
    return None


def _sim_root(job_dir, rollout_dir):
    rollout_sim = rollout_dir / "simulation"
    if rollout_sim.exists() or not (job_dir / "simulation").exists():
        return rollout_sim
    return job_dir / "simulation"


def _trace_root(job_dir, rollout_dir):
    if (rollout_dir / "hierarchy").exists() or (rollout_dir / "model_calls").exists():
        return rollout_dir
    if (job_dir / "hierarchy").exists() or (job_dir / "model_calls").exists():
        return job_dir
    return rollout_dir


def _is_vlm_trace_metadata(meta):
    return (
        isinstance(meta, dict)
        and isinstance(meta.get("status"), str)
        and bool(meta["status"])
        and isinstance(meta.get("model"), str)
        and bool(meta["model"])
        and isinstance(meta.get("started_at"), str)
        and bool(meta["started_at"])
    )


def _extract_tokens(usage):
    if not isinstance(usage, dict):
        return None, None
    in_tok = usage.get("input_tokens")
    if in_tok is None:
        in_tok = usage.get("prompt_tokens")
    out_tok = usage.get("output_tokens")
    if out_tok is None:
        out_tok = usage.get("completion_tokens")
    if type(in_tok) is int and type(out_tok) is int and in_tok >= 0 and out_tok >= 0:
        return in_tok, out_tok
    return None, None


def collect_usage(job_dir, rollout_dir, kind):
    base = _trace_root(job_dir, rollout_dir)
    hierarchy_traces, planner_traces = [], []
    for folder, target in (("hierarchy", hierarchy_traces), ("model_calls", planner_traces)):
        root = base / folder
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("metadata.json")):
            meta = _read_json(path)
            if _is_vlm_trace_metadata(meta):
                target.append(meta)
    all_traces = hierarchy_traces + planner_traces
    if kind == "fixture":
        return {
            "input_tokens": None,
            "output_tokens": None,
            "model_api_calls": 0,
            "hierarchy_api_calls": 0,
            "planner_api_calls": 0,
            "synthetic_api_calls": len(all_traces),
            "missing_usage": 0,
            "usage_synthetic": True,
            "usage_mode": "synthetic",
            "paid_requests": 0,
        }
    in_sum, out_sum, have_tokens, missing = 0, 0, False, 0
    for meta in all_traces:
        in_tok, out_tok = _extract_tokens(meta.get("usage"))
        if in_tok is None or out_tok is None:
            missing += 1
        else:
            in_sum += in_tok
            out_sum += out_tok
            have_tokens = True
    total_calls = len(all_traces)
    return {
        "input_tokens": in_sum if have_tokens else None,
        "output_tokens": out_sum if have_tokens else None,
        "model_api_calls": total_calls,
        "hierarchy_api_calls": len(hierarchy_traces),
        "planner_api_calls": len(planner_traces),
        "synthetic_api_calls": 0,
        "missing_usage": missing,
        "usage_synthetic": False,
        "usage_mode": "actual",
        "paid_requests": total_calls,
    }


def _collect_sim_counts(job_dir, rollout_dir):
    sim_dir = _sim_root(job_dir, rollout_dir)
    tools = _read_json(sim_dir / "tools.json")
    tool_attempts = len(tools) if isinstance(tools, list) else 0
    req_dir = sim_dir / "requests"
    req_count = len(list(req_dir.glob("*.json"))) if req_dir.is_dir() else 0
    cleanup = _read_json(sim_dir / "cleanup.json")
    cleanup_ok = isinstance(cleanup, dict) and cleanup.get("stopped") is True
    return tool_attempts, req_count, cleanup, cleanup_ok


def _pending_row(job):
    kind = job["kind"]
    is_fixture = kind == "fixture"
    return {
        "job_id": job["job_id"],
        "kind": kind,
        "status": "pending",
        "nonbenchmark": is_fixture,
        "model_declared_status": None,
        "task_success": None,
        "input_tokens": None,
        "output_tokens": None,
        "model_api_calls": 0,
        "hierarchy_api_calls": 0,
        "planner_api_calls": 0,
        "synthetic_api_calls": 0,
        "missing_usage": 0,
        "usage_synthetic": is_fixture,
        "usage_mode": "synthetic" if is_fixture else "actual",
        "paid_requests": 0,
        "model_tool_attempts": 0,
        "executor_request_count": 0,
        "elapsed_s": None,
        "elapsed_wall_time_s": None,
        "cleanup_confirmed": False,
        "child_result": None,
        "operator_scene_metrics": None,
        "reconciled": False,
        "error": None,
    }


def _build_row(job, job_dir, rollout_dir, status, elapsed_s, *, reconciled=False, error=None, prior_row=None):
    kind = job["kind"]
    is_fixture = kind == "fixture"
    result_path = rollout_dir / "result.json"
    child_result = _read_json(result_path)
    if not isinstance(child_result, dict):
        child_result = None
    tool_attempts, req_count, _, cleanup_ok = _collect_sim_counts(job_dir, rollout_dir)
    usage = collect_usage(job_dir, rollout_dir, kind)
    declared = child_result.get("model_declared_status") if (child_result and not is_fixture) else None
    task_success = None
    if not is_fixture and isinstance(prior_row, dict) and isinstance(prior_row.get("task_success"), bool):
        task_success = prior_row["task_success"]
    return {
        "job_id": job["job_id"],
        "kind": kind,
        "status": status,
        "nonbenchmark": is_fixture,
        "model_declared_status": declared,
        "task_success": None if is_fixture else task_success,
        **usage,
        "model_tool_attempts": tool_attempts,
        "executor_request_count": req_count,
        "elapsed_s": elapsed_s,
        "elapsed_wall_time_s": elapsed_s,
        "cleanup_confirmed": cleanup_ok,
        "child_result": str(result_path) if result_path.is_file() else None,
        "operator_scene_metrics": child_result.get("operator_scene_metrics") if child_result else None,
        "reconciled": bool(reconciled),
        "error": error,
    }


def summarize(rows, *, paused=False, pause_reason=None):
    model_rows = [r for r in rows if r.get("kind") == "model"]
    fixture_rows = [r for r in rows if r.get("kind") == "fixture"]
    completed_models = [r for r in model_rows if r.get("status") == "completed"]
    completed_fixtures = [r for r in fixture_rows if r.get("status") == "completed"]
    scored_models = [r for r in completed_models if isinstance(r.get("task_success"), bool)]
    task_successes = sum(1 for r in scored_models if r.get("task_success") is True)
    errors = sum(1 for r in rows if r.get("status") in {"error", "interrupted"})
    interrupted = sum(1 for r in rows if r.get("status") == "interrupted")
    pending = sum(1 for r in rows if r.get("status") == "pending")
    in_vals = [r["input_tokens"] for r in model_rows if type(r.get("input_tokens")) is int]
    out_vals = [r["output_tokens"] for r in model_rows if type(r.get("output_tokens")) is int]
    return {
        "total_jobs": len(rows),
        "completed": len(completed_models) + len(completed_fixtures),
        "completed_task": len(completed_models),
        "completed_tasks": len(completed_models),
        "scored_tasks": len(scored_models),
        "completed_task_successes": task_successes,
        "task_successes": task_successes,
        "task_success_rate": (task_successes / len(scored_models)) if scored_models else None,
        "errors": errors,
        "interrupted": interrupted,
        "pending": pending,
        "fixtures": len(fixture_rows),
        "completed_fixtures": len(completed_fixtures),
        "input_tokens": sum(in_vals) if in_vals else None,
        "output_tokens": sum(out_vals) if out_vals else None,
        "model_api_calls": sum(int(r.get("model_api_calls") or 0) for r in model_rows),
        "missing_usage": sum(int(r.get("missing_usage") or 0) for r in model_rows),
        "paused": bool(paused),
        "pause_reason": pause_reason,
    }


def _build_command(python, job, child_config, config_path, rollout_dir):
    kind = job["kind"]
    if kind not in {"model", "fixture"}:
        raise ValueError(f"Unsupported job kind: {kind}")
    cmd = [
        str(python),
        "-u",
        "-m",
        "rvicl_ig.piper_cli",
        "run" if kind == "model" else "smoke",
        "--rpent-root",
        str(child_config["rpent_root"]),
        "--config",
        str(config_path),
        "--output",
        str(rollout_dir),
    ]
    if kind == "model":
        video = job.get("video") or {}
        cmd.extend(["--video", str(video["path"])])
        if video.get("start_s") is not None:
            cmd.extend(["--start-s", str(video["start_s"])])
        if video.get("end_s") is not None:
            cmd.extend(["--end-s", str(video["end_s"])])
        if child_config.get("env_file"):
            cmd.extend(["--env-file", str(child_config["env_file"])])
    return cmd


def _graceful_stop_owned_child(proc, grace_s):
    try:
        if hasattr(proc, "send_signal"):
            proc.send_signal(signal.SIGINT)
        elif hasattr(proc, "terminate"):
            proc.terminate()
    except OSError:
        return
    try:
        proc.wait(timeout=grace_s)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        if hasattr(proc, "terminate"):
            proc.terminate()
    except OSError:
        return
    try:
        proc.wait(timeout=grace_s)
    except subprocess.TimeoutExpired:
        if hasattr(proc, "kill"):
            try:
                proc.kill()
                proc.wait(timeout=grace_s)
            except Exception:
                pass


def run_jobs(jobs, run_dir: Path, *, python=sys.executable, launch=None) -> dict:
    """One POSIX batch owner; the OS releases the lock even after a crash."""
    import fcntl
    run_dir = Path(run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "batch.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("This batch already has an active owner; no worker started") from error
        return _run_jobs(jobs, run_dir, python=python, launch=launch)


def _run_jobs(jobs, run_dir: Path, *, python=sys.executable, launch=None) -> dict:
    run_dir = Path(run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    jobs = list(jobs)
    launcher = subprocess.Popen if launch is None else launch

    prior_results = _read_json(run_dir / "results.json")
    prior_by_id = (
        {r["job_id"]: r for r in prior_results if isinstance(r, dict) and "job_id" in r}
        if isinstance(prior_results, list)
        else {}
    )
    rows_by_id = {}
    paused = False
    pause_reason = None

    def flush_checkpoint():
        ordered = [rows_by_id.get(job["job_id"], _pending_row(job)) for job in jobs]
        summary = summarize(ordered, paused=paused, pause_reason=pause_reason)
        save(run_dir / "results.json", ordered)
        save(run_dir / "summary.json", summary)
        return ordered, summary

    for job in jobs:
        job_id = job["job_id"]
        kind = job["kind"]
        job_dir = run_dir / job_id
        rollout_dir = job_dir / "rollout"
        receipt_path = job_dir / "receipt.json"
        row_path = job_dir / "row.json"
        saved_row = _read_json(row_path)
        prior_row = dict(saved_row) if isinstance(saved_row, dict) else {}
        if isinstance(prior_by_id.get(job_id), dict):
            prior_row.update({k: v for k, v in prior_by_id[job_id].items() if v is not None})

        existing_receipt = _read_json(receipt_path)
        if (receipt_path.exists() and (not isinstance(existing_receipt, dict)
                or existing_receipt.get("status") not in TERMINAL_RECEIPT_STATUSES | {"running"})):
            raise RuntimeError(f"Invalid receipt for {job_id}; unknown outcome must not replay")
        if existing_receipt is None and job_dir.exists() and any(job_dir.iterdir()):
            raise RuntimeError(f"Orphaned artifacts for {job_id}; reconcile them before starting a worker")
        if isinstance(existing_receipt, dict):
            r_status = existing_receipt.get("status")
            if r_status in TERMINAL_RECEIPT_STATUSES:
                elapsed = existing_receipt.get("elapsed_s")
                if elapsed is None and isinstance(prior_row, dict):
                    elapsed = prior_row.get("elapsed_s")
                row = _build_row(
                    job,
                    job_dir,
                    rollout_dir,
                    r_status,
                    elapsed,
                    reconciled=existing_receipt.get("reconciled", False),
                    error=existing_receipt.get("error"),
                    prior_row=prior_row,
                )
                save(row_path, row)
                rows_by_id[job_id] = row
                if not row["cleanup_confirmed"]:
                    paused = True
                    pause_reason = f"{job_id}: cleanup remains unconfirmed; no next worker started"
                    flush_checkpoint()
                    break
                flush_checkpoint()
                continue

            if r_status == "running":
                child_result = _read_json(rollout_dir / "result.json")
                _, _, cleanup, cleanup_ok = _collect_sim_counts(job_dir, rollout_dir)
                result_ok = isinstance(child_result, dict) and child_result.get("status") == "completed"
                if result_ok and cleanup_ok:
                    elapsed = existing_receipt.get("elapsed_s")
                    updated_receipt = {
                        **existing_receipt,
                        "status": "completed",
                        "reconciled": True,
                        "cleanup_confirmed": True,
                        "cleanup": cleanup,
                        "finished_at": _now_iso(),
                        "error": None,
                    }
                    save(receipt_path, updated_receipt)
                    row = _build_row(
                        job,
                        job_dir,
                        rollout_dir,
                        "completed",
                        elapsed,
                        reconciled=True,
                        prior_row=prior_row,
                    )
                    save(row_path, row)
                    rows_by_id[job_id] = row
                    flush_checkpoint()
                    continue

                paused = True
                pause_reason = f"Unreconciled running receipt for {job_id}: unknown outcome must not replay"
                unreconciled_receipt = {
                    **existing_receipt,
                    "status": "running",
                    "reconciled": False,
                    "cleanup_confirmed": cleanup_ok,
                    "cleanup": cleanup,
                    "paused": True,
                    "error": "unreconciled_running_receipt",
                }
                save(receipt_path, unreconciled_receipt)
                row = _build_row(
                    job,
                    job_dir,
                    rollout_dir,
                    "error",
                    existing_receipt.get("elapsed_s"),
                    reconciled=False,
                    error="unreconciled_running_receipt",
                    prior_row=prior_row,
                )
                row["receipt_status"] = "running"
                row["paused"] = True
                save(row_path, row)
                rows_by_id[job_id] = row
                flush_checkpoint()
                break

        job_dir.mkdir(parents=True, exist_ok=True)
        child_config = {k: v for k, v in dict(job["config"]).items() if k != "evaluation"}
        config_path = job_dir / "config.json"
        save(config_path, child_config)

        timeout_s = float(child_config.get("trial_timeout", 7200))
        grace_s = float(child_config.get("cleanup_grace_s", 5.0))
        cmd = _build_command(python, job, child_config, config_path, rollout_dir)

        started_iso = _now_iso()
        started_mono = time.monotonic()
        receipt = {
            "job_id": job_id,
            "kind": kind,
            "status": "running",
            "pid": None,
            "started_at": started_iso,
            "reconciled": False,
        }
        save(receipt_path, receipt)

        returncode = None
        timed_out = False
        interrupted_flag = False
        launch_error = None

        try:
            with (job_dir / "worker.log").open("a", encoding="utf-8") as log_stream:
                proc = launcher(cmd, stdout=log_stream, stderr=subprocess.STDOUT,
                                start_new_session=(os.name == "posix"))
                pid = getattr(proc, "pid", None)
                if pid is not None:
                    receipt["pid"] = pid
                    save(receipt_path, receipt)
                try:
                    rc = proc.wait(timeout=timeout_s)
                    returncode = rc if rc is not None else getattr(proc, "returncode", 0)
                except subprocess.TimeoutExpired:
                    timed_out = True
                    _graceful_stop_owned_child(proc, grace_s)
                    returncode = getattr(proc, "returncode", None)
                except KeyboardInterrupt:
                    interrupted_flag = True
                    _graceful_stop_owned_child(proc, grace_s)
                    returncode = getattr(proc, "returncode", None)
        except KeyboardInterrupt:
            interrupted_flag = True
        except Exception as exc:
            launch_error = f"{type(exc).__name__}: {exc}"

        elapsed_s = round(time.monotonic() - started_mono, 6)
        child_result = _read_json(rollout_dir / "result.json")
        result_ok = isinstance(child_result, dict) and child_result.get("status") == "completed"
        _, _, cleanup, cleanup_ok = _collect_sim_counts(job_dir, rollout_dir)

        if interrupted_flag:
            final_status = "interrupted"
            error_msg = "interrupted"
        elif launch_error is not None:
            final_status = "error"
            error_msg = launch_error
        elif timed_out:
            final_status = "error"
            error_msg = "trial_timeout"
        elif returncode != 0 or not result_ok:
            final_status = "error"
            child_err = _read_json(rollout_dir / "error.json")
            if isinstance(child_err, dict) and child_err.get("message"):
                error_msg = f"{child_err.get('error_type', 'ChildError')}: {child_err['message']}"
            else:
                error_msg = f"child_exit_{returncode}" if returncode != 0 else "missing_or_incomplete_result"
        elif not cleanup_ok:
            final_status = "error"
            error_msg = "cleanup_unconfirmed"
        else:
            final_status = "completed"
            error_msg = None

        receipt.update(
            status=final_status,
            returncode=returncode,
            elapsed_s=elapsed_s,
            finished_at=_now_iso(),
            cleanup_confirmed=cleanup_ok,
            cleanup=cleanup,
            error=error_msg,
        )
        save(receipt_path, receipt)

        row = _build_row(
            job,
            job_dir,
            rollout_dir,
            final_status,
            elapsed_s,
            reconciled=False,
            error=error_msg,
            prior_row=prior_row,
        )
        save(row_path, row)
        rows_by_id[job_id] = row

        if final_status != "completed":
            paused = True
            pause_reason = f"{job_id}: {error_msg}"
            flush_checkpoint()
            break

        flush_checkpoint()

    ordered_rows, summary = flush_checkpoint()
    return {
        **summary,
        "run_dir": str(run_dir),
        "results": ordered_rows,
        "summary": summary,
    }
