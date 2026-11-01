"""CLI: official split -> cached video hierarchy -> isolated rollout -> native scores."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path

from .dataset import ground_truth, prepare_jobs, source_identity
from .io import Client, save
from .metrics import summarize, trajectory_scores

ROOT = Path(__file__).resolve().parents[1]


def run_episode(job):
    directory = Path(job["trial_dir"])
    row = {key: job[key] for key in ("env_id", "level", "human_id", "human_episode", "seed", "reset_index", "model")}
    row.update(status="error", scored=False, method="rvicl-vlm-human-video-native-ig",
               executor="rpent-dual-arm-primitives-ig-panda", learned_vla=False,
               signature=job["signature"], scoring_endpoint="official_horizon",
               max_episode_steps=job["max_episode_steps"], max_turns=job["max_turns"],
               seed_protocol="author-single-cpu-vector-seed2022")
    row.update(sam3_enabled=job.get("sam3_enabled", True),
               sam3_endpoint=job.get("sam3_endpoint"), sam3_protocol=job.get("sam3_protocol", "rest"))
    row["primitive_image_history"] = job.get("primitive_image_history", "budget")
    scene = None
    try:
        sys.path.insert(0, job["game_root"])
        from .simulator import Scene
        if job.get("validate_only"):
            scene = Scene(job, directory / "simulator")
            observation = scene.observe()
            if set(observation["cameras"]) != {"head", "left_wrist", "right_wrist"}:
                raise ValueError("Missing live camera")
            if job.get("validate_sam3"):
                sam_step = scene.steps
                with scene.record_tool("segment", {"prompt": job.get("sam3_validation_prompt", "object"), "camera": "head"}, source="validation") as record:
                    text = scene.perception.segment(**record["arguments"])
                    record["result"] = text
                with scene.record_tool("segment", {"camera": "head", "point": [
                        observation["cameras"]["head"]["height"]//2, observation["cameras"]["head"]["width"]//2]}, source="validation") as record:
                    point = scene.perception.segment(**record["arguments"])
                    record["result"] = point
                if scene.steps != sam_step or not text.get("found") or not point.get("found"):
                    raise ValueError(f"SAM text/point inference validation failed: {text}, {point}")
                if text.get("world_xyz") is None or point.get("world_xyz") is None:
                    raise ValueError("SAM masks must project to valid measured world coordinates")
                row["sam3_validation"] = {"text": text, "point": point, "read_only": True}
            waypoints = {side: list(arm["tcp_pose_world"]) for side, arm in scene.state()["arms"].items()}
            for pose in waypoints.values():
                pose[2] += .005
            from .rpent_backend import create_primitives
            primitives = create_primitives(scene, job["rpent_root"])
            def execute(name, **arguments):
                with scene.record_tool(name, arguments, source="validation") as record:
                    result = getattr(primitives, name)(**arguments)
                    record["result"] = result
                    return result
            primitive_results = {}
            for side, pose in waypoints.items():
                primitive_results["move_to_"+side] = execute("move_to",
                    arm=side, xyz=pose[:3], quat=pose[3:], substeps=0)
            primitive_results["move_both"] = execute("move_both", **waypoints)
            primitive_results["rotate_wrist"] = execute("rotate_wrist", arm="left", delta_yaw_deg=.5, substeps=0)
            primitive_results["set_gripper"] = execute("set_gripper", arm="left", val=1., steps=3)
            primitive_results["release"] = execute("release", arm="right", steps=3)
            if any(not result.get("completed") for result in primitive_results.values()):
                raise ValueError(f"RPent primitive did not complete: {primitive_results}")
            measured = scene.state()
            errors = {side: float(sum((measured["arms"][side]["tcp_pose_world"][k]-pose[k])**2
                                     for k in range(3))**.5) for side, pose in waypoints.items()}
            if max(errors.values()) > .02:
                raise ValueError(f"Native TCP motion validation failed: {errors}")
            result = scene.finish()
            row.update(result, status="validation_complete", tcp_position_errors_m=errors,
                       camera_names=list(observation["cameras"]), rpent_validation=primitive_results)
        else:
            from .agent import solve
            from .hierarchy import build
            from .video import EpisodeVideo
            client = Client(job)
            # Cache is shared across task difficulty levels for the same human episode.
            video_key = hashlib.sha256(json.dumps(job["video"], sort_keys=True).encode()).hexdigest()[:20]
            cache = Path(job["output_root"]) / "video_cache" / job["signature"] / video_key
            video = EpisodeVideo(job["video"], cache)
            hierarchy = build(video, job, client, cache)
            row["hierarchy"] = str(cache / "hierarchy.json")
            save(directory / "demonstration.json", {"hierarchy": str(cache / "hierarchy.json"),
                "vlm_call": hierarchy.data.get("vlm_trace", str(cache / "hierarchy_response")),
                "reused_cached_hierarchy": client.calls == 0})
            if job.get("prepare_only"):
                row.update(status="prepared", vlm_calls=client.calls)
            else:
                # The robot reference trajectory is used only after policy rollout, for scoring.
                scene = Scene(job, directory / "simulator")
                row.update(solve(client, scene, hierarchy, job, directory))
                row.update(scene.finish())
                row.update(status="complete", vlm_calls=client.calls, tss_available=False)
                if job["compute_tss"]:
                    try:
                        reference = ground_truth(job, directory)
                        row.update(trajectory_scores(scene.actions, reference, scene.first_success), tss_available=True)
                        row["tss_protocol"] = "author-raw16d-no-reordering-band015"
                        row["tss_prediction_order"] = [f"{key}:7joints+gripper" for key in scene.keys]
                    except (ValueError, FileNotFoundError) as error:
                        # Preserve valid native SR even when optional GT is absent.
                        row["tss_unavailable_reason"] = str(error)
                else:
                    row["tss_unavailable_reason"] = "Disabled by --skip-tss"
                row["clip_views"] = hierarchy.views
                row["sam3_calls"] = scene.perception.calls if scene.perception else 0
    except Exception:
        row.update(status="error", error=traceback.format_exc())
    finally:
        if scene is not None:
            try:
                scene.close()
            except Exception:
                row.update(status="error", cleanup_error=traceback.format_exc())
            if getattr(scene, "recorder", None):
                row["execution_recording"] = str(scene.recorder.directory / "videos/manifest.json")
                row["tool_recordings"] = str(scene.recorder.directory / "tool_calls/index.json")
        save(directory / "result.json", row)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/imitator_game.json")
    parser.add_argument("--game-root")
    parser.add_argument("--rpent-root")
    parser.add_argument("--demo-episode-offset", type=int)
    parser.add_argument("--demo-per-episode", action="store_true", help="Use a different human episode per reset offset")
    parser.add_argument("--eval-config", help="Environment list (.txt); absolute path or relative to --game-root")
    parser.add_argument("--human-root")
    parser.add_argument("--human-config")
    parser.add_argument("--sim-root")
    parser.add_argument("--sim-config")
    parser.add_argument("--output-root", help="Output folder; saves run-<signature>/ and shared video_cache/ here")
    parser.add_argument("--model")
    parser.add_argument("--env-file", type=Path, help="Read API credentials without executing shell code")
    parser.add_argument("--ffmpeg-dir", type=Path, help="Directory containing ffmpeg and ffprobe")
    parser.add_argument("--reasoning-effort")
    parser.add_argument("--primitive-image-history", choices=["budget", "latest"],
                        help="budget: native image-byte budget; latest: only the latest robot camera batch and current tool image, preserving demonstration images and text")
    parser.add_argument("--sam3-endpoint")
    parser.add_argument("--sam3-protocol", choices=["rest", "rpent"])
    parser.add_argument("--no-sam3", action="store_true")
    parser.add_argument("--validate-sam3", action="store_true", help="Exercise real text and point SAM inference during --validate-only")
    parser.add_argument("--sam3-validation-prompt", default="object", help="Visible object category for the optional live SAM validation")
    parser.add_argument("--levels", nargs="+", choices=["L0", "L1", "L2", "L3"],
                        help="Select dataset difficulty levels from --eval-config")
    parser.add_argument("--tasks", nargs="+", help="Select exact environment IDs from --eval-config")
    parser.add_argument("--episodes-per-env", type=int, help="Number of reset episodes per selected environment")
    parser.add_argument("--episode-start", type=int, help="First reset episode index (zero-based); demo stays fixed by default")
    for flag in ("max-turns", "max-episode-steps", "max-candidates", "max-keyframes"):
        parser.add_argument("--" + flag, type=int)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true", help="Save jobs.json and print selected data without VLM or simulation")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--skip-tss", action="store_true")
    parser.add_argument("--retry-errors", action="store_true")
    parser.add_argument("--report-only", type=Path)
    parser.add_argument("--worker-job", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker_job:
        row = run_episode(json.loads(args.worker_job.read_text()))
        return int(row["status"] == "error")
    if args.report_only:
        jobs = json.loads((args.report_only / "jobs.json").read_text())
        return report(jobs, args.report_only)
    config = json.loads(args.config.read_text())
    config.setdefault("primitive_image_history", "budget")
    for key in list(config):
        value = getattr(args, key, None)
        if value is not None:
            config[key] = value
    config.update(validate_only=args.validate_only, prepare_only=args.prepare_only)
    if config["primitive_image_history"] not in {"budget", "latest"}:
        parser.error("primitive_image_history must be budget or latest")
    config["validate_sam3"] = args.validate_sam3
    if args.validate_sam3:
        config["sam3_validation_prompt"] = args.sam3_validation_prompt
    if args.no_sam3:
        config["sam3_enabled"] = False
    if args.validate_sam3 and (not args.validate_only or args.no_sam3):
        parser.error("--validate-sam3 requires --validate-only with SAM3 enabled")
    if args.demo_per_episode:
        config["demo_episode_offset"] = None
    if args.env_file:
        config["env_file"] = str(args.env_file.resolve())
    if args.ffmpeg_dir:
        config["ffmpeg_dir"] = str(args.ffmpeg_dir.resolve())
        os.environ["PATH"] = config["ffmpeg_dir"] + os.pathsep + os.environ.get("PATH", "")
    if args.validate_only and args.prepare_only:
        parser.error("--validate-only and --prepare-only are mutually exclusive")
    if args.limit < 0:
        parser.error("--limit must be nonnegative")
    if args.no_video:
        config["save_video"] = False
    if args.skip_tss:
        config["compute_tss"] = False
    config["output_root"] = str(Path(config["output_root"]).resolve())
    jobs = prepare_jobs(config, args.tasks or (), args.levels or (), args.limit)
    signature = source_identity(config)
    identity = hashlib.sha256(json.dumps([j["video"] for j in jobs], sort_keys=True).encode()).hexdigest()[:10]
    run_dir = Path(config["output_root"]) / f"run-{signature}-{identity}"
    for job in jobs:
        job.update(signature=signature, trial_dir=str(run_dir / job["env_id"] /
            f"episode-{job['human_episode']:03d}-seed-{job['seed']}"))
    save(run_dir / "jobs.json", jobs)
    print(f"Resolved {len(jobs)} official evaluation jobs; artifacts: {run_dir}", flush=True)
    if args.dry_run:
        print(json.dumps([{k: j[k] for k in ("env_id", "level", "reset_index", "human_id", "human_episode", "seed")}
                          for j in jobs], indent=2))
        return 0
    for number, job in enumerate(jobs, 1):
        directory = Path(job["trial_dir"])
        directory.mkdir(parents=True, exist_ok=True)
        result_path = directory / "result.json"
        if result_path.exists():
            previous = json.loads(result_path.read_text())
            if previous.get("signature") == signature and (previous["status"] != "error" or not args.retry_errors):
                print(f"[{number}/{len(jobs)}] Resume: {previous['status']} {job['env_id']}", flush=True)
                continue
        save(directory / "job.json", job)
        print(f"[{number}/{len(jobs)}] {job['env_id']} human episode {job['human_episode']}; log: {directory / 'worker.log'}", flush=True)
        try:
            with (directory / "worker.log").open("w") as log:
                process = subprocess.run([sys.executable, "-u", str(ROOT / "scripts/eval_imitator_game.py"),
                    "--worker-job", str(directory / "job.json")], stdout=log, stderr=subprocess.STDOUT,
                    timeout=config["trial_timeout"])
            if process.returncode and not result_path.exists():
                raise RuntimeError(f"Worker exited {process.returncode}; inspect worker.log")
        except (subprocess.TimeoutExpired, RuntimeError) as error:
            save(result_path, {"env_id": job["env_id"], "level": job["level"], "signature": signature,
                "status": "error", "scored": False, "error": str(error)})
        row = json.loads(result_path.read_text())
        print(f"  {row['status']}; SR={row.get('SR')}; TSS available={row.get('tss_available')}", flush=True)
        report(jobs, run_dir, quiet=True)
    return report(jobs, run_dir)


def report(jobs, directory, quiet=False):
    rows = []
    for job in jobs:
        path = Path(job["trial_dir"]) / "result.json"
        rows.append(json.loads(path.read_text()) if path.exists() else {
            "env_id": job["env_id"], "level": job["level"], "status": "pending"})
    summary = summarize(rows)
    save(Path(directory) / "results.json", rows)
    save(Path(directory) / "summary.json", summary)
    if not quiet:
        print(json.dumps(summary["overall"], indent=2))
    return int(any(r["status"] in {"error", "pending"} for r in rows))
