"""Batch the existing Piper runner; keep video reasoning and primitives in RPent."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from urllib.request import ProxyHandler, build_opener

from .dataset import prepare_jobs
from .io import save
from .piper_cli import ROOT, verify_sources


def load_json(path):
    return json.loads(Path(path).read_text())


def file_identity(path):
    path = Path(path).resolve(strict=True)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path), "sha256": digest.hexdigest()}


def evaluation_spec(path):
    spec = load_json(path)
    if spec.get("scope") == "watchact_imitation_piperx_v1":
        from .watchact_score import validate_spec
        validate_spec(spec)
        return spec
    if spec.get("scope") != "piper_scene_metrics" or not spec.get("predicates"):
        raise ValueError("Evaluator requires explicit piper_scene_metrics predicates; BDDL is not yet a Piper scorer")
    for row in spec["predicates"]:
        if (set(row) != {"path", "equals"} or not isinstance(row["path"], list)
                or not row["path"] or not all(isinstance(k, str) and k for k in row["path"])
                or type(row["equals"]) is not bool):
            raise ValueError("Each scoring predicate needs a nonempty key path and boolean equals")
    return spec


def build_jobs(rows, config, *, bindings=None, relative_to=None):
    """Resolve operator scene/scoring bindings without giving scoring data to RPent."""
    base = Path(relative_to or ".").resolve()
    bindings = bindings or {}
    allowed = set(load_json(ROOT / "piper_simulation.json")) | {"rpent_root", "env_file", "ffmpeg_dir", "schema_in_text"}
    jobs, identifiers = [], set()
    for index, row in enumerate(rows):
        default_id = (f"{row['env_id']}-episode-{row.get('human_episode', index):03d}-seed-{row.get('seed', 0)}"
                      if "env_id" in row else f"episode-{index:04d}")
        ident = row.get("job_id", default_id)
        if (not isinstance(ident, str) or not ident or len(ident) > 160
                or any(not (c.isascii() and (c.isalnum() or c in "_-")) for c in ident)
                or ident in identifiers):
            raise ValueError("Job IDs must be unique ASCII filenames containing letters, digits, '_' or '-'")
        identifiers.add(ident)
        kind = row.get("kind", "model")
        if kind not in {"model", "fixture"}:
            raise ValueError("kind must be model or explicitly nonbenchmark fixture")
        public = {k: v for k, v in config.items() if k in allowed}
        public["seed"] = row.get("seed", config.get("seed", 0))
        binding = dict(bindings.get(row.get("env_id", ident), {}))
        binding.update({k: row[k] for k in ("simulation_asset", "evaluation_file", "arm_count") if k in row})
        problems, identities = [], {}
        if kind == "model":
            if type(binding.get("arm_count")) is not int or binding["arm_count"] != 1:
                problems.append("single-arm scene binding (arm_count=1) missing")
            for name in ("simulation_asset", "evaluation_file"):
                if not binding.get(name):
                    problems.append(name + " missing; no default scene substitution")
                    continue
                try:
                    path = (base / binding[name]).resolve(strict=True)
                    identities[name] = file_identity(path)
                    binding[name] = str(path)
                    if name == "evaluation_file":
                        if evaluation_spec(path)["scope"] == "watchact_imitation_piperx_v1":
                            public["operator_evaluation_trace"] = True
                except (OSError, ValueError) as error:
                    problems.append(f"{name}: {error}")
            if binding.get("simulation_asset"):
                public["simulation_asset"] = binding["simulation_asset"]
        for key in ("language_instruction", "camera_perspective", "spatial_reference"):
            if key in row:
                if not isinstance(row[key], str) or not row[key].strip():
                    raise ValueError(key + " must be nonempty public text")
                public[key] = row[key]
        video = row.get("video")
        if isinstance(video, str):
            video = {"path": video, "start_s": 0.0, "end_s": None}
        if kind == "model":
            if not isinstance(video, dict) or not video.get("path"):
                problems.append("demonstration video missing")
            else:
                video = {k: video.get(k) for k in ("path", "start_s", "end_s")}
                start = 0.0 if video.get("start_s") is None else video["start_s"]
                end = video.get("end_s")
                if (isinstance(start, bool) or not isinstance(start, (int, float)) or not math.isfinite(start)
                        or start < 0 or (end is not None and (isinstance(end, bool)
                            or not isinstance(end, (int, float)) or not math.isfinite(end) or end <= start))):
                    raise ValueError("Invalid recorded video interval")
                video["start_s"] = start
                try:
                    video["path"] = str((base / video["path"]).resolve(strict=True))
                    identities["video"] = file_identity(video["path"])
                except OSError as error:
                    problems.append("video: " + str(error))
        else:
            video = None
        jobs.append({"job_id": ident, "kind": kind, "config": public, "video": video,
                     "evaluation": binding.get("evaluation_file") if kind == "model" else None,
                     "input_identity": identities, "blocked": problems})
    if not jobs:
        raise ValueError("No jobs selected")
    return jobs


def score_metrics(metrics, spec):
    checks = []
    for predicate in spec["predicates"]:
        value = metrics
        for key in predicate["path"]:
            if not isinstance(value, dict) or key not in value:
                raise ValueError("Private metric unavailable: " + ".".join(predicate["path"]))
            value = value[key]
        if type(value) is not bool:
            raise ValueError("Scoring predicate must refer to a measured boolean")
        checks.append({**predicate, "actual": value, "holds": value == predicate["equals"]})
    return {"scope": "piper_scene_metrics", "success": all(r["holds"] for r in checks), "checks": checks}


def assess(jobs, directory, report):
    """Only post-rollout evaluator reads private goals/metrics. No policy feedback."""
    for row in report.get("results", []):
        job = next(j for j in jobs if j["job_id"] == row["job_id"])
        result_path = Path(directory) / job["job_id"] / "rollout/result.json"
        if row.get("status") != "completed" or job["kind"] != "model" or not result_path.exists():
            continue
        result = load_json(result_path)
        try:
            spec = evaluation_spec(job["evaluation"])
            if spec["scope"] == "watchact_imitation_piperx_v1":
                from .watchact_score import score_task
                sim = result_path.parent / "simulation"
                if (sim / "operator_capture_error.json").exists():
                    raise ValueError("Incomplete independent physics evidence; inspect operator_capture_error.json")
                scoring = {"scope": spec["scope"], **score_task(load_json(sim / "operator_initial.json"),
                    load_json(sim / "operator_trace.json"), load_json(sim / "operator_terminal.json"), spec)}
            else:
                scoring = score_metrics(result["operator_scene_metrics"], spec)
            row.update(task_success=scoring["success"], scoring_scope=scoring["scope"])
            save(Path(directory) / job["job_id"] / "score.json", scoring)
        except (OSError, KeyError, ValueError) as error:
            row.update(task_success=None, scoring_error=str(error))
    from .piper_batch import summarize
    summary = summarize(report.get("results", []), paused=report.get("paused", False),
                        pause_reason=report.get("pause_reason"))
    report.update(summary)
    report["summary"] = summary
    save(Path(directory) / "results.json", report.get("results", []))
    report["scoring"] = {"scope": "Piper adapted scenes, not official IG/LIBERO scores",
        "scored": sum(type(r.get("task_success")) is bool for r in report.get("results", [])),
        "successful": sum(r.get("task_success") is True for r in report.get("results", []))}
    save(Path(directory) / "summary.json", report)
    return report


def validate_scenes(jobs):
    """Compile the actual single-arm assets and validate available scoring fields before API use."""
    from piperx_middleware.mujoco_backend import MujocoBackend
    for job in jobs:
        if job["kind"] == "fixture":
            continue
        backend = MujocoBackend(asset_path=Path(job["config"]["simulation_asset"]), seed=job["config"]["seed"])
        try:
            required = ("tcp_site_id", "flange_site_id", "camera_id", "wrist_camera_id",
                        "cube_red_body_id", "cube_blue_body_id", "tray_body_id")
            if any(getattr(backend, name) < 0 for name in required):
                raise ValueError("Scene does not support the existing Piper robot/camera/metrics contract; no default substitution")
            spec = evaluation_spec(job["evaluation"])
            if spec["scope"] == "watchact_imitation_piperx_v1":
                from .watchact_score import contained
                initial = {"success": contained(backend.evaluate(), spec)}
            else:
                initial = score_metrics(backend.evaluate(), spec)
            if initial["success"]:
                raise ValueError("Goal is already satisfied in the configured initial scene")
        finally:
            backend.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/imitator_game.json")
    parser.add_argument("--manifest", type=Path, help="Explicit single-arm tasks; no GT fields are policy inputs")
    parser.add_argument("--scene-map", type=Path, default=ROOT / "configs/piper_scenes.json")
    parser.add_argument("--rpent-root")
    parser.add_argument("--env-file")
    parser.add_argument("--ffmpeg-dir")
    parser.add_argument("--eval-config")
    parser.add_argument("--model")
    parser.add_argument("--reasoning-effort")
    parser.add_argument("--sam3-endpoint")
    parser.add_argument("--sam3-protocol", choices=["rest", "rpent"])
    parser.add_argument("--primitive-image-history", choices=["latest", "budget"])
    parser.add_argument("--output-root", type=Path, default=ROOT / "var/piperx")
    parser.add_argument("--levels", nargs="+", choices=["L0", "L1", "L2", "L3"])
    parser.add_argument("--tasks", nargs="+")
    parser.add_argument("--episode-start", type=int)
    parser.add_argument("--episodes-per-env", type=int)
    parser.add_argument("--max-turns", type=int)
    parser.add_argument("--max-episode-steps", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="Two scripted physics fixtures; excluded from benchmark results")
    args = parser.parse_args(argv)
    config = {**load_json(ROOT / "piper_simulation.json"), **load_json(args.config)}
    for key in ("rpent_root", "env_file", "ffmpeg_dir", "eval_config", "model", "reasoning_effort",
                "sam3_endpoint", "sam3_protocol", "primitive_image_history", "episode_start",
                "episodes_per_env", "max_turns", "max_episode_steps"):
        if getattr(args, key) is not None:
            config[key] = getattr(args, key)
    if args.ffmpeg_dir:
        os.environ["PATH"] = str(Path(args.ffmpeg_dir).resolve()) + os.pathsep + os.environ.get("PATH", "")
    lock = verify_sources(config["rpent_root"])
    relative_to = args.manifest.parent if args.manifest else args.scene_map.parent
    bindings = load_json(args.scene_map) if args.scene_map.is_file() else {}
    if args.smoke:
        rows = [{"job_id": "fixture-01", "kind": "fixture"}, {"job_id": "fixture-02", "kind": "fixture"}]
    elif args.manifest:
        rows = load_json(args.manifest)
        if not isinstance(rows, list):
            parser.error("manifest must be a JSON list")
    else:
        rows = prepare_jobs(config, args.tasks or (), args.levels or ())
    jobs = build_jobs(rows, config, bindings=bindings, relative_to=relative_to)
    source_files = {str(p.relative_to(ROOT)): file_identity(p)["sha256"] for p in sorted((ROOT / "rvicl_ig").rglob("*"))
                    if p.is_file() and p.suffix in {".py", ".txt"}}
    signature = hashlib.sha256(json.dumps({"jobs": jobs, "sources": source_files,
        "rpent": lock["rpent"]["commit"]}, sort_keys=True).encode()).hexdigest()[:16]
    directory = args.output_root.resolve() / ("run-" + signature)
    save(directory / "jobs.json", jobs)
    save(directory / "source_manifest.json", source_files)
    blocked = [{"job_id": j["job_id"], "reasons": j["blocked"]} for j in jobs if j["blocked"]]
    print(json.dumps({"jobs": len(jobs), "directory": str(directory), "blocked": blocked}, ensure_ascii=False), flush=True)
    if args.dry_run:
        return 0
    if blocked:
        save(directory / "preflight.json", {"status": "blocked", "jobs": blocked})
        print("No model request or simulation started: bind single-arm scenes and independent goals first.", flush=True)
        return 2
    try:
        validate_scenes(jobs)
    except Exception as error:
        save(directory / "preflight.json", {"status": "blocked", "reason": "Scene/scorer validation failed", "error": str(error)})
        return 2
    if not args.smoke and config.get("sam3_enabled"):
        try:
            opener = build_opener(ProxyHandler({}))
            if config.get("sam3_protocol") != "rest":
                raise ValueError("This batch preflight supports the deployed REST SAM service only")
            with opener.open(config["sam3_endpoint"].rstrip("/") + "/openapi.json", timeout=5) as response:
                routes = json.load(response).get("paths", {})
                if not all("post" in routes.get(route, {}) for route in ("/segment", "/segment_point")):
                    raise ValueError("SAM3 REST segmentation routes unavailable")
        except Exception as error:
            save(directory / "preflight.json", {"status": "blocked", "reason": "SAM3 unavailable", "error": str(error)})
            return 2
    from .piper_batch import run_jobs
    report = assess(jobs, directory, run_jobs(jobs, directory))
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return int(any(r.get("status") != "completed" or r.get("scoring_error") for r in report.get("results", [])))


if __name__ == "__main__":
    raise SystemExit(main())
