"""Run the frozen native planner with a Piper simulation, or an explicit fixture smoke."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

from .hierarchy import build
from .io import Client, save
from .piper_prompt import piper_prompt
from .piper_scene import PiperScene
from .planner import solve_native
from .video import EpisodeVideo

ROOT = Path(__file__).resolve().parents[1]


class SchemaTextClient(Client):
    """Expose the existing format contract to gateways that ignore response_format.

    Opt-in transport compatibility: same schema, one request, no extra inference
    stage or response repair. Native RPent execution uses its original provider.
    """
    def complete(self, messages, *, schema=None, **kwargs):
        if schema is not None:
            messages = [dict(m) for m in messages]
            messages[0]["content"] += ("\n\nOUTPUT FORMAT\nReturn exactly the following JSON schema, "
                "using its property names, not the conceptual level labels L0/L1/L2. "
                "Do not include Markdown or any other fields.\n" + json.dumps(schema))
        return super().complete(messages, schema=schema, **kwargs)


def verify_sources(rpent_root):
    lock = json.loads((ROOT / "BASELINE.lock.json").read_text())
    for rel in ("hierarchy.py", "video.py", "prompts/hierarchy.txt", "prompts/agent.txt",
                "io.py", "recording.py", "rpent_backend.py"):
        expected = lock["rvicl"]["source_files"]["rvicl_ig/" + rel]
        if hashlib.sha256((ROOT / "rvicl_ig" / rel).read_bytes()).hexdigest() != expected:
            raise ValueError("Frozen RV-ICL source changed: " + rel)
    for rel, expected in lock["rpent"]["files"].items():
        path = Path(rpent_root) / rel
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("Frozen RPent source changed: " + rel)
    return lock


def source(path, start=0., end=None):
    import subprocess
    from .video import binary
    path = Path(path).resolve(strict=True)
    if end is None:
        result = subprocess.run([binary("ffprobe"), "-v", "error", "-show_entries", "format=duration",
                                 "-of", "json", str(path)], capture_output=True, text=True, check=True)
        end = float(json.loads(result.stdout)["format"]["duration"])
    return {"path": str(path), "start_s": start, "end_s": end}


def smoke(scene, job, directory):
    """Scripted provider + REAL physics. Never reported as model/task success."""
    import subprocess
    import httpx2 as httpx
    from .video import binary
    observation = scene.observe()
    before = scene.state()
    # A deliberately labeled simulated fixture, not a human demonstration.
    frame = observation["_images"][0]["path"]
    video_path = directory / "synthetic-fixture.mp4"
    subprocess.run([binary("ffmpeg"), "-nostdin", "-v", "error", "-loop", "1", "-i", frame,
        "-t", "2", "-r", "4", "-pix_fmt", "yuv420p", "-threads", "1", str(video_path)], check=True)
    video = EpisodeVideo(source(video_path), directory / "demo_frames")
    candidates = video.candidates(job["candidate_fps"], job["max_candidates"])
    last = len(candidates)-1
    evidence = {"evidence": "SCRIPTED INTERFACE FIXTURE; simulated image", "uncertainty": "Not human/model evaluation"}
    raw = {"summary": "Protocol fixture: inspect, move a few mm in free space, yaw, open gripper, settle.",
        "keyframes": [{"index": index, "stage": "fixture", "left": "single arm", "right": "none",
                       "result": "not a task score", **evidence} for index in (0, last)],
        "phases": [{"id": 1, "label": "fixture", "start_index": 0, "end_index": last,
                    "left": "small free-space motion", "right": "none", "result": "interface check", **evidence}],
        "moments": [{"name": "fixture end", "kind": "end", "index": last, "phase": 1, **evidence}]}
    annotation = SimpleNamespace(model="scripted-fixture", effort="none",
        complete=lambda *args, **kwargs: {"content": json.dumps(raw)})
    hierarchy = build(video, job, annotation, directory / "hierarchy")
    calls = [("view_task_video", {"part": 1}), ("view_task_map", {}),
             ("view_demo_clip", {"phase": 1}), ("view_env_state", {}),
             ("move_to", None), ("rotate_wrist", {"arm": "left", "delta_yaw_deg": 2., "substeps": 2}),
             ("set_gripper", {"arm": "left", "val": 0., "steps": 2}),
             ("release", {"arm": "left", "val": 1., "steps": 2}),
             ("wait", {"steps": 2}), ("finish", {"status": "success", "summary": "Scripted interface smoke only"})]
    sent = []

    def respond(request):
        sent.append(json.loads(request.content))
        name, arguments = calls[len(sent)-1]
        if arguments is None:
            xyz = scene.state()["arms"]["left"]["tcp_pose_world"][:3]
            arguments = {"arm": "left", "xyz": [xyz[0]+.002, xyz[1], xyz[2]+.004], "quat": None, "substeps": 0}
        return httpx.Response(200, json={"id": "fixture-"+str(len(sent)), "object": "chat.completion",
            "created": 1, "model": job["model"], "choices": [{"index": 0, "finish_reason": "tool_calls",
            "message": {"role": "assistant", "content": None, "tool_calls": [{"id": "call-"+str(len(sent)),
            "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}]}}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}})

    client = SimpleNamespace(base="http://fixture.test/v1", key="fixture-not-a-secret", model=job["model"],
                             effort="none", timeout=30, calls=0)
    result = solve_native(client, scene, hierarchy, {**job, "max_turns": 12}, directory,
                          prompt=piper_prompt(), http_transport=httpx.MockTransport(respond))
    after = scene.state()
    transcript = json.loads((directory / "transcript.json").read_text())
    errors = [row for row in transcript if row["result"].get("tool_rejected") or
              row["result"].get("execution", {}).get("completed") is False]
    if len(sent) != len(calls) or len(transcript) != len(calls) or errors or scene.paused or result["stop_reason"] != "finish":
        raise RuntimeError("Scripted native-planner smoke failed; inspect transcript")
    if after["arms"]["left"]["gripper_opening_m"] < scene.gripper_max_m - .001:
        raise RuntimeError("Gripper opening was not confirmed from measured feedback")
    from scipy.spatial.transform import Rotation
    import numpy as np
    initial = before["arms"]["left"]["tcp_pose_world"]
    final = after["arms"]["left"]["tcp_pose_world"]
    error_m = float(np.linalg.norm(np.array(final[:3]) - np.array(initial[:3]) - [.002, 0., .004]))
    expected = Rotation.from_euler("z", 2., degrees=True) * Rotation.from_quat(initial[3:], scalar_first=True)
    error_deg = float(np.rad2deg((expected.inv() * Rotation.from_quat(final[3:], scalar_first=True)).magnitude()))
    if error_m > .001 or error_deg > .5:
        raise RuntimeError(f"Measured TCP does not match fixture motion: {error_m} m, {error_deg} deg")
    return {"kind": "scripted_provider_real_mujoco_protocol_smoke", "model_task_success": None,
            "independent_task_score": None, "provider_requests": len(sent), "tool_calls": len(transcript),
            "before": before, "after": after, "position_error_m": error_m,
            "orientation_error_deg": error_deg, "planner": result}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["smoke", "run"])
    parser.add_argument("--rpent-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "piper_simulation.json")
    parser.add_argument("--video", type=Path)
    parser.add_argument("--start-s", type=float, default=0.)
    parser.add_argument("--end-s", type=float)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--max-turns", type=int)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("Use a NEW output directory; previous runs are never overwritten or resumed implicitly")
    if args.command == "run" and args.video is None:
        parser.error("run requires --video")
    lock = verify_sources(args.rpent_root)
    config = json.loads(args.config.read_text())
    config["rpent_root"] = str(args.rpent_root.resolve(strict=True))
    if args.env_file:
        config["env_file"] = str(args.env_file.resolve(strict=True))
    if args.max_turns:
        if args.max_turns < 1:
            parser.error("max-turns must be positive")
        config["max_turns"] = args.max_turns
    output = args.output.resolve()
    output.mkdir(parents=True)
    shutil.copytree(ROOT / "rvicl_ig", output / "source/rvicl_ig", ignore=shutil.ignore_patterns("__pycache__"))
    save(output / "source_manifest.json", {"stage": "before any model request or executor startup",
        "files": {str(p.relative_to(output / "source")): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in (output / "source").rglob("*") if p.is_file()}})
    save(output / "config.json", config)
    save(output / "BASELINE.lock.json", lock)
    (output / "agent_prompt.txt").write_text(piper_prompt())
    try:
        if args.command == "run":
            # Validate credentials/video and build the original hierarchy BEFORE starting physics.
            client = (SchemaTextClient if config.get("schema_in_text") else Client)(config)
            video = EpisodeVideo(source(args.video, args.start_s, args.end_s), output / "demo_frames")
            save(output / "demo_source.json", {**video.source,
                "sha256": hashlib.sha256(video.path.read_bytes()).hexdigest()})
            hierarchy = build(video, config, client, output / "hierarchy")
        with PiperScene(config, output / "simulation") as scene:
            if args.command == "smoke":
                result = smoke(scene, config, output)
            else:
                result = solve_native(client, scene, hierarchy, config, output, prompt=piper_prompt())
                scene.check_cancelled()
                declared = json.loads((output / "planner_result.json").read_text())["finish_result"] or {}
                result = {"kind": "native_rpent_model_run", "planner": result,
                          "model_declared_status": declared.get("status"),
                          "model_summary": declared.get("summary"),
                          "model_task_success": None, "operator_scene_metrics": scene.finish(),
                          "metrics_scope": "Piper scene only; not official IG/LIBERO scores"}
        save(output / "result.json", {"status": "completed", **result})
    except BaseException as error:
        save(output / "error.json", {"status": "paused", "error_type": type(error).__name__, "message": str(error)})
        raise
    print(json.dumps({"status": "completed", "output": str(output), "kind": result["kind"]}))


if __name__ == "__main__":
    main()
