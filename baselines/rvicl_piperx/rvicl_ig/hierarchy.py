"""One VLM call builds L0/L1/L2; L3 reads original frames on demand."""

import hashlib
import json
from pathlib import Path

from .io import content, save

PROMPT = (Path(__file__).parent / "prompts/hierarchy.txt").read_text()
CAVEAT = ("Human-video visual estimates; no robot state annotations or benchmark predicates. "
          "Learn the demonstrated behaviors and skills; adapt contact/motion strategy and hand roles to the live robot scene. "
          "Achieve analogous behavior and outcomes instead of copying every human movement. Relocalize in the live scene. "
          "Never copy demonstration coordinates or assume a named grasp succeeded.")


def object_schema(properties):
    return {"type": "object", "additionalProperties": False,
            "properties": properties, "required": list(properties)}


def schema(limit):
    text = {"type": "string"}
    integer = {"type": "integer", "minimum": 0}
    def array(item, minimum=1, maximum=None):
        return {"type": "array", "minItems": minimum, "items": item,
                **({"maxItems": maximum} if maximum else {})}
    evidence = {"evidence": text, "uncertainty": text}
    return object_schema({"summary": text,
        "keyframes": array(object_schema({"index": integer, "stage": text,
            "left": text, "right": text, "result": text, **evidence}), 2, limit),
        "phases": array(object_schema({"id": {"type": "integer", "minimum": 1}, "label": text,
            "start_index": integer, "end_index": integer, "left": text, "right": text,
            "result": text, **evidence})),
        "moments": array(object_schema({"name": text, "kind": {"type": "string", "enum": [
            "grasp", "release", "handoff", "articulation", "contact", "end"]},
            "index": integer, "phase": {"type": "integer", "minimum": 1}, **evidence}))})


def validate(value, candidates, duration, limit):
    """Reject broken chronology/ranges rather than silently inventing missing stages."""
    if not isinstance(value, dict) or set(value) != {"summary", "keyframes", "phases", "moments"}:
        raise ValueError("Hierarchy must contain summary, keyframes, phases and moments")
    if not isinstance(value["summary"], str) or not value["summary"].strip():
        raise ValueError("Hierarchy summary must be nonempty")
    count = len(candidates)
    def index(value):
        if type(value) is not int or not 0 <= value < count:
            raise ValueError("Hierarchy index outside supplied candidate list")
        return value
    for collection in ("keyframes", "phases", "moments"):
        if not isinstance(value[collection], list) or not value[collection]:
            raise ValueError(f"Empty or invalid {collection}")
        spec = schema(limit)["properties"][collection]["items"]["properties"]
        for item in value[collection]:
            if not isinstance(item, dict) or set(item) != set(spec):
                raise ValueError(f"Invalid {collection} fields")
            for key, prop in spec.items():
                if prop["type"] == "string" and not isinstance(item[key], str):
                    raise ValueError(f"{key} must be text")
    keys = [index(k["index"]) for k in value["keyframes"]]
    if not 2 <= len(keys) <= limit or keys != sorted(set(keys)) or keys[0] != 0 or keys[-1] != count-1:
        raise ValueError("Keyframes must be unique, ordered and retain both episode boundaries")
    phases = value["phases"]
    previous = 0
    for i, phase in enumerate(phases, 1):
        a, b = index(phase["start_index"]), index(phase["end_index"])
        if type(phase["id"]) is not int or phase["id"] != i or a != previous or a >= b:
            raise ValueError("Phases must be consecutive, nonempty and cover the full timeline")
        phase["start_s"] = 0. if i == 1 else candidates[a]["timestamp_s"]
        phase["end_s"] = duration if i == len(phases) else candidates[b]["timestamp_s"]
        previous = b
    if previous != count-1:
        raise ValueError("Last phase must reach the end of the episode")
    seen, anchors, ends = set(), [], 0
    for moment in value["moments"]:
        anchor = index(moment["index"])
        pid = moment["phase"]
        if type(pid) is not int or not 1 <= pid <= len(phases):
            raise ValueError("Moment references an unknown phase")
        phase = phases[pid-1]
        if not phase["start_index"] <= anchor <= phase["end_index"]:
            raise ValueError("Moment falls outside its phase")
        name = moment["name"].strip().casefold()
        if not name or name in seen or moment["kind"] not in {"grasp", "release", "handoff", "articulation", "contact", "end"}:
            raise ValueError("Moment names must be unique and kinds valid")
        seen.add(name)
        anchors.append(anchor)
        time = candidates[anchor]["timestamp_s"]
        moment.update(time_s=time, start_s=max(0., time-1.), end_s=min(duration, time+1.))
        if moment["kind"] == "end":
            ends += 1
            if anchor != count-1:
                raise ValueError("End moment must use the final candidate")
            moment.update(start_s=max(0., duration-2.), end_s=duration)
    if anchors != sorted(anchors) or ends != 1:
        raise ValueError("Moments must be chronological with exactly one final end moment")
    return value


def build(video, config, client, directory):
    directory = Path(directory)
    identity = {"source": video.source, "prompt": PROMPT, "model": client.model,
                "effort": client.effort, "fps": config["candidate_fps"],
                "max_candidates": config["max_candidates"], "max_keyframes": config["max_keyframes"],
                "version": 1}
    signature = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    path = directory / "hierarchy.json"
    if path.exists():
        cached = json.loads(path.read_text())
        if cached.get("signature") == signature and all(Path(f["path"]).is_file() for f in cached["candidates"]):
            return Hierarchy(video, cached, config["clip_budget"])
    frames = video.candidates(config["candidate_fps"], config["max_candidates"])
    if len(frames) < 2:
        raise ValueError("Need at least two human video frames")
    request = {"task": "Learn the human behaviors and manipulation skills demonstrated in the video, and apply them with actions adapted to the current robot scene to achieve analogous behavior and outcomes, rather than copying the demonstrated motions verbatim.",
        "keyframe_limit": config["max_keyframes"], "duration_s": video.duration,
        "candidate_count": len(frames), "effective_max_gap_s": max(
            b["timestamp_s"]-a["timestamp_s"] for a,b in zip(frames, frames[1:])),
        "_images": [{"path": f["path"], "caption": f"candidate {f['index']}, t={f['timestamp_s']:.6f}s"} for f in frames]}
    print(f"Building video L0-L2 with ONE VLM request ({len(frames)} candidates)", flush=True)
    message = client.complete([{"role": "system", "content": PROMPT},
        {"role": "user", "content": content(request)}], schema=schema(config["max_keyframes"]),
        artifact=directory / "hierarchy_response.json")
    raw = json.loads(message["content"])
    data = validate(raw, frames, video.duration, config["max_keyframes"])
    data.update(signature=signature, source=video.source, candidates=frames,
                duration_s=video.duration, annotation_source="vlm_rgb_only", caveat=CAVEAT)
    data["vlm_trace"] = getattr(client, "last_trace_directory", str(directory / "hierarchy_response"))
    save(path, data)
    return Hierarchy(video, data, config["clip_budget"])


class Hierarchy:
    def __init__(self, video, data, budget=3):
        self.video, self.data, self.budget = video, data, budget
        self.views, self.pages, self.map_read = {}, set(), False

    @property
    def page_count(self):
        return (len(self.data["keyframes"]) + 7) // 8

    @property
    def ready(self):
        return self.map_read and self.pages == set(range(1, self.page_count+1))

    def view_task_video(self, part=1):
        if type(part) is not int or not 1 <= part <= self.page_count:
            raise ValueError("Unknown keyframe page")
        self.pages.add(part)
        keys = self.data["keyframes"][(part-1)*8:part*8]
        return {"part": part, "parts": self.page_count, "caveat": CAVEAT, "keyframes": keys,
            "_images": [{"path": self.data["candidates"][k["index"]]["path"],
                         "caption": json.dumps(k)} for k in keys]}

    def view_task_map(self):
        self.map_read = True
        return {key: self.data[key] for key in ("summary", "phases", "moments", "duration_s", "caveat")}

    def view_demo_clip(self, moment=None, phase=None):
        from .video import contact_sheet
        if not self.ready:
            raise ValueError("Read every L0 keyframe page and the L1/L2 map before requesting clips")
        if (moment is None) == (phase is None):
            raise ValueError("Specify exactly one moment or phase")
        if moment is not None:
            if not isinstance(moment, str):
                raise ValueError("moment must be a name")
            matches = [m for m in self.data["moments"] if m["name"].casefold() == moment.casefold()]
            key = "moment-" + str(self.data["moments"].index(matches[0])) if matches else ""
        else:
            matches = [p for p in self.data["phases"] if type(phase) is int and p["id"] == phase]
            key = f"phase-{phase}"
        if not matches:
            raise ValueError("Unknown demonstration node; use names/ids from view_task_map")
        if self.views.get(key, 0) >= self.budget:
            raise ValueError("Viewing budget exhausted for this node; act using the available evidence")
        node = matches[0]
        frames = self.video.clip(node["start_s"], node["end_s"], key)
        sheet = contact_sheet(frames, Path(frames[0]["path"]).parent / f"{key}-strip.png")
        self.views[key] = self.views.get(key, 0) + 1
        return {"node": node, "caveat": CAVEAT, "frames_shown": len(frames),
                "frame_times_s": [f["timestamp_s"] for f in frames],
                "_images": [{"path": sheet, "caption": "Original human video frames in chronological row order"}]}
