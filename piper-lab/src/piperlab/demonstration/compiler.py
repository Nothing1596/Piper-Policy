"""Demonstration compiler (human-video v1).

``compile_demo`` turns one local human-demonstration video into a versioned
demo bundle:

1. ``video.candidates.build_candidates`` produces bounded, CV-scored
   candidate frames (no learned detector, no model in the loop);
2. the caller-supplied model selects keyframes batch by batch, at most
   ``images_per_call`` images per ``model.infer`` call;
3. a global semantic pass orders the selected evidence into stages,
   constraints and unknowns;
4. every model claim is validated locally. Unknown frame IDs, broken
   chronology, out-of-enum verdicts and missing fields are rejected and
   re-queried inside a bounded budget; once the budget is exhausted the gap
   is recorded as ``unresolved`` in ``unknowns``. Nothing is fabricated and
   nothing is silently truncated.

A compiled demo is historical evidence only: ``outcome_verdict`` describes
the recorded demonstration and never certifies a current or future
execution. ``model_identity`` is whatever the model object self-reports;
it is explicitly *not* transport-level proof of which model answered.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import copy
from pathlib import Path

PROMPT_VERSION = "human-video-v1.4"


def _detection_brief(detections):
    """Bound model hints; retain full detector diagnostics in candidate artifacts."""
    if not isinstance(detections, dict):
        return None
    regions = detections.get('regions', [])
    ranked = sorted(regions, key=lambda r: r.get('score', 0), reverse=True)[:12]
    return {
        'interpretation': 'Detector labels are fallible proposals, not ground truth. Prefer visible image evidence.',
        'count': detections.get('count', len(regions)),
        'labels': [str(label)[:80] for label in detections.get('labels', [])[:12]],
        'regions': [dict(label=str(r.get('label', 'unknown'))[:80],
                         score=round(float(r.get('score', 0)), 3),
                         bbox_xyxy=[round(float(v), 1) for v in r.get('bbox_xyxy', [])[:4]],
                         track_id=r.get('track_id'), track_epoch=r.get('track_epoch')) for r in ranked],
        'omitted_regions': max(0, len(regions)-len(ranked)),
        'diagnostics': 'Full detector/tracker diagnostic records remain in the candidate artifact.'}

VERDICTS = ("supported", "refuted", "unknown")
UNRESOLVED = "unresolved"

SELECT_SCHEMA: dict = {
    "type": "object",
    "required": ["selections"],
    "properties": {
        "selections": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["frame_id", "keep", "reason"],
                "properties": {
                    "frame_id": {"type": "string"},
                    "keep": {"type": "boolean"},
                    "reason": {"type": "string"},
                    "operation": {"type": "string"},
                    "stage_hint": {"type": "string"},
                    "object_roles": {"type": "array", "items": {"type": "string"}},
                    "visible_outcome": {"type": "string"},
                },
            },
        }
    },
}

SEMANTIC_SCHEMA: dict = {
    "type": "object",
    "required": ["stages", "goal_constraints", "unknowns", "outcome_verdict", "summary"],
    "properties": {
        "stages": {
            "type": "array",
            "items": {
                "type": "object",
                "required": [
                    "id", "operation", "object_roles", "preconditions",
                    "expected_effects", "evidence_refs", "uncertainty",
                ],
                "properties": {
                    "id": {"type": "string"},
                    "operation": {"type": "string"},
                    "object_roles": {"type": "array", "items": {"type": "string"}},
                    "preconditions": {"type": "array", "items": {"type": "string"}},
                    "expected_effects": {"type": "array", "items": {"type": "string"}},
                    "evidence_refs": {"type": "array", "items": {"type": "string"}},
                    "uncertainty": {"type": "string"},
                },
            },
        },
        "goal_constraints": {"type": "array", "items": {"type": "string"}},
        "unknowns": {"type": "array", "items": {"type": "string"}},
        "outcome_verdict": {"type": "string", "enum": list(VERDICTS)},
        "summary": {"type": "string"},
    },
}

SELECT_PROMPT = """\
You are selecting keyframes from a demonstration video for the task (it may be synthetic; do not assume a human or a grasp is visible):
"{task}"

Candidate frames for THIS batch (frame_id, timestamp_s, selection hints):
{batch}

Rules:
- Only reference frame_id values listed above; never invent an ID.
- Keep frames that mark stage boundaries, object contact/release, or visible
  outcomes; skip redundant frames.
- Reply with JSON matching the supplied schema: one entry per frame you have
  an opinion on, keep=true for keyframes, with a concrete reason.
"""

SEMANTIC_PROMPT = """\
You are compiling a demonstration into a structured stage plan for the
task: "{task}"

Selected keyframes, in chronological order (frame_id, timestamp_s,
selection notes):
{evidence}

Images attached to this call, in order (direct visual evidence):
{images}

Text-only selected frames (batch-selection notes; NOT shown to you in this
call, treat any claim about them as second-hand):
{text_only}

Rules:
- evidence_refs may only use frame_id values listed above; never invent one.
- Prefer image-backed frames as evidence; text-only frames are indirect
  knowledge from the batch selection pass.
- Stages must be chronological: each stage's evidence_refs must be
  non-decreasing in timestamp, and stages must be ordered by their earliest
  evidence.
- outcome_verdict is one of supported|refuted|unknown and refers ONLY to the
  recorded historical demonstration.
- List every genuine uncertainty in unknowns instead of guessing.
- Reply with JSON matching the supplied schema.
"""

REPAIR_SUFFIX = """\

Your previous reply failed local validation with these errors:
{errors}

Return a corrected JSON document only. Do not invent frame_id values; use
exactly the IDs supplied in the original request.
"""


class CompileError(RuntimeError):
    """Demonstration compilation failed explicitly."""


def _fail_if_exists(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"output already exists, refusing to overwrite: {path}")


def _write_json_atomic(path: Path, payload: dict) -> None:
    tmp = path.with_name(path.name + f".tmp-{os.getpid()}")
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp, path)


def _canonical(payload) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


class _RequeryBudget:
    """Total re-query budget across the whole compile (plan A.6: max 3)."""

    def __init__(self, total: int) -> None:
        if isinstance(total, bool) or not isinstance(total, int) or total < 0:
            raise ValueError("requery_budget must be a non-negative int")
        self.total = total
        self.used = 0

    @property
    def exhausted(self) -> bool:
        return self.used >= self.total

    def spend(self) -> None:
        if self.exhausted:
            raise CompileError("re-query budget already exhausted")
        self.used += 1


# ---------------------------------------------------------------------------
# manifest / response validation


def _validate_manifest(manifest: dict, video_path: Path) -> list[dict]:
    """Return the candidate list, or raise CompileError on a bogus manifest."""
    if not isinstance(manifest, dict):
        raise CompileError("candidate builder returned a non-dict manifest")
    source = manifest.get("source")
    if not isinstance(source, dict) or not source.get("path") or not source.get("sha256"):
        raise CompileError("candidate manifest lacks source path/sha256")
    candidates = manifest.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise CompileError("candidate manifest has no candidates; nothing to compile")
    seen: set[str] = set()
    last_ts = -math.inf
    for cand in candidates:
        frame_id = cand.get("frame_id")
        ts = cand.get("timestamp_s")
        image_path = cand.get("image_path")
        if not isinstance(frame_id, str) or not frame_id:
            raise CompileError(f"candidate without a valid frame_id: {cand!r}")
        if frame_id in seen:
            raise CompileError(f"duplicate candidate frame_id: {frame_id}")
        seen.add(frame_id)
        if not isinstance(ts, (int, float)) or isinstance(ts, bool) or not math.isfinite(ts):
            raise CompileError(f"candidate {frame_id} has no finite timestamp_s")
        if ts < last_ts:
            raise CompileError("candidate manifest is not chronologically ordered")
        last_ts = ts
        if not isinstance(image_path, str) or not Path(image_path).is_file():
            raise CompileError(f"candidate {frame_id} image missing on disk: {image_path!r}")
    return candidates


def _validate_selection_response(response, supplied_ids: list[str]) -> list[str]:
    """Validate one batch selection response against the IDs actually supplied."""
    errors: list[str] = []
    if not isinstance(response, dict):
        return ["selection response is not a JSON object"]
    selections = response.get("selections")
    if not isinstance(selections, list):
        return ["selection response lacks a 'selections' array"]
    supplied = set(supplied_ids)
    seen: set[str] = set()
    for index, entry in enumerate(selections):
        if not isinstance(entry, dict):
            errors.append(f"selections[{index}] is not an object")
            continue
        frame_id = entry.get("frame_id")
        if not isinstance(frame_id, str) or frame_id not in supplied:
            errors.append(
                f"selections[{index}].frame_id {frame_id!r} was not supplied in this batch"
            )
            continue
        if frame_id in seen:
            errors.append(f"selections[{index}] duplicates frame_id {frame_id!r}")
        seen.add(frame_id)
        if not isinstance(entry.get("keep"), bool):
            errors.append(f"selections[{index}].keep is not a boolean")
        if entry.get("keep") is True:
            reason = entry.get("reason")
            if not isinstance(reason, str) or not reason.strip():
                errors.append(f"selections[{index}] kept without a non-empty reason")
        for key in ("operation", "stage_hint", "visible_outcome"):
            if key in entry and entry[key] is not None and not isinstance(entry[key], str):
                errors.append(f"selections[{index}].{key} is not a string")
        roles = entry.get("object_roles")
        if roles is not None and (
            not isinstance(roles, list) or not all(isinstance(r, str) for r in roles)
        ):
            errors.append(f"selections[{index}].object_roles is not a list of strings")
    return errors


def _validate_semantic_response(response, supplied_ids: list[str], ts_by_id: dict) -> list[str]:
    """Validate the global stage plan against the evidence actually supplied."""
    errors: list[str] = []
    if not isinstance(response, dict):
        return ["semantic response is not a JSON object"]
    stages = response.get("stages")
    if not isinstance(stages, list):
        errors.append("semantic response lacks a 'stages' array")
        stages = []
    supplied = set(supplied_ids)
    stage_ids: set[str] = set()
    previous_earliest = -math.inf
    for index, stage in enumerate(stages):
        if not isinstance(stage, dict):
            errors.append(f"stages[{index}] is not an object")
            continue
        stage_id = stage.get("id")
        if not isinstance(stage_id, str) or not stage_id.strip():
            errors.append(f"stages[{index}].id is missing or empty")
        elif stage_id in stage_ids:
            errors.append(f"duplicate stage id {stage_id!r}")
        else:
            stage_ids.add(stage_id)
        operation = stage.get("operation")
        if not isinstance(operation, str) or not operation.strip():
            errors.append(f"stages[{index}].operation is missing or empty")
        for key in ("object_roles", "preconditions", "expected_effects"):
            value = stage.get(key)
            if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
                errors.append(f"stages[{index}].{key} is not a list of strings")
        if not isinstance(stage.get("uncertainty"), str):
            errors.append(f"stages[{index}].uncertainty is missing (use '' if none)")
        refs = stage.get("evidence_refs")
        if not isinstance(refs, list) or not all(isinstance(r, str) for r in refs):
            errors.append(f"stages[{index}].evidence_refs is not a list of strings")
            continue
        bad = [r for r in refs if r not in supplied]
        if bad:
            errors.append(f"stages[{index}].evidence_refs not supplied as evidence: {bad!r}")
            continue
        times = [ts_by_id[r] for r in refs]
        if times != sorted(times):
            errors.append(f"stages[{index}].evidence_refs are not chronological")
        if times:
            earliest = times[0]
            if earliest < previous_earliest - 1e-12:
                errors.append(f"stages[{index}] starts before the previous stage; not ordered")
            previous_earliest = max(previous_earliest, earliest)
    for key in ("goal_constraints", "unknowns"):
        value = response.get(key)
        if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
            errors.append(f"semantic response {key!r} is not a list of strings")
    verdict = response.get("outcome_verdict")
    if verdict not in VERDICTS:
        errors.append(f"outcome_verdict {verdict!r} is not one of {VERDICTS}")
    if not isinstance(response.get("summary"), str):
        errors.append("semantic response 'summary' is not a string")
    return errors


# ---------------------------------------------------------------------------
# model calls


def _infer_validated(model, prompt: str, images: list[Path], schema: dict,
                     validate, budget: _RequeryBudget, call_log: list[dict],
                     role: str):
    """Call model.infer, validate, and re-query with feedback within budget.

    Returns (response, errors). response is None when the budget was exhausted
    with the response still invalid; the errors explain exactly why.
    """
    errors: list[str] = []
    while True:
        try:
            response = model.infer(prompt, images, schema)
            call_log.append({"role": role, "images": len(images), "error": None})
        except Exception as exc:  # transport/runtime failure: retry within budget
            response = None
            call_log.append({"role": role, "images": len(images), "error": repr(exc)})
            errors = [f"model.infer raised {type(exc).__name__}: {exc}"]
        else:
            errors = validate(response)
        if not errors:
            return response, []
        if budget.exhausted:
            return None, errors
        budget.spend()
        prompt = prompt + REPAIR_SUFFIX.format(errors=json.dumps(errors, indent=2))


def _check_images_per_call(images: list[Path], limit: int) -> None:
    if len(images) > limit:
        raise CompileError(
            f"internal error: {len(images)} images exceed the per-call limit {limit}"
        )


def _model_identity(model) -> dict:
    identity = getattr(model, "identity", None)
    if not isinstance(identity, dict) or not identity:
        raise CompileError("model must expose a non-empty .identity dict property")
    try:
        json.dumps(identity)
    except (TypeError, ValueError) as exc:
        raise CompileError(f"model.identity is not JSON-serializable: {exc}") from exc
    return dict(identity)


def _rebase_under(path_str: str, staging: Path, output: Path) -> str:
    """Map a path inside the staging tree to its final committed location.

    The candidate builder records absolute paths under the directory it was
    given (``staging/candidates``); the bundle is only renamed into place at
    commit time, so recorded paths must point at the final location to stay
    valid after commit. Paths outside the staging tree pass through unchanged.
    """
    rel = os.path.relpath(path_str, staging)
    if rel.startswith(".."):
        return path_str
    return str((output / rel).absolute())


def _cache_key(source_hash: str, task: str, identity: dict, parameters: dict) -> str:
    payload = {
        "source_hash": source_hash,
        "task": task,
        "model_identity": identity,
        "prompt_version": PROMPT_VERSION,
        "parameters": parameters,
    }
    return hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()


def _load_cache(cache_dir: Path, key: str, identity_payload: dict):
    """Return a transcript only on an exact identity match; else None."""
    path = cache_dir / f"{key}.json"
    if not path.is_file():
        return None
    with open(path, "r", encoding="utf-8") as handle:
        entry = json.load(handle)
    if entry.get("identity") != identity_payload or not isinstance(entry.get("transcript"), dict):
        raise CompileError(
            f"cache entry {path} identity mismatch or malformed transcript; "
            "refusing to reuse it (cache is only valid on exact identity match)"
        )
    return entry["transcript"]


def _store_cache(cache_dir: Path, key: str, identity_payload: dict, transcript: dict) -> str:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{key}.json"
    if path.exists():
        return "exists"
    _write_json_atomic(path, {"identity": identity_payload, "transcript": transcript})
    return "written"


# ---------------------------------------------------------------------------
# keyframe bookkeeping


def _spread_pick(frame_ids: list[str], limit: int) -> list[str]:
    """Evenly spread picks, always including the first and last frame."""
    if len(frame_ids) <= limit:
        return list(frame_ids)
    if limit < 2:
        return frame_ids[:limit]
    picks = []
    for i in range(limit):
        idx = round(i * (len(frame_ids) - 1) / (limit - 1))
        if frame_ids[idx] not in picks:
            picks.append(frame_ids[idx])
    return picks


def _trim_to_budget(frames: list[dict], referenced: set[str],
                    max_keyframes: int) -> tuple[list[dict], list[dict], bool]:
    """Enforce max_keyframes explicitly.

    Boundary (first/last) frames and frames referenced as stage evidence are
    never trimmed. Interior unreferenced frames with the smallest neighbour
    gap are dropped first. Returns (frames, dropped, exceeded): ``exceeded``
    is True when the budget still cannot be met because every remaining frame
    is evidence-referenced; that fact is reported, not hidden.
    """
    if len(frames) <= max_keyframes:
        return frames, [], False
    kept = list(frames)
    dropped: list[dict] = []
    while len(kept) > max_keyframes:
        interior = [
            (i, f) for i, f in enumerate(kept)
            if i not in (0, len(kept) - 1) and f["frame_id"] not in referenced
        ]
        if not interior:
            return kept, dropped, True
        # Drop the interior frame whose nearest kept neighbour is closest in
        # time (most redundant); ties resolve to the earlier frame.
        def redundancy(item) -> float:
            i, _f = item
            before = kept[i - 1]["timestamp_s"]
            after = kept[i + 1]["timestamp_s"]
            return min(kept[i]["timestamp_s"] - before, after - kept[i]["timestamp_s"])

        victim_index, victim = min(interior, key=lambda item: (redundancy(item), item[0]))
        dropped.append({
            "frame_id": victim["frame_id"],
            "timestamp_s": victim["timestamp_s"],
            "reason": "max_keyframes",
            "detail": f"dropped to honour max_keyframes={max_keyframes}",
        })
        kept.pop(victim_index)
    return kept, dropped, False


# ---------------------------------------------------------------------------
# entry point


def compile_demo(
    video_path: str,
    output_dir: str,
    task: str,
    model,
    *,
    window_s: float = 10,
    max_keyframes: int = 24,
    candidate_builder=None,
    cache_dir: str | None = None,
    requery_budget: int = 3,
    images_per_call: int = 6,
    detector=None,
    detector_hz: float = 2,
    _single_segment: bool = False,
    _shared_source_path=None,
    _segment_context=None,
) -> dict:
    """Compile one human-demonstration video into a versioned demo bundle.

    ``model`` must expose ``.identity -> dict`` and
    ``.infer(prompt: str, images: list[Path], schema: dict) -> dict``; root
    provides the HTTP implementation. Returns the persisted demo dict (also
    written to ``output_dir/demo.json``). The output directory must not
    exist; all artifacts are staged and renamed into place atomically.
    """
    if not isinstance(task, str) or not task.strip():
        raise CompileError("task must be a non-empty string")
    if isinstance(max_keyframes, bool) or not isinstance(max_keyframes, int) \
            or max_keyframes < 2:
        raise CompileError("max_keyframes must be an int >= 2 (first and last are mandatory)")
    if isinstance(images_per_call, bool) or not isinstance(images_per_call, int) \
            or images_per_call <= 0:
        raise CompileError("images_per_call must be a positive int")
    if images_per_call > 6:
        raise CompileError("images_per_call is bounded at 6 per the human-video v1 plan")
    if isinstance(window_s, bool) or not isinstance(window_s, (int, float)) \
            or not math.isfinite(window_s) or window_s <= 0:
        raise CompileError("window_s must be a positive finite number")
    budget = _RequeryBudget(requery_budget)
    identity = _model_identity(model)
    if cache_dir is not None and identity.get('cache_identity_complete') is False:
        raise CompileError('persistent cache requires a verified model artifact manifest; pass --model-manifest or omit --cache')
    source_path = Path(video_path).absolute()
    if not source_path.is_file():
        raise CompileError(f"video source does not exist: {source_path}")

    if candidate_builder is None:
        from ..video.candidates import build_candidates as candidate_builder

    output = Path(output_dir).absolute()
    _fail_if_exists(output)
    staging = output.parent / (output.name + f".tmp-{os.getpid()}")
    if staging.exists():
        raise FileExistsError(f"stale staging directory in the way: {staging}")
    staging.mkdir(parents=True)

    # 1. candidates (traditional CV only; detector optional, no model involved)
    candidates_dir = staging / "candidates"
    manifest = candidate_builder(
        str(source_path), str(candidates_dir), window_s=window_s,
        detector=detector, detector_hz=detector_hz,
    )
    candidates = _validate_manifest(manifest, source_path)
    manifest_file = candidates_dir / "manifest.json"
    source_hash = manifest["source"]["sha256"]
    if not _single_segment and candidates[-1]['timestamp_s']-candidates[0]['timestamp_s']>window_s:
        return _compile_segments(source_path,output,staging,task,model,manifest,
            window_s=window_s,max_keyframes=max_keyframes,cache_dir=cache_dir,
            requery_budget=requery_budget,images_per_call=images_per_call)
    ts_by_id = {c["frame_id"]: c["timestamp_s"] for c in candidates}
    cand_by_id = {c["frame_id"]: c for c in candidates}

    parameters = {
        "window_s": window_s,
        "max_keyframes": max_keyframes,
        "images_per_call": images_per_call,
        "requery_budget": requery_budget,
        "detector": getattr(detector, "name", type(detector).__name__) if detector else None,
        "detector_sha256":getattr(detector,'sha256',None),
        "selected_image_long_edge":768,
        "detector_hz": detector_hz if detector is not None else None,
        "candidate_parameters": manifest.get("parameters", {}),
        "segment_context":_segment_context,
    }
    model_task=task
    if _segment_context:
        model_task+='\nThis is one bounded chronological segment of a longer video: '+json.dumps(_segment_context)
        model_task+=' Describe only this segment. Its first/last image may be mid-action; do not infer unseen contact, earlier preconditions, or completion of the whole task. Mark missing boundary evidence unknown.'
    cache_identity = {
        "source_hash": source_hash,
        "task": task,
        "model_identity": identity,
        "prompt_version": PROMPT_VERSION,
        "parameters": parameters,
    }
    cache_status = {"status": "disabled", "key": None}
    transcript = None
    if cache_dir is not None:
        key = _cache_key(source_hash, task, identity, parameters)
        transcript = _load_cache(Path(cache_dir), key, cache_identity)
        cache_status = {"status": "hit" if transcript is not None else "miss", "key": key}

    call_log: list[dict] = []
    unknowns: list[str] = []

    # 2. per-batch selection (or validated reuse from an identity-matched cache)
    selections: dict[str, dict] = {}
    if transcript is not None:
        batch_responses = transcript.get("batch_responses")
        if not isinstance(batch_responses, list):
            raise CompileError("cached transcript lacks batch_responses")
        expected_batches = math.ceil(len(candidates) / images_per_call)
        if len(batch_responses) != expected_batches:
            raise CompileError("cached transcript batch count does not match candidates")
        for start in range(0, len(candidates), images_per_call):
            batch = candidates[start:start + images_per_call]
            ids = [c["frame_id"] for c in batch]
            response = batch_responses[start // images_per_call]
            errors = _validate_selection_response(response, ids)
            if errors:
                raise CompileError(
                    f"cached selection for batch at {start} failed revalidation: {errors}"
                )
            for entry in response["selections"]:
                if entry["keep"]:
                    selections[entry["frame_id"]] = entry
    else:
        batch_responses = []
        for start in range(0, len(candidates), images_per_call):
            batch = candidates[start:start + images_per_call]
            ids = [c["frame_id"] for c in batch]
            images = [Path(c["image_path"]) for c in batch]
            _check_images_per_call(images, images_per_call)
            brief = [
                {
                    "frame_id": c["frame_id"],
                    "timestamp_s": c["timestamp_s"],
                    "reasons": c.get("reasons", []),
                    "scores": {
                        "motion": c.get("scores", {}).get("motion"),
                        "clarity": c.get("scores", {}).get("clarity"),
                    },
                    "detections": _detection_brief(c.get("detections")),
                }
                for c in batch
            ]
            prompt = SELECT_PROMPT.format(task=model_task, batch=json.dumps(brief, indent=2))
            selection_schema=copy.deepcopy(SELECT_SCHEMA)
            selection_schema['properties']['selections']['items']['properties']['frame_id']['enum']=ids
            response, errors = _infer_validated(
                model, prompt, images, selection_schema,
                lambda r, ids=ids: _validate_selection_response(r, ids),
                budget, call_log, role="select",
            )
            batch_responses.append(response)
            if response is None:
                unknowns.append(
                    f"unresolved: selection batch starting at {ids[0]} exhausted the "
                    f"re-query budget; last errors: {errors}"
                )
                continue
            for entry in response["selections"]:
                if entry["keep"]:
                    selections[entry["frame_id"]] = entry

    # 3. boundary frames are always included, explicitly marked
    first_id = candidates[0]["frame_id"]
    last_id = candidates[-1]["frame_id"]
    for boundary_id in (first_id, last_id):
        if boundary_id not in selections:
            selections[boundary_id] = {
                "frame_id": boundary_id,
                "keep": True,
                "reason": "initial/final frame preserved by the compiler",
                "selected_by": "compiler_boundary",
            }
            unknowns.append(
                f"model did not select boundary frame {boundary_id}; "
                "added by the compiler as initial/final evidence"
            )

    selected_ids = sorted(selections, key=lambda fid: ts_by_id[fid])

    # 4. global semantic pass over the selected evidence
    evidence_brief = []
    for fid in selected_ids:
        entry = selections[fid]
        evidence_brief.append({
            "frame_id": fid,
            "timestamp_s": ts_by_id[fid],
            "reason": entry.get("reason"),
            "operation": entry.get("operation"),
            "stage_hint": entry.get("stage_hint"),
            "object_roles": entry.get("object_roles"),
            "visible_outcome": entry.get("visible_outcome"),
            "detections": _detection_brief(cand_by_id[fid].get("detections")),
        })
    # Exactly these frames are attached as images, in this order; every other
    # selected frame reaches the model only as text from the batch pass.
    imaged_ids = _spread_pick(selected_ids, images_per_call)
    semantic_images = [Path(cand_by_id[fid]["image_path"]) for fid in imaged_ids]
    _check_images_per_call(semantic_images, images_per_call)
    text_only_ids = [fid for fid in selected_ids if fid not in set(imaged_ids)]
    images_section = "\n".join(
        f"  image {i + 1}: frame_id={fid} timestamp_s={ts_by_id[fid]:.6f}"
        for i, fid in enumerate(imaged_ids)
    ) or "  (none)"
    text_only_brief = [b for b in evidence_brief if b["frame_id"] in set(text_only_ids)]

    if transcript is not None:
        semantic = transcript.get("semantic")
        errors = _validate_semantic_response(semantic, selected_ids, ts_by_id)
        if errors:
            raise CompileError(f"cached semantic response failed revalidation: {errors}")
    else:
        prompt = SEMANTIC_PROMPT.format(
            task=model_task,
            evidence=json.dumps(evidence_brief, indent=2),
            images=images_section,
            text_only=json.dumps(text_only_brief, indent=2),
        )
        semantic_schema=copy.deepcopy(SEMANTIC_SCHEMA)
        semantic_schema['properties']['stages']['items']['properties']['evidence_refs']['items']['enum']=selected_ids
        semantic, errors = _infer_validated(
            model, prompt, semantic_images, semantic_schema,
            lambda r: _validate_semantic_response(r, selected_ids, ts_by_id),
            budget, call_log, role="semantic",
        )

    if semantic is None:
        stages: list[dict] = []
        goal_constraints: list[str] = []
        summary = ""
        verdict = UNRESOLVED
        unknowns.append(
            f"unresolved: semantic pass exhausted the re-query budget; "
            f"last errors: {errors}"
        )
    else:
        stages = [
            {
                "id": s["id"],
                "operation": s["operation"],
                "object_roles": list(s["object_roles"]),
                "preconditions": list(s["preconditions"]),
                "expected_effects": list(s["expected_effects"]),
                "evidence_refs": list(s["evidence_refs"]),
                "uncertainty": s["uncertainty"],
            }
            for s in semantic["stages"]
        ]
        goal_constraints = list(semantic["goal_constraints"])
        summary = semantic["summary"]
        verdict = semantic["outcome_verdict"]
        unknowns.extend(semantic["unknowns"])

    # 5. explicit max-keyframe enforcement (never trim evidence silently)
    referenced = {ref for stage in stages for ref in stage["evidence_refs"]}
    frames = [
        {
            "frame_id": fid,
            "image_path": cand_by_id[fid]["image_path"],
            "timestamp_s": ts_by_id[fid],
            "selected_by": selections[fid].get("selected_by", "model"),
            "reason": selections[fid].get("reason", ""),
        }
        for fid in selected_ids
    ]
    frames, dropped_keyframes, budget_exceeded = _trim_to_budget(frames, referenced, max_keyframes)
    if budget_exceeded:
        unknowns.append(
            f"keyframe budget exceeded: {len(frames)} frames kept although "
            f"max_keyframes={max_keyframes}, because every remaining frame is "
            "boundary or stage evidence; reported, not silently trimmed"
        )

    if cache_dir is not None and cache_status["status"] == "miss":
        # Only fully validated transcripts are cached: unresolved batches or an
        # unresolved semantic pass must not become sticky through the cache.
        complete = verdict != UNRESOLVED and all(r is not None for r in batch_responses)
        if complete:
            transcript = {"batch_responses": batch_responses, "semantic": semantic}
            try:
                cache_status["store"] = _store_cache(
                    Path(cache_dir), cache_status["key"], cache_identity, transcript
                )
            except OSError as exc:
                cache_status["store"] = f"failed: {exc}"
        else:
            cache_status["store"] = "skipped_unresolved"

    # 6. assemble + persist
    demo_core = {
        "source_hash": source_hash,
        "task": task,
        "model_identity": identity,
        "prompt_version": PROMPT_VERSION,
        "parameters": parameters,
        "frames": [{"frame_id": f["frame_id"], "timestamp_s": f["timestamp_s"]} for f in frames],
        "stages": stages,
        "goal_constraints": goal_constraints,
        "outcome_verdict": verdict,
    }
    demo_id = "demo-" + hashlib.sha256(_canonical(demo_core).encode("utf-8")).hexdigest()[:16]

    demo = {
        "schema_version": 1,
        "demo_id": demo_id,
        "created_by": "piperlab.demonstration.compiler.compile_demo",
        "source_path": str(_shared_source_path or output / ('source'+source_path.suffix)),
        "source_hash": source_hash,
        "task": task,
        "model_identity": {
            **identity,
            "identity_verified": bool(identity.get('identity_verified',False)),
            "identity_note": (
                "See identity_source for adapter verification. When identity_verified is false this is not transport-level proof. Weight hashes require a separate artifact manifest."
            ),
        },
        "prompt_version": PROMPT_VERSION,
        "parameters": parameters,
        "frames": frames,
        "stages": stages,
        "goal_constraints": goal_constraints,
        "unknowns": unknowns,
        "dropped_keyframes": dropped_keyframes,
        "outcome_verdict": verdict,
        "outcome_scope": "historical_demo_only",
        "summary": summary,
        "cache": cache_status,
        "requery": {"budget": requery_budget, "used": budget.used},
        "semantic_evidence": {
            "images_supplied": imaged_ids,
            "text_only": text_only_ids,
            "note": (
                "images_supplied frames were directly visible to the model in "
                "the semantic pass; text_only frames reached it only as "
                "batch-selection notes"
            ),
        },
        "model_calls": [
            {"role": c["role"], "images": c["images"], "error": c["error"]} for c in call_log
        ],
        "candidate_manifest": "candidates/manifest.json",
    }
    json.dumps(demo)  # fail here, not after commit, if anything is not serializable
    # Inference reads existing staging files. Rebase only after the last call.
    for frame in demo['frames']:
        frame['image_path']=_rebase_under(frame['image_path'],staging,output)
    for cand in candidates:
        cand['image_path']=_rebase_under(cand['image_path'],staging,output)
    if manifest_file.is_file():_write_json_atomic(manifest_file,manifest)
    if _shared_source_path is None:
        archived=staging / ('source'+source_path.suffix)
        shutil.copy2(source_path,archived)
        with archived.open('rb') as stream:
            if hashlib.file_digest(stream,'sha256').hexdigest()!=source_hash:
                raise CompileError('source changed between analysis and archive')
    _write_json_atomic(staging / "demo.json", demo)
    os.rename(staging, output)
    return demo


def _compile_segments(source,output,staging,task,model,manifest,*,window_s,max_keyframes,
                      cache_dir,requery_budget,images_per_call):
    """Keep every time window in a separate bounded semantic context and archive once."""
    archived=staging/('source'+source.suffix)
    shutil.copy2(source,archived)
    with archived.open('rb') as stream:
        if hashlib.file_digest(stream,'sha256').hexdigest()!=manifest['source']['sha256']:
            raise CompileError('source changed between analysis and archive')
    windows={}
    origin=manifest['candidates'][0]['timestamp_s']
    for candidate in manifest['candidates']:
        window=int((candidate['timestamp_s']-origin)//window_s)
        windows.setdefault(window,[]).append(candidate)
    chunks=[values[start:start+max_keyframes] for values in windows.values()
            for start in range(0,len(values),max_keyframes)]
    segments=[]
    for number,chunk in enumerate(chunks):
        segment_id=f'segment-{number:04d}'
        def builder(video,directory,**ignored):
            directory=Path(directory);(directory/'frames').mkdir(parents=True)
            subset=copy.deepcopy(manifest)
            subset['source']['path']=str(archived)
            if 'index' in subset['source']:subset['source']['index']='../../../candidates/source-index.json'
            subset['parameters']={**subset.get('parameters',{}),'segment_id':segment_id,
                'segment_frame_ids':[c['frame_id'] for c in chunk]}
            subset['candidates']=[]
            for original in chunk:
                item=copy.deepcopy(original)
                image=directory/'frames'/Path(item['image_path']).name
                shutil.copy2(item['image_path'],image);item['image_path']=str(image)
                subset['candidates'].append(item)
            _write_json_atomic(directory/'manifest.json',subset)
            return subset
        child=compile_demo(str(archived),str(staging/'segments'/segment_id),task,model,
            window_s=window_s,max_keyframes=max_keyframes,candidate_builder=builder,
            cache_dir=cache_dir,requery_budget=requery_budget,images_per_call=images_per_call,
            _single_segment=True,_shared_source_path=archived,
            _segment_context={'index':number,'count':len(chunks),
                              'time_range_s':[chunk[0]['timestamp_s'],chunk[-1]['timestamp_s']]})
        # Stage IDs are local model output; namespace them before flattening.
        for stage in child['stages']:stage['id']=segment_id+'/'+stage['id']
        child['segment_id']=segment_id
        child['time_range_s']=[chunk[0]['timestamp_s'],chunk[-1]['timestamp_s']]
        child['candidate_count']=len(chunk)
        child['boundary_policy']='Time/count partition; cross-boundary contact is unknown unless supported by each segment. Query original source when needed.'
        segments.append(child)
    verdicts=[s['outcome_verdict'] for s in segments]
    verdict=next((v for v in (UNRESOLVED,'refuted','unknown') if v in verdicts),'supported')
    core={'source_hash':manifest['source']['sha256'],'task':task,'segments':[s['demo_id'] for s in segments],
          'segmentation_version':'1.0','window_s':window_s,'max_keyframes_per_segment':max_keyframes}
    demo={'schema_version':1,'demo_id':'demo-'+hashlib.sha256(_canonical(core).encode()).hexdigest()[:16],
        'created_by':'piperlab.demonstration.compiler.compile_demo','source_path':str(archived),
        'source_hash':core['source_hash'],'task':task,'model_identity':_model_identity(model),
        'prompt_version':PROMPT_VERSION,'parameters':core,'segments':segments,
        'frames':[f for s in segments for f in s['frames']],
        'stages':[stage for s in segments for stage in s['stages']],
        'goal_constraints':[], 'unknowns':[f"{s['segment_id']}: {u}" for s in segments for u in s['unknowns']],
        'dropped_keyframes':[d for s in segments for d in s['dropped_keyframes']],
        'outcome_verdict':verdict,'outcome_scope':'historical_demo_only',
        'summary':f'{len(segments)} ordered video segments. Resolve each segment in sequence; do not infer cross-segment contact.',
        'requery':{'budget':requery_budget*len(segments),'used':sum(s['requery']['used'] for s in segments)},
        'model_calls':[c for s in segments for c in s['model_calls']],
        'candidate_manifest':'candidates/manifest.json'}
    from .paths import relocate_bundle
    demo=relocate_bundle(demo,staging,output)
    # Children and manifests remain independently inspectable after atomic rename.
    for number,child in enumerate(demo['segments']):
        _write_json_atomic(staging/'segments'/f'segment-{number:04d}'/'demo.json',child)
    for path in staging.rglob('manifest.json'):
        value=json.loads(path.read_text(encoding='utf-8'))
        _write_json_atomic(path,relocate_bundle(value,staging,output))
    _write_json_atomic(staging/'demo.json',demo)
    os.rename(staging,output)
    return demo
