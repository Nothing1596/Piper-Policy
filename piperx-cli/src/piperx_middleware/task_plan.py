"""Durable observe-and-plan task state.

One active task per root. The model owns goal, stage, question text and a
concise semantic interpretation; the backend owns answer state, results and
evidence refs. Optimistic versioning: a writer that missed an update loses
and must re-read, never overwrites.
"""
import contextlib
import json
import os
from pathlib import Path
import re
import threading
import time
import uuid

from .models import DomainError

MAX_GOAL = 2000
MAX_QUESTION = 1000
MAX_INTERPRETATION = 2000
MAX_QUESTIONS = 32
MAX_REASON = 500
MAX_CHANGES_BYTES = 16384
MAX_QUERY_BYTES = 4096
MAX_RESULT_BYTES = 8192

QUESTION_SOURCES = ("demo", "live", "geometry")
RESULT_STATUSES = ("answered", "unavailable")

CHANGE_KEYS = frozenset({"goal", "current_stage", "interpretation", "questions"})
QUESTION_KEYS = frozenset({"id", "question", "affects_action", "source", "query"})
RESULT_KEYS = frozenset({"value", "evidence", "status"})
BACKEND_OWNED = frozenset({"state", "result", "evidence", "answered_at", "stale_at"})

_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z")
_STAGE_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}\Z")

TASK_PLAN_TOOL = {
    "name": "task_plan",
    "description": (
        "Track one observe-and-plan task: the goal, which decision-relevant "
        "questions remain, and which live/geometry answers went stale. Planning "
        "state only — it never measures, approves, or starts motion. "
        "create requires goal only; update requires task_id, expected_version and "
        "changes. Put questions/current_stage/interpretation inside changes. "
        "Use the returned task_id, never a session UUID. Omit query for visual questions."
    ),
    "inputSchema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["operation"],
        "allOf": [
            {"if": {"properties": {"operation": {"const": "create"}}}, "then": {"required": ["goal"]}},
            {"if": {"properties": {"operation": {"const": "update"}}}, "then": {"required": ["task_id", "expected_version", "changes"]}},
            {"if": {"properties": {"operation": {"const": "resume"}}}, "then": {"required": ["task_id"]}},
        ],
        "properties": {
            "operation": {"enum": ["guide", "get", "create", "update", "resume"]},
            "context": {"enum": ["live", "offline_demo"]},
            "task_id": {"type": "string", "minLength": 36, "maxLength": 36},
            "goal": {"type": "string", "minLength": 1, "maxLength": MAX_GOAL},
            "expected_version": {"type": "integer", "minimum": 1},
            "changes": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "goal": {"type": "string", "minLength": 1, "maxLength": MAX_GOAL},
                    "current_stage": {"type": "string", "minLength": 1, "maxLength": 64},
                    "interpretation": {"type": "string", "maxLength": MAX_INTERPRETATION},
                    "questions": {
                        "type": "array",
                        "maxItems": MAX_QUESTIONS,
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["id", "question", "affects_action", "source"],
                            "properties": {
                                "id": {"type": "string", "minLength": 1, "maxLength": 64},
                                "question": {"type": "string", "minLength": 1, "maxLength": MAX_QUESTION},
                                "affects_action": {"type": "boolean"},
                                "source": {"enum": list(QUESTION_SOURCES)},
                                "query": {"type": "object"},
                            },
                        },
                    },
                },
            },
        },
    },
}

_root_locks = {}
_root_locks_guard = threading.Lock()


def guide(context="live", persistent=True, *, camera_mode="standard"):
    """One skill source for native/MCP clients and controlled offline ablations."""
    if context not in ("live", "offline_demo"):
        raise ValueError("Unknown observation context")
    if camera_mode not in ("standard", "fixed_adapter"):
        raise ValueError("Unknown camera mode")
    source = (Path(__file__).parent / "skills" / "observe-and-plan" / "SKILL.md").read_text(encoding="utf-8")
    sections = {}
    for section in source.split("\n## ")[1:]:
        heading, body = section.split("\n", 1)
        sections[heading] = "## " + heading + "\n" + body.strip()
    names = ["Shared observation guidance", "Offline demonstration context" if context == "offline_demo" else "Live metric evidence"]
    if persistent:
        names.append("Plan persistence")
    text = "# observe-and-plan\n\n" + "\n\n".join(sections[name] for name in names)
    if camera_mode == "fixed_adapter":
        text = text.replace(
            "Read demo_reference(overview=true), then camera_observe",
            "Read the supplied recorded overview (or demo_watch at its supplied address), then camera_observe",
        )
        start = text.find("For a numeric question,")
        end = text.find("\n\nIf a view is insufficient", start)
        if start < 0 or end <= start:
            raise ValueError("Shared skill structure changed; review camera adaptation before freezing")
        text = text[:start] + text[end + 2:]
        if "demo_reference(" in text or "camera_observe(question_id" in text:
            raise ValueError("Unsupported skill syntax remains")
    return text


def _root_lock(root):
    key = str(root.resolve())
    with _root_locks_guard:
        return _root_locks.setdefault(key, threading.RLock())


def _text(value, field, lo, hi):
    if not isinstance(value, str) or not lo <= len(value) <= hi:
        raise DomainError("invalid_request", f"{field} must be a string of {lo}..{hi} characters.", 422)
    if any(ord(c) < 32 and c not in "\n\t" for c in value):
        raise DomainError("invalid_request", f"{field} contains control characters.", 422)
    return value


def _bounded_json(value, field, max_bytes):
    try:
        raw = json.dumps(value, allow_nan=False)
    except (TypeError, ValueError):
        raise DomainError("invalid_request", f"{field} must be finite JSON.", 422) from None
    if len(raw.encode("utf-8")) > max_bytes:
        raise DomainError("invalid_request", f"{field} exceeds {max_bytes} bytes.", 422)


def _task_id(value):
    if isinstance(value, str):
        try:
            if str(uuid.UUID(value)) == value:
                return value
        except ValueError:
            pass
    raise DomainError("invalid_task_id", "task_id must be a canonical lowercase UUID.", 422)


def _safe_id(value, field):
    if not isinstance(value, str) or not _ID_RE.match(value):
        raise DomainError("invalid_request", f"{field} must be 1..64 chars from [A-Za-z0-9._-], starting alphanumeric.", 422)
    return value


def _question(raw, index):
    field = f"questions[{index}]"
    if not isinstance(raw, dict):
        raise DomainError("invalid_request", f"{field} must be an object.", 422)
    unknown = set(raw) - QUESTION_KEYS
    if unknown & BACKEND_OWNED:
        raise DomainError("invalid_request", f"{field}: state/result/evidence are backend-owned.", 422)
    if unknown:
        raise DomainError("invalid_request", f"{field}: unknown keys {sorted(unknown)}.", 422)
    for required in ("id", "question", "affects_action", "source"):
        if required not in raw:
            raise DomainError("invalid_request", f"{field}.{required} is required.", 422)
    if not isinstance(raw["affects_action"], bool):
        raise DomainError("invalid_request", f"{field}.affects_action must be a boolean.", 422)
    if raw["source"] not in QUESTION_SOURCES:
        raise DomainError("invalid_request", f"{field}.source must be one of {list(QUESTION_SOURCES)}.", 422)
    q = {
        "id": _safe_id(raw["id"], f"{field}.id"),
        "question": _text(raw["question"], f"{field}.question", 1, MAX_QUESTION),
        "affects_action": raw["affects_action"],
        "source": raw["source"],
        "state": "pending",
    }
    query = raw.get("query")
    if query is not None:
        if not isinstance(query, dict) or not isinstance(query.get("operation"), str):
            raise DomainError("invalid_request", f"{field}.query must be an object with an operation string.", 422)
        _safe_id(query["operation"], f"{field}.query.operation")
        from .scene_bridge import SCENE_TOOL
        from jsonschema import validate, ValidationError
        try:
            validate(query, SCENE_TOOL['inputSchema'])
        except ValidationError as exc:
            raise DomainError('invalid_request', 'Unsupported geometry query: '+exc.message, 422) from exc
        _bounded_json(query, f"{field}.query", MAX_QUERY_BYTES)
        q["query"] = query
    return q


def _merge_question(new, prior):
    same = prior and all(prior.get(k) == new.get(k) for k in ("question", "affects_action", "source", "query"))
    if same:
        for key in ("state", "result", "evidence", "answered_at", "stale_at"):
            if key in prior:
                new[key] = prior[key]
    return new


class TaskPlanStore:
    def __init__(self, root):
        self.root = Path(root)
        (self.root / "tasks").mkdir(parents=True, exist_ok=True)
        self._lock = _root_lock(self.root)
        self._lock_file = (self.root / "task-plan.lock").open("a+b")
        self._lock_file.seek(0, 2)
        if not self._lock_file.tell():
            self._lock_file.write(b"0")
            self._lock_file.flush()

    def close(self):
        self._lock_file.close()

    @contextlib.contextmanager
    def _cross_process(self):
        # OS-owned byte lock: released on crash, never a stale PID file.
        deadline = time.monotonic() + 5.0
        while True:
            try:
                if os.name == "nt":
                    import msvcrt
                    self._lock_file.seek(0)
                    msvcrt.locking(self._lock_file.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise DomainError("store_busy", "Task plan store is busy.", 409) from None
                time.sleep(0.05)
        try:
            yield
        finally:
            try:
                if os.name == "nt":
                    import msvcrt
                    self._lock_file.seek(0)
                    msvcrt.locking(self._lock_file.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_UN)
            except OSError:
                pass

    def _run(self, op, work):
        with self._lock, self._cross_process():
            try:
                out = work()
            except DomainError as exc:
                self._event(op, ok=False, error=exc.code)
                raise
            self._event(op, ok=True, task_id=out.get("task_id"), version=out.get("version"))
            return out

    def _event(self, op, ok, task_id=None, version=None, error=None):
        # Compact identifiers/timing/status only: never payloads, keys, tokens.
        line = {"at": round(time.time(), 3), "op": op, "ok": ok}
        if task_id is not None:
            line["task"] = task_id
        if version is not None:
            line["v"] = version
        if error is not None:
            line["error"] = error
        try:
            with (self.root / "events.jsonl").open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(line, allow_nan=False, separators=(",", ":")) + "\n")
        except OSError:
            pass

    def _atomic_write(self, path, obj):
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(obj, indent=1, allow_nan=False) + "\n", encoding="utf-8")
        os.replace(tmp, path)

    def _write_plan(self, task):
        path = self.root / "tasks" / task["task_id"] / "plan.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        self._atomic_write(path, task)

    def _write_active(self, task_id):
        self._atomic_write(self.root / "active-task.json", {"task_id": task_id})

    def _read_plan(self, task_id):
        path = self.root / "tasks" / task_id / "plan.json"
        if not path.is_file():
            raise DomainError("task_not_found", f"No task {task_id}.", 404)
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            raise DomainError("store_corrupt", "Task plan file is unreadable.", 500) from None

    def _active_id(self):
        pointer = self.root / "active-task.json"
        if not pointer.is_file():
            return None
        try:
            return _task_id(json.loads(pointer.read_text(encoding="utf-8"))["task_id"])
        except (OSError, ValueError, KeyError, TypeError, DomainError):
            raise DomainError("store_corrupt", "Active task pointer is unreadable.", 500) from None

    def _read_for_write(self, task_id, expected_version):
        task = self._read_plan(_task_id(task_id))
        if task["state"] != "active":
            raise DomainError("task_invalidated", "Task was invalidated; create a new one.", 409)
        if self._active_id() != task_id:
            raise DomainError('task_not_active', 'Resume this task explicitly before updating it.', 409)
        if isinstance(expected_version, bool) or not isinstance(expected_version, int) or expected_version != task["version"]:
            raise DomainError(
                "version_conflict",
                f"Expected version {expected_version}, current is {task['version']}; re-read and retry.",
                409,
            )
        return task

    def _check_changes(self, changes):
        if not isinstance(changes, dict) or not changes:
            raise DomainError("invalid_request", "changes must be a non-empty object.", 422)
        unknown = set(changes) - CHANGE_KEYS
        if unknown:
            raise DomainError("invalid_request", f"changes may only touch {sorted(CHANGE_KEYS)}; got {sorted(unknown)}.", 422)
        _bounded_json(changes, "changes", MAX_CHANGES_BYTES)
        out = {}
        if "goal" in changes:
            out["goal"] = _text(changes["goal"], "goal", 1, MAX_GOAL)
        if "current_stage" in changes:
            stage = changes["current_stage"]
            if not isinstance(stage, str) or not _STAGE_RE.match(stage):
                raise DomainError("invalid_request", "current_stage must be 1..64 chars from [A-Za-z0-9_-].", 422)
            out["current_stage"] = stage
        if "interpretation" in changes:
            out["interpretation"] = _text(changes["interpretation"], "interpretation", 0, MAX_INTERPRETATION)
        if "questions" in changes:
            raw = changes["questions"]
            if not isinstance(raw, list) or len(raw) > MAX_QUESTIONS:
                raise DomainError("invalid_request", f"questions must be a list of at most {MAX_QUESTIONS}.", 422)
            checked = [_question(q, i) for i, q in enumerate(raw)]
            if len({q["id"] for q in checked}) != len(checked):
                raise DomainError("invalid_request", "question ids must be unique.", 422)
            out["questions"] = checked
        return out

    def _check_result(self, result):
        if not isinstance(result, dict):
            raise DomainError("invalid_request", "result must be an object.", 422)
        unknown = set(result) - RESULT_KEYS
        if unknown:
            raise DomainError("invalid_request", f"result may only contain {sorted(RESULT_KEYS)}; got {sorted(unknown)}.", 422)
        _bounded_json(result, "result", MAX_RESULT_BYTES)
        status = result.get("status", "answered")
        if status not in RESULT_STATUSES:
            raise DomainError("invalid_request", f"result.status must be one of {list(RESULT_STATUSES)}.", 422)
        evidence = result.get("evidence")
        return {
            "status": status,
            "value": result.get("value"),
            "evidence": _text(evidence, "result.evidence", 1, 256) if evidence is not None else None,
        }

    def create(self, goal, demo=None):
        if isinstance(demo, dict):
            if set(demo) - {'artifact','artifact_sha256','source_sha256'}:
                raise DomainError('invalid_request', 'Unsupported demonstration reference fields', 422)
            _bounded_json(demo, 'demo', 4096)
            demo = dict(demo)
        elif demo is not None:
            demo = _text(demo, 'demo', 1, 256)
        def work():
            now = time.time()
            task = {
                "task_id": str(uuid.uuid4()),
                "version": 1,
                "state": "active",
                "goal": _text(goal, "goal", 1, MAX_GOAL),
                "demo_ref": demo,
                "current_stage": "observe",
                "interpretation": "",
                "questions": [],
                "created_at": now,
                "updated_at": now,
            }
            self._write_plan(task)
            self._write_active(task["task_id"])
            return task
        return self._run("create", work)

    def get(self, task_id=None):
        def work():
            tid = _task_id(task_id) if task_id is not None else self._active_id()
            if tid is None:
                raise DomainError("task_not_found", "No active task.", 404)
            return self._read_plan(tid)
        return self._run("get", work)

    def update(self, task_id, expected_version, changes):
        def work():
            task = self._read_for_write(task_id, expected_version)
            checked = self._check_changes(changes)
            if "goal" in checked:
                if task['goal'] != checked['goal']:
                    for question in task['questions']:
                        if question['state'] in RESULT_STATUSES:
                            question['state'] = 'stale'
                            question['stale_at'] = time.time()
                    task['interpretation'] = ''
                task["goal"] = checked["goal"]
            if "current_stage" in checked:
                task["current_stage"] = checked["current_stage"]
            if "interpretation" in checked:
                task["interpretation"] = checked["interpretation"]
            if "questions" in checked:
                prior = {q["id"]: q for q in task["questions"]}
                task["questions"] = [_merge_question(q, prior.get(q["id"])) for q in checked["questions"]]
            task["version"] += 1
            task["updated_at"] = time.time()
            self._write_plan(task)
            return task
        return self._run("update", work)

    def resume(self, task_id):
        def work():
            task = self._read_plan(_task_id(task_id))
            if task["state"] != "active":
                raise DomainError("task_invalidated", "Task was invalidated; create a new one.", 409)
            now = time.time()
            for q in task["questions"]:
                if q["source"] in ("live", "geometry") and q["state"] in ("answered", "unavailable"):
                    q["state"] = "stale"
                    q["stale_at"] = now
            task["version"] += 1
            task["updated_at"] = now
            self._write_plan(task)
            self._write_active(task["task_id"])
            return task
        return self._run("resume", work)

    def stale_questions(self, task_id, expected_version, question_ids):
        """Expire backend evidence without changing the selected task."""
        def work():
            task = self._read_for_write(task_id, expected_version)
            now = time.time()
            for question in task['questions']:
                if question['id'] in question_ids and question['state'] == 'answered':
                    question.update(state='stale', stale_at=now)
            task['version'] += 1
            task['updated_at'] = now
            self._write_plan(task)
            return task
        return self._run('stale_questions', work)

    def record_result(self, task_id, expected_version, question_id, result):
        def work():
            task = self._read_for_write(task_id, expected_version)
            qid = _safe_id(question_id, "question_id")
            question = next((q for q in task["questions"] if q["id"] == qid), None)
            if question is None:
                raise DomainError("question_not_found", f"No question {qid}.", 404)
            payload = self._check_result(result)
            if question["state"] in RESULT_STATUSES:
                raise DomainError("result_conflict", "Question already has a result; refusing to overwrite.", 409)
            now = time.time()
            question["state"] = payload["status"]
            for key in ('result', 'answered_at', 'evidence'):
                question.pop(key, None)
            if payload["value"] is not None:
                question["result"] = payload["value"]
            if payload["status"] == "answered":
                question["answered_at"] = now
            if payload["evidence"] is not None:
                question["evidence"] = payload["evidence"]
            question.pop("stale_at", None)
            task["version"] += 1
            task["updated_at"] = now
            self._write_plan(task)
            return task
        return self._run("record_result", work)

    def invalidate(self, reason):
        def work():
            tid = self._active_id()
            if tid is None:
                raise DomainError("task_not_found", "No active task to invalidate.", 404)
            task = self._read_plan(tid)
            task["state"] = "invalidated"
            task["invalidate_reason"] = _text(reason, "reason", 1, MAX_REASON)
            task["version"] += 1
            task["updated_at"] = time.time()
            self._write_plan(task)
            (self.root / "active-task.json").unlink(missing_ok=True)
            return task
        return self._run("invalidate", work)
