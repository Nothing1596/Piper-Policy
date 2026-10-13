"""TaskPlanStore: durable observe-and-plan state, optimistic versioning, backend-owned answers."""
import json
import threading

import pytest

from piperx_middleware.models import DomainError
from piperx_middleware.task_plan import TASK_PLAN_TOOL, TaskPlanStore, guide


def q(qid, source, text="Where is the target?", query=None):
    out = {"id": qid, "question": text, "affects_action": True, "source": source}
    if query is not None:
        out["query"] = query
    return out


def codes(exc):
    return exc.value.code


class TestLifecycle:
    def test_create_persists_across_instances(self, tmp_path):
        first = TaskPlanStore(tmp_path)
        task = first.create("pick the red cube", demo="demo-0007")
        assert task["version"] == 1 and task["current_stage"] == "observe"
        reopened = TaskPlanStore(tmp_path)
        active = reopened.get()
        assert active["task_id"] == task["task_id"]
        assert active["goal"] == "pick the red cube" and active["demo_ref"] == "demo-0007"

    def test_create_preserves_original_task(self, tmp_path):
        store = TaskPlanStore(tmp_path)
        old = store.create("old goal")
        new = store.create("new goal")
        assert store.get()["task_id"] == new["task_id"]
        assert store.get(old["task_id"])["goal"] == "old goal"

    def test_get_without_active_task(self, tmp_path):
        with pytest.raises(DomainError) as exc:
            TaskPlanStore(tmp_path).get()
        assert codes(exc) == "task_not_found"

    def test_invalidate_current_only(self, tmp_path):
        store = TaskPlanStore(tmp_path)
        task = store.create("goal")
        done = store.invalidate("camera source changed")
        assert done["state"] == "invalidated" and done["invalidate_reason"] == "camera source changed"
        with pytest.raises(DomainError) as exc:
            store.get()
        assert codes(exc) == "task_not_found"
        assert store.get(task["task_id"])["state"] == "invalidated"
        with pytest.raises(DomainError) as exc:
            store.update(task["task_id"], done["version"], {"goal": "x"})
        assert codes(exc) == "task_invalidated"
        with pytest.raises(DomainError) as exc:
            store.invalidate("again")
        assert codes(exc) == "task_not_found"


class TestVersioning:
    def test_stale_version_rejected(self, tmp_path):
        store = TaskPlanStore(tmp_path)
        task = store.create("goal")
        store.update(task["task_id"], 1, {"interpretation": "mug left of laptop"})
        with pytest.raises(DomainError) as exc:
            store.update(task["task_id"], 1, {"goal": "other"})
        assert codes(exc) == "version_conflict"
        assert store.get()["goal"] == "goal"

    def test_record_result_requires_current_version(self, tmp_path):
        store = TaskPlanStore(tmp_path)
        task = store.create("goal")
        store.update(task["task_id"], 1, {"questions": [q("q1", "live")]})
        with pytest.raises(DomainError) as exc:
            store.record_result(task["task_id"], 1, "q1", {"value": [1, 2, 3]})
        assert codes(exc) == "version_conflict"
        assert store.get()["questions"][0]["state"] == "pending"

    def test_late_result_does_not_overwrite(self, tmp_path):
        store = TaskPlanStore(tmp_path)
        task = store.create("goal")
        store.update(task["task_id"], 1, {"questions": [q("q1", "live")]})
        store.record_result(task["task_id"], 2, "q1", {"value": {"x": 1}, "evidence": "frame-12"})
        with pytest.raises(DomainError) as exc:
            store.record_result(task["task_id"], 3, "q1", {"value": {"x": 2}})
        assert codes(exc) == "result_conflict"
        question = store.get()["questions"][0]
        assert question["result"] == {"x": 1} and question["evidence"] == "frame-12"

    def test_non_integer_version_rejected(self, tmp_path):
        store = TaskPlanStore(tmp_path)
        task = store.create("goal")
        with pytest.raises(DomainError) as exc:
            store.update(task["task_id"], True, {"goal": "x"})
        assert codes(exc) == "version_conflict"


class TestQuestions:
    def test_resume_stales_live_and_geometry_not_demo(self, tmp_path):
        store = TaskPlanStore(tmp_path)
        task = store.create("goal")
        task = store.update(task["task_id"], 1, {"questions": [q("d", "demo"), q("l", "live"), q("g", "geometry")]})
        for qid in ("d", "l", "g"):
            task = store.record_result(task["task_id"], task["version"], qid, {"value": 1})
        resumed = store.resume(task["task_id"])
        states = {x["id"]: x["state"] for x in resumed["questions"]}
        assert states == {"d": "answered", "l": "stale", "g": "stale"}
        again = store.record_result(task["task_id"], resumed["version"], "l", {"value": 2})
        assert {x["id"]: x["state"] for x in again["questions"]}["l"] == "answered"

    def test_changed_question_cannot_reuse_result(self, tmp_path):
        store = TaskPlanStore(tmp_path)
        task = store.create("goal")
        task = store.update(task["task_id"], 1, {"questions": [q("q1", "live"), q("q2", "live")]})
        task = store.record_result(task["task_id"], 2, "q1", {"value": 1})
        task = store.record_result(task["task_id"], 3, "q2", {"value": 2})
        changed = q("q1", "live", text="Where is the blue target?")
        task = store.update(task["task_id"], 4, {"questions": [changed, q("q2", "live")]})
        by_id = {x["id"]: x for x in task["questions"]}
        assert by_id["q1"]["state"] == "pending" and "result" not in by_id["q1"]
        assert by_id["q2"]["state"] == "answered" and by_id["q2"]["result"] == 2

    def test_changed_query_cannot_reuse_result(self, tmp_path):
        store = TaskPlanStore(tmp_path)
        task = store.create("goal")
        geo = q("g", "geometry", query={"operation": "surface_point", "pixel": [10, 20]})
        task = store.update(task["task_id"], 1, {"questions": [geo]})
        store.record_result(task["task_id"], 2, "g", {"value": True})
        moved = q("g", "geometry", query={"operation": "surface_point", "pixel": [40, 20]})
        task = store.update(task["task_id"], 3, {"questions": [moved]})
        assert task["questions"][0]["state"] == "pending" and "result" not in task["questions"][0]

    def test_unavailable_is_a_valid_result(self, tmp_path):
        store = TaskPlanStore(tmp_path)
        task = store.create("goal")
        store.update(task["task_id"], 1, {"questions": [q("q1", "live")]})
        task = store.record_result(task["task_id"], 2, "q1", {"status": "unavailable", "evidence": "no depth frame"})
        question = task["questions"][0]
        assert question["state"] == "unavailable" and "result" not in question
        with pytest.raises(DomainError) as exc:
            store.record_result(task["task_id"], 3, "q1", {"value": 1})
        assert codes(exc) == "result_conflict"


class TestValidation:
    def test_backend_owned_fields_rejected(self, tmp_path):
        store = TaskPlanStore(tmp_path)
        task = store.create("goal")
        for key in ("state", "result", "evidence"):
            bad = q("q1", "live")
            bad[key] = "answered" if key == "state" else {}
            with pytest.raises(DomainError) as exc:
                store.update(task["task_id"], 1, {"questions": [bad]})
            assert codes(exc) == "invalid_request"

    def test_task_fields_outside_allowlist_rejected(self, tmp_path):
        store = TaskPlanStore(tmp_path)
        task = store.create("goal")
        for changes in ({"version": 9}, {"state": "active"}, {"task_id": "x"}, {}):
            with pytest.raises(DomainError) as exc:
                store.update(task["task_id"], 1, changes)
            assert codes(exc) == "invalid_request"

    def test_invalid_task_ids_never_touch_paths(self, tmp_path):
        store = TaskPlanStore(tmp_path)
        store.create("goal")
        for bad in ("../events", "..", "not-a-uuid", str.upper("123e4567-e89b-12d3-a456-426614174000"), 123):
            with pytest.raises(DomainError) as exc:
                store.get(bad)
            assert codes(exc) == "invalid_task_id"
        assert [p.name for p in (tmp_path / "tasks").iterdir()] and not (tmp_path / "events").exists()

    def test_unsafe_question_ids_rejected(self, tmp_path):
        store = TaskPlanStore(tmp_path)
        task = store.create("goal")
        for bad in ("../x", "a/b", ".hidden", "x" * 65):
            with pytest.raises(DomainError) as exc:
                store.update(task["task_id"], 1, {"questions": [q(bad, "live")]})
            assert codes(exc) == "invalid_request"

    def test_non_finite_and_oversized_json_rejected(self, tmp_path):
        store = TaskPlanStore(tmp_path)
        task = store.create("goal")
        with pytest.raises(DomainError) as exc:
            store.update(task["task_id"], 1, {"questions": [q("g", "geometry", query={"operation": "accumulate_region", "voxel_size_m": float("nan")})]})
        assert codes(exc) == "invalid_request"
        with pytest.raises(DomainError) as exc:
            store.update(task["task_id"], 1, {"interpretation": "x" * 2001})
        assert codes(exc) == "invalid_request"

    def test_question_shape_enforced(self, tmp_path):
        store = TaskPlanStore(tmp_path)
        task = store.create("goal")
        bad = [q("q1", "nowhere"), q("q1", "live") | {"affects_action": 1}, q("q1", "live") | {"query": ["nope"]}]
        for questions in ([bad[0]], [bad[1]], [bad[2]], [q("dup", "live"), q("dup", "live")]):
            with pytest.raises(DomainError) as exc:
                store.update(task["task_id"], 1, {"questions": questions})
            assert codes(exc) == "invalid_request"


class TestConcurrency:
    def test_atomic_reads_under_parallel_updates(self, tmp_path):
        store_a = TaskPlanStore(tmp_path)
        store_b = TaskPlanStore(tmp_path)
        task = store_a.create("goal")
        tid = task["task_id"]
        failures = []

        def writer(store, n):
            done = 0
            while done < n:
                seen = store.get(tid)
                if seen["task_id"] != tid or seen["version"] < 1:
                    failures.append("torn read")
                    return
                try:
                    store.update(tid, seen["version"], {"interpretation": f"i{done}"})
                    done += 1
                except DomainError as exc:
                    if exc.code != "version_conflict":
                        failures.append(exc.code)
                        return

        threads = [threading.Thread(target=writer, args=(s, 25)) for s in (store_a, store_b)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not failures
        assert store_a.get(tid)["version"] == 51

    def test_leftover_tmp_file_does_not_corrupt_reads(self, tmp_path):
        store = TaskPlanStore(tmp_path)
        task = store.create("goal")
        tmp = tmp_path / "tasks" / task["task_id"] / "plan.json.tmp"
        tmp.write_text("{not json", encoding="utf-8")
        assert store.get()["goal"] == "goal"


class TestToolAndEvents:
    def test_tool_schema_is_strict_and_model_facing(self):
        from jsonschema import Draft202012Validator
        schema = TASK_PLAN_TOOL["inputSchema"]
        Draft202012Validator.check_schema(schema)
        assert schema["additionalProperties"] is False
        assert schema["required"] == ["operation"]
        assert set(schema["properties"]) == {"operation", "context", "task_id", "goal", "expected_version", "changes"}
        assert set(schema["properties"]["operation"]["enum"]) == {"guide", "get", "create", "update", "resume"}

    def test_guide_reads_packaged_skill(self):
        text = guide()
        assert "observe" in text.lower() and "task_plan" in text

    def test_offline_ablation_uses_identical_shared_skill(self):
        shared=guide(context='offline_demo',persistent=False)
        persistent=guide(context='offline_demo',persistent=True)
        assert persistent.startswith(shared+'\n\n')
        assert '## Live metric evidence' not in shared
        assert '## Plan persistence' not in shared
        assert 'RECORDED INITIAL SCENE' in shared
        assert 'remove resolved demo questions' in persistent

    def test_operation_required_fields_fail_before_dispatch(self):
        from jsonschema import validate,ValidationError
        for body in ({'operation':'create'},{'operation':'update','changes':{}},{'operation':'resume'}):
            with pytest.raises(ValidationError):validate(body,TASK_PLAN_TOOL['inputSchema'])
        validate({'operation':'guide','context':'offline_demo'},TASK_PLAN_TOOL['inputSchema'])

    def test_events_are_compact_identifiers_only(self, tmp_path):
        store = TaskPlanStore(tmp_path)
        task = store.create("secret goal text with token sk-abc123")
        store.update(task["task_id"], 1, {"questions": [q("q1", "live")]})
        store.record_result(task["task_id"], 2, "q1", {"value": {"x": 1}})
        with pytest.raises(DomainError):
            store.update(task["task_id"], 99, {"goal": "y"})
        lines = (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()
        events = [json.loads(line) for line in lines]
        assert [e["op"] for e in events] == ["create", "update", "record_result", "update"]
        assert events[-1]["ok"] is False and events[-1]["error"] == "version_conflict"
        for line in lines:
            assert "secret goal" not in line and "sk-abc123" not in line
        for event in events:
            assert set(event) <= {"at", "op", "ok", "task", "v", "error"}
