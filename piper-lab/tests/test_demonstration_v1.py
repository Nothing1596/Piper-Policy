"""Demonstration compiler + store tests (human-video v1).

Everything here is synthetic: the "video" is a few bytes in a tmp file (or,
for the end-to-end class, a generated MPEG-4), the candidate builder is an
in-test fixture that writes PIL-generated JPEGs, and the model is FakeModel,
an in-process test double. FakeModel.identity is self-reported and
deliberately flagged as not transport-level proof; the compiler must record
it with identity_verified=False.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import re
from fractions import Fraction
from pathlib import Path

import pytest
from PIL import Image

from piperlab.demonstration import CompileError, DemoStore, StoreError, compile_demo

TASK = "pick the red block and place it on the blue plate (synthetic task text)"

FAKE_IDENTITY = {
    "name": "fake-vlm-v1",
    "provider": "test-double",
    "transport": "in-process",
    "note": "self-reported test identity; not transport proof",
}


# ---------------------------------------------------------------------------
# fixtures: synthetic candidate builder + FakeModel


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_fake_builder(frame_count: int = 12, hz: float = 2.0):
    """A candidate-builder stand-in writing PIL JPEGs + a valid manifest.

    Keeps demonstration tests independent of PyAV/OpenCV; the real builder is
    covered in tests/test_video_v1.py. Frames are solid-colour squares, i.e.
    synthetic provenance end to end.
    """

    def build(video_path: str, output_dir: str, window_s: float = 10, **kwargs) -> dict:
        output = Path(output_dir).absolute()
        if output.exists():
            raise FileExistsError(f"output already exists, refusing to overwrite: {output}")
        frames_dir = output / "frames"
        frames_dir.mkdir(parents=True)
        source = Path(video_path).absolute()
        candidates = []
        for index in range(frame_count):
            frame_id = f"f{index:06d}"
            image_path = frames_dir / f"{frame_id}.jpg"
            Image.new("RGB", (16, 12), ((index * 20) % 255, 10, 200)).save(
                image_path, format="JPEG"
            )
            candidates.append({
                "frame_id": frame_id,
                "sequence": index,
                "pts": index * int(1000 / hz),
                "time_base": "1/1000",
                "timestamp_s": index / hz,
                "image_path": str(image_path.absolute()),
                "scores": {"motion": 0.1, "clarity": 5.0, "camera": None},
                "detections": (
                    {"count": 1, "labels": ["fake_blob"], "max_score": 0.5,
                     "truncated_by_scorer": 0}
                    if index % 3 == 0 else None
                ),
                "reasons": ["baseline"],
                "window": int((index / hz) // window_s),
            })
        manifest = {
            "schema_version": 1,
            "kind": "piperlab.video.candidates",
            "created_by": "tests.test_demonstration_v1.make_fake_builder",
            "provenance": "synthetic test fixture (PIL frames); not real footage",
            "source": {"path": str(source), "sha256": _sha256(source)},
            "parameters": {"window_s": window_s, "analysis_hz": hz},
            "candidates": candidates,
            "dropped": [],
        }
        with open(output / "manifest.json", "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
        return manifest

    return build


class FakeModel:
    """In-process test double for the model protocol.

    ``select_fn(batch_ids, call_index)`` and ``semantic_fn(frame_ids,
    call_index)`` return raw responses; the compiler must validate them.
    """

    def __init__(self, select_fn=None, semantic_fn=None):
        self.calls: list[dict] = []
        self._select_fn = select_fn or self._default_select
        self._semantic_fn = semantic_fn or self._default_semantic

    @property
    def identity(self) -> dict:
        return dict(FAKE_IDENTITY)

    @staticmethod
    def _default_select(batch_ids, _call_index):
        return {
            "selections": [
                {
                    "frame_id": fid,
                    "keep": i % 2 == 0,
                    "reason": f"synthetic stage boundary near {fid}",
                    "operation": "move",
                    "object_roles": ["red_block"],
                    "visible_outcome": "block grasped",
                }
                for i, fid in enumerate(batch_ids)
            ]
        }

    @staticmethod
    def _default_semantic(frame_ids, _call_index):
        middle = frame_ids[len(frame_ids) // 2]
        return {
            "stages": [
                {
                    "id": "stage-1",
                    "operation": "pick",
                    "object_roles": ["red_block"],
                    "preconditions": ["gripper open above block"],
                    "expected_effects": ["red_block grasped"],
                    "evidence_refs": frame_ids[: max(1, len(frame_ids) // 2 + 1)],
                    "uncertainty": "synthetic fixture; no real contact",
                },
                {
                    "id": "stage-2",
                    "operation": "place",
                    "object_roles": ["red_block", "blue_plate"],
                    "preconditions": ["red_block grasped"],
                    "expected_effects": ["red_block on blue_plate"],
                    "evidence_refs": [middle, frame_ids[-1]],
                    "uncertainty": "",
                },
            ],
            "goal_constraints": ["red_block ends on blue_plate"],
            "unknowns": [],
            "outcome_verdict": "supported",
            "summary": "synthetic two-stage pick/place",
        }

    def infer(self, prompt: str, images: list, schema: dict) -> dict:
        assert all(Path(p).is_file() for p in images), 'Every model image must exist at inference time'
        call_index = sum(
            1 for c in self.calls
            if ("selections" in c["schema"]["properties"]) == ("selections" in schema["properties"])
        )
        self.calls.append({
            "prompt": prompt,
            "images": [str(p) for p in images],
            "schema": schema,
        })
        if "selections" in schema["properties"]:
            body = prompt.split("selection hints):\n", 1)[1].split("\n\n", 1)[0]
            batch_ids = [entry["frame_id"] for entry in json.loads(body)]
            return self._select_fn(batch_ids, call_index)
        body = prompt.split("selection notes):\n", 1)[1].split("\n\n", 1)[0]
        frame_ids = [entry["frame_id"] for entry in json.loads(body)]
        return self._semantic_fn(frame_ids, call_index)

    @property
    def select_calls(self) -> list[dict]:
        return [c for c in self.calls if "selections" in c["schema"]["properties"]]

    @property
    def semantic_calls(self) -> list[dict]:
        return [c for c in self.calls if "selections" not in c["schema"]["properties"]]


@pytest.fixture
def synthetic_video(tmp_path) -> Path:
    video = tmp_path / "synthetic-demo.mp4"
    video.write_bytes(b"synthetic demonstration placeholder; not real footage")
    return video


def _compile(video, output, model, builder=None, task=TASK, **kwargs):
    return compile_demo(
        str(video), str(output), task, model,
        candidate_builder=builder or make_fake_builder(), **kwargs,
    )


# ---------------------------------------------------------------------------
# compiler


class TestCompileHappyPath:
    def test_demo_json_contract(self, tmp_path, synthetic_video):
        model = FakeModel()
        demo = _compile(synthetic_video, tmp_path / "bundle", model)
        json.dumps(demo)  # returned report must be JSON-serializable
        on_disk = json.loads((tmp_path / "bundle" / "demo.json").read_text("utf-8"))
        assert on_disk == demo

        assert demo["schema_version"] == 1
        assert demo["demo_id"].startswith("demo-")
        assert demo["source_hash"] == _sha256(synthetic_video)
        assert Path(demo['source_path']).parent == (tmp_path/'bundle').absolute()
        assert _sha256(Path(demo['source_path'])) == _sha256(synthetic_video)
        assert demo["task"] == TASK
        assert demo["prompt_version"]
        assert demo["outcome_scope"] == "historical_demo_only"

        identity = demo["model_identity"]
        assert identity["name"] == "fake-vlm-v1"
        assert identity["identity_verified"] is False
        assert "not transport-level proof" in identity["identity_note"]

        frames = demo["frames"]
        timestamps = [f["timestamp_s"] for f in frames]
        assert timestamps == sorted(timestamps), "frames must be chronological"
        candidate_ids = {f"f{i:06d}" for i in range(12)}
        assert {f["frame_id"] for f in frames} <= candidate_ids
        assert frames[0]["frame_id"] == "f000000"  # initial always present
        assert frames[-1]["frame_id"] == "f000011"  # final always present
        for frame in frames:
            assert Path(frame["image_path"]).is_file()

        assert [s["id"] for s in demo["stages"]] == ["stage-1", "stage-2"]
        selected_ids = {f["frame_id"] for f in frames}
        for stage in demo["stages"]:
            assert set(stage["evidence_refs"]) <= selected_ids
            refs_ts = [demo["frames"][next(
                i for i, f in enumerate(demo["frames"]) if f["frame_id"] == r
            )]["timestamp_s"] for r in stage["evidence_refs"]]
            assert refs_ts == sorted(refs_ts)
        assert demo["outcome_verdict"] == "supported"
        assert demo["goal_constraints"] == ["red_block ends on blue_plate"]

    def test_images_per_call_never_exceeds_six(self, tmp_path, synthetic_video):
        model = FakeModel()
        _compile(synthetic_video, tmp_path / "bundle", model)
        assert model.calls, "expected model calls"
        assert all(len(c["images"]) <= 6 for c in model.calls)

    def test_detection_summaries_reach_selection_prompt(self, tmp_path, synthetic_video):
        model = FakeModel()
        _compile(synthetic_video, tmp_path / "bundle", model)
        first_select = model.select_calls[0]
        body = first_select["prompt"].split("selection hints):\n", 1)[1].split("\n\n", 1)[0]
        brief = json.loads(body)
        assert any(entry["detections"] for entry in brief)
        fake_entry = next(entry for entry in brief if entry["detections"])
        assert fake_entry["detections"]["labels"] == ["fake_blob"]

    def test_semantic_prompt_maps_images_to_frame_ids(self, tmp_path, synthetic_video):
        model = FakeModel()
        demo = _compile(synthetic_video, tmp_path / "bundle", model)
        prompt = model.semantic_calls[0]["prompt"]
        images_section = prompt.split("direct visual evidence):", 1)[1].split("\n\n", 1)[0]
        mapped_ids = re.findall(r"frame_id=(f\d+)", images_section)
        n_images = len(model.semantic_calls[0]["images"])
        assert 0 < len(mapped_ids) == n_images <= 6
        support = demo["semantic_evidence"]
        assert support["images_supplied"] == mapped_ids
        frame_ids = {f["frame_id"] for f in demo["frames"]}
        assert set(support["images_supplied"]) <= frame_ids
        assert not (set(support["images_supplied"]) & set(support["text_only"]))
        # text-only frames are explicitly listed as second-hand in the prompt
        assert "second-hand" in prompt
        for fid in support["text_only"]:
            assert f'"frame_id": "{fid}"' in prompt.split("second-hand", 1)[1]

    def test_output_dir_refuses_overwrite(self, tmp_path, synthetic_video):
        model = FakeModel()
        _compile(synthetic_video, tmp_path / "bundle", model)
        with pytest.raises(FileExistsError, match="refusing to overwrite"):
            _compile(synthetic_video, tmp_path / "bundle", FakeModel())


class TestValidationAndRequery:
    def test_fabricated_frame_ids_never_persisted(self, tmp_path, synthetic_video):
        def bad_select(batch_ids, _call_index):
            return {
                "selections": [
                    {"frame_id": "f999999", "keep": True, "reason": "hallucinated"},
                    {"frame_id": batch_ids[0], "keep": True, "reason": "real one"},
                ]
            }

        model = FakeModel(select_fn=bad_select)
        demo = _compile(synthetic_video, tmp_path / "bundle", model, requery_budget=2)
        frame_ids = {f["frame_id"] for f in demo["frames"]}
        assert "f999999" not in frame_ids
        assert "f000000" in frame_ids and "f000011" in frame_ids  # boundaries
        # both batches exhausted the shared re-query budget; gaps are recorded
        assert demo["requery"]["used"] == 2
        batch_unknowns = [u for u in demo["unknowns"] if "selection batch" in u]
        assert len(batch_unknowns) == 2
        # 3 calls for batch 1 (initial + 2 retries), 1 for batch 2, 1 semantic
        assert len(model.calls) == 5
        # the semantic pass still validated over real evidence only
        assert demo["outcome_verdict"] in ("supported", "unresolved")
        all_refs = {r for s in demo["stages"] for r in s["evidence_refs"]}
        assert all_refs <= frame_ids

    def test_out_of_enum_verdict_is_not_silently_coerced(self, tmp_path, synthetic_video):
        def bad_semantic(_frame_ids, _call_index):
            return {
                "stages": [], "goal_constraints": [], "unknowns": [],
                "outcome_verdict": "definitely_worked", "summary": "bogus",
            }

        model = FakeModel(semantic_fn=bad_semantic)
        demo = _compile(synthetic_video, tmp_path / "bundle", model, requery_budget=1)
        assert demo["outcome_verdict"] == "unresolved"
        assert demo["stages"] == []
        assert any("semantic pass" in u for u in demo["unknowns"])

    def test_unsupplied_evidence_refs_rejected(self, tmp_path, synthetic_video):
        def bad_semantic(_frame_ids, _call_index):
            return {
                "stages": [{
                    "id": "s1", "operation": "pick", "object_roles": [],
                    "preconditions": [], "expected_effects": [],
                    "evidence_refs": ["f424242"], "uncertainty": "",
                }],
                "goal_constraints": [], "unknowns": [],
                "outcome_verdict": "supported", "summary": "bogus refs",
            }

        model = FakeModel(semantic_fn=bad_semantic)
        demo = _compile(synthetic_video, tmp_path / "bundle", model, requery_budget=0)
        assert demo["outcome_verdict"] == "unresolved"
        assert demo["stages"] == []
        all_refs = {r for s in demo["stages"] for r in s["evidence_refs"]}
        assert "f424242" not in all_refs

    def test_non_chronological_stage_evidence_rejected(self, tmp_path, synthetic_video):
        def bad_semantic(frame_ids, _call_index):
            return {
                "stages": [{
                    "id": "s1", "operation": "pick", "object_roles": [],
                    "preconditions": [], "expected_effects": [],
                    "evidence_refs": [frame_ids[-1], frame_ids[0]], "uncertainty": "",
                }],
                "goal_constraints": [], "unknowns": [],
                "outcome_verdict": "supported", "summary": "out of order",
            }

        model = FakeModel(semantic_fn=bad_semantic)
        demo = _compile(synthetic_video, tmp_path / "bundle", model, requery_budget=0)
        assert demo["outcome_verdict"] == "unresolved"

    def test_requery_recovers_after_one_bad_reply(self, tmp_path, synthetic_video):
        def flaky_semantic(frame_ids, call_index):
            if call_index == 0:
                return {"nonsense": True}
            return FakeModel._default_semantic(frame_ids, call_index)

        model = FakeModel(semantic_fn=flaky_semantic)
        demo = _compile(synthetic_video, tmp_path / "bundle", model, requery_budget=3)
        assert demo["outcome_verdict"] == "supported"
        assert demo["requery"]["used"] == 1

    def test_model_exception_counts_against_budget(self, tmp_path, synthetic_video):
        class ExplodingModel(FakeModel):
            def infer(self, prompt, images, schema):
                self.calls.append({"prompt": prompt, "images": list(images), "schema": schema})
                raise ConnectionError("synthetic transport failure")

        demo = _compile(synthetic_video, tmp_path / "bundle", ExplodingModel(),
                        requery_budget=1)
        assert demo["outcome_verdict"] == "unresolved"
        assert demo["requery"]["used"] == 1
        assert any("ConnectionError" in u or "unresolved" in u for u in demo["unknowns"])
        assert any(c["error"] for c in demo["model_calls"])


class TestKeyframeBudget:
    def test_trim_is_explicit(self, tmp_path, synthetic_video):
        def keep_all(batch_ids, _call_index):
            return {
                "selections": [
                    {"frame_id": fid, "keep": True, "reason": "keep everything"}
                    for fid in batch_ids
                ]
            }

        def sparse_semantic(frame_ids, _call_index):
            return {
                "stages": [{
                    "id": "s1", "operation": "pick", "object_roles": [],
                    "preconditions": [], "expected_effects": [],
                    "evidence_refs": [frame_ids[0], frame_ids[-1]], "uncertainty": "",
                }],
                "goal_constraints": [], "unknowns": [],
                "outcome_verdict": "unknown", "summary": "sparse evidence",
            }

        model = FakeModel(select_fn=keep_all, semantic_fn=sparse_semantic)
        demo = _compile(synthetic_video, tmp_path / "bundle", model, max_keyframes=6)
        assert len(demo["frames"]) == 6
        assert demo["frames"][0]["frame_id"] == "f000000"
        assert demo["frames"][-1]["frame_id"] == "f000011"
        dropped = demo["dropped_keyframes"]
        assert len(dropped) == 6
        assert all(d["reason"] == "max_keyframes" for d in dropped)
        kept = {f["frame_id"] for f in demo["frames"]}
        assert not (kept & {d["frame_id"] for d in dropped})
        # evidence-referenced frames survive trimming
        assert {"f000000", "f000011"} <= kept

    def test_evidence_frames_are_never_trimmed_and_overflow_is_reported(
        self, tmp_path, synthetic_video
    ):
        def keep_all(batch_ids, _call_index):
            return {
                "selections": [
                    {"frame_id": fid, "keep": True, "reason": "keep"}
                    for fid in batch_ids
                ]
            }

        def dense_semantic(frame_ids, _call_index):
            return {
                "stages": [{
                    "id": "s1", "operation": "slide", "object_roles": [],
                    "preconditions": [], "expected_effects": [],
                    "evidence_refs": list(frame_ids), "uncertainty": "",
                }],
                "goal_constraints": [], "unknowns": [],
                "outcome_verdict": "unknown", "summary": "everything is evidence",
            }

        model = FakeModel(select_fn=keep_all, semantic_fn=dense_semantic)
        demo = _compile(synthetic_video, tmp_path / "bundle", model, max_keyframes=4)
        assert len(demo["frames"]) > 4, "evidence must not be trimmed to fit"
        assert demo["dropped_keyframes"] == []
        assert any("keyframe budget exceeded" in u for u in demo["unknowns"])


class TestCacheIdentity:
    def test_hit_only_on_exact_identity_match(self, tmp_path, synthetic_video):
        cache = tmp_path / "cache"
        first_model = FakeModel()
        first = _compile(synthetic_video, tmp_path / "b1", first_model, cache_dir=str(cache))
        assert first["cache"]["status"] == "miss"
        assert first["cache"]["store"] == "written"
        calls_first = len(first_model.calls)
        assert calls_first > 0

        second_model = FakeModel()
        second = _compile(synthetic_video, tmp_path / "b2", second_model, cache_dir=str(cache))
        assert second["cache"]["status"] == "hit"
        assert len(second_model.calls) == 0, "cache hit must not call the model"
        assert second["demo_id"] == first["demo_id"]

        other_task = _compile(
            synthetic_video, tmp_path / "b3", FakeModel(),
            cache_dir=str(cache), task="a different synthetic task",
        )
        assert other_task["cache"]["status"] == "miss"

        class OtherModel(FakeModel):
            @property
            def identity(self):
                return {**FAKE_IDENTITY, "name": "fake-vlm-v2"}

        other_model = _compile(synthetic_video, tmp_path / "b4", OtherModel(),
                               cache_dir=str(cache))
        assert other_model["cache"]["status"] == "miss"

    def test_unresolved_transcripts_are_not_cached(self, tmp_path, synthetic_video):
        cache = tmp_path / "cache"

        def bad_semantic(_frame_ids, _call_index):
            return {"broken": True}

        demo = _compile(synthetic_video, tmp_path / "b1",
                        FakeModel(semantic_fn=bad_semantic),
                        cache_dir=str(cache), requery_budget=0)
        assert demo["outcome_verdict"] == "unresolved"
        assert demo["cache"]["store"] == "skipped_unresolved"
        assert not cache.exists() or not list(cache.glob("*.json"))


class TestCompilerInputValidation:
    def test_empty_task_rejected(self, tmp_path, synthetic_video):
        with pytest.raises(CompileError, match="task"):
            _compile(synthetic_video, tmp_path / "bundle", FakeModel(), task=" ")

    def test_missing_model_identity_rejected(self, tmp_path, synthetic_video):
        class NoIdentity(FakeModel):
            identity = None

        with pytest.raises(CompileError, match="identity"):
            _compile(synthetic_video, tmp_path / "bundle", NoIdentity())

    def test_missing_video_rejected(self, tmp_path):
        with pytest.raises(CompileError, match="does not exist"):
            compile_demo(str(tmp_path / "nope.mp4"), str(tmp_path / "bundle"),
                         TASK, FakeModel(), candidate_builder=make_fake_builder())

    def test_bogus_manifest_rejected(self, tmp_path, synthetic_video):
        def bad_builder(video_path, output_dir, **kwargs):
            return {"source": {"path": video_path, "sha256": "x"}, "candidates": []}

        with pytest.raises(CompileError, match="no candidates"):
            _compile(synthetic_video, tmp_path / "bundle", FakeModel(),
                     builder=bad_builder)


# ---------------------------------------------------------------------------
# store


def _registered_bundle(tmp_path, synthetic_video, task=TASK, model=None) -> tuple[Path, dict]:
    bundle = tmp_path / f"bundle-{len(list(tmp_path.glob('bundle-*')))}"
    demo = _compile(synthetic_video, bundle, model or FakeModel(), task=task)
    return bundle, demo


class TestDemoStore:
    def test_register_get_roundtrip(self, tmp_path, synthetic_video):
        bundle, demo = _registered_bundle(tmp_path, synthetic_video)
        with DemoStore(tmp_path / "store") as store:
            demo_id = store.register(bundle)
            assert demo_id == demo["demo_id"]
            loaded = store.get(demo_id)
            assert loaded["demo_id"] == demo_id
            assert loaded["task"] == TASK
            assert loaded["source_hash"] == _sha256(synthetic_video)

    def test_register_rewrites_image_paths_into_store(self, tmp_path, synthetic_video):
        bundle, _demo = _registered_bundle(tmp_path, synthetic_video)
        with DemoStore(tmp_path / "store") as store:
            demo_id = store.register(bundle)
            loaded = store.get(demo_id)
            for frame in loaded["frames"]:
                path = Path(frame["image_path"])
                assert path.is_file(), "stored bundle must be self-contained"
                assert str(tmp_path / "store" / "bundles") in str(path)

    def test_register_is_idempotent_for_identical_content(self, tmp_path, synthetic_video):
        bundle, demo = _registered_bundle(tmp_path, synthetic_video)
        with DemoStore(tmp_path / "store") as store:
            assert store.register(bundle) == demo["demo_id"]
            assert store.register(bundle) == demo["demo_id"]

    def test_conflicting_demo_id_rejected(self, tmp_path, synthetic_video):
        bundle, demo = _registered_bundle(tmp_path, synthetic_video)
        forged = tmp_path / "forged"
        forged.mkdir()
        forged_demo = dict(demo)
        forged_demo["task"] = "forged different task under the same demo_id"
        (forged / "demo.json").write_text(json.dumps(forged_demo), encoding="utf-8")
        with DemoStore(tmp_path / "store") as store:
            store.register(bundle)
            with pytest.raises(StoreError, match="different content"):
                store.register(forged)

    def test_get_unknown_id_raises_keyerror(self, tmp_path):
        with DemoStore(tmp_path / "store") as store:
            with pytest.raises(KeyError, match="unknown demo_id"):
                store.get("demo-0000000000000000")

    def test_find_by_task_text_and_limit(self, tmp_path, synthetic_video):
        def push_semantic(frame_ids, _call_index):
            return {
                "stages": [{
                    "id": "s1", "operation": "push", "object_roles": ["green_cup"],
                    "preconditions": ["gripper behind cup"],
                    "expected_effects": ["cup moved left"],
                    "evidence_refs": [frame_ids[0], frame_ids[-1]], "uncertainty": "",
                }],
                "goal_constraints": ["green_cup moved left"], "unknowns": [],
                "outcome_verdict": "supported", "summary": "synthetic push",
            }

        bundle_a, _ = _registered_bundle(tmp_path, synthetic_video)
        bundle_b, _ = _registered_bundle(
            tmp_path, synthetic_video,
            task="push the green cup left across the table (synthetic)",
            model=FakeModel(semantic_fn=push_semantic),
        )
        with DemoStore(tmp_path / "store") as store:
            id_a = store.register(bundle_a)
            id_b = store.register(bundle_b)
            hits = store.find("place red block on blue plate")
            assert [h["demo_id"] for h in hits] == [id_a]
            hits = store.find("push green cup")
            assert [h["demo_id"] for h in hits] == [id_b]
            both = store.find("the synthetic task", limit=5)
            assert {h["demo_id"] for h in both} == {id_a, id_b}
            assert len(store.find("the synthetic task", limit=1)) == 1
            assert store.find("unrelated welding procedure") == []

    def test_register_rejects_malformed_bundle(self, tmp_path):
        bogus = tmp_path / "bogus"
        bogus.mkdir()
        (bogus / "demo.json").write_text(json.dumps({"schema_version": 99}), "utf-8")
        with DemoStore(tmp_path / "store") as store:
            with pytest.raises(StoreError):
                store.register(bogus)

    def test_find_chinese_substring_in_mixed_language_task(self, tmp_path, synthetic_video):
        bundle, _ = _registered_bundle(
            tmp_path, synthetic_video, task="按顺序摆放红色方块 Piper demo"
        )
        with DemoStore(tmp_path / "store") as store:
            demo_id = store.register(bundle)
            assert [hit["demo_id"] for hit in store.find("摆放")] == [demo_id]
            assert [hit["demo_id"] for hit in store.find("顺序摆放 Piper")] == [demo_id]
            assert store.find("焊接钢管") == []


# ---------------------------------------------------------------------------
# end-to-end with the real candidate builder (needs the video extras)

HAVE_VIDEO_EXTRAS = (
    importlib.util.find_spec("av") is not None
    and importlib.util.find_spec("cv2") is not None
)
needs_video = pytest.mark.skipif(
    not HAVE_VIDEO_EXTRAS, reason="PyAV/OpenCV not installed (video extras)"
)


def _write_real_video(path: Path, *, frames: int = 40, fps: int = 10) -> Path:
    """Generated MPEG-4: static scene with a moving bright square burst.

    Fully synthetic fixture; no real footage anywhere in this test suite.
    """
    import av  # local: only called from the gated end-to-end test
    import numpy as np

    container = av.open(str(path), "w")
    stream = container.add_stream("mpeg4", rate=Fraction(fps, 1))
    stream.width, stream.height = 96, 64
    stream.pix_fmt = "yuv420p"
    for index in range(frames):
        image = np.full((64, 96, 3), (30, 30, 40), dtype=np.uint8)
        if 15 <= index <= 25:
            x = min(4 + (index - 15) * 8, 96 - 32 - 2)
            image[12:44, x:x + 32] = (235, 235, 235)
        frame = av.VideoFrame.from_ndarray(image, format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()
    return path


@needs_video
class TestEndToEndRealVideo:
    def test_compile_with_real_candidate_builder(self, tmp_path):
        video = _write_real_video(tmp_path / "synthetic-demo.mp4")
        model = FakeModel()
        demo = compile_demo(
            str(video), str(tmp_path / "bundle"), TASK, model,
            window_s=2.0, max_keyframes=24,
        )
        json.dumps(demo)
        bundle = tmp_path / "bundle"
        # real candidate manifest persisted inside the bundle, source retained
        manifest = json.loads(
            (bundle / "candidates" / "manifest.json").read_text("utf-8")
        )
        assert manifest["source"]["sha256"] == _sha256(video)
        assert manifest["detector"] is None  # no detector supplied
        # demo frames point at real JPEGs inside the committed bundle (the
        # staging suffix must survive nowhere past the atomic rename)
        assert demo["frames"]
        for frame in demo["frames"]:
            path = Path(frame["image_path"])
            assert path.is_file() and ".tmp-" not in str(path)
            assert str(bundle) in str(path)
            assert path.read_bytes()[:2] == b"\xff\xd8"  # JPEG SOI
        # bundle demo.json matches the returned report exactly
        on_disk = json.loads((bundle / "demo.json").read_text("utf-8"))
        assert on_disk == demo
        assert demo["outcome_verdict"] == "supported"
        # and the store accepts the real bundle
        with DemoStore(tmp_path / "store") as store:
            demo_id = store.register(bundle)
            assert demo_id == demo["demo_id"]
            loaded = store.get(demo_id)
            assert all(Path(f["image_path"]).is_file() for f in loaded["frames"])
