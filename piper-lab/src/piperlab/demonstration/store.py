"""SQLite-backed demonstration store (human-video v1).

Persistence for compiled demo bundles: whole-bundle copy plus a small SQLite
index for tag/text lookup. No embeddings, no vector store; the exact
``demo_id`` is the main reproducibility path, ``find`` is only a discovery
aid.

Bundles are self-contained after registration: frame ``image_path`` values
are rewritten to the store's own copy, so deleting the compile output does
not corrupt stored demos. Identity/idempotency checks hash the semantic
content with absolute paths normalised to bundle-relative ones, so
re-registering the same bundle from a different location is still a no-op.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sqlite3
import time
from pathlib import Path

from .compiler import _write_json_atomic  # same atomic-write convention


class StoreError(RuntimeError):
    """A store operation failed explicitly."""


_DEMO_JSON = "demo.json"
_TOKEN_RE = re.compile(r"[a-z0-9]+|[\u3400-\u4dbf\u4e00-\u9fff\U00020000-\U0002fa1f]+")


def _fail_if_exists(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"output already exists, refusing to overwrite: {path}")


def _tokens(text: str) -> list[str]:
    # Chinese has no whitespace word boundaries. Adjacent character pairs
    # support substring discovery without a tokenizer/model dependency. Keep
    # single-character runs searchable, but avoid single-character overlap
    # dominating multi-character queries. This remains lexical, not semantic.
    tokens: list[str] = []
    for token in _TOKEN_RE.findall(text.lower()):
        if token.isascii():
            if len(token) >= 2:
                tokens.append(token)
        elif len(token) == 1:
            tokens.append(token)
        else:
            tokens.extend(token[index:index + 2] for index in range(len(token) - 1))
    return tokens


def _content_hash(demo: dict, bundle_dir: Path) -> str:
    """Hash the semantic content with absolute paths normalised away.

    ``image_path`` becomes bundle-relative and ``source_path`` is dropped, so
    the hash is stable across bundle locations (compile dir vs store copy).
    """
    from .paths import relocate_bundle
    normalised = relocate_bundle(demo,bundle_dir,Path('.'))
    normalised.pop("source_path", None)
    payload = json.dumps(normalised, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _validate_demo(demo, bundle_dir: Path) -> dict:
    if not isinstance(demo, dict):
        raise StoreError(f"{bundle_dir}/{_DEMO_JSON} is not a JSON object")
    if demo.get("schema_version") != 1:
        raise StoreError(f"unsupported schema_version: {demo.get('schema_version')!r}")
    demo_id = demo.get("demo_id")
    if not isinstance(demo_id, str) or not demo_id.startswith("demo-"):
        raise StoreError(f"demo_id {demo_id!r} is missing or malformed")
    if not isinstance(demo.get("task"), str) or not demo["task"].strip():
        raise StoreError("demo has no task text")
    if not isinstance(demo.get("source_hash"), str) or not demo["source_hash"]:
        raise StoreError("demo has no source_hash")
    for key in ("frames", "stages", "goal_constraints", "unknowns"):
        if not isinstance(demo.get(key), list):
            raise StoreError(f"demo {key!r} is missing or not a list")
    for frame in demo["frames"]:
        if not isinstance(frame, dict) or not isinstance(frame.get("frame_id"), str):
            raise StoreError(f"demo frame entry is malformed: {frame!r}")
    return demo


def _tags(demo: dict) -> list[str]:
    tags: set[str] = set()
    for stage in demo.get("stages", []):
        if isinstance(stage, dict):
            operation = stage.get("operation")
            if isinstance(operation, str) and operation.strip():
                tags.add(operation.strip().lower())
            for role in stage.get("object_roles", []):
                if isinstance(role, str) and role.strip():
                    tags.add(role.strip().lower())
    return sorted(tags)


class DemoStore:
    """A root directory holding registered demo bundles plus a SQLite index."""

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root).absolute()
        self._bundles = self._root / "bundles"
        self._bundles.mkdir(parents=True, exist_ok=True)
        self._db_path = self._root / "store.sqlite3"
        self._db = sqlite3.connect(str(self._db_path))
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS demos ("
            " demo_id TEXT PRIMARY KEY,"
            " task TEXT NOT NULL,"
            " source_hash TEXT NOT NULL,"
            " model_name TEXT,"
            " tags TEXT NOT NULL,"
            " text TEXT NOT NULL,"
            " content_hash TEXT NOT NULL,"
            " created_utc REAL NOT NULL"
            ")"
        )
        self._db.commit()

    @property
    def root(self) -> Path:
        return self._root

    def close(self) -> None:
        self._db.close()

    def __enter__(self) -> "DemoStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- write ------------------------------------------------------------
    def register(self, bundle_path: str | Path) -> str:
        """Register a compiled bundle; returns its exact demo_id.

        Re-registering identical content is an idempotent no-op. Registering
        different content under an existing demo_id raises StoreError; a
        demo_id must never silently change meaning.
        """
        source_dir = Path(bundle_path).absolute()
        demo_file = source_dir / _DEMO_JSON
        if not demo_file.is_file():
            raise StoreError(f"bundle lacks {_DEMO_JSON}: {source_dir}")
        with open(demo_file, "r", encoding="utf-8") as handle:
            demo = _validate_demo(json.load(handle), source_dir)
        from .paths import resolve_bundle
        demo=resolve_bundle(demo,source_dir)
        demo_id = demo["demo_id"]
        content_hash = _content_hash(demo, source_dir)

        row = self._db.execute(
            "SELECT content_hash FROM demos WHERE demo_id = ?", (demo_id,)
        ).fetchone()
        if row is not None:
            if row[0] != content_hash:
                raise StoreError(
                    f"demo_id {demo_id} already registered with different content; "
                    "refusing to overwrite a reproducibility anchor"
                )
            return demo_id

        target = self._bundles / demo_id
        _fail_if_exists(target)
        staging = target.parent / (target.name + f".tmp-{os.getpid()}")
        if staging.exists():
            raise FileExistsError(f"stale staging directory in the way: {staging}")
        shutil.copytree(source_dir, staging)

        # Rewrite frame image paths to the store's own copy so the stored
        # bundle is self-contained. Paths outside the bundle are kept as-is
        # and explicitly flagged.
        from .paths import relocate_bundle
        stored_demo = relocate_bundle(demo,source_dir,target)
        for original,frame in zip(demo['frames'],stored_demo['frames']):
            path = original.get("image_path")
            if not isinstance(path, str):
                continue
            rel = os.path.relpath(path, source_dir)
            if rel.startswith(".."):
                frame["image_outside_bundle"] = True
        _write_json_atomic(staging / _DEMO_JSON, stored_demo)
        os.rename(staging, target)

        tags = _tags(demo)
        text = " ".join([demo["task"], *tags])
        model_name = demo.get("model_identity", {}).get("name")
        self._db.execute(
            "INSERT INTO demos"
            " (demo_id, task, source_hash, model_name, tags, text, content_hash, created_utc)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                demo_id,
                demo["task"],
                demo["source_hash"],
                model_name if isinstance(model_name, str) else None,
                json.dumps(tags),
                text,
                content_hash,
                time.time(),
            ),
        )
        self._db.commit()
        return demo_id

    # -- read -------------------------------------------------------------
    def _load(self, demo_id: str) -> dict:
        demo_file = self._bundles / demo_id / _DEMO_JSON
        if not demo_file.is_file():
            raise StoreError(
                f"bundle files for {demo_id} are missing under {self._bundles}; "
                "the store is inconsistent"
            )
        with open(demo_file, "r", encoding="utf-8") as handle:
            from .paths import resolve_bundle
            return resolve_bundle(json.load(handle),demo_file.parent)

    def get(self, demo_id: str) -> dict:
        """Return the stored demo document; KeyError for an unknown id."""
        row = self._db.execute(
            "SELECT demo_id FROM demos WHERE demo_id = ?", (demo_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"unknown demo_id: {demo_id}")
        return self._load(demo_id)

    def find(self, task: str, limit: int = 5) -> list[dict]:
        """Best-effort discovery by task text/tags; exact demo_id via get().

        Scoring is a plain token-overlap count over the stored task+tags text;
        there is no embedding assumption anywhere in v1.
        """
        if not isinstance(task, str) or not task.strip():
            raise ValueError("task must be a non-empty string")
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise ValueError("limit must be a positive int")
        tokens = set(_tokens(task))
        if not tokens:
            return []
        scored: list[tuple[int, str]] = []
        for demo_id, text in self._db.execute("SELECT demo_id, text FROM demos"):
            stored = set(_tokens(text))
            score = len(tokens & stored)
            if score:
                scored.append((score, demo_id))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [self._load(demo_id) for _score, demo_id in scored[:limit]]
