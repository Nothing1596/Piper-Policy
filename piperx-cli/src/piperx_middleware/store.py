import json
import logging
from pathlib import Path
import sqlite3
import threading
import time


class Store:
    """Durable request identity: a restart never replays an uncertain motion."""
    def __init__(self, root: Path):
        root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.db = sqlite3.connect(root / "execution.sqlite3", check_same_thread=False)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.execute("CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, request_id TEXT UNIQUE, plan_id TEXT, result TEXT)")
        self.db.execute("CREATE TABLE IF NOT EXISTS events (seq INTEGER PRIMARY KEY AUTOINCREMENT, at REAL, event TEXT, data TEXT)")
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS calls ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "at REAL, "
            "source TEXT, "
            "operation TEXT, "
            "method TEXT, "
            "path TEXT, "
            "status TEXT, "
            "duration_ms REAL, "
            "request_id TEXT, "
            "job_id TEXT, "
            "error_code TEXT"
            ")"
        )
        for ident, raw in self.db.execute("SELECT id,result FROM jobs").fetchall():
            job = json.loads(raw)
            if job["status"] == "awaiting_approval":
                job.update(status="cancelled", error_code="executor_restarted", error="Pending approval invalidated by executor restart; no replay.")
                self.db.execute("UPDATE jobs SET result=? WHERE id=?", (json.dumps(job), ident))
            elif job["status"] in ("accepted", "running"):
                job.update(status="outcome_unknown", error="Executor restarted; command is never replayed.")
                self.db.execute("UPDATE jobs SET result=? WHERE id=?", (json.dumps(job), ident))
        self.db.commit()

    def event(self, name, **data):
        with self.lock:
            self.db.execute("INSERT INTO events(at,event,data) VALUES(?,?,?)", (time.time(), name, json.dumps(data, allow_nan=False)))
            self.db.commit()

    def put(self, job, new=False):
        raw = json.dumps(job, allow_nan=False)
        with self.lock:
            if new:
                self.db.execute("INSERT INTO jobs VALUES(?,?,?,?)", (job["job_id"], job["request_id"], job["plan_id"], raw))
            else:
                self.db.execute("UPDATE jobs SET result=? WHERE id=?", (raw, job["job_id"]))
            self.db.commit()

    def get(self, *, job_id=None, request_id=None):
        key, value = ("id", job_id) if job_id else ("request_id", request_id)
        with self.lock:
            row = self.db.execute(f"SELECT result FROM jobs WHERE {key}=?", (value,)).fetchone()
            return json.loads(row[0]) if row else None

    def jobs(self, limit: int = 100):
        with self.lock:
            lim = max(1, min(int(limit), 500))
            rows = self.db.execute("SELECT result FROM jobs ORDER BY rowid DESC LIMIT ?", (lim,)).fetchall()
            return [json.loads(r[0]) for r in rows]

    def events(self, after=0, limit=100):
        with self.lock:
            return [dict(seq=s, at=t, event=n, data=json.loads(d)) for s, t, n, d in
                    self.db.execute("SELECT seq,at,event,data FROM events WHERE seq>? ORDER BY seq LIMIT ?", (after, limit))]

    def record_call(self, *, at: float, source: str, operation: str, method: str, path: str,
                    status: str, duration_ms: float, request_id: str | None = None,
                    job_id: str | None = None, error_code: str | None = None) -> int | None:
        with self.lock:
            try:
                cur = self.db.execute(
                    "INSERT INTO calls(at, source, operation, method, path, status, duration_ms, request_id, job_id, error_code) "
                    "VALUES(?,?,?,?,?,?,?,?,?,?)",
                    (at, source, operation, method, path, status, duration_ms, request_id, job_id, error_code)
                )
                self.db.execute("DELETE FROM calls WHERE id <= (SELECT id FROM calls ORDER BY id DESC LIMIT 1 OFFSET 2000)")
                self.db.commit()
                return cur.lastrowid
            except Exception as exc:
                try:
                    self.db.rollback()
                except sqlite3.Error:
                    pass
                logging.getLogger(__name__).warning(
                    "Call history write failed (%s); execution result preserved.", type(exc).__name__)
                return None

    def calls(self, limit: int = 100):
        with self.lock:
            lim = max(1, min(int(limit), 500))
            rows = self.db.execute(
                "SELECT id, at, source, operation, method, path, status, duration_ms, request_id, job_id, error_code "
                "FROM calls ORDER BY id DESC LIMIT ?",
                (lim,)
            ).fetchall()
            return [
                {
                    "id": r[0],
                    "at": r[1],
                    "source": r[2],
                    "operation": r[3],
                    "method": r[4],
                    "path": r[5],
                    "status": r[6],
                    "duration_ms": r[7],
                    "request_id": r[8],
                    "job_id": r[9],
                    "error_code": r[10],
                }
                for r in rows
            ]

    def close(self):
        self.db.close()
