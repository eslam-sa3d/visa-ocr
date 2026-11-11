"""Job stores. Results contain personal data, so every job expires (JOB_TTL_MINUTES) and is deleted.

JobStore keeps jobs in memory (lost on restart). SqliteJobStore (set JOB_DB_PATH) survives restarts and can be
shared by several worker processes on one host. For several hosts set DATABASE_URL: see pgstore.py (PostgreSQL + work queue).
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

from . import config

PENDING = ("queued", "processing")


@dataclass
class Job:
    id: str
    correlation_id: str
    owner: str = ""
    status: str = "queued"  # queued | processing | done | failed
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    result: dict | None = None
    error: dict | None = None
    consent_training: bool = False  # applicant agreed their images may be kept for model training

    def to_dict(self) -> dict:
        return {
            "jobId": self.id,
            "correlationId": self.correlation_id,
            "status": self.status,
            "createdAt": self.created_at,
            "result": self.result,
            "error": self.error,
        }


class JobStore:
    def __init__(self, ttl_seconds: int | None = None) -> None:
        self._jobs: dict[str, tuple[float, Job]] = {}
        self._lock = threading.Lock()
        self._ttl = ttl_seconds

    @property
    def ttl(self) -> int:
        return config.job_ttl_seconds() if self._ttl is None else self._ttl

    def _purge_locked(self) -> None:
        cutoff = time.time() - self.ttl
        for jid in [k for k, (t, _) in self._jobs.items() if t < cutoff]:
            del self._jobs[jid]

    def sweep(self) -> None:
        with self._lock:
            self._purge_locked()

    def create(self, correlation_id: str, owner: str = "") -> Job:
        job = Job(id=uuid.uuid4().hex, correlation_id=correlation_id, owner=owner)
        with self._lock:
            self._purge_locked()
            self._jobs[job.id] = (time.time(), job)
        return job

    def create_if_room(self, correlation_id: str, owner: str, limit: int) -> Job | None:
        """Count pending jobs and insert under one lock, so concurrent uploads cannot overrun the cap."""
        job = Job(id=uuid.uuid4().hex, correlation_id=correlation_id, owner=owner)
        with self._lock:
            self._purge_locked()
            if sum(1 for _, j in self._jobs.values() if j.status in PENDING) >= limit:
                return None
            self._jobs[job.id] = (time.time(), job)
        return job

    def discard(self, job_id: str) -> None:
        with self._lock:
            self._jobs.pop(job_id, None)

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            self._purge_locked()
            entry = self._jobs.get(job_id)
            return entry[1] if entry else None

    def update(self, job_id: str, **changes) -> None:
        with self._lock:
            entry = self._jobs.get(job_id)
            if entry is None:        # expired while processing: nothing to update
                return
            for key, value in changes.items():
                setattr(entry[1], key, value)

    def pending_count(self) -> int:
        with self._lock:
            return sum(1 for _, j in self._jobs.values() if j.status in PENDING)

    def count(self) -> int:
        with self._lock:
            return len(self._jobs)


class SqliteJobStore:
    def __init__(self, path: str, ttl_seconds: int | None = None) -> None:
        self._path = path
        self._ttl = ttl_seconds
        self._lock = threading.Lock()
        old_umask = os.umask(0o077)                    # the database holds personal data: owner-only files
        try:
            c0 = self._conn()
            c0.close()
        finally:
            os.umask(old_umask)
        self._lock_down()
        with self._conn() as c:
            c.execute("""CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY, correlation_id TEXT, owner TEXT, status TEXT, created_at TEXT, created_ts REAL,
                result TEXT, error TEXT, consent_training INTEGER)""")

    def _conn(self) -> sqlite3.Connection:
        c = sqlite3.connect(self._path, timeout=10, check_same_thread=False, isolation_level=None)
        c.execute("PRAGMA journal_mode=WAL")
        c.execute("PRAGMA secure_delete=ON")            # overwrite deleted rows instead of leaving them in free pages
        return c

    def _lock_down(self) -> None:
        for suffix in ("", "-wal", "-shm"):
            try:
                os.chmod(self._path + suffix, 0o600)
            except OSError:
                pass

    def sweep(self) -> None:
        with self._lock, self._conn() as c:
            self._purge(c)
            c.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        self._lock_down()

    @property
    def ttl(self) -> int:
        return config.job_ttl_seconds() if self._ttl is None else self._ttl

    def _purge(self, c: sqlite3.Connection) -> None:
        c.execute("DELETE FROM jobs WHERE created_ts < ?", (time.time() - self.ttl,))

    def create(self, correlation_id: str, owner: str = "") -> Job:
        job = Job(id=uuid.uuid4().hex, correlation_id=correlation_id, owner=owner)
        with self._lock, self._conn() as c:
            self._purge(c)
            c.execute("INSERT INTO jobs VALUES (?,?,?,?,?,?,?,?,?)",
                      (job.id, correlation_id, owner, job.status, job.created_at, time.time(), None, None, 0))
        return job

    def create_if_room(self, correlation_id: str, owner: str, limit: int) -> Job | None:
        """Count and insert in one write transaction (other processes sharing the file wait on BEGIN IMMEDIATE)."""
        job = Job(id=uuid.uuid4().hex, correlation_id=correlation_id, owner=owner)
        with self._lock:
            c = self._conn()
            try:
                c.execute("BEGIN IMMEDIATE")
                self._purge(c)
                n = c.execute("SELECT COUNT(*) FROM jobs WHERE status IN ('queued','processing')").fetchone()[0]
                if n >= limit:
                    c.execute("ROLLBACK")
                    return None
                c.execute("INSERT INTO jobs VALUES (?,?,?,?,?,?,?,?,?)",
                          (job.id, correlation_id, owner, job.status, job.created_at, time.time(), None, None, 0))
                c.execute("COMMIT")
            finally:
                c.close()
        return job

    def discard(self, job_id: str) -> None:
        with self._lock, self._conn() as c:
            c.execute("DELETE FROM jobs WHERE id = ?", (job_id,))

    def get(self, job_id: str) -> Job | None:
        with self._lock, self._conn() as c:
            self._purge(c)
            row = c.execute("SELECT id, correlation_id, owner, status, created_at, result, error, consent_training "
                            "FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if not row:
            return None
        return Job(id=row[0], correlation_id=row[1], owner=row[2], status=row[3], created_at=row[4],
                   result=json.loads(row[5]) if row[5] else None, error=json.loads(row[6]) if row[6] else None,
                   consent_training=bool(row[7]))

    def update(self, job_id: str, **changes) -> None:
        cols, vals = [], []
        for key, value in changes.items():
            if key in ("result", "error"):
                value = json.dumps(value, ensure_ascii=False) if value is not None else None
            elif key == "consent_training":
                value = int(bool(value))
            elif key not in ("status",):
                continue
            cols.append(f"{key} = ?")
            vals.append(value)
        if cols:
            with self._lock, self._conn() as c:
                c.execute(f"UPDATE jobs SET {', '.join(cols)} WHERE id = ?", (*vals, job_id))  # noqa: S608 - fixed column names

    def pending_count(self) -> int:
        with self._lock, self._conn() as c:
            return c.execute("SELECT COUNT(*) FROM jobs WHERE status IN ('queued','processing')").fetchone()[0]

    def count(self) -> int:
        with self._lock, self._conn() as c:
            return c.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]


def make_store():
    if config.database_url():
        from .pgstore import PgJobStore
        return PgJobStore()
    path = config.job_db_path()
    return SqliteJobStore(path) if path else JobStore()
