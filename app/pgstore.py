"""PostgreSQL job store with a work queue, for several API/worker processes or hosts.

The API inserts a job (images included, as bytea) and returns at once; workers claim jobs with
SELECT ... FOR UPDATE SKIP LOCKED, so each job is processed once even with many workers. Images are erased
as soon as the job finishes; every row is deleted after JOB_TTL_MINUTES. A running worker renews its claim
(heartbeat); a worker that dies stops renewing, and after JOB_STALE_SECONDS the job is re-queued (up to JOB_MAX_ATTEMPTS,
then it fails). Every write by a run carries its attempt number, so a run that lost its claim cannot overwrite the job.

Enable with DATABASE_URL=postgresql://user:pass@host/db. Use disk encryption / TLS to the database
(sslmode=require) because rows hold personal data while a job is running.
"""
from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone

from . import config
from .jobs import Job

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
  id TEXT PRIMARY KEY, correlation_id TEXT, owner TEXT, status TEXT NOT NULL, created_at TEXT, created_ts DOUBLE PRECISION,
  result JSONB, error JSONB, consent_training BOOLEAN DEFAULT FALSE,
  hint TEXT, applicant JSONB, images BYTEA[], attempts INT DEFAULT 0, locked_ts DOUBLE PRECISION);
CREATE INDEX IF NOT EXISTS jobs_queue ON jobs (status, created_ts);
"""


class PgJobStore:
    queue = True

    def __init__(self, url: str | None = None, ttl_seconds: int | None = None) -> None:
        import psycopg
        self._psycopg = psycopg
        self._url = url or config.database_url()
        self._ttl = ttl_seconds
        with self._conn() as c:
            c.execute(SCHEMA)

    def _conn(self):
        return self._psycopg.connect(self._url, autocommit=True, connect_timeout=5)

    @property
    def ttl(self) -> int:
        return config.job_ttl_seconds() if self._ttl is None else self._ttl

    def _purge(self, c) -> None:
        c.execute("DELETE FROM jobs WHERE created_ts < %s", (time.time() - self.ttl,))

    def sweep(self) -> None:
        with self._conn() as c:
            self._purge(c)

    # ---- API side
    def create(self, correlation_id: str, owner: str = "") -> Job:
        job = Job(id=uuid.uuid4().hex, correlation_id=correlation_id, owner=owner)
        with self._conn() as c:
            self._purge(c)
            c.execute("INSERT INTO jobs (id, correlation_id, owner, status, created_at, created_ts) VALUES (%s,%s,%s,%s,%s,%s)",
                      (job.id, correlation_id, owner, "preparing", job.created_at, time.time()))
        return job

    def create_if_room(self, correlation_id: str, owner: str, limit: int) -> Job | None:
        """Count pending jobs and insert in one transaction under an advisory lock, so API nodes cannot overrun the cap."""
        job = Job(id=uuid.uuid4().hex, correlation_id=correlation_id, owner=owner)
        with self._conn() as c:
            self._purge(c)
            with c.transaction():
                c.execute("SELECT pg_advisory_xact_lock(7406001)")
                n = c.execute("SELECT COUNT(*) FROM jobs WHERE status IN ('preparing','queued','processing')").fetchone()[0]
                if n >= limit:
                    return None
                c.execute("INSERT INTO jobs (id, correlation_id, owner, status, created_at, created_ts) VALUES (%s,%s,%s,%s,%s,%s)",
                          (job.id, correlation_id, owner, "preparing", job.created_at, time.time()))
        return job

    def discard(self, job_id: str) -> None:
        with self._conn() as c:
            c.execute("DELETE FROM jobs WHERE id=%s AND status='preparing'", (job_id,))

    def submit(self, job_id: str, images: list[bytes], hint: str | None, applicant: dict) -> None:
        from psycopg.types.json import Jsonb
        with self._conn() as c:
            c.execute("UPDATE jobs SET images=%s, hint=%s, applicant=%s, status='queued' WHERE id=%s",
                      (images, hint, Jsonb(applicant), job_id))

    def get(self, job_id: str) -> Job | None:
        with self._conn() as c:
            self._purge(c)
            row = c.execute("SELECT id, correlation_id, owner, status, created_at, result, error, consent_training "
                            "FROM jobs WHERE id=%s", (job_id,)).fetchone()
        if not row:
            return None
        status = "queued" if row[3] == "preparing" else row[3]
        return Job(id=row[0], correlation_id=row[1], owner=row[2], status=status, created_at=row[4],
                   result=row[5], error=row[6], consent_training=bool(row[7]))

    def update(self, job_id: str, **changes) -> None:
        from psycopg.types.json import Jsonb
        sets, vals = [], []
        for key, value in changes.items():
            if key == "status":
                sets.append("status=%s"); vals.append(value)
            elif key in ("result", "error"):
                sets.append(f"{key}=%s"); vals.append(Jsonb(value) if value is not None else None)  # noqa: S608 - fixed names
            elif key == "consent_training":
                sets.append("consent_training=%s"); vals.append(bool(value))
        if changes.get("status") in ("done", "failed"):
            sets.append("images=NULL, hint=NULL, applicant=NULL")   # nothing the caller sent outlives the job
        if sets:
            with self._conn() as c:
                c.execute(f"UPDATE jobs SET {', '.join(sets)} WHERE id=%s", (*vals, job_id))  # noqa: S608

    def pending_count(self) -> int:
        with self._conn() as c:
            return c.execute("SELECT COUNT(*) FROM jobs WHERE status IN ('preparing','queued','processing')").fetchone()[0]

    def count(self) -> int:
        with self._conn() as c:
            return c.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]

    # ---- worker side
    def requeue_stale(self) -> int:
        """Retry jobs whose worker died; fail them after too many attempts."""
        from psycopg.types.json import Jsonb
        now = time.time()
        cutoff = now - config.job_stale_seconds()
        with self._conn() as c, c.transaction():        # both steps see one state
            c.execute("UPDATE jobs SET status='failed', images=NULL, hint=NULL, applicant=NULL, error=%s "
                      "WHERE status='processing' AND locked_ts < %s AND attempts >= %s",
                      (Jsonb({"code": "internal_error", "message": "Processing failed"}), cutoff, config.job_max_attempts()))
            # a row left 'preparing' by an API process that stopped between create and submit would hold a slot until the TTL
            c.execute("DELETE FROM jobs WHERE status='preparing' AND created_ts < %s", (cutoff,))
            return c.execute("UPDATE jobs SET status='queued' WHERE status='processing' AND locked_ts < %s", (cutoff,)).rowcount

    def heartbeat(self, job_id: str, token: int) -> bool:
        """Renew a claim while the job runs. False when this run no longer holds the claim."""
        with self._conn() as c:
            return c.execute("UPDATE jobs SET locked_ts=%s WHERE id=%s AND status='processing' AND attempts=%s",
                             (time.time(), job_id, token)).rowcount == 1

    def finish(self, job_id: str, token: int, status: str, result: dict | None = None, error: dict | None = None) -> bool:
        """Write the outcome only if this run still holds the claim (same attempt number, still processing). A run that was
        re-queued or failed by the stale sweep gets False and its result is dropped."""
        from psycopg.types.json import Jsonb
        with self._conn() as c:
            return c.execute("UPDATE jobs SET status=%s, result=%s, error=%s, images=NULL, hint=NULL, applicant=NULL "
                             "WHERE id=%s AND status='processing' AND attempts=%s",
                             (status, Jsonb(result) if result is not None else None,
                              Jsonb(error) if error is not None else None, job_id, token)).rowcount == 1

    def claim(self) -> "Claim | None":
        """Take the oldest queued job, or None. Safe with any number of concurrent workers. The returned tuple also carries
        .token (the attempt number), which heartbeat() and finish() need."""
        with self._conn() as c:
            row = c.execute(
                "UPDATE jobs SET status='processing', attempts=attempts+1, locked_ts=%s "
                "WHERE id = (SELECT id FROM jobs WHERE status='queued' AND images IS NOT NULL "
                "ORDER BY created_ts FOR UPDATE SKIP LOCKED LIMIT 1) "
                "RETURNING id, images, hint, applicant, attempts", (time.time(),)).fetchone()
        if not row:
            return None
        claim = Claim((row[0], [bytes(b) for b in row[1]], row[2], row[3] or {}))
        claim.token = row[4]
        return claim


class Claim(tuple):
    """(job_id, images, hint, applicant) plus .token, the attempt number that fences this run's writes."""
    token: int = 0


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()
