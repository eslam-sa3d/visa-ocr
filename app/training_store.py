"""Opt-in store for reviewer corrections and the images they refer to (future training data).

Disabled unless TRAINING_STORE_DIR is set. Even then, images are kept only for jobs the
applicant consented to. The data is personal data: put the directory on an encrypted volume,
restrict access, and purge on a schedule (purge_older_than).

A correction is never trusted as it arrives. It is saved as "pending" under jobs/<id>/corrections/, one file per
submission (nothing is overwritten), and becomes training data only after a second person approves it with
tools/approve_corrections.py. The dataset builder reads approved corrections only.
"""
from __future__ import annotations

import json
import os
import shutil
import time

from . import crypto
from datetime import datetime, timezone
from pathlib import Path


def load_corrections(job_dir: Path) -> list[tuple[Path, dict]]:
    """Every readable correction of one job, oldest first. A file that cannot be read is skipped, never fatal. A
    correction.json written before approvals existed is read as pending."""
    out: list[tuple[Path, dict]] = []
    files = sorted((job_dir / "corrections").glob("*.json")) if (job_dir / "corrections").is_dir() else []
    legacy = job_dir / "correction.json"
    for path in ([legacy] if legacy.is_file() else []) + files:
        try:
            rec = json.loads(crypto.unseal_text(path.read_text(encoding="utf-8")))
        except (OSError, ValueError, crypto.DataKeyError):
            continue
        if isinstance(rec, dict):
            rec.setdefault("status", "pending")
            out.append((path, rec))
    return out


def approved_correction(job_dir: Path) -> dict | None:
    """The newest approved correction of a job (approved by someone other than its submitter), or None."""
    for _, rec in reversed(load_corrections(job_dir)):
        by = str(rec.get("approvedBy") or "").strip().casefold()
        if rec.get("status") == "approved" and by and by not in (str(rec.get("reviewer", "")).strip().casefold(),
                                                                 str(rec.get("submittedBy", "")).strip().casefold()):
            return rec
    return None


class StoreFull(RuntimeError):
    """No more training data is accepted right now: too many corrections for one job, or too little free space."""


def record_digest(rec: dict) -> str:
    """A short fingerprint of what a correction says. --show prints it and --approve must name it, so the record approved is
    the record the approver read, and not one posted after they looked."""
    import hashlib  # noqa: WPS433
    body = {k: rec.get(k) for k in ("reviewer", "submittedBy", "document", "lines", "createdAt")}
    return hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:16]


def _ext(data: bytes) -> str:
    return "png" if data[:8] == b"\x89PNG\r\n\x1a\n" else "jpg"


class TrainingStore:
    def __init__(self, root: str | Path | None = None) -> None:
        from . import config  # noqa: WPS433
        raw = root if root is not None else config.training_store_dir()
        self.root = Path(raw) if raw else None

    @property
    def enabled(self) -> bool:
        return self.root is not None

    def _job_dir(self, job_id: str) -> Path:
        assert self.root is not None
        if not job_id.isalnum():
            raise ValueError("Invalid job id")
        d = self.root / "jobs" / job_id
        for parent in (self.root, self.root / "jobs"):          # mkdir(parents=True) applies the mode to the leaf only
            parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        d.mkdir(parents=True, exist_ok=True, mode=0o700)
        return d

    def _room_left(self) -> bool:
        from . import config  # noqa: WPS433
        assert self.root is not None
        probe = self.root if self.root.exists() else self.root.parent
        try:
            return shutil.disk_usage(probe).free >= config.training_min_free_bytes()
        except OSError:
            return True

    def save_images(self, job_id: str, images: list[bytes]) -> int:
        if self.root is not None and (self.root / "withdrawn" / job_id).exists():
            raise FileNotFoundError("training data for this job was removed")
        if not self._room_left():
            raise StoreFull("the training store's volume is nearly full")
        d = self._job_dir(job_id) / "images"
        d.mkdir(exist_ok=True, mode=0o700)
        for i, data in enumerate(images):
            path = d / f"{i}.{_ext(data)}"
            path.write_bytes(crypto.seal(data))            # encrypted when DATA_KEY_FILE is set (see app/crypto.py)
            path.chmod(0o600)
        return len(images)

    def save_correction(self, job_id: str, payload: dict) -> Path:
        """Store a correction beside the consented images. Never re-creates a folder that was purged (retention, or consent
        withdrawn): that would write the applicant's data back. Raises FileNotFoundError instead."""
        assert self.root is not None
        if not job_id.isalnum():
            raise ValueError("Invalid job id")
        d = self.root / "jobs" / job_id
        if (self.root / "withdrawn" / job_id).exists() or not (d / "images").is_dir():
            raise FileNotFoundError("training data for this job was removed")
        record = {"jobId": job_id, "createdAt": datetime.now(timezone.utc).isoformat(), **payload,
                  "status": "pending", "approvedBy": None}        # the caller cannot approve its own submission
        folder = d / "corrections"
        folder.mkdir(exist_ok=True, mode=0o700)
        from . import config  # noqa: WPS433
        if sum(1 for _ in folder.glob("*.json")) >= config.max_corrections_per_job():
            raise StoreFull("no more corrections can be stored for this job")
        if not self._room_left():
            raise StoreFull("the training store's volume is nearly full")
        path = folder / f"{time.time_ns()}.json"                   # append-only: an earlier correction is never overwritten
        path.write_text(crypto.seal_text(json.dumps(record, indent=2, ensure_ascii=False)), encoding="utf-8")
        path.chmod(0o600)
        return path

    def pending(self) -> list[tuple[Path, dict]]:
        """Corrections waiting for a decision, least confident reading first: those teach the model the most."""
        if self.root is None:
            return []
        out = []
        for job in sorted((self.root / "jobs").glob("*")):
            for path, rec in load_corrections(job):
                if rec.get("status") == "pending":
                    out.append((path, rec))
        def conf(item):
            c = (item[1].get("ocr") or {}).get("confidence")
            return (0, c) if isinstance(c, (int, float)) else (1, 0.0)     # known confidence first, lowest first
        return sorted(out, key=conf)

    def decide(self, job_id: str, approver: str, approve: bool = True, note: str = "", correction: str = "",
               approved_crops: list | None = None) -> Path:
        """Approve or reject one pending correction, named by its file name or digest as --show prints them: the record decided
        is the one the approver read, never one posted afterwards. The approver must be a different person from the one who
        submitted it: a single account must not be able to put a label into the training data on its own."""
        assert self.root is not None
        approver = approver.strip()
        if not approver or not job_id.isalnum():
            raise ValueError("an approver name and a valid job id are needed")
        waiting = [(p, r) for p, r in load_corrections(self.root / "jobs" / job_id) if r.get("status") == "pending"]
        if not waiting:
            raise FileNotFoundError("no pending correction for this job")
        chosen = [(p, r) for p, r in waiting if correction and correction.strip() in (p.name, record_digest(r))]
        if len(chosen) != 1:
            raise ValueError("name the correction to decide with --correction: its file name or digest, as --show prints it")
        path, rec = chosen[0]
        same = {str(rec.get("reviewer", "")).strip().casefold(), str(rec.get("submittedBy", "")).strip().casefold()}
        if approve and approver.casefold() in same:
            raise PermissionError("a correction cannot be approved by the person who submitted it")
        digest = record_digest(rec)
        rec.update(status="approved" if approve else "rejected", approvedBy=approver if approve else None,
                   decidedBy=approver, decidedAt=datetime.now(timezone.utc).isoformat(), decisionNote=note,
                   approvedDigest=digest if approve else None,
                   approvedCrops=list(approved_crops or []) if approve else [])
        tmp = path.with_suffix(".tmp")
        tmp.write_text(crypto.seal_text(json.dumps(rec, indent=2, ensure_ascii=False)), encoding="utf-8")
        tmp.chmod(0o600)
        os.replace(tmp, path)
        from . import audit  # noqa: WPS433
        audit.event("correction_decided", job=job_id, decision=rec["status"], lines=len(rec.get("lines") or []),
                    approver=approver, submitted_by=str(rec.get("submittedBy", ""))[:16], file=path.name, digest=digest)
        return path

    def remove(self, job_id: str, tombstone: bool = False) -> bool:
        """Delete one job's images and corrections. With tombstone=True (consent withdrawn) a marker is left so nothing can be
        written for that job again; the marker holds no personal data."""
        if self.root is None or not job_id.isalnum():
            return False
        if tombstone:
            w = self.root / "withdrawn"
            w.mkdir(parents=True, exist_ok=True, mode=0o700)
            (w / job_id).touch(mode=0o600)
        d = self.root / "jobs" / job_id
        if d.is_dir():
            shutil.rmtree(d, ignore_errors=True)
            from . import audit  # noqa: WPS433
            audit.event("training_data_removed", job=job_id, consent_withdrawn=bool(tombstone))
            return True
        return False

    def purge_older_than(self, days: float) -> int:
        """Delete job folders not modified for `days` days. Returns how many were removed."""
        if self.root is None or not (self.root / "jobs").is_dir():
            return 0
        cutoff = time.time() - days * 86400
        removed = 0
        for d in (self.root / "jobs").iterdir():
            try:                                   # one bad folder must not stop the rest of the purge
                if d.is_dir() and d.stat().st_mtime < cutoff:
                    shutil.rmtree(d)
                    removed += 1
            except OSError:
                continue
        if removed:
            from . import audit  # noqa: WPS433
            audit.event("training_data_purged", folders=removed, older_than_days=days)
        marks = self.root / "withdrawn"
        if marks.is_dir():                         # a marker is only needed while the job row can still exist
            for m in marks.iterdir():
                try:
                    if m.stat().st_mtime < cutoff:
                        m.unlink()
                except OSError:
                    continue
        return removed
