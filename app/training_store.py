"""Opt-in store for reviewer corrections and the images they refer to (future training data).

Disabled unless TRAINING_STORE_DIR is set. Even then, images are kept only for jobs the
applicant consented to. The data is personal data: put the directory on an encrypted volume,
restrict access, and purge on a schedule (purge_older_than).
"""
from __future__ import annotations

import json
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path


def _ext(data: bytes) -> str:
    return "png" if data[:8] == b"\x89PNG\r\n\x1a\n" else "jpg"


class TrainingStore:
    def __init__(self, root: str | Path | None = None) -> None:
        raw = root if root is not None else os.environ.get("TRAINING_STORE_DIR")
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

    def save_images(self, job_id: str, images: list[bytes]) -> int:
        if self.root is not None and (self.root / "withdrawn" / job_id).exists():
            raise FileNotFoundError("training data for this job was removed")
        d = self._job_dir(job_id) / "images"
        d.mkdir(exist_ok=True, mode=0o700)
        for i, data in enumerate(images):
            path = d / f"{i}.{_ext(data)}"
            path.write_bytes(data)
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
        record = {"jobId": job_id, "createdAt": datetime.now(timezone.utc).isoformat(), **payload}
        path = d / "correction.json"
        path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
        path.chmod(0o600)
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
        marks = self.root / "withdrawn"
        if marks.is_dir():                         # a marker is only needed while the job row can still exist
            for m in marks.iterdir():
                try:
                    if m.stat().st_mtime < cutoff:
                        m.unlink()
                except OSError:
                    continue
        return removed
