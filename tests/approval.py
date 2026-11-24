"""Approve a correction the way tools/approve_corrections.py does: by its digest, with the crops the approver was shown."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.training_store import load_corrections, record_digest  # noqa: E402
from training import prepare_dataset  # noqa: E402


def crops(job_dir: Path, rec: dict) -> list:
    images = prepare_dataset.job_images(job_dir)
    out = []
    for ln in rec.get("lines") or []:
        crop = prepare_dataset.cut_line(job_dir, ln, images)[0] if prepare_dataset._well_formed(ln) else None
        out.append(prepare_dataset.crop_digest(crop) if crop is not None else None)
    return out


def approve(store, job_id: str, approver: str, approve: bool = True, note: str = ""):
    job_dir = Path(store.root) / "jobs" / job_id
    pending = [(p, r) for p, r in load_corrections(job_dir) if r.get("status") == "pending"]
    path, rec = pending[-1]
    return store.decide(job_id, approver, approve=approve, note=note, correction=record_digest(rec),
                        approved_crops=crops(job_dir, rec))
