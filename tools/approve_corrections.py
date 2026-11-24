#!/usr/bin/env python3
"""Second-person approval of reviewer corrections, run ON THE SERVER by someone other than the reviewer.

A correction posted to /v1/jobs/{id}/corrections is only "pending". It becomes training data when a different person
approves it here; the dataset builder ignores everything else. This is the control against a wrong or planted label.

    python tools/approve_corrections.py --store-dir "$TRAINING_STORE_DIR" --list
    python tools/approve_corrections.py --store-dir "$TRAINING_STORE_DIR" --show JOB_ID [--crops-dir DIR]
    python tools/approve_corrections.py --store-dir "$TRAINING_STORE_DIR" --approve JOB_ID --correction DIGEST --approver "Your Name"
    python tools/approve_corrections.py --store-dir "$TRAINING_STORE_DIR" --reject  JOB_ID --correction DIGEST --approver "Your Name" --note "why"

--list shows the least confident readings first: those are the ones worth labelling first. --show prints each correction with
its digest, the document fields, then each line with the image crop its box covers, written as a PNG to a private folder
(--crops-dir, or a new temporary folder): look at the crop, not only at the numbers. A warning is printed for any MRZ that
fails its own check digits (such a correction is never used for training, approved or not). --approve and --reject decide
only the correction whose digest (or file name) you give, and only if it has not changed since --show. The crop digests are
recorded with the approval, and a line whose crop differs from what you saw is never used. Stored text is shown with every
control or formatting character as a visible escape. The output and the crops contain personal data: do not paste them into
tickets or chats, and delete the crops folder when you are done.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import tempfile
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.mrz import check_score  # noqa: E402
from app.training_store import TrainingStore, load_corrections, record_digest  # noqa: E402
from training import prepare_dataset  # noqa: E402


def safe(value) -> str:
    """Stored text as one visible line: anything that is not printable (control, formatting or direction characters) is
    shown as an escape, so a stored value cannot change what the approver sees."""
    return "".join(c if c.isprintable() and unicodedata.category(c) != "Cf" else c.encode("unicode_escape").decode("ascii")
                   for c in str(value))


def crops_for(job_dir: Path, rec: dict, folder: Path | None, stem: str) -> list[tuple[str | None, Path | None]]:
    """(digest, written file) for each line of a correction, cut exactly as the dataset builder cuts it."""
    out = []
    images = prepare_dataset.job_images(job_dir)
    for i, ln in enumerate(rec.get("lines") or []):
        if not prepare_dataset._well_formed(ln):
            out.append((None, None))
            continue
        crop, _ = prepare_dataset.cut_line(job_dir, ln, images)
        if crop is None:
            out.append((None, None))
            continue
        path = None
        if folder is not None:
            path = folder / f"{stem}_{i}.png"
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "wb") as fh:
                crop.save(fh, "PNG")
        out.append((prepare_dataset.crop_digest(crop), path))
    return out


def mrz_problem(rec: dict) -> str | None:
    lines = [ln.get("text", "") for ln in rec.get("lines") or [] if isinstance(ln, dict)]
    mrz = [t for t in lines if re.fullmatch(r"[A-Z0-9<]{28,44}", t or "")]
    if len(mrz) in (2, 3) and not check_score(mrz)[1]:
        return "the corrected MRZ fails its own check digits"
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store-dir", type=Path, required=True)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--show", metavar="JOB_ID")
    ap.add_argument("--approve", metavar="JOB_ID")
    ap.add_argument("--reject", metavar="JOB_ID")
    ap.add_argument("--approver", default="")
    ap.add_argument("--correction", default="", help="digest or file name of the correction, as --show prints it")
    ap.add_argument("--crops-dir", type=Path, default=None, help="where --show writes the line crops (default: a new private temp folder)")
    ap.add_argument("--note", default="")
    a = ap.parse_args()
    store = TrainingStore(a.store_dir)

    if a.list:
        rows = store.pending()
        for path, rec in rows:
            ocr = rec.get("ocr") or {}
            conf = ocr.get("confidence")
            warn = mrz_problem(rec)
            print(f"{safe(rec.get('jobId', path.parent.parent.name))}  [{record_digest(rec)}]  "
                  f"confidence={'-' if conf is None else safe(conf)}  lines={len(rec.get('lines') or [])}  "
                  f"changed={','.join(safe(f) for f in ocr.get('fieldsChanged') or []) or '-'}  "
                  f"reviewer={safe(rec.get('reviewer', '-'))}" + (f"  WARNING: {warn}" if warn else ""))
        print(f"{len(rows)} pending")
        return
    if a.show:
        if not a.show.isalnum():
            ap.error("a job id is letters and digits only")
        job_dir = a.store_dir / "jobs" / a.show
        folder = a.crops_dir
        if folder is None:
            folder = Path(tempfile.mkdtemp(prefix="visa-ocr-crops-"))      # created 0700
        else:
            folder.mkdir(parents=True, exist_ok=True, mode=0o700)
        for path, rec in load_corrections(job_dir):
            print(f"--- {safe(path.name)} [{record_digest(rec)}]: {safe(rec.get('status'))} "
                  f"(reviewer {safe(rec.get('reviewer', '-'))}, approved by {safe(rec.get('approvedBy') or '-')})")
            for k, v in (rec.get("document") or {}).items():
                print(f"  {safe(k)}: {safe(v)}")
            crops = crops_for(job_dir, rec, folder, path.stem)
            for ln, (digest, crop) in zip(rec.get("lines") or [], crops):
                ln = ln if isinstance(ln, dict) else {}
                where = f"crop {crop} sha256 {digest[:16]}" if crop else "NO CROP (box outside the image): never used"
                print(f"  image {safe(ln.get('image'))} box {safe(ln.get('box'))} [{safe(ln.get('lang'))}] "
                      f"{safe(ln.get('text'))}  -> {where}")
            warn = mrz_problem(rec)
            if warn:
                print(f"  WARNING: {warn}")
        print(f"crops written to {folder} (personal data: delete the folder when done)")
        return
    job = a.approve or a.reject
    if not job:
        ap.error("choose --list, --show, --approve or --reject")
    if not a.approver.strip():
        ap.error("--approver is required")
    if not a.correction.strip():
        ap.error("--correction is required: the digest printed by --show for the correction you checked")
    if not job.isalnum():
        ap.error("a job id is letters and digits only")
    job_dir = a.store_dir / "jobs" / job
    crops = []
    if a.approve:
        for p, rec in load_corrections(job_dir):
            if a.correction.strip() in (p.name, record_digest(rec)):
                crops = [d for d, _ in crops_for(job_dir, rec, None, "")]
    try:
        path = store.decide(job, a.approver, approve=bool(a.approve), note=a.note, correction=a.correction,
                            approved_crops=crops)
    except PermissionError as exc:
        raise SystemExit(f"Refused: {exc}")
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"Nothing done: {exc}")
    print(f"{'approved' if a.approve else 'rejected'}: {path.name} [{safe(a.correction)}]")


if __name__ == "__main__":
    main()
