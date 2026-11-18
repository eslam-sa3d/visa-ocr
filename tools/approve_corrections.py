#!/usr/bin/env python3
"""Second-person approval of reviewer corrections, run ON THE SERVER by someone other than the reviewer.

A correction posted to /v1/jobs/{id}/corrections is only "pending". It becomes training data when a different person
approves it here; the dataset builder ignores everything else. This is the control against a wrong or planted label.

    python tools/approve_corrections.py --store-dir "$TRAINING_STORE_DIR" --list
    python tools/approve_corrections.py --store-dir "$TRAINING_STORE_DIR" --show JOB_ID
    python tools/approve_corrections.py --store-dir "$TRAINING_STORE_DIR" --approve JOB_ID --approver "Your Name"
    python tools/approve_corrections.py --store-dir "$TRAINING_STORE_DIR" --reject  JOB_ID --approver "Your Name" --note "why"

--list shows the least confident readings first: those are the ones worth labelling first. --show prints the corrected lines
beside a warning for any MRZ that fails its own check digits (such a correction is never used for training, approved or not).
The output contains personal data: do not paste it into tickets or chats.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.mrz import check_score  # noqa: E402
from app.training_store import TrainingStore, load_corrections  # noqa: E402


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
    ap.add_argument("--note", default="")
    a = ap.parse_args()
    store = TrainingStore(a.store_dir)

    if a.list:
        rows = store.pending()
        for path, rec in rows:
            ocr = rec.get("ocr") or {}
            conf = ocr.get("confidence")
            warn = mrz_problem(rec)
            print(f"{rec.get('jobId', path.parent.parent.name)}  confidence={'-' if conf is None else conf}  "
                  f"lines={len(rec.get('lines') or [])}  changed={','.join(ocr.get('fieldsChanged') or []) or '-'}  "
                  f"reviewer={rec.get('reviewer', '-')}" + (f"  WARNING: {warn}" if warn else ""))
        print(f"{len(rows)} pending")
        return
    if a.show:
        for path, rec in load_corrections(a.store_dir / "jobs" / a.show):
            print(f"--- {path.name}: {rec.get('status')} (reviewer {rec.get('reviewer', '-')}, approved by {rec.get('approvedBy') or '-'})")
            for ln in rec.get("lines") or []:
                print(f"  image {ln.get('image')} box {ln.get('box')} [{ln.get('lang')}] {ln.get('text')}")
            for k, v in (rec.get("document") or {}).items():
                print(f"  {k}: {v}")
            warn = mrz_problem(rec)
            if warn:
                print(f"  WARNING: {warn}")
        return
    job = a.approve or a.reject
    if not job:
        ap.error("choose --list, --show, --approve or --reject")
    if not a.approver.strip():
        ap.error("--approver is required")
    try:
        path = store.decide(job, a.approver, approve=bool(a.approve), note=a.note)
    except PermissionError as exc:
        raise SystemExit(f"Refused: {exc}")
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"Nothing done: {exc}")
    print(f"{'approved' if a.approve else 'rejected'}: {path}")


if __name__ == "__main__":
    main()
