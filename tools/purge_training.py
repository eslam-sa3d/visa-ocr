#!/usr/bin/env python3
"""Delete training-store job folders older than N days (images and corrections).

The service already does this in the background when TRAINING_RETENTION_DAYS > 0 (default 30).
Use this script for a cron job, or after a consent withdrawal:

  python tools/purge_training.py --store-dir /data/training --days 30
  python tools/purge_training.py --store-dir /data/training --job-id <job id>     # one job, now
  python tools/purge_training.py --store-dir /data/training --job-id <job id> --work-dir /data/autotrain

--job-id leaves a marker so no later correction can write that job's data back. --work-dir also removes the job's line
crops and label rows from datasets built for training (auto_train run folders, or a prepare_dataset.py output folder).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.training_store import TrainingStore  # noqa: E402


def purge_derived(work_dir: Path, job_id: str) -> int:
    """Remove one job's line crops (real_<jobid>_*.png) and its rows in label and prediction files under work_dir."""
    removed, marker = 0, f"real_{job_id}_"
    for crop in work_dir.rglob(marker + "*"):
        if crop.is_file():
            crop.unlink()
            removed += 1
    for txt in work_dir.rglob("*.txt"):
        try:
            rows = txt.read_text(encoding="utf-8").split("\n")
        except (OSError, UnicodeDecodeError):
            continue
        kept = [r for r in rows if marker not in r]
        if len(kept) != len(rows):
            txt.write_text("\n".join(kept), encoding="utf-8")
            removed += len(rows) - len(kept)
    return removed


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store-dir", required=True)
    ap.add_argument("--days", type=float, help="remove folders not modified for this many days")
    ap.add_argument("--job-id", help="remove one job's folder now (consent withdrawn)")
    ap.add_argument("--work-dir", type=Path, help="with --job-id: also clean datasets under this folder")
    a = ap.parse_args()
    store = TrainingStore(a.store_dir)
    if a.job_id:
        if not a.job_id.isalnum():
            sys.exit("invalid job id")
        d = Path(a.store_dir) / "jobs" / a.job_id
        store.remove(a.job_id, tombstone=True)
        print(f"removed {a.job_id}" if not d.exists() else "could not remove")
        if a.work_dir:
            print(f"removed {purge_derived(a.work_dir, a.job_id)} derived file(s) and label row(s)")
    elif a.days is not None:
        print(f"removed {store.purge_older_than(a.days)} job folder(s)")
    else:
        sys.exit("give --days or --job-id")


if __name__ == "__main__":
    main()
