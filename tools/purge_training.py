#!/usr/bin/env python3
"""Delete training-store job folders older than N days (images and corrections).

The service already does this in the background when TRAINING_RETENTION_DAYS > 0 (default 30).
Use this script for a cron job, or after a consent withdrawal:

  python tools/purge_training.py --store-dir /data/training --days 30
  python tools/purge_training.py --store-dir /data/training --job-id <job id>     # one job, now
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.training_store import TrainingStore  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store-dir", required=True)
    ap.add_argument("--days", type=float, help="remove folders not modified for this many days")
    ap.add_argument("--job-id", help="remove one job's folder now (consent withdrawn)")
    a = ap.parse_args()
    store = TrainingStore(a.store_dir)
    if a.job_id:
        if not a.job_id.isalnum():
            sys.exit("invalid job id")
        d = Path(a.store_dir) / "jobs" / a.job_id
        shutil.rmtree(d, ignore_errors=True)
        print(f"removed {a.job_id}" if not d.exists() else "could not remove")
    elif a.days is not None:
        print(f"removed {store.purge_older_than(a.days)} job folder(s)")
    else:
        sys.exit("give --days or --job-id")


if __name__ == "__main__":
    main()
