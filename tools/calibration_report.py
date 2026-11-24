#!/usr/bin/env python3
"""How often is a reading at a given confidence actually right? Run it on the training store before trusting a threshold.

Every correction records how confident the reader was (confidence, needsReview) and which of the fields the reviewer
checked had to be changed. Grouped by confidence, that is a reliability table:

    python tools/calibration_report.py --store-dir "$TRAINING_STORE_DIR" [--threshold 0.97] [--json out.json]

Read it like this: in the row for 0.95-1.00, "wrong" should be close to zero. If readings the service would have accepted
without review (confidence at or above the threshold, needsReview false) were corrected, the threshold is too low or the
confidence is not telling the truth. Only corrections a second person approved are counted. Only documents that reached a
reviewer are in the store, so the table says nothing
about documents nobody looked at: sample some accepted ones for review as well.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.training_store import approved_correction  # noqa: E402

BUCKETS = ((0.0, 0.5), (0.5, 0.8), (0.8, 0.9), (0.9, 0.95), (0.95, 1.0001))


def report(store_dir: Path, threshold: float) -> dict:
    rows = {f"{lo:.2f}-{min(hi, 1.0):.2f}": {"documents": 0, "wrong": 0} for lo, hi in BUCKETS}
    rows["no confidence"] = {"documents": 0, "wrong": 0}
    accepted = {"documents": 0, "wrong": 0}
    for job in sorted((store_dir / "jobs").glob("*")):
        # Only a correction a second person approved is evidence: a pending one is one reviewer key's word, and could say
        # that an accepted reading was right when nobody else has looked.
        rec = approved_correction(job)
        if rec is None or not isinstance(rec.get("ocr"), dict):
            continue
        ocr = rec["ocr"]
        if not ocr.get("fieldsChecked"):
            continue                                 # the reviewer confirmed no fields: nothing to learn from
        wrong = bool(ocr.get("fieldsChanged"))
        conf = ocr.get("confidence")
        key = "no confidence"
        if isinstance(conf, (int, float)):
            key = next(f"{lo:.2f}-{min(hi, 1.0):.2f}" for lo, hi in BUCKETS if lo <= conf < hi)
            if conf >= threshold and ocr.get("needsReview") is False:
                accepted["documents"] += 1
                accepted["wrong"] += wrong
        rows[key]["documents"] += 1
        rows[key]["wrong"] += wrong
    for r in list(rows.values()) + [accepted]:
        r["errorRate"] = round(r["wrong"] / r["documents"], 4) if r["documents"] else None
    return {"threshold": threshold, "byConfidence": rows, "acceptedWithoutReview": accepted}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store-dir", type=Path, required=True)
    ap.add_argument("--threshold", type=float, default=0.97)
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()
    out = report(a.store_dir, a.threshold)
    print(f"{'confidence':<16}{'documents':>10}{'wrong':>8}{'error rate':>12}")
    for name, r in out["byConfidence"].items():
        print(f"{name:<16}{r['documents']:>10}{r['wrong']:>8}{'-' if r['errorRate'] is None else r['errorRate']:>12}")
    acc = out["acceptedWithoutReview"]
    print(f"accepted without review at >= {a.threshold}: {acc['documents']} documents, {acc['wrong']} wrong")
    if a.json:
        a.json.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
