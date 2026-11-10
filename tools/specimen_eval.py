#!/usr/bin/env python3
"""Measure accuracy on a folder of REAL ID card images (Iqama, Saudi ID, other GCC IDs).

    python tools/specimen_eval.py --dir specimens/ --expected specimens/expected.csv

expected.csv columns: file, then any of idNumber, nameAr, nameEn, nationality, birthDate, expiryDate, occupation,
employer, birthPlace, issuePlace, religion (leave a cell empty to skip that field). Dates as YYYY-MM-DD.
Reports per-field accuracy, how many results were sent to review, and SILENT WRONG: a wrong field on a result
that was NOT flagged for review. Decide go-live only when silent-wrong is acceptable to the business owner.
"""
import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app import pipeline  # noqa: E402
from app.engines import get_engine  # noqa: E402
from app.review import Applicant  # noqa: E402


def norm(v) -> str:
    return " ".join(str(v or "").split()).casefold()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--expected", type=Path, required=True)
    ap.add_argument("--hint", default="iqama")
    ap.add_argument("--out", type=Path, default=Path("specimen_report.json"))
    a = ap.parse_args()
    rows = list(csv.DictReader(a.expected.open(encoding="utf-8-sig")))
    engine = get_engine()
    per_field: dict[str, list[bool]] = defaultdict(list)
    tally: Counter = Counter()
    detail = []
    for row in rows:
        data = (a.dir / row["file"]).read_bytes()
        status, payload = pipeline.run([data], a.hint, Applicant(), engine)
        if status != "done":
            tally["not_read"] += 1
            detail.append({"file": row["file"], "status": "not_read", "error": payload.get("code")})
            print(f"{row['file']}: NOT READ ({payload.get('code')})")
            continue
        doc, review = payload["document"], payload["needsReview"]
        wrong = []
        for field, want in row.items():
            if field == "file" or not (want or "").strip():
                continue
            ok = norm(doc.get(field)) == norm(want)
            per_field[field].append(ok)
            if not ok:
                wrong.append(field)
        tally["review" if review else "no_review"] += 1
        verdict = "OK" if not wrong else ("WRONG_FLAGGED" if review else "WRONG_SILENT")
        tally[verdict] += 1
        detail.append({"file": row["file"], "verdict": verdict, "wrong": wrong, "needsReview": review})
        print(f"{row['file']}: {verdict} {wrong}")
    print("\nPer-field accuracy")
    for f, oks in per_field.items():
        print(f"  {f:14s} {sum(oks)}/{len(oks)}  {100 * sum(oks) / len(oks):.0f}%")
    print(f"\nimages {len(rows)}  {dict(tally)}")
    a.out.write_text(json.dumps({"summary": dict(tally), "fields": {f: [sum(o), len(o)] for f, o in per_field.items()},
                                 "detail": detail}, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
