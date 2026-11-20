#!/usr/bin/env python3
"""Accuracy on REAL documents: per-field accuracy, documents fully correct, how many were accepted without review, and the
number that matters most: fields accepted without review that were WRONG.

Keep the folder outside the repository (it is personal data). It holds the images and a labels.json:

    {"passport_eg_01.jpg": {"documentNumber": "A1234567", "surname": "HASSAN", "givenNames": "AHMED ALI",
                            "nationality": "EGY", "birthDate": "1990-05-05", "expiryDate": "2030-01-01", "sex": "M",
                            "type": "passport"},
     "iqama_01.jpg": {"idNumber": "2123456789", "hint": "iqama", "type": "iqama"}}

Only the fields you list are scored. "hint" is sent as doc_type_hint; "type" groups the report.

    python tools/accuracy_real.py /secure/specimens [--engine ensemble] [--json report.json]

The exit code is 1 when any field was accepted without review and wrong. tests/test_review_fixes.py runs the same check when
REAL_SAMPLES_DIR is set, so it can be a release gate.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

META = ("hint", "type")


def run(folder: Path, engine_name: str | None = None, with_form: bool = False, labels_file: Path | None = None) -> dict:
    from app import pipeline
    from app.engines import get_engine, get_engine_by_name
    from app.review import Applicant
    labels = json.loads((labels_file or folder / "labels.json").read_text(encoding="utf-8"))
    engine = get_engine_by_name(engine_name) if engine_name else get_engine()
    fields = defaultdict(lambda: {"total": 0, "correct": 0})
    types = defaultdict(lambda: {"documents": 0, "correct": 0, "acceptedWithoutReview": 0})
    wrongly_accepted, unread, docs_correct, accepted = [], [], 0, 0
    for name, truth in sorted(labels.items()):
        # --with-form: what the portal does in production, sending what the applicant typed. Fields no check digit covers
        # (name, nationality) are then confirmed by the form instead of waiting for a person.
        form = Applicant(surname=truth.get("surname"), given_names=truth.get("givenNames"), nationality=truth.get("nationality"),
                         document_number=truth.get("documentNumber")) if with_form else Applicant()
        status, result = pipeline.run([(folder / name).read_bytes()], truth.get("hint"), form, engine)
        kind = truth.get("type", "document")
        types[kind]["documents"] += 1
        doc = result.get("document", {}) if status == "done" else {}
        if status != "done":
            unread.append(name)
        review = bool(result.get("needsReview", True)) if status == "done" else True
        all_ok = status == "done"
        for key, want in truth.items():
            if key in META:
                continue
            ok = str(doc.get(key, "") or "").strip().upper() == str(want).strip().upper()
            fields[key]["total"] += 1
            fields[key]["correct"] += ok
            all_ok = all_ok and ok
            if not ok and status == "done" and not review:
                wrongly_accepted.append({"file": name, "field": key})       # values are not reported: personal data
        docs_correct += all_ok
        types[kind]["correct"] += all_ok
        if status == "done" and not review:
            accepted += 1
            types[kind]["acceptedWithoutReview"] += 1
    return {"documents": len(labels), "documentsCorrect": docs_correct, "acceptedWithoutReview": accepted,
            "unread": unread, "wronglyAccepted": wrongly_accepted,
            "fields": {k: {**v, "accuracy": round(v["correct"] / v["total"], 4)} for k, v in sorted(fields.items())},
            "byType": dict(types)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", type=Path)
    ap.add_argument("--engine")
    ap.add_argument("--json", type=Path)
    ap.add_argument("--labels", type=Path, help="answer file, when it is not labels.json inside the folder")
    ap.add_argument("--with-form", action="store_true",
                    help="send the correct name, nationality and number as the application form, as the portal does")
    a = ap.parse_args()
    report = run(a.folder, a.engine, a.with_form, a.labels)
    print(json.dumps(report, indent=2))
    if a.json:
        a.json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    sys.exit(1 if report["wronglyAccepted"] else 0)


if __name__ == "__main__":
    main()
