#!/usr/bin/env python3
"""Read every image in a folder with the configured engine and write, in an output folder you choose:

    results.jsonl   one line per image: what happened, with no personal data (status, format, checks, flags, seconds)
    sheet.csv       what was read from each image, with an empty "confirmed" column for a person to fill in
    summary.txt     counts per sub-folder

    python tools/read_folder.py /secure/testdata --out /secure/testdata-results [--engine auto] [--minutes 30]

The run can be stopped and started again: images already in results.jsonl are skipped. sheet.csv holds personal data, so keep
the output folder next to the images, outside the repository.

A person then checks sheet.csv against the documents: correct any wrong value in place and write "yes" in the confirmed column.
Rows marked yes become the answer file that tools/accuracy_real.py scores against:

    python tools/read_folder.py /secure/testdata --out /secure/testdata-results --make-labels
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

MRZ_FORMATS = {"TD1", "TD2", "TD3", "MRVA", "MRVB"}
SHEET_FIELDS = ["documentNumber", "idNumber", "surname", "givenNames", "nationality", "birthDate", "expiryDate", "sex"]
SHEET_HEADER = ["file", "type", *SHEET_FIELDS, "confirmed"]
IMAGE_TYPES = (".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp")


def _images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_TYPES and not p.name.startswith("."))


def _read_one(path: Path, engine) -> tuple[dict, dict]:
    """(row with no personal data, the fields read)."""
    from app import pipeline
    from app.review import Applicant
    from app.uploads import shrink_for_ocr, validate_image
    started = time.monotonic()
    try:
        data = path.read_bytes()
        try:
            validate_image(data, None)
        except Exception as exc:  # noqa: BLE001 - a rejected upload is a result, not a crash
            status, res = "rejected", {"code": getattr(exc, "code", None) or type(exc).__name__}
        else:
            status, res = pipeline.run([shrink_for_ocr(data)], None, Applicant(), engine)
    except Exception as exc:  # noqa: BLE001 - one bad image must not stop the run
        status, res = "crash", {"code": type(exc).__name__}
    doc = res.get("document") or {}
    checks = doc.get("checks") or {}
    row = {"seconds": round(time.monotonic() - started, 2), "status": status, "code": res.get("code"), "engine": res.get("engine"),
           "format": doc.get("format"), "kind": doc.get("kind") or doc.get("documentType"),
           "checksPass": all(checks.values()) if checks else None, "needsReview": res.get("needsReview"),
           "errors": [f["code"] for f in res.get("flags", []) if f.get("severity") == "error"]}
    fields = {k: doc.get(k, "") for k in SHEET_FIELDS}
    fields["type"] = row["kind"] or ""
    return row, fields


def summarise(rows: dict[str, dict]) -> str:
    groups: dict[str, Counter] = defaultdict(Counter)
    for name, r in rows.items():
        c = groups[name.split("/")[0] if "/" in name else "."]
        c["images"] += 1
        if r.get("format") in MRZ_FORMATS:
            c["mrz"] += 1
            c["checks pass" if r.get("checksPass") else "check fails"] += 1
            c["accepted"] += r.get("needsReview") is False
        elif r.get("status") == "done":
            c["printed text"] += 1
        else:
            c["not read"] += 1
    cols = ["images", "mrz", "checks pass", "check fails", "accepted", "printed text", "not read"]
    width = max(len(g) for g in groups) if groups else 6
    out = [f"{'folder':<{width}}  " + "  ".join(f"{c:>12}" for c in cols)]
    total: Counter = Counter()
    for g in sorted(groups):
        out.append(f"{g:<{width}}  " + "  ".join(f"{groups[g][c]:>12}" for c in cols))
        total.update(groups[g])
    out.append(f"{'TOTAL':<{width}}  " + "  ".join(f"{total[c]:>12}" for c in cols))
    engines = Counter(r.get("engine") or "none" for r in rows.values())
    out.append("")
    out.append("engine that produced each result: " + ", ".join(f"{k} {v}" for k, v in engines.most_common()))
    return "\n".join(out) + "\n"


def make_labels(out: Path) -> int:
    labels = {}
    with open(out / "sheet.csv", newline="", encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            if (r.get("confirmed") or "").strip().lower() in ("yes", "y", "1", "true"):
                entry = {k: r[k].strip() for k in SHEET_FIELDS if (r.get(k) or "").strip()}
                if r.get("type"):
                    entry["type"] = r["type"].strip()
                labels[r["file"]] = entry
    (out / "labels.json").write_text(json.dumps(labels, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(labels)} confirmed documents written to {out / 'labels.json'}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("folder", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--engine", default=None, help="engine name; default is OCR_ENGINE")
    ap.add_argument("--minutes", type=float, default=0, help="stop after this long (0 = until finished)")
    ap.add_argument("--make-labels", action="store_true", help="turn the confirmed rows of sheet.csv into labels.json")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    if a.make_labels:
        return make_labels(a.out)

    results, sheet = a.out / "results.jsonl", a.out / "sheet.csv"
    rows: dict[str, dict] = {}
    if results.exists():
        for line in results.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            rows[r["file"]] = r
    todo = [p for p in _images(a.folder) if p.relative_to(a.folder).as_posix() not in rows]
    print(f"{len(rows)} already read, {len(todo)} to read", flush=True)
    if todo:
        from app.engines import get_engine, get_engine_by_name
        engine = get_engine_by_name(a.engine) if a.engine else get_engine()
        new_sheet = not sheet.exists()
        started = time.monotonic()
        with open(results, "a", encoding="utf-8") as rf, open(sheet, "a", newline="", encoding="utf-8-sig") as sf:
            writer = csv.DictWriter(sf, fieldnames=SHEET_HEADER)
            if new_sheet:
                writer.writeheader()
            for n, path in enumerate(todo, 1):
                if a.minutes and time.monotonic() - started > a.minutes * 60:
                    print("time is up; run the same command again to continue")
                    break
                name = path.relative_to(a.folder).as_posix()
                row, fields = _read_one(path, engine)
                rows[name] = {"file": name, **row}
                rf.write(json.dumps(rows[name], ensure_ascii=False) + "\n")
                rf.flush()
                if row["status"] == "done":
                    writer.writerow({"file": name, **fields, "confirmed": ""})
                    sf.flush()
                if n % 10 == 0:
                    print(f"  {n}/{len(todo)}", flush=True)
    text = summarise(rows)
    (a.out / "summary.txt").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
