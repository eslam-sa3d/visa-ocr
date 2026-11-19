#!/usr/bin/env python3
"""Why did PaddleOCR hand a document over to the second pair of readers? Run after tools/read_folder.py.

For the images whose result came from the Tesseract + RapidOCR pair, PaddleOCR reads the image again on its own and the reason
it had no usable MRZ is recorded. Nothing personal is written: only counts, line lengths and check-digit totals.

    python tools/paddle_handover.py /secure/testdata --out /secure/testdata-results [--every 3]

Writes handover.jsonl and prints a summary. --every 3 takes every third such image (a sample, to keep the run short).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def reason_for(out, check_score, clean_line) -> dict:
    """Why PaddleOCR's own reading was not used, from its output alone."""
    lines = [clean_line(str(l.get("text", ""))) for l in (out.fields or {}).get("textLines") or []]
    like = [l for l in lines if len(l) >= 20 and re.fullmatch(r"[A-Z0-9<]+", l) and "<" in l]
    row = {"textLines": len(lines), "mrzLikeLengths": sorted(len(l) for l in like),
           "fillerRuns": sorted((max((len(r) for r in re.findall(r"<+", l)), default=0) for l in like), reverse=True)[:3]}
    if out.mrz_lines:
        passing, perfect = check_score(out.mrz_lines)
        row.update(reason="read, all checks pass" if perfect else "read, a check digit fails", checksPassing=passing,
                   confidence=out.mrz_confidence)
    elif not lines:
        row["reason"] = "no text found at all"
    elif not like:
        row["reason"] = "text found, but no line that looks like an MRZ"
    else:
        row["reason"] = "MRZ-like lines found, but not the right number or length"
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("folder", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--every", type=int, default=1)
    a = ap.parse_args()
    from PIL import Image
    from app.engines import get_engine_by_name
    from app.mrz import check_score, clean_line
    from app.uploads import shrink_for_ocr
    results = [json.loads(l) for l in (a.out / "results.jsonl").read_text(encoding="utf-8").splitlines()]
    todo = [r for r in results if r.get("engine") == "ensemble"][::max(1, a.every)]
    target = a.out / "handover.jsonl"
    done = {json.loads(l)["file"] for l in target.read_text(encoding="utf-8").splitlines()} if target.exists() else set()
    todo = [r for r in todo if r["file"] not in done]
    print(f"{len(done)} already done, {len(todo)} to read", flush=True)
    engine = get_engine_by_name("paddle-ocr") if todo else None
    with open(target, "a", encoding="utf-8") as fh:
        for n, r in enumerate(todo, 1):
            path = a.folder / r["file"]
            started = time.monotonic()
            row = {"file": r["file"], "pairFormat": r.get("format"), "pairChecksPass": r.get("checksPass")}
            try:
                data = path.read_bytes()
                with Image.open(path) as im:
                    row["size"] = list(im.size)
                small = shrink_for_ocr(data)
                with Image.open(__import__("io").BytesIO(small)) as im:
                    row["workSize"] = list(im.size)
                row.update(reason_for(engine.extract([small], None), check_score, clean_line))
            except Exception as exc:  # noqa: BLE001 - the failure is the finding
                row["reason"] = "PaddleOCR failed: " + type(exc).__name__
            row["seconds"] = round(time.monotonic() - started, 1)
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            if n % 10 == 0:
                print(f"  {n}/{len(todo)}", flush=True)
    rows = [json.loads(l) for l in target.read_text(encoding="utf-8").splitlines()]
    print()
    for reason, count in Counter(r["reason"] for r in rows).most_common():
        print(f"{count:>4}  {reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
