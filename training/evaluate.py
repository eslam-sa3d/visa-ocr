#!/usr/bin/env python3
"""Score recognition output against labels: character error rate (CER), exact match, per script.

Both files use the label format "<image path>\\t<text>". Predictions are matched by image path.

    python training/evaluate.py --labels data/rec/val_real.txt --predictions preds.txt --out metrics.json

Lower CER is better. A model is only worth promoting if CER on REAL validation lines improves.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.textutil import normalize_arabic, script_of, to_ascii_digits  # noqa: E402


def edit_distance(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def read(path: Path) -> dict[str, str]:
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "\t" in line:
            rel, text = line.split("\t", 1)
            rows[rel] = text
    return rows


def normalise(text: str, arabic: bool, digits: bool) -> str:
    if digits:
        text = to_ascii_digits(text)
    if arabic:
        text = normalize_arabic(text)
    return " ".join(text.split())


def evaluate(labels: dict[str, str], preds: dict[str, str], arabic: bool = False, digits: bool = False) -> dict:
    totals = defaultdict(lambda: {"samples": 0, "chars": 0, "edits": 0, "exact": 0})
    missing = 0
    for rel, ref in labels.items():
        if rel not in preds:
            missing += 1
            continue
        ref_n, hyp_n = normalise(ref, arabic, digits), normalise(preds[rel], arabic, digits)
        for bucket in ("all", script_of(ref)):
            t = totals[bucket]
            t["samples"] += 1
            t["chars"] += max(len(ref_n), 1)
            t["edits"] += edit_distance(ref_n, hyp_n)
            t["exact"] += ref_n == hyp_n

    def summary(t: dict) -> dict:
        return {"samples": t["samples"], "cer": round(t["edits"] / t["chars"], 4) if t["chars"] else None,
                "exactMatch": round(t["exact"] / t["samples"], 4) if t["samples"] else None}

    result = {"missingPredictions": missing, **summary(totals["all"])}
    result["byScript"] = {k: summary(v) for k, v in sorted(totals.items()) if k != "all"}
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--predictions", type=Path, required=True)
    ap.add_argument("--normalize-arabic", action="store_true",
                    help="ignore diacritics, tatweel and alef/yeh variants when comparing")
    ap.add_argument("--ascii-digits", action="store_true", help="compare Arabic-Indic digits as 0-9")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    metrics = evaluate(read(args.labels), read(args.predictions), args.normalize_arabic, args.ascii_digits)
    text = json.dumps(metrics, indent=2, ensure_ascii=False)
    print(text)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
