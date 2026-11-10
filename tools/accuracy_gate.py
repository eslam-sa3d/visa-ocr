#!/usr/bin/env python3
"""Pass/fail accuracy gate for CI and for go-live (checklist E1 and E2). Thresholds live in config/acceptance.json.

    python tools/accuracy_gate.py                 # generated documents (regression guard), exit 1 if a threshold is missed
    python tools/accuracy_gate.py --n 3           # per format and condition; 3 -> 60 images, about 2 minutes
    python tools/accuracy_gate.py --real-dir real-samples   # go-live bar on real specimens (reads scenarios/report.json)
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools"))


def synthetic(n: int, th: dict) -> list[str]:
    import e2e_accuracy as E
    failures = []
    jobs = [(f, k, v, 9000 + i * 7 + k) for i, f in enumerate(["TD3", "TD2", "TD1", "MRVA", "MRVB"])
            for k in range(n) for v in ("clean", "rot+4", "blur", "noise")]
    with Pool(2) as p:
        out = p.map(E.job, jobs, chunksize=1)
    total = len(out)
    not_read = sum(r["status"] != "done" for r in out)
    done = [r for r in out if r["status"] == "done"]
    silent = sum(1 for r in done if r["wrong"] and not r["needsReview"])
    correct = sum(1 for r in done if not r["wrong"])
    print(f"documents {total}: fully correct {correct / total:.1%}, not read {not_read / total:.1%}, silent wrong {silent}, "
          f"median {statistics.median(r['sec'] for r in out):.1f}s")
    if correct / total < th["min_fully_correct"]:
        failures.append(f"fully correct {correct / total:.1%} < {th['min_fully_correct']:.0%}")
    if not_read / total > th["max_not_read"]:
        failures.append(f"not read {not_read / total:.1%} > {th['max_not_read']:.0%}")
    if silent > th["max_silent_wrong"]:
        failures.append(f"{silent} silent wrong reads (allowed {th['max_silent_wrong']})")
    return failures


def real(real_dir: Path, th: dict) -> list[str]:
    report = json.loads((real_dir / "report.json").read_text(encoding="utf-8"))
    rows = report if isinstance(report, list) else report.get("results", report.get("scenarios", []))
    failures, fam = [], {}
    for r in rows:
        fam.setdefault(r.get("docType", "?"), []).append(r)
    for family, items in sorted(fam.items()):
        n = len(items)
        ok = sum(1 for r in items if r.get("verdict") == "PASS")
        silent = sum(1 for r in items if r.get("verdict") == "PARTIAL" and not r.get("needsReview", True))
        print(f"{family:28} n={n:3} pass {ok / n:.0%} silent-wrong {silent}")
        if n < th["min_specimens_per_family"]:
            failures.append(f"{family}: only {n} specimens (need {th['min_specimens_per_family']})")
        bar = th["min_iqama_key_fields"] if family.startswith(("iqama", "gcc_id")) else th["min_fully_correct_per_family"]
        if ok / n < bar:
            failures.append(f"{family}: {ok / n:.0%} < {bar:.0%}")
        if silent > th["max_silent_wrong"]:
            failures.append(f"{family}: {silent} silent wrong reads")
    return failures


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--real-dir", type=Path)
    a = ap.parse_args()
    cfg = json.loads((ROOT / "config" / "acceptance.json").read_text(encoding="utf-8"))
    failures = real(a.real_dir, cfg["real_specimens"]) if a.real_dir else synthetic(a.n, cfg["synthetic"])
    if failures:
        print("\nGATE FAILED:\n - " + "\n - ".join(failures)); sys.exit(1)
    print("\nGATE PASSED (" + ("real specimens" if a.real_dir else "generated documents") + ")")


if __name__ == "__main__":
    main()
