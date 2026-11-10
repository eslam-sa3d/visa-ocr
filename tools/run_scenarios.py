#!/usr/bin/env python3
"""Run every scenario image through the service (in-process, same code as the real API) and write a report.

    python tools/run_scenarios.py                       # all synthetic scenarios, default engine
    python tools/run_scenarios.py --engine tesseract-mrz
    python tools/run_scenarios.py --only passport --variants clean,blur
    python tools/run_scenarios.py --real-dir real-samples   # your own images, see docs/TEST_PLAN.md
    python tools/run_scenarios.py --min-pass 0.8         # exit code 1 if the pass rate is lower

Outcome per scenario:  PASS = every expected field matches;  PARTIAL = document read, some field wrong;
FAIL = not read (or wrong status).  Report: scenarios/report.md and scenarios/report.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / "scenarios"


def run_one(client, path: Path, hint: str | None) -> tuple[dict, float]:
    ctype = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    t0 = time.time()
    r = client.post("/v1/documents/extract", files=[("files", (path.name, path.read_bytes(), ctype))],
                    data={"doc_type_hint": hint} if hint else {})
    if r.status_code != 202:
        return {"status": "rejected", "error": {"code": str(r.status_code), "message": json.dumps(r.json())[:200]}}, time.time() - t0
    jid = r.json()["jobId"]
    for _ in range(300):
        j = client.get(f"/v1/jobs/{jid}").json()
        if j["status"] in ("done", "failed"):
            return j, time.time() - t0
        time.sleep(0.5)
    return {"status": "timeout"}, time.time() - t0


def judge(sc: dict, job: dict) -> tuple[str, list[str]]:
    status = job.get("status")
    if sc["expectStatus"] == "failed":
        return ("PASS", []) if status in ("failed", "rejected") else ("FAIL", [f"expected a clean failure, got {status}"])
    if status != "done":
        err = (job.get("error") or {})
        return "FAIL", [f"status {status}: {err.get('code', '')} {err.get('message', '')[:90]}"]
    doc = job["result"]["document"]
    wrong = [f"{k}: expected {v!r}, got {doc.get(k)!r}" for k, v in sc["expected"].items() if doc.get(k) != v]
    return ("PASS", []) if not wrong else ("PARTIAL", wrong)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", help="OCR_ENGINE to use (default: auto)")
    ap.add_argument("--only", help="substring of the scenario id or document type")
    ap.add_argument("--variants", help="comma list, e.g. clean,blur")
    ap.add_argument("--real-dir", type=Path, help="folder with your own images and expected.json")
    ap.add_argument("--min-pass", type=float, default=0.0)
    args = ap.parse_args()
    if args.engine:
        os.environ["OCR_ENGINE"] = args.engine
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

    from fastapi.testclient import TestClient
    import app.main as main
    client = TestClient(main.app)

    if args.real_dir:
        manifest = json.loads((args.real_dir / "expected.json").read_text(encoding="utf-8"))["scenarios"]
        base = args.real_dir
    else:
        manifest = json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))["scenarios"]
        base = OUT
    if args.only:
        manifest = [s for s in manifest if args.only in s["id"] or args.only in s["docType"]]
    if args.variants:
        keep = set(args.variants.split(","))
        manifest = [s for s in manifest if s["variant"] in keep]

    results = []
    for n, sc in enumerate(manifest, 1):
        job, secs = run_one(client, base / sc["image"], sc.get("hint"))
        verdict, notes = judge(sc, job)
        res = (job.get("result") or {})
        results.append({**{k: sc[k] for k in ("id", "group", "docType", "variant")}, "verdict": verdict, "notes": notes,
                        "seconds": round(secs, 1), "engine": res.get("engine", ""),
                        "flags": [f["code"] for f in res.get("flags", [])], "needsReview": res.get("needsReview")})
        print(f"[{n}/{len(manifest)}] {verdict:7} {sc['id']}  {secs:4.1f}s  " + ("; ".join(notes)[:110]), flush=True)

    total = len(results)
    counts = defaultdict(int)
    for r in results:
        counts[r["verdict"]] += 1
    rate = counts["PASS"] / total if total else 0.0
    lines = ["# Scenario test report", "",
             f"Engine setting: `{os.environ.get('OCR_ENGINE', 'auto')}`  |  scenarios: {total}  |  "
             f"PASS {counts['PASS']}  PARTIAL {counts['PARTIAL']}  FAIL {counts['FAIL']}  |  pass rate {rate:.0%}", ""]
    mark = {"PASS": "✓", "PARTIAL": "~", "FAIL": "✗"}
    variants = list(dict.fromkeys(r["variant"] for r in results if r["group"] != "negative"))
    types = list(dict.fromkeys(r["docType"] for r in results if r["group"] != "negative"))
    lines += ["## Matrix (✓ all fields right, ~ read but a field is wrong, ✗ not read)", "",
              "| Document type | " + " | ".join(variants) + " |", "|---|" + "---|" * len(variants)]
    for t in types:
        cells = []
        for v in variants:
            hit = [r for r in results if r["docType"] == t and r["variant"] == v]
            cells.append(" ".join(mark[r["verdict"]] for r in hit) or "–")
        lines.append(f"| {t} | " + " | ".join(cells) + " |")
    lines += ["", "## Pass rate by variant (bad-photo condition)", "", "| Variant | Pass | Total |", "|---|---|---|"]
    for v in variants:
        rs = [r for r in results if r["variant"] == v]
        lines.append(f"| {v} | {sum(r['verdict'] == 'PASS' for r in rs)} | {len(rs)} |")
    neg = [r for r in results if r["group"] == "negative"]
    if neg:
        lines += ["", "## Negative cases (must fail cleanly, never crash or invent data)", ""]
        lines += [f"- {mark[r['verdict']]} {r['id']}" for r in neg]
    bad = [r for r in results if r["verdict"] != "PASS"]
    if bad:
        lines += ["", "## Failures and partial reads", ""]
        for r in bad:
            lines.append(f"- **{r['id']}** ({r['verdict']}): " + "; ".join(r["notes"]))
    if results:
        lines += ["", f"Average time per image: {sum(r['seconds'] for r in results) / total:.1f}s "
                      f"(slowest {max(r['seconds'] for r in results):.1f}s)."]
    (OUT if not args.real_dir else args.real_dir).mkdir(exist_ok=True)
    out_dir = args.real_dir or OUT
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out_dir / "report.json").write_text(json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8")
    print("\n".join(lines[:4]))
    print(f"report: {out_dir / 'report.md'}")
    sys.exit(1 if rate < args.min_pass else 0)


if __name__ == "__main__":
    main()
