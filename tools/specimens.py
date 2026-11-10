#!/usr/bin/env python3
"""Plan, label and track the real-document test set (checklist group C).

    python tools/specimens.py init  real-samples         # folders, quota table, expected.json skeleton, README
    python tools/specimens.py add   real-samples IMG.jpg --type passport:TD3 --country EGY --condition phone \
                                    --doc-no AB1234567 --birth 1990-01-31 --expiry 2030-05-01 [--surname X --given Y --nat EGY]
    python tools/specimens.py status real-samples        # how many collected per type/country group vs the quota
    python tools/specimens.py draft real-samples         # fills expected.json guesses from the service itself (REVIEW EVERY LINE)

`add` copies the image into the set and records the values a person READ OFF THE DOCUMENT. Never record what the service
returned: the point of the set is to measure the service against the truth. Keep the folder off the internet-facing host
and out of git (it holds personal data). Then run:  python tools/run_scenarios.py --real-dir real-samples
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

QUOTA = 30
# type -> country groups that matter for this programme
PLAN = {
    "passport:TD3": ["SAU", "EGY", "IND", "PAK", "PHL", "BGD", "LKA", "NPL", "IDN", "SDN", "YEM", "SYR", "JOR", "TUR", "EUROPE", "OTHER"],
    "visa:MRVA": ["SCHENGEN", "UK", "OTHER"],
    "visa:MRVB": ["SCHENGEN", "OTHER"],
    "id_card:TD1": ["USA-GREENCARD", "ARE", "OTHER"],
    "id_card:TD2": ["EGY", "OTHER"],
    "iqama:visual": ["NEW", "OLD"],
    "gcc_id:SAU-NATIONAL-ID": ["FRONT", "BACK"],
    "gcc_id:ARE-EMIRATES-ID": ["FRONT", "BACK"],
    "gcc_id:QAT-QID": ["FRONT", "BACK"],
    "gcc_id:KWT-CIVIL-ID": ["FRONT", "BACK"],
    "gcc_id:BHR-CPR": ["FRONT", "BACK"],
    "gcc_id:OMN-ID": ["FRONT", "BACK"],
}
CONDITIONS = ["scan", "phone", "glare", "folded", "low_light"]
README = """# Real specimen set (personal data: keep private)
- Use only documents whose holder gave written consent, or your own. Never commit this folder.
- One image per file. Name files anything; `tools/specimens.py add` copies and registers them.
- The expected values are what a person reads off the PAPER, not what the service returned.
- Aim: %d per type and country group, spread over scan, phone photo, glare, folded, low light.
- Delete images when the test campaign ends (or on the holder's request).
"""


def load(d: Path) -> dict:
    p = d / "expected.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {"scenarios": []}


def save(d: Path, m: dict) -> None:
    (d / "expected.json").write_text(json.dumps(m, indent=1, ensure_ascii=False), encoding="utf-8")


def cmd_init(a) -> None:
    d = Path(a.dir); (d / "images").mkdir(parents=True, exist_ok=True)
    (d / "README.md").write_text(README % QUOTA, encoding="utf-8")
    if not (d / "expected.json").exists():
        save(d, {"scenarios": []})
    rows = ["type,group,quota"] + [f"{t},{g},{QUOTA}" for t, gs in PLAN.items() for g in gs]
    (d / "quota.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"created {d}: {len(rows) - 1} groups x {QUOTA} = {(len(rows) - 1) * QUOTA} specimens in the full plan")


def cmd_add(a) -> None:
    d = Path(a.dir); m = load(d)
    src = Path(a.image)
    n = sum(1 for s in m["scenarios"] if s["docType"] == a.type and s.get("group") == a.country) + 1
    sid = f"{a.type.replace(':', '_')}_{a.country}_{n:03d}"
    dst = d / "images" / (sid + src.suffix.lower())
    shutil.copy2(src, dst)
    exp = {k: v for k, v in {"documentNumber": a.doc_no, "idNumber": a.id_no, "surname": a.surname, "givenNames": a.given,
                             "nationality": a.nat, "birthDate": a.birth, "expiryDate": a.expiry}.items() if v}
    m["scenarios"].append({"id": sid, "group": a.country, "docType": a.type, "variant": a.condition,
                           "image": f"images/{dst.name}", "expectStatus": "done", "expected": exp})
    save(d, m); print("added", sid)


def cmd_status(a) -> None:
    d = Path(a.dir); m = load(d)
    have: dict[tuple, int] = {}
    for s in m["scenarios"]:
        have[(s["docType"], s.get("group"))] = have.get((s["docType"], s.get("group")), 0) + 1
    total = done = 0
    for t, gs in PLAN.items():
        for g in gs:
            n = have.get((t, g), 0); total += QUOTA; done += min(n, QUOTA)
            print(f"{t:26} {g:14} {n:3}/{QUOTA} {'OK' if n >= QUOTA else ''}")
    print(f"\ncollected {done}/{total} ({100 * done // total}%) of the full plan")
    conds = {}
    for s in m["scenarios"]:
        conds[s["variant"]] = conds.get(s["variant"], 0) + 1
    print("conditions:", conds or "none yet")


def cmd_draft(a) -> None:
    """Pre-fill empty expected values from the service. These are GUESSES to speed up typing; a person must correct them."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from app.engines import get_engine
    from app.pipeline import run
    from app.review import Applicant
    d = Path(a.dir); m = load(d)
    for s in m["scenarios"]:
        if s["expected"]:
            continue
        st, res = run([(d / s["image"]).read_bytes()], None, Applicant(), get_engine())
        doc = res.get("document", {}) if st == "done" else {}
        s["expected"] = {k: doc[k] for k in ("documentNumber", "surname", "givenNames", "nationality", "birthDate", "expiryDate", "idNumber") if doc.get(k)}
        s["DRAFT_FROM_SERVICE_REVIEW_ME"] = True
    save(d, m); print("drafted; every entry marked DRAFT_FROM_SERVICE_REVIEW_ME must be checked against the paper, then delete the marker")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for n in ("init", "status", "draft"):
        sub.add_parser(n).add_argument("dir")
    p = sub.add_parser("add"); p.add_argument("dir"); p.add_argument("image")
    p.add_argument("--type", required=True, choices=sorted(PLAN)); p.add_argument("--country", required=True)
    p.add_argument("--condition", default="phone", choices=CONDITIONS)
    for f in ("doc-no", "id-no", "surname", "given", "nat", "birth", "expiry"):
        p.add_argument("--" + f)
    a = ap.parse_args()
    {"init": cmd_init, "add": cmd_add, "status": cmd_status, "draft": cmd_draft}[a.cmd](a)


if __name__ == "__main__":
    main()
