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

Checking is quicker on a page than in a spreadsheet. --review-page writes review.html into the output folder: each image
beside what was read from it, the values editable, one click to confirm. It opens from disk, sends nothing anywhere, and its
"Save sheet" button downloads the confirmed sheet.csv to put back in the output folder before --make-labels.

    python tools/read_folder.py /secure/testdata --out /secure/testdata-results --review-page
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
SHEET_FIELDS = ["documentNumber", "idNumber", "surname", "givenNames", "nameEn", "nameAr", "nationality", "birthDate",
                "expiryDate", "sex"]
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
    row = {"seconds": round(time.monotonic() - started, 2), "timings": res.get("timings"), "status": status, "code": res.get("code"), "engine": res.get("engine"),
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


REVIEW_PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Confirm what was read</title>
<style>
body{font:15px/1.4 system-ui,sans-serif;margin:0;background:#f4f5f7;color:#1b1f24}
header{position:sticky;top:0;background:#fff;border-bottom:1px solid #d0d4da;padding:10px 16px;display:flex;gap:12px;align-items:center;z-index:2}
header b{flex:1} button{font:inherit;padding:6px 14px;border:1px solid #8a93a0;border-radius:6px;background:#fff;cursor:pointer}
button.go{background:#0b6e4f;color:#fff;border-color:#0b6e4f}
.row{display:grid;grid-template-columns:minmax(0,1.3fr) minmax(280px,1fr);gap:16px;background:#fff;margin:14px 16px;padding:14px;border-radius:8px;border:2px solid #d0d4da}
.row.yes{border-color:#0b6e4f}.row.skip{opacity:.55}
.row img{max-width:100%;max-height:520px;cursor:zoom-in;border-radius:4px}
.row img.turn1{transform:rotate(90deg)}.row img.turn2{transform:rotate(180deg)}.row img.turn3{transform:rotate(270deg)}
label{display:grid;grid-template-columns:150px 1fr;gap:8px;margin:4px 0;align-items:center}
input{font:15px ui-monospace,Menlo,monospace;padding:5px 7px;border:1px solid #b6bcc6;border-radius:5px;width:100%;box-sizing:border-box}
.name{font-size:12px;color:#5b6470;word-break:break-all;margin-bottom:6px}.acts{margin-top:10px;display:flex;gap:8px;flex-wrap:wrap}
</style></head><body>
<header><b id="count"></b><button onclick="only()">Show only unconfirmed</button><button class="go" onclick="save()">Save sheet</button></header>
<div id="rows"></div>
<script>
const FIELDS=__FIELDS__, ROWS=__ROWS__, BASE=__BASE__, KEY="visa-ocr-review:"+location.pathname;
let state={}; try{state=JSON.parse(localStorage.getItem(KEY)||"{}")}catch(e){}
function keep(){try{localStorage.setItem(KEY,JSON.stringify(state))}catch(e){} count()}
function count(){const n=ROWS.filter(r=>(state[r.file]||{}).confirmed==="yes").length;
  document.getElementById("count").textContent=n+" of "+ROWS.length+" confirmed. Correct any wrong value, then press Confirm."}
function draw(){const box=document.getElementById("rows");
  ROWS.forEach((r,i)=>{const s=state[r.file]=state[r.file]||{};const d=document.createElement("div");d.className="row "+(s.confirmed||"");d.id="r"+i;
    const left=document.createElement("div");const img=document.createElement("img");img.loading="lazy";img.src=BASE+r.file.split("/").map(encodeURIComponent).join("/");
    let turn=0;img.onclick=()=>window.open(img.src);left.appendChild(img);
    const right=document.createElement("div");const nm=document.createElement("div");nm.className="name";nm.textContent=r.file+"  ("+(r.type||"")+")";right.appendChild(nm);
    FIELDS.forEach(f=>{const l=document.createElement("label");l.textContent=f;const inp=document.createElement("input");
      inp.value=(f in s)?s[f]:(r[f]||"");inp.oninput=()=>{s[f]=inp.value;keep()};if(f==="nameAr")inp.dir="rtl";l.appendChild(inp);right.appendChild(l)});
    const acts=document.createElement("div");acts.className="acts";
    const mk=(t,c,fn)=>{const b=document.createElement("button");b.textContent=t;if(c)b.className=c;b.onclick=fn;acts.appendChild(b)};
    mk("Confirm","go",()=>{s.confirmed="yes";d.className="row yes";keep();const n=document.getElementById("r"+(i+1));if(n)n.scrollIntoView({behavior:"smooth"})});
    mk("Cannot tell",null,()=>{s.confirmed="skip";d.className="row skip";keep()});
    mk("Undo",null,()=>{s.confirmed="";d.className="row";keep()});
    mk("Turn image",null,()=>{turn=(turn+1)%4;img.className=turn?"turn"+turn:""});
    right.appendChild(acts);d.appendChild(left);d.appendChild(right);box.appendChild(d)});count()}
function only(){ROWS.forEach((r,i)=>{if((state[r.file]||{}).confirmed)document.getElementById("r"+i).style.display="none"})}
function save(){const q=v=>'"'+String(v==null?"":v).replace(/"/g,'""')+'"';const head=["file","type",...FIELDS,"confirmed"];
  const lines=[head.join(",")];ROWS.forEach(r=>{const s=state[r.file]||{};
    lines.push([r.file,r.type||"",...FIELDS.map(f=>(f in s)?s[f]:(r[f]||"")),s.confirmed==="yes"?"yes":""].map(q).join(","))});
  const a=document.createElement("a");a.href=URL.createObjectURL(new Blob(["\\ufeff"+lines.join("\\r\\n")+"\\r\\n"],{type:"text/csv"}));a.download="sheet.csv";a.click()}
draw();
</script></body></html>
"""


def review_page(folder: Path, out: Path) -> int:
    import os
    with open(out / "sheet.csv", newline="", encoding="utf-8-sig") as fh:
        rows = [{k: (v or "") for k, v in r.items() if k != "confirmed"} for r in csv.DictReader(fh)]
    base = os.path.relpath(folder.resolve(), out.resolve()).replace(os.sep, "/") + "/"
    page = (REVIEW_PAGE.replace("__FIELDS__", json.dumps(SHEET_FIELDS)).replace("__BASE__", json.dumps(base))
            .replace("__ROWS__", json.dumps(rows, ensure_ascii=False).replace("</", "<\\/")))
    (out / "review.html").write_text(page, encoding="utf-8")
    print(f"{len(rows)} documents. Open {out / 'review.html'} in a browser. It holds personal data: keep it in this folder.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("folder", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--engine", default=None, help="engine name; default is OCR_ENGINE")
    ap.add_argument("--minutes", type=float, default=0, help="stop after this long (0 = until finished)")
    ap.add_argument("--make-labels", action="store_true", help="turn the confirmed rows of sheet.csv into labels.json")
    ap.add_argument("--review-page", action="store_true", help="write review.html: each image beside what was read, to confirm")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    if a.make_labels:
        return make_labels(a.out)
    if a.review_page:
        return review_page(a.folder, a.out)

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
