#!/usr/bin/env python3
"""How many documents a running service reads a minute, and how long an applicant waits, at rising numbers of clients.

Run it against the real servers before go-live. It needs only Python 3 and the "requests" package, and a folder of sample
images (your own photos of documents give the true figure; generated ones read faster).

    python tools/capacity_test.py --url https://<host> --key <portal key> --images /path/to/sample/images
    options: --levels 1,4,16,64   clients uploading at the same time, one test for each number
             --uploads 6          uploads per client at each level
             --insecure           accept a self-signed certificate
             --json result.json   also write the figures to a file

What to read from the table: "per minute" stops rising at the service's capacity; from there on, more clients only wait
longer. Size the servers so that the expected uploads a minute at peak are about 60% of that capacity. "busy" counts uploads
the service turned away because its queue was full; "unclear" counts photos refused at once as unusable.
Nothing about the documents is printed or written: only counts and seconds.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

import requests

IMAGE_TYPES = (".jpg", ".jpeg", ".png")


def one(a, session, name: str, data: bytes) -> tuple[str, float]:
    headers = {"X-API-Key": a.key} if a.key else {}
    kind = "image/png" if name.lower().endswith(".png") else "image/jpeg"
    started = time.monotonic()
    try:
        r = session.post(a.url + a.endpoint, files=[("files", (name, data, kind))], headers=headers, timeout=120, verify=not a.insecure)
        if r.status_code == 503:
            return "busy", time.monotonic() - started
        if r.status_code == 422 and (r.json().get("detail") or {}).get("code") == "image_unclear":
            return "unclear", time.monotonic() - started
        if r.status_code != 202:
            return f"http {r.status_code}", time.monotonic() - started
        job = r.json()["jobId"]
        while time.monotonic() - started < a.timeout:
            j = session.get(f"{a.url}/v1/jobs/{job}", headers=headers, timeout=60, verify=not a.insecure).json()
            if j.get("status") in ("done", "failed"):
                return ("read" if j["status"] == "done" else "turned back"), time.monotonic() - started
            time.sleep(0.4)
        return "timeout", time.monotonic() - started
    except requests.RequestException as exc:
        return "error " + type(exc).__name__, time.monotonic() - started


def level(a, images: list[tuple[str, bytes]], clients: int) -> dict:
    total = clients * a.uploads
    started = time.monotonic()
    with requests.Session() as session, cf.ThreadPoolExecutor(clients) as pool:
        results = list(pool.map(lambda i: one(a, session, *images[i % len(images)]), range(total)))
    wall = time.monotonic() - started
    answered = sorted(t for kind, t in results if kind in ("read", "turned back"))
    counts = Counter(kind for kind, _ in results)
    return {"clients": clients, "uploads": total, "read": counts.get("read", 0), "turnedBack": counts.get("turned back", 0),
            "busy": counts.get("busy", 0), "unclear": counts.get("unclear", 0),
            "other": {k: v for k, v in counts.items() if k not in ("read", "turned back", "busy", "unclear")},
            "perMinute": round(len(answered) / wall * 60, 1) if wall else 0.0,
            "medianSeconds": round(statistics.median(answered), 1) if answered else None,
            "p95Seconds": round(answered[max(0, int(len(answered) * 0.95) - 1)], 1) if answered else None}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--key")
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--levels", default="1,4,16,64")
    ap.add_argument("--uploads", type=int, default=6)
    ap.add_argument("--endpoint", default="/v1/documents/extract", help="or /v1/passports/extract, /v1/cards/extract")
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--insecure", action="store_true")
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()
    a.url = a.url.rstrip("/")
    if a.insecure:
        requests.packages.urllib3.disable_warnings()                      # the operator asked for it: one notice is enough
        print("certificate checks are off (--insecure)")
    images = [(p.name, p.read_bytes()) for p in sorted(a.images.rglob("*")) if p.suffix.lower() in IMAGE_TYPES]
    if not images:
        print(f"no JPEG or PNG images under {a.images}")
        return 2
    try:
        ready = requests.get(a.url + "/readyz", timeout=20, verify=not a.insecure).json()
    except requests.RequestException as exc:
        print(f"the service does not answer at {a.url}: {type(exc).__name__}")
        return 2
    print(f"service: {ready.get('status')}, readers: {ready.get('readers', 'not reported')} | sample images: {len(images)}")
    one(a, requests.Session(), *images[0])                                 # the first read after a start is not typical
    rows = []
    print(f"{'clients':>8} {'uploads':>8} {'read':>6} {'turned back':>12} {'busy':>6} {'unclear':>8} {'per minute':>11} {'median s':>9} {'95% within s':>13}")
    for clients in [int(v) for v in a.levels.split(",") if v.strip()]:
        row = level(a, images, clients)
        rows.append(row)
        print(f"{row['clients']:>8} {row['uploads']:>8} {row['read']:>6} {row['turnedBack']:>12} {row['busy']:>6} {row['unclear']:>8} "
              f"{row['perMinute']:>11} {str(row['medianSeconds']):>9} {str(row['p95Seconds']):>13}" + (f"  {row['other']}" if row["other"] else ""))
    best = max((r["perMinute"] for r in rows), default=0.0)
    print(f"\ncapacity: about {best} documents a minute. Plan for a peak of about {round(best * 0.6, 1)} a minute (60% of it).")
    if a.json:
        a.json.write_text(json.dumps({"url": a.url, "readers": ready.get("readers"), "images": len(images), "levels": rows,
                                      "capacityPerMinute": best}, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
