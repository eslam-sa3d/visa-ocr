#!/usr/bin/env python3
"""Performance test against a RUNNING server. Generated documents only. Times are what a client sees: upload until the job is done.

  python tools/perf_test.py --url http://127.0.0.1:8000 [--key KEY] --phase latency --n 10
  python tools/perf_test.py --url ... --phase load --levels 1,2,4,8 --n 40
  python tools/perf_test.py --url ... --phase burst --n 80                 # more uploads at once than the pending cap
  python tools/perf_test.py --url ... --phase soak --seconds 180 --pid <server pid> [--pid <worker pid>]
Every result is checked: a job that ends 'done' must carry the right document number, or it is counted as WRONG.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import io
import json
import random
import statistics
import sys
import threading
import time
from pathlib import Path

import requests
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import accuracy_matrix as A  # noqa: E402
import make_scenarios as M  # noqa: E402


def jpg(img, quality=90) -> bytes:
    b = io.BytesIO()
    img.convert("RGB").save(b, "JPEG", quality=quality)
    return b.getvalue()


def make(kind: str, seed: int):
    """Return (image bytes, form data, key field name, expected value)."""
    if kind == "iqama":
        rng = random.Random(seed)
        p = dict(M.IQAMA_PEOPLE[seed % len(M.IQAMA_PEOPLE)])
        p["id"] = M.make_id("2" + "".join(rng.choice("0123456789") for _ in range(8)))
        return jpg(M.draw_iqama(p)), {"doc_type_hint": "iqama"}, "idNumber", p["id"]
    fmt = "TD3" if kind == "phone" else kind
    rng = random.Random(seed)
    kw = A.rnd_case(rng, fmt)
    if fmt in ("TD1", "TD2"):
        kw["doc"] = A.rnd_doc(rng, fmt)
    lines, truth = A.build(fmt, **kw)
    img = M.draw_document({"lines": lines, "label": f"{fmt} TEST"})
    if kind == "phone":                                   # a 12-megapixel phone photo with the page inside it
        page = img.resize((3400, round(img.height * 3400 / img.width)), Image.LANCZOS)
        canvas = Image.new("RGB", (4000, 3000), (120, 110, 100))
        canvas.paste(page, (300, max(0, (3000 - page.height) // 2)))
        return jpg(canvas, 88), {}, "documentNumber", truth["documentNumber"]
    return jpg(img), {}, "documentNumber", truth["documentNumber"]


def pct(vals, p):
    if not vals:
        return None
    s = sorted(vals)
    return round(s[min(len(s) - 1, int(round(p / 100 * (len(s) - 1))))], 2)


class Client:
    def __init__(self, a) -> None:
        self.url, self.h, self.timeout = a.url.rstrip("/"), ({"X-API-Key": a.key} if a.key else {}), a.timeout

    def one(self, item) -> dict:
        data, form, field, expect = item
        t0 = time.time()
        try:
            r = requests.post(self.url + "/v1/documents/extract", files=[("files", ("d.jpg", data, "image/jpeg"))], data=form,
                              headers=self.h, timeout=120)
        except requests.RequestException as exc:
            return {"outcome": "error:" + type(exc).__name__, "sec": time.time() - t0}
        accepted = time.time() - t0
        if r.status_code != 202:
            return {"outcome": f"http_{r.status_code}", "sec": accepted, "accept": accepted}
        jid = r.json()["jobId"]
        while time.time() - t0 < self.timeout:
            try:
                j = requests.get(f"{self.url}/v1/jobs/{jid}", headers=self.h, timeout=30).json()
            except requests.RequestException:
                time.sleep(0.2)
                continue
            if j.get("status") == "done":
                ok = (j["result"]["document"].get(field) or "") == expect
                return {"outcome": "ok" if ok else ("review_wrong" if j["result"]["needsReview"] else "WRONG"),
                        "sec": time.time() - t0, "accept": accepted, "review": j["result"]["needsReview"]}
            if j.get("status") == "failed":
                return {"outcome": "failed:" + j["error"]["code"], "sec": time.time() - t0, "accept": accepted}
            time.sleep(0.1)
        return {"outcome": "timeout", "sec": time.time() - t0, "accept": accepted}


def summary(rows: list[dict], wall: float | None = None) -> dict:
    done = [r["sec"] for r in rows if r["outcome"] in ("ok", "review_wrong", "WRONG") or r["outcome"].startswith("failed")]
    out: dict = {"n": len(rows)}
    kinds: dict[str, int] = {}
    for r in rows:
        kinds[r["outcome"]] = kinds.get(r["outcome"], 0) + 1
    out["outcomes"] = kinds
    out.update(p50=pct(done, 50), p95=pct(done, 95), max=round(max(done), 2) if done else None,
               mean=round(statistics.mean(done), 2) if done else None)
    acc = [r["accept"] for r in rows if "accept" in r]
    out["accept_p95"] = pct(acc, 95)
    if wall:
        out["wall_s"] = round(wall, 1)
        out["per_minute"] = round(len(done) / wall * 60, 1)
    return out


def probe_health(url: str, stop: threading.Event, out: list[float]) -> None:
    while not stop.is_set():
        t = time.time()
        try:
            requests.get(url + "/healthz", timeout=10)
            out.append(time.time() - t)
        except requests.RequestException:
            out.append(10.0)
        stop.wait(0.25)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--key")
    ap.add_argument("--phase", required=True, choices=["latency", "load", "burst", "soak"])
    ap.add_argument("--kinds", default="TD3,MRVA,TD1,iqama,phone")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--levels", default="1,2,4,8")
    ap.add_argument("--seconds", type=int, default=180)
    ap.add_argument("--pid", type=int, action="append", default=[])
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--json")
    a = ap.parse_args()
    c = Client(a)
    report: dict = {"phase": a.phase}
    kinds = a.kinds.split(",")

    if a.phase == "latency":
        c.one(make("TD3", 1))                             # warm-up: the first read loads the models
        for kind in kinds:
            rows = [c.one(make(kind, 100 + i)) for i in range(a.n)]
            report[kind] = summary(rows)
            print(kind, json.dumps(report[kind]), flush=True)

    elif a.phase in ("load", "burst"):
        mix = [k for k in kinds if k != "phone"]
        items = [make(mix[i % len(mix)], 500 + i) for i in range(a.n)]
        c.one(items[0])
        levels = [a.n] if a.phase == "burst" else [int(x) for x in a.levels.split(",")]
        for level in levels:
            stop, health = threading.Event(), []
            t = threading.Thread(target=probe_health, args=(c.url, stop, health), daemon=True)
            t.start()
            t0 = time.time()
            with cf.ThreadPoolExecutor(level) as pool:
                rows = list(pool.map(c.one, items))
            wall = time.time() - t0
            stop.set()
            t.join()
            s = summary(rows, wall)
            s["healthz_p95_ms"], s["healthz_max_ms"] = round((pct(health, 95) or 0) * 1000), round(max(health) * 1000) if health else None
            report[f"c{level}"] = s
            print(f"concurrency {level}:", json.dumps(s), flush=True)

    else:  # soak
        import psutil
        procs = [psutil.Process(p) for p in a.pid]
        mix = [k for k in kinds if k != "phone"]
        rows, rss, i = [], [], 0
        end = time.time() + a.seconds

        def worker(offset: int) -> None:
            n = offset
            while time.time() < end:
                rows.append(c.one(make(mix[n % len(mix)], 9000 + n)))
                n += 2

        threads = [threading.Thread(target=worker, args=(k,)) for k in range(2)]
        t0 = time.time()
        [t.start() for t in threads]
        while time.time() < end:
            if procs:
                total = 0
                for p in procs:
                    try:
                        total += p.memory_info().rss + sum(ch.memory_info().rss for ch in p.children(recursive=True))
                    except psutil.Error:
                        pass
                rss.append(round(total / 1e6))
            time.sleep(5)
            i += 1
        [t.join() for t in threads]
        s = summary(rows, time.time() - t0)
        if rss:
            third = max(1, len(rss) // 3)
            s.update(rss_mb_start=rss[0], rss_mb_first_third=round(statistics.mean(rss[:third])),
                     rss_mb_last_third=round(statistics.mean(rss[-third:])), rss_mb_peak=max(rss))
        half = len(rows) // 2
        s["p50_first_half"], s["p50_second_half"] = pct([r["sec"] for r in rows[:half]], 50), pct([r["sec"] for r in rows[half:]], 50)
        report["soak"] = s
        print("soak:", json.dumps(s), flush=True)

    if a.json:
        Path(a.json).write_text(json.dumps(report, indent=1))
    bad = sum(v.get("outcomes", {}).get("WRONG", 0) for v in report.values() if isinstance(v, dict))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
