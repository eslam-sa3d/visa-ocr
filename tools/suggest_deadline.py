#!/usr/bin/env python3
"""Measure how long THIS server needs to read a document, and suggest a value for OCR_DEADLINE_SECONDS.

A hard time limit that is too short for the machine costs accuracy: the readers stop before their later attempts (enlarged
page, sideways page). Run this once on each kind of server, with the engine it will use, before setting a limit:

    python tools/suggest_deadline.py [--n 30] [--engine ensemble]

It reads generated passports, visas and ID cards (clean, tilted, blurred, noisy) with NO limit, then prints the time under
which 95% finished and how many would have been cut off at a few candidate limits. Suggested value: the 95% time plus a
quarter, rounded up to 0.5 s. Leave OCR_DEADLINE_SECONDS unset if response time is not a requirement.
"""
from __future__ import annotations

import argparse
import io
import math
import os
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--engine")
    a = ap.parse_args()
    os.environ.pop("OCR_DEADLINE_SECONDS", None)
    import accuracy_matrix as A
    import make_scenarios as M
    from PIL import ImageFilter
    from app import pipeline
    from app.engines import get_engine, get_engine_by_name
    from app.review import Applicant
    engine = get_engine_by_name(a.engine) if a.engine else get_engine()
    rng = random.Random(11)
    times, wrong = [], 0
    for i in range(a.n + 1):
        fmt = ("TD3", "MRVA", "TD1", "TD3")[i % 4]
        kw = A.rnd_case(rng, fmt)
        if fmt == "TD1":
            kw["doc"] = A.rnd_doc(rng, fmt)
        lines, truth = A.build(fmt, **kw)
        img = M.draw_document({"lines": lines, "label": f"{fmt} TEST"}).convert("RGB")
        if i % 3 == 1:
            img = img.rotate(rng.choice((-4, 3, 5)), expand=True, fillcolor=(255, 255, 255))
        elif i % 3 == 2:
            img = img.filter(ImageFilter.GaussianBlur(1.2))
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=85)
        t0 = time.monotonic()
        status, result = pipeline.run([buf.getvalue()], None, Applicant(), engine)
        if i == 0:
            continue                                   # the first read loads the models
        times.append(time.monotonic() - t0)
        wrong += status != "done" or result["document"].get("documentNumber") != truth["documentNumber"]
    times.sort()
    p95 = times[min(len(times) - 1, int(round(0.95 * (len(times) - 1))))]
    print(f"documents: {len(times)}   not read correctly with no limit: {wrong}")
    print(f"median {times[len(times) // 2]:.2f} s   95% under {p95:.2f} s   slowest {times[-1]:.2f} s")
    for limit in (1.0, 1.5, 2.0, 3.0, 5.0):
        print(f"  at a limit of {limit:.1f} s, {sum(t > limit for t in times)} of {len(times)} reads would have been cut short")
    print(f"suggested OCR_DEADLINE_SECONDS: {math.ceil(p95 * 1.25 * 2) / 2:.1f}")


if __name__ == "__main__":
    main()
