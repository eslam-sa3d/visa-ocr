"""Concurrent upload test against a running server: checks for crashes, wrong answers and latency.

  python tools/load_test.py --url http://127.0.0.1:8000 --image scenarios/images/<file> --expect-surname ERIKSSON -n 40 -c 8
Add --api-key KEY when the server has API_KEYS set.
"""
import argparse
import concurrent.futures as cf
import statistics
import time

import requests


def one(a, data):
    h = {"X-API-Key": a.api_key} if a.api_key else {}
    t = time.time()
    r = requests.post(a.url + "/v1/documents/extract", files=[("files", ("x.png", data, "image/png"))], headers=h, timeout=60)
    if r.status_code != 202:
        return ("http_%d" % r.status_code, time.time() - t)
    jid = r.json()["jobId"]
    while time.time() - t < a.timeout:
        j = requests.get(f"{a.url}/v1/jobs/{jid}", headers=h, timeout=30).json()
        if j["status"] == "done":
            ok = j["result"]["document"].get("surname") == a.expect_surname
            return ("ok" if ok else "WRONG", time.time() - t)
        if j["status"] == "failed":
            return ("failed:" + j["error"]["code"], time.time() - t)
        time.sleep(0.3)
    return ("timeout", time.time() - t)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:8000")
    p.add_argument("--image", required=True)
    p.add_argument("--expect-surname", required=True)
    p.add_argument("-n", type=int, default=40)
    p.add_argument("-c", type=int, default=8)
    p.add_argument("--timeout", type=int, default=180)
    p.add_argument("--api-key")
    a = p.parse_args()
    data = open(a.image, "rb").read()
    with cf.ThreadPoolExecutor(a.c) as ex:
        res = list(ex.map(lambda _: one(a, data), range(a.n)))
    from collections import Counter
    print(Counter(r[0] for r in res))
    lat = sorted(r[1] for r in res)
    print(f"p50 {statistics.median(lat):.1f}s  p95 {lat[int(len(lat) * .95) - 1]:.1f}s  max {lat[-1]:.1f}s")


if __name__ == "__main__":
    main()
