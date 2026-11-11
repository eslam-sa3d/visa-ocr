#!/usr/bin/env python3
"""Black-box system test against a RUNNING server: full functional cycles and security checks over HTTP.

  python tools/system_test.py --url http://127.0.0.1:8000                       # server without keys (ALLOW_ANONYMOUS)
  python tools/system_test.py --url https://host --key KEY_A --key2 KEY_B       # server with two API keys
  options: --training (server has TRAINING_STORE_DIR), --store-dir DIR (same host: also tests the purge), --insecure (self-signed TLS),
           --json out.json

Documents are generated (ICAO test data, synthetic fonts); nothing real is sent. Exit code 1 when any check fails.
Checks that need a certain server set-up are reported as SKIP when that set-up is absent.
"""
from __future__ import annotations

import argparse
import io
import json
import random
import struct
import subprocess
import sys
import time
import zlib
from pathlib import Path

import requests
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import accuracy_matrix as A  # noqa: E402
import make_scenarios as M  # noqa: E402

RESULTS: list[dict] = []


def record(group: str, name: str, ok: bool | None, detail: str = "") -> None:
    status = "SKIP" if ok is None else ("PASS" if ok else "FAIL")
    RESULTS.append({"group": group, "name": name, "status": status, "detail": detail})
    print(f"{status:4} [{group}] {name}" + (f"  -- {detail}" if detail and status != "PASS" else ""), flush=True)


def check(group: str, name: str):
    def deco(fn):
        try:
            out = fn()
            if out is None or out is True:
                record(group, name, True)
            elif out == "skip" or (isinstance(out, tuple) and out[0] == "skip"):
                record(group, name, None, out[1] if isinstance(out, tuple) else "")
            else:
                record(group, name, False, str(out))
        except AssertionError as exc:
            record(group, name, False, "assert: " + str(exc)[:300])
        except Exception as exc:  # noqa: BLE001
            record(group, name, False, f"{type(exc).__name__}: {str(exc)[:300]}")
        return fn
    return deco


class Api:
    def __init__(self, url: str, key: str | None, verify: bool) -> None:
        self.url, self.key, self.verify = url.rstrip("/"), key, verify

    def h(self, key: str | None = "__own__") -> dict:
        k = self.key if key == "__own__" else key
        return {"X-API-Key": k} if k else {}

    def get(self, path, key="__own__", **kw):
        return requests.get(self.url + path, headers={**self.h(key), **kw.pop("headers", {})}, verify=self.verify, timeout=60, **kw)

    def post(self, path, key="__own__", **kw):
        return requests.post(self.url + path, headers={**self.h(key), **kw.pop("headers", {})}, verify=self.verify, timeout=120, **kw)

    def upload(self, images: list[bytes], data: dict | None = None, key="__own__", ctype="image/jpeg"):
        return self.post("/v1/documents/extract", key=key, files=[("files", (f"d{i}.jpg", b, ctype)) for i, b in enumerate(images)],
                         data=data or {})

    def wait(self, job_id: str, key="__own__", timeout=90) -> dict:
        t = time.time()
        while time.time() - t < timeout:
            j = self.get(f"/v1/jobs/{job_id}", key=key).json()
            if j.get("status") in ("done", "failed"):
                return j
            time.sleep(0.15)
        raise AssertionError("job did not finish in time")

    def read(self, images, data=None) -> dict:
        r = self.upload(images, data)
        assert r.status_code == 202, f"upload {r.status_code} {r.text[:200]}"
        return self.wait(r.json()["jobId"])


def jpg(img: Image.Image, quality=90) -> bytes:
    b = io.BytesIO()
    img.convert("RGB").save(b, "JPEG", quality=quality)
    return b.getvalue()


def png_bytes(size=(300, 200), color="white") -> bytes:
    b = io.BytesIO()
    Image.new("RGB", size, color).save(b, "PNG")
    return b.getvalue()


def make_doc(fmt: str, seed: int, variant: str = "clean"):
    rng = random.Random(seed)
    kw = A.rnd_case(rng, fmt)
    if fmt in ("TD1", "TD2"):
        kw["doc"] = A.rnd_doc(rng, fmt)
    lines, truth = A.build(fmt, **kw)
    img = M.VARIANTS[variant](M.draw_document({"lines": lines, "label": f"{fmt} TEST"}))
    return jpg(img), truth, lines


def bomb_png(width=60000, height=60000) -> bytes:
    """A tiny PNG file whose header declares 3.6 gigapixels."""
    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)) \
        + chunk(b"IDAT", zlib.compress(b"\x00" * 1000)) + chunk(b"IEND", b"")


FIELDS = ["documentNumber", "surname", "givenNames", "nationality", "birthDate", "expiryDate", "sex"]


def functional(api: Api, a) -> None:
    g = "functional"

    @check(g, "service answers: /healthz, /readyz, /, /ui, /v1/models")
    def _():
        assert api.get("/healthz", key=None).json() == {"status": "ok"}
        r = api.get("/readyz", key=None)
        assert r.status_code == 200 and r.json()["status"] == "ready", r.text
        assert "endpoints" in api.get("/", key=None).json()
        ui = api.get("/ui", key=None)
        assert ui.status_code == 200 and "<html" in ui.text.lower()
        assert "models" in api.get("/v1/models").json()

    for fmt, label in (("TD3", "passport"), ("MRVA", "visa MRV-A"), ("MRVB", "visa MRV-B"), ("TD1", "ID card TD1"), ("TD2", "ID card TD2")):
        @check(g, f"full cycle, {label}: upload, poll, every field correct, check digits pass")
        def _(fmt=fmt):
            data, truth, _lines = make_doc(fmt, 4100 + len(fmt) + ord(fmt[-1]))
            j = api.read([data])
            assert j["status"] == "done", j.get("error")
            doc = j["result"]["document"]
            wrong = [f for f in FIELDS if f in truth and (doc.get(f) or "") != (truth[f] or "")]
            assert not wrong, f"wrong fields {wrong}"
            assert all(doc["checks"].values()), doc["checks"]
            assert isinstance(j["result"]["needsReview"], bool) and "fieldStatus" in j["result"]

    for variant in ("rot+4", "blur", "noise"):
        @check(g, f"passport photo variant '{variant}': read, and never accepted with a wrong field")
        def _(variant=variant):
            data, truth, _ = make_doc("TD3", 5200, variant)
            j = api.read([data])
            if j["status"] != "done":
                return ("skip", "not read: " + j["error"]["code"])
            doc = j["result"]["document"]
            wrong = [f for f in FIELDS if f in truth and (doc.get(f) or "") != (truth[f] or "")]
            assert not (wrong and not j["result"]["needsReview"]), f"accepted with wrong {wrong}"
            return ("skip", f"read with wrong {wrong}, sent to review") if wrong else True

    @check(g, "form data that matches confirms names and nationality")
    def _():
        data, truth, _ = make_doc("TD3", 6100)
        j = api.read([data], {"expected_surname": truth["surname"], "expected_given_names": truth["givenNames"],
                              "expected_nationality": truth["nationality"], "expected_document_number": truth["documentNumber"],
                              "expected_birth_date": truth["birthDate"], "expected_expiry_date": truth["expiryDate"]})
        fs = j["result"]["fieldStatus"]
        assert fs["names"] == "confirmed by form" and fs["nationality"] == "confirmed by form", fs
        assert not [f for f in j["result"]["flags"] if f["code"].startswith("applicant_")]

    @check(g, "form data that differs forces review with a specific flag per field")
    def _():
        data, truth, _ = make_doc("TD3", 6100)
        j = api.read([data], {"expected_surname": "ZZZZQ", "expected_document_number": "X0000000", "expected_birth_date": "1901-01-01",
                              "expected_nationality": "ZZZ" if truth["nationality"] != "ZZZ" else "QQQ"})
        codes = {f["code"] for f in j["result"]["flags"]}
        need = {"applicant_name_mismatch", "applicant_document_number_mismatch", "applicant_birth_date_mismatch", "nationality_mismatch"}
        assert need <= codes and j["result"]["needsReview"], sorted(codes)

    @check(g, "expired document is flagged and sent to review")
    def _():
        lines = ["P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<", "L898902C36UTO7408122F1204159ZE184226B<<<<<10"]
        j = api.read([jpg(M.draw_document({"lines": lines, "label": "TD3 TEST"}))])
        assert j["status"] == "done" and j["result"]["needsReview"]
        assert "document_expired" in {f["code"] for f in j["result"]["flags"]}

    @check(g, "Iqama card (no MRZ): ID number, dates and nationality read; always sent to review")
    def _():
        p = dict(M.IQAMA_PEOPLE[0])
        j = api.read([jpg(M.draw_iqama(p))], {"doc_type_hint": "iqama"})
        assert j["status"] == "done", j.get("error")
        doc = j["result"]["document"]
        assert doc.get("idNumber") == p["id"], doc.get("idNumber")
        assert doc.get("expiryDate") and doc.get("birthDate") and doc.get("nationalityCode")
        assert j["result"]["needsReview"] is True

    @check(g, "two images in one upload are read as one job")
    def _():
        data, truth, _ = make_doc("TD3", 6200)
        j = api.read([png_bytes(), data])
        assert j["status"] == "done" and j["result"]["document"]["documentNumber"] == truth["documentNumber"]

    @check(g, "blank page fails with mrz_not_found and a usable message")
    def _():
        j = api.read([png_bytes((900, 600))], {"doc_type_hint": "passport"})
        assert j["status"] == "failed" and j["error"]["code"] == "mrz_not_found" and len(j["error"]["message"]) > 20

    @check(g, "/v1/mrz/parse: valid lines parse; a changed digit fails its check; nonsense is refused")
    def _():
        lines = ["P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<", "L898902C36UTO7408122F1204159ZE184226B<<<<<10"]
        ok = api.post("/v1/mrz/parse", json={"lines": lines}).json()
        assert ok["document"]["surname"] == "ERIKSSON" and all(ok["document"]["checks"].values())
        bad = api.post("/v1/mrz/parse", json={"lines": [lines[0], lines[1].replace("7408122", "7408132")]}).json()
        assert "mrz_checksum_failed" in {f["code"] for f in bad["flags"]} and bad["needsReview"]
        assert api.post("/v1/mrz/parse", json={"lines": ["HELLO", "WORLD"]}).status_code == 422
        assert api.post("/v1/mrz/parse", json={"lines": ["A"]}).status_code == 422

    @check(g, "upload errors have the right status and code")
    def _():
        cases = [
            ("no file", lambda: api.post("/v1/documents/extract", data={"doc_type_hint": "passport"}), 422, None),
            ("5 files", lambda: api.upload([png_bytes()] * 5), 422, "bad_file_count"),
            ("text as image", lambda: api.upload([b"hello world, not an image"]), 415, None),
            ("GIF", lambda: api.upload([b"GIF89a" + b"\x00" * 200], ctype="image/gif"), 415, "bad_type"),
            ("PDF", lambda: api.upload([b"%PDF-1.7\n" + b"0" * 200], ctype="application/pdf"), 415, "bad_type"),
            ("truncated JPEG header", lambda: api.upload([b"\xff\xd8\xff\xe0" + b"\x00" * 30]), 422, "invalid_image"),
            ("too small", lambda: api.upload([png_bytes((20, 20))]), 422, "image_too_small"),
            ("long and narrow", lambda: api.upload([png_bytes((3000, 100))]), 422, "image_bad_aspect"),
            ("over 10 MB", lambda: api.upload([b"\xff\xd8\xff" + b"\x00" * (10 * 1024 * 1024 + 10)]), 413, "too_large"),
            ("form field too long", lambda: api.upload([png_bytes()], {"expected_document_number": "A" * 31}), 422, None),
        ]
        bad = []
        for name, call, status, code in cases:
            r = call()
            got = (r.json().get("detail") or {}) if r.headers.get("content-type", "").startswith("application/json") else {}
            got_code = got.get("code") if isinstance(got, dict) else None
            if r.status_code != status or (code and got_code != code):
                bad.append(f"{name}: {r.status_code}/{got_code}")
        assert not bad, "; ".join(bad)

    @check(g, "unknown job id gives 404; a result can be read again until it expires")
    def _():
        assert api.get("/v1/jobs/" + "0" * 32).status_code == 404
        data, _, _ = make_doc("TD3", 6300)
        r = api.upload([data])
        jid = r.json()["jobId"]
        assert r.json()["status"] == "queued" and r.headers.get("X-Correlation-ID") == r.json()["correlationId"]
        first = api.wait(jid)
        assert api.get(f"/v1/jobs/{jid}").json() == first

    @check(g, "caller's correlation id is kept on the job")
    def _():
        r = api.upload([png_bytes()], None)
        r2 = api.post("/v1/documents/extract", files=[("files", ("a.png", png_bytes(), "image/png"))], headers={"X-Correlation-ID": "portal-req-123"})
        assert r.status_code == 202 and r2.json()["correlationId"] == "portal-req-123"

    @check(g, "training cycle: consent, correction saved, no-consent refused, purge blocks later corrections")
    def _():
        if not a.training:
            return ("skip", "server has no TRAINING_STORE_DIR (run with --training)")
        data, _, lines = make_doc("TD3", 6400)
        body = {"reviewer": "tester-1", "document": {"surname": "DUMMY"},
                "lines": [{"image": 0, "box": [10, 10, 400, 60], "text": lines[1], "lang": "en"}]}
        r = api.upload([data], {"consent_training": "true"})
        jid = r.json()["jobId"]
        done = api.wait(jid)
        boxes = (done["result"].get("fields") or {}).get("mrzLineBoxes")
        assert boxes and all("text" in b for b in boxes), "line positions for the reviewer are missing"
        c = api.post(f"/v1/jobs/{jid}/corrections", json=body)
        assert c.status_code == 201 and c.json() == {"stored": True, "lines": 1}, c.text
        r2 = api.upload([data])
        api.wait(r2.json()["jobId"])
        assert api.post(f"/v1/jobs/{r2.json()['jobId']}/corrections", json=body).status_code == 409
        bad = dict(body, lines=[dict(body["lines"][0], text="A\tB\nC")])
        assert api.post(f"/v1/jobs/{jid}/corrections", json=bad).status_code == 422
        if not a.store_dir:
            return ("skip", "corrections OK; purge step needs --store-dir on the same host")
        store = Path(a.store_dir)
        assert (store / "jobs" / jid / "correction.json").exists() and list((store / "jobs" / jid / "images").iterdir())
        modes = {oct(p.stat().st_mode & 0o777) for p in (store / "jobs" / jid).rglob("*")}
        assert modes <= {"0o600", "0o700"}, f"training files not owner-only: {modes}"
        subprocess.run([sys.executable, str(HERE / "purge_training.py"), "--store-dir", str(store), "--job-id", jid], check=True, capture_output=True)
        assert not (store / "jobs" / jid).exists()
        assert api.post(f"/v1/jobs/{jid}/corrections", json=body).status_code == 409
        assert not (store / "jobs" / jid).exists(), "a correction re-created purged data"


def security_checks(api: Api, a) -> None:
    g = "security"
    keys = bool(a.key)

    @check(g, "no key and wrong key are refused on every /v1 route; a valid key works (header and Bearer)")
    def _():
        if not keys:
            return ("skip", "server runs without keys")
        routes = [("GET", "/v1/models"), ("GET", "/v1/jobs/" + "0" * 32), ("POST", "/v1/mrz/parse"), ("POST", "/v1/documents/extract"),
                  ("POST", "/v1/jobs/" + "0" * 32 + "/corrections")]
        for method, path in routes:
            r = requests.request(method, api.url + path, verify=api.verify, timeout=30, json={} if method == "POST" else None)
            assert r.status_code == 401, f"{method} {path} without key gave {r.status_code}"
        assert api.get("/v1/models", key="wrong-" + "x" * 30).status_code == 401
        assert api.get("/v1/models").status_code == 200
        assert requests.get(api.url + "/v1/models", headers={"Authorization": "Bearer " + a.key}, verify=api.verify, timeout=30).status_code == 200
        r = api.get("/readyz", key=None).json()
        assert r.get("authEnabled") is True

    @check(g, "a job is visible only to the key that created it (read and corrections)")
    def _():
        if not (keys and a.key2):
            return ("skip", "needs --key and --key2")
        data, _, _ = make_doc("TD3", 7100)
        jid = api.upload([data]).json()["jobId"]
        api.wait(jid)
        other = api.get(f"/v1/jobs/{jid}", key=a.key2)
        missing = api.get("/v1/jobs/" + "f" * 32, key=a.key2)
        assert other.status_code == 404 and other.json() == missing.json(), "foreign job is distinguishable from a missing one"
        c = api.post(f"/v1/jobs/{jid}/corrections", key=a.key2, json={"reviewer": "x", "document": {}, "lines": []})
        assert c.status_code == 404
        assert api.get(f"/v1/jobs/{jid}").status_code == 200

    @check(g, "job ids are random 128-bit values")
    def _():
        ids = [api.upload([png_bytes()]).json()["jobId"] for _ in range(6)]
        assert all(len(i) == 32 and int(i, 16) >= 0 for i in ids) and len(set(ids)) == 6
        assert len({i[:8] for i in ids}) == 6 and len({i[-8:] for i in ids}) == 6

    @check(g, "security headers on every response; results are not cacheable; the test page has a content policy")
    def _():
        for path in ("/healthz", "/readyz", "/v1/models", "/ui", "/v1/jobs/" + "0" * 32):
            h = api.get(path).headers
            assert h.get("X-Content-Type-Options") == "nosniff" and h.get("X-Frame-Options") == "DENY" and h.get("Referrer-Policy") == "no-referrer", path
            if path.startswith("/v1"):
                assert h.get("Cache-Control") == "no-store", path
        csp = api.get("/ui").headers.get("Content-Security-Policy", "")
        assert "default-src 'none'" in csp and "frame-ancestors 'none'" in csp
        assert "access-control-allow-origin" not in {k.lower() for k in api.get("/v1/models", headers={"Origin": "https://evil.example"}).headers}

    @check(g, "oversized request body is refused with 413 (declared and streamed)")
    def _():
        import socket
        import ssl
        from urllib.parse import urlparse
        u = urlparse(api.url)
        s = socket.create_connection((u.hostname, u.port or (443 if u.scheme == "https" else 80)), timeout=20)
        if u.scheme == "https":
            ctx = ssl.create_default_context()
            if not api.verify:
                ctx.check_hostname, ctx.verify_mode = False, ssl.CERT_NONE
            s = ctx.wrap_socket(s, server_hostname=u.hostname)
        key = f"X-API-Key: {api.key}\r\n" if api.key else ""
        s.sendall((f"POST /v1/mrz/parse HTTP/1.1\r\nHost: {u.hostname}\r\n{key}Content-Type: application/json\r\n"
                   f"Content-Length: {60 * 1024 * 1024}\r\n\r\n").encode() + b"{")
        first = s.recv(200).decode(errors="replace")
        s.close()
        assert " 413 " in first.split("\r\n")[0], "declared 60 MB body: " + first[:60]

        def gen():
            yield b'--xx\r\nContent-Disposition: form-data; name="files"; filename="a.jpg"\r\nContent-Type: image/jpeg\r\n\r\n'
            for _ in range(50):
                yield b"x" * (1024 * 1024)
            yield b"\r\n--xx--\r\n"
        try:
            r2 = requests.post(api.url + "/v1/documents/extract", headers={**api.h(), "Content-Type": "multipart/form-data; boundary=xx"},
                               data=gen(), verify=api.verify, timeout=60)
            assert r2.status_code == 413, f"streamed 50 MB upload gave {r2.status_code}"
        except requests.exceptions.ConnectionError:
            pass                                    # the server closed the connection part-way: also a refusal

    @check(g, "hostile image files are refused before decoding (pixel bomb, wrong content, polyglot)")
    def _():
        bomb = api.upload([bomb_png()], ctype="image/png")
        assert bomb.status_code == 413 and bomb.json()["detail"]["code"] == "image_too_many_pixels", bomb.text[:200]
        svg = api.upload([b"<svg xmlns='http://www.w3.org/2000/svg'><script>alert(1)</script></svg>"], ctype="image/svg+xml")
        assert svg.status_code == 415
        html = api.upload([b"<html><script>alert(1)</script></html>"], ctype="image/png")
        assert html.status_code == 415
        poly = api.upload([png_bytes() + b"<script>alert(1)</script>" + b"PK\x03\x04" + b"\x00" * 100], ctype="image/png")
        assert poly.status_code == 202                  # a valid PNG with trailing bytes is just an image: it must not crash
        j = api.wait(poly.json()["jobId"])
        assert j["status"] == "failed" and j["error"]["code"] in ("mrz_not_found", "invalid_image")
        assert api.get("/healthz", key=None).status_code == 200

    @check(g, "corrupt image body ends as a clean failed job, not a crash or a stack trace")
    def _():
        data, _, _ = make_doc("TD3", 7200)
        r = api.upload([data[: len(data) // 3]])
        if r.status_code != 202:
            assert r.status_code in (415, 422)
            return True
        j = api.wait(r.json()["jobId"])
        assert j["status"] in ("failed", "done")
        if j["status"] == "failed":
            assert j["error"]["code"] in ("invalid_image", "mrz_not_found"), j["error"]["code"]
        assert "Traceback" not in json.dumps(j) and "File \"" not in json.dumps(j)

    @check(g, "injection attempts in ids, headers and fields have no effect")
    def _():
        for bad in ("../../etc/passwd", "..%2F..%2Fetc%2Fpasswd", "' OR '1'='1", "0" * 31 + "'", "%00", "a" * 5000):
            r = api.get("/v1/jobs/" + bad)
            assert r.status_code in (404, 405, 400, 414), f"job id {bad[:20]!r} gave {r.status_code}"
            assert "root:" not in r.text and "Traceback" not in r.text
        r = api.post(f"/v1/jobs/{'../' * 3}x/corrections", json={"reviewer": "x"})
        assert r.status_code in (404, 405, 422)
        cid = api.get("/healthz", headers={"X-Correlation-ID": "abc\r\nSet-Cookie: x=1"} if False else {"X-Correlation-ID": "<script>alert(1)</script>"})
        assert "<script>" not in cid.headers.get("X-Correlation-ID", "") and "set-cookie" not in {k.lower() for k in cid.headers}
        data, _, _ = make_doc("TD3", 7300)
        xss = "<img src=x onerror=alert(1)>"
        j = api.read([data], {"expected_surname": xss, "expected_document_number": "'; DROP TABLE jobs;--"})
        assert j["status"] == "done"
        raw = api.get("/v1/models")
        assert raw.status_code == 200                     # the store still works after the SQL-looking value
        res = api.get("/v1/jobs/" + j["jobId"])
        assert res.headers["content-type"].startswith("application/json")
        long = api.post("/v1/mrz/parse", json={"lines": ["A" * 100000, "B" * 100000]})
        assert long.status_code == 422

    @check(g, "malformed requests give a clean 4xx with no internals")
    def _():
        cases = [
            requests.post(api.url + "/v1/mrz/parse", headers={**api.h(), "Content-Type": "application/json"}, data=b"{not json", verify=api.verify, timeout=30),
            requests.post(api.url + "/v1/documents/extract", headers={**api.h(), "Content-Type": "multipart/form-data; boundary=xx"},
                          data=b"--xx\r\nbroken", verify=api.verify, timeout=30),
            requests.post(api.url + "/v1/mrz/parse", headers=api.h(), json={"lines": [1, 2]}, verify=api.verify, timeout=30),
            requests.post(api.url + "/v1/jobs/" + "0" * 32 + "/corrections", headers=api.h(), json={"reviewer": "", "lines": "x"}, verify=api.verify, timeout=30),
        ]
        for r in cases:
            assert 400 <= r.status_code < 500, r.status_code
            assert "Traceback" not in r.text and "site-packages" not in r.text and "/home/" not in r.text

    @check(g, "methods that should not exist are refused")
    def _():
        for method, path in (("DELETE", "/v1/jobs/" + "0" * 32), ("PUT", "/v1/models"), ("PATCH", "/v1/documents/extract"), ("GET", "/v1/documents/extract")):
            r = requests.request(method, api.url + path, headers=api.h(), verify=api.verify, timeout=30)
            assert r.status_code == 405, f"{method} {path} -> {r.status_code}"
        t = requests.request("TRACE", api.url + "/", verify=api.verify, timeout=30)
        assert t.status_code in (405, 501, 400)

    @check(g, "interactive API docs are not open to anonymous callers")
    def _():
        open_ = [p for p in ("/docs", "/redoc", "/openapi.json") if requests.get(api.url + p, verify=api.verify, timeout=30).status_code == 200]
        assert not open_, f"open without a key: {open_}"

    @check(g, "wrong keys are throttled (429 after a few attempts)")
    def _():
        if not keys:
            return ("skip", "server runs without keys")
        got = [api.get("/v1/models", key=f"guess-{i}-" + "x" * 20).status_code for i in range(15)]
        assert 429 in got, f"15 wrong keys, no 429: {got}"
        assert api.get("/v1/models").status_code == 200, "the valid key was locked out by another caller's wrong guesses"

    @check(g, "request rate limit answers 429 with Retry-After (run last)")
    def _():
        got = []
        for _ in range(a.rate_probe):
            r = api.post("/v1/mrz/parse", json={"lines": ["P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<", "L898902C36UTO7408122F1204159ZE184226B<<<<<10"]})
            got.append(r.status_code)
            if r.status_code == 429:
                assert int(r.headers.get("Retry-After", "0")) >= 1
                return True
        return ("skip", f"no 429 in {a.rate_probe} requests (limit is off or higher)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--key")
    ap.add_argument("--key2", help="a second valid key, for the isolation test")
    ap.add_argument("--training", action="store_true")
    ap.add_argument("--store-dir")
    ap.add_argument("--insecure", action="store_true")
    ap.add_argument("--only", choices=["functional", "security"])
    ap.add_argument("--rate-probe", type=int, default=80)
    ap.add_argument("--json")
    a = ap.parse_args()
    if a.insecure:
        import urllib3
        urllib3.disable_warnings()
    api = Api(a.url, a.key, not a.insecure)
    if a.only != "security":
        functional(api, a)
    if a.only != "functional":
        security_checks(api, a)
    n = {s: sum(1 for r in RESULTS if r["status"] == s) for s in ("PASS", "FAIL", "SKIP")}
    print(f"\n{n['PASS']} passed, {n['FAIL']} failed, {n['SKIP']} skipped")
    if a.json:
        Path(a.json).write_text(json.dumps(RESULTS, indent=1))
    sys.exit(1 if n["FAIL"] else 0)


if __name__ == "__main__":
    main()
