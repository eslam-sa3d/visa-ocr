"""Review policy, authentication, rate limiting, job expiry and upload validation."""
import io
import time

import pytest
from fastapi.testclient import TestClient
from PIL import Image

import app.main as main
from app import security
from app.engines.base import EngineOutput
from app.jobs import JobStore, SqliteJobStore
from app.review import Applicant, apply_mrz_policy, compare_applicant, names_match
from app.pipeline import analyse_mrz
from app.visual_zone import parse_gcc_id, saudi_id_check

TD3 = ["P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<", "L898902C36UTO7408122F1204159ZE184226B<<<<<10"]


def img(fmt="PNG", size=(300, 200)):
    b = io.BytesIO()
    Image.new("RGB", size, "white").save(b, format=fmt)
    return b.getvalue()


class EngineWith:
    name = "fake"

    def __init__(self, conf, lines=None):
        self.conf, self.lines = conf, lines

    def extract(self, images, hint):
        return EngineOutput(engine="fake", mrz_lines=TD3, mrz_confidence=self.conf,
                            fields={"textLines": self.lines or []})


def job_for(client, monkeypatch, engine, data=None, headers=None):
    monkeypatch.setattr(main, "get_engine", lambda: engine)
    r = client.post("/v1/documents/extract", files=[("files", ("a.png", img(), "image/png"))], data=data or {}, headers=headers or {})
    assert r.status_code == 202, r.text
    return client.get(f"/v1/jobs/{r.json()['jobId']}", headers=headers or {}).json()


# ---------------------------------------------------------------- 1. unprotected fields (names, nationality, sex)
def codes(job):
    return {f["code"] for f in job["result"]["flags"]}


def test_low_confidence_names_need_review(monkeypatch):
    job = job_for(TestClient(main.app), monkeypatch, EngineWith(0.80))
    assert "unprotected_fields_unverified" in codes(job) and job["result"]["needsReview"]
    assert job["result"]["fieldStatus"]["names"] == "unverified"


def test_unknown_confidence_needs_review(monkeypatch):
    job = job_for(TestClient(main.app), monkeypatch, EngineWith(None))
    assert "unprotected_fields_unverified" in codes(job)


def test_high_confidence_passes_without_review(monkeypatch):
    job = job_for(TestClient(main.app), monkeypatch, EngineWith(0.99))
    flags = codes(job)
    assert "unprotected_fields_unverified" not in flags
    assert job["result"]["fieldStatus"]["names"] == "ocr confident"
    assert job["result"]["confidence"]["mrz"] == 0.99


def test_applicant_form_confirms_names_even_with_low_confidence():
    result = analyse_mrz(TD3)
    apply_mrz_policy(result, 0.5, [], Applicant(surname="Eriksson", given_names="Anna Maria", nationality="UTO"))
    st = result["fieldStatus"]
    assert st["names"] == "confirmed by form" and st["nationality"] == "confirmed by form"
    assert st["sex"] == "unverified"          # the form cannot confirm sex or issuing state


def test_applicant_name_mismatch_is_an_error(monkeypatch):
    job = job_for(TestClient(main.app), monkeypatch, EngineWith(0.99),
                  data={"expected_surname": "ERIKSEN", "expected_given_names": "ANNA MARIA"})
    assert "applicant_name_mismatch" in codes(job) and job["result"]["needsReview"]


def test_applicant_data_mismatches_each_field():
    doc = analyse_mrz(TD3)["document"]
    flags, _ = compare_applicant(doc, Applicant(document_number="L000000", birth_date="1980-01-01", expiry_date="2012-04-15"))
    assert {f.code for f in flags} == {"applicant_document_number_mismatch", "applicant_birth_date_mismatch"}


def test_name_matching_rules():
    assert names_match(["ANNA", "MARIA", "ERIKSSON"], ["ERIKSSON", "ANNA", "MARIA"])        # order does not matter
    assert names_match(["OMAR", "SOBHY", "IBRAHIM"], ["OMAR", "SOBHY", "IBRA"])             # name cut to fit the MRZ
    assert not names_match(["ANNA", "MARIA"], ["ANNA", "MARTA"])
    assert not names_match(["ANNA", "MARIA", "EXTRA"], ["ANNA", "MARIA"])                   # a name is missing in the document


def test_visual_name_match_confirms_names():
    result = analyse_mrz(TD3)
    apply_mrz_policy(result, 0.5, [{"text": "ERIKSSON ANNA MARIA"}], Applicant())
    assert result["fieldStatus"]["names"] == "confirmed by printed page"


# ---------------------------------------------------------------- 2. Iqama / visual-zone results
def L(text, y, s=0.95):
    return {"text": text, "score": s, "box": [[0, y], [100, y], [100, y + 20], [0, y + 20]]}


def make_id(prefix):
    return next(prefix + str(c) for c in range(10) if saudi_id_check(prefix + str(c)))


def test_visual_zone_always_needs_review_by_default():
    r = parse_gcc_id([L("Iqama No: " + make_id("261242825"), 10), L("Date of Birth: 1990/03/12", 40), L("Expiry Date: 2030/01/01", 70)])
    assert r["needsReview"] and "visual_zone_only" in {f["code"] for f in r["flags"]}


def test_visual_zone_auto_accept_still_reviews_checksum_and_low_confidence():
    good = make_id("261242825")
    ok = parse_gcc_id([L("Iqama No: " + good, 10), L("Expiry Date: 2030/01/01", 40)], auto_accept=True)
    assert not ok["needsReview"]
    bad = good[:-1] + str((int(good[-1]) + 1) % 10)
    assert parse_gcc_id([L("Iqama No: " + bad, 10)], auto_accept=True)["needsReview"]
    assert parse_gcc_id([L("Iqama No: " + good, 10, 0.5)], auto_accept=True)["needsReview"]


# ---------------------------------------------------------------- 3. authentication, rate limit, expiry, owner
def test_auth_required_when_keys_configured(monkeypatch):
    monkeypatch.setenv("API_KEYS", "secret-one,secret-two")
    c = TestClient(main.app)
    assert c.get("/v1/models").status_code == 401
    assert c.get("/v1/models", headers={"X-API-Key": "wrong"}).status_code == 401
    assert c.get("/v1/models", headers={"X-API-Key": "secret-two"}).status_code == 200
    assert c.get("/v1/models", headers={"Authorization": "Bearer secret-one"}).status_code == 200
    assert c.get("/healthz").status_code == 200                      # health stays open for load balancers
    assert c.post("/v1/mrz/parse", json={"lines": TD3}).status_code == 401


def test_sha256_keys_supported(monkeypatch):
    import hashlib
    monkeypatch.delenv("API_KEYS", raising=False)
    monkeypatch.setenv("API_KEY_SHA256", hashlib.sha256(b"abc").hexdigest())
    c = TestClient(main.app)
    assert c.get("/v1/models", headers={"X-API-Key": "abc"}).status_code == 200
    assert c.get("/v1/models", headers={"X-API-Key": "abd"}).status_code == 401


def test_jobs_are_private_to_their_owner(monkeypatch):
    monkeypatch.setenv("API_KEYS", "key-a,key-b")
    c = TestClient(main.app)
    job = job_for(c, monkeypatch, EngineWith(0.99), headers={"X-API-Key": "key-a"})
    jid = job["jobId"]
    assert c.get(f"/v1/jobs/{jid}", headers={"X-API-Key": "key-a"}).status_code == 200
    assert c.get(f"/v1/jobs/{jid}", headers={"X-API-Key": "key-b"}).status_code == 404   # same answer as "unknown"


def test_rate_limit(monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_PER_MIN", "3")
    security.limiter._hits.clear()
    c = TestClient(main.app)
    statuses = [c.post("/v1/mrz/parse", json={"lines": TD3}).status_code for _ in range(5)]
    assert statuses == [200, 200, 200, 429, 429]
    r = c.post("/v1/mrz/parse", json={"lines": TD3})
    assert r.headers["retry-after"].isdigit()
    monkeypatch.setenv("RATE_LIMIT_PER_MIN", "0")
    assert c.post("/v1/mrz/parse", json={"lines": TD3}).status_code == 200


@pytest.mark.parametrize("make", [lambda p: JobStore(ttl_seconds=0.2), lambda p: SqliteJobStore(str(p / "jobs.db"), ttl_seconds=0.2)])
def test_jobs_expire(tmp_path, make):
    s = make(tmp_path)
    job = s.create("cid", "owner-1")
    s.update(job.id, status="done", result={"document": {"surname": "X"}})
    got = s.get(job.id)
    assert got.status == "done" and got.result["document"]["surname"] == "X" and got.owner == "owner-1"
    time.sleep(0.4)
    assert s.get(job.id) is None
    s.update(job.id, status="failed")        # updating an expired job must not raise


def test_sqlite_store_survives_restart(tmp_path):
    path = str(tmp_path / "jobs.db")
    a = SqliteJobStore(path, ttl_seconds=600)
    job = a.create("cid", "me")
    a.update(job.id, status="done", result={"x": 1}, consent_training=True)
    b = SqliteJobStore(path, ttl_seconds=600)              # a new process opening the same file
    got = b.get(job.id)
    assert got.result == {"x": 1} and got.consent_training and b.pending_count() == 0


def test_busy_when_too_many_pending(monkeypatch):
    monkeypatch.setenv("MAX_PENDING_JOBS", "0")
    r = TestClient(main.app).post("/v1/documents/extract", files=[("files", ("a.png", img(), "image/png"))])
    assert r.status_code == 503 and r.headers["retry-after"]


def test_correlation_id_is_validated():
    c = TestClient(main.app)
    assert c.get("/healthz", headers={"X-Correlation-ID": "ok-123"}).headers["x-correlation-id"] == "ok-123"
    assert c.get("/healthz", headers={"X-Correlation-ID": "x" * 500}).headers["x-correlation-id"] != "x" * 500
    assert c.get("/healthz", headers={"X-Correlation-ID": "bad id\twith spaces"}).headers["x-correlation-id"] != "bad id\twith spaces"


def test_security_headers():
    c = TestClient(main.app)
    h = c.get("/v1/models").headers
    assert h["cache-control"] == "no-store" and h["x-content-type-options"] == "nosniff"
    assert "default-src 'none'" in c.get("/ui").headers["content-security-policy"]


def test_text_lines_not_returned_by_default(monkeypatch):
    c = TestClient(main.app)
    job = job_for(c, monkeypatch, EngineWith(0.99, lines=[{"text": "PRIVATE", "box": []}]))
    assert "textLines" not in job["result"]["fields"]
    monkeypatch.setenv("INCLUDE_TEXT_LINES", "true")
    job = job_for(c, monkeypatch, EngineWith(0.99, lines=[{"text": "PRIVATE", "box": []}]))
    assert job["result"]["fields"]["textLines"][0]["text"] == "PRIVATE"


# ---------------------------------------------------------------- 4. upload validation
def post_file(data, ctype="image/png"):
    return TestClient(main.app).post("/v1/documents/extract", files=[("files", ("x", data, ctype))])


@pytest.mark.parametrize("data,ctype,status,code", [
    (b"hello world, not an image" * 20, "image/png", 415, "type_mismatch"),
    (b"%PDF-1.4 " + b"0" * 500, "image/jpeg", 415, "type_mismatch"),
    (b"GIF89a" + b"0" * 100, "image/gif", 415, "bad_type"),
    (b"\x89PNG\r\n\x1a\n" + b"garbage" * 50, "image/png", 422, "invalid_image"),
    (None, "image/png", 422, "image_too_small"),                     # replaced below with a real tiny image
])
def test_bad_uploads_get_specific_errors(data, ctype, status, code):
    if data is None:
        data = img(size=(30, 20))
    r = post_file(data, ctype)
    assert r.status_code == status and r.json()["detail"]["code"] == code, r.text


def test_jpeg_labelled_as_png_is_accepted_by_its_real_bytes():
    assert post_file(img("JPEG"), "image/png").status_code == 202


def test_pixel_cap_stops_decompression_bombs(monkeypatch):
    monkeypatch.setenv("MAX_IMAGE_PIXELS", "100000")
    r = post_file(img(size=(400, 400)))
    assert r.status_code == 413 and r.json()["detail"]["code"] == "image_too_many_pixels"
    big = Image.new("1", (20000, 20000), 1)
    b = io.BytesIO()
    big.save(b, format="PNG")                                        # tiny file, 400 megapixels
    monkeypatch.setenv("MAX_IMAGE_PIXELS", "40000000")
    assert post_file(b.getvalue()).status_code == 413


def test_decode_failure_inside_engine_is_a_specific_error(monkeypatch):
    from PIL import UnidentifiedImageError

    class Bad:
        name = "bad"

        def extract(self, images, hint):
            raise UnidentifiedImageError("cannot identify image file")

    job = job_for(TestClient(main.app), monkeypatch, Bad())
    assert job["status"] == "failed" and job["error"]["code"] == "invalid_image"


def test_readyz_reports_missing_models(monkeypatch, tmp_path):
    monkeypatch.setenv("MODELS_DIR", str(tmp_path))
    (tmp_path / "manifest.json").write_text('{"models":[{"id":"m","task":"text_detection","lang":"any","version":"1",'
                                            '"path":"nope","arch":"x","sha256":null,"active":true}]}')
    main._model_cache["at"] = 0.0
    monkeypatch.setenv("OCR_ENGINE", "paddle-ocr")           # this engine cannot work without its models: not ready
    r = TestClient(main.app).get("/readyz")
    assert r.status_code == 503 and "missing" in r.json()["detail"]["message"]
    monkeypatch.setenv("OCR_ENGINE", "auto")                 # falls back to the other readers: ready, with a warning
    r = TestClient(main.app).get("/readyz")
    assert r.status_code == 200 and "missing" in r.json()["warnings"][0]
    monkeypatch.setenv("OCR_ENGINE", "ensemble")             # needs no model files at all
    assert TestClient(main.app).get("/readyz").json().get("warnings") is None
    main._model_cache["at"] = 0.0


def test_very_old_or_newborn_age_gets_century_warning():
    from datetime import date
    from app.mrz import parse_mrz
    from app.validation import validate
    r = parse_mrz(TD3, today=date(2026, 11, 9))           # born 1974: normal
    assert "birth_century_check" not in {f.code for f in validate(r, date(2026, 11, 9))}
    assert "birth_century_check" in {f.code for f in validate(r, date(2074, 1, 1))}   # would be 100 years old


def test_large_photo_is_reduced_before_reading_and_small_one_is_untouched(monkeypatch):
    import io
    from PIL import Image
    from app.uploads import shrink_for_ocr
    big = io.BytesIO()
    Image.new("RGB", (4000, 3000), (200, 200, 200)).save(big, "JPEG")
    out = shrink_for_ocr(big.getvalue())
    assert max(Image.open(io.BytesIO(out)).size) == 2000 and len(out) < len(big.getvalue())
    small = io.BytesIO()
    Image.new("RGB", (1200, 800), (200, 200, 200)).save(small, "PNG")
    assert shrink_for_ocr(small.getvalue()) is small.getvalue() or shrink_for_ocr(small.getvalue()) == small.getvalue()
    monkeypatch.setenv("WORK_MAX_SIDE", "0")
    assert shrink_for_ocr(big.getvalue()) == big.getvalue()          # switched off
    assert shrink_for_ocr(b"not an image") == b"not an image"        # left for the normal error handling
