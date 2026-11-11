"""Regression tests for the second security audit (11 Nov 2026). One block per finding."""
import asyncio
import io
import json
import os
import stat
import subprocess
import sys
import threading
import time
from argparse import Namespace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

import app.main as main
from app import mrz, security
from app.engines.base import EngineOutput
from app.jobs import JobStore, SqliteJobStore
from app.pipeline import analyse_mrz
from app.review import Applicant, apply_mrz_policy, compare_applicant
from app.training_store import TrainingStore
from app.validation import visual_name_status
from app.visual_zone import parse_gcc_id, saudi_id_check

ROOT = Path(__file__).resolve().parent.parent
LINE2 = "L898902C36UTO7408122F1204159ZE184226B<<<<<10"
TD3 = ["P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<", LINE2]


def png(size=(300, 200)):
    b = io.BytesIO()
    Image.new("RGB", size, "white").save(b, format="PNG")
    return b.getvalue()


def line1(names: str) -> str:
    return ("P<UTO" + names).ljust(44, "<")


def codes(result):
    return {f["code"] for f in result["flags"]}


def L(text, y, score=0.95):
    return {"text": text, "score": score, "box": [[10, y], [600, y], [600, y + 20], [10, y + 20]]}


def good_id(prefix="2"):
    for n in range(100000000, 100000999):
        cand = prefix + str(n)
        if saudi_id_check(cand):
            return cand
    raise AssertionError


class Fixed:
    name = "fake"

    def __init__(self, lines=TD3, conf=0.99, text=None):
        self.lines, self.conf, self.text = lines, conf, text

    def extract(self, images, hint):
        return EngineOutput(engine="fake", mrz_lines=self.lines, mrz_confidence=self.conf, fields={"textLines": self.text or []})


# ------------------------------------------------------------------ 1. blank API key must not turn authentication off
@pytest.mark.parametrize("value", [",", " ", " , ,"])
def test_blank_api_key_fails_closed(monkeypatch, value):
    monkeypatch.setenv("API_KEYS", value)
    monkeypatch.delenv("API_KEY_SHA256", raising=False)
    with pytest.raises(RuntimeError):
        security.auth_enabled()
    with pytest.raises(RuntimeError):                       # a request is never answered as "anonymous"
        TestClient(main.app).get("/v1/models")


def test_entrypoint_refuses_blank_key(tmp_path):
    env = {"PATH": os.environ["PATH"], "API_KEYS": ",", "TLS_CERT": str(tmp_path / "c"), "TLS_KEY": str(tmp_path / "k")}
    r = subprocess.run(["sh", str(ROOT / "docker-entrypoint.sh")], env=env, capture_output=True, text=True, timeout=20)
    assert r.returncode == 1 and "no usable API key" in r.stderr


def test_wrong_keys_are_throttled_even_with_rate_limit_off(monkeypatch):
    monkeypatch.setenv("API_KEYS", "k" * 40)
    monkeypatch.setenv("RATE_LIMIT_PER_MIN", "0")
    monkeypatch.setenv("BAD_KEY_LIMIT_PER_MIN", "3")
    monkeypatch.setattr(security, "limiter", security.RateLimiter())
    c = TestClient(main.app)
    got = [c.get("/v1/models", headers={"X-API-Key": "wrong"}).status_code for _ in range(5)]
    assert got[:3] == [401, 401, 401] and 429 in got[3:]


# ------------------------------------------------------------------ 2. confidence covers every MRZ line that is used
def test_padded_or_unmatched_line_has_no_confidence():
    short = "P<UTOERIKSSON<<ANNA<<<"
    found = mrz.find_mrz_lines([short, LINE2])
    assert found and len(found[0]) == 44
    assert mrz.lines_confidence(found, [short, LINE2], [0.30, 0.99]) is None       # line 1 was padded: no score speaks for it
    assert mrz.lines_confidence(TD3, TD3, [0.30, 0.99]) == 0.30                    # the lowest line decides
    assert mrz.lines_confidence(TD3, TD3, [0.98, 0.99]) == 0.98


def test_short_low_score_line1_goes_to_review():
    short = "P<UTOERIKSSON<<ANNA<<<"
    found = mrz.find_mrz_lines([short, LINE2])
    result = analyse_mrz(found)
    apply_mrz_policy(result, mrz.lines_confidence(found, [short, LINE2], [0.30, 0.99]), [], Applicant(), mrz_lines=found)
    assert result["needsReview"] and result["fieldStatus"]["names"] == "unverified"


# ------------------------------------------------------------------ 3. card text reader: bounded Tesseract calls and time
def test_text_reader_stops_at_its_budget_and_caps_rereads(monkeypatch):
    pytest.importorskip("pytesseract")
    from app.engines import tesseract_text as tt
    calls = {"data": 0, "string": 0}
    rows = 40

    def fake_data(img, **kw):
        calls["data"] += 1
        n = rows if img.height > 200 else 1
        return {"text": ["4007 3158 2604"] * n, "conf": [30.0] * n, "block_num": [1] * n, "par_num": [1] * n,
                "line_num": list(range(n)), "left": [10] * n, "top": [20 + 30 * i for i in range(n)],
                "width": [400] * n, "height": [20] * n}

    def fake_string(img, **kw):
        calls["string"] += 1
        return "4007 3158 2604"

    import pytesseract
    monkeypatch.setattr(pytesseract, "image_to_data", fake_data)
    monkeypatch.setattr(pytesseract, "image_to_string", fake_string)
    monkeypatch.setattr(tt, "available_langs", lambda: {"ara", "eng"})
    lines = tt.read_text_lines([png((1300, 1400))])
    assert len(lines) >= rows
    # before the fix: 8 digit re-reads for each of the 40 rows (320) and 2 weak-line re-reads per row per scale (160)
    assert calls["string"] <= tt.MAX_ID_ROWS * 8
    assert calls["data"] <= 2 + 2 * 2 * tt.MAX_REREADS + 4 * 2 * tt.MAX_STRIPS

    calls.update(data=0, string=0)
    monkeypatch.setenv("OCR_DEADLINE_SECONDS", "0.0001")     # budget already spent after the first page read
    tt.read_text_lines([png((1300, 1400))])
    assert calls["string"] == 0 and calls["data"] <= 2


def test_ensemble_card_text_is_not_read_twice(monkeypatch):
    from app import pipeline
    from app.engines import tesseract_text
    monkeypatch.setattr(tesseract_text, "read_text_lines", lambda images: pytest.fail("second text read"))
    result = parse_gcc_id([L("Iqama No: " + good_id(), 10)])
    assert pipeline._fill_from_second_reader(result, [b""], "ensemble") is result


# ------------------------------------------------------------------ 4. form fields are bounded and not echoed at length
def test_upload_form_fields_have_length_limits(monkeypatch):
    monkeypatch.setattr(main, "get_engine", lambda: Fixed())
    c = TestClient(main.app)
    files = [("files", ("a.png", png(), "image/png"))]
    assert c.post("/v1/documents/extract", files=files, data={"expected_document_number": "A" * 31}).status_code == 422
    assert c.post("/v1/documents/extract", files=files, data={"doc_type_hint": "Z" * 33}).status_code == 422
    assert c.post("/v1/documents/extract", files=files, data={"expected_surname": "Q" * 101}).status_code == 422
    r = c.post("/v1/documents/extract", files=files, data={"expected_document_number": "\x01" * 30, "expected_surname": "Q" * 100})
    assert r.status_code == 202
    job = c.get(f"/v1/jobs/{r.json()['jobId']}").json()
    assert len(json.dumps(job["result"])) < 4000 and "\\u0001" not in json.dumps(job["result"])


def test_flag_messages_clip_caller_text():
    flags, _ = compare_applicant({"documentNumber": "L898902C3", "surname": "ERIKSSON", "givenNames": "ANNA"},
                                 Applicant(document_number="A" * 5000, surname="Q" * 5000))
    assert flags and all(len(f.message) < 400 for f in flags)


# ------------------------------------------------------------------ 5. pending cap is atomic; jobs have their own pool
@pytest.mark.parametrize("make", [lambda p: JobStore(), lambda p: SqliteJobStore(str(p / "jobs.db"))])
def test_pending_cap_cannot_be_overrun(tmp_path, make):
    store, got = make(tmp_path), []
    start = threading.Barrier(12)

    def go():
        start.wait()
        got.append(store.create_if_room("c", "o", 2))

    threads = [threading.Thread(target=go) for _ in range(12)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    made = [j for j in got if j is not None]
    assert len(made) == 2 and store.pending_count() == 2
    store.discard(made[0].id)
    assert store.pending_count() == 1 and store.create_if_room("c", "o", 2) is not None


def test_busy_answer_and_slot_returned_on_bad_upload(monkeypatch):
    monkeypatch.setenv("MAX_PENDING_JOBS", "1")
    monkeypatch.setattr(main, "store", JobStore())
    c = TestClient(main.app)
    bad = c.post("/v1/documents/extract", files=[("files", ("a.png", b"not an image", "image/png"))])
    assert bad.status_code == 415 and main.store.pending_count() == 0          # the reserved slot was given back

    class Held:                                                                # a pool that never runs the job
        def submit(self, *a, **k):
            pass
    monkeypatch.setattr(main, "_ocr_pool", Held())
    files = [("files", ("a.png", png(), "image/png"))]
    assert c.post("/v1/documents/extract", files=files).status_code == 202
    assert c.post("/v1/documents/extract", files=files).status_code == 503
    assert c.get("/healthz").status_code == 200


def test_jobs_run_on_a_dedicated_pool():
    from concurrent.futures import ThreadPoolExecutor
    import importlib
    fresh = importlib.reload(main)
    try:
        assert isinstance(fresh._ocr_pool, ThreadPoolExecutor) and fresh._ocr_pool._thread_name_prefix == "ocr"
    finally:
        fresh._ocr_pool.shutdown(wait=False)


# ------------------------------------------------------------------ 6. name clean-up must not erase real names silently
@pytest.mark.parametrize("names,surname,given", [
    ("SMITH<<JOHN<LESLIE", "SMITH", "JOHN LESLIE"), ("MARTIN<<GISELLE", "MARTIN", "GISELLE"),
    ("SMITH<<CECILE", "SMITH", "CECILE"), ("KESSEL", "KESSEL", ""), ("SELLECK", "SELLECK", "")])
def test_real_names_made_of_filler_like_letters_are_kept(names, surname, given):
    doc = analyse_mrz([line1(names), LINE2])["document"]
    assert doc["surname"] == surname and doc["givenNames"] == given


def test_misread_filler_run_is_removed_but_reviewed():
    result = analyse_mrz(["P<UTOXUAN<<JIANG".ljust(26, "K") + "SSKKKKKKKKKKKKKKKK", LINE2])
    assert result["document"]["givenNames"] == "JIANG"
    assert "mrz_name_letters_removed" in codes(result) and result["needsReview"]


def test_run_that_reaches_the_line_end_is_reviewed_not_accepted():
    result = analyse_mrz([("P<UTO" + "A" * 33 + "LESLIE"), LINE2])        # name field full: cannot tell name from fillers
    assert len("P<UTO" + "A" * 33 + "LESLIE") == 44 and result["needsReview"]


# ------------------------------------------------------------------ 7. PostgreSQL queue: heartbeat and fenced writes
PG = os.environ.get("TEST_DATABASE_URL")


@pytest.mark.skipif(not PG, reason="set TEST_DATABASE_URL to run PostgreSQL tests")
def test_pg_claim_is_renewed_and_stale_writes_are_dropped(monkeypatch):
    from app.pgstore import PgJobStore
    store = PgJobStore(PG, ttl_seconds=600)
    with store._conn() as c:
        c.execute("DELETE FROM jobs")
    job = store.create_if_room("c", "o", 5)
    store.submit(job.id, [b"img"], "passport", {"surname": "X" * 50})
    first = store.claim()
    with store._conn() as c:                                   # the claim is old ...
        c.execute("UPDATE jobs SET locked_ts = locked_ts - 10000 WHERE id=%s", (job.id,))
    assert store.heartbeat(job.id, first.token)                # ... but the worker is alive and renews it
    assert store.requeue_stale() == 0
    with store._conn() as c:                                   # now the worker really stops
        c.execute("UPDATE jobs SET locked_ts = locked_ts - 10000 WHERE id=%s", (job.id,))
    assert store.requeue_stale() == 1
    second = store.claim()
    assert second.token == first.token + 1
    assert not store.heartbeat(job.id, first.token)
    assert not store.finish(job.id, first.token, "failed", error={"code": "late"})   # the run that lost its claim is ignored
    assert store.get(job.id).status == "processing"
    assert store.finish(job.id, second.token, "done", result={"ok": True})
    assert store.get(job.id).result == {"ok": True}
    with store._conn() as c:                                   # nothing the caller sent outlives the job
        assert c.execute("SELECT images, hint, applicant FROM jobs WHERE id=%s", (job.id,)).fetchone() == (None, None, None)
    assert not store.finish(job.id, first.token, "failed", error={"code": "late"}) and store.get(job.id).status == "done"


@pytest.mark.skipif(not PG, reason="set TEST_DATABASE_URL to run PostgreSQL tests")
def test_pg_pending_cap_is_atomic():
    from app.pgstore import PgJobStore
    store = PgJobStore(PG, ttl_seconds=600)
    with store._conn() as c:
        c.execute("DELETE FROM jobs")
    got, start = [], threading.Barrier(8)

    def go():
        start.wait()
        got.append(store.create_if_room("c", "o", 2))

    threads = [threading.Thread(target=go) for _ in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert len([j for j in got if j]) == 2 and store.pending_count() == 2


def test_worker_uses_the_claim_token():
    from app import worker

    class Store:
        def __init__(self):
            self.finished, self.updates = [], []

        def claim(self):
            from app.pgstore import Claim
            c = Claim(("job1", [b""], None, {}))
            c.token = 7
            return c

        def heartbeat(self, job_id, token):
            return True

        def finish(self, job_id, token, status, result=None, error=None):
            self.finished.append((job_id, token, status))
            return False                                       # claim lost: must not fall back to an unfenced write

        def update(self, *a, **k):
            self.updates.append((a, k))

    s = Store()
    assert worker.work_once(s, lambda: Fixed())
    assert s.finished == [("job1", 7, "done")] and s.updates == []


def test_stale_and_attempt_settings_have_a_floor(monkeypatch):
    from app import config
    monkeypatch.setenv("JOB_STALE_SECONDS", "0")
    monkeypatch.setenv("JOB_MAX_ATTEMPTS", "-1")
    assert config.job_stale_seconds() >= 30 and config.job_max_attempts() >= 1


# ------------------------------------------------------------------ 8. second-reader fill is validated after merging
@pytest.mark.parametrize("second_line,score,expected", [
    ("Expiry Date: 2020/01/01", 0.95, "document_expired"),
    ("Expiry Date: 1430/01/01", 0.95, "expiry_hijri_not_checked"),
    ("Expiry Date: 2035/01/01", 0.40, "low_confidence_fields")])
def test_filled_expiry_is_checked_under_auto_accept(monkeypatch, second_line, score, expected):
    from app import pipeline
    from app.engines import tesseract_text
    monkeypatch.setenv("VISUAL_ZONE_AUTO_ACCEPT", "1")
    idn = good_id()
    first = [L("Iqama No: " + idn, 10), L("Name: AHMED ALI HASSAN", 40), L("Nationality: Egypt", 70), L("Date of Birth: 1990/05/05", 100)]
    monkeypatch.setattr(tesseract_text, "read_text_lines", lambda images: first + [L(second_line, 130, score)])
    result = pipeline._fill_from_second_reader(parse_gcc_id(first), [b""], "paddle-ocr")
    by_code = {f["code"]: f["severity"] for f in result["flags"]}
    assert result["document"].get("expiryDate") and by_code.get(expected) == "error"
    assert by_code["filled_from_second_reader"] == "error"


# ------------------------------------------------------------------ 9. a correction cannot undo a purge
def test_correction_after_purge_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "get_engine", lambda: Fixed())
    monkeypatch.setattr(main, "training", TrainingStore(tmp_path / "store"))
    c = TestClient(main.app)
    r = c.post("/v1/documents/extract", files=[("files", ("a.png", png(), "image/png"))], data={"consent_training": "true"})
    jid = r.json()["jobId"]
    body = {"reviewer": "r", "document": {"surname": "DUMMY"}, "lines": []}
    assert c.post(f"/v1/jobs/{jid}/corrections", json=body).status_code == 201
    out = subprocess.run([sys.executable, str(ROOT / "tools" / "purge_training.py"), "--store-dir", str(tmp_path / "store"),
                          "--job-id", jid], capture_output=True, text=True)
    assert "removed" in out.stdout
    assert c.post(f"/v1/jobs/{jid}/corrections", json=body).status_code == 409
    assert not (tmp_path / "store" / "jobs" / jid).exists()
    with pytest.raises(FileNotFoundError):                    # the marker also blocks a new image write for that job
        TrainingStore(tmp_path / "store").save_images(jid, [png()])


# ------------------------------------------------------------------ 10 + 11. derived training data: lifetime and permissions
def _train_args(tmp_path, **kw):
    base = dict(store_dir=tmp_path / "store", work_dir=tmp_path / "work", lang="ar", arch="arch", dict=None, synth_dir=[],
                models_dir=tmp_path / "models", min_new=1, min_val=5, min_gain=0.002, activate=False, keep_days=30)
    return Namespace(**{**base, **kw})


def test_auto_train_removes_crops_labels_and_old_runs(tmp_path):
    sys.path.insert(0, str(ROOT / "training"))
    import auto_train
    d = tmp_path / "store" / "jobs" / "job1"
    d.mkdir(parents=True)
    (d / "correction.json").write_text(json.dumps({"lines": [{"text": "SECRET<<NAME"}] * 3}))
    old = tmp_path / "work" / "20200101-0000" / "data"
    old.mkdir(parents=True)
    (old / "real_job0_0.png").write_bytes(b"x")
    past = time.time() - 40 * 86400
    os.utime(old.parent, (past, past))

    def runner(cmd, env=None):
        cmd = [str(c) for c in cmd]
        if Path(cmd[1]).name == "prepare_dataset.py":
            out = Path(cmd[cmd.index("--out") + 1])
            (out / "images").mkdir(parents=True)
            (out / "images" / "real_job1_0.png").write_bytes(b"crop")
            (out / "val_real.txt").write_text("images/real_job1_0.png\tSECRET<<NAME\n")
            (out / "stats.json").write_text(json.dumps({"realTrain": 3, "realVal": 0, "synthTrain": 0}))
        else:
            raise RuntimeError("the training step failed")

    report = auto_train.cycle(_train_args(tmp_path), runner=runner)
    assert report["outcome"] == "skipped_too_few_real_validation_lines"
    left = [p for p in (tmp_path / "work").rglob("*") if p.is_file()]
    assert not old.parent.exists()
    assert all("SECRET" not in p.read_text(errors="ignore") and "real_job1" not in p.name for p in left)

    with pytest.raises(RuntimeError):                         # a failed run cleans up as well
        auto_train.cycle(_train_args(tmp_path, min_val=0), runner=runner, now=datetime.now(timezone.utc) + timedelta(minutes=5))
    assert not [p for p in (tmp_path / "work").rglob("real_*")]


def test_purge_tool_cleans_a_hand_built_dataset(tmp_path):
    ds = tmp_path / "work" / "ds"
    (ds / "images").mkdir(parents=True)
    (ds / "images" / "real_abc123_0.png").write_bytes(b"x")
    (ds / "images" / "real_other_0.png").write_bytes(b"x")
    (ds / "train.txt").write_text("images/real_abc123_0.png\tSECRET\nimages/real_other_0.png\tKEEP\n")
    (tmp_path / "store").mkdir()
    subprocess.run([sys.executable, str(ROOT / "tools" / "purge_training.py"), "--store-dir", str(tmp_path / "store"),
                    "--job-id", "abc123", "--work-dir", str(tmp_path / "work")], check=True, capture_output=True)
    assert not (ds / "images" / "real_abc123_0.png").exists() and (ds / "images" / "real_other_0.png").exists()
    assert (ds / "train.txt").read_text() == "images/real_other_0.png\tKEEP\n"


def test_prepare_dataset_output_is_owner_only(tmp_path):
    store = TrainingStore(tmp_path / "store")
    store.save_images("job1", [png((400, 200))])
    store.save_correction("job1", {"reviewer": "r", "lines": [{"image": 0, "box": [10, 10, 300, 60], "text": "ABC", "lang": "en"}]})
    out = tmp_path / "out"
    code = ("import os,sys,runpy; os.umask(0o022); sys.argv=['prepare_dataset.py','--store-dir',sys.argv[1],'--out',sys.argv[2]];"
            "runpy.run_path(sys.argv[0] if False else %r, run_name='__main__')" % str(ROOT / "training" / "prepare_dataset.py"))
    subprocess.run([sys.executable, "-c", code, str(tmp_path / "store"), str(out)], check=True, capture_output=True, cwd=ROOT)
    paths = [out, *out.rglob("*")]
    assert len(paths) > 2 and all(stat.S_IMODE(p.stat().st_mode) & 0o077 == 0 for p in paths)
    assert "UMask=0077" in (ROOT / "deploy" / "visa-ocr-autotrain.service").read_text()


# ------------------------------------------------------------------ 12. image decode runs off the event loop
def test_shrink_runs_in_a_worker_thread(monkeypatch):
    seen = []
    loop_thread = []

    def fake_shrink(data):
        seen.append(threading.get_ident())
        return data

    monkeypatch.setattr(main, "shrink_for_ocr", fake_shrink)
    monkeypatch.setattr(main, "get_engine", lambda: Fixed())

    @main.app.get("/_loop_thread", include_in_schema=False)
    async def _loop_thread():
        loop_thread.append(threading.get_ident())
        return {}

    try:
        c = TestClient(main.app)
        c.get("/_loop_thread")
        assert c.post("/v1/documents/extract", files=[("files", ("a.png", png(), "image/png"))]).status_code == 202
        assert seen and loop_thread and seen[0] != loop_thread[0]
    finally:
        main.app.router.routes[:] = [r for r in main.app.router.routes if getattr(r, "path", "") != "/_loop_thread"]


# ------------------------------------------------------------------ 13. printed-page name check
def test_one_or_two_letters_at_the_end_is_a_mismatch_not_a_match():
    assert visual_name_status("KHAN", "ALIK", [{"text": "KHAN ALI"}])[0] == "mismatch"
    assert visual_name_status("KHAN", "MALI", [{"text": "KHAN MALIK"}])[0] == "mismatch"
    assert visual_name_status("KHAN", "ALI", [{"text": "KHAN ALYK"}, {"text": "KHAN ALI"}])[0] == "match"   # identical line wins


def test_name_ending_misread_is_flagged_at_high_confidence():
    lines = [line1("KHAN<<ALIK"), LINE2]
    result = analyse_mrz(lines)
    apply_mrz_policy(result, 0.98, [{"text": "KHAN ALI"}], Applicant(), mrz_lines=lines)
    assert "name_differs_from_visual_zone" in codes(result) and result["needsReview"]


def test_real_truncation_at_a_full_name_field_still_matches():
    full = ("P<UTO" + "ABCDEFGHIJKLMNOPQRS<<ABCDEFGHIJKLMNOPQR")[:44]
    assert len(full) == 44 and not full.endswith("<")
    result = analyse_mrz([full, LINE2])
    doc = result["document"]
    apply_mrz_policy(result, 0.5, [{"text": doc["surname"] + " " + doc["givenNames"] + "STUVW"}], Applicant(), mrz_lines=[full, LINE2])
    assert result["fieldStatus"]["names"] == "confirmed by printed page"


# ------------------------------------------------------------------ 14. ID number source is bound under auto-accept
def test_employer_number_is_not_taken_as_the_holders_number():
    holder, employer = good_id("2"), good_id("1")
    r = parse_gcc_id([L("Iqama No: " + holder[:-1], 10), L("Employer ID: " + employer, 40), L("Expiry Date: 2035/01/01", 70)],
                     auto_accept=True, min_confidence=0)
    assert r["needsReview"] and r["document"].get("idNumber") != employer


def test_repaired_or_unlabelled_number_always_reaches_a_person():
    holder = good_id("2")
    repaired = parse_gcc_id([L("Iqama No: " + holder[:4] + "?" + holder[5:], 10), L("Expiry Date: 2035/01/01", 40)],
                            auto_accept=True, min_confidence=0)
    if repaired["document"].get("idNumber") == holder:
        assert "id_number_repaired" in {f["code"] for f in repaired["flags"] if f["severity"] == "error"}
    assert repaired["needsReview"]
    other = good_id("1")
    unlabelled = parse_gcc_id([L("Iqama No: 12AB", 10), L(other, 40), L("Expiry Date: 2035/01/01", 70)], auto_accept=True, min_confidence=0)
    assert unlabelled["needsReview"]
    if unlabelled["document"].get("idNumber") == other:
        assert "id_number_unlabelled" in codes(unlabelled)


def test_clean_labelled_card_is_still_accepted_under_auto_accept():
    r = parse_gcc_id([L("Iqama No: " + good_id(), 10), L("Expiry Date: 2035/01/01", 40)], auto_accept=True)
    assert not r["needsReview"]


# ------------------------------------------------------------------ needs-validation leads that had a cheap source fix
def test_card_parser_line_count_is_bounded(monkeypatch):
    from app import pipeline
    monkeypatch.setenv("MAX_TEXT_LINES", "20")
    seen = {}
    real = pipeline.parse_gcc_id
    monkeypatch.setattr(pipeline, "parse_gcc_id", lambda lines, *a, **k: seen.setdefault("n", len(lines)) and real(lines, *a, **k))
    many = [L("Name", 10 * i) for i in range(500)]
    pipeline.run([png()], "gcc_id", Applicant(), Fixed(lines=None, text=many))
    assert seen["n"] == 20
