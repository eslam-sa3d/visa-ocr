"""Regression tests for the third security audit (24 Nov 2026, commit 248aabe). One block per finding."""
import sys
from datetime import date
from pathlib import Path

from app.mrz import check_digit, parse_mrz
from app.pipeline import analyse_mrz
from app.review import Applicant, apply_mrz_policy, compare_applicant, names_match
from app.validation import validate

sys.path.insert(0, str(Path(__file__).resolve().parent))
TODAY = date(2026, 11, 24)


def cd(s: str) -> str:
    return str(check_digit(s))


def td3(names="ERIKSSON<<ANNA<MARIA", doc="L898902C3", expiry="320415", opt="", type_code="P"):
    l1 = (type_code.ljust(2, "<") + "UTO" + names).ljust(44, "<")
    d9 = doc.ljust(9, "<")
    o = opt.ljust(14, "<")
    body = d9 + cd(d9) + "UTO" + "740812" + cd("740812") + "F" + expiry + cd(expiry) + o + (cd(o) if opt else "<")
    return [l1, body + cd(body[0:10] + body[13:20] + body[21:43])]


def codes(lines):
    return {f.code for f in validate(parse_mrz(lines, TODAY), TODAY)}


def policy(lines, applicant=Applicant(), text_lines=None, agreed=None, confidence=None):
    result = analyse_mrz(lines)
    apply_mrz_policy(result, confidence, text_lines or [], applicant, lines, agreed)
    return result


# ------------------------------------------------ expiry year 99 is read as 2099
def test_expiry_more_than_twenty_years_ahead_goes_to_review():
    assert "expiry_century_check" in codes(td3(expiry="990415"))
    assert "expiry_century_check" not in codes(td3(expiry="320415"))          # a normal ten-year expiry


# ------------------------------------------------ impossible type code, digits in a name
def test_impossible_type_code_and_digits_in_names_are_flagged():
    assert "type_code_invalid" in codes(td3(type_code="ZZ"))
    assert "type_code_invalid" in codes(td3(type_code="9"))
    assert "type_code_invalid" in codes(td3(type_code="A"))                   # A is not a passport letter
    assert "type_code_invalid" not in codes(td3())
    assert "mrz_name_invalid_characters" in codes(td3(names="ER4KSS7N<<ANNA<MAR9A"))
    assert "mrz_name_invalid_characters" not in codes(td3())


# ------------------------------------------------ tidy of a number that passes its check digit
def test_tidy_of_a_check_digit_field_is_never_silent():
    lines = td3(doc="123456<K<")
    r = parse_mrz(lines, TODAY)
    assert not r.valid and "mrz_field_repaired" in {f.code for f in validate(r, TODAY)}
    clean = td3(doc="12345<7<<")              # the tidy would fail the check digit: left as read, no repair
    r = parse_mrz(clean, TODAY)
    assert r.valid and r.document_number == "12345<7"


# ------------------------------------------------ name clean-up erases a short given name
def test_stray_letters_removed_from_the_name_line_go_to_review():
    lines = td3(names="WANG<<<LI")
    r = parse_mrz(lines, TODAY)
    assert "mrz_name_letters_removed" in {f.code for f in validate(r, TODAY)}
    assert policy(lines, confidence=0.99, agreed=["names"])["needsReview"] is True


# ------------------------------------------------ document type fixed by the format without the letter
def test_type_letter_the_format_does_not_allow_is_not_confirmed_by_format():
    for letter in "FRBI":
        res = policy(td3(type_code=letter), text_lines=[{"text": "UTO"}], agreed=["nationality"])
        assert res["fieldStatus"]["documentType"] != "fixed by the document format"
        assert res["needsReview"] is True
    res = policy(td3(), text_lines=[{"text": "UTO"}], agreed=["nationality"])
    assert res["fieldStatus"]["documentType"] == "fixed by the document format"


# ------------------------------------------------ form name prefix with fillers left in the name field
def test_form_prefix_counts_only_when_the_name_field_is_full():
    short = td3(names="ALI<<OMAR")
    res = policy(short, Applicant(surname="ALIYEV", given_names="OMARION"))
    assert "applicant_name_mismatch" in {f["code"] for f in res["flags"]}
    assert res["fieldStatus"]["names"] != "confirmed by form"
    full = td3(names="ABDULRAHMANALMUTAIRI<<MOHAMMEDABDULLAHS")  # 39 characters: the field is full, the last word cut
    assert not full[0].endswith("<")
    res = policy(full, Applicant(surname="ABDULRAHMANALMUTAIRI", given_names="MOHAMMEDABDULLAHSALEH"))
    assert res["fieldStatus"]["names"] == "confirmed by form"
    assert names_match(["ALIYEV"], ["ALI"], name_field_full=True) and not names_match(["ALIYEV"], ["ALI"])


# ------------------------------------------------ surname alone confirms the given names
def test_surname_only_form_does_not_confirm_given_names():
    res = policy(td3(), Applicant(surname="ERIKSSON"))
    assert res["fieldStatus"]["names"] != "confirmed by form" and res["needsReview"] is True
    _, verified = compare_applicant({"surname": "ERIKSSON", "givenNames": ""}, Applicant(surname="ERIKSSON"))
    assert verified["names"] is True                       # no given names on the document: the surname is the whole name
    res = policy(td3(), Applicant(surname="ERIKSSON", given_names="ANNA MARIA"))
    assert res["fieldStatus"]["names"] == "confirmed by form"


# ------------------------------------------------ printed-page prefix after a cleaned name line
def test_printed_page_prefix_after_name_clean_up_is_not_a_match():
    lines = td3(names="ERIKSSON<<ANNA<MARI").copy()
    lines[0] = lines[0][:-1] + "K"                         # last filler misread as a letter
    res = policy(lines, text_lines=[{"text": "ERIKSSON ANNA MARIA"}])
    assert res["fieldStatus"]["names"] != "confirmed by printed page"
    assert res["needsReview"] is True


# ------------------------------------------------ MRZ text without '<' counted as the page's copy of the number
def test_mrz_line_without_fillers_is_not_a_page_copy_of_the_number():
    lines = td3()
    plain = lines[1].replace("<", "")
    res = policy(lines, text_lines=[{"text": plain}, {"text": lines[1][:10].replace("<", "")}])
    assert res["fieldStatus"].get("documentNumber") != "confirmed by printed page"
    from app.review import _page_values
    assert "L898902C3" not in _page_values([{"text": plain}], lines)["tokens"]
    assert "L898902C3" in _page_values([{"text": "No. L898902C3"}], lines)["tokens"]


# ------------------------------------------------ two readers agree on letters but not on the name split
def test_merged_filler_reading_does_not_confirm_the_name_split():
    from test_ensemble_and_tools import _ensemble, _td3_with_optional
    from app import pipeline
    good = _td3_with_optional("ZE184226B")
    merged = [good[0].replace("<<ANNA", "<ANNA") + "<", good[1]]
    for a, b in ((good, merged), (merged, good)):
        res = pipeline.run([b""], None, Applicant(), _ensemble(a, b))[1]
        assert res["document"]["surname"] == "ERIKSSON" and res["needsReview"] is True
    res = pipeline.run([b""], None, Applicant(), _ensemble(good, good))[1]
    assert res["needsReview"] is False                    # identical readings are still accepted


# ------------------------------------------------ card ID number taken from another label under auto-accept
def _card_line(text, y, x, score=0.95):
    return {"text": text, "score": score, "box": [x, y, x + 12 * len(text), y + 20]}


def test_card_number_from_another_label_is_never_auto_accepted():
    from app.visual_zone import parse_gcc_id
    emp, own = "1012345672", "2087654329"
    day = date(2026, 1, 1)
    base = [_card_line("Kingdom of Saudi Arabia", 10, 100), _card_line("MOHAMED ALI HASSAN", 60, 100)]
    tail = [_card_line("Expiry Date: 2035/01/01", 220, 100)]
    a = base + [_card_line("ID No", 110, 500), _card_line(emp, 160, 100), _card_line("Employer ID", 160, 500)] + tail
    d = [_card_line("المملكة العربية السعودية", 10, 300), _card_line("MOHAMED ALI HASSAN", 60, 100),
         _card_line("رقم الإقامة", 110, 600), _card_line(emp, 160, 100), _card_line("هوية صاحب العمل", 160, 600),
         _card_line("2035/01/01", 220, 100), _card_line("تاريخ الانتهاء", 220, 600)]
    e = base + [_card_line(emp, 110, 100), _card_line("Employer ID", 110, 300), _card_line("ID No", 110, 600)] + tail
    for lines in (a, d, e):
        res = parse_gcc_id(lines, today=day, auto_accept=True)
        assert res["needsReview"] is True
        assert any(f["code"] == "id_number_unlabelled" and f["severity"] == "error" for f in res["flags"])
    good = base + [_card_line(own, 110, 100), _card_line("ID No", 110, 500), _card_line(emp, 160, 100),
                   _card_line("Employer ID", 160, 500)] + tail
    res = parse_gcc_id(good, today=day, auto_accept=True)
    assert res["document"]["idNumber"] == own
    assert not any(f["code"] == "id_number_unlabelled" for f in res["flags"])


# ------------------------------------------------ sealed values bound to their row; plain data refused with a key
def _key(tmp_path, monkeypatch, cipher=""):
    from cryptography.fernet import Fernet
    k = tmp_path / "data.key"
    k.write_text(Fernet.generate_key().decode() + "\n")
    monkeypatch.setenv("DATA_KEY_FILE", str(k))
    if cipher:
        monkeypatch.setenv("DATA_CIPHER", cipher)
    else:
        monkeypatch.delenv("DATA_CIPHER", raising=False)


def test_sealed_result_does_not_open_after_owner_change_or_copy(tmp_path, monkeypatch):
    import sqlite3
    import pytest
    from app import crypto
    from app.jobs import SqliteJobStore
    for cipher in ("", "aes256gcm"):
        _key(tmp_path, monkeypatch, cipher)
        db = tmp_path / f"jobs{cipher}.db"
        store = SqliteJobStore(str(db))
        a, b = store.create("c1", owner="ownerA"), store.create("c2", owner="ownerB")
        store.update(a.id, status="done", result={"surname": "ALPHA"})
        store.update(b.id, status="done", result={"surname": "BETA"})
        assert store.get(a.id).result == {"surname": "ALPHA"}
        with sqlite3.connect(db) as c:
            c.execute("UPDATE jobs SET owner='ownerB' WHERE id=?", (a.id,))
        with pytest.raises(crypto.DataKeyError):
            store.get(a.id)
        with sqlite3.connect(db) as c:
            c.execute("UPDATE jobs SET owner='ownerA' WHERE id=?", (a.id,))
            val = c.execute("SELECT result FROM jobs WHERE id=?", (a.id,)).fetchone()[0]
            c.execute("UPDATE jobs SET result=? WHERE id=?", (val, b.id))      # copy A's sealed result into B's row
        with pytest.raises(crypto.DataKeyError):
            store.get(b.id)


def test_plain_stored_result_is_refused_with_a_key(tmp_path, monkeypatch):
    import json
    import sqlite3
    import pytest
    from app import crypto
    from app.jobs import SqliteJobStore
    _key(tmp_path, monkeypatch)
    store = SqliteJobStore(str(tmp_path / "jobs.db"))
    job = store.create("c", owner="o")
    store.update(job.id, status="done", result={"needsReview": True})
    with sqlite3.connect(tmp_path / "jobs.db") as c:
        c.execute("UPDATE jobs SET result=? WHERE id=?", (json.dumps({"needsReview": False, "document": {"surname": "FORGED"}}), job.id))
    with pytest.raises(crypto.DataKeyError):
        store.get(job.id)
    with pytest.raises(crypto.DataKeyError):
        crypto.unseal_json({"needsReview": False})


def test_timed_out_error_is_sealed_and_readable(tmp_path, monkeypatch):
    import sqlite3
    from app.jobs import SqliteJobStore
    _key(tmp_path, monkeypatch)
    monkeypatch.setenv("JOB_MAX_SECONDS", "0")
    store = SqliteJobStore(str(tmp_path / "jobs.db"))
    job = store.create("c", owner="o")
    with sqlite3.connect(tmp_path / "jobs.db") as c:
        c.execute("UPDATE jobs SET created_ts = created_ts - 120 WHERE id=?", (job.id,))
    got = store.get(job.id)
    assert got.status == "failed" and got.error["code"] == "timed_out"
    with sqlite3.connect(tmp_path / "jobs.db") as c:
        assert c.execute("SELECT error FROM jobs WHERE id=?", (job.id,)).fetchone()[0].startswith("VOENC")


# ================================================ audit trail, training approval, capacity, models
from tests.test_nca_controls import KEYS, events, h, roles, upload  # noqa: E402,F401  (roles is a fixture)


def test_cross_owner_read_and_refused_read_are_recorded(roles, monkeypatch):
    import hashlib
    import json
    from fastapi.testclient import TestClient
    import app.main as main
    client, log = roles
    job = upload(client).json()["jobId"]
    assert client.get(f"/v1/jobs/{job}", headers=h("reviewer")).status_code == 200
    assert client.get(f"/v1/jobs/{job}", headers=h("portal")).status_code == 200            # owner polling: not recorded
    got = [e for e in events(log) if e["event"] in ("job_read", "job_read_refused")]
    assert [e["event"] for e in got] == ["job_read"] and got[0]["cross_owner"] is True and got[0]["role"] == "reviewer"
    # a second uploader key asks for the first key's job
    keys = json.loads(open(main.os.environ["API_KEYS_FILE"]).read())
    keys.append({"name": "other", "sha256": hashlib.sha256(("q" * 40).encode()).hexdigest(), "role": "uploader"})
    open(main.os.environ["API_KEYS_FILE"], "w").write(json.dumps(keys))
    import os, time
    os.utime(main.os.environ["API_KEYS_FILE"], (time.time() + 5, time.time() + 5))
    assert client.get(f"/v1/jobs/{job}", headers={"X-API-Key": "q" * 40}).status_code == 404
    assert "job_read_refused" in [e["event"] for e in events(log)]


def test_throttled_caller_writes_one_line_per_window(roles):
    client, log = roles
    for _ in range(40):
        client.get("/v1/models", headers={"X-API-Key": "wrong" * 8})
    names = [e["event"] for e in events(log)]
    assert names.count("auth_throttled") == 1 and len(names) < 20


def test_unwritable_audit_file_is_reported_and_the_event_is_kept(tmp_path, monkeypatch, capsys):
    from fastapi.testclient import TestClient
    from app import audit
    import app.main as main
    monkeypatch.setenv("AUDIT_LOG_FILE", str(tmp_path / "missing-folder" / "audit.log"))
    audit.event("forbidden", key="k", role="uploader")
    assert audit.write_failed and '"forbidden"' in capsys.readouterr().err
    r = TestClient(main.app).get("/readyz")
    assert r.status_code == 503 and "security event log" in r.json()["detail"]["message"]


def test_events_without_a_file_go_to_stderr_and_name_the_account(monkeypatch, capsys):
    from app import audit
    monkeypatch.delenv("AUDIT_LOG_FILE", raising=False)
    rec = audit.event("training_data_purged", folders=2)
    assert '"training_data_purged"' in capsys.readouterr().err and rec["os_user"]


def _store_with_correction(tmp_path, lines, reviewer="amal"):
    from app.training_store import TrainingStore
    from tests.test_model_and_training import png_bytes
    ts = TrainingStore(tmp_path / "store")
    ts.save_images("job1", [png_bytes()])
    ts.save_correction("job1", {"reviewer": reviewer, "submittedBy": "key1", "lines": lines})
    return ts


def test_approval_names_the_record_and_a_later_correction_is_not_approved(tmp_path):
    import pytest
    from app.training_store import approved_correction, load_corrections, record_digest
    line = {"image": 0, "box": [5, 5, 200, 60], "text": "REVIEWED", "lang": "en"}
    ts = _store_with_correction(tmp_path, [line])
    job = tmp_path / "store" / "jobs" / "job1"
    seen = record_digest(load_corrections(job)[0][1])                      # what the approver read with --show
    ts.save_correction("job1", {"reviewer": "amal", "submittedBy": "key1", "lines": [{**line, "text": "PLANTED"}]})
    with pytest.raises(ValueError):
        ts.decide("job1", "omar")                                          # no record named: refused
    ts.decide("job1", "omar", correction=seen)
    assert approved_correction(job)["lines"][0]["text"] == "REVIEWED"
    assert [r["lines"][0]["text"] for _, r in ts.pending()] == ["PLANTED"]


def test_box_over_other_pixels_is_not_used_unless_its_crop_was_approved(tmp_path):
    from collections import Counter
    from app.training_store import load_corrections, record_digest
    from training import prepare_dataset
    ts = _store_with_correction(tmp_path, [{"image": 0, "box": [5, 5, 200, 60], "text": "SMYTH", "lang": "en"}])
    job = tmp_path / "store" / "jobs" / "job1"
    ts.decide("job1", "omar", correction=record_digest(load_corrections(job)[0][1]), approved_crops=[])
    out = tmp_path / "ds"
    (out / "images").mkdir(parents=True)
    skipped = Counter()
    assert prepare_dataset.from_store(tmp_path / "store", out, skipped=skipped) == []
    assert skipped["cropNotApproved"] == 1


def test_approval_tool_shows_crops_and_escapes_stored_text(tmp_path, capsys, monkeypatch):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
    import approve_corrections as tool
    ts = _store_with_correction(tmp_path, [{"image": 0, "box": [5, 5, 200, 60], "text": "A‮B\u009bC", "lang": "en"}],
                                reviewer="x‏y")
    monkeypatch.setattr(sys, "argv", ["t", "--store-dir", str(tmp_path / "store"), "--show", "job1",
                                      "--crops-dir", str(tmp_path / "crops")])
    tool.main()
    out = capsys.readouterr().out
    assert "‮" not in out and "\u009b" not in out and "‏" not in out and "\\u202e" in out
    assert list((tmp_path / "crops").glob("*.png")) and "sha256" in out


def test_api_refuses_control_and_direction_characters_and_bounds_corrections(roles, monkeypatch):
    import app.main as main
    client, _ = roles
    job = upload(client).json()["jobId"]
    for body in ({"reviewer": "a‮b", "document": {}, "lines": []},
                 {"reviewer": "a", "document": {"surname\u009b": "X"}, "lines": []},
                 {"reviewer": "a", "document": {}, "lines": [{"image": 0, "box": [0, 0, 10, 10], "text": "A⁦B", "lang": "en"}]},
                 {"reviewer": "a", "document": {}, "lines": [{"image": 999, "box": [0, 0, 10, 10], "text": "AB", "lang": "en"}]},
                 {"reviewer": "a", "document": {}, "lines": [{"image": 0, "box": [0, 0, 10, 10**9], "text": "AB", "lang": "en"}]}):
        assert client.post(f"/v1/jobs/{job}/corrections", json=body, headers=h("reviewer")).status_code == 422
    assert client.post(f"/v1/jobs/{job}/corrections", json={"reviewer": "أمل", "document": {"surname": "محمد"}, "lines": []},
                       headers=h("reviewer")).status_code == 201                  # Arabic needs no formatting characters
    monkeypatch.setenv("MAX_CORRECTIONS_PER_JOB", "2")
    codes = [client.post(f"/v1/jobs/{job}/corrections", json={"reviewer": "a", "document": {}, "lines": []},
                         headers=h("reviewer")).status_code for _ in range(3)]
    assert codes == [201, 429, 429]


def test_correction_limit_applies_with_the_general_limit_off(roles, monkeypatch):
    client, _ = roles
    job = upload(client).json()["jobId"]
    monkeypatch.setenv("RATE_LIMIT_PER_MIN", "0")
    monkeypatch.setenv("CORRECTIONS_PER_MIN", "2")
    monkeypatch.setenv("MAX_CORRECTIONS_PER_JOB", "100")
    codes = [client.post(f"/v1/jobs/{job}/corrections", json={"reviewer": "a", "document": {}, "lines": []},
                         headers=h("reviewer")).status_code for _ in range(4)]
    assert codes[:2] == [201, 201] and codes[2:] == [429, 429]


def test_unknown_checked_keys_do_not_count_for_calibration(roles):
    import app.main as main
    from app.training_store import load_corrections
    client, _ = roles
    job = upload(client).json()["jobId"]
    client.post(f"/v1/jobs/{job}/corrections", json={"reviewer": "a", "document": {"zz_not_a_field": "x"}, "lines": []},
                headers=h("reviewer"))
    rec = load_corrections(main.training.root / "jobs" / job)[-1][1]
    assert rec["ocr"]["fieldsChecked"] == [] and rec["ocr"]["fieldsChanged"] == []


def test_job_failed_as_timed_out_is_never_read_or_turned_done():
    from app import worker
    from app.jobs import JobStore
    store = JobStore()
    job = store.create("c", "o")
    store.update(job.id, status="failed", error={"code": "timed_out"})
    called = []

    class Engine:
        def extract(self, *a):
            called.append(1)
            raise AssertionError("must not read")
    worker.process_job(store, job.id, [b""], None, Applicant(), lambda: Engine())
    assert not called and store.get(job.id).status == "failed"


def test_admission_counts_work_the_process_still_holds(monkeypatch):
    from fastapi.testclient import TestClient
    import app.main as main
    from tests.test_model_and_training import png_bytes
    monkeypatch.setenv("MAX_PENDING_JOBS", "2")
    held = {"a", "b"}
    monkeypatch.setattr(main, "_inflight", set(held))
    r = TestClient(main.app).post("/v1/documents/extract", files=[("files", ("p.png", png_bytes(), "image/png"))])
    assert r.status_code == 503 and r.json()["detail"]["code"] == "busy"


def test_check_routes_decode_at_working_size_and_report_the_uploaded_size():
    import io
    from PIL import Image
    from app import photo, quality
    b = io.BytesIO()
    Image.new("RGBA", (4000, 3000), (0, 0, 0, 0)).save(b, "PNG")
    q = quality.measure(b.getvalue())
    assert (q.width, q.height) == (4000, 3000) and q.brightness > 250       # transparency laid on white
    img, size = quality.open_reduced(b.getvalue(), "L", 1000)
    assert size == (4000, 3000) and max(img.size) == 1000
    try:
        assert photo.check(b.getvalue())["measures"]["width"] == 4000
    except photo.Unavailable:
        pass                                               # no face model on this machine; the size rule is the same code


def test_check_routes_share_the_decode_limit(monkeypatch):
    import asyncio
    import app.main as main
    monkeypatch.setattr(main, "_shrink_slots", None)
    slots = main._decode_slots()
    assert slots is main._decode_slots() and slots._value == 2
    import inspect
    for route in (main.check_image, main.check_photo, main.extract):
        assert "_decode_slots()" in inspect.getsource(route)


def test_older_signed_manifest_put_back_is_refused(tmp_path, monkeypatch):
    import pytest
    from app.model_registry import ManifestError, ModelInfo, Registry, _HIGHEST
    monkeypatch.setenv("MANIFEST_HMAC_KEY", "k" * 32)
    monkeypatch.setenv("MANIFEST_STATE_FILE", str(tmp_path / "state.json"))
    _HIGHEST.clear()
    reg = Registry(tmp_path / "models")
    reg.save([ModelInfo(id="rec_A", task="text_recognition", lang="en", version="1", path="a", active=True)])
    state1 = reg.manifest_path.read_bytes()
    assert reg.models()[0].id == "rec_A"
    reg.save([ModelInfo(id="rec_B", task="text_recognition", lang="en", version="2", path="b", active=True)])
    assert reg.models()[0].id == "rec_B"
    reg.manifest_path.write_bytes(state1)                  # someone without the key restores the older signed file
    with pytest.raises(ManifestError):
        reg.models()
    _HIGHEST.clear()                                       # a restarted server remembers through the state file
    with pytest.raises(ManifestError):
        Registry(tmp_path / "models").models()


def test_tesseract_model_needs_a_signed_checksum_file_when_a_key_is_set(tmp_path, monkeypatch):
    import hashlib
    from app.engines.tesseract_mrz import tessdata_verified
    from app.model_registry import sign_bytes
    folder = tmp_path / "tessdata"
    folder.mkdir()
    (folder / "ocrb.traineddata").write_bytes(b"model")
    (folder / "SHA256SUMS").write_text(hashlib.sha256(b"model").hexdigest() + "  ocrb.traineddata\n")
    monkeypatch.delenv("ALLOW_UNVERIFIED_MODELS", raising=False)
    monkeypatch.delenv("MANIFEST_HMAC_KEY", raising=False)
    assert tessdata_verified(folder, "ocrb")              # no key: the checksum file is enough, as before
    monkeypatch.setenv("MANIFEST_HMAC_KEY", "k" * 32)
    assert not tessdata_verified(folder, "ocrb")          # key set, checksum file unsigned: refused
    (folder / "SHA256SUMS.sig").write_text(sign_bytes((folder / "SHA256SUMS").read_bytes()))
    assert tessdata_verified(folder, "ocrb")


def test_production_gate_refuses_reading_plain_stored_data(monkeypatch):
    import app.main as main
    monkeypatch.setenv("DATA_ALLOW_PLAINTEXT_READ", "1")
    assert any("DATA_ALLOW_PLAINTEXT_READ" in p for p in main.production_problems())


# ================================================ hardening for the leads that need a fact from the servers
def test_model_id_that_leaves_its_folder_is_refused(tmp_path):
    import json
    import pytest
    from app.model_registry import ManifestError, Registry
    reg = Registry(tmp_path / "models")
    reg.base.mkdir()
    for bad in ("../../etc", "a/b", "..", ""):
        reg.manifest_path.write_text(json.dumps({"models": [{"id": bad, "task": "text_recognition", "lang": "ar",
                                                              "version": "1", "path": "x"}]}))
        with pytest.raises(ManifestError):
            reg.models()


def test_production_flag_with_a_comment_left_in_it_stops_the_service(monkeypatch):
    import pytest
    from app import config
    for value, want in (("1", True), (" true ", True), ("0", False), ("", False)):
        monkeypatch.setenv("PRODUCTION", value)
        assert config.production() is want
    monkeypatch.setenv("PRODUCTION", "1   # refuses to start in any unsafe set-up")
    with pytest.raises(RuntimeError):
        config.production()
    env = (Path(__file__).resolve().parents[1] / "deploy/production/visa-ocr.env").read_text()
    assert not [ln for ln in env.splitlines() if "=" in ln and not ln.lstrip().startswith("#") and "#" in ln]


def test_entrypoint_reads_the_production_flag_like_the_service(tmp_path):
    import subprocess
    script = Path(__file__).resolve().parents[1] / "docker-entrypoint.sh"
    r = subprocess.run(["sh", str(script)], env={"PATH": "/usr/bin:/bin", "PRODUCTION": "1 # comment"},
                       capture_output=True, text=True, timeout=30)
    assert r.returncode != 0 and "neither 1 nor 0" in r.stderr


def test_spreadsheet_cells_are_never_formulas_and_round_trip():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
    import read_folder
    for v in ("=HYPERLINK(1)", "+1", "-2", "@SUM(1)"):
        assert read_folder.cell(v).startswith("'") and read_folder.uncell(read_folder.cell(v)) == v
    assert read_folder.cell("ERIKSSON") == "ERIKSSON" and read_folder.uncell("O'NEIL") == "O'NEIL"
    assert "/^[=+@\\t\\r-]/" in read_folder.REVIEW_PAGE and "forget()" in read_folder.REVIEW_PAGE


def test_test_tools_take_the_key_from_the_environment():
    import re
    root = Path(__file__).resolve().parents[1] / "tools"
    for name in ("system_test.py", "capacity_test.py", "perf_test.py", "load_test.py"):
        assert "VISA_OCR_API_KEY" in (root / name).read_text()
    steps = (Path(__file__).resolve().parents[1] / "deploy/release/DEPLOYMENT.txt").read_text()
    assert not re.search(r"--key <", steps) and 'X-API-Key: <' not in steps


def test_offline_selftest_drops_root_and_uses_a_private_log():
    script = (Path(__file__).resolve().parents[1] / "tools/offline_selftest.sh").read_text()
    assert "setpriv" in script and "mktemp" in script and "/tmp/offline_selftest_server.log" not in script
    assert script.count("$DROP") == 2 and script.count(" -I -c") == 3


def test_upload_without_a_valid_key_is_refused_before_its_body_is_read(monkeypatch):
    from fastapi.testclient import TestClient
    import app.main as main
    from app import security
    monkeypatch.setenv("API_KEYS", "s" * 32)
    monkeypatch.setattr(security, "limiter", security.RateLimiter())
    read = []

    async def app_that_reads(scope, receive, send):
        read.append(await receive())
    guard = security.BodyLimit(app_that_reads)
    import asyncio

    async def run(headers):
        sent = []

        async def receive():
            return {"type": "http.request", "body": b"x" * 1000, "more_body": False}

        async def send(msg):
            sent.append(msg)
        await guard({"type": "http", "method": "POST", "path": "/v1/documents/extract", "headers": headers,
                     "query_string": b"", "client": ("1.2.3.4", 1), "server": ("t", 80), "scheme": "http"}, receive, send)
        return sent
    sent = asyncio.run(run([(b"x-api-key", b"wrong")]))
    assert sent[0]["status"] == 401 and not read
    sent = asyncio.run(run([(b"x-api-key", ("s" * 32).encode())]))
    assert read                                            # a valid key goes on to the route
    r = TestClient(main.app).post("/v1/documents/extract", files=[("files", ("a.png", b"x" * 100, "image/png"))])
    assert r.status_code == 401
