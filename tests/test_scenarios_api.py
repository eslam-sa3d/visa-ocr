"""Scenario tests that need no OCR engine: validation rules and API behaviour for every document type,
including the bad-input cases. Image-level scenarios are in tools/run_scenarios.py."""
import io
from datetime import date

import pytest
from fastapi.testclient import TestClient

import app.main as main
from app.mrz import MrzError, check_digit, parse_mrz
from app.validation import cross_match, validate

TODAY = date(2026, 11, 9)
client = TestClient(main.app)

TD3 = ["P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<", "L898902C36UTO7408122F1204159ZE184226B<<<<<10"]


def flags(lines, **kw):
    return {f.code for f in validate(parse_mrz(lines, today=TODAY), TODAY)}


def with_char(line, idx, ch):
    return line[:idx] + ch + line[idx + 1:]


# ---- S1: every document format parses and passes its check digits ---------------------------------------------
@pytest.mark.parametrize("lines,fmt", [
    (TD3, "TD3"),
    (["V<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<", "L898902C36UTO7408122F1204159<<<<<<<<<<<<<<<<"], "MRVA"),
    (["I<UTOD231458907<<<<<<<<<<<<<<<", "7408122F1204159UTO<<<<<<<<<<<6", "ERIKSSON<<ANNA<MARIA<<<<<<<<<<"], "TD1"),
])
def test_formats_parse(lines, fmt):
    r = parse_mrz(lines, today=TODAY)
    assert r.format == fmt and r.surname == "ERIKSSON"


# ---- S2: each check digit field caught when corrupted -----------------------------------------------------------
@pytest.mark.parametrize("idx,field", [(9, "document_number"), (19, "birth_date"), (27, "expiry_date"), (43, "composite")])
def test_corrupt_check_digit_is_flagged(idx, field):
    bad = TD3[1][:idx] + str((int(TD3[1][idx]) + 1) % 10) + TD3[1][idx + 1:]
    r = parse_mrz([TD3[0], bad], today=TODAY)
    assert r.checks[field] is False
    assert "mrz_checksum_failed" in {f.code for f in validate(r, TODAY)}


# ---- S3: business rules --------------------------------------------------------------------------------------------
def test_expired_document_flagged():
    assert "document_expired" in flags(TD3)                      # expires 2012


def test_digit_lookalikes_are_corrected_and_reported():
    noisy = TD3[1][:15] + "O" + TD3[1][16:]               # a zero in the birth date read as the letter O
    r = parse_mrz([TD3[0], noisy], today=TODAY)
    assert r.valid and r.birth_date == "1974-08-12" and r.corrected_fields
    assert "ocr_digits_corrected" in {f.code for f in validate(r, TODAY)}


def test_two_digit_year_century_rule():
    def line2(dob):
        doc = "L898902C3"
        exp = "320101"
        body = doc + str(check_digit(doc)) + "UTO" + dob + str(check_digit(dob)) + "F" + exp + str(check_digit(exp)) + "<" * 14 + "0"
        comp = body[:10] + body[13:20] + body[21:43]
        return body + str(check_digit(comp))
    assert parse_mrz([TD3[0], line2("080102")], today=TODAY).birth_date == "2008-01-02"   # not in the future: 20xx
    assert parse_mrz([TD3[0], line2("501231")], today=TODAY).birth_date == "1950-12-31"   # would be future as 2050: 19xx


def test_non_mrz_characters_rejected_with_help():
    with pytest.raises(MrzError, match="Shift"):
        parse_mrz([TD3[0].replace("<<", "くく", 1), TD3[1]])


@pytest.mark.parametrize("lines", [[], ["only one line"], ["A" * 44], ["A" * 10, "B" * 10, "C" * 10, "D" * 10]])
def test_wrong_line_counts_rejected(lines):
    with pytest.raises(MrzError):
        parse_mrz(lines)


# ---- S4: cross-document match (visa vs passport) ------------------------------------------------------------------------
def test_cross_match_passport_and_visa():
    visa = parse_mrz(["V<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<", "L898902C36UTO7408122F2706013<<<<<<<<<<<<<<<<"], today=TODAY)
    ok = cross_match(parse_mrz(TD3, today=TODAY), visa, "passport", "visa")
    assert ok == [] or all(f.severity != "error" for f in ok)


# ---- S5: API behaviour ------------------------------------------------------------------------------------------------------
def png(size=(200, 100)):
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", size, "white").save(buf, format="PNG")
    return buf.getvalue()


def test_api_rejects_wrong_type_too_many_and_too_large():
    r = client.post("/v1/documents/extract", files=[("files", ("a.gif", b"GIF89a", "image/gif"))])
    assert r.status_code == 415 and r.json()["detail"]["code"] == "bad_type"
    many = [("files", (f"{i}.png", png(), "image/png")) for i in range(9)]
    assert client.post("/v1/documents/extract", files=many).status_code == 422
    big = [("files", ("big.png", b"0" * (10 * 1024 * 1024 + 1), "image/png"))]
    assert client.post("/v1/documents/extract", files=big).status_code == 413
    assert client.post("/v1/documents/extract").status_code == 422


def test_api_unknown_job_and_health():
    assert client.get("/v1/jobs/does-not-exist").status_code == 404
    assert client.get("/healthz").status_code == 200


def test_api_correlation_id_echoed_and_generated():
    assert client.get("/healthz", headers={"X-Correlation-Id": "abc-123"}).headers.get("x-correlation-id") == "abc-123"
    assert client.get("/healthz").headers.get("x-correlation-id")


def test_api_mrz_parse_bad_text_is_422_not_500():
    r = client.post("/v1/mrz/parse", json={"lines": ["hello", "world"]})
    assert r.status_code == 422 and r.json()["detail"]["code"] == "mrz_unsupported"


def test_api_job_with_blank_image_fails_cleanly(monkeypatch):
    from app.engines.base import EngineOutput

    class Blank:
        name = "blank"

        def extract(self, images, hint):
            return EngineOutput(engine="blank", mrz_lines=[])

    monkeypatch.setattr(main, "get_engine", lambda: Blank())
    r = client.post("/v1/documents/extract", files=[("files", ("a.png", png(), "image/png"))])
    job = client.get(f"/v1/jobs/{r.json()['jobId']}").json()
    assert job["status"] == "failed" and job["error"]["code"] == "mrz_not_found"
    assert job["result"] is None


def test_api_engine_crash_does_not_leak_details(monkeypatch):
    class Boom:
        name = "boom"

        def extract(self, images, hint):
            raise RuntimeError("secret path /etc/passwd")

    monkeypatch.setattr(main, "get_engine", lambda: Boom())
    r = client.post("/v1/documents/extract", files=[("files", ("a.png", png(), "image/png"))])
    job = client.get(f"/v1/jobs/{r.json()['jobId']}").json()
    assert job["status"] == "failed" and job["error"]["code"] == "internal_error"
    assert "passwd" not in str(job)


def test_api_expected_nationality_end_to_end(monkeypatch):
    from app.engines.base import EngineOutput

    class Fixed:
        name = "fixed"

        def extract(self, images, hint):
            return EngineOutput(engine="fixed", mrz_lines=TD3)

    monkeypatch.setattr(main, "get_engine", lambda: Fixed())
    r = client.post("/v1/documents/extract", files=[("files", ("a.png", png(), "image/png"))],
                    data={"expected_nationality": "SAU"})
    job = client.get(f"/v1/jobs/{r.json()['jobId']}").json()
    assert "nationality_mismatch" in [f["code"] for f in job["result"]["flags"]] and job["result"]["needsReview"]
