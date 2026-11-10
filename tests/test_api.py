import io

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

TD3 = [
    "P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<",
    "L898902C36UTO7408122F1204159ZE184226B<<<<<10",
]


def test_healthz_and_correlation_id():
    r = client.get("/healthz", headers={"X-Correlation-ID": "abc-123"})
    assert r.status_code == 200 and r.headers["X-Correlation-ID"] == "abc-123"


def test_mrz_parse_endpoint():
    r = client.post("/v1/mrz/parse", json={"lines": TD3})
    assert r.status_code == 200
    body = r.json()
    assert body["document"]["valid"] is True
    assert body["document"]["documentNumber"] == "L898902C3"
    # Specimen expired in 2012, so it needs review.
    assert body["needsReview"] is True
    assert "document_expired" in {f["code"] for f in body["flags"]}


def test_mrz_parse_rejects_unsupported():
    r = client.post("/v1/mrz/parse", json={"lines": ["ABC", "DEF"]})
    assert r.status_code == 422 and r.json()["detail"]["code"] == "mrz_unsupported"


def test_extract_rejects_non_image():
    r = client.post("/v1/documents/extract",
                    files=[("files", ("a.txt", b"hello", "text/plain"))])
    assert r.status_code == 415


def test_unknown_job():
    assert client.get("/v1/jobs/nope").status_code == 404


def test_extract_job_lifecycle_with_blank_image():
    Image = pytest.importorskip("PIL.Image")
    buf = io.BytesIO()
    Image.new("RGB", (800, 500), "white").save(buf, format="PNG")
    r = client.post("/v1/documents/extract",
                    files=[("files", ("p.png", buf.getvalue(), "image/png"))],
                    data={"doc_type_hint": "passport"})
    assert r.status_code == 202
    job = client.get(f"/v1/jobs/{r.json()['jobId']}").json()
    # No MRZ in a blank image; if Tesseract is not installed the engine reports that instead.
    assert job["status"] == "failed"
    assert job["error"]["code"] in {"mrz_not_found", "engine_unavailable"}


def test_ui_page_is_served_without_external_assets():
    r = client.get("/ui")
    assert r.status_code == 200 and "Visa OCR test page" in r.text
    assert "http://" not in r.text.replace("http://www.w3.org", "") and "https://" not in r.text


def test_expected_nationality_match_and_mismatch():
    from fastapi.testclient import TestClient
    import app.main as m
    c = TestClient(m.app)
    lines = ["P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<", "L898902C36UTO7408122F1204159ZE184226B<<<<<10"]
    ok = c.post("/v1/mrz/parse", json={"lines": lines, "expected_nationality": "uto"}).json()
    assert "nationality_mismatch" not in [f["code"] for f in ok["flags"]]
    bad = c.post("/v1/mrz/parse", json={"lines": lines, "expected_nationality": "SAU"}).json()
    assert "nationality_mismatch" in [f["code"] for f in bad["flags"]] and bad["needsReview"]
    junk = c.post("/v1/mrz/parse", json={"lines": lines, "expected_nationality": "Egypt"}).json()
    assert "expected_nationality_invalid" in [f["code"] for f in junk["flags"]]


def test_extract_accepts_jpeg_with_odd_content_type():
    import io
    from PIL import Image
    b = io.BytesIO(); Image.new("RGB", (64, 64), "white").save(b, "JPEG")
    for label in ("image/jpg", "application/octet-stream"):
        r = client.post("/v1/documents/extract", files=[("files", ("a.jpg", b.getvalue(), label))])
        assert r.status_code == 202, (label, r.text)
