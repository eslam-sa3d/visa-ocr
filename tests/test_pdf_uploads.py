"""PDF uploads: each page is drawn as an image in a separate, limited process and read like a JPEG or PNG."""
import io
import subprocess

import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from tests.test_photo_and_passport import PASSPORT, _Reader


def _pdf(pages: int = 1, size=(1240, 1754)) -> bytes:
    imgs = []
    for n in range(pages):
        img = Image.new("RGB", size, "white")
        ImageDraw.Draw(img).text((100, 100 + 40 * n), f"page {n + 1}", fill="black")
        imgs.append(img)
    out = io.BytesIO()
    imgs[0].save(out, "PDF", resolution=150, save_all=True, append_images=imgs[1:])
    return out.getvalue()


def _png() -> bytes:
    out = io.BytesIO()
    Image.new("RGB", (400, 300), "white").save(out, "PNG")
    return out.getvalue()


@pytest.fixture
def client(monkeypatch):
    import app.main as main
    seen = []

    class Reader(_Reader):
        def extract(self, images, hint):
            seen.append(["image" if Image.open(io.BytesIO(i)).format in ("PNG", "JPEG") else "other" for i in images])
            return super().extract(images, hint)
    monkeypatch.setattr(main, "get_engine", lambda: Reader(PASSPORT))
    c = TestClient(main.app)
    c.seen = seen
    return c


def test_a_pdf_page_is_read_like_an_image_and_the_pdf_goes_no_further(client):
    r = client.post("/v1/documents/extract", files=[("files", ("passport.pdf", _pdf(), "application/pdf"))])
    assert r.status_code == 202
    job = client.get("/v1/jobs/" + r.json()["jobId"]).json()
    assert job["status"] == "done" and job["result"]["document"]["surname"] == "ERIKSSON"
    assert client.seen == [["image"]]                       # the readers got the drawn page, never the PDF


def test_pdf_label_does_not_matter_only_the_content(client):
    r = client.post("/v1/passports/extract", files=[("files", ("scan.jpg", _pdf(), "image/jpeg"))])
    assert r.status_code == 202 and client.seen == [["image"]]


def test_each_pdf_page_counts_as_one_image(client):
    r = client.post("/v1/documents/extract", files=[("files", ("a.pdf", _pdf(2), "application/pdf"))])
    assert r.status_code == 202 and client.seen == [["image", "image"]]
    r = client.post("/v1/documents/extract", files=[("files", ("a.pdf", _pdf(2), "application/pdf")),
                                                    ("files", ("b.pdf", _pdf(2), "application/pdf")),
                                                    ("files", ("c.png", _png(), "image/png"))])
    assert r.status_code == 422 and r.json()["detail"]["code"] == "bad_file_count"


def test_pdf_with_too_many_pages_is_turned_back(client):
    r = client.post("/v1/documents/extract", files=[("files", ("a.pdf", _pdf(3), "application/pdf"))])
    assert r.status_code == 422 and r.json()["detail"]["code"] == "pdf_too_many_pages"
    assert "3 pages" in r.json()["detail"]["message"]


def test_broken_and_password_protected_pdfs_are_refused(client):
    r = client.post("/v1/documents/extract", files=[("files", ("a.pdf", b"%PDF-1.7\nnot really", "application/pdf"))])
    assert r.status_code == 415 and r.json()["detail"]["code"] == "bad_pdf"
    pypdf = pytest.importorskip("pypdf")
    reader = pypdf.PdfReader(io.BytesIO(_pdf()))
    writer = pypdf.PdfWriter()
    writer.append_pages_from_reader(reader)
    writer.encrypt("secret")
    out = io.BytesIO()
    writer.write(out)
    r = client.post("/v1/documents/extract", files=[("files", ("a.pdf", out.getvalue(), "application/pdf"))])
    assert r.status_code == 415 and r.json()["detail"]["code"] == "pdf_encrypted"


def test_a_pdf_that_is_too_heavy_ends_only_its_own_process(monkeypatch):
    from app import pdf
    real = subprocess.run

    def timeout(*a, **k):
        raise subprocess.TimeoutExpired(a[0], k.get("timeout"))
    monkeypatch.setattr(pdf.subprocess, "run", timeout)
    with pytest.raises(pdf.PdfError) as exc:
        pdf.render(_pdf())
    assert exc.value.code == "pdf_too_complex"

    def killed(*a, **k):
        return subprocess.CompletedProcess(a[0], -9, b"", b"")       # killed by the memory or CPU limit
    monkeypatch.setattr(pdf.subprocess, "run", killed)
    with pytest.raises(pdf.PdfError) as exc:
        pdf.render(_pdf())
    assert exc.value.code == "pdf_too_complex"
    monkeypatch.setattr(pdf.subprocess, "run", real)
    monkeypatch.setenv("PDF_RENDER_MEMORY_MB", "256")
    assert len(pdf.render(_pdf())) == 1                    # a normal scan fits well inside the smallest limit


def test_the_drawing_process_runs_isolated_and_limited(monkeypatch):
    from app import pdf
    calls = []
    real = subprocess.run

    def spy(*a, **k):
        calls.append((a[0], k))
        return real(*a, **k)
    monkeypatch.setattr(pdf.subprocess, "run", spy)
    pdf.render(_pdf())
    cmd, kw = calls[0]
    assert cmd[1] == "-I" and kw["cwd"] == "/" and set(kw["env"]) <= {"PATH", "OMP_NUM_THREADS", "LANG"}
    assert "RLIMIT_AS" in cmd[3] and "RLIMIT_CPU" in cmd[3] and "RLIMIT_FSIZE" in cmd[3]


def test_large_pages_are_drawn_within_the_pixel_limits(monkeypatch):
    from app import config, pdf
    page = pdf.render(_pdf(size=(4000, 4000)))[0]          # a 26.7-inch square page
    w, h = Image.open(io.BytesIO(page)).size
    assert max(w, h) <= config.pdf_max_side() and w * h <= config.max_pixels()


def test_malware_scanner_sees_the_pdf_as_uploaded(client, monkeypatch):
    import app.main as main
    scanned = []
    monkeypatch.setattr(main, "scan_for_malware", lambda data: scanned.append(data[:5]))
    data = _pdf()
    r = client.post("/v1/documents/extract", files=[("files", ("a.pdf", data, "application/pdf"))])
    assert r.status_code == 202 and scanned == [b"%PDF-"]


def test_check_routes_take_the_first_page_of_a_pdf(monkeypatch):
    import app.main as main
    c = TestClient(main.app)
    r = c.post("/v1/documents/check", files={"file": ("a.pdf", _pdf(2), "application/pdf")})
    assert r.status_code == 200 and "quality" in r.json()
    r = c.post("/v1/documents/check", files={"file": ("a.pdf", _pdf(5), "application/pdf")})
    assert r.status_code == 200                             # only the first page is drawn for a check


def test_other_formats_are_still_refused(client):
    out = io.BytesIO()
    Image.new("RGB", (400, 300), "white").save(out, "GIF")
    r = client.post("/v1/documents/extract", files=[("files", ("a.gif", out.getvalue(), "image/gif"))])
    assert r.status_code == 415 and "PDF" in r.json()["detail"]["message"]


def test_review_everything_sends_every_finished_result_to_a_person(client, monkeypatch):
    r = client.post("/v1/documents/extract", files=[("files", ("p.png", _png(), "image/png"))],
                    data={"expected_surname": "ERIKSSON", "expected_given_names": "ANNA MARIA"})
    assert "review_everything" not in [f["code"] for f in client.get("/v1/jobs/" + r.json()["jobId"]).json()["result"]["flags"]]
    monkeypatch.setenv("REVIEW_EVERYTHING", "1")
    r = client.post("/v1/documents/extract", files=[("files", ("p.png", _png(), "image/png"))],
                    data={"expected_surname": "ERIKSSON", "expected_given_names": "ANNA MARIA"})
    res = client.get("/v1/jobs/" + r.json()["jobId"]).json()["result"]
    assert res["needsReview"] is True and "review_everything" in [f["code"] for f in res["flags"]]
