"""The photo is checked for clarity, and an unclear one comes back with what the applicant should do about it."""
import io

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFilter

from app import pipeline, quality
from app.engines.base import EngineOutput
from app.review import Applicant

GOOD = ["P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<", "L898902C36UTO7408122F1204159ZE184226B<<<<<10"]


def _page(blur=0.0, level=1.0, size=(1400, 1000)) -> bytes:
    """A page of printed lines on grey paper, optionally blurred or darkened."""
    rng = np.random.default_rng(1)
    img = Image.fromarray((rng.random((size[1], size[0])) * 20 + 190).astype("uint8"))
    draw = ImageDraw.Draw(img)
    for row in range(60, size[1] - 40, 38):
        for x in range(60, size[0] - 60, 26):
            draw.rectangle((x, row, x + 12, row + 18), fill=int(rng.integers(20, 70)))
    if blur:
        img = img.filter(ImageFilter.GaussianBlur(blur))
    if level != 1.0:
        img = img.point(lambda v: int(v * level))
    out = io.BytesIO()
    img.save(out, format="PNG")
    return out.getvalue()


class _Reader:
    name = "fake"

    def __init__(self, lines=None):
        self.lines, self.calls = lines, 0

    def extract(self, images, hint):
        self.calls += 1
        return EngineOutput(engine="ensemble", mrz_lines=self.lines, mrz_confidence=0.995, fields={"textLines": []}, confidence=0.9)


def test_clear_page_has_no_complaint():
    q = quality.measure(_page())
    assert q.clear and q.sharpness > 60


@pytest.mark.clarity
@pytest.mark.parametrize("image, reason", [
    (_page(blur=9), "blurred"),
    (_page(level=0.12), "too_dark"),
    (_page(size=(320, 240)), "too_small"),
])
def test_unusable_photo_is_turned_back_before_reading(image, reason):
    reader = _Reader(GOOD)
    status, res = pipeline.run([image], None, Applicant(), reader)
    assert status == "failed" and res["code"] == "image_unclear" and reader.calls == 0
    assert reason in res["retake"]["reasons"] and res["retake"]["recommended"]
    assert "take the photo again" in res["message"].lower() and res["message"] == quality.MESSAGES[reason]


@pytest.mark.clarity
def test_blank_white_image_is_turned_back():
    out = io.BytesIO()
    Image.new("RGB", (1200, 900), "white").save(out, format="PNG")
    status, res = pipeline.run([out.getvalue()], None, Applicant(), _Reader(GOOD))
    assert status == "failed" and res["code"] == "image_unclear"


@pytest.mark.clarity
def test_readable_photo_is_never_complained_about():
    """Soft focus, but every check digit passes: the reading stands and nobody is asked for another photo."""
    soft = _page(blur=3)
    assert quality.measure(soft).doubts and not quality.measure(soft).issues
    status, res = pipeline.run([soft], None, Applicant(), _Reader(GOOD))
    assert status == "done" and "retake" not in res
    assert not any(f["code"] == "retake_recommended" for f in res["flags"])


@pytest.mark.clarity
def test_failing_check_digit_asks_for_another_photo():
    wrong = [GOOD[0], GOOD[1][:9] + "0" + GOOD[1][10:]]
    status, res = pipeline.run([_page()], None, Applicant(), _Reader(wrong))
    assert status == "done" and res["needsReview"] and res["retake"]["reasons"] == ["mrz_unclear"]
    assert any(f["code"] == "retake_recommended" and f["severity"] == "warning" for f in res["flags"])
    soft = pipeline.run([_page(blur=3)], None, Applicant(), _Reader(wrong))[1]
    assert soft["retake"]["reasons"][0] == "blurred" and soft["retake"]["message"] == quality.MESSAGES["blurred"]


@pytest.mark.clarity
def test_nothing_read_says_what_to_do():
    status, res = pipeline.run([_page()], None, Applicant(), _Reader(None))
    assert status == "failed" and res["code"] == "mrz_not_found" and res["retake"]["reasons"] == ["not_found"]
    status, res = pipeline.run([_page(blur=3)], None, Applicant(), _Reader(None))
    assert res["code"] == "image_unclear" and res["retake"]["reasons"][0] == "blurred"


@pytest.mark.clarity
def test_check_can_be_switched_off(monkeypatch):
    monkeypatch.setenv("IMAGE_QUALITY_CHECK", "0")
    reader = _Reader(GOOD)
    status, _ = pipeline.run([_page(blur=9)], None, Applicant(), reader)
    assert status == "done" and reader.calls == 1


def test_unopenable_bytes_are_left_to_the_upload_checks():
    assert quality.measure(b"not an image").issues == []


def test_portal_can_check_a_photo_at_once():
    from fastapi.testclient import TestClient
    import app.main as main
    client = TestClient(main.app)
    good = client.post("/v1/documents/check", files={"file": ("a.png", _page(), "image/png")}).json()
    assert good["clear"] and good["acceptable"] and "retake" not in good
    bad = client.post("/v1/documents/check", files={"file": ("a.png", _page(blur=9), "image/png")}).json()
    assert not bad["acceptable"] and bad["retake"]["recommended"] and bad["retake"]["message"] == quality.MESSAGES["blurred"]
    soft = client.post("/v1/documents/check", files={"file": ("a.png", _page(blur=3), "image/png")}).json()
    assert soft["acceptable"] and not soft["clear"] and soft["retake"]["recommended"] is False
    assert client.post("/v1/documents/check", files={"file": ("a.png", b"junk", "image/png")}).status_code in (415, 422)
