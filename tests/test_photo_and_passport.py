"""The personal-photo check and the passport-only endpoint."""
import io

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFilter

from app import photo, pipeline
from app.engines.base import EngineOutput
from app.review import Applicant

PASSPORT = ["P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<", "L898902C36UTO7408122F1204159ZE184226B<<<<<10"]
VISA = ["V<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<", "L8988901C4XXX4009078F96121096ZE184226B<<<<<<"]
FACE = (200, 150, 200, 200)                      # where the stand-in detector says the face is, in a 600 x 750 photo


def _photo(size=(600, 750), background=(238, 238, 236), blur=0.0, level=1.0, colour=True, clutter=False) -> bytes:
    """A stand-in portrait: a light wall, a textured oval where the face is, shoulders below."""
    rng = np.random.default_rng(3)
    img = Image.new("RGB", size, background)
    draw = ImageDraw.Draw(img)
    if clutter:
        for _ in range(60):
            x, y = int(rng.integers(0, size[0])), int(rng.integers(0, 300))
            draw.rectangle((x, y, x + 60, y + 40), fill=tuple(int(v) for v in rng.integers(0, 255, 3)))
    draw.ellipse((size[0] * 0.1, size[1] * 0.75, size[0] * 0.9, size[1] * 1.4), fill=(40, 60, 110))
    x, y, w, h = FACE
    skin = (rng.random((h, w, 3)) * 70 + np.array([150, 120, 100])).clip(0, 255).astype("uint8")
    if not colour:
        skin[...] = skin.mean(axis=2, keepdims=True)
    mask = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mask).ellipse((0, 0, w, h), fill=255)
    img.paste(Image.fromarray(skin), (x, y), mask)
    if blur:
        img = img.filter(ImageFilter.GaussianBlur(blur))
    if level != 1.0:
        img = img.point(lambda v: int(min(255, v * level)))
    if not colour:
        img = img.convert("L").convert("RGB")
    out = io.BytesIO()
    img.save(out, format="PNG")
    return out.getvalue()


@pytest.fixture
def detector(monkeypatch):
    """The face detector ships with OpenCV 4 and is not in every test environment: here it is a stand-in that reports the
    face and eyes where the test put them."""
    found = {"faces": [FACE], "eyes": [(40, 50, 40, 30), (120, 52, 40, 30)]}
    monkeypatch.setattr(photo, "find_faces", lambda grey: list(found["faces"]))
    monkeypatch.setattr(photo, "find_eyes", lambda face: list(found["eyes"]))
    return found


def test_a_good_portrait_is_accepted(detector):
    v = photo.check(_photo())
    assert v["acceptable"] and v["issues"] == [] and v["doubts"] == [] and "retake" not in v
    assert v["measures"]["faces"] == 1 and v["measures"]["backgroundBrightness"] > 200


@pytest.mark.parametrize("kwargs,faces,expect", [
    ({"blur": 6}, None, "blurred"),
    ({"level": 0.3}, None, "too_dark"),
    ({"background": (50, 50, 50)}, None, "background_not_light"),
    ({"clutter": True}, None, "background_not_plain"),
    ({"size": (900, 600)}, None, "wrong_shape"),
    ({"size": (200, 250)}, [(70, 50, 60, 60)], "too_small"),
    ({}, [], "no_face"),
    ({}, [FACE, (20, 300, 190, 190)], "several_faces"),
    ({}, [(260, 300, 70, 70)], "face_too_small"),
    ({}, [(10, 60, 560, 600)], "face_too_close"),
    ({}, [(380, 150, 200, 200)], "not_centred"),
    ({}, [(200, 0, 200, 200)], "face_cut_off"),
])
def test_each_fault_is_named_with_what_to_do(detector, kwargs, faces, expect):
    if faces is not None:
        detector["faces"] = faces
    v = photo.check(_photo(**kwargs))
    assert not v["acceptable"] and expect in v["issues"]
    assert v["retake"]["recommended"] and v["retake"]["message"] == photo.MESSAGES[v["issues"][0]] and len(v["messages"]) >= 1


def test_doubts_are_reported_but_do_not_refuse(detector):
    detector["eyes"] = [(40, 50, 40, 30)]                                     # one eye found: glasses, hair, a closed eye
    v = photo.check(_photo(colour=False))
    assert v["acceptable"] and v["doubts"] == ["eyes_not_clear", "not_colour"] and v["retake"]["recommended"] is False
    detector["eyes"] = [(40, 40, 40, 30), (120, 90, 40, 30)]                  # the line through the eyes slopes
    assert photo.check(_photo())["doubts"] == ["head_tilted"]


def test_a_face_lost_in_the_dark_is_called_too_dark(detector):
    detector["faces"] = []
    assert photo.check(_photo(level=0.1))["issues"] == ["too_dark"]
    assert photo.check(b"not a picture")["issues"] == ["unreadable"]


def test_photo_endpoint_answers_at_once_and_stores_nothing(detector):
    from fastapi.testclient import TestClient
    import app.main as main
    client = TestClient(main.app)
    before = main.store.count()
    good = client.post("/v1/photos/check", files={"file": ("me.png", _photo(), "image/png")})
    assert good.status_code == 200 and good.json()["acceptable"] is True
    bad = client.post("/v1/photos/check", files={"file": ("me.png", _photo(blur=6), "image/png")}).json()
    assert bad["acceptable"] is False and bad["retake"]["reasons"][0] == "blurred"
    assert client.post("/v1/photos/check", files={"file": ("me.png", b"<html>", "image/png")}).status_code == 415
    assert main.store.count() == before


def test_photo_endpoint_says_so_when_the_detector_is_missing(monkeypatch):
    from fastapi.testclient import TestClient
    import app.main as main

    def missing(_grey):
        raise photo.Unavailable("haarcascade_frontalface_default.xml was not found")
    monkeypatch.setattr(photo, "find_faces", missing)
    r = TestClient(main.app).post("/v1/photos/check", files={"file": ("me.png", _photo(), "image/png")})
    assert r.status_code == 503 and r.json()["detail"]["code"] == "photo_check_unavailable"


class _Reader:
    name = "paddle-ocr"

    def __init__(self, lines, texts=()):
        self.lines, self.texts = lines, texts

    def extract(self, images, hint):
        assert hint != pipeline.PASSPORT_MRZ         # the readers are told "passport", never the endpoint's own word
        return EngineOutput(engine=self.name, mrz_lines=self.lines, mrz_confidence=0.99 if self.lines else None,
                            fields={"textLines": [{"text": t, "score": 0.98} for t in self.texts]}, mrz_searched=True)


def _page() -> bytes:
    out = io.BytesIO()
    Image.new("RGB", (400, 300), "white").save(out, "PNG")
    return out.getvalue()


def test_passport_only_reads_a_passport_and_turns_back_everything_else():
    status, res = pipeline.run([_page()], pipeline.PASSPORT_MRZ, Applicant(), _Reader(PASSPORT))
    assert status == "done" and res["document"]["kind"] == "passport" and all(res["document"]["checks"].values())
    status, res = pipeline.run([_page()], pipeline.PASSPORT_MRZ, Applicant(), _Reader(VISA))
    assert status == "failed" and res["code"] == "not_a_passport" and res["retake"]["recommended"]
    card = ["KINGDOM OF BAHRAIN", "IDENTITY CARD", "Name: Sample Person", "Nationality: Egypt", "Date of Birth: 05/03/1990",
            "Expiry Date: 11/06/2030", "المهنة مهندس برمجيات"]
    status, res = pipeline.run([_page()], pipeline.PASSPORT_MRZ, Applicant(), _Reader(None, card))
    assert status == "failed" and res["code"] == "mrz_not_found"
    status, res = pipeline.run([_page()], None, Applicant(), type("R", (_Reader,), {"extract": lambda self, images, hint: EngineOutput(
        engine="paddle-ocr", fields={"textLines": [{"text": t, "score": 0.98} for t in card]}, mrz_searched=True)})(None))
    assert status == "done" and res["document"]["format"] == "visual"        # the endpoint for all documents still reads the card


def test_passport_endpoint_runs_a_passport_only_job(monkeypatch):
    from fastapi.testclient import TestClient
    import app.main as main
    seen = []

    class Reader(_Reader):
        def extract(self, images, hint):
            seen.append(hint)
            return super().extract(images, hint)
    monkeypatch.setattr(main, "get_engine", lambda: Reader(PASSPORT))
    client = TestClient(main.app)
    r = client.post("/v1/passports/extract", files=[("files", ("p.png", _page(), "image/png"))], data={"expected_surname": "ERIKSSON"})
    assert r.status_code == 202
    job = client.get("/v1/jobs/" + r.json()["jobId"]).json()
    assert job["status"] == "done" and job["result"]["document"]["surname"] == "ERIKSSON" and seen == ["passport"]
    monkeypatch.setattr(main, "get_engine", lambda: Reader(VISA))
    r = client.post("/v1/passports/extract", files=[("files", ("p.png", _page(), "image/png"))])
    job = client.get("/v1/jobs/" + r.json()["jobId"]).json()
    assert job["status"] == "failed" and job["error"]["code"] == "not_a_passport"
    r = client.post("/v1/documents/extract", files=[("files", ("p.png", _page(), "image/png"))])
    assert client.get("/v1/jobs/" + r.json()["jobId"]).json()["status"] == "done"      # a visa is still read by the general endpoint
