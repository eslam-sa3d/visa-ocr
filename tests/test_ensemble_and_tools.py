"""Ensemble engine logic (with fake readers), Arabic value-before-label, and the specimen / synthetic-data tools."""
import json
import subprocess
import sys
from pathlib import Path

from app.engines.base import EngineOutput, EngineUnavailable
from app.engines.ensemble import EnsembleEngine
from app.visual_zone import parse_gcc_id

ROOT = Path(__file__).resolve().parent.parent
GOOD = ["P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<", "L898902C36UTO7408122F1204159ZE184226B<<<<<10"]
BAD = [GOOD[0], "L898903C36UTO7408122F1204159ZE184226B<<<<<10"]       # document number check digit fails


class Fake:
    def __init__(self, lines=None, error=False):
        self.lines, self.error = lines, error

    def extract(self, images, hint):
        if self.error:
            raise EngineUnavailable("not installed")
        return EngineOutput(engine="fake", mrz_lines=self.lines)


def ensemble(t, r):
    e = EnsembleEngine.__new__(EnsembleEngine)
    e.tesseract, e.rapid = t, r
    e._text_fields = lambda images: {}
    return e


def test_agreement_with_all_check_digits_gives_confidence():
    out = ensemble(Fake(GOOD), Fake(GOOD)).extract([b""], None)
    assert out.mrz_lines == GOOD and out.mrz_confidence == 0.99


def test_agreement_on_a_failing_check_digit_gives_no_confidence():
    assert ensemble(Fake(BAD), Fake(BAD)).extract([b""], None).mrz_confidence is None


def test_disagreement_prefers_more_passing_check_digits_and_gives_no_confidence():
    out = ensemble(Fake(BAD), Fake(GOOD)).extract([b""], None)
    assert out.mrz_lines == GOOD and out.mrz_confidence is None


def test_without_second_reader_tesseract_result_is_unchanged():
    out = ensemble(Fake(GOOD), Fake(error=True)).extract([b""], None)
    assert out.mrz_lines == GOOD and out.mrz_confidence is None


def test_value_before_label_is_read():
    lines = [{"text": "2456789011 رقم الهوية", "score": 0.9}, {"text": "1988/11/02 تاريخ الميلاد", "score": 0.9},
             {"text": "تاريخ الانتهاء 2028/01/30", "score": 0.9}]
    doc = parse_gcc_id(lines)["document"]
    assert doc["idNumber"] == "2456789011" and doc["birthDate"] == "1988-11-02" and doc["expiryDate"] == "2028-01-30"


def test_garbled_expiry_label_is_guessed_and_sent_to_review():
    lines = [{"text": "2456789011 رقم الهوية", "score": 0.9}, {"text": "1988/11/02 تاريخ الميلاد", "score": 0.9},
             {"text": "تاريخ <ul 2028/01/30", "score": 0.8}]
    res = parse_gcc_id(lines)
    assert res["document"]["expiryDate"] == "2028-01-30"
    assert any(f["code"] == "expiry_label_guessed" and f["severity"] == "error" for f in res["flags"])
    assert res["needsReview"]


def test_specimen_tool_plans_and_counts(tmp_path):
    run = lambda *a: subprocess.run([sys.executable, str(ROOT / "tools" / "specimens.py"), *a], capture_output=True, text=True, check=True).stdout
    d = tmp_path / "rs"
    run("init", str(d))
    img = tmp_path / "a.jpg"
    img.write_bytes(b"x")
    run("add", str(d), str(img), "--type", "passport:TD3", "--country", "EGY", "--doc-no", "A1234567", "--birth", "1990-01-31")
    exp = json.loads((d / "expected.json").read_text())["scenarios"][0]
    assert exp["expected"]["documentNumber"] == "A1234567" and (d / exp["image"]).exists()
    assert "passport:TD3" in run("status", str(d))


def test_synthetic_mrz_generator_writes_labelled_lines(tmp_path):
    subprocess.run([sys.executable, str(ROOT / "training" / "synth_mrz.py"), "--out", str(tmp_path), "--count", "12"], check=True,
                   capture_output=True)
    rows = (tmp_path / "train.txt").read_text().splitlines() + (tmp_path / "val.txt").read_text().splitlines()
    assert len([r for r in rows if r]) == 12 and all(len(r.split("\t")[1]) in (30, 36, 44) for r in rows if r)


def test_auto_falls_back_to_ensemble_when_paddle_finds_no_mrz(monkeypatch):
    from app import engines
    from app.engines.base import EngineOutput

    class Fake:
        def __init__(self, out): self.out, self.calls = out, 0
        def extract(self, images, hint): self.calls += 1; return self.out

    paddle = Fake(EngineOutput(engine="paddle-ocr", mrz_lines=None))
    ens = Fake(EngineOutput(engine="ensemble", mrz_lines=["A" * 44, "B" * 44]))
    reg = {"paddle-ocr": paddle, "ensemble": ens}
    monkeypatch.setattr(engines, "get_engine_by_name", lambda n: reg[n])
    out = engines.AutoEngine().extract([b"x"], None)
    assert out.engine == "ensemble" and paddle.calls == 1 and ens.calls == 1

    # Paddle found the MRZ: no second pass
    paddle2 = Fake(EngineOutput(engine="paddle-ocr", mrz_lines=["A" * 44, "B" * 44], mrz_confidence=0.99))
    ens2 = Fake(EngineOutput(engine="ensemble"))
    reg.update({"paddle-ocr": paddle2, "ensemble": ens2})
    assert engines.AutoEngine().extract([b"x"], None).engine == "paddle-ocr" and ens2.calls == 0


def test_low_paddle_confidence_is_raised_only_when_the_second_pair_agrees(monkeypatch):
    from app import engines
    from app.engines.base import EngineOutput

    class Fake:
        def __init__(self, out): self.out = out
        def extract(self, images, hint): return self.out

    paddle = Fake(EngineOutput(engine="paddle-ocr", mrz_lines=GOOD, mrz_confidence=0.63))
    reg = {"paddle-ocr": paddle, "ensemble": Fake(EngineOutput(engine="ensemble", mrz_lines=GOOD, mrz_confidence=0.99))}
    monkeypatch.setattr(engines, "get_engine_by_name", lambda n: reg[n])
    assert engines.AutoEngine().extract([b"x"], None).mrz_confidence >= 0.97

    # the pair disagrees with Paddle: Paddle's low confidence stays, so a person checks
    reg["paddle-ocr"] = Fake(EngineOutput(engine="paddle-ocr", mrz_lines=GOOD, mrz_confidence=0.63))
    reg["ensemble"] = Fake(EngineOutput(engine="ensemble", mrz_lines=BAD, mrz_confidence=None))
    assert engines.AutoEngine().extract([b"x"], None).mrz_confidence == 0.63


def test_tilt_estimate_finds_a_rotated_page():
    from PIL import Image, ImageDraw
    from app.engines.tesseract_mrz import _estimate_tilt
    page = Image.new("L", (900, 600), 255)
    d = ImageDraw.Draw(page)
    for y in range(60, 560, 40):
        d.rectangle([60, y, 840, y + 12], fill=0)              # rows of "text"
    assert _estimate_tilt(page, Image) == 0
    tilted = page.rotate(-4, Image.BICUBIC, fillcolor=255)
    assert abs(_estimate_tilt(tilted, Image) - 4) <= 1          # the angle that turns it back level


def test_hard_deadline_setting_is_read(monkeypatch):
    from app import config
    monkeypatch.setenv("OCR_DEADLINE_SECONDS", "2")
    assert config.ocr_deadline() == 2.0
    monkeypatch.setenv("OCR_DEADLINE_SECONDS", "junk")
    assert config.ocr_deadline() == 0.0


# ---- readers differ only inside the optional data, at a look-alike its check digit cannot tell apart (real case, 11 Nov 2026)
def _td3_with_optional(optional: str, number: str = "KB3912527"):
    from app.mrz import check_digit as _cd

    def check_digit(v):
        return str(_cd(v))
    opt = optional.ljust(14, "<")
    body = number + check_digit(number) + "UTO" + "880402" + check_digit("880402") + "M" + "270202" + check_digit("270202") + opt + check_digit(opt)
    comp = body[0:10] + body[13:20] + body[21:43]
    return ["P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<", body + check_digit(comp)]


class _Stub:
    def __init__(self, lines):
        self.lines = lines

    def extract(self, images, hint):
        from app.engines.base import EngineOutput
        return EngineOutput(engine="stub", mrz_lines=self.lines)


def _ensemble(a, b):
    from app.engines.ensemble import EnsembleEngine
    e = EnsembleEngine.__new__(EnsembleEngine)
    e.name, e.tesseract, e.rapid = "ensemble", _Stub(a), _Stub(b)
    return e


def test_look_alike_in_optional_data_does_not_block_the_other_fields():
    from app import pipeline
    from app.review import Applicant
    a, b = _td3_with_optional("6901317"), _td3_with_optional("G901317")
    assert a != b and a[1][28] == "6" and b[1][28] == "G"
    status, res = pipeline.run([b""], None, Applicant(), _ensemble(a, b))
    assert status == "done" and res["needsReview"] is False
    assert res["document"]["optionalData"] == "6901317"                      # the digit fits a field of digits
    by = {f["code"]: f["severity"] for f in res["flags"]}
    assert by == {"optional_data_unconfirmed": "warning"}
    assert res["fieldStatus"]["optionalData"] == "unverified" and res["fieldStatus"]["names"] == "ocr confident"
    assert res["readers"]["differences"] == [{"line": 2, "position": 29, "tesseract": "6", "rapidocr": "G"}]


def test_optional_data_review_setting_forces_review(monkeypatch):
    from app import pipeline
    from app.review import Applicant
    monkeypatch.setenv("OPTIONAL_DATA_REVIEW", "1")
    status, res = pipeline.run([b""], None, Applicant(), _ensemble(_td3_with_optional("6901317"), _td3_with_optional("G901317")))
    assert res["needsReview"] is True


def test_look_alike_in_the_document_number_still_goes_to_review():
    from app import pipeline
    from app.review import Applicant
    a, b = _td3_with_optional("6901317", "KB3912527"), _td3_with_optional("6901317", "KB39L2527")
    status, res = pipeline.run([b""], None, Applicant(), _ensemble(a, b))
    assert res["needsReview"] is True and "unprotected_fields_unverified" in {f["code"] for f in res["flags"]}


def test_a_name_difference_still_goes_to_review():
    from app import pipeline
    from app.review import Applicant
    a = _td3_with_optional("6901317")
    b = [a[0].replace("ANNA", "ANNE"), a[1]]
    status, res = pipeline.run([b""], None, Applicant(), _ensemble(a, b))
    assert res["needsReview"] is True
