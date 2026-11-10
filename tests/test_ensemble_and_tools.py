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
