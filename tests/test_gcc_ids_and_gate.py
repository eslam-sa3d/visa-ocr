import json
import subprocess
import sys
from pathlib import Path

from app import gcc_ids
from app.visual_zone import parse_gcc_id

ROOT = Path(__file__).resolve().parent.parent


def luhn_complete(body: str) -> str:
    for c in "0123456789":
        if gcc_ids.luhn_ok(body + c):
            return body + c


def kuwait(first11: str) -> str:
    c = gcc_ids.kuwait_check_digit(first11)
    return first11 + str(0 if c == 11 else c % 10)


def test_emirates_id_is_recognised_and_checked():
    good = luhn_complete("78419901234567")
    info = gcc_ids.classify(good)
    assert info["kind"] == "uae_emirates_id" and info["valid"]
    bad = good[:-1] + str((int(good[-1]) + 1) % 10)
    assert not gcc_ids.classify(bad)["valid"]


def test_kuwait_civil_id_birth_date_is_inside_the_number():
    info = gcc_ids.classify(kuwait("29003151234"))
    assert info["kind"] == "kuwait_civil_id" and info["birthDate"] == "1990-03-15"
    assert gcc_ids.classify("290133151234") is None            # month 13


def test_other_lengths_are_not_guessed():
    assert gcc_ids.classify("12345678") is None and gcc_ids.classify("5" * 11) is None


def test_card_with_emirates_number_is_read_and_always_sent_to_review():
    num = luhn_complete("78419901234567")
    lines = [{"text": f"رقم الهوية {num[:3]}-{num[3:7]}-{num[7:14]}-{num[14]}", "score": 0.95},
             {"text": "تاريخ الميلاد 1990/03/15", "score": 0.95}, {"text": "تاريخ الانتهاء 2030/01/01", "score": 0.95}]
    res = parse_gcc_id(lines)
    assert res["document"]["kind"] == "uae_emirates_id" and res["document"]["idNumber"] == num
    codes = {f["code"]: f["severity"] for f in res["flags"]}
    assert codes["gcc_layout_unverified"] == "error" and res["needsReview"]


def test_birth_date_must_match_the_kuwait_number():
    num = kuwait("29003151234")
    lines = [{"text": f"الرقم المدني {num}", "score": 0.95}, {"text": "تاريخ الميلاد 1991/03/15", "score": 0.95}]
    res = parse_gcc_id(lines)
    assert any(f["code"] == "birth_date_differs_from_id_number" for f in res["flags"])


def test_saudi_iqama_path_is_unchanged():
    res = parse_gcc_id([{"text": "2612428256 رقم الهوية", "score": 0.95}])
    assert res["document"]["kind"] == "iqama" and not any(f["code"] == "gcc_layout_unverified" for f in res["flags"])


def test_acceptance_file_is_valid_and_gate_reads_a_report(tmp_path):
    cfg = json.loads((ROOT / "config" / "acceptance.json").read_text())
    assert cfg["synthetic"]["max_silent_wrong"] == 0 and cfg["real_specimens"]["min_specimens_per_family"] == 30
    (tmp_path / "report.json").write_text(json.dumps([{"docType": "passport:TD3", "verdict": "PASS", "needsReview": False}] * 3))
    out = subprocess.run([sys.executable, str(ROOT / "tools" / "accuracy_gate.py"), "--real-dir", str(tmp_path)],
                         capture_output=True, text=True)
    assert out.returncode == 1 and "only 3 specimens" in out.stdout            # too few real specimens must fail the gate


def _ln(text, x0, y0, x1, y1, score=0.9):
    return {"text": text, "score": score, "box": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]}


def test_premium_card_header_is_not_taken_for_the_name_and_nationality_needs_a_country():
    from app.visual_zone import parse_gcc_id
    lines = [_ln("KINGDOM OF SAUDI ARABIA", 800, 60, 1000, 90), _ln("المملكة العربية السعودية", 800, 100, 1000, 130),
             _ln("المحمد عبد الحميد محمد بسيوني", 600, 160, 1000, 200), _ln("MOHAMED ABDELHAMID MOHAMED BASYUNI", 500, 210, 1000, 245),
             _ln("Number: 2177914971", 500, 270, 800, 300), _ln("Nationality:", 500, 400, 640, 425), _ln("Egypt", 650, 400, 720, 425),
             _ln("يجب التحقق", 120, 400, 200, 425)]
    doc = parse_gcc_id(lines)["document"]
    assert doc["nameEn"] == "MOHAMED ABDELHAMID MOHAMED BASYUNI"
    assert "السعودية" not in doc.get("nameAr", "")
    assert doc["nationalityCode"] == "EGY"


def test_missing_dates_are_filled_from_the_second_reader(monkeypatch):
    from app import pipeline
    first = {"document": {"idNumber": "2177914971", "nameEn": "A B", "nameAr": "ا ب ج", "nationality": "Egypt", "nationalityCode": "EGY"},
             "flags": [{"code": "birthDate_not_found", "severity": "warning", "message": "x"},
                       {"code": "expiryDate_not_found", "severity": "warning", "message": "x"}]}
    second = {"document": {"birthDate": "1975-05-03", "birthCalendar": "gregorian", "expiryDate": "2029-09-07",
                           "expiryCalendar": "gregorian", "fieldConfidence": {"birthDate": 0.6, "expiryDate": 0.55}}, "flags": []}
    monkeypatch.setattr("app.engines.tesseract_text.read_text_lines", lambda images: [{"text": "x"}])
    monkeypatch.setattr(pipeline, "parse_gcc_id", lambda lines: second)
    out = pipeline._fill_from_second_reader(first, [b"x"], "paddle-ocr")
    assert out["document"]["birthDate"] == "1975-05-03" and out["document"]["expiryDate"] == "2029-09-07"
    codes = [f["code"] for f in out["flags"]]
    assert "filled_from_second_reader" in codes and not any(c.endswith("_not_found") for c in codes)


def test_one_unreadable_id_digit_is_restored_from_the_check_digit():
    from app.visual_zone import _repair_one_unreadable_digit
    assert _repair_one_unreadable_digit("297598/005") == "2975987005"
    assert _repair_one_unreadable_digit("1506523891") is None          # nothing unreadable
    assert _repair_one_unreadable_digit("12345") is None


def test_unlabelled_gcc_number_is_taken_only_when_it_is_the_only_one():
    num = "275070302656"
    lines = [{"text": "الجنسية الهند", "score": 0.9}, {"text": num + "  يندملا الرقم", "score": 0.87},
             {"text": "تاريخ الميلاد 1975/07/03", "score": 0.93}]
    assert parse_gcc_id(lines)["document"].get("idNumber") == num
