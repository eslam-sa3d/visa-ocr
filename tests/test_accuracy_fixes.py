"""Regression tests for the accuracy review (docs/ACCURACY_REVIEW.md)."""
import shutil
import sys
from datetime import date
from pathlib import Path

import pytest

from app.countries import country_name
from app.mrz import check_digit, parse_mrz
from app.review import Applicant, apply_mrz_policy
from app.validation import validate

TODAY = date(2026, 11, 9)


def cd(s):
    return str(check_digit(s))


def td2_long(doc="ABC1234567", opt="XY"):
    d9, over = doc[:9], doc[9:]
    l1 = ("I<UTOERIKSSON<<ANNA<MARIA").ljust(36, "<")
    optf = (over + cd(doc) + "<" + opt).ljust(7, "<")
    body = d9 + "<" + "UTO" + "740812" + cd("740812") + "F" + "320415" + cd("320415") + optf
    return [l1, body + cd(body[0:10] + body[13:20] + body[21:35])]


def td1_long(doc="SRC0000000001", opt="ZZ"):
    d9, over = doc[:9], doc[9:]
    l1 = ("I<USA" + d9 + "<" + (over + cd(doc) + "<" + opt).ljust(15, "<"))
    l2a = "740812" + cd("740812") + "F" + "320415" + cd("320415") + "USA" + "<" * 11
    comp = cd(l1[5:30] + l2a[0:7] + l2a[8:15] + l2a[18:29])
    return [l1, l2a + comp, "ERIKSSON<<ANNA<MARIA".ljust(30, "<")]


def td3(doc="L898902C3", nat="UTO", opt="", names="ERIKSSON<<ANNA<MARIA"):
    l1 = ("P<UTO" + names).ljust(44, "<")
    d9 = doc.ljust(9, "<")
    o = opt.ljust(14, "<")
    body = d9 + cd(d9) + nat + "740812" + cd("740812") + "F" + "320415" + cd("320415") + o + (cd(o) if opt else "<")
    return [l1, body + cd(body[0:10] + body[13:20] + body[21:43])]


def test_td2_long_document_number_is_reassembled():
    r = parse_mrz(td2_long(), TODAY)
    assert r.valid and r.document_number == "ABC1234567" and r.optional_data == "XY"


def test_td1_overflow_optional_data_has_no_leading_filler():
    r = parse_mrz(td1_long(), TODAY)
    assert r.valid and r.document_number == "SRC0000000001" and r.optional_data == "ZZ"


def test_look_alike_letters_in_document_number_are_repaired_and_sent_to_review():
    lines = td3(doc="L4JRS08N0")
    bad = [lines[0], lines[1].replace("L4JRS08N0", "L4JRSO8NO", 1)]
    r = parse_mrz(bad, TODAY)
    assert r.document_number == "L4JRS08N0" and r.repaired_fields
    assert not r.valid                                  # as read, the check digit failed: never reported as clean
    codes = {f.code: f.severity for f in validate(r, TODAY)}
    assert codes["mrz_field_repaired"] == "error"


def test_valid_number_is_left_untouched():
    # a number that already passes its check digit is never rewritten
    r = parse_mrz(td3(doc="AB12345CD"), TODAY)
    assert not r.repaired_fields and r.valid


def test_digit_in_country_code_becomes_letter():
    lines = td3(doc="L898902C3", nat="COD")
    bad = [lines[0], lines[1][:10] + "C0D" + lines[1][13:]]
    r = parse_mrz(bad, TODAY)
    assert r.nationality == "COD" and any("0->O" in c for c in r.corrected_fields)


def test_stray_letter_after_names_is_a_misread_filler():
    lines = td3(names="KIM<<OLUWASEUN")
    l1 = lines[0][:30] + "K" + lines[0][31:]
    assert parse_mrz([l1, lines[1]], TODAY).given_names == "OLUWASEUN"


def test_k_for_filler_in_optional_data_is_removed():
    lines = td3(opt="SJ9LM")
    bad = [lines[0], lines[1][:34] + "K" + lines[1][35:]]
    r = parse_mrz(bad, TODAY)
    assert r.valid and r.optional_data == "SJ9LM"


def test_passport_read_as_visa_is_flagged():
    lines = td3(doc="L898902C3", opt="ZE184226B")
    bad = ["V" + lines[0][1:], lines[1]]
    r = parse_mrz(bad, TODAY)
    assert r.format == "MRVA"
    assert "type_code_inconsistent" in {f.code for f in validate(r, TODAY)}


def test_genuine_visa_is_not_flagged_as_passport():
    visa = ["V<UTOERIKSSON<<ANNA<MARIA".ljust(44, "<"),
            ("L898902C3" + cd("L898902C3") + "UTO740812" + cd("740812") + "F320415" + cd("320415") + "ABCDEF").ljust(44, "<")]
    assert "type_code_inconsistent" not in {f.code for f in validate(parse_mrz(visa, TODAY), TODAY)}


@pytest.mark.parametrize("code", ["D", "GBD", "GBN", "GBO", "GBP", "GBS", "UNO", "UNA", "UNK", "EUE", "RKS", "XKX"])
def test_icao_special_codes_have_names(code):
    assert country_name(code)


def _policy(lines, confidence, applicant=None):
    from app.mrz import parse_mrz as p
    r = p(lines, TODAY)
    res = {"document": r.to_dict(), "flags": [f.to_dict() for f in validate(r, TODAY)]}
    apply_mrz_policy(res, confidence, [], applicant or Applicant())
    return res


def test_policy_covers_document_type_and_ambiguous_characters_without_confidence():
    res = _policy(td3(doc="L61G8SCT3"), None)
    codes = {f["code"] for f in res["flags"]}
    assert "document_number_ambiguous_characters" in codes
    assert res["fieldStatus"]["documentType"] == "unverified"


def test_policy_accepts_when_confident_or_form_confirms():
    res = _policy(td3(doc="L61G8SCT3"), 0.99)
    assert "document_number_ambiguous_characters" not in {f["code"] for f in res["flags"]}
    assert res["fieldStatus"]["documentType"] == "ocr confident"
    res = _policy(td3(doc="L61G8SCT3"), None, Applicant(document_number="L61G8SCT3"))
    assert "document_number_ambiguous_characters" not in {f["code"] for f in res["flags"]}


@pytest.mark.skipif(shutil.which("tesseract") is None, reason="Tesseract not installed")
def test_tilted_photo_is_read():
    import io
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
    import make_scenarios as M
    from app.engines.tesseract_mrz import TesseractMrzEngine
    lines = td3(doc="L898902C3", opt="ZE184226B")
    img = M.v_rot_p4(M.draw_document({"lines": lines, "label": "TILT"}))
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "JPEG", quality=90)
    out = TesseractMrzEngine().extract([buf.getvalue()], None)
    assert out.mrz_lines and parse_mrz(out.mrz_lines, TODAY).document_number == "L898902C3"
