from datetime import date

import pytest

from app.mrz import MrzError, check_digit, find_mrz_lines, parse_mrz
from app.validation import cross_match, validate

# ICAO 9303 specimen documents (holder: Anna Maria Eriksson, Utopia).
TD3 = [
    "P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<",
    "L898902C36UTO7408122F1204159ZE184226B<<<<<10",
]
TD2 = [
    "I<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<",
    "D231458907UTO7408122F1204159<<<<<<<6",
]
TD1 = [
    "I<UTOD231458907<<<<<<<<<<<<<<<",
    "7408122F1204159UTO<<<<<<<<<<<6",
    "ERIKSSON<<ANNA<MARIA<<<<<<<<<<",
]
MRVA = [
    "V<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<",
    "L8988901C4XXX4009078F96121096ZE184226B<<<<<<",
]
EARLY = date(2010, 1, 1)  # specimen documents expire in 2012


def test_check_digit_icao_example():
    assert check_digit("520727") == 3


def test_td3_passport():
    r = parse_mrz(TD3, today=EARLY)
    assert r.format == "TD3" and r.kind == "passport"
    assert r.issuing_state == "UTO" and r.nationality == "UTO"
    assert r.surname == "ERIKSSON" and r.given_names == "ANNA MARIA"
    assert r.document_number == "L898902C3"
    assert r.birth_date == "1974-08-12" and r.expiry_date == "2012-04-15"
    assert r.sex == "F" and r.optional_data == "ZE184226B"
    assert r.valid, r.checks


def test_td2_card():
    r = parse_mrz(TD2, today=EARLY)
    assert r.format == "TD2" and r.kind == "id_card"
    assert r.document_number == "D23145890"
    assert r.valid, r.checks


def test_td1_card_and_green_card_layout():
    r = parse_mrz(TD1, today=EARLY)
    assert r.format == "TD1"
    assert r.surname == "ERIKSSON" and r.given_names == "ANNA MARIA"
    assert r.birth_date == "1974-08-12" and r.expiry_date == "2012-04-15"
    assert r.valid, r.checks


def test_schengen_visa_mrva():
    r = parse_mrz(MRVA, today=EARLY)
    assert r.format == "MRVA" and r.kind == "visa"
    assert r.document_number == "L8988901C"
    assert r.nationality == "XXX"
    assert r.birth_date == "1940-09-07" and r.expiry_date == "2096-12-10"
    assert r.valid, r.checks


def test_tampered_document_number_fails_checksum():
    bad = [TD3[0], "L898903C36" + TD3[1][10:]]
    r = parse_mrz(bad, today=EARLY)
    assert not r.valid and not r.checks["document_number"]
    codes = {f.code for f in validate(r, today=EARLY)}
    assert "mrz_checksum_failed" in codes


def test_ocr_letter_for_digit_is_corrected():
    # "740812" read as "74O812" (letter O instead of zero) in a numeric field.
    noisy = [TD3[0], TD3[1].replace("7408122", "74O8122")]
    r = parse_mrz(noisy, today=EARLY)
    assert r.valid and r.birth_date == "1974-08-12"
    assert r.corrected_fields
    assert "ocr_digits_corrected" in {f.code for f in validate(r, today=EARLY)}


def test_expired_document_flagged():
    r = parse_mrz(TD3, today=date(2026, 11, 9))
    assert "document_expired" in {f.code for f in validate(r, today=date(2026, 11, 9))}


def test_whitespace_and_case_are_tolerated():
    r = parse_mrz([l.lower().replace("<", " < ") for l in TD3], today=EARLY)
    assert r.valid


def test_unsupported_shape():
    with pytest.raises(MrzError):
        parse_mrz(["ABC", "DEF"])


def test_find_mrz_in_noisy_ocr_output():
    noise = ["PASSPORT", "SURNAME ERIKSSON", *TD3, "random trailing text"]
    assert find_mrz_lines(noise) == TD3


def test_find_mrz_returns_none_when_absent():
    assert find_mrz_lines(["HELLO", "WORLD"]) is None


def test_cross_match_passport_and_visa_same_holder():
    p, v = parse_mrz(TD3, today=EARLY), parse_mrz(MRVA, today=EARLY)
    flags = cross_match(p, v, "passport", "visa")
    # Same name and sex; the visa specimen has a different birth date, so it must be flagged.
    assert [f.message for f in flags] == ["birth_date differs between passport and visa"]


def test_find_mrz_lines_repairs_dropped_and_extra_fillers():
    from app.mrz import find_mrz_lines
    l1 = "P<USASPECIMEN<<JANE<TEST<<<<<<<<<<<<<<<<<<<<"
    l2 = "1234567897USA9003152F3106143<<<<<<<<<<<<<<02"
    short = find_mrz_lines([l1[:-2], l2])           # OCR dropped two '<'
    assert short == [l1, l2]
    long_ = find_mrz_lines(["noise", l1 + "<", l2[:30] + "<" + l2[30:]])  # extra '<'
    assert long_ == [l1, l2]
    assert find_mrz_lines([l1[:30], l2]) == [l1, l2]    # line 1 truncated, line 2 intact: filler restored (line 1 has no check digit)
    assert find_mrz_lines([l1[:10], l2]) is None        # too little of line 1 left to trust
    assert find_mrz_lines(["XXXXXXXXXXXXXXXXXXXXXXXX<<<", l2]) is None   # does not start like an MRZ


def test_unknown_country_code_is_flagged():
    from app.mrz import parse_mrz
    from app.validation import validate
    lines = ["P<DELAPAZ<<MICHELLE<<<<<<<<<<<<<<<<<<<<<<<<<", "9102392482USA6401171F1812051900781200<129676"]
    codes = [f.code for f in validate(parse_mrz(lines))]
    assert "issuing_code_unknown" in codes
