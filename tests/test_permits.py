"""Fronts of residence cards (no MRZ on that side): read from printed labels, always sent to a person."""
from datetime import date

from app import pipeline
from app.engines.base import EngineOutput
from app.permits import kind_of, parse_date, parse_permit
from app.review import Applicant

TODAY = date(2026, 11, 22)


def L(text, x0, y0, x1, y1, score=0.99):
    return {"text": text, "score": score, "box": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]], "image": 0}


US = [L("UNITED STATES OF AMERICA", 16, 14, 307, 33), L("PERMANENT RESIDENT", 16, 27, 308, 59), L("Surname", 235, 80, 297, 99),
      L("SAMPLE", 235, 98, 339, 118), L("Given Name", 243, 121, 330, 140), L("JANE", 248, 142, 311, 159),
      L("Country of Birth", 249, 164, 352, 181), L("Utopia", 245, 183, 312, 201), L("USCIS#", 265, 206, 325, 222),
      L("Category", 400, 204, 469, 226), L("000-111-222", 269, 222, 381, 242), L("IR1", 403, 223, 434, 242),
      L("Date of Birth", 259, 242, 354, 261), L("Sex", 403, 246, 433, 265), L("20 OCT 1990", 268, 262, 386, 286),
      L("F", 404, 267, 418, 282), L("Card Expires", 267, 285, 358, 304), L("Resident Since:", 377, 290, 479, 304),
      L("10/26/32", 269, 305, 349, 323), L("10/25/20", 378, 305, 457, 323)]
DE = [L("XXZ1AB2CD", 1429, 35, 1926, 115), L("AUFENTHALTSTITEL", 510, 56, 1156, 129), L("XXZ1AB2CD", 60, 184, 323, 229),
      L("NAMEN Vornamen/SURNAMES Forenames", 800, 257, 1491, 297), L("MUSTERFRAU", 816, 306, 1127, 360), L("Erika", 815, 380, 1013, 438),
      L("STAATSANGEHORIGKEIT/", 1080, 513, 1481, 553), L("GEBURTSDATUM/", 1665, 513, 1949, 550), L("GESCHLECHT/", 804, 519, 1024, 554),
      L("NATIONALITY", 1081, 550, 1312, 587), L("SEX", 804, 554, 873, 588), L("DATE OF BIRTH", 1683, 552, 1948, 583),
      L("F", 817, 596, 856, 644), L("UTO", 1092, 593, 1216, 648), L("01 06 1990", 1546, 589, 1943, 646),
      L("ART DES TITELS/TYPE OF PERMIT", 805, 649, 1345, 686), L("KARTE GÜLTIG BIS/CARD EXPIRY", 1408, 649, 1942, 682),
      L("AUFENTHALTSERLAUBNIS", 816, 694, 1404, 741), L("1412", 1551, 684, 1764, 741), L("2030", 1780, 685, 1942, 740),
      L("RESIDENCE PERMIT", 822, 1183, 1158, 1217)]
UK = [L("RESIDENCE PERMIT", 164, 3, 333, 19), L("ZZ1234567", 379, 7, 453, 20), L("DOE", 166, 50, 226, 64), L("JANE", 165, 63, 242, 79),
      L("VALID UNTIL", 167, 81, 211, 91), L("05-10-2030", 165, 91, 238, 104), L("PLACE AND DATE OF ISSUE", 166, 108, 258, 118),
      L("UK 14-02-2025", 166, 119, 262, 132), L("TYPE OF PERMIT", 166, 135, 224, 145), L("T4 GENERAL STUDENT", 167, 145, 323, 158),
      L("LEAVE TO REMAIN", 164, 156, 301, 174)]
OM = [L("SULTANATE OF OMAN", 410, 745, 773, 781), L("RESIDENT", 396, 856, 545, 888), L("CARD", 397, 892, 484, 925),
      L("CIVIL NUMBER", 369, 979, 532, 1009), L("الرقم المدني", 874, 974, 1006, 1028), L("12345678", 608, 985, 817, 1029),
      L("EXPIRY DATE", 367, 1017, 517, 1047), L("22/10/2030", 631, 1027, 801, 1065), L("DATE OF BIRTH", 368, 1057, 540, 1084),
      L("08/03/1985", 633, 1066, 801, 1104)]


def test_us_permanent_resident_card():
    d = parse_permit(US, TODAY)["document"]
    assert d["kind"] == "us_permanent_resident_card" and d["issuingCountry"] == "USA"
    assert (d["documentNumber"], d["surname"], d["givenNames"], d["sex"]) == ("000-111-222", "SAMPLE", "JANE", "F")
    assert d["birthDate"] == "1990-10-20" and d["expiryDate"] == "2032-10-26"          # US numeric dates are month first


def test_eu_residence_permit_with_shared_label_lines_and_a_split_date():
    d = parse_permit(DE, TODAY)["document"]
    assert d["kind"] == "eu_residence_permit" and d["documentNumber"] == "XXZ1AB2CD"
    assert (d["surname"], d["givenNames"], d["sex"], d["nationality"]) == ("MUSTERFRAU", "ERIKA", "F", "UTO")
    assert d["birthDate"] == "1990-06-01" and d["expiryDate"] == "2030-12-14" and d["permitType"] == "AUFENTHALTSERLAUBNIS"


def test_uk_residence_permit_name_has_no_label():
    d = parse_permit(UK, TODAY)["document"]
    assert d["kind"] == "uk_residence_permit" and d["issuingCountry"] == "GBR" and d["documentNumber"] == "ZZ1234567"
    assert (d["surname"], d["givenNames"], d["expiryDate"], d["permitType"]) == ("DOE", "JANE", "2030-10-05", "T4 GENERAL STUDENT")


def test_oman_card():
    d = parse_permit(OM, TODAY)["document"]
    assert d["kind"] == "oman_resident_card" and d["idNumber"] == "12345678" and d["issuingCountry"] == "OMN"
    assert d["birthDate"] == "1985-03-08" and d["expiryDate"] == "2030-10-22"


def test_every_front_side_reading_goes_to_a_person():
    for lines in (US, DE, UK, OM):
        r = parse_permit(lines, TODAY)
        assert r["needsReview"] is True and r["document"]["checks"] == {}
        assert any(f["code"] == "visual_zone_only" and f["severity"] == "error" for f in r["flags"])


def test_other_text_is_not_a_permit():
    assert parse_permit([L("INVOICE", 0, 0, 100, 20), L("Date of birth", 0, 30, 100, 50)]) is None
    assert kind_of([L("KINGDOM OF SAUDI ARABIA", 0, 0, 200, 20)]) is None


def test_dates():
    assert parse_date("20 OCT 2002") == "2002-10-20" and parse_date("01.06.2000") == "2000-06-01"
    assert parse_date("10/26/32", month_first=True) == "2032-10-26" and parse_date("1412 2024") == "2024-12-14"
    assert parse_date("31 02 2020") is None and parse_date("no date") is None
    assert parse_date("05/07/85", month_first=True, born=True, today=TODAY) == "1985-05-07"


class _Reader:
    name = "fake"

    def __init__(self, lines):
        self.lines = lines

    def extract(self, images, hint):
        return EngineOutput(engine="paddle-ocr", mrz_lines=None, fields={"textLines": self.lines}, confidence=0.95)


def test_pipeline_reads_a_permit_front_and_sends_it_to_review():
    status, res = pipeline.run([b"x"], None, Applicant(), _Reader(DE))
    assert status == "done" and res["document"]["kind"] == "eu_residence_permit" and res["needsReview"] is True
    assert "retake" not in res


def test_saudi_card_with_english_residence_permit_title_stays_with_the_gulf_reader():
    lines = [L("KINGDOM OF SAUDI ARABIA", 0, 0, 300, 20), L("RESIDENCE PERMIT", 0, 30, 300, 50), L("2345678901", 0, 60, 200, 80),
             L("Date of birth", 0, 90, 120, 110), L("01/01/1990", 130, 90, 260, 110)]
    status, res = pipeline.run([b"x"], None, Applicant(), _Reader(lines))
    assert not (status == "done" and str(res.get("document", {}).get("kind", "")).endswith("residence_permit"))
