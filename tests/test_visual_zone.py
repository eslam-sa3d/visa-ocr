from datetime import date

from fastapi.testclient import TestClient

import app.main as main
from app.engines.base import EngineOutput
from app.visual_zone import parse_date, parse_gcc_id, saudi_id_check

TODAY = date(2026, 11, 9)


def make_valid_id(prefix9: str) -> str:
    for c in range(10):
        if saudi_id_check(prefix9 + str(c)):
            return prefix9 + str(c)
    raise AssertionError


IQAMA = make_valid_id("234567890"[:9])


def L(text, y, x=0, score=0.95):
    return {"text": text, "score": score, "box": [[x, y], [x + 100, y], [x + 100, y + 20], [x, y + 20]]}


def test_check_digit():
    assert saudi_id_check(IQAMA)
    assert not saudi_id_check(IQAMA[:-1] + str((int(IQAMA[-1]) + 1) % 10))
    assert not saudi_id_check("12345")


def test_parse_dates_both_calendars():
    assert parse_date("2027/05/14") == ("2027-05-14", "gregorian")
    assert parse_date("٠٣/٠٢/١٤٤٨") == ("1448-02-03", "hijri")
    assert parse_date("not a date") is None


def test_iqama_english_labels():
    lines = [L("Iqama No: " + IQAMA, 10), L("Name: ALI HASSAN KHAN", 40), L("Nationality: PAK", 70),
             L("Occupation: ENGINEER", 100), L("Date of Birth: 1990/03/12", 130), L("Expiry Date: 2027/05/14", 160)]
    r = parse_gcc_id(lines, TODAY)
    d = r["document"]
    assert d["kind"] == "iqama" and d["idNumber"] == IQAMA and d["valid"]
    assert d["nameEn"] == "ALI HASSAN KHAN" and d["nationality"] == "PAK" and d["nationalityName"] == "Pakistan"
    assert d["birthDate"] == "1990-03-12" and d["expiryDate"] == "2027-05-14" and d["expiryCalendar"] == "gregorian"
    codes = [f["code"] for f in r["flags"]]
    assert "document_expired" not in codes and "visual_zone_only" in codes


def test_iqama_arabic_labels_and_arabic_indic_digits():
    ar_digits = IQAMA.translate(str.maketrans("0123456789", "٠١٢٣٤٥٦٧٨٩"))
    lines = [L("رقم الإقامة: " + ar_digits, 10), L("الاسم: علي حسن خان", 40), L("المهنة: مهندس", 70),
             L("تاريخ الميلاد: ١٩٩٠/٠٣/١٢", 100), L("تاريخ الانتهاء: ٢٠٢٥/٠١/٠١", 130)]
    r = parse_gcc_id(lines, TODAY)
    d = r["document"]
    assert d["idNumber"] == IQAMA and d["nameAr"] == "علي حسن خان" and d["occupation"] == "مهندس"
    assert d["expiryDate"] == "2025-01-01"
    assert "document_expired" in [f["code"] for f in r["flags"]] and r["needsReview"]


def test_value_on_next_line_and_misread_digit_flagged():
    wrong = IQAMA[:-1] + str((int(IQAMA[-1]) + 3) % 10)
    r = parse_gcc_id([L("Iqama No", 10), L(wrong, 40), L("Date of Birth", 70), L("12/03/1990", 100)], TODAY)
    assert r["document"]["idNumber"] == wrong and r["document"]["birthDate"] == "1990-03-12"
    assert "id_number_checksum_failed" in [f["code"] for f in r["flags"]]


def test_missing_id_number_needs_review():
    r = parse_gcc_id([L("Nationality: EGY", 10)], TODAY)
    assert r["needsReview"] and "id_number_not_found" in [f["code"] for f in r["flags"]]


def test_low_confidence_flagged():
    r = parse_gcc_id([L("Iqama No: " + IQAMA, 10, score=0.5)], TODAY)
    assert "low_confidence_fields" in [f["code"] for f in r["flags"]]


class FakeIdEngine:
    name = "fake-paddle"

    def extract(self, images, hint):
        return EngineOutput(engine=self.name, mrz_lines=[],
                            fields={"textLines": [L("Iqama No: " + IQAMA, 10), L("Name: ALI HASSAN KHAN", 40)]})


class NoTextEngine:
    name = "tess"

    def extract(self, images, hint):
        return EngineOutput(engine=self.name, mrz_lines=[])


def _upload(client, hint):
    from PIL import Image
    import io
    buf = io.BytesIO(); Image.new("RGB", (200, 120), "white").save(buf, format="PNG")
    r = client.post("/v1/documents/extract", files=[("files", ("a.png", buf.getvalue(), "image/png"))],
                    data={"doc_type_hint": hint})
    return client.get(f"/v1/jobs/{r.json()['jobId']}").json()


def test_job_uses_visual_zone_when_no_mrz(monkeypatch):
    monkeypatch.setattr(main, "get_engine", lambda: FakeIdEngine())
    job = _upload(TestClient(main.app), "iqama")
    assert job["status"] == "done" and job["result"]["document"]["idNumber"] == IQAMA


def test_job_explains_when_engine_cannot_read_iqama(monkeypatch):
    monkeypatch.setattr(main, "get_engine", lambda: NoTextEngine())
    job = _upload(TestClient(main.app), "iqama")
    assert job["status"] == "failed" and "paddle-ocr" in job["error"]["message"]


def test_dedupe_keeps_best_reading_per_box():
    from app.textutil import dedupe_overlapping
    box = [[0, 0], [100, 0], [100, 20], [0, 20]]
    lines = [{"text": "junk", "score": 0.5, "box": box}, {"text": "good", "score": 0.9, "box": box},
             {"text": "other", "score": 0.4, "box": [[0, 50], [100, 50], [100, 70], [0, 70]]}]
    assert [l["text"] for l in dedupe_overlapping(lines)] == ["good", "other"]


def test_dates_with_slashes_misread_as_digit_one():
    assert parse_date("٢٠٢٧١٠٤١٢٢") == ("2027-04-22", "gregorian")
    assert parse_date("٢٢١٠٤١٢٠٢٧") == ("2027-04-22", "gregorian")


def test_name_cross_check_flags_misread_mrz_name():
    from app.validation import check_name_against_visual as chk
    lines = [{"text": "XUAN, JIANG"}, {"text": "PEOPLE'S REPUBLIC OF CHINA"}]
    flags = chk("XUAN", "ZANG", lines)
    assert [f.code for f in flags] == ["name_differs_from_visual_zone"] and "XUAN, JIANG" in flags[0].message
    assert chk("XUAN", "JIANG", lines) == []                    # identical: no flag
    assert chk("XUAN", "JIANG", [{"text": "NATIONALITY CHINESE"}]) == []   # nothing comparable on the page


def test_truncated_names_are_not_a_mismatch():
    from app.validation import check_name_against_visual as chk
    assert chk("SHOEIB", "OMAR SOBHY IBRAHIM IBRA", [{"text": "SHOEIB, OMAR SOBHY IBRAHIM IB"}]) == []
