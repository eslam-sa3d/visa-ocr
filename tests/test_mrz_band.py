"""The MRZ strip locator, filler restoration and filler-tolerant agreement (added after the real-scan review, 11 Nov 2026)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
pytest.importorskip("cv2")
from PIL import Image  # noqa: E402

import make_scenarios as M  # noqa: E402
from app.engines.mrz_band import find_bands, text_rows  # noqa: E402

LINES = ["P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<", "L898902C36UTO7408122F1204159ZE184226B<<<<<10"]


def _page_on_sheet():
    """A passport page placed small on a large white sheet, like a flatbed scan."""
    page = M.draw_document({"lines": LINES, "label": "TD3 TEST"}).convert("L")
    page = page.resize((700, round(page.height * 700 / page.width)))
    sheet = Image.new("L", (1654, 2339), 245)
    sheet.paste(page, (400, 900))
    return sheet, (400, 900, page.width, page.height)


def test_strip_is_found_on_a_large_sheet_and_holds_two_rows():
    sheet, (x, y, w, h) = _page_on_sheet()
    bands = find_bands(sheet)
    assert bands and bands[0].quarter == 0
    b = bands[0]
    assert b.image.width > 4 * b.image.height                       # a strip, not a block of the page
    assert x - 40 <= b.box[0] and b.box[2] <= x + w + 40            # inside the passport page
    assert b.box[1] > y + h * 0.6                                   # in its lower part, where the MRZ is
    assert len(text_rows(b.image)) == 2
    assert b.rows and len(b.rows) == 2 and all(r[2] - r[0] > 400 for r in b.rows)


def test_strip_is_found_when_the_sheet_is_sideways():
    sheet, _ = _page_on_sheet()
    bands = find_bands(sheet.rotate(90, expand=True))
    assert bands and bands[0].quarter == 1 and bands[0].image.width > 4 * bands[0].image.height
    assert bands[0].rows is None                                    # row boxes are only reported for an upright page


def test_blank_and_plain_text_pages_give_no_strip():
    assert find_bands(Image.new("L", (1200, 900), 255)) == []


def test_merged_fillers_are_restored_by_the_check_digits():
    from app.engines.rapid_ocr import restore_fillers
    full = "S006857<<6UTO8106034M2408231<<<<<<<<<<<<<<06"
    from app.mrz import check_digit
    number, birth, expiry, opt = "S006857<<", "810603", "240823", "<" * 14
    body = number + str(check_digit(number)) + "UTO" + birth + str(check_digit(birth)) + "M" + expiry + str(check_digit(expiry)) + opt + str(check_digit(opt))
    full = body + str(check_digit(body[0:10] + body[13:20] + body[21:43]))
    merged = full.replace("S006857<<", "S006857<", 1)               # what a reader that merges repeated fillers returns
    assert len(merged) == 43 and restore_fillers(merged) == full
    assert restore_fillers("HELLO<WORLD") == "HELLO<WORLD"          # not an MRZ line: untouched
    assert restore_fillers(full) == full


def test_readings_that_differ_only_in_filler_counts_agree():
    from app.engines.ensemble import _squash
    a = ["P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<", LINES[1]]
    b = ["P<UTOERIKSSON<ANNA<MARIA<<<<<<<<<<<<<<<<<<<<", LINES[1]]
    c = ["P<UTOERIKSSON<<ANNE<MARIA<<<<<<<<<<<<<<<<<<<", LINES[1]]
    assert _squash(a) == _squash(b) and _squash(a) != _squash(c)


def test_ensemble_accepts_filler_count_difference_and_keeps_the_valid_line(monkeypatch):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from test_ensemble_and_tools import _ensemble, _td3_with_optional
    from app import pipeline
    from app.review import Applicant
    good = _td3_with_optional("ZE184226B")
    merged = [good[0].replace("<<ANNA", "<ANNA") + "<", good[1]]
    status, res = pipeline.run([b""], None, Applicant(), _ensemble(good, merged))
    assert status == "done" and res["needsReview"] is False
    assert res["document"]["surname"] == "ERIKSSON" and res["document"]["givenNames"] == "ANNA MARIA"
    changed = [good[0].replace("ANNA", "ANNE"), good[1]]
    assert pipeline.run([b""], None, Applicant(), _ensemble(good, changed))[1]["needsReview"] is True
