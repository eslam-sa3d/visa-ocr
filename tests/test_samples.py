import json
from pathlib import Path

import pytest

from app.mrz import parse_mrz
from app.validation import validate

SAMPLES = json.loads((Path(__file__).resolve().parent.parent / "samples" / "mrz_samples.json").read_text())


@pytest.mark.parametrize("sample", SAMPLES, ids=[s["id"] for s in SAMPLES])
def test_sample_parses_valid_and_matches_expected(sample):
    parsed = parse_mrz(sample["lines"])
    assert parsed.valid, parsed.checks
    assert not validate(parsed)
    assert parsed.to_dict() == sample["expected"]


def test_green_card_long_receipt_number_is_reassembled():
    s = next(x for x in SAMPLES if x["id"] == "greencard_usa")
    assert s["expected"]["documentNumber"] == "SRC0000000001"


def test_german_issuer_code_is_single_letter():
    s = next(x for x in SAMPLES if x["id"] == "passport_deu")
    assert s["expected"]["issuingState"] == "D"
    assert s["expected"]["surname"] == "SPECIMEN MUELLER"
