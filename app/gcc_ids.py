"""ID-number rules for GCC identity cards other than the Saudi Iqama / national ID.

STATUS: written from publicly described number formats, NOT checked against real cards. Every result built on these
rules is returned for human review (flag `gcc_layout_unverified`). Use them to recognise the card type and to catch
misread digits, never to approve a card.

    UAE Emirates ID   15 digits, 784-YYYY-NNNNNNN-C, Luhn check over all 15 digits          (strong)
    Kuwait civil ID   12 digits, century digit 2 or 3, YYMMDD birth date, serial, check digit (date strong, check weak)
    Qatar ID (QID)    11 digits, century digit 2 or 3, YY of birth, nationality code, serial     (structure only)
    Bahrain CPR        9 digits, YYMM of birth, serial, check digit                                (structure only)
    Oman civil number  no public structure: not recognised from the number
"""
from __future__ import annotations

from datetime import date

_KUWAIT_WEIGHTS = (2, 1, 6, 3, 7, 9, 10, 5, 8, 4, 2)


def luhn_ok(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def kuwait_check_digit(first11: str) -> int:
    return 11 - (sum(int(c) * w for c, w in zip(first11, _KUWAIT_WEIGHTS)) % 11)


def _valid_date(y: int, m: int, d: int) -> bool:
    try:
        date(y, m, d)
        return True
    except ValueError:
        return False


def classify(digits: str) -> dict | None:
    """Recognise a GCC ID number from its digits alone. Returns {kind, country, valid, birthDate, note} or None."""
    if not digits.isdigit():
        return None
    n = len(digits)
    if n == 15 and digits.startswith("784"):
        return {"kind": "uae_emirates_id", "country": "ARE", "valid": luhn_ok(digits), "birthDate": None,
                "note": "Luhn check over 15 digits"}
    if n == 12 and digits[0] in "23":
        century = 1900 if digits[0] == "2" else 2000
        yy, mm, dd = int(digits[1:3]), int(digits[3:5]), int(digits[5:7])
        if not _valid_date(century + yy, mm, dd):
            return None
        check = kuwait_check_digit(digits[:11])
        return {"kind": "kuwait_civil_id", "country": "KWT", "valid": check == int(digits[11]) or (check == 11 and digits[11] == "0"),
                "birthDate": f"{century + yy:04d}-{mm:02d}-{dd:02d}", "note": "birth date inside the number; check digit weak"}
    if n == 11 and digits[0] in "23":
        return {"kind": "qatar_id", "country": "QAT", "valid": True, "birthDate": None, "note": "structure only"}
    if n == 9 and 1 <= int(digits[2:4]) <= 12:
        return {"kind": "bahrain_cpr", "country": "BHR", "valid": True, "birthDate": None, "note": "structure only"}
    return None
