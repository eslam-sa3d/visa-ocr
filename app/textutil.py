"""Small text helpers for mixed Arabic / English document text."""
from __future__ import annotations

import re

_ARABIC_INDIC = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
_EXT_ARABIC_INDIC = str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789")
_ARABIC_RE = re.compile("[؀-ۿݐ-ݿࢠ-ࣿﭐ-﷿ﹰ-﻿]")
_LATIN_RE = re.compile("[A-Za-z]")
_DIACRITICS = re.compile("[ً-ٰٟـ]")  # harakat, dagger alef, tatweel
_ALEF_FORMS = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا",
                             "ى": "ي"})


def to_ascii_digits(text: str) -> str:
    """Arabic-Indic and Persian digits to 0-9 (needed for dates and ID numbers)."""
    return text.translate(_ARABIC_INDIC).translate(_EXT_ARABIC_INDIC)


def has_arabic(text: str) -> bool:
    return bool(_ARABIC_RE.search(text))


def script_of(text: str) -> str:
    """'arabic', 'latin', 'mixed' or 'other' (digits and punctuation only)."""
    ar, lat = has_arabic(text), bool(_LATIN_RE.search(text))
    if ar and lat:
        return "mixed"
    if ar:
        return "arabic"
    if lat:
        return "latin"
    return "other"


def normalize_arabic(text: str) -> str:
    """Comparison-only normalisation: drop diacritics and tatweel, unify alef and yeh forms."""
    return _DIACRITICS.sub("", text).translate(_ALEF_FORMS)


def _same_box(a: list, b: list) -> bool:
    """True when two text polygons describe the same region (same detector run read it twice).
    Compares corners, not bounding rectangles: tilted neighbouring lines have overlapping rectangles."""
    if len(a) < 4 or len(b) < 4:
        return False
    h = min(abs(a[3][1] - a[0][1]) + abs(a[3][0] - a[0][0]), abs(b[3][1] - b[0][1]) + abs(b[3][0] - b[0][0])) or 1
    dist = sum(((a[k][0] - b[k][0]) ** 2 + (a[k][1] - b[k][1]) ** 2) ** 0.5 for k in range(4)) / 4
    return dist <= 0.4 * h


def dedupe_overlapping(lines: list[dict]) -> list[dict]:
    """When several recognisers read the same box, keep only the highest-scoring reading (original order kept)."""
    kept: list[dict] = []
    for ln in sorted(lines, key=lambda l: -float(l.get("score") or 0)):
        if not any(_same_box(ln.get("box") or [], k.get("box") or []) for k in kept):
            kept.append(ln)
    return [l for l in lines if any(l is k for k in kept)]


HIJRI_MARGIN_DAYS = 3


def hijri_to_gregorian(year: int, month: int, day: int):
    """Approximate Gregorian date for a Hijri date, by the arithmetic (tabular) calendar. Saudi documents follow the Umm
    al-Qura calendar, which is set by astronomical calculation and can differ from this by a day or two, so use the result
    only with HIJRI_MARGIN_DAYS of doubt on either side. Returns None for a date that cannot exist."""
    from datetime import date
    if not (1 <= month <= 12 and 1 <= day <= 30 and 1 <= year <= 2000):
        return None
    jdn = (11 * year + 3) // 30 + 354 * year + 30 * month - (month - 1) // 2 + day + 1948440 - 385
    try:
        return date.fromordinal(jdn - 1721425)
    except (ValueError, OverflowError):
        return None
