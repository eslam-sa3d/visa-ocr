"""Validation rules applied on top of parsed MRZ data."""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date

from .mrz import MrzResult, check_digit


@dataclass
class Flag:
    code: str
    severity: str  # "error" | "warning"
    message: str

    def to_dict(self) -> dict:
        return {"code": self.code, "severity": self.severity, "message": self.message}


def validate(r: MrzResult, today: date | None = None) -> list[Flag]:
    today = today or date.today()
    flags: list[Flag] = []

    for name, ok in r.checks.items():
        if not ok:
            flags.append(Flag("mrz_checksum_failed", "error", f"Check digit failed: {name}"))

    # The name line has no check digit. When letters were removed from it as "misread fillers", or no surname is left, nothing
    # proves the result, so a person confirms the name.
    removed = [c for c in r.corrected_fields if "filler read as letters removed" in c]
    if removed:
        flags.append(Flag("mrz_name_letters_removed", "error",
                          "Letters at the end of the name line were treated as misread fillers and removed ("
                          + ", ".join(removed) + "). Confirm the name against the document."))
    if not (r.surname or "").strip():
        flags.append(Flag("mrz_surname_missing", "error", "No surname could be read from the MRZ. Check the name."))

    if r.corrected_fields:
        flags.append(Flag("ocr_digits_corrected", "warning",
                          "Look-alike characters were corrected in numeric positions: "
                          + ", ".join(r.corrected_fields)))

    if r.repaired_fields:
        flags.append(Flag("mrz_field_repaired", "error",
                          "The check digit failed and the field was repaired by swapping look-alike characters "
                          "(" + ", ".join(r.repaired_fields) + "). A person must confirm it against the document."))

    # A passport whose first letter was read as V is parsed as a visa (same 2 x 44 shape). A visa has no check digits for
    # its optional data, so test the passport check digits: if both pass on a "visa", it is probably a passport.
    if r.format == "MRVA" and len(r.lines) == 2:
        l2 = r.lines[1]
        if l2[42:44].isdigit() and check_digit(l2[0:10] + l2[13:20] + l2[21:43]) == int(l2[43]) \
                and (check_digit(l2[28:42]) == int(l2[42]) or set(l2[28:42]) <= {"<"}):
            flags.append(Flag("type_code_inconsistent", "error",
                              "The type code says visa, but the second line passes the passport check digits. "
                              "The first letter may have been misread (P read as V). Check the document type."))

    if r.birth_date is None:
        flags.append(Flag("birth_date_invalid", "error", "Date of birth is not a valid date"))
    elif date.fromisoformat(r.birth_date) > today:
        flags.append(Flag("birth_date_in_future", "error", "Date of birth is in the future"))
    else:
        age = (today - date.fromisoformat(r.birth_date)).days / 365.25
        if age >= 95 or age < 1:   # the MRZ has a 2-digit year, so 1926 and 2026 look the same
            flags.append(Flag("birth_century_check", "warning",
                              f"Date of birth gives an age of {age:.0f}; the MRZ year has only two digits, check the century"))

    if r.expiry_date is None:
        flags.append(Flag("expiry_date_invalid", "error", "Expiry date is not a valid date"))
    elif date.fromisoformat(r.expiry_date) < today:
        flags.append(Flag("document_expired", "error", f"Document expired on {r.expiry_date}"))

    if r.sex not in ("M", "F", "<", "X"):
        flags.append(Flag("sex_invalid", "error", f"Unexpected sex marker: {r.sex!r}"))

    code_re = re.compile(r"[A-Z]{1,3}")
    if not code_re.fullmatch(r.issuing_state):
        flags.append(Flag("issuing_state_invalid", "error", "Issuing state code is not 1-3 letters"))
    if not code_re.fullmatch(r.nationality):
        flags.append(Flag("nationality_invalid", "error", "Nationality code is not 1-3 letters"))

    from .countries import country_name
    for label, code in (("Issuing state", r.issuing_state), ("Nationality", r.nationality)):
        if code and code_re.fullmatch(code) and not code.startswith("XX") and not country_name(code):
            flags.append(Flag(f"{label.split()[0].lower()}_code_unknown", "warning",
                              f"{label} code {code!r} is not a known country code. The MRZ may be misread or malformed."))

    if not r.document_number:
        flags.append(Flag("document_number_missing", "error", "Document number is empty"))

    return flags


def _norm(s: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", s.upper())


def cross_match(a: MrzResult, b: MrzResult, label_a: str = "A", label_b: str = "B") -> list[Flag]:
    """Compare holder identity fields between two documents (e.g. passport and visa)."""
    flags: list[Flag] = []
    pairs = [
        ("surname", _norm(a.surname), _norm(b.surname)),
        ("given_names", _norm(a.given_names), _norm(b.given_names)),
        ("birth_date", a.birth_date, b.birth_date),
        ("sex", a.sex, b.sex),
    ]
    # XXX / XXA / XXB / XXC mean "unspecified" on some documents, so skip those.
    if not (a.nationality.startswith("XX") or b.nationality.startswith("XX")):
        pairs.append(("nationality", a.nationality, b.nationality))
    for name, va, vb in pairs:
        if va != vb:
            flags.append(Flag("cross_document_mismatch", "error",
                              f"{name} differs between {label_a} and {label_b}"))
    return flags


def check_expected_nationality(r: MrzResult, expected: str | None) -> list[Flag]:
    """Compare the nationality in the MRZ with the one the applicant declared (3-letter ISO code)."""
    if not expected:
        return []
    code = expected.strip().upper()
    if not re.fullmatch(r"[A-Z]{3}", code):
        return [Flag("expected_nationality_invalid", "warning",
                     "Declared nationality must be a 3-letter ISO code such as SAU or EGY; check skipped")]
    if r.nationality == code:
        return []
    severity = "warning" if r.nationality.startswith("XX") else "error"
    return [Flag("nationality_mismatch", severity,
                 f"Declared nationality {code} does not match the document ({r.nationality})")]


def visual_name_status(surname: str, given: str, text_lines: list[dict], name_field_full: bool = False) -> tuple[str, str]:
    """Compare the MRZ name with the name printed on the page. Returns (status, printed_text) where status is
    'match' (identical, or a real truncation: the MRZ name field is full and the printed name starts with the MRZ name),
    'mismatch' (a near miss, including one or two letters added or lost at the end: one side was probably misread) or
    'none' (nothing comparable was read from the page)."""
    from difflib import SequenceMatcher

    def letters(t: str) -> str:
        return re.sub(r"[^A-Z]", "", t.upper())

    mrz_name = letters(surname + given)
    if len(mrz_name) < 4:
        return "none", ""
    best, best_text, cut_text = 0.0, "", ""
    for ln in text_lines or []:
        text = str(ln.get("text", ""))
        if not re.search(r"[A-Za-z]{2}", text) or re.search(r"[<0-9]", text):
            continue
        v = letters(text)
        if len(v) < 4:
            continue
        if v == mrz_name:
            return "match", text.strip()
        if name_field_full and v.startswith(mrz_name):      # only the MRZ side is ever cut to fit
            cut_text = cut_text or text.strip()
            continue
        if abs(len(v) - len(mrz_name)) > 2:
            continue
        ratio = SequenceMatcher(None, mrz_name, v).ratio()
        if ratio > best:
            best, best_text = ratio, text.strip()
    if cut_text:
        return "match", cut_text
    if best >= 0.75:
        return "mismatch", best_text
    return "none", ""


def check_name_against_visual(surname: str, given: str, text_lines: list[dict]) -> list[Flag]:
    """The MRZ name line has no check digit. Flag a name that nearly matches the printed one but is not identical."""
    status, printed = visual_name_status(surname, given, text_lines)
    if status != "mismatch":
        return []
    return [Flag("name_differs_from_visual_zone", "error",
                 f"MRZ name '{surname} {given}' differs from the name printed on the page ('{printed}'). "
                 "One of them was probably misread. Check the document.")]
