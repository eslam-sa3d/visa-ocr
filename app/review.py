"""When must a person look at the result? Rules for fields that no check digit protects.

MRZ check digits cover the document number, birth date, expiry date (and a composite). Names, nationality, sex and
issuing state are NOT covered, so a misread there is invisible to the parser. They are accepted without review only when
(a) the applicant's own form data matches, (b) the name printed on the page matches the MRZ name, or (c) the OCR was
confident about every MRZ line (MIN_CONFIDENCE_UNPROTECTED). Otherwise the result is marked for review.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date

from . import config
from .validation import Flag, visual_name_status


@dataclass
class Applicant:
    """Data the applicant typed on the form. Optional; every field given is checked against the document."""
    nationality: str | None = None
    surname: str | None = None
    given_names: str | None = None
    document_number: str | None = None
    birth_date: str | None = None     # YYYY-MM-DD
    expiry_date: str | None = None    # YYYY-MM-DD

    def any(self) -> bool:
        return any(v for v in self.__dict__.values())


def _tokens(*parts: str | None) -> list[str]:
    return [t for t in re.sub(r"[^A-Z ]", " ", " ".join(p or "" for p in parts).upper()).split() if t]


def names_match(applicant_tokens: list[str], doc_tokens: list[str]) -> bool:
    """Same words in any order. A document name cut off to fit the MRZ (a prefix of an applicant word) still matches."""
    if not applicant_tokens or not doc_tokens:        # nothing comparable (e.g. non-Latin names) is not a match
        return False
    remaining = list(applicant_tokens)
    for t in doc_tokens:
        if t in remaining:
            remaining.remove(t)
            continue
        cut = next((r for r in remaining if r.startswith(t) and len(t) >= 3), None)
        if cut is None:
            return False
        remaining.remove(cut)
    return not remaining


def _valid_date(s: str | None) -> bool:
    try:
        return s is not None and date.fromisoformat(s) is not None
    except ValueError:
        return False


def _clip(s: str, n: int = 40) -> str:
    """Caller text for a message: printable characters only, and short. Messages are stored with the result."""
    s = "".join(c if c.isprintable() else "?" for c in str(s))
    return s if len(s) <= n else s[:n] + "..."


def compare_applicant(doc: dict, applicant: Applicant) -> tuple[list[Flag], dict[str, bool]]:
    """Compare form data with the document. Returns (flags, which unprotected groups the form confirms)."""
    flags: list[Flag] = []
    verified = {"names": False, "nationality": False}
    if not applicant.any():
        return flags, verified

    doc_number = doc.get("documentNumber") or doc.get("idNumber") or ""
    if applicant.document_number:
        if re.sub(r"[^A-Z0-9]", "", applicant.document_number.upper()) != re.sub(r"[^A-Z0-9]", "", doc_number.upper()):
            flags.append(Flag("applicant_document_number_mismatch", "error",
                              f"Form says document number '{_clip(applicant.document_number)}', document says '{_clip(doc_number)}'"))
    for label, given, found in (("birth date", applicant.birth_date, doc.get("birthDate")),
                                ("expiry date", applicant.expiry_date, doc.get("expiryDate"))):
        if given:
            if not _valid_date(given):
                flags.append(Flag("applicant_data_invalid", "warning", f"Form {label} must be YYYY-MM-DD; check skipped"))
            elif given != found:
                flags.append(Flag(f"applicant_{label.split()[0]}_date_mismatch", "error",
                                  f"Form says {label} {given}, document says {found}"))
    if applicant.nationality:
        code = applicant.nationality.strip().upper()
        found = (doc.get("nationalityCode") or doc.get("nationality") or "").upper()
        if not re.fullmatch(r"[A-Z]{3}", code):
            flags.append(Flag("expected_nationality_invalid", "warning",
                              "Declared nationality must be a 3-letter ISO code such as SAU or EGY; check skipped"))
        elif code == found:
            verified["nationality"] = True
        elif not found.startswith("XX"):
            flags.append(Flag("nationality_mismatch", "error", f"Declared nationality {code} does not match the document ({found})"))
    if applicant.surname or applicant.given_names:
        a_tokens = _tokens(applicant.surname, applicant.given_names)
        d_tokens = _tokens(doc.get("surname"), doc.get("givenNames")) or _tokens(doc.get("nameEn"))
        if applicant.surname and not applicant.given_names:
            d_tokens = _tokens(doc.get("surname")) if doc.get("surname") else d_tokens
        if names_match(a_tokens, d_tokens):
            verified["names"] = True
        else:
            flags.append(Flag("applicant_name_mismatch", "error",
                              f"Form name '{_clip(' '.join(a_tokens), 120)}' does not match the document name "
                              f"'{_clip(' '.join(d_tokens), 120)}'"))
    return flags, verified


def apply_mrz_policy(result: dict, mrz_confidence: float | None, text_lines: list[dict] | None,
                     applicant: Applicant, mrz_lines: list[str] | None = None, line2_agreed: bool = False) -> None:
    """Add review flags to a parsed MRZ result (mutates it)."""
    doc = result["document"]
    flags: list[Flag] = []

    a_flags, verified = compare_applicant(doc, applicant)
    flags += a_flags

    # A name is cut only when it does not fit: the MRZ name line (line 1; line 3 on a three-line card) then has no filler at its end.
    name_line = (mrz_lines[2] if len(mrz_lines) == 3 else mrz_lines[0]) if mrz_lines else ""
    v_status, printed = visual_name_status(doc.get("surname", ""), doc.get("givenNames", ""), text_lines or [],
                                           name_field_full=bool(name_line) and not name_line.endswith("<"))
    if v_status == "mismatch":
        flags.append(Flag("name_differs_from_visual_zone", "error",
                          f"MRZ name '{doc.get('surname', '')} {doc.get('givenNames', '')}' differs from the name printed on "
                          f"the page ('{printed}'). One of them was probably misread. Check the document."))

    thr = config.min_confidence_unprotected()
    confident = mrz_confidence is not None and mrz_confidence >= thr
    status: dict[str, str] = {}

    def decide(name: str, by_form: bool, by_page: bool = False) -> bool:
        if by_form:
            status[name] = "confirmed by form"
        elif by_page:
            status[name] = "confirmed by printed page"
        elif confident:
            status[name] = "ocr confident"
        else:
            status[name] = "unverified"
        return status[name] != "unverified"

    names_ok = decide("names", verified["names"], v_status == "match")
    # line2_agreed: two readers read an identical second line that passes every check. Nationality and sex are in that line.
    nat_ok = decide("nationality", verified["nationality"], line2_agreed)
    sex_ok = decide("sex", False, line2_agreed)
    if line2_agreed:
        for k in ("nationality", "sex"):
            if status.get(k) == "confirmed by printed page":
                status[k] = "both readers agree"
    iss_ok = decide("issuingState", False)
    type_ok = decide("documentType", False)
    opt_ok = True
    if doc.get("format") in ("MRVA", "MRVB") and doc.get("optionalData"):
        opt_ok = decide("visaOptionalData", False)       # visa optional data has no check digit
    result["fieldStatus"] = status
    result["confidence"] = {"mrz": mrz_confidence}

    # Characters the check digit cannot tell apart (1/L, 6/G, ...): a document number containing them can be wrong silently.
    number = re.sub(r"[^A-Z0-9]", "", (doc.get("documentNumber") or "").upper())
    form_number = re.sub(r"[^A-Z0-9]", "", (applicant.document_number or "").upper())
    if re.search(r"[1L6G]", number) and not confident and not line2_agreed and form_number != number:
        flags.append(Flag("document_number_ambiguous_characters", "error",
                          "The document number contains characters the check digit cannot tell apart (1/L, 6/G). "
                          "Confirm it against the document or enter it on the form."))

    unverified = [n for n, ok in (("names", names_ok), ("nationality", nat_ok), ("sex", sex_ok), ("issuing state", iss_ok),
                                  ("document type", type_ok), ("visa optional data", opt_ok)) if not ok]
    if unverified:
        how = ("OCR confidence is not available for this reading" if mrz_confidence is None
               else f"OCR confidence {mrz_confidence:.2f} is below {thr:.2f}")
        flags.append(Flag("unprotected_fields_unverified", "error",
                          f"No check digit protects: {', '.join(unverified)}. {how}, and nothing else confirms them. "
                          "A person must check these fields against the document."))
    result["flags"] += [f.to_dict() for f in flags]
    result["needsReview"] = any(f["severity"] == "error" for f in result["flags"])
