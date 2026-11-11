"""Field extraction for ID cards that have no MRZ (Saudi Iqama, Saudi national ID, other GCC IDs).

Input is the list of text lines an OCR engine found (text, optional score, optional box). Fields are found by
their printed labels in Arabic or English, with regex fallbacks for the ID number and dates.

IMPORTANT: the label words and layout rules are written from general knowledge of these cards, not from real
specimens. Run it on real card images and adjust LABELS before relying on it.
"""
from __future__ import annotations

import functools

import re
from datetime import date

from . import gcc_ids
from .countries import arabic_country_code, country_name
from .textutil import normalize_arabic, to_ascii_digits
from .validation import Flag

# field -> label patterns (matched on text after Arabic normalisation, lower-cased)
LABELS: dict[str, list[str]] = {
    "idNumber": [r"رقم الهوية", r"رقم الهويه", r"رقم الاقامة", r"رقم الاقامه", r"iqama", r"residence permit", r"\bid no\b", r"id number",
                 r"الرقم المدني", r"الرقم الشخصي", r"رقم البطاقة", r"رقم البطاقه", r"الرقم الوطني", r"civil no", r"civil id", r"personal no", r"\bcpr\b"],
    "employerId": [r"هوية صاحب", r"هويه صاحب", r"employer id"],
    "employer": [r"اسم صاحب العمل", r"صاحب العمل", r"الكفيل", r"employer", r"sponsor"],
    "name": [r"الاسم", r"\bname\b"],
    "nationality": [r"الجنسية", r"الجنسيه", r"nationality"],
    "religion": [r"الديانة", r"الديانه", r"religion"],
    "occupation": [r"المهنة", r"المهنه", r"occupation", r"profession"],
    "birthPlace": [r"مكان الميلاد", r"place of birth"],
    "birthDate": [r"تاريخ الميلاد", r"date of birth", r"birth"],
    "expiryDate": [r"تاريخ الانتهاء", r"تاريخ انتهاء", r"انتهاء", r"expiry", r"expiration", r"expires"],
    "issuePlace": [r"مكان الاصد\w*", r"place of issue"],       # OCR often garbles the last letter, so no word end
    "workPlace": [r"مكان العمل", r"place of work", r"work location"],
}
def _load_label_overrides() -> None:
    """Add label patterns for other ID layouts without code changes: config/id_labels.json (or $ID_LABELS_FILE),
    {"idNumber": ["رقم البطاقة", ...], ...}. Extra patterns are tried after the built-in ones."""
    import json
    import os
    from pathlib import Path
    path = Path(os.environ.get("ID_LABELS_FILE", Path(__file__).parent.parent / "config" / "id_labels.json"))
    if not path.is_file():
        return
    try:
        extra = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    for field, patterns in extra.items():
        if field in LABELS and isinstance(patterns, list):
            for pat in patterns:
                try:
                    re.compile(pat)
                except re.error:
                    continue
                LABELS[field].append(pat)


NOISE = {"العمل", "الاصدار", "صاحب"}  # label fragments that OCR splits into their own box
_load_label_overrides()
ALL_LABELS = sum(LABELS.values(), [])

_DATE_YMD = re.compile(r"(?<!\d)(\d{4})\s*[/\-.]\s*(\d{1,2})\s*[/\-.]\s*(\d{1,2})(?!\d)")
_DATE_DMY = re.compile(r"(?<!\d)(\d{1,2})\s*[/\-.]\s*(\d{1,2})\s*[/\-.]\s*(\d{4})(?!\d)")
_ID10 = re.compile(r"(?<!\d)([12]\d{9})(?!\d)")
_ARABIC = re.compile("[؀-ۿ]")


def saudi_id_check(number: str) -> bool:
    """Check digit used for Saudi national IDs and Iqamas (Luhn variant). This is the widely used community
    algorithm, not an official published specification, so a failure is only a warning."""
    if not re.fullmatch(r"\d{10}", number):
        return False
    total = 0
    for i, ch in enumerate(number):
        d = int(ch)
        if i % 2 == 0:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def parse_date(text: str) -> tuple[str, str] | None:
    """Return (ISO-style date, calendar) where calendar is 'hijri' (year < 1700) or 'gregorian'."""
    t = to_ascii_digits(text)
    packed = re.fullmatch(r"\s*(\d{4})1(\d{2})1(\d{2})\s*", t)           # OCR read both "/" as the digit 1
    if packed:
        t = f"{packed[1]}/{packed[2]}/{packed[3]}"
    else:
        packed = re.fullmatch(r"\s*(\d{2})1(\d{2})1(\d{4})\s*", t)
        if packed:
            t = f"{packed[1]}/{packed[2]}/{packed[3]}"
    m = _DATE_YMD.search(t)
    if m:
        y, mo, d = int(m[1]), int(m[2]), int(m[3])
    else:
        m = _DATE_DMY.search(t)
        if not m:
            return None
        d, mo, y = int(m[1]), int(m[2]), int(m[3])
    if not (1 <= mo <= 12 and 1 <= d <= 31):
        return None
    return f"{y:04d}-{mo:02d}-{d:02d}", ("hijri" if y < 1700 else "gregorian")


def _norm(text: str) -> str:
    return normalize_arabic(to_ascii_digits(text)).lower().strip()


def _center(line: dict) -> tuple[float, float]:
    box = line.get("box") or []
    if not box:
        return (0.0, 0.0)
    if isinstance(box[0], (list, tuple)):
        xs, ys = [p[0] for p in box], [p[1] for p in box]
    else:
        xs, ys = [box[0], box[2]], [box[1], box[3]]
    return (sum(xs) / len(xs), sum(ys) / len(ys))


def _ordered(lines: list[dict]) -> list[dict]:
    if not any(l.get("box") for l in lines):
        return list(lines)
    return sorted(lines, key=lambda l: (round(_center(l)[1] / 12), _center(l)[0]))


@functools.lru_cache(maxsize=4096)
def _classify(text: str) -> tuple[str | None, str]:
    """Return (field, text after the label) when the line contains a label; (None, "") otherwise.
    The label that starts earliest wins, then the longest."""
    n = normalize_arabic(text)
    best = None
    for field, pats in LABELS.items():
        for p in pats:
            m = re.search(p, n, flags=re.IGNORECASE)
            if m and (best is None or (m.start(), -(m.end() - m.start())) < best[0]):
                best = ((m.start(), -(m.end() - m.start())), field, m.end())
    if best is None:
        return None, ""
    rest = re.sub(r"^[\s:：\-–/|]+", "", n[best[2]:]).strip()
    if not rest:                                     # some readers put the value before the label ("1988/11/02 تاريخ الميلاد")
        before = re.sub(r"[\s:：\-–/|]+$", "", n[:best[0][0]]).strip()
        rest = before
    return best[1], rest


def _bbox(line: dict) -> tuple[float, float, float, float] | None:
    box = line.get("box") or []
    if not box:
        return None
    if isinstance(box[0], (list, tuple)):
        xs, ys = [p[0] for p in box], [p[1] for p in box]
    else:
        xs, ys = [box[0], box[2]], [box[1], box[3]]
    return min(xs), min(ys), max(xs), max(ys)


def _repair_one_unreadable_digit(text: str) -> str | None:
    """A 10-character ID number with exactly one character that is not a digit (a slash, bar or similar). The check digit allows
    exactly one value for the gap, so the result is unique; the caller still flags it for review."""
    t = re.sub(r"[\s-]", "", to_ascii_digits(text))
    for start in range(0, len(t) - 9):
        cand = t[start:start + 10]
        bad = [i for i, c in enumerate(cand) if not c.isdigit()]
        if len(bad) != 1 or not re.fullmatch(r"[\d/|\\!?.,:;lI]{10}", cand):
            continue
        options = [cand[:bad[0]] + str(d) + cand[bad[0] + 1:] for d in range(10)]
        options = [o for o in options if o[0] in "12" and saudi_id_check(o)]
        if len(options) == 1:
            return options[0]
    return None


def _is_label(line: dict) -> bool:
    n = normalize_arabic(line["text"]).strip()
    return n in NOISE or _classify(line["text"])[0] is not None


def _find_value(field: str, lines: list[dict]) -> tuple[str, dict] | None:
    for idx, ln in enumerate(lines):
        f, rest = _classify(ln["text"])
        if f != field:
            continue
        if rest:                                   # "label: value" in one box
            return rest, ln
        bb = _bbox(ln)
        if bb:                                     # value on the same row; Arabic cards put it LEFT of the label
            cy, h = (bb[1] + bb[3]) / 2, bb[3] - bb[1]
            cands = []
            for o in lines:
                ob = _bbox(o)
                if o is ln or ob is None or _is_label(o) or not o["text"].strip():
                    continue
                if abs((ob[1] + ob[3]) / 2 - cy) <= 0.6 * max(h, ob[3] - ob[1]):
                    gap_left = bb[0] - ob[2]
                    gap_right = ob[0] - bb[2]
                    slack = -0.8 * max(h, 1)           # blur or tilt makes neighbouring boxes overlap a little
                    if gap_left >= slack:
                        cands.append((max(gap_left, 0), o))
                    elif gap_right >= slack:
                        cands.append((1000 + gap_right, o))
            if cands:
                return min(cands, key=lambda c: c[0])[1]["text"].strip(), min(cands, key=lambda c: c[0])[1]
        for nxt in lines[idx + 1: idx + 3]:        # no geometry: value on the next line
            if not _is_label(nxt) and nxt["text"].strip():
                return nxt["text"].strip(), nxt
    return None


def _find_other_gcc_number(lines, labelled):
    """UAE, Kuwait, Qatar or Bahrain number: from the value next to an ID label, or (UAE only) anywhere, since 784 is distinctive."""
    candidates = []
    if labelled:
        candidates.append((labelled[0], labelled[1].get("score", 1.0)))
    candidates += [(ln["_n"], ln.get("score", 1.0)) for ln in lines]
    for text, score in candidates:
        digits = re.sub(r"[\s-]", "", to_ascii_digits(text))
        for m in re.finditer(r"\d{9,15}", digits):
            info = gcc_ids.classify(m[0])
            if info and (text is candidates[0][0] and labelled or info["kind"] == "uae_emirates_id"):
                return m[0], info, score
    # The label can be garbled beyond recognition (Arabic words come out reversed or mangled). Take a number that fits one of the
    # GCC formats, from a line that holds no date, but only when exactly one such number is on the card.
    found: dict[str, tuple[dict, float]] = {}
    for ln in lines:
        if _DATE_YMD.search(ln["_n"]) or _DATE_DMY.search(ln["_n"]):
            continue
        digits = re.sub(r"(?<=\d)[\s-]+(?=\d)", "", ln["_n"])
        for m in re.finditer(r"(?<!\d)\d{9,15}(?!\d)", digits):
            info = gcc_ids.classify(m[0])
            if info:
                found[m[0]] = (info, ln.get("score", 1.0))
    if len(found) == 1:
        num, (info, score) = next(iter(found.items()))
        return num, info, min(score, 0.6)
    return None


_HEADER_LATIN = {"KINGDOM", "SAUDI", "ARABIA", "PREMIUM", "RESIDENT", "IDENTITY", "RESIDENCY", "RESIDENCE", "PERMIT", "IQAMA",
                 "MINISTRY", "INTERIOR", "REPUBLIC", "NATIONAL", "CARD", "ID", "OF"}
_HEADER_ARABIC = ("المملكه العربيه السعوديه", "هويه مقيم", "الاقامه المميزه", "رقم النسخه", "يجب التحقق", "وزاره الداخليه",
                  "بطاقه هويه", "الرمز السريع")


def _is_header_text(text: str) -> bool:
    """Card titles and printed notices that are not a holder's name or value."""
    words = re.findall(r"[A-Za-z]+", text.upper())
    if words and sum(w in _HEADER_LATIN for w in words) >= max(1, (len(words) + 1) // 2):
        return True
    n = normalize_arabic(text)
    return any(h in n for h in _HEADER_ARABIC)


def _resolve_country(text: str) -> str:
    """ISO code when the text is, or contains as whole words, a known country name (Arabic or English) or a 3-letter code."""
    from .countries import ARABIC_COUNTRIES, COUNTRIES
    t = text.strip()
    if re.fullmatch(r"[A-Za-z]{3}", t) and t.upper() in COUNTRIES:
        return t.upper()
    n = normalize_arabic(t)
    for name in sorted(ARABIC_COUNTRIES, key=len, reverse=True):
        if re.search(r"(?<!\w)" + re.escape(normalize_arabic(name)) + r"(?!\w)", n):
            return ARABIC_COUNTRIES[name]
    low = " " + re.sub(r"[^a-z ]+", " ", t.lower()) + " "
    for code, name in sorted(COUNTRIES.items(), key=lambda kv: -len(kv[1])):
        if len(name) >= 4 and f" {name.lower()} " in low:
            return code
    return ""


def _row_neighbours(label: dict, lines: list[dict]) -> list[dict]:
    bb = _bbox(label)
    if not bb:
        return []
    cy, h = (bb[1] + bb[3]) / 2, bb[3] - bb[1]
    out = []
    for o in lines:
        ob = _bbox(o)
        if o is label or ob is None or not o["text"].strip():
            continue
        if abs((ob[1] + ob[3]) / 2 - cy) <= 0.9 * max(h, ob[3] - ob[1]):
            out.append(o)
    return out



def parse_gcc_id(raw_lines: list[dict], today: date | None = None, auto_accept: bool | None = None,
                 min_confidence: float | None = None) -> dict:
    """Read an ID card with no MRZ. Results always need review unless auto_accept (VISUAL_ZONE_AUTO_ACCEPT) is on;
    even then a failed check digit, a low-confidence field or a missing ID number sends the card to a person."""
    from . import config
    today = today or date.today()
    auto_accept = config.visual_zone_auto_accept() if auto_accept is None else auto_accept
    min_confidence = config.min_confidence_visual() if min_confidence is None else min_confidence
    lines = _ordered([{**l, "text": str(l.get("text", "")), "_n": to_ascii_digits(str(l.get("text", "")))}
                      for l in raw_lines if str(l.get("text", "")).strip()])
    flags: list[Flag] = []
    doc: dict = {"kind": "gcc_id", "format": "visual"}
    scores: dict[str, float] = {}

    # ID number: labelled value first, otherwise any 10-digit number starting with 1 or 2
    id_number = None
    repaired_id = False
    found = _find_value("idNumber", lines)
    if found:
        m = _ID10.search(re.sub(r"[\s-]", "", to_ascii_digits(found[0])))
        if m:
            id_number, scores["idNumber"] = m[1], found[1].get("score", 1.0)
        else:
            fixed = _repair_one_unreadable_digit(found[0])
            if fixed:
                id_number, scores["idNumber"] = fixed, min(found[1].get("score", 1.0), 0.5)
                repaired_id = True
    unlabelled_id = False
    if id_number is None:
        for ln in lines:
            if _classify(ln["text"])[0] not in (None, "idNumber"):
                continue                     # a number on another field's line (employer ID, a date) is not the holder's number
            m = _ID10.search(re.sub(r"[\s-]", "", ln["_n"]))
            if m:
                id_number, scores["idNumber"] = m[1], ln.get("score", 1.0)
                unlabelled_id = True         # not next to an ID label: nothing says this is the holder's number
                break
    checks: dict[str, bool] = {}
    if id_number:
        doc["idNumber"] = id_number
        doc["kind"] = "iqama" if id_number[0] == "2" else "saudi_national_id"
        checks["id_number"] = saudi_id_check(id_number)
        if repaired_id:
            # error, not a score cap: a rebuilt number must reach a person whatever the confidence threshold is
            flags.append(Flag("id_number_repaired", "error",
                              "One digit of the ID number was unreadable and was restored from the check digit. Confirm it against the card."))
        if unlabelled_id:
            flags.append(Flag("id_number_unlabelled", "error",
                              "The ID number was not found next to an ID label; a 10-digit number from elsewhere on the card was used. "
                              "Confirm it against the card."))
        if not checks["id_number"]:
            flags.append(Flag("id_number_checksum_failed", "error",
                              "The ID number fails the check digit. A digit was probably misread."))
    else:
        other = _find_other_gcc_number(lines, found)
        if other:
            digits, info, score = other
            id_number = digits
            doc["idNumber"], doc["kind"], doc["issuingCountry"] = digits, info["kind"], info["country"]
            scores["idNumber"] = score
            checks["id_number"] = info["valid"]
            if not info["valid"]:
                flags.append(Flag("id_number_checksum_failed", "error",
                                  "The ID number fails its check digit. A digit was probably misread."))
            flags.append(Flag("gcc_layout_unverified", "error",
                              f"Recognised as {info['kind'].replace('_', ' ')} from the number format ({info['note']}). "
                              "This card type has not been tested on real specimens, so a person must check every field."))
            doc["_numberBirthDate"] = info["birthDate"]
        else:
            flags.append(Flag("id_number_not_found", "error", "No 10-digit ID number starting with 1 or 2 was found"))

    # names: keep Arabic and Latin separately. Cards often print them with no label.
    name = _find_value("name", lines)
    if name and _is_header_text(name[0]):
        name = None
    if name:
        key = "nameAr" if _ARABIC.search(name[0]) else "nameEn"
        doc[key], scores[key] = name[0], name[1].get("score", 1.0)
    en_line = None
    if "nameEn" not in doc:
        for ln in lines:
            t = ln["text"].strip()
            if _is_label(ln) or _is_header_text(t):
                continue
            # the longest run of capital-letter words; readers add stray marks around it ("= MOHAMED ... BASYUNI pases i")
            runs = re.findall(r"[A-Z][A-Z'\-]+(?: [A-Z][A-Z'\-]+)+", t)
            if runs:
                best = max(runs, key=len)
                if len(best.split()) >= 2 and not _is_header_text(best):
                    doc["nameEn"], scores["nameEn"], en_line = best, ln.get("score", 1.0), ln
                    break
    eb = _bbox(en_line) if en_line is not None else None
    if "nameAr" not in doc and eb is not None:
        ecy, eh = (eb[1] + eb[3]) / 2, eb[3] - eb[1]
        above = []
        for o in lines:
            ob = _bbox(o)
            if (not ob or o is en_line or _is_label(o) or not _ARABIC.search(o["text"]) or len(o["text"].split()) < 3
                    or _is_header_text(o["text"])):
                continue
            dy = ecy - (ob[1] + ob[3]) / 2           # centres, so tilt does not matter
            overlaps_x = min(eb[2], ob[2]) - max(eb[0], ob[0]) > 0
            if 0.2 * eh <= dy <= 2.2 * eh and overlaps_x:
                above.append((dy, o))
        if above:
            o = min(above, key=lambda a: a[0])[1]
            ar = re.sub(r"\s+", " ", re.sub(r"[A-Za-z0-9\"'=|_\\]+", " ", o["text"])).strip()    # drop Latin noise the reader added
            doc["nameAr"], scores["nameAr"] = ar or o["text"].strip(), o.get("score", 1.0)
    for field in ("nationality", "occupation", "religion", "birthPlace", "issuePlace", "workPlace", "employer"):
        f = _find_value(field, lines)
        if field == "nationality":
            # The value must be a country. A neighbouring notice or a garbled box ("يجب التحقق") is not; look along the same row.
            if not f or not _resolve_country(f[0]):
                alt = None
                for lab in [l for l in lines if _classify(l["text"])[0] == "nationality"]:
                    for o in _row_neighbours(lab, lines):
                        if not _is_label(o) and not _is_header_text(o["text"]) and _resolve_country(o["text"]):
                            alt = (o["text"].strip(), o)
                            break
                    if alt:
                        break
                f = alt or (f if f and not _is_header_text(f[0]) else None)
        if f:
            value = f[0]
            if field == "nationality" and ":" in value and _resolve_country(value):
                value = value.rsplit(":", 1)[1].strip() or value        # keep the country, drop a label remnant
            doc[field], scores[field] = value, f[1].get("score", 1.0)
    f = _find_value("employerId", lines)
    if f:
        digits = re.sub(r"\D", "", to_ascii_digits(f[0]))
        if digits:
            doc["employerId"], scores["employerId"] = digits, f[1].get("score", 1.0)

    # dates, assigned by label first
    dates: dict[str, tuple[str, str]] = {}
    for field in ("birthDate", "expiryDate"):
        f = _find_value(field, lines)
        if f:
            parsed = parse_date(f[0])
            if parsed:
                dates[field], scores[field] = parsed, f[1].get("score", 1.0)
    if "expiryDate" not in dates and "birthDate" in dates:
        # The expiry label is often garbled by OCR ("تاريخ <ul 2028/01/30"). A second line that says "date" and has one date is
        # taken as the expiry, but only as a guess: it always reaches a person.
        for ln in lines:
            if _classify(ln["text"])[0] is not None or not re.search(r"تاريخ|date", normalize_arabic(ln["text"]), re.I):
                continue
            parsed = parse_date(ln["text"])
            if parsed and parsed != dates["birthDate"]:
                dates["expiryDate"], scores["expiryDate"] = parsed, min(ln.get("score", 1.0), 0.5)
                flags.append(Flag("expiry_label_guessed", "error",
                                  "The expiry label could not be read; the date on the other 'date' line was used. Check it."))
                break
    for field, (iso, cal) in dates.items():
        doc[field], doc[field.replace("Date", "Calendar")] = iso, cal
    for field in ("birthDate", "expiryDate"):
        if field not in doc:
            # With auto-accept on, an unreadable expiry must still reach a person
            sev = "error" if (auto_accept and field == "expiryDate") else "warning"
            flags.append(Flag(f"{field}_not_found", sev, f"Could not read {field}"))

    if doc.get("expiryDate"):
        if doc.get("expiryCalendar") == "gregorian":
            if date.fromisoformat(doc["expiryDate"]) < today:
                flags.append(Flag("document_expired", "error", "The document has expired"))
        else:
            flags.append(Flag("expiry_hijri_not_checked", "error" if auto_accept else "warning",
                              "Expiry is a Hijri date; expiry check needs the Gregorian date"))
    if doc.get("birthDate") and doc.get("birthCalendar") == "gregorian" and date.fromisoformat(doc["birthDate"]) > today:
        flags.append(Flag("birth_date_in_future", "error", "Date of birth is in the future"))

    from_number = doc.pop("_numberBirthDate", None)
    if from_number and doc.get("birthDate") and doc["birthDate"] != from_number and doc.get("birthCalendar") == "gregorian":
        flags.append(Flag("birth_date_differs_from_id_number", "error",
                          f"Birth date {doc['birthDate']} differs from the date inside the ID number ({from_number})."))
    elif from_number and "birthDate" not in doc:
        doc["birthDate"], doc["birthCalendar"] = from_number, "gregorian"
        flags = [f for f in flags if f.code != "birthDate_not_found"]
        flags.append(Flag("birth_date_from_id_number", "warning", "Birth date was taken from the ID number."))

    low = sorted(k for k, v in scores.items() if v is not None and v < min_confidence)
    if low:
        flags.append(Flag("low_confidence_fields", "error", "Low OCR confidence for: " + ", ".join(low)))
    flags.append(Flag("visual_zone_only", "warning" if auto_accept else "error",
                      "This ID has no MRZ, so fields come from OCR of printed text only. "
                      + ("Review is recommended." if auto_accept else "A person must review it before approval.")))

    nat = doc.get("nationality", "")
    code = (nat.upper() if re.fullmatch(r"[A-Za-z]{3}", nat) else arabic_country_code(nat)) or _resolve_country(nat)
    doc["nationalityCode"] = code
    doc["nationalityName"] = country_name(code) if code else ""
    doc["checks"] = checks
    doc["fieldConfidence"] = {k: round(float(v), 3) for k, v in scores.items() if v is not None}
    doc["correctedFields"] = []
    doc["valid"] = bool(checks.get("id_number"))
    return {"document": doc, "flags": [f.to_dict() for f in flags],
            "needsReview": any(f.severity == "error" for f in flags) or "idNumber" not in doc}
