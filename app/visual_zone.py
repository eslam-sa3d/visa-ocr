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
from .textutil import HIJRI_MARGIN_DAYS, hijri_to_gregorian, normalize_arabic, to_ascii_digits
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
    from . import config  # noqa: WPS433
    from pathlib import Path
    path = Path(config.id_labels_file() or Path(__file__).parent.parent / "config" / "id_labels.json")
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


def hijri_expiry_flag(doc: dict, today: date, auto_accept: bool) -> Flag:
    """Expiry check for a Hijri date. The date is converted by the arithmetic calendar, which can be a day or two away from
    the official Umm al-Qura date, so only a clear answer is given: expired by more than the margin is an error, valid by
    more than the margin is noted with the converted date, and anything closer still needs a person."""
    try:
        y, m, d = (int(x) for x in str(doc.get("expiryDate", "")).split("-"))
        approx = hijri_to_gregorian(y, m, d)
    except ValueError:
        approx = None
    if approx is None:
        return Flag("expiry_hijri_not_checked", "error" if auto_accept else "warning",
                    "Expiry is a Hijri date that could not be converted; check it against the Gregorian date")
    doc["expiryDateGregorianApprox"] = approx.isoformat()
    days = (approx - today).days
    if days < -HIJRI_MARGIN_DAYS:
        return Flag("document_expired", "error", f"The document has expired (Hijri {doc['expiryDate']}, about {approx.isoformat()})")
    if days > HIJRI_MARGIN_DAYS:
        return Flag("expiry_hijri_converted", "warning",
                    f"Expiry is a Hijri date, about {approx.isoformat()} (within {HIJRI_MARGIN_DAYS} days): not expired")
    return Flag("expiry_hijri_not_checked", "error" if auto_accept else "warning",
                f"Expiry is a Hijri date within {HIJRI_MARGIN_DAYS} days of today (about {approx.isoformat()}); "
                "check it against the Gregorian date")


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


_GULF_NAMES = {
    "SAU": (r"\bsaudi\b|\bksa\b", ("السعوديه", "هويه مقيم", "هويه وطنيه")),
    "ARE": (r"\bemirates\b|\buae\b", ("الامارات",)),
    "KWT": (r"\bkuwait\b", ("الكويت",)),
    "QAT": (r"\bqatar\b", ("قطر",)),
    "BHR": (r"\bbahrain\b", ("البحرين",)),
    "OMN": (r"\boman\b|\bsultanate\b", ("سلطنه عمان",)),
}
_GULF_ID_LABEL = re.compile(r"\bcpr\b|\bcivil (?:no|id)\b|\bpersonal no\b|\biqama\b", re.I)
_GULF_ID_LABEL_AR = ("رقم الهويه", "رقم الاقامه", "الرقم المدني", "الرقم الشخصي", "الرقم الوطني")


def gulf_evidence(lines: list[dict]) -> tuple[set[str], bool]:
    """What the printed text says about where the card is from: the Gulf states it names, and whether it carries a Gulf ID label.
    A number that merely has the right length proves nothing (a European permit number is nine digits too), so a number with no
    check digit of its own is only taken as a Gulf ID when the card itself says so."""
    text = " ".join(str(l.get("text", "")) for l in lines)
    arabic = normalize_arabic(text).replace("ة", "ه")      # teh marbuta is often read (and written) as heh
    named = {code for code, (latin, ar) in _GULF_NAMES.items() if re.search(latin, text, re.I) or any(a in arabic for a in ar)}
    return named, bool(_GULF_ID_LABEL.search(text)) or any(a in arabic for a in _GULF_ID_LABEL_AR)


def _number_fits_card(info: dict, named: set[str], labelled_gulf: bool) -> bool:
    if info["kind"] in ("uae_emirates_id", "kuwait_civil_id") and info["valid"]:
        return True         # these two prove themselves; a state named on the card may only be the holder's nationality
    if named:
        return info["country"] in named            # a Kuwaiti card does not carry a Bahraini number
    return labelled_gulf


def _find_other_gcc_number(lines, labelled, named=frozenset(), labelled_gulf=True):
    """UAE, Kuwait, Qatar or Bahrain number: from the value next to an ID label, or (UAE only) anywhere, since 784 is distinctive."""
    candidates = []
    if labelled:
        candidates.append((labelled[0], labelled[1].get("score", 1.0)))
    candidates += [(ln["_n"], ln.get("score", 1.0)) for ln in lines]
    for text, score in candidates:
        digits = re.sub(r"[\s-]", "", to_ascii_digits(text))
        for m in re.finditer(r"\d{9,15}", digits):
            info = gcc_ids.classify(m[0])
            if info and not _number_fits_card(info, named, labelled_gulf):
                continue
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
            if info and _number_fits_card(info, named, labelled_gulf):
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



class _Card:
    """What has been read from one card so far."""
    def __init__(self, lines: list[dict]) -> None:
        self.lines = lines
        self.doc: dict = {"kind": "gcc_id", "format": "visual"}
        self.scores: dict[str, float] = {}
        self.flags: list[Flag] = []
        self.checks: dict[str, bool] = {}

    def set(self, field: str, value, line: dict | None) -> None:
        self.doc[field] = value
        self.scores[field] = (line or {}).get("score", 1.0)


def _saudi_number(c: _Card):
    """The 10-digit Saudi number: by its label first, otherwise any such number not on another field's line. Returns
    (number or None, the labelled value found, was a digit rebuilt, was it taken without a label)."""
    found = _find_value("idNumber", c.lines)
    if found:
        m = _ID10.search(re.sub(r"[\s-]", "", to_ascii_digits(found[0])))
        if m:
            c.scores["idNumber"] = found[1].get("score", 1.0)
            return m[1], found, False, False
        fixed = _repair_one_unreadable_digit(found[0])
        if fixed:
            c.scores["idNumber"] = min(found[1].get("score", 1.0), 0.5)
            return fixed, found, True, False
    for ln in c.lines:
        if _classify(ln["text"])[0] not in (None, "idNumber"):
            continue                     # a number on another field's line (employer ID, a date) is not the holder's number
        m = _ID10.search(re.sub(r"[\s-]", "", ln["_n"]))
        if m:
            c.scores["idNumber"] = ln.get("score", 1.0)
            return m[1], found, False, True          # not next to an ID label: nothing says this is the holder's number
    return None, found, False, False


def _read_id_number(c: _Card) -> None:
    """ID number, card kind and check digit, with a flag for every way the number can be less than certain."""
    id_number, found, repaired, unlabelled = _saudi_number(c)
    named, labelled_gulf = gulf_evidence(c.lines)
    if len(named) == 1:
        c.doc["issuingCountry"] = next(iter(named))
    # A ten-digit number is a Saudi number only if the card can be Saudi: not when it names another Gulf state only, and not
    # when the number has no ID label and nothing on the card says Saudi Arabia (passports and foreign permits carry such numbers).
    if id_number and ((named and "SAU" not in named) or (unlabelled and "SAU" not in named and not labelled_gulf)):
        id_number = None
        c.scores.pop("idNumber", None)
    if id_number:
        c.doc["idNumber"] = id_number
        c.doc["kind"] = "iqama" if id_number[0] == "2" else "saudi_national_id"
        c.checks["id_number"] = saudi_id_check(id_number)
        if repaired:
            # error, not a score cap: a rebuilt number must reach a person whatever the confidence threshold is
            c.flags.append(Flag("id_number_repaired", "error",
                                "One digit of the ID number was unreadable and was restored from the check digit. Confirm it against the card."))
        if unlabelled:
            c.flags.append(Flag("id_number_unlabelled", "error",
                                "The ID number was not found next to an ID label; a 10-digit number from elsewhere on the card was used. "
                                "Confirm it against the card."))
        if not c.checks["id_number"]:
            c.flags.append(Flag("id_number_checksum_failed", "error",
                                "The ID number fails the check digit. A digit was probably misread."))
        return
    other = _find_other_gcc_number(c.lines, found, named, labelled_gulf)
    if not other:
        c.flags.append(Flag("id_number_not_found", "error", "No 10-digit ID number starting with 1 or 2 was found"))
        return
    digits, info, score = other
    c.doc["idNumber"], c.doc["kind"], c.doc["issuingCountry"] = digits, info["kind"], info["country"]
    c.scores["idNumber"] = score
    c.checks["id_number"] = info["valid"]
    if not info["valid"]:
        c.flags.append(Flag("id_number_checksum_failed", "error",
                            "The ID number fails its check digit. A digit was probably misread."))
    c.flags.append(Flag("gcc_layout_unverified", "error",
                        f"Recognised as {info['kind'].replace('_', ' ')} from the number format ({info['note']}). "
                        "This card type has not been tested on real specimens, so a person must check every field."))
    c.doc["_numberBirthDate"] = info["birthDate"]


def _unlabelled_latin_name(c: _Card) -> dict | None:
    """Cards often print the name with no label: take the longest run of capital-letter words (readers add stray marks around
    it, "= MOHAMED ... BASYUNI pases i"). Returns the line it came from."""
    for ln in c.lines:
        t = ln["text"].strip()
        if _is_label(ln) or _is_header_text(t):
            continue
        runs = re.findall(r"[A-Z][A-Z'\-]+(?: [A-Z][A-Z'\-]+)+", t)
        if runs:
            best = max(runs, key=len)
            if len(best.split()) >= 2 and not _is_header_text(best):
                c.set("nameEn", best, ln)
                return ln
    return None


def _arabic_name_above(c: _Card, en_line: dict) -> None:
    """The Arabic name is printed just above the Latin one: the nearest Arabic line of three or more words over it."""
    eb = _bbox(en_line)
    if eb is None:
        return
    ecy, eh = (eb[1] + eb[3]) / 2, eb[3] - eb[1]
    above = []
    for o in c.lines:
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
        c.set("nameAr", ar or o["text"].strip(), o)


def _read_names(c: _Card) -> None:
    """Arabic and Latin names, kept separately."""
    name = _find_value("name", c.lines)
    if name and _is_header_text(name[0]):
        name = None
    if name:
        c.set("nameAr" if _ARABIC.search(name[0]) else "nameEn", name[0], name[1])
    en_line = _unlabelled_latin_name(c) if "nameEn" not in c.doc else None
    if "nameAr" not in c.doc and en_line is not None:
        _arabic_name_above(c, en_line)


def _nationality_value(c: _Card, f):
    """The value must be a country. A neighbouring notice or a garbled box is not; look along the label's row."""
    if f and _resolve_country(f[0]):
        return f
    for lab in [l for l in c.lines if _classify(l["text"])[0] == "nationality"]:
        for o in _row_neighbours(lab, c.lines):
            if not _is_label(o) and not _is_header_text(o["text"]) and _resolve_country(o["text"]):
                return o["text"].strip(), o
    return f if f and not _is_header_text(f[0]) else None


def _read_labelled_fields(c: _Card) -> None:
    for field in ("nationality", "occupation", "religion", "birthPlace", "issuePlace", "workPlace", "employer"):
        f = _find_value(field, c.lines)
        if field == "nationality":
            f = _nationality_value(c, f)
        if f:
            value = f[0]
            if field == "nationality" and ":" in value and _resolve_country(value):
                value = value.rsplit(":", 1)[1].strip() or value        # keep the country, drop a label remnant
            c.set(field, value, f[1])
    f = _find_value("employerId", c.lines)
    if f:
        digits = re.sub(r"\D", "", to_ascii_digits(f[0]))
        if digits:
            c.set("employerId", digits, f[1])


def _date_below_label(field: str, lines: list[dict]) -> tuple[tuple[str, str], dict] | None:
    """The nearest date printed under a field's label (overlapping it sideways, within three line heights)."""
    for label in lines:
        if _classify(label["text"])[0] != field:
            continue
        lb = _bbox(label)
        if not lb:
            continue
        h = max(lb[3] - lb[1], 1.0)
        best = None
        for o in lines:
            ob = _bbox(o)
            if o is label or not ob or _is_label(o):
                continue
            if lb[1] + 0.4 * h <= ob[1] <= lb[3] + 3 * h and ob[2] >= lb[0] - h and ob[0] <= lb[2] + h:
                parsed = parse_date(o["text"])
                if parsed and (best is None or ob[1] < best[2]):
                    best = (parsed, o, ob[1])
        if best:
            return best[0], best[1]
    return None


def _read_sex(c: _Card) -> None:
    """"Sex: M" or "الجنس ذكر" printed in one piece."""
    for ln in c.lines:
        m = re.search(r"\bsex\s*[:.]?\s*([MF])\b", ln["text"], re.I)
        if m:
            c.doc["sex"], c.scores["sex"] = m[1].upper(), ln.get("score", 1.0)
            return
    for ln in c.lines:
        t = normalize_arabic(ln["text"])
        if re.search(r"الجنس(?!ي)", t) and re.search(r"ذكر|انثي|انثى", t):
            c.doc["sex"], c.scores["sex"] = ("M" if "ذكر" in t else "F"), ln.get("score", 1.0)
            return


def _read_dates(c: _Card, auto_accept: bool) -> None:
    """Birth and expiry dates, by label first."""
    dates: dict[str, tuple[str, str]] = {}
    for field in ("birthDate", "expiryDate"):
        f = _find_value(field, c.lines)
        if f:
            parsed = parse_date(f[0])
            if parsed:
                dates[field], c.scores[field] = parsed, f[1].get("score", 1.0)
    for field in ("birthDate", "expiryDate"):
        if field not in dates:
            below = _date_below_label(field, c.lines)        # Emirates ID: the date is printed under its label, not beside it
            if below:
                dates[field], c.scores[field] = below[0], below[1].get("score", 1.0)
    if "expiryDate" not in dates and "birthDate" in dates:
        # The expiry label is often garbled by OCR ("تاريخ <ul 2028/01/30"). A second line that says "date" and has one date is
        # taken as the expiry, but only as a guess: it always reaches a person.
        for ln in c.lines:
            if _classify(ln["text"])[0] is not None or not re.search(r"تاريخ|date", normalize_arabic(ln["text"]), re.I):
                continue
            parsed = parse_date(ln["text"])
            if parsed and parsed != dates["birthDate"]:
                dates["expiryDate"], c.scores["expiryDate"] = parsed, min(ln.get("score", 1.0), 0.5)
                c.flags.append(Flag("expiry_label_guessed", "error",
                                    "The expiry label could not be read; the date on the other 'date' line was used. Check it."))
                break
    for field, (iso, cal) in dates.items():
        c.doc[field], c.doc[field.replace("Date", "Calendar")] = iso, cal
    for field in ("birthDate", "expiryDate"):
        if field not in c.doc:
            # With auto-accept on, an unreadable expiry must still reach a person
            sev = "error" if (auto_accept and field == "expiryDate") else "warning"
            c.flags.append(Flag(f"{field}_not_found", sev, f"Could not read {field}"))


def _check_dates(c: _Card, today: date, auto_accept: bool) -> None:
    """Expired, born in the future, and the birth date some ID numbers carry inside them."""
    doc = c.doc
    if doc.get("expiryDate"):
        if doc.get("expiryCalendar") == "gregorian":
            if date.fromisoformat(doc["expiryDate"]) < today:
                c.flags.append(Flag("document_expired", "error", "The document has expired"))
        else:
            c.flags.append(hijri_expiry_flag(doc, today, auto_accept))
    if doc.get("birthDate") and doc.get("birthCalendar") == "gregorian" and date.fromisoformat(doc["birthDate"]) > today:
        c.flags.append(Flag("birth_date_in_future", "error", "Date of birth is in the future"))
    from_number = doc.pop("_numberBirthDate", None)
    if from_number and doc.get("birthDate") and doc["birthDate"] != from_number and doc.get("birthCalendar") == "gregorian":
        c.flags.append(Flag("birth_date_differs_from_id_number", "error",
                            f"Birth date {doc['birthDate']} differs from the date inside the ID number ({from_number})."))
    elif from_number and "birthDate" not in doc:
        doc["birthDate"], doc["birthCalendar"] = from_number, "gregorian"
        c.flags = [f for f in c.flags if f.code != "birthDate_not_found"]
        c.flags.append(Flag("birth_date_from_id_number", "warning", "Birth date was taken from the ID number."))


def parse_gcc_id(raw_lines: list[dict], today: date | None = None, auto_accept: bool | None = None,
                 min_confidence: float | None = None) -> dict:
    """Read an ID card with no MRZ. Results always need review unless auto_accept (VISUAL_ZONE_AUTO_ACCEPT) is on;
    even then a failed check digit, a low-confidence field or a missing ID number sends the card to a person."""
    from . import config
    today = today or date.today()
    auto_accept = config.visual_zone_auto_accept() if auto_accept is None else auto_accept
    min_confidence = config.min_confidence_visual() if min_confidence is None else min_confidence
    c = _Card(_ordered([{**l, "text": str(l.get("text", "")), "_n": to_ascii_digits(str(l.get("text", "")))}
                        for l in raw_lines if str(l.get("text", "")).strip()]))
    _read_id_number(c)
    _read_names(c)
    _read_labelled_fields(c)
    _read_dates(c, auto_accept)
    _read_sex(c)
    _check_dates(c, today, auto_accept)

    doc, flags = c.doc, c.flags
    low = sorted(k for k, v in c.scores.items() if v is not None and v < min_confidence)
    if low:
        flags.append(Flag("low_confidence_fields", "error", "Low OCR confidence for: " + ", ".join(low)))
    flags.append(Flag("visual_zone_only", "warning" if auto_accept else "error",
                      "This ID has no MRZ, so fields come from OCR of printed text only. "
                      + ("Review is recommended." if auto_accept else "A person must review it before approval.")))
    nat = doc.get("nationality", "")
    code = (nat.upper() if re.fullmatch(r"[A-Za-z]{3}", nat) else arabic_country_code(nat)) or _resolve_country(nat)
    doc["nationalityCode"] = code
    doc["nationalityName"] = country_name(code) if code else ""
    doc["checks"] = c.checks
    doc["fieldConfidence"] = {k: round(float(v), 3) for k, v in c.scores.items() if v is not None}
    doc["correctedFields"] = []
    doc["valid"] = bool(c.checks.get("id_number"))
    return {"document": doc, "flags": [f.to_dict() for f in flags],
            "needsReview": any(f.severity == "error" for f in flags) or "idNumber" not in doc}
