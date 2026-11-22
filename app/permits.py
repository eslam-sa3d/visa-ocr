"""Read the FRONT of a residence card that has no MRZ on that side: the EU uniform-format residence permit (all member
states, Norway, Switzerland), the UK biometric residence permit, the US Permanent Resident Card, and the Omani resident or
identity card.

These fronts carry no check digit, so nothing read here can prove itself. Every result is marked for review and says so.
The MRZ side of the same card (the back) is read by the MRZ reader and is the better upload: when a card has one, the
message asks for it.

How a field is found: each printed label ("DATE OF BIRTH", "GEBURTSDATUM", "Card Expires") is located, and the value is
the nearest text below it, or beside it on the same row, that has the right shape for that field (a date, one letter, a
three-letter code). Several labels often share one printed line; the shape of the value tells them apart.
"""
from __future__ import annotations

import re
import unicodedata
from datetime import date

from .textutil import normalize_arabic, to_ascii_digits
from .validation import Flag

# What the title of the card says, by kind. Matched on text with accents removed, lower case.
_TITLES = {
    "us_permanent_resident_card": (r"permanent resident", r"united states|uscis"),
    "uk_residence_permit": (r"residence permit", r"leave to (?:remain|enter)|settlement|\buk\b|united kingdom|work permitted|no public funds"),
    "oman_resident_card": (r"sultanate of oman|سلطنه عمان", r"civil number|الرقم المدني|resident card|identity card"),
    "eu_residence_permit": (
        r"residence permit|aufenthaltstitel|titre de sejour|permesso di soggiorno|permiso de residencia|titulo de residencia"
        r"|verblijfs(?:titel|vergunning|document)|karta pobytu|povoleni k pobytu|tartozkodasi engedely|opholds(?:tilladelse|kort)"
        r"|uppehallstillstand|oleskelulupa|elamisluba|uzturesanas atlauja|leidimas gyventi|permis de sedere"
        r"|povolenie na pobyt|dovoljenje za prebivanje|dozvola boravka|oppholdstillatelse|aufenthaltsbewilligung|auslanderausweis"
        r"|titre de resident|residence card", None),
}
_ISSUER = {"us_permanent_resident_card": "USA", "uk_residence_permit": "GBR", "oman_resident_card": "OMN"}

# Printed labels, by field. One label line may match several fields.
_LABELS = {
    "name": r"surnames?\W+(?:and\W+)?(?:fore|given)\s*names?|noms?\W+pr.?noms?|namen?\W+vornamen|^name$|^hame$|apellidos|cognome|nazwisko",
    "surname": r"^surnames?$|^nom$|^name/surname$",
    "givenNames": r"^(?:given|fore)\s*names?$|^pr.?noms?$",
    "sex": r"\bsex\b|\bsexe\b|geschlecht|\bsexo\b|\bsesso\b",
    "nationality": r"nationalit|staatsangeh|nacionalidad|cittadinanza|\bnat\b\.?",
    "birthDate": r"date of birth|birth date|geburtsdatum|date de naissance|fecha de nacimiento|data di nascita|تاريخ الميلاد|تارخ ال",
    "expiryDate": r"card expir|valid until|date of expiry|expiry date|gultig bis|valable jusqu|valido hasta|valid to|تاريخ الا?نتهاء|تاريخ الاتهاء",
    "permitType": r"type of permit|art des titels|cat\. du titre|tipo de permiso|tipo di permesso",
    "birthCountry": r"country of birth",
    "category": r"^category$",
    "number": r"uscis\s*#|civil number|الرقم المدني|personal number",
}
_ANY_LABEL = re.compile("|".join(_LABELS.values()) + r"|remarks|observations|anmerkungen|place and date of issue|resident since"
                        r"|place of birth|address|signature|type of permit|leave to|work permitted")
_MONTHS = {m: i for i, m in enumerate("jan feb mar apr may jun jul aug sep oct nov dec".split(), 1)}
_MONTHS.update({"mai": 5, "okt": 10, "dez": 12, "fev": 2, "avr": 4, "juin": 6, "juil": 7, "aou": 8, "ene": 1, "abr": 4,
                "ago": 8, "dic": 12, "gen": 1, "mag": 5, "giu": 6, "lug": 7, "set": 9, "ott": 10})
_DATE = re.compile(r"(?<!\d)(\d{1,2})\s*[ ./\-]\s*(\d{1,2}|[A-Za-z]{3,4})\s*[ ./\-]\s*(\d{4}|\d{2})(?!\d)")
_DATE_PACKED = re.compile(r"(?<!\d)(\d{2})(\d{2})\s+(\d{4})(?!\d)")          # "1412 2024": the first space was not read


def _plain(text: str) -> str:
    """Lower case, accents removed, Arabic letter forms unified: what labels are matched on."""
    text = normalize_arabic(str(text)).replace("ة", "ه")
    text = "".join(ch for ch in unicodedata.normalize("NFKD", text) if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", text).strip().lower()


def _box(line: dict) -> tuple[float, float, float, float]:
    pts = line.get("box") or []
    if len(pts) != 4:
        return (0.0, 0.0, 0.0, 0.0)
    xs, ys = [float(p[0]) for p in pts], [float(p[1]) for p in pts]
    return min(xs), min(ys), max(xs), max(ys)


def kind_of(lines: list[dict]) -> str | None:
    """Which of the supported cards the printed text belongs to, or None."""
    text = " | ".join(_plain(l.get("text", "")) for l in lines)
    for kind, (title, second) in _TITLES.items():
        if re.search(title, text) and (second is None or re.search(second, text)):
            return kind
    return None


def parse_date(text: str, month_first: bool = False, born: bool = False, today: date | None = None) -> str | None:
    """A printed date as YYYY-MM-DD, or None. month_first: US cards print numeric dates as MM/DD/YY."""
    text = to_ascii_digits(str(text))
    m = _DATE.search(text)
    if m:
        a, b, y = m[1], m[2], m[3]
    else:
        m = _DATE_PACKED.search(text)
        if not m:
            return None
        a, b, y = m[1], m[2], m[3]
    year = int(y)
    if len(y) == 2:
        now = (today or date.today()).year
        year += 2000 if (not born and year <= 79) or (born and 2000 + year <= now) else 1900
    if b.isdigit():
        day, month = (int(b), int(a)) if month_first and "/" in m[0] else (int(a), int(b))
    else:
        day, month = int(a), _MONTHS.get(_plain(b)[:4]) or _MONTHS.get(_plain(b)[:3]) or 0
    try:
        return date(year, month, day).isoformat()
    except ValueError:
        return None


class _Page:
    def __init__(self, lines: list[dict]) -> None:
        self.lines = [{**l, "_p": _plain(l.get("text", "")), "_b": _box(l)} for l in lines if str(l.get("text", "")).strip()]
        self.labels = [l for l in self.lines if _ANY_LABEL.search(l["_p"])]
        heights = sorted(l["_b"][3] - l["_b"][1] for l in self.lines if l["_b"][3] > l["_b"][1])
        self.h = heights[len(heights) // 2] if heights else 20.0

    def label(self, field: str) -> list[dict]:
        return [l for l in self.lines if re.search(_LABELS[field], l["_p"])]

    def near(self, label: dict) -> list[tuple[str, dict]]:
        """Text that can be the value of a label, nearest first: the rest of the label's own line, then rows below it that
        overlap it sideways, then text to its right on the same row. Text on one row is joined (a date read in two pieces)."""
        x0, y0, x1, y1 = label["_b"]
        h = max(y1 - y0, self.h * 0.6)
        out: list[tuple[str, dict]] = []
        tail = re.split(r"[:#]\s*", str(label["text"]), maxsplit=1)
        if len(tail) == 2 and tail[1].strip():
            out.append((tail[1].strip(), label))
        below, beside = [], []
        for l in self.lines:
            if l is label or l in self.labels:
                continue
            a0, b0, a1, b1 = l["_b"]
            if b0 >= y1 - 0.45 * h and b0 <= y1 + 3.2 * h and a1 >= x0 - 0.6 * h and a0 <= x1 + 0.6 * h:
                below.append(l)
            elif abs((b0 + b1) / 2 - (y0 + y1) / 2) <= 0.6 * h and (0 <= a0 - x1 <= 14 * h or 0 <= x0 - a1 <= 14 * h):
                beside.append(l)
        rows: list[list[dict]] = []
        for l in sorted(below, key=lambda l: (l["_b"][1], l["_b"][0])):
            if rows and abs(l["_b"][1] - rows[-1][0]["_b"][1]) <= 0.6 * h:
                rows[-1].append(l)
            else:
                rows.append([l])
        for row in rows:
            row.sort(key=lambda l: l["_b"][0])
            if len(row) > 1:
                out.append((" ".join(str(l["text"]) for l in row), row[0]))
            out += [(str(l["text"]), l) for l in row]
        out += [(str(l["text"]), l) for l in sorted(beside, key=lambda l: abs(l["_b"][0] - x1))]
        return out

    def value(self, field: str, accept) -> tuple[object, dict] | None:
        for label in self.label(field):
            for text, line in self.near(label):
                got = accept(text)
                if got:
                    return got, line
        return None


def _word(text: str) -> str | None:
    t = re.sub(r"[^A-Za-zÀ-ɏ' \-]", " ", str(text)).strip()
    return re.sub(r"\s+", " ", t) if len(t.replace(" ", "")) >= 2 else None


def _card_number(page: _Page, kind: str) -> tuple[str, dict] | None:
    """The card's own number. EU cards print it twice (top right, and along the left edge); UK cards print it top right."""
    cands = []
    for l in page.lines:
        t = re.sub(r"[^A-Z0-9]", "", str(l["text"]).upper())
        if 7 <= len(t) <= 12 and re.search(r"\d", t) and re.search(r"[A-Z]", t) and l not in page.labels and "<" not in str(l["text"]):
            cands.append((t, l))
    if not cands:
        return None
    if kind == "uk_residence_permit":
        shaped = [c for c in cands if re.fullmatch(r"[A-Z]{2}[0-9X]\d{6}", c[0])]
        if shaped:
            return shaped[0]
    top = min(l["_b"][1] for l in page.lines)
    span = max(l["_b"][3] for l in page.lines) - top or 1.0
    twice = [c for c in cands if sum(1 for o in cands if sum(a != b for a, b in zip(c[0], o[0])) <= 2 and len(o[0]) == len(c[0])) >= 2]
    pool = twice or [c for c in cands if (c[1]["_b"][1] - top) / span <= 0.2]
    return min(pool, key=lambda c: c[1]["_b"][1]) if pool else None


def parse_permit(raw_lines: list[dict], today: date | None = None) -> dict | None:
    """Read the front of a supported residence card. None when the text is not one of them."""
    kind = kind_of(raw_lines)
    if kind is None:
        return None
    today = today or date.today()
    page = _Page(raw_lines)
    us = kind == "us_permanent_resident_card"
    doc: dict = {"format": "visual", "kind": kind, "documentType": kind}
    scores: dict[str, float] = {}

    def put(field: str, found) -> None:
        if found:
            doc[field] = found[0]
            scores[field] = float(found[1].get("score") or 0.0)

    if _ISSUER.get(kind):
        doc["issuingCountry"] = _ISSUER[kind]
    if us:
        put("documentNumber", page.value("number", lambda t: (re.search(r"\d{3}[\- ]?\d{3}[\- ]?\d{3}", t) or [None])[0]))
        put("surname", page.value("surname", lambda t: (_word(t) or "").upper() or None))
        put("givenNames", page.value("givenNames", lambda t: (_word(t) or "").upper() or None))
        put("birthCountry", page.value("birthCountry", _word))
        put("category", page.value("category", lambda t: (re.fullmatch(r"\s*([A-Z]{1,3}\d{1,2}|[A-Z]\d[A-Z])\s*", t.upper()) or [None, None])[1]))
    elif kind == "oman_resident_card":
        put("idNumber", page.value("number", lambda t: (re.fullmatch(r"\D*(\d{6,9})\D*", to_ascii_digits(t)) or [None, None])[1]))
    else:
        put("documentNumber", _card_number(page, kind))
        name = page.label("name")
        if name:
            words = [(w, l) for t, l in page.near(name[0]) if l is not name[0] for w in [_word(t)] if w and not re.search(r"\d", t)]
            words = [(w, l) for w, l in words if l not in page.labels][:3]
            rows = sorted({id(l): (w, l) for w, l in words}.values(), key=lambda x: x[1]["_b"][1])
            if rows:
                put("surname", (rows[0][0].upper().strip("* "), rows[0][1]))
            if len(rows) > 1:
                put("givenNames", (rows[1][0].upper(), rows[1][1]))
        elif kind == "uk_residence_permit":
            put("surname", _uk_name(page, 0))
            put("givenNames", _uk_name(page, 1))
        put("permitType", page.value("permitType", lambda t: re.sub(r"\s+", " ", t).strip().upper()
                                     if len(t.strip()) >= 2 and not _ANY_LABEL.search(_plain(t)) else None))
        put("personalNumber", page.value("number", lambda t: (re.fullmatch(r"\D*(\d{8,12})\D*", t) or [None, None])[1]))
    put("sex", page.value("sex", lambda t: (re.fullmatch(r"\s*([MFX])\s*", t.upper()) or [None, None])[1]))
    put("nationality", page.value("nationality", lambda t: (re.fullmatch(r"\s*([A-Z]{3})\s*", t.upper()) or [None, None])[1]))
    put("birthDate", page.value("birthDate", lambda t: parse_date(t, False, True, today)))
    put("expiryDate", page.value("expiryDate", lambda t: parse_date(t, us, False, today)))

    flags: list[Flag] = []
    if doc.get("expiryDate") and doc["expiryDate"] < today.isoformat():
        flags.append(Flag("document_expired", "error", f"The card expired on {doc['expiryDate']}."))
    if doc.get("birthDate") and doc["birthDate"] > today.isoformat():
        doc.pop("birthDate")
    key = "idNumber" if kind == "oman_resident_card" else "documentNumber"
    if not doc.get(key):
        flags.append(Flag("document_number_not_found", "error", "The card number could not be read."))
    low = sorted(k for k, v in scores.items() if v and v < 0.9)
    if low:
        flags.append(Flag("low_confidence_fields", "error", "Low OCR confidence for: " + ", ".join(low)))
    flags.append(Flag("visual_zone_only", "error",
                      "This side of the card has no MRZ and no check digit, so the fields come from the printed text only and "
                      "nothing confirms them. A person must review it. The other side of the card carries the MRZ: prefer that side."))
    nat = doc.get("nationality", "")
    doc["nationalityCode"] = nat if re.fullmatch(r"[A-Z]{3}", nat) else ""
    doc["checks"] = {}
    doc["fieldConfidence"] = {k: round(v, 3) for k, v in scores.items() if v}
    doc["correctedFields"] = []
    doc["valid"] = False
    return {"document": doc, "flags": [f.to_dict() for f in flags], "needsReview": True}


def _uk_name(page: _Page, which: int) -> tuple[str, dict] | None:
    """UK cards print the name without a label: the first lines of capitals under the title, above "VALID UNTIL"."""
    title = next((l for l in page.lines if "residence permit" in l["_p"]), None)
    stop = next((l for l in page.lines if re.search(r"valid\s*\w?ntil|place and date", l["_p"])), None)
    if not title or not stop:
        return None
    rows = [l for l in page.lines if title["_b"][3] <= l["_b"][1] < stop["_b"][1] and l not in page.labels
            and re.fullmatch(r"[A-Z][A-Z' \-]+", str(l["text"]).strip()) and l["_b"][0] <= stop["_b"][0] + 6 * page.h
            and l["_b"][0] >= stop["_b"][0] - 2 * page.h]
    rows.sort(key=lambda l: l["_b"][1])
    return (str(rows[which]["text"]).strip(), rows[which]) if len(rows) > which else None


FOUND_FIELDS = ("documentNumber", "idNumber", "surname", "givenNames", "sex", "nationality", "birthDate", "expiryDate")
