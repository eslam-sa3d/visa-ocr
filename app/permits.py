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

from . import config
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
    "name": r"surnames?\W+(?:and\W+)?(?:fore|given)\s*names?|noms?\W+pr.?noms?|namen?\W+vornamen|^name$|^hame$|apellidos|cognom[ei]|nazwisko"
            r"|efternamn|efternavn|etterna[vm]n|prijmeni|priezvisko|vezeteknev|perekonnanimi|apelidos|sukunimi|achternaam|uzvards|pavarde|prezime|priimek",
    "surname": r"^surnames?$|^nom$|^name/surname$",
    "givenNames": r"^(?:given|fore)\s*names?$|^pr.?noms?$",
    "sex": r"\bsex\b|\bsexe\b|geschlecht|\bsexo\b|\bsesso\b|^kon$|^kjonn$|^nem$|pohlavi|^sugu\b|^spol$|^plec$|sukupuoli|geslacht|^dzimums$|^lytis$",
    "nationality": r"nationalit|staatsangeh|nacionalidad|cittadinanza|\bnat\b\.?|nasjonalitet|kodakondsus|citizenship|statni prislusnost"
                   r"|statna prislusnost|allampolgarsag|obywatelstwo|kansalaisuus|drzavljanstvo|pilsoniba|pilietybe|cetatenia",
    "birthDate": r"date of birth|birth date|geburtsdatum|date de naissance|fecha de nacimiento|data di nascita|تاريخ الميلاد|تارخ ال"
                 r"|fodelsedatum|fodselsdato|fodselsdag|datum narozeni|datum narodenia|szuletesi|data de nascimento|data nasci|syntymaaika"
                 r"|geboortedatum|sunniaeg|data urodzenia|datum rojstva|datum rodenja|data nasterii",
    "expiryDate": r"card expir|valid until|date of expiry|expiry date|gultig bis|valable jusqu|valido hasta|valid to|تاريخ الا?نتهاء|تاريخ الاتهاء"
                  r"|giltighetsdag|gyldig til|platnost do|ervenyesseg|validade|scadenza|kehtiv kuni|voimassa|geldig tot|waznosc|veljavnost",
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


_NOT_A_CARD_FRONT = re.compile(r"passport|passeport|pasaporte|passaporte|reisepass|paspoort|paszport|جواز")


def says_it_is_a_passport(lines: list[dict]) -> bool:
    """True when the printed text names the document a passport (in the languages seen on passports, Arabic included)."""
    # "Passport No." on a residence card is a field label, not the document's title
    return any(_NOT_A_CARD_FRONT.search(t) and not re.search(r"\bno\b|number|n°|\bnr\b|رقم", t)
               for t in (str(l.get("text", "")).lower() for l in lines))


def kind_of(lines: list[dict]) -> str | None:
    """Which of the supported cards the printed text belongs to, or None. A page that says it is a passport, or that shows
    part of an MRZ (lines of capitals with '<' fillers), is never a card front: its MRZ was simply not read, and the right
    answer is another photo, not a reading without check digits."""
    text = " | ".join(_plain(l.get("text", "")) for l in lines)
    if _NOT_A_CARD_FRONT.search(text):
        return None
    for l in lines:
        raw = re.sub(r"\s+", "", str(l.get("text", "")))
        if len(raw) >= 20 and raw.count("<") >= 3:
            return None
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
        if 7 <= len(t) <= 12 and re.search(r"\d", t) and l not in page.labels and "<" not in str(l["text"]) and not _DATE.search(str(l["text"])):
            cands.append((t, l))
    # a number of digits only is the card's number only when it is printed twice, as the card's number is
    cands = [c for c in cands if re.search(r"[A-Z]", c[0]) or sum(1 for o in cands if o[0] == c[0]) >= 2]
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
    put("sex", page.value("sex", _sex))
    put("nationality", page.value("nationality", lambda t: (re.fullmatch(r"\s*([A-Z]{3})\s*", t.upper()) or [None, None])[1]))
    put("birthDate", page.value("birthDate", lambda t: parse_date(t, False, True, today)))
    put("expiryDate", page.value("expiryDate", lambda t: parse_date(t, us, False, today)))

    if kind == "eu_residence_permit":
        _by_layout(page, doc, scores, today)

    flags: list[Flag] = []
    if doc.get("expiryDate") and doc["expiryDate"] < today.isoformat():
        flags.append(Flag("document_expired", "error", f"The card expired on {doc['expiryDate']}."))
    if doc.get("birthDate") and doc["birthDate"] > today.isoformat():
        doc.pop("birthDate")
    key = "idNumber" if kind == "oman_resident_card" else "documentNumber"
    if not doc.get(key):
        flags.append(Flag("document_number_not_found", "error", "The card number could not be read."))
    low = sorted(k for k, v in scores.items() if v and v < config.min_confidence_visual())
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
    look = doc.pop("_lookAgain", None) or None     # where a missed field is printed, for a second look by the reader
    return {"document": doc, "flags": [f.to_dict() for f in flags], "needsReview": True, **({"lookAgain": look} if look else {})}


def sex_letter(text: str) -> str | None:
    return _sex(text)


_SEX_LETTERS = {"M": "M", "F": "F", "X": "X", "K": "F", "W": "F", "V": "F", "H": "M"}     # K kvinna, W weiblich, V vrouw, H homme
_SEX_WORDS = {"MOT": "F", "VYR": "M"}                                                       # Lithuanian moteris, vyras


def _sex(text: str) -> str | None:
    """The sex as M, F or X. Cards print it in the national language and in English ("K/F", "M/M", "Α/M"): the English
    letter is the one after the stroke. A single national letter is translated where it can mean only one thing."""
    t = str(text).strip().upper()
    m = re.fullmatch(r"\W?\w\s*/\s*([MFX])", t)
    if m:
        return m[1]
    return _SEX_LETTERS.get(t) if len(t) == 1 else _SEX_WORDS.get(t)


_PURE_DATE = re.compile(r"\s*\d{1,2}\s*[ ./\-]\s*\d{1,2}\s*[ ./\-]\s*\d{4}\s*")
_NAME_LINE = re.compile(r"[A-Za-zÀ-ɏ][A-Za-zÀ-ɏ' \-]+")


def _by_layout(page: _Page, doc: dict, scores: dict, today: date) -> None:
    """The EU uniform-format card puts every field in the same place, whatever language its labels are in, and the small
    labels are the part most often misread. So what the labels did not give is taken from the layout: the date of birth is
    the first date on the card; on its row, to the left, stand the sex and the three-letter nationality; the next date below
    is the card's expiry; the two lines above that row's labels are the forenames and, above them, the surname."""
    dates = [l for l in page.lines if _PURE_DATE.fullmatch(str(l["text"])) and parse_date(l["text"])]
    # A date read in pieces ("11 03" and "1990", or three boxes): the pieces of one row are put together again.
    pieces = sorted((l for l in page.lines if re.fullmatch(r"[\d ./\-]+", str(l["text"]).strip()) and l not in dates
                     and len(re.sub(r"\D", "", str(l["text"]))) <= 6), key=lambda l: (l["_b"][1], l["_b"][0]))
    while pieces:
        first = pieces[0]
        tall = max(1.0, first["_b"][3] - first["_b"][1])
        group = sorted((l for l in pieces if abs(l["_b"][1] - first["_b"][1]) <= 0.5 * tall), key=lambda l: l["_b"][0])
        pieces = [l for l in pieces if not any(l is g for g in group)]
        text = " ".join(str(l["text"]).strip() for l in group)
        if len(group) > 1 and _PURE_DATE.fullmatch(text) and parse_date(text):
            box = (min(l["_b"][0] for l in group), min(l["_b"][1] for l in group), max(l["_b"][2] for l in group), max(l["_b"][3] for l in group))
            dates.append({"text": text, "score": min(float(l.get("score") or 0.0) for l in group), "image": group[0].get("image"),
                          "_b": box, "_p": text})
    dates.sort(key=lambda l: l["_b"][1])
    if not dates:
        return
    anchor = dates[0]
    born = parse_date(anchor["text"], False, True, today)
    if not born or born > today.isoformat():
        return
    x0, y0, x1, y1 = anchor["_b"]
    h, mid = max(1.0, y1 - y0), (y0 + y1) / 2

    def put(field: str, value, line: dict) -> None:
        doc[field] = value
        scores[field] = float(line.get("score") or 0.0)

    if not doc.get("birthDate"):
        put("birthDate", born, anchor)
    if not doc.get("expiryDate"):
        later = [l for l in dates[1:] if l["_b"][1] > y0 + 0.5 * h and (parse_date(l["text"]) or "") > born]
        if later:
            put("expiryDate", parse_date(later[0]["text"]), later[0])
    row = [l for l in page.lines if l is not anchor and l not in page.labels and l["_b"][2] <= x0 + 0.5 * h
           and abs((l["_b"][1] + l["_b"][3]) / 2 - mid) <= 0.5 * h]
    for l in sorted(row, key=lambda l: abs((l["_b"][1] + l["_b"][3]) / 2 - mid)):
        text = str(l["text"]).strip()
        if not doc.get("nationality") and re.fullmatch(r"[A-Z]{3}", text) and not _sex(text):
            put("nationality", text, l)
        elif not doc.get("sex") and _sex(text):
            put("sex", _sex(text), l)
    looks: list[dict] = []
    if not doc.get("nationality"):
        # a code read with a letter too many or too few ("GGBR"): that spot is read again on its own, enlarged
        near = [l for l in row if re.fullmatch(r"[A-Z]{2,5}", str(l["text"]).strip()) and not _sex(str(l["text"]))]
        if near:
            b = near[0]["_b"]
            looks.append({"field": "nationality", "image": int(anchor.get("image") or 0),
                          "box": [b[0] - 0.5 * h, b[1] - 0.4 * h, b[2] + 0.5 * h, b[3] + 0.4 * h]})
    doc["_lookAgain"] = looks
    if not row:
        return
    # Names: the nearest lines of letters above this row's labels, in the same column. The surname is printed in capitals,
    # the forenames in mixed case, on one line or two.
    left = min(l["_b"][0] for l in row)
    above = [l for l in page.lines if l["_b"][3] <= y0 - 0.6 * h and l["_b"][1] >= y0 - 12 * h and left - 8 * h <= l["_b"][0] <= left + 2 * h
             and _NAME_LINE.fullmatch(str(l["text"]).strip()) and l not in page.labels]
    above.sort(key=lambda l: -l["_b"][1])                     # nearest first
    if not above:
        return
    first = str(above[0]["text"]).strip()
    words = first.split()
    caps = [w for w in words if w.isupper()]
    if caps and len(caps) < len(words) and words[:len(caps)] == caps:          # "SURNAME Forenames" on one line
        surname, given, s_line, g_line = " ".join(caps), " ".join(words[len(caps):]), above[0], above[0]
    elif len(above) > 1 and above[0]["_b"][1] - above[1]["_b"][3] <= 1.5 * h:
        surname, given, s_line, g_line = str(above[1]["text"]).strip(), first, above[1], above[0]
    else:
        surname, given, s_line, g_line = first, None, above[0], None
    if not surname.isupper() and doc.get("surname"):
        return                                 # the line above is not a surname in capitals: keep what the label gave
    # A long surname runs over two lines: a line of capitals directly above belongs to it.
    higher = [l for l in above if l["_b"][3] <= s_line["_b"][1] + 0.3 * h]
    if higher and str(higher[0]["text"]).strip().isupper() and "/" not in str(higher[0]["text"]) \
            and s_line["_b"][1] - higher[0]["_b"][3] <= 0.8 * h:
        surname = str(higher[0]["text"]).strip() + " " + surname
    put("surname", surname.upper().strip("* "), s_line)
    if given:
        put("givenNames", given.upper(), g_line)
    elif "givenNames" in doc and doc.get("givenNames") == doc.get("surname"):
        doc.pop("givenNames")
    if not doc.get("sex"):
        # The sex is one letter at the start of this row, under the names' left edge, and one letter is easily missed on a
        # whole page. Say where it is: the reader looks at that spot again, enlarged (pipeline, _look_again).
        column = s_line["_b"][0]
        right = min((l["_b"][0] for l in row if l["_b"][0] > column + 1.5 * h), default=column + 6 * h)
        looks.append({"field": "sex", "image": int(anchor.get("image") or 0),
                      "box": [column - 0.6 * h, mid - 0.9 * h, right - 0.2 * h, mid + 0.9 * h]})


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
