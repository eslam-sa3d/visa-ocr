"""ICAO 9303 machine-readable zone (MRZ) parsing and check-digit verification.

Supported formats:
  TD1   3 x 30  ID cards, US green card (back)
  TD2   2 x 36  ID cards, travel documents
  TD3   2 x 44  passports
  MRVA  2 x 44  visa stickers (Schengen visa)
  MRVB  2 x 36  small visa stickers

The MRZ is the same standard for every issuing country, which is why it is the
global baseline for passport extraction.
"""
from __future__ import annotations

from .countries import country_name

import re
from itertools import combinations
from dataclasses import dataclass, field
from datetime import date

FILLER = "<"
_WEIGHTS = (7, 3, 1)

# Visually similar characters that OCR often confuses in positions that must be digits.
_DIGIT_FIX = str.maketrans(
    {"O": "0", "Q": "0", "D": "0", "I": "1", "L": "1", "Z": "2", "S": "5", "B": "8", "G": "6"}
)
_CONFUSE_TO_DIGIT = {"O": "0", "Q": "0", "D": "0", "I": "1", "L": "1", "Z": "2", "S": "5", "B": "8", "G": "6"}
_ALPHA_FIX = str.maketrans({"0": "O", "1": "I", "5": "S", "8": "B", "2": "Z", "6": "G"})
_FILLER_LOOKALIKES = str.maketrans({"«": "<", "‹": "<", "＜": "<"})

# Characters OCR confuses in alphanumeric fields (document number, optional data). Used by the check-digit repair.
_CONFUSE = {"0": "OQD", "O": "0", "Q": "0", "D": "0", "1": "IL", "I": "1", "L": "1", "2": "Z", "Z": "2",
            "5": "S", "S": "5", "8": "B", "B": "8", "6": "G", "G": "6"}


class MrzError(ValueError):
    """Raised when lines cannot be interpreted as a supported MRZ."""


# --------------------------------------------------------------------------- #
# Check digits
# --------------------------------------------------------------------------- #
def _char_value(c: str) -> int:
    if c.isdigit():
        return int(c)
    if "A" <= c <= "Z":
        return ord(c) - 55
    if c == FILLER:
        return 0
    raise MrzError(f"Invalid MRZ character: {c!r}")


def check_digit(data: str) -> int:
    """ICAO 9303 check digit: weights 7,3,1 repeating, sum modulo 10."""
    return sum(_char_value(c) * _WEIGHTS[i % 3] for i, c in enumerate(data)) % 10


# --------------------------------------------------------------------------- #
# Result model
# --------------------------------------------------------------------------- #
@dataclass
class MrzResult:
    format: str
    kind: str  # passport | visa | id_card
    type_code: str
    issuing_state: str
    surname: str
    given_names: str
    document_number: str
    nationality: str
    birth_date: str | None
    sex: str
    expiry_date: str | None
    optional_data: str
    optional_data_2: str
    checks: dict[str, bool]
    corrected_fields: list[str]
    lines: list[str]
    repaired_fields: list[str] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return all(self.checks.values())

    def to_dict(self) -> dict:
        return {
            "format": self.format,
            "kind": self.kind,
            "typeCode": self.type_code,
            "issuingState": self.issuing_state,
            "issuingStateName": country_name(self.issuing_state),
            "surname": self.surname,
            "givenNames": self.given_names,
            "documentNumber": self.document_number,
            "nationality": self.nationality,
            "nationalityName": country_name(self.nationality),
            "birthDate": self.birth_date,
            "sex": self.sex,
            "expiryDate": self.expiry_date,
            "optionalData": self.optional_data,
            "optionalData2": self.optional_data_2,
            "checks": self.checks,
            "correctedFields": self.corrected_fields,
            "repairedFields": self.repaired_fields,
            "valid": self.valid,
        }


@dataclass
class _State:
    checks: dict[str, bool] = field(default_factory=dict)
    corrected: list[str] = field(default_factory=list)
    repaired: list[str] = field(default_factory=list)
    failed_before_repair: set[str] = field(default_factory=set)


def _repair_by_check_digit(data: str, chk: str) -> str | None:
    """Fix OCR look-alike mistakes in an alphanumeric field using its check digit.

    Tries one look-alike substitution, then two. Only an answer that is the single candidate passing the check digit
    is returned; if several candidates pass, or none, returns None (the field stays flagged)."""
    if not chk.isdigit() or check_digit(data) == int(chk):
        return None
    want = int(chk)
    # First assume the OCR turned digits into look-alike letters (O for 0, I for 1 ...), the usual mistake in numbers.
    to_digit = {c: d for c, d in _CONFUSE_TO_DIGIT.items()}
    ls = [(i, to_digit[c]) for i, c in enumerate(data) if c in to_digit]
    cands: set[str] = set()
    for k in range(1, min(len(ls), 3) + 1):
        for combo in combinations(ls, k):
            cand = list(data)
            for i, d in combo:
                cand[i] = d
            cand = "".join(cand)
            if check_digit(cand) == want:
                cands.add(cand)
        if cands:
            break
    if len(cands) == 1:
        return cands.pop()
    slots = [(i, _CONFUSE[c]) for i, c in enumerate(data) if c in _CONFUSE]
    found = {data[:i] + alt + data[i + 1:] for i, alts in slots for alt in alts
             if check_digit(data[:i] + alt + data[i + 1:]) == want}
    if len(found) == 1:
        return found.pop()
    if found:
        return None
    for a in range(len(slots)):
        for b in range(a + 1, len(slots)):
            (i, ai), (j, aj) = slots[a], slots[b]
            for x in ai:
                for y in aj:
                    cand = list(data)
                    cand[i], cand[j] = x, y
                    cand = "".join(cand)
                    if check_digit(cand) == want:
                        found.add(cand)
    return found.pop() if len(found) == 1 else None


_CHECK_KEY = {"documentNumber": "document_number", "optionalData": "optional_data"}


def _tidy_padding(data: str) -> str:
    """Numbers and optional data are padded with '<' only at the end. One or two letters after the first '<' are misread
    fillers (K, C, E, X look like '<'), and the check digit cannot see that for K because the values differ by 20."""
    i = data.find(FILLER)
    if i < 0:
        return data
    tail = data[i:]
    stray = sum(1 for c in tail if c != FILLER)
    if stray == 0 or stray > 2 or len(tail) < 3:
        return data
    return data[:i] + re.sub(r"[^<]", FILLER, tail)


def _repair_field(data: str, chk: str, name: str, st: _State) -> str:
    tidy = _tidy_padding(data)
    if tidy != data:
        if chk.isdigit() and check_digit(data) == int(chk) and check_digit(tidy) != int(chk):
            tidy = data                       # the unchanged text passes its check digit and the tidy one does not: keep it
        else:
            st.corrected.append(f"{name}:stray character in filler removed")
            data = tidy
    fixed = _repair_by_check_digit(data, chk)
    if fixed is None:
        return data
    st.failed_before_repair.add(_CHECK_KEY[name])
    st.repaired.append(f"{name}:{data.rstrip(FILLER)}->{fixed.rstrip(FILLER)}")
    return fixed


def _finish(st: _State) -> dict:
    """A repaired field did not pass its check digit as read, so it stays marked as failed. The repair is only a suggestion."""
    for key in st.failed_before_repair:
        st.checks[key] = False
    return st.checks


def _overflow(doc9: str, chk: str, opt: str) -> tuple[str, str, str]:
    """ICAO long document number (TD1 and TD2): the check position holds '<', the number continues at the start of the
    optional data, and the real check digit is the last character of that continuation, followed by '<'."""
    if chk != FILLER:
        return doc9, chk, opt
    m = re.match(r"([A-Z0-9]{2,})(?:<|$)", opt)
    if not m:
        return doc9, chk, opt
    ov = m.group(1)
    return doc9.rstrip(FILLER) + ov[:-1], ov[-1].translate(_DIGIT_FIX), opt[len(ov) + 1:]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def clean_line(line: str) -> str:
    """Upper-case, drop whitespace, normalise filler look-alikes."""
    return re.sub(r"\s+", "", line.upper()).translate(_FILLER_LOOKALIKES)


def _fix(line: str, positions: list[int], line_no: int, st: _State) -> str:
    """Correct OCR letter/digit confusion at positions that must be numeric."""
    chars = list(line)
    for i in positions:
        fixed = chars[i].translate(_DIGIT_FIX)
        if fixed != chars[i]:
            st.corrected.append(f"line{line_no}[{i}]:{chars[i]}->{fixed}")
            chars[i] = fixed
    return "".join(chars)


def _fix_alpha(line: str, start: int, end: int, line_no: int, st: _State) -> str:
    """Digits cannot appear in names or country codes: turn look-alike digits back into letters."""
    chars = list(line)
    for i in range(start, min(end, len(chars))):
        fixed = chars[i].translate(_ALPHA_FIX)
        if fixed != chars[i]:
            st.corrected.append(f"line{line_no}[{i}]:{chars[i]}->{fixed}")
            chars[i] = fixed
    return "".join(chars)


def _tidy_names(line: str, start: int, line_no: int, st: _State) -> str:
    """After the last name there are only fillers. A few stray letters inside that run are misread fillers."""
    field_ = line[start:]
    # A long filler run read as letters ("JIANG" + "<" * 20 -> "JLANGKSSKKKKKKKKKK"): letters that look like "<" and fill the whole
    # tail are fillers. Only runs of 6 or more such letters at the end of the name field are treated this way.
    # A real name is followed by fillers; a misread filler run reaches the very end of the line. So a run that is followed by
    # '<' is left alone (LESLIE, GISELLE, CECILE are names), and a removal is always reported for a person to confirm.
    m = re.search(r"[KSECLXI]{6,}$", field_)
    if m:
        st.corrected.append(f"line{line_no}[{start + m.start()}..]:filler read as letters removed ({m[0][:12]})")
        line = line[:start + m.start()] + FILLER * (len(field_) - m.start())
        field_ = line[start:]
    i = field_.find("<<<")
    if i < 0:
        return line
    tail = field_[i:]
    stray = sum(1 for c in tail if c != FILLER)
    if stray == 0 or stray > 3:
        return line
    st.corrected.append(f"line{line_no}[{start + i}..]:stray letters in filler removed")
    return line[:start + i] + re.sub(r"[^<]", FILLER, tail)


def _verify(data: str, chk: str, name: str, st: _State, allow_empty: bool = False) -> None:
    if allow_empty and set(data) <= {FILLER} and chk in (FILLER, "0"):
        st.checks[name] = True
        return
    if not chk.isdigit():
        st.checks[name] = False
        return
    st.checks[name] = check_digit(data) == int(chk)


def _parse_date(yymmdd: str, role: str, today: date) -> str | None:
    if not re.fullmatch(r"\d{6}", yymmdd):
        return None
    yy, mm, dd = int(yymmdd[:2]), int(yymmdd[2:4]), int(yymmdd[4:6])
    if role == "birth":
        # Two-digit year: anything later than the current year is the previous century.
        year = 1900 + yy if yy > today.year % 100 else 2000 + yy
    else:
        year = 2000 + yy
    try:
        return date(year, mm, dd).isoformat()
    except ValueError:
        return None


def _split_names(field_value: str) -> tuple[str, str]:
    primary, _, secondary = field_value.partition("<<")
    squash = lambda s: re.sub(r" +", " ", s.replace(FILLER, " ")).strip()  # noqa: E731
    return squash(primary), squash(secondary)


def _kind(type_code: str) -> str:
    first = type_code[:1]
    if first == "P":
        return "passport"
    if first == "V":
        return "visa"
    return "id_card"


def _code(raw: str) -> str:
    return raw.strip(FILLER)


# --------------------------------------------------------------------------- #
# Format parsers
# --------------------------------------------------------------------------- #
def _two_line_common(l2: str, st: _State, today: date, doc: tuple[str, str] | None = None) -> dict:
    """Shared line-2 layout for TD3, MRVA, TD2, MRVB (first 28 characters). `doc` overrides (number, check digit)."""
    doc_no, doc_chk = doc if doc else (l2[0:9], l2[9])
    _verify(doc_no, doc_chk, "document_number", st)
    dob = l2[13:19]
    _verify(dob, l2[19], "birth_date", st)
    exp = l2[21:27]
    _verify(exp, l2[27], "expiry_date", st)
    return {
        "document_number": doc_no.rstrip(FILLER),
        "nationality": _code(l2[10:13]),
        "birth_date": _parse_date(dob, "birth", today),
        "sex": l2[20],
        "expiry_date": _parse_date(exp, "expiry", today),
    }


_TWO_LINE_NUMERIC = [9, *range(13, 20), *range(21, 28)]


def _parse_td3(lines: list[str], today: date) -> MrzResult:
    st = _State()
    l1 = _tidy_names(_fix_alpha(lines[0], 2, 44, 1, st), 5, 1, st)
    l2 = _fix_alpha(_fix(lines[1], [*_TWO_LINE_NUMERIC, 42, 43], 2, st), 10, 13, 2, st)
    l2 = l2[:0] + _repair_field(l2[0:9], l2[9], "documentNumber", st) + l2[9:]
    if l2[42].isdigit():
        l2 = l2[:28] + _repair_field(l2[28:42], l2[42], "optionalData", st) + l2[42:]
    f = _two_line_common(l2, st, today)
    _verify(l2[28:42], l2[42], "optional_data", st, allow_empty=True)
    _verify(l2[0:10] + l2[13:20] + l2[21:43], l2[43], "composite", st)
    surname, given = _split_names(l1[5:44])
    return MrzResult("TD3", _kind(l1[0:2]), _code(l1[0:2]), _code(l1[2:5]), surname, given,
                     optional_data=l2[28:42].rstrip(FILLER), optional_data_2="",
                     checks=_finish(st), corrected_fields=st.corrected, lines=[l1, l2], repaired_fields=st.repaired, **f)


def _parse_mrva(lines: list[str], today: date) -> MrzResult:
    st = _State()
    l1 = _tidy_names(_fix_alpha(lines[0], 2, 44, 1, st), 5, 1, st)
    l2 = _fix_alpha(_fix(lines[1], _TWO_LINE_NUMERIC, 2, st), 10, 13, 2, st)
    l2 = _repair_field(l2[0:9], l2[9], "documentNumber", st) + l2[9:]
    f = _two_line_common(l2, st, today)
    surname, given = _split_names(l1[5:44])
    return MrzResult("MRVA", "visa", _code(l1[0:2]), _code(l1[2:5]), surname, given,
                     optional_data=l2[28:44].rstrip(FILLER), optional_data_2="",
                     checks=_finish(st), corrected_fields=st.corrected, lines=[l1, l2], repaired_fields=st.repaired, **f)


def _parse_td2(lines: list[str], today: date) -> MrzResult:
    st = _State()
    l1 = _tidy_names(_fix_alpha(lines[0], 2, 36, 1, st), 5, 1, st)
    l2 = _fix_alpha(_fix(lines[1], [*_TWO_LINE_NUMERIC, 35], 2, st), 10, 13, 2, st)
    overflowed = l2[9] == FILLER
    doc_no, chk, opt = _overflow(l2[0:9], l2[9], l2[28:35])
    if chk.isdigit():
        fixed = _repair_field(doc_no, chk, "documentNumber", st)
        if fixed != doc_no:
            l2 = fixed[:9].ljust(9, FILLER) + l2[9:]
            if overflowed:
                tail = fixed[9:]
                l2 = l2[:28] + tail + l2[28 + len(tail):]
            doc_no = fixed
    f = _two_line_common(l2, st, today, doc=(doc_no, chk))
    f["document_number"] = doc_no.rstrip(FILLER)
    _verify(l2[0:10] + l2[13:20] + l2[21:35], l2[35], "composite", st)
    surname, given = _split_names(l1[5:36])
    return MrzResult("TD2", _kind(l1[0:2]), _code(l1[0:2]), _code(l1[2:5]), surname, given,
                     optional_data=opt.rstrip(FILLER), optional_data_2="",
                     checks=_finish(st), corrected_fields=st.corrected, lines=[l1, l2], repaired_fields=st.repaired, **f)


def _parse_mrvb(lines: list[str], today: date) -> MrzResult:
    st = _State()
    l1 = _tidy_names(_fix_alpha(lines[0], 2, 36, 1, st), 5, 1, st)
    l2 = _fix_alpha(_fix(lines[1], _TWO_LINE_NUMERIC, 2, st), 10, 13, 2, st)
    l2 = _repair_field(l2[0:9], l2[9], "documentNumber", st) + l2[9:]
    f = _two_line_common(l2, st, today)
    surname, given = _split_names(l1[5:36])
    return MrzResult("MRVB", "visa", _code(l1[0:2]), _code(l1[2:5]), surname, given,
                     optional_data=l2[28:36].rstrip(FILLER), optional_data_2="",
                     checks=_finish(st), corrected_fields=st.corrected, lines=[l1, l2], repaired_fields=st.repaired, **f)


def _parse_td1(lines: list[str], today: date) -> MrzResult:
    st = _State()
    l1 = _fix_alpha(_fix(lines[0], [14], 1, st), 2, 5, 1, st)
    l2 = _fix_alpha(_fix(lines[1], [*range(0, 7), *range(8, 15), 29], 2, st), 15, 18, 2, st)
    l3 = _tidy_names(_fix_alpha(lines[2], 0, 30, 3, st), 0, 3, st)

    doc_no, chk, opt1 = _overflow(l1[5:14], l1[14], l1[15:30])
    if chk.isdigit():
        fixed = _repair_field(doc_no, chk, "documentNumber", st)
        if fixed != doc_no:
            head = fixed[:9].ljust(9, FILLER)
            tail = fixed[9:]
            l1 = l1[:5] + head + l1[14:]
            if tail:
                l1 = l1[:15] + tail + l1[15 + len(tail):]
            doc_no = fixed
    _verify(doc_no, chk, "document_number", st)

    dob, exp = l2[0:6], l2[8:14]
    _verify(dob, l2[6], "birth_date", st)
    _verify(exp, l2[14], "expiry_date", st)
    _verify(l1[5:30] + l2[0:7] + l2[8:15] + l2[18:29], l2[29], "composite", st)

    surname, given = _split_names(l3)
    return MrzResult("TD1", _kind(l1[0:2]), _code(l1[0:2]), _code(l1[2:5]), surname, given,
                     document_number=doc_no.rstrip(FILLER), nationality=_code(l2[15:18]),
                     birth_date=_parse_date(dob, "birth", today), sex=l2[7],
                     expiry_date=_parse_date(exp, "expiry", today),
                     optional_data=opt1.rstrip(FILLER), optional_data_2=l2[18:29].rstrip(FILLER),
                     checks=_finish(st), corrected_fields=st.corrected, lines=[l1, l2, l3], repaired_fields=st.repaired)


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def parse_mrz(lines: list[str], today: date | None = None) -> MrzResult:
    """Parse MRZ lines (as read by OCR) into fields with check-digit results."""
    today = today or date.today()
    cleaned = [clean_line(l) for l in lines if l and l.strip()]
    sizes = tuple(len(l) for l in cleaned)
    for n, line in enumerate(cleaned, start=1):
        bad = sorted({c for c in line if not re.fullmatch(r"[A-Z0-9<]", c)})
        if bad:
            raise MrzError(
                f"MRZ line {n} contains characters that are not A-Z, 0-9 or '<': "
                + " ".join(repr(c) for c in bad)
                + ". Type the filler with the '<' key (Shift + comma on a US keyboard)."
            )

    if len(cleaned) == 3 and sizes == (30, 30, 30):
        return _parse_td1(cleaned, today)
    if len(cleaned) == 2 and sizes == (44, 44):
        return _parse_mrva(cleaned, today) if cleaned[0][0] == "V" else _parse_td3(cleaned, today)
    if len(cleaned) == 2 and sizes == (36, 36):
        return _parse_mrvb(cleaned, today) if cleaned[0][0] == "V" else _parse_td2(cleaned, today)
    raise MrzError(f"Unsupported MRZ shape: {len(cleaned)} lines, lengths {sizes}")


def _fit_length(line: str, size: int, max_diff: int = 3) -> str | None:
    """Real-world OCR often drops or adds a filler '<'. Fix the length by growing or shrinking the longest run of
    '<' (the optional-data / name filler). The check digits later decide whether the repaired line is trustworthy."""
    diff = size - len(line)
    if diff == 0:
        return line
    if abs(diff) > max_diff:
        return None
    runs = list(re.finditer(r"<+", line))
    if not runs:
        return None
    run = max(runs, key=lambda m: m.end() - m.start())
    new_len = (run.end() - run.start()) + diff
    if new_len < 1:
        return None
    return line[: run.start()] + "<" * new_len + line[run.end():]


def lines_confidence(lines: list[str] | None, texts: list[str], scores: list) -> float | None:
    """Lowest OCR score over the detected texts that the returned MRZ lines came from. None when any returned line has no
    detected text that reads exactly the same (a line whose filler run was padded or adjusted was partly made up, so no score
    speaks for it), or when a score is missing."""
    if not lines:
        return None
    by_text: dict[str, float] = {}
    for t, s in zip(texts, scores):
        if s is None:
            continue
        c = clean_line(str(t))
        if c:
            by_text[c] = min(float(s), by_text.get(c, 1.0))
    used = [by_text.get(l) for l in lines]
    if any(u is None for u in used):
        return None
    return round(min(used), 4)


def find_mrz_lines(candidate_lines: list[str]) -> list[str] | None:
    """Locate a block of MRZ lines inside raw OCR output. Exact lengths first, then lines that are a few
    characters off (repaired by adjusting the filler run)."""
    cleaned = [clean_line(l) for l in candidate_lines]
    cleaned = [l for l in cleaned if l]
    # stray words (a stamp, a name) can sit between the MRZ lines in OCR order: look at MRZ-like lines only first
    mrz_like = [l for l in cleaned if len(l) >= 26 and re.fullmatch(r"[A-Z0-9<]+", l) and (FILLER in l or len(l) in (30, 36, 44))]
    if mrz_like != cleaned:
        found = find_mrz_lines(mrz_like)
        if found:
            return found
    for size, count in ((30, 3), (44, 2), (36, 2)):
        for i in range(len(cleaned) - count + 1):
            window = cleaned[i : i + count]
            # a data line can be completely full and hold no '<' at all, so one filler in the block is enough
            if all(len(l) == size and re.fullmatch(r"[A-Z0-9<]+", l) for l in window) and any(FILLER in l for l in window):
                return window
    for size, count in ((44, 2), (36, 2), (30, 3)):
        for i in range(len(cleaned) - count + 1):
            window = cleaned[i : i + count]
            if not all(re.fullmatch(r"[A-Z0-9<]+", l) and FILLER in l and abs(len(l) - size) <= 3 for l in window):
                continue
            fitted = [_fit_length(l, size) for l in window]
            if all(f is not None for f in fitted) and (size != 44 or window[0][0] in "PV" or window[0][0] == "I"):
                return [f for f in fitted if f is not None]
    # Line 1 (type, country, names) is mostly filler, and OCR often loses part of that long run. When line 2 is exactly the right
    # length and line 1 starts like an MRZ and still has its name and some filler, restore line 1's filler. Line 1 has no check
    # digit and its names are never trusted without a second reading, so a restored tail cannot approve anything by itself.
    for size in (44, 36):
        for i in range(len(cleaned) - 1):
            first, second = cleaned[i], cleaned[i + 1]
            if (re.fullmatch(r"[A-Z0-9<]+", first) and len(second) == size and re.fullmatch(r"[A-Z0-9<]+", second)
                    and 12 <= len(first) < size - 3 and first[0] in "PVIAC" and re.search(r"<<<", first)
                    and not re.search(r"[0-9]{4,}", first)):
                return [first.ljust(size, FILLER), second]
    return None
