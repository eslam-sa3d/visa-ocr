#!/usr/bin/env python3
"""Independent accuracy matrix for the MRZ layer (no OCR, no models).

What it measures, using its OWN check-digit implementation (so it is not grading the parser with the parser's code):
  1. ICAO 9303 published specimens          -> must parse exactly
  2. Country x format matrix                -> every country code in app/countries.py in every MRZ format
  3. Name, date, document-number edge cases -> truncation, hyphens, long numbers, century pivot, unknown dates
  4. Error injection (OCR-style mistakes)   -> how many bad reads are caught, and how many are SILENTLY wrong
  5. Missing / extra filler '<' characters  -> the most common real OCR fault

    python tools/accuracy_matrix.py [--seed 1] [--n-errors 3000] [--json out.json]

Every document is fictional. This measures the TEXT layer only. It says nothing about how well the OCR engine reads a
real photo; for that use tools/run_scenarios.py and real specimens.
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.countries import COUNTRIES  # noqa: E402
from app.mrz import MrzError, find_mrz_lines, parse_mrz  # noqa: E402
from app.validation import validate  # noqa: E402

TODAY = date(2026, 11, 9)
W = (7, 3, 1)


# ----------------------------------------------------------------------------- independent oracle
def val(c: str) -> int:
    return int(c) if c.isdigit() else 0 if c == "<" else ord(c) - 55


def cd(s: str) -> str:
    return str(sum(val(c) * W[i % 3] for i, c in enumerate(s)) % 10)


def pad(s: str, n: int) -> str:
    return (s + "<" * n)[:n]


def names_field(surname: str, given: str, n: int) -> str:
    s = surname.replace("-", "<").replace(" ", "<").replace("'", "")
    g = given.replace("-", "<").replace(" ", "<").replace("'", "")
    return pad(s + ("<<" + g if g else "<<"), n)


def squash(x: str) -> str:
    return " ".join(x.replace("<", " ").split())


def build(fmt, *, tcode, issuing, nat, surname, given, doc, dob, sex, exp, opt="", opt2="", opt_check_zero=False):
    """Return (lines, truth) for a document in the given format, with correct check digits."""
    over = ""
    doc9 = doc
    if fmt in ("TD1", "TD2") and len(doc) > 9:
        doc9, over = doc[:9], doc[9:]
    if fmt in ("TD1", "TD2"):        # optional data shares its field with the overflow, so only what fits can be on the document
        room = (15 if fmt == "TD1" else 7) - ((len(over) + 2) if over else 0)
        opt = opt[:max(room, 0)]
    d9 = pad(doc9, 9)
    dchk = "<" if over else cd(d9)
    if fmt == "TD3":
        l1 = pad(tcode, 2) + pad(issuing, 3) + names_field(surname, given, 39)
        optf = pad(opt, 14)
        ochk = ("0" if opt_check_zero else "<") if not opt else cd(optf)
        body = d9 + cd(d9) + pad(nat, 3) + dob + cd(dob) + sex + exp + cd(exp) + optf + ochk
        l2 = body + cd(body[0:10] + body[13:20] + body[21:43])
    elif fmt == "MRVA":
        l1 = pad(tcode, 2) + pad(issuing, 3) + names_field(surname, given, 39)
        l2 = d9 + cd(d9) + pad(nat, 3) + dob + cd(dob) + sex + exp + cd(exp) + pad(opt, 16)
    elif fmt == "MRVB":
        l1 = pad(tcode, 2) + pad(issuing, 3) + names_field(surname, given, 31)
        l2 = d9 + cd(d9) + pad(nat, 3) + dob + cd(dob) + sex + exp + cd(exp) + pad(opt, 8)
    elif fmt == "TD2":
        l1 = pad(tcode, 2) + pad(issuing, 3) + names_field(surname, given, 31)
        optf = pad((over + cd(doc) + "<" if over else "") + opt, 7)
        body = d9 + dchk + pad(nat, 3) + dob + cd(dob) + sex + exp + cd(exp) + optf
        l2 = body + cd(body[0:10] + body[13:20] + body[21:35])
    elif fmt == "TD1":
        optf1 = pad((over + cd(doc) + "<" if over else "") + opt, 15)
        l1 = pad(tcode, 2) + pad(issuing, 3) + d9 + dchk + optf1
        o2 = pad(opt2, 11)
        l2a = dob + cd(dob) + sex + exp + cd(exp) + pad(nat, 3) + o2
        comp = cd(l1[5:30] + l2a[0:7] + l2a[8:15] + l2a[18:29])
        l2 = l2a + comp
        l3 = names_field(surname, given, 30)
        return [l1, l2, l3], truth_of(fmt, tcode, issuing, nat, l3, doc, dob, sex, exp, opt, opt2)
    else:
        raise ValueError(fmt)
    nf = l1[5:]
    return [l1, l2], truth_of(fmt, tcode, issuing, nat, nf, doc, dob, sex, exp, opt, opt2)


def iso(yymmdd: str, role: str) -> str:
    yy, mm, dd = int(yymmdd[:2]), int(yymmdd[2:4]), int(yymmdd[4:])
    year = (1900 + yy if yy > TODAY.year % 100 else 2000 + yy) if role == "birth" else 2000 + yy
    return date(year, mm, dd).isoformat()


def truth_of(fmt, tcode, issuing, nat, namefield, doc, dob, sex, exp, opt, opt2):
    sur, _, giv = namefield.partition("<<")
    return {
        "typeCode": tcode.strip("<"), "issuingState": issuing.strip("<"), "surname": squash(sur),
        "givenNames": squash(giv), "documentNumber": doc.strip("<"), "nationality": nat.strip("<"),
        "birthDate": iso(dob, "birth"), "sex": sex, "expiryDate": iso(exp, "expiry"),
        "optionalData": opt.strip("<"), "optionalData2": opt2.strip("<"),
    }


# ----------------------------------------------------------------------------- comparison
FIELDS = ["typeCode", "issuingState", "surname", "givenNames", "documentNumber", "nationality", "birthDate", "sex",
          "expiryDate", "optionalData", "optionalData2"]
PROTECTED = {"documentNumber", "birthDate", "expiryDate"}      # own check digit (optionalData: TD3 only; see report)


def read(lines):
    """What the service does with MRZ text: parse + validate. Returns (doc dict | None, error flags, all flag codes)."""
    try:
        r = parse_mrz(lines, TODAY)
    except MrzError as e:
        return None, [f"unparsed:{str(e)[:40]}"], []
    fl = validate(r, TODAY)
    d = r.to_dict()
    return d, [f.code for f in fl if f.severity == "error"], [f.code for f in fl]


def diff(d, truth):
    return [k for k in FIELDS if (d.get(k) or "") != (truth[k] or "")]


# ----------------------------------------------------------------------------- data pools
SURNAMES = ["ERIKSSON", "AL-HARBI", "O'BRIEN", "VAN DER BERG", "MUELLER", "NGUYEN", "DE LA CRUZ", "SMITH", "KIM", "ABDELRAHMAN",
            "SANCHEZ-GARCIA", "ZHANG", "ALMUTAIRI", "IBRAHIM", "KOWALCZYK-NOWAK", "D'ANGELO", "BIN LADEN", "SINGH", "PETROVA", "X"]
GIVENS = ["ANNA MARIA", "MOHAMMED", "JEAN-PAUL", "AHMED ALI HASSAN", "LI", "MARIA DEL CARMEN", "OLUWASEUN", "SARA", "A", "",
          "ABDULRAHMAN BIN KHALID", "NURUL AIN BINTI MOHD"]
LONG_SUR = "ABCDEFGHIJKLMNOPQRSTUVWXYZ-ZYXWVUTSRQPONMLKJIHGFEDCBA"
TYPES = {"TD3": ["P<", "PD", "PO"], "MRVA": ["V<", "VD"], "MRVB": ["V<", "VD"], "TD2": ["I<", "ID", "A<", "AC"],
         "TD1": ["I<", "ID", "IR", "C1", "AC", "CR"]}
SPECIAL = ["D", "GBD", "GBN", "GBO", "GBP", "GBS", "UNO", "UNA", "UNK", "XXA", "XXB", "XXC", "XXX", "EUE", "RKS", "XKX"]
ALNUM = "ABCDEFGHJKLMNPRSTUVWXYZ0123456789"


def rnd_date(rng, y0, y1):
    y, m = rng.randint(y0, y1), rng.randint(1, 12)
    d = rng.randint(1, [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31][m - 1])
    return f"{y % 100:02d}{m:02d}{d:02d}"


def rnd_doc(rng, fmt):
    n = rng.randint(6, 9) if fmt in ("TD3", "MRVA", "MRVB") else rng.choice([8, 9, 9, 9, 10, 12, 13])
    return "".join(rng.choice(ALNUM) for _ in range(n))


def rnd_case(rng, fmt, issuing=None, nat=None, surname=None, given=None, doc=None):
    codes = list(COUNTRIES)
    kw = dict(
        tcode=rng.choice(TYPES[fmt]), issuing=issuing or rng.choice(codes), nat=nat or rng.choice(codes),
        surname=surname if surname is not None else rng.choice(SURNAMES), given=given if given is not None else rng.choice(GIVENS),
        doc=doc or rnd_doc(rng, fmt), dob=rnd_date(rng, 1935, 2024), sex=rng.choice("MF<X"), exp=rnd_date(rng, 2027, 2046),
        opt="".join(rng.choice(ALNUM) for _ in range(rng.choice([0, 0, 5, 7]))),
        opt2="".join(rng.choice(ALNUM) for _ in range(rng.choice([0, 3, 11]))) if fmt == "TD1" else "",
        opt_check_zero=rng.random() < 0.3)
    if fmt == "TD1":
        kw["opt"] = kw["opt"][:15]
    return kw


# ----------------------------------------------------------------------------- 1. ICAO specimens
SPECIMENS = {
    "TD3": (["P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<", "L898902C36UTO7408122F1204159ZE184226B<<<<<10"],
            dict(documentNumber="L898902C3", surname="ERIKSSON", givenNames="ANNA MARIA", nationality="UTO", birthDate="1974-08-12", expiryDate="2012-04-15")),
    "TD1": (["I<UTOD231458907<<<<<<<<<<<<<<<", "7408122F1204159UTO<<<<<<<<<<<6", "ERIKSSON<<ANNA<MARIA<<<<<<<<<<"],
            dict(documentNumber="D23145890", surname="ERIKSSON", givenNames="ANNA MARIA", nationality="UTO", birthDate="1974-08-12", expiryDate="2012-04-15")),
    "TD2": (["I<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<", "D231458907UTO7408122F1204159<<<<<<<6"],
            dict(documentNumber="D23145890", surname="ERIKSSON", givenNames="ANNA MARIA", nationality="UTO", birthDate="1974-08-12", expiryDate="2012-04-15")),
    "MRVA": (["V<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<", "L8988901C4XXX4009078F96121096ZE184226B<<<<<<"],
             dict(documentNumber="L8988901C", surname="ERIKSSON", givenNames="ANNA MARIA", nationality="XXX", birthDate="1940-09-07", expiryDate="2096-12-10")),
    "MRVB": (["V<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<", "L8988901C4XXX4009078F9612109<<<<<<<<"],
             dict(documentNumber="L8988901C", surname="ERIKSSON", givenNames="ANNA MARIA", nationality="XXX", birthDate="1940-09-07", expiryDate="2096-12-10")),
}


def test_specimens():
    out = {}
    for fmt, (lines, exp) in SPECIMENS.items():
        try:
            r = parse_mrz(lines, date(2005, 1, 1)).to_dict()
        except MrzError as e:
            out[fmt] = f"PARSE ERROR {e}"
            continue
        bad = {k: (r.get(k), v) for k, v in exp.items() if r.get(k) != v}
        out[fmt] = "OK" if not bad and r["valid"] and r["format"] == fmt else {"mismatch": bad, "valid": r["valid"], "format": r["format"]}
    return out


# ----------------------------------------------------------------------------- 2/3. matrix + edge cases
def run_case(fmt, kw, tag):
    lines, truth = build(fmt, **kw)
    assert [len(x) for x in lines] == {"TD3": [44, 44], "MRVA": [44, 44], "MRVB": [36, 36], "TD2": [36, 36], "TD1": [30, 30, 30]}[fmt], (fmt, lines)
    d, errs, codes = read(lines)
    return {"tag": tag, "fmt": fmt, "lines": lines, "truth": truth, "parsed": d is not None,
            "wrong": diff(d, truth) if d else FIELDS, "errors": errs, "codes": codes, "kw": kw}


def matrix(rng):
    res = []
    for fmt in TYPES:
        for code in list(COUNTRIES) + SPECIAL:
            for k in range(2):
                nat = code if k == 0 else None
                kw = rnd_case(rng, fmt, issuing=code if k == 1 else None, nat=nat)
                res.append(run_case(fmt, kw, f"country:{code}"))
    return res


def edge_cases(rng):
    res = []
    for fmt in TYPES:
        # long / awkward names
        for sur, giv, tag in [(LONG_SUR, "ANNA MARIA", "long-surname"), ("SMITH", "A" * 60, "long-given"), ("X", "", "no-given"),
                              ("", "ANNA", "no-surname"), ("DE LA CRUZ-SANCHEZ", "MARIA-JOSE", "hyphens"), ("O'NEIL", "D'ARCY", "apostrophes"),
                              ("A" * 38, "B", "fills-field"), ("A" * 37, "B", "fills-field-2"), ("A" * 39, "B", "fills-39")]:
            res.append(run_case(fmt, rnd_case(rng, fmt, surname=sur, given=giv), f"name:{tag}"))
        # document number lengths
        for n in range(1, 16):
            if fmt in ("TD3", "MRVA", "MRVB") and n > 9:
                continue
            res.append(run_case(fmt, rnd_case(rng, fmt, doc="".join(rng.choice(ALNUM) for _ in range(n))), f"docno:len{n}"))
        # documents number with leading/trailing zero-like and letters O/I that OCR confuses
        for doc in ["0000000001", "O00000001", "1I1I1I1I1", "B8B8B8B8B", "S5S5S5S5S", "Z2Z2Z2Z2Z", "G6G6G6G6G"]:
            if fmt in ("TD3", "MRVA", "MRVB") and len(doc) > 9:
                continue
            res.append(run_case(fmt, rnd_case(rng, fmt, doc=doc), "docno:lookalikes"))
        # sex markers
        for sx in "MF<X":
            kw = rnd_case(rng, fmt); kw["sex"] = sx
            res.append(run_case(fmt, kw, f"sex:{sx}"))
        # dates: leap day, century pivot, month ends
        for dob, exp, tag in [("000229", "281231", "leap-2000"), ("240229", "270101", "leap-2024"), ("261108", "300101", "born-2026"),
                              ("270101", "300101", "pivot-2027"), ("260101", "300101", "pivot-2026"), ("250101", "300101", "pivot-2025"),
                              ("301231", "300101", "pivot-1930")]:
            kw = rnd_case(rng, fmt); kw["dob"] = dob; kw["exp"] = exp
            try:
                r = run_case(fmt, kw, f"date:{tag}")
            except ValueError:
                continue
            res.append(r)
    return res


def summarize(rs):
    tot = len(rs); ok = 0; bad_parse = 0; bad_fields = collections.Counter(); flagged_clean = 0; examples = []
    by_tag = collections.defaultdict(lambda: [0, 0])
    for r in rs:
        key = r["tag"].split(":")[0]
        clean = r["parsed"] and not r["wrong"] and not r["errors"]
        by_tag[key][1] += 1
        if clean:
            ok += 1; by_tag[key][0] += 1
        else:
            if not r["parsed"]:
                bad_parse += 1
            for f in r["wrong"]:
                bad_fields[f] += 1
            if r["parsed"] and not r["wrong"] and r["errors"]:
                flagged_clean += 1
            if len(examples) < 25:
                examples.append({k: r[k] for k in ("tag", "fmt", "lines", "wrong", "errors")})
    return {"total": tot, "exact_and_unflagged": ok, "exact_but_flagged_for_review": flagged_clean, "not_parsed": bad_parse,
            "wrong_field_counts": dict(bad_fields), "by_group": {k: f"{v[0]}/{v[1]}" for k, v in by_tag.items()}, "examples": examples}


# ----------------------------------------------------------------------------- 4. error injection
CONFUSE = {"O": "0", "0": "O", "I": "1", "1": "I", "B": "8", "8": "B", "S": "5", "5": "S", "Z": "2", "2": "Z", "G": "6", "6": "G",
           "D": "0", "Q": "0", "L": "1", "T": "7", "7": "T", "U": "V", "V": "U", "H": "N", "N": "H", "M": "N", "<": "K", "K": "<",
           "A": "4", "4": "A", "E": "F", "F": "E", "C": "G", "R": "P", "P": "R", "W": "V", "Y": "V", "X": "K", "J": "I", "9": "g"}
CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<"


def corrupt(lines, k, rng):
    ls = [list(l) for l in lines]
    pos = [(i, j) for i, l in enumerate(ls) for j in range(len(l))]
    for i, j in rng.sample(pos, k):
        c = ls[i][j]
        n = CONFUSE.get(c) if rng.random() < 0.7 else None
        n = n.upper() if n else rng.choice([x for x in CHARS if x != c])
        ls[i][j] = n if n != c else rng.choice([x for x in CHARS if x != c])
    return ["".join(l) for l in ls]


def field_of_position(fmt, line, j):
    """Which field a character position belongs to (for reporting where silent errors sit)."""
    if fmt == "TD1":
        if line == 0:
            return "typeCode" if j < 2 else "issuingState" if j < 5 else "documentNumber" if j < 15 else "optionalData"
        if line == 1:
            return ("birthDate" if j < 7 else "sex" if j == 7 else "expiryDate" if j < 15 else "nationality" if j < 18
                    else "optionalData2" if j < 29 else "composite")
        return "name"
    if line == 0:
        return "typeCode" if j < 2 else "issuingState" if j < 5 else "name"
    return ("documentNumber" if j < 10 else "nationality" if j < 13 else "birthDate" if j < 20 else "sex" if j == 20
            else "expiryDate" if j < 28 else "optionalData" if j < (42 if fmt == "TD3" else 35) else "check")


def error_injection(rng, n_per):
    out = {}
    for k in (1, 2, 3):
        stats = collections.Counter(); silent_fields = collections.Counter(); silent_examples = []
        for fmt in TYPES:
            for _ in range(n_per):
                kw = rnd_case(rng, fmt)
                lines, truth = build(fmt, **kw)
                bad = corrupt(lines, k, rng)
                if bad == lines:
                    continue
                found = find_mrz_lines(bad) or bad
                d, errs, codes = read(found)
                stats["total"] += 1
                if d is None:
                    stats["rejected_unparsed"] += 1; continue
                wrong = diff(d, truth)
                if errs:
                    stats["flagged_error"] += 1
                elif not wrong:
                    stats["recovered_exact"] += 1
                else:
                    stats["SILENT_WRONG"] += 1
                    prot = [w for w in wrong if w in PROTECTED or (w == "optionalData" and fmt == "TD3")]
                    for w in wrong:
                        silent_fields[w] += 1
                    if prot:
                        stats["SILENT_WRONG_in_check_protected_field"] += 1
                        if len(silent_examples) < 6:
                            silent_examples.append({"fmt": fmt, "truth_lines": lines, "read_lines": bad, "wrong": wrong})
        out[f"{k}_error"] = {"stats": dict(stats), "silent_wrong_fields": dict(silent_fields), "protected_examples": silent_examples}
    return out


# ----------------------------------------------------------------------------- 5. filler '<' drop/add
def filler_faults(rng, n_per):
    res = collections.Counter(); ex = []
    for fmt in TYPES:
        for _ in range(n_per):
            kw = rnd_case(rng, fmt)
            lines, truth = build(fmt, **kw)
            for mode in ("drop1", "drop2", "add1"):
                bad = []
                for l in lines:
                    runs = [i for i, c in enumerate(l) if c == "<"]
                    if not runs:
                        bad.append(l); continue
                    p = rng.choice(runs)
                    if mode == "drop1":
                        bad.append(l[:p] + l[p + 1:])
                    elif mode == "drop2":
                        bad.append(l[:p] + l[p + 2:] if p + 2 <= len(l) else l[:p])
                    else:
                        bad.append(l[:p] + "<" + l[p:])
                found = find_mrz_lines(bad)
                key = f"{mode}"
                res[key + ":total"] += 1
                if not found:
                    res[key + ":not_found(flagged)"] += 1; continue
                d, errs, _ = read(found)
                if d is None:
                    res[key + ":unparsed(flagged)"] += 1
                elif errs:
                    res[key + ":flagged_error"] += 1
                elif not diff(d, truth):
                    res[key + ":recovered_exact"] += 1
                else:
                    res[key + ":SILENT_WRONG"] += 1
                    if len(ex) < 5:
                        ex.append({"mode": mode, "fmt": fmt, "read": found, "wrong": diff(d, truth)})
    return {"counts": dict(sorted(res.items())), "silent_examples": ex}


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--n-errors", type=int, default=600, help="documents per format per error level")
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    report = {"specimens": test_specimens()}
    report["country_matrix"] = summarize(matrix(rng))
    report["edge_cases"] = summarize(edge_cases(rng))
    report["error_injection"] = error_injection(rng, a.n_errors)
    report["filler_faults"] = filler_faults(rng, max(50, a.n_errors // 6))
    print(json.dumps(report, indent=1, ensure_ascii=False))
    if a.json:
        a.json.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
