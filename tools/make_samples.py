"""Generate synthetic MRZ sample data and real API request/response examples.

All holders are fictional ("SPECIMEN") and all numbers are made up. Check digits are
computed, so every sample parses as valid. Run from the project root:

    python tools/make_samples.py
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient  # noqa: E402
from PIL import Image  # noqa: E402

import app.main as main  # noqa: E402
from app.engines.base import EngineOutput  # noqa: E402
from app.mrz import check_digit as cd  # noqa: E402
from app.mrz import parse_mrz  # noqa: E402
from app.validation import validate  # noqa: E402

SAMPLES = ROOT / "samples"
API = SAMPLES / "api"
DOCS = ROOT / "docs"
for d in (SAMPLES, API, DOCS):
    d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# MRZ builders (with computed check digits)
# --------------------------------------------------------------------------- #
def pad(s: str, n: int) -> str:
    assert len(s) <= n, (s, len(s), n)
    return s.ljust(n, "<")


def code3(c: str) -> str:
    return c.ljust(3, "<")


def names(surname: str, given: str) -> str:
    return surname.replace(" ", "<") + "<<" + given.replace(" ", "<")


def td3(issuer, surname, given, docno, nat, dob, sex, exp, personal=""):
    l1 = "P<" + code3(issuer) + pad(names(surname, given), 39)
    d9, p14 = pad(docno, 9), pad(personal, 14)
    pchk = str(cd(p14)) if personal else "<"
    body = d9 + str(cd(d9)) + code3(nat) + dob + str(cd(dob)) + sex + exp + str(cd(exp)) + p14 + pchk
    comp = str(cd(body[0:10] + body[13:20] + body[21:43]))
    return [l1, body + comp]


def mrva(issuer, surname, given, docno, nat, dob, sex, exp, optional=""):
    l1 = "V<" + code3(issuer) + pad(names(surname, given), 39)
    d9 = pad(docno, 9)
    l2 = d9 + str(cd(d9)) + code3(nat) + dob + str(cd(dob)) + sex + exp + str(cd(exp)) + pad(optional, 16)
    return [l1, l2]


def td1(type2, issuer, docno, nat, dob, sex, exp, surname, given, opt1="", opt2=""):
    if len(docno) > 9:  # long document number overflows into optional data 1
        d9, chk, o1 = docno[:9], "<", docno[9:] + str(cd(docno)) + "<" + opt1
    else:
        d9, chk, o1 = pad(docno, 9), str(cd(pad(docno, 9))), opt1
    l1 = pad(type2, 2) + code3(issuer) + d9 + chk + pad(o1, 15)
    l2 = dob + str(cd(dob)) + sex + exp + str(cd(exp)) + code3(nat) + pad(opt2, 11)
    comp = str(cd(l1[5:30] + l2[0:7] + l2[8:15] + l2[18:29]))
    return [l1, l2 + comp, pad(names(surname, given), 30)]


# --------------------------------------------------------------------------- #
# Sample data
# --------------------------------------------------------------------------- #
SPEC = [
    ("passport_sau", "Passport, Saudi Arabia (synthetic)", "SAU", "passport",
     td3("SAU", "SPECIMEN", "NORA SAMPLE", "SP0000001", "SAU", "900315", "F", "310614")),
    ("passport_egy", "Passport, Egypt (synthetic)", "EGY", "passport",
     td3("EGY", "SPECIMEN", "OMAR TEST", "SP0000002", "EGY", "850702", "M", "330128")),
    ("passport_gbr", "Passport, United Kingdom (synthetic, born 2001)", "GBR", "passport",
     td3("GBR", "SPECIMEN", "ALEX JAMES", "SP0000003", "GBR", "010928", "M", "340511")),
    ("passport_deu", "Passport, Germany (issuer code 'D', optional data, compound surname)", "D", "passport",
     td3("D", "SPECIMEN MUELLER", "ERIKA", "SP0000004", "D", "781120", "F", "320903", "ZZ12345")),
    ("passport_ind", "Passport, India (synthetic)", "IND", "passport",
     td3("IND", "SPECIMEN", "ASHA", "SP0000005", "IND", "960421", "F", "350317")),
    ("visa_fra", "Schengen visa sticker, France, holder of passport_sau (synthetic)", "FRA", "visa",
     mrva("FRA", "SPECIMEN", "NORA SAMPLE", "VS0000001", "SAU", "900315", "F", "270512")),
    ("visa_esp", "Schengen visa sticker, Spain, holder of passport_egy (synthetic)", "ESP", "visa",
     mrva("ESP", "SPECIMEN", "OMAR TEST", "VS0000002", "EGY", "850702", "M", "270930")),
    ("greencard_usa", "US permanent resident card, back MRZ (TD1, 13-character receipt number)", "USA", "id_card",
     td1("C1", "USA", "SRC0000000001", "IND", "880810", "M", "331015", "SPECIMEN", "RAVI TEST")),
    ("idcard_are", "National ID card, TD1 MRZ, issuing state ARE (synthetic)", "ARE", "id_card",
     td1("I<", "ARE", "SP0000011", "ARE", "920205", "F", "291130", "SPECIMEN", "LAYLA SAMPLE")),
]

samples = []
for sid, label, country, kind, lines in SPEC:
    parsed = parse_mrz(lines)
    flags = validate(parsed)
    assert parsed.valid, (sid, parsed.checks)
    assert not flags, (sid, [f.message for f in flags])
    samples.append({
        "id": sid, "label": label, "country": country, "kind": kind,
        "synthetic": True, "lines": lines, "expected": parsed.to_dict(),
    })
(SAMPLES / "mrz_samples.json").write_text(json.dumps(samples, indent=2, ensure_ascii=False) + "\n")
by_id = {s["id"]: s for s in samples}


# --------------------------------------------------------------------------- #
# Real API request/response examples (engine faked so the MRZ is known)
# --------------------------------------------------------------------------- #
class FakeEngine:
    name = "tesseract-mrz"

    def __init__(self, lines):
        self.lines = lines

    def extract(self, images, doc_type_hint):
        return EngineOutput(engine=self.name, mrz_lines=self.lines)


client = TestClient(main.app)
CID = "portal-req-0001"
FIXED_JOB = "7f3c2a9e1b4d4c6a8e5f0b1d2c3a4b5e"
FIXED_TS = "2026-11-09T09:52:14.512345+00:00"

buf = io.BytesIO()
Image.new("RGB", (64, 40), "white").save(buf, format="PNG")
PNG = buf.getvalue()


def run_job(lines, hint="passport"):
    main.get_engine = lambda: FakeEngine(lines)
    r = client.post("/v1/documents/extract", headers={"X-Correlation-ID": CID},
                    files=[("files", ("data-page.jpg", PNG, "image/png"))],
                    data={"doc_type_hint": hint})
    accepted = r.json()
    job = client.get(f"/v1/jobs/{accepted['jobId']}").json()
    for obj in (accepted, job):
        obj["jobId"] = FIXED_JOB
    job["createdAt"] = FIXED_TS
    return r.status_code, accepted, job


def save(name, obj):
    text = obj if isinstance(obj, str) else json.dumps(obj, indent=2, ensure_ascii=False)
    (API / name).write_text(text.rstrip("\n") + "\n")
    return text


passport = by_id["passport_sau"]["lines"]
code, accepted, job_done = run_job(passport)
assert code == 202 and job_done["status"] == "done"

curl = (
    "curl -X POST https://ocr.internal.example/v1/documents/extract \\\n"
    "  -H 'X-Correlation-ID: portal-req-0001' \\\n"
    "  -F 'files=@data-page.jpg;type=image/jpeg' \\\n"
    "  -F 'doc_type_hint=passport'\n\n"
    "# Poll for the result:\n"
    "curl https://ocr.internal.example/v1/jobs/" + FIXED_JOB
)
ex = {}
ex["01_extract_request.curl.txt"] = save("01_extract_request.curl.txt", curl)
ex["02_extract_response_202.json"] = save("02_extract_response_202.json", accepted)
ex["03_job_done_passport.json"] = save("03_job_done_passport.json", job_done)

tampered = [passport[0], "T" + passport[1][1:]]  # wrong document number, check digit now fails
_, _, job_review = run_job(tampered)
assert job_review["result"]["needsReview"] is True
ex["04_job_needs_review.json"] = save("04_job_needs_review.json", job_review)

main.get_engine = lambda: FakeEngine(None)
r = client.post("/v1/documents/extract", files=[("files", ("blank.jpg", PNG, "image/png"))])
failed = client.get(f"/v1/jobs/{r.json()['jobId']}").json()
failed["jobId"], failed["correlationId"], failed["createdAt"] = FIXED_JOB, CID, FIXED_TS
ex["05_job_failed_no_mrz.json"] = save("05_job_failed_no_mrz.json", failed)

r = client.post("/v1/documents/extract", files=[("files", ("a.pdf", b"%PDF-1.4", "application/pdf"))])
ex["06_error_unsupported_type_415.json"] = save("06_error_unsupported_type_415.json", r.json())

visa_req = {"lines": by_id["visa_fra"]["lines"]}
visa_resp = client.post("/v1/mrz/parse", json=visa_req).json()
ex["07_mrz_parse_request_visa.json"] = save("07_mrz_parse_request_visa.json", visa_req)
ex["08_mrz_parse_response_visa.json"] = save("08_mrz_parse_response_visa.json", visa_resp)


# --------------------------------------------------------------------------- #
# Markdown doc
# --------------------------------------------------------------------------- #
def block(lang, text):
    return f"```{lang}\n{text.rstrip()}\n```\n"


md = ["# API examples and sample data\n",
      "Generated by `tools/make_samples.py` from the running service. All people are fictional "
      "and all numbers are made up.\n",
      "## 1. Upload a document\n",
      "Send 1 to 4 JPEG or PNG images (for example the passport data page). The call returns "
      "immediately with a job.\n",
      block("bash", ex["01_extract_request.curl.txt"]),
      "Response `202 Accepted`:\n", block("json", ex["02_extract_response_202.json"]),
      "## 2. Read the result\n",
      "`GET /v1/jobs/{jobId}` while the job is `queued` or `processing` returns the status only. "
      "When `done`, `result` carries the fields, per-check results and flags.\n",
      block("json", ex["03_job_done_passport.json"]),
      "## 3. Result that needs human review\n",
      "A wrong character in the document number makes its check digit fail, so the flag "
      "`mrz_checksum_failed` is raised and `needsReview` is `true`.\n",
      block("json", ex["04_job_needs_review.json"]),
      "## 4. Failures\n",
      "No MRZ found in the image (job status `failed`):\n", block("json", ex["05_job_failed_no_mrz.json"]),
      "Rejected upload (HTTP 415):\n", block("json", ex["06_error_unsupported_type_415.json"]),
      "## 5. Parse MRZ text you already have\n",
      "`POST /v1/mrz/parse` (no OCR involved), shown with a Schengen visa sticker:\n",
      block("json", ex["07_mrz_parse_request_visa.json"]), "Response `200 OK`:\n",
      block("json", ex["08_mrz_parse_response_visa.json"]),
      "## 6. Sample data\n",
      "`samples/mrz_samples.json` holds synthetic documents for every supported format, each with "
      "the expected parsed result. Use them for tests and as the seed of the POC set.\n",
      "| Id | Document | Format |\n|---|---|---|"]
for s in samples:
    md.append(f"| `{s['id']}` | {s['label']} | {s['expected']['format']} |")
md.append("")
for s in samples:
    md.append(f"**{s['id']}**\n")
    md.append(block("", "\n".join(s["lines"])))
(DOCS / "API_EXAMPLES.md").write_text("\n".join(md))
print("samples:", len(samples), "| api examples:", len(ex))
