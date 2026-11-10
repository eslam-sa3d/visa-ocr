# Test plan and scenarios

Three layers. Run them in this order.

| Layer | What it proves | Command | Needs OCR models |
|---|---|---|---|
| 1. Unit and API scenarios | Parsing, check digits, business rules, API errors, privacy rules | `python -m pytest -q` | No |
| 2. Image scenarios (synthetic) | Every document type, read from a picture, under bad-photo conditions | `python tools/run_scenarios.py` | Yes |
| 3. Real-document scenarios | Accuracy on real specimens (the number that matters for go-live) | `python tools/run_scenarios.py --real-dir real-samples` | Yes |

## Layer 1: unit and API scenarios (about 85 tests)
| ID | Scenario | Expected |
|---|---|---|
| S1 | Passport (TD3), visa (MRV-A, MRV-B), ID card (TD1, TD2), green card (TD1) text | Parsed, all check digits pass |
| S2 | Each check digit corrupted in turn (document number, birth, expiry, composite) | That check fails, `mrz_checksum_failed`, needs review |
| S3 | Letter O, I, B read instead of 0, 1, 8 in numeric fields | Corrected, `ocr_digits_corrected` flag |
| S4 | Two-digit years (08 and 50) | 2008 and 1950, never a future birth date |
| S5 | Expired document, future birth date, bad sex marker, unknown country code | Flag raised, needs review where it is an error |
| S6 | Declared nationality matches, differs, or is not a 3-letter code | No flag, `nationality_mismatch` (review), `expected_nationality_invalid` |
| S7 | Wrong characters in MRZ text (for example a Japanese `く` instead of `<`) | 422 with a help message |
| S8 | Wrong line counts, truncated lines, extra or missing `<` | Rejected, or repaired and then judged by check digits |
| S9 | Name in MRZ differs from the name printed on the page, truncated names | `name_differs_from_visual_zone`, but no flag for a cut name |
| S10 | Iqama / GCC ID: labelled fields, Arabic and English labels, Arabic-Indic digits, Hijri dates, value on next line, misread digit | Fields extracted, check digit warning |
| S11 | API: wrong file type (415), more than 8 files (422), over 10 MB (413), unknown job (404), no files (422) | Correct status and error code |
| S12 | Engine crash or blank image | Clean `failed` job, no internal details leaked |
| S13 | Training data: consent missing, store disabled, purge by age | 409 or removed, nothing stored without consent |

## Layer 2: image scenarios (111 generated images)
`python tools/make_scenarios.py` renders fictional documents; `python tools/run_scenarios.py` scores them.

- **Document types (11):** passport (SAU, EGY, GBR, DEU, IND), visa MRV-A (FRA, ESP), visa MRV-B (DEU), ID card TD1 (ARE) and TD2 (EGY), US green card, plus Iqama (visual, no MRZ) with two people.
- **Photo conditions (9):** clean, tilt +4 degrees, tilt -3 degrees, blur, noise, low resolution, low contrast, glare over the MRZ, shear (camera angle).
- **Negative cases (4):** blank page, random noise, a letter with no MRZ, a tiny image. These must fail cleanly.
- **Result per image:** PASS (every field correct), PARTIAL (read but a field is wrong), FAIL (not read).
- Output: `scenarios/report.md` (matrix by type and condition, failure list) and `report.json`.

The images use a plain monospaced font, not the real OCR-B face, so treat results as a floor check and a regression guard. Run it after every change and compare reports.

## Layer 3: real documents (do this before go-live)
1. Make a folder `real-samples/` with your images (never commit personal documents).
2. Add `real-samples/expected.json`:
```json
{"scenarios": [
  {"id": "cn_passport_1", "group": "real", "docType": "passport:TD3", "variant": "scan",
   "image": "cn_passport_1.jpg", "expectStatus": "done",
   "expected": {"documentNumber": "EJ8942101", "nationality": "CHN", "birthDate": "1981-01-13"}}
]}
```
3. Run `python tools/run_scenarios.py --real-dir real-samples`.
4. For go-live, collect at least 30 specimens per document type and per issuing country group, in the conditions your applicants actually use (phone photo, scanner, glare, folded).

## Suggested acceptance thresholds (to agree with the business)
- MRZ documents on real specimens: at least 95% fully correct, and 100% of wrong reads flagged for review (no silent errors).
- Iqama / GCC ID: at least 90% on the key fields (ID number, names, dates) with human review of the rest.
- Zero crashes and zero leaked internals on the negative cases.
- Median time per document under 10 seconds on the target server.

## Not covered yet
- Security tests (penetration test, upload malware scanning), load and concurrency tests, and offline-install tests on the real air-gapped server.
- GCC IDs from other countries and older Iqama designs: need real specimens.
- Back side of the Iqama and the Saudi national ID layout.
