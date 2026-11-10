# Accuracy review (test-expert view)

Updated 2026-11-09 after the fixes: see section 7 for before/after numbers. Sections 1 to 5 describe the state BEFORE the fixes.

Scope: passports (all countries, TD3), Schengen/national visas (MRV-A, MRV-B), US green card (TD1), ID cards (TD1/TD2), GCC Iqama/IDs.
Tools: `tools/accuracy_matrix.py` (independent ICAO 9303 oracle, text level), `tools/e2e_accuracy.py` (rendered images through `pipeline.run`), `tools/run_scenarios.py`.
Date: 2026-11-09. Engine available here: Tesseract (eng, no OCR-B). PaddleOCR models were never run (not available offline here).

## 1. Parser on clean MRZ text (no OCR)
| Check | Result |
|---|---|
| ICAO specimens, 5 formats | 5/5 exact |
| TD3 / MRV-A / MRV-B, 2,810 generated cases, all countries | all correct |
| TD2 with document number > 9 chars | DEFECT: check digit fails, goes to review, number wrong (634 of 1,500 generated TD2; the generator over-samples long numbers) |
| TD1 (green card) overflow document number | DEFECT: optionalData keeps a leading `<` (328 of 1,500 generated TD1) |
| ICAO special codes (`D`, `GBD/GBN/GBO/GBP/GBS`, `UNO/UNA/UNK`, `EUE`, `RKS`, `XKX`) | warn as unknown country; `XKX` has an empty name |

## 2. Errors the check digit cannot see
- 52 of 666 character pairs differ by a multiple of 10 in check-digit value. Look-alike ones: `1`/`L`, `6`/`G`, `<`/`K`, `<`/`U`. A document number or optional data with such a swap passes all checks silently.
- Names, nationality, sex, issuing state and type code have no check digit. Policy covers the first four (needs OCR confidence or form confirmation). NOT covered: type code and MRV optional data.
- Passport `P` misread as `V`: 300/300 parsed as a visa with no error (same 2x44 geometry). The reverse, visa read as passport, is caught.

## 3. End to end, Tesseract, 160 rendered images (5 formats x 8 documents x clean/tilt+4/blur/noise)
| Outcome | Count |
|---|---|
| Correct, sent to review | 45 |
| Wrong field, flagged for review | 74 |
| Silent wrong (wrong, not flagged) | **0** |
| No result (`mrz_not_found`) | 41 (35 of them tilt +4) |
| Auto-accepted | **0** |
Field accuracy where a result came back (119): documentNumber 60 (50%), surname 117, givenNames 109, nationality 90 (76%), birthDate 107, expiryDate 106, sex 106, issuingState 116. Typical error: `0`/`O` in document numbers (caught by check digit).
Original 50-image scenario run (Tesseract): PASS 14, PARTIAL 15, FAIL 21 (28%).
Nothing is auto-accepted because Tesseract gives no confidence, and sex/issuing state can never be confirmed by the form.

## 4. Iqama and GCC IDs
- Tesseract path: 0 of 8 Iqama images read (visual zone needs PaddleOCR with Arabic).
- Visual-zone parser: 13 unit tests on synthetic text lines only. One real Iqama tested so far.
- Emirates ID, Qatar, Kuwait, Bahrain, Oman IDs: untested/unsupported (see `docs/GCC_ID_COVERAGE.md`).

## 5. Verdict
- Safety property holds in these tests: no silent wrong read from OCR noise, because check digits plus policy force review.
- Accuracy and straight-through rate are NOT acceptable yet with Tesseract: about 28-42% fully correct, 0% auto-accept. Whether PaddleOCR fixes this is unproven.
- Not provable without real specimens: Schengen, green card, most GCC cards, real OCR-B print, phone photos.

## 6. Recommended fixes (not yet made)
1. TD2 overflow document number parsing.
2. Strip leading `<` from TD1 optional data.
3. Check-digit-guided repair: try O/0, I/1, B/8, S/5, Z/2, G/6 swaps in document number until the check digit passes (flag as corrected).
4. Deskew / rotation search (+-5 deg) before MRZ detection; tilt +4 fails 35/40.
5. Add type code and MRV optional data to the unverified-field policy; reject P/V type mismatch with page evidence.
6. Add ICAO special country codes.
7. Real specimens (30 per type/country group) and a Paddle run before go-live.

## 7. After the fixes (A1 to A8 of the checklist)
Changes: TD2 long numbers, TD1 optional data, ICAO special country codes, check-digit guided repair of look-alike letters
(always sent to review), digit-to-letter fix in names and country codes, stray letters inside the filler run, tilt search
(+-2, 4, 6 degrees) and best-of-several reads in the Tesseract engine, MRZ blocks whose data line has no `<`, type code
and visa optional data in the review policy, P-read-as-V detection, and a review flag for document numbers with 1/L/6/G
when OCR confidence is not available. Tests: `tests/test_accuracy_fixes.py` (24 tests); whole suite 158 passed.

Hold-out run: 240 new documents (`E2E_SEED=5000 python tools/e2e_accuracy.py 12`), Tesseract, synthetic fonts.
| Measure | Before | After |
|---|---|---|
| Fully correct (all fields) | 28% | **83%** |
| Wrong but flagged for review | 46% | 17% |
| Not read | 26% | **0%** |
| Silent wrong | 0% | **0%** |
| Auto-accepted | 0% | 0% (needs OCR confidence: PaddleOCR) |
| Tilted +4 degrees: fully correct | 0% | 82% |
By family (after): TD3 passport 90%, MRV-A 85%, MRV-B 88%, TD2 83%, TD1 69%.
By field (after): document number 88%, surname 98%, given names 94%, nationality 100%, birth date 100%, expiry date 100%, sex 100%, issuing state 99%.
Original 61-image scenario run: PASS 28% -> 77%. Clean-text parser: 2,820 of 2,820 country cases and 189 of 189 edge cases exact.
Check-digit-protected fields still wrong without a flag after one injected character error: 87 -> 36 of 1,500 (2.4%): only the pairs the check digit cannot see.

Still open: 100% is not reached. Document numbers (88%) lose to J/1, 9/0 and 6/G style misreads that no rule can undo; TD1 (69%) has long
receipt numbers; names and Iqama/GCC cards need PaddleOCR, fine-tuning and real specimens (checklist groups B, C, D).

## 8. Groups B and C (engine, models, specimens)
Done here (Tesseract + RapidOCR, no downloads needed):
- New `ensemble` engine (now the default when PaddleOCR is not available). Tesseract and RapidOCR read the page. When both
  give the same MRZ lines and every check digit passes, the reading gets confidence 0.99 (`AGREEMENT_CONFIDENCE`). That is
  what lets clear reads skip review. If they differ, the reading with more passing check digits is used and a person reviews it.
- Arabic + English text lines from Tesseract (`ara` + `eng` data) feed the existing ID-card reader, so Iqama works without PaddleOCR.
- Value-before-label lines ("1988/11/02 تاريخ الميلاد") are read; a garbled expiry label is guessed and forced to review.
- `tools/specimens.py`: plan (1,200 specimens: 30 per type and country group), add, status, draft. `training/synth_mrz.py`:
  synthetic MRZ line images for fine-tuning.

Hold-out run, 240 new documents, ensemble engine:
| Measure | Tesseract only (section 7) | Ensemble |
|---|---|---|
| Fully correct | 83% | **91%** |
| Accepted without review | 0% | **42%** (100 documents, 0 wrong) |
| Wrong but flagged | 17% | 9% |
| Silent wrong | 0% | 0% |
| Document number correct | 88% | 97% |
| Mean time per document | 1.9 s | 4.9 s (max 17 s) |
Calibration: documents where the two readers agreed had 0 wrong unprotected fields out of 100; documents where they did not
had 13.6% (19 of 140). So agreement is a usable confidence signal; with 0 of 100 the true error rate could still be up to
about 3% (95% bound), so keep human sampling of accepted documents. Synthetic fonts only.
Original 61-image scenario suite: 28% -> 77% -> **87%**. Iqama rendered cards (8 images): 0% -> **88%** (7 of 8).

NOT done, and why:
- B1 PaddleOCR models and B2 Arabic Paddle model: model download hosts are blocked in this environment (checked). Run
  `python tools/fetch_models.py` on a connected machine.
- B3 fine-tuning: needs the pretrained weights and a GPU. `training/synth_mrz.py` is ready. Baseline on 40 harsh synthetic
  lines: character error rate Tesseract 17%, RapidOCR 21% (the number to beat).
- C1 and C2 collecting real specimens, C3 labelling: need real documents. Use `tools/specimens.py`.
- Other GCC ID layouts (group D) need specimens first.

## 9. Remaining checklist items
Done: D1 to D6 as number-format recognition only (not layouts), E1 (CI gate: `tools/accuracy_gate.py`, 40 generated documents,
95% fully correct, 2.5% not read, 0 silent wrong, in 2 minutes), E2 as a proposal (`config/acceptance.json`, status PROPOSED).
Cannot be finished without outside input:
- B1, B2, B3: PaddleOCR weights (download hosts blocked here), GPU, OCR-B font.
- C1, C2: real documents with the holder's consent (`tools/specimens.py` tracks them).
- D1 to D7 as real layouts, and Oman: need 20 or more real front and back images per card.
- E2 sign-off: the business owner must agree the numbers in `config/acceptance.json`.
Run `python tools/accuracy_gate.py --real-dir real-samples` after `tools/run_scenarios.py --real-dir real-samples` to apply the go-live bar.
