# Visa Portal offline document OCR

First implementation slice of the offline OCR plan: web-only, server-side, no internet access at runtime.

## What exists

- `app/mrz.py`: ICAO 9303 parser for TD1, TD2, TD3 (passports, all countries), MRV-A and MRV-B (Schengen visa stickers). Check digits, composite check, century logic for dates, OCR look-alike correction in numeric positions, long document numbers on TD1.
- `app/validation.py`: validation flags (`mrz_checksum_failed`, `document_expired`, `ocr_digits_corrected`, ...) and `cross_match` for passport vs visa identity checks.
- `app/engines/`: engine interface plus a Tesseract MRZ baseline. Vendor engines (for example Regula on-prem) and PaddleOCR plug in here after the POC.
- `app/main.py`: FastAPI service with correlation IDs, upload limits, jobs, and no image logging or storage.
- `tests/`: ICAO specimen documents for every format, tamper and OCR-error cases, API tests.
- `samples/` and `docs/API_EXAMPLES.md`: synthetic sample documents (passports from several countries, Schengen visas, TD1 cards) and real request/response examples. Regenerate with `python tools/make_samples.py`.

## Run

```bash
pip install -r requirements-dev.txt
pytest
uvicorn app.main:app --port 8080
```

Try the parser without any OCR:

```bash
curl -s localhost:8080/v1/mrz/parse -H 'content-type: application/json' \
  -d '{"lines":["P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<","L898902C36UTO7408122F1204159ZE184226B<<<<<10"]}'
```

## API

| Endpoint | Purpose |
|---|---|
| `POST /v1/mrz/parse` | Parse and validate MRZ lines you already have |
| `POST /v1/documents/extract` | Upload 1 to 4 JPEG/PNG images, returns a job |
| `GET /v1/jobs/{id}` | Status, fields, per-check results, flags, `needsReview` |

## Not built yet (next slices)

1. Engine POC harness and scorecard (Regula on-prem vs PaddleOCR baseline) on the sample set.
2. Visual-zone extraction for priority passport countries, Schengen visa fields, GCC IDs, US green card front and barcode.
3. Document classifier and image quality scoring (glare, blur, skew).
4. Persistent jobs (PostgreSQL), queue workers, encrypted storage with auto-purge, audit log.
5. Review queue UI and the web capture component.
6. Egress-denied network policy and an air-gapped test run.

## Offline notes

- Build the image on a connected host, push to the internal registry, run air-gapped. Bundle models and fonts locally.
- Tesseract's default model is a weak fit for the OCR-B MRZ font; install `ocrb.traineddata` and set `TESSERACT_LANG=ocrb`.
- The in-memory job store is for development only.

## English / Arabic AI model (trainable)
`OCR_ENGINE=paddle-ocr` enables a PaddleOCR-based engine for English and Arabic lines. Models live in `models/`
and are checksum-verified; see `docs/MODEL_TRAINING.md` for the training workflow, and `GET /v1/models` for status.
