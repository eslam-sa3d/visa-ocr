# Visa Portal offline document OCR

Reads passports, visa stickers, ID cards, the Saudi Iqama and GCC ID cards (English and Arabic) from a photo, and returns
the fields, the check results and whether a person must review them. Runs on your own servers with no internet access.

**Status (11 Nov 2026): supervised pilot.** Accuracy has been measured on generated documents only. See
`docs/GO_LIVE_PLAN.md` for what is still needed before automatic decisions.

## Quick start

```bash
pip install -r requirements-dev.txt
pytest                                                   # 190 pass, 6 skipped
OCR_ENGINE=ensemble uvicorn app.main:app --port 8080     # then open http://127.0.0.1:8080/ui
```

Docker: `docker build --build-arg VARIANT=lite -t visa-ocr:lite .` (small, no PaddleOCR) or `docker build -t visa-ocr .`
(full). Setup, running and use are in `docs/USER_MANUAL.md`.

## Documents

| Document | Read it for |
|---|---|
| `docs/USER_MANUAL.md` | Set up, run, use the page and the API, settings, troubleshooting |
| `docs/ARCHITECTURE.md` | Components, how one read works, data and privacy, limits |
| `docs/SERVER_SIZING.md` | Servers, cores, memory and storage for 2,000 concurrent users |
| `docs/GO_LIVE_PLAN.md` | Measured speed, open items and owners |
| `docs/ACCEPTANCE_SIGNOFF.md` | Thresholds for the business owner to sign |
| `docs/MODEL_TRAINING.md` | Fine-tuning and automatic retraining |
| `docs/RUNBOOK_CONNECTED_MACHINE.md` | Fetching models and training on a connected machine |
| `docs/OFFLINE_INSTALL.md` | Installing on a server with no internet |
| `docs/SECURITY.md` | API keys, TLS, limits, data handling |
| `docs/API_EXAMPLES.md` | Request and response samples |
| `docs/ACCURACY_REVIEW.md`, `docs/TEST_PLAN.md`, `docs/GCC_ID_COVERAGE.md` | Accuracy work and test coverage |

## What is in the box

- `app/`: the service. `mrz.py` (ICAO 9303: TD1, TD2, TD3, MRV-A, MRV-B), `visual_zone.py` and `gcc_ids.py` (cards
  without an MRZ), `engines/` (Tesseract, RapidOCR, PaddleOCR and their combinations), `pipeline.py`, `main.py` (API).
- `training/`: dataset building, fine-tuning, scoring, model registration, `auto_train.py` (scheduled retraining).
- `deploy/`: systemd timer and installer for retraining.
- `tools/`: test document generators, accuracy runs, load test, model download.
- `tests/`: 190 tests, including ICAO specimen documents for every format.

## API in one look

| Endpoint | Purpose |
|---|---|
| `POST /v1/documents/extract` | Upload 1 to 4 JPEG or PNG images; returns a job id |
| `GET /v1/jobs/{id}` | Status, fields, checks, flags, `needsReview` |
| `POST /v1/mrz/parse` | Check MRZ text you already have |
| `POST /v1/jobs/{id}/corrections` | Reviewer corrections (with applicant consent) |
| `GET /v1/models`, `/healthz`, `/readyz` | Model status and health |

## Not supported

US driving licences and state IDs, old non-standard EU cards, PDF and HEIC uploads.
