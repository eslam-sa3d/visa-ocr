# User manual

How to set up, run and use the document reader. For the design see `ARCHITECTURE.md`; for hardware see
`SERVER_SIZING.md`.

Status: supervised pilot. A person should confirm results until the real-specimen accuracy test is done.

## 1. What it reads

| Document | Choose in "Document type" | Notes |
|---|---|---|
| Passport | Passport | All countries, MRZ at the bottom |
| Visa sticker (Schengen, UK, US) | Visa sticker | MRZ on the sticker |
| ID card, residence permit, US green card | ID card / green card | Cards with an MRZ, usually on the back |
| Saudi Iqama, GCC ID cards | Iqama / GCC ID (no MRZ) | Always sent to review |

Not supported: US driving licences and state IDs, old non-standard EU cards, PDFs.

## 2. Set up

### Option A. Docker (recommended)

Build on a machine with internet access, then move the image to the offline server.

```bash
git clone https://github.com/eslam-sa3d/visa-ocr && cd visa-ocr

# Small image: Tesseract + RapidOCR. No model download needed.
docker build --build-arg VARIANT=lite -t visa-ocr:lite .

# Full image: adds PaddleOCR (needed for self-training). Fetch the models first.
python tools/fetch_models.py
docker build -t visa-ocr:full .

# Move to the offline server
docker save visa-ocr:lite | gzip > visa-ocr-lite.tar.gz      # copy the file across, then:
docker load < visa-ocr-lite.tar.gz
```

### Option B. Without Docker

Needs Python 3.12 and Tesseract with the English and Arabic language packs.

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-lite.txt     # or requirements.txt for PaddleOCR
```

For a server with no internet, see `OFFLINE_INSTALL.md` (wheelhouse) and `RUNBOOK_CONNECTED_MACHINE.md` (models).

## 3. Run

### One container (demo or small pilot)

```bash
docker run -d --name visa-ocr -p 443:443 \
  -e API_KEYS="$(openssl rand -hex 32)" \
  -e OCR_ENGINE=ensemble \
  -e OCR_DEADLINE_SECONDS=1.7 \
  -v /secure/tls:/data/tls:ro \
  visa-ocr:lite
```

- **TLS.** Put `server.crt` and `server.key` in one folder and mount the folder, as above. Do not mount single files:
  if the path is missing, Docker creates a folder with that name and the service stops with a clear error.
  With no certificate, a self-signed one is generated (browsers warn; test only).
- **API key.** The container refuses to start without `API_KEYS`. For a local test only, use `-e ALLOW_ANONYMOUS=1`.
- **Check it.** `curl -k https://localhost/readyz` should answer `{"status":"ready",...}`.

On a Mac, double-click `run_local.command`: it starts the service on port 8080 and opens the test page.

Without Docker: `OCR_ENGINE=ensemble .venv/bin/python -m uvicorn app.main:app --port 8080`, then open
`http://127.0.0.1:8080/ui`. The service does not reload code by itself: restart it after an update.

### Production (several machines)

```bash
# API nodes
docker run -d -p 443:443 -e API_KEYS=... -e DATABASE_URL=postgresql://user:pass@db/visaocr?sslmode=require \
  -e WORKER_THREADS=0 -e MAX_PENDING_JOBS=4000 -e RATE_LIMIT_PER_MIN=0 -e TRUST_PROXY=1 \
  -v /secure/tls:/data/tls:ro visa-ocr:lite

# Worker nodes: one container per worker process, 2 cores each
docker run -d --cpus 2 -e DATABASE_URL=postgresql://user:pass@db/visaocr?sslmode=require \
  -e OCR_ENGINE=ensemble -e OCR_DEADLINE_SECONDS=1.7 visa-ocr:lite python -m app.worker
```

Worker counts and server sizes are in `SERVER_SIZING.md`.

## 4. Use the web page

Open `https://<server>/ui`.

1. Drop the photo (JPEG or PNG, up to 10 MB, up to 4 images).
2. Choose the document type. "Not sure" works for documents with an MRZ.
3. Optional: enter the declared nationality and the applicant's details. The reader compares them with the document.
4. If the server asks for a key, paste it under "API key".
5. Click **Read document**.

What you see:

| On screen | Meaning |
|---|---|
| Green banner | Every check passed and no field needs a person |
| Amber banner, "Why review is needed" | A person must compare the listed fields with the document |
| "Could not read the document" | The line under the banner gives the real reason and what to do |
| Check digits list | Each ✓ is a field the document's own check digit confirms |
| "OCR digits were corrected" | A look-alike character (O/0, I/1) was fixed and the check digit confirms the fix |

Tips for a good photo: flat, sharp, well lit, no glare, the whole page in frame with the lines of `<<<` fully visible.

### Saving corrections (to improve the reader)

1. Tick **Applicant consented to training use** before reading. Tick it only when the applicant agreed.
2. After the result, the **Corrections for training** panel shows each line the reader located.
3. Fix any wrong character so the line matches the document exactly. Leave correct lines alone.
4. Enter your name or staff ID and click **Save corrections**.

The panel appears only when the server has `TRAINING_STORE_DIR` set and the line positions are known (level pages).

## 5. Use the API

```bash
# 1. Upload. The answer comes at once with a job id.
curl -k -H "X-API-Key: $KEY" -F files=@passport.jpg -F doc_type_hint=passport \
  -F expected_nationality=EGY https://server/v1/documents/extract
# {"jobId":"...","status":"queued","correlationId":"..."}

# 2. Ask for the result (repeat every half second until status is done or failed)
curl -k -H "X-API-Key: $KEY" https://server/v1/jobs/<jobId>
```

Main parts of a finished job:

| Field | Meaning |
|---|---|
| `status` | `queued`, `processing`, `done` or `failed` |
| `result.document` | The fields: `documentNumber`, `surname`, `givenNames`, `nationality`, `birthDate`, `expiryDate`, `sex`, `issuingState`, `checks` |
| `result.mrz` | The MRZ exactly as read: `format` (TD1, TD2, TD3, MRVA, MRVB) and `lines` (2 or 3 strings). Absent for cards without an MRZ |
| `result.needsReview` | `true` when a person must check it |
| `result.flags` | Each problem found: `code`, `severity` (`error` forces review), `message` |
| `result.fieldStatus` | How each unprotected field was verified |
| `result.engine` | Which reader produced the result |
| `error.code`, `error.message` | Why a failed job failed |

Form fields for upload: `files` (1 to 4), `doc_type_hint` (`passport`, `visa`, `id_card`, `iqama`, `gcc_id`),
`expected_nationality`, `expected_surname`, `expected_given_names`, `expected_document_number`,
`expected_birth_date`, `expected_expiry_date` (dates as `YYYY-MM-DD`), `consent_training`.

Other endpoints: `POST /v1/mrz/parse` (check MRZ text you already have), `POST /v1/jobs/{id}/corrections`,
`GET /v1/models`, `GET /healthz`, `GET /readyz`. Full request and response samples are in `API_EXAMPLES.md`.

To test an installation end to end: `python tools/system_test.py --url https://server --key <key>` (functional and
security checks) and `python tools/perf_test.py --url https://server --key <key> --phase latency`. Results of the last
run are in `TEST_REPORT.md`.

Results are deleted after 60 minutes (`JOB_TTL_MINUTES`). Store what you need in the portal.

## 6. Main settings

| Setting | Default | Purpose |
|---|---|---|
| `API_KEYS` | none | Comma-separated keys. Sent as `X-API-Key` |
| `OCR_ENGINE` | `auto` | `ensemble` (Tesseract + RapidOCR), `paddle-ocr`, or `auto` |
| `OCR_DEADLINE_SECONDS` | 0 | Hard limit per read. 1.7 keeps a read within about 2 seconds |
| `WORK_MAX_SIDE` | 2000 | Photos are reduced to this many pixels on the long side |
| `DATABASE_URL` | none | PostgreSQL queue for several machines |
| `JOB_DB_PATH` | none | SQLite file for one machine (Docker sets `/data/jobs.db`) |
| `MAX_PENDING_JOBS` | 32 | Uploads are refused with "busy" beyond this |
| `RATE_LIMIT_PER_MIN` | 60 | Per client. 0 turns it off |
| `MAX_UPLOAD_BYTES` | 10 MB | Per image |
| `JOB_TTL_MINUTES` | 60 | How long results are kept |
| `JOB_THREADS` | half the cores | Jobs read at the same time in one container (no PostgreSQL queue) |
| `OCR_TEXT_SECONDS` | 20 | Time limit for reading a card without an MRZ when `OCR_DEADLINE_SECONDS` is 0 |
| `WORKER_POLL_SECONDS` | 0.2 | How often an idle worker checks the PostgreSQL queue |
| `API_DOCS` | off | Set to 1 to serve `/docs` and `/openapi.json` (they need no key; keep off in production) |
| `TESSERACT_LANG`, `TESSERACT_DATA_DIR` | auto | Tesseract model for the MRZ. An OCR-B model is used automatically when `models/tessdata/ocrb_int.traineddata` (or `ocrb.traineddata`) exists; it reads MRZ print much better than the stock English model |
| `OPTIONAL_DATA_REVIEW` | off | Set to 1 to force review when the two readers differ only inside a passport's optional data (personal number) |
| `BAD_KEY_LIMIT_PER_MIN` | 10 | Wrong-key attempts per client. Applies even when `RATE_LIMIT_PER_MIN` is 0 |
| `TRAINING_STORE_DIR` | none | Turns on saving consented images and corrections |
| `TRAINING_RETENTION_DAYS` | 30 | How long training data is kept |
| `MODELS_DIR` | `./models` | PaddleOCR models and `manifest.json` |
| `TLS_CERT`, `TLS_KEY` | `/data/tls/server.crt`, `.key` | Certificate paths in Docker |

## 7. Self-training

Set up once on the training server (GPU): `sudo sh deploy/install_autotrain.sh`, then edit
`/etc/visa-ocr/autotrain.env`. Every Sunday at 02:00 it retrains if at least 500 new corrected lines exist, and keeps
the new model only if it scores better on real validation lines. Details and the approval step are in
`MODEL_TRAINING.md`. Workers pick up a new model without a restart. This needs the full image.

Line crops and label files made for a training run are deleted when the run ends. Run folders older than `--keep-days`
(30) are removed. To withdraw one applicant's consent:
`python tools/purge_training.py --store-dir <store> --job-id <id> --work-dir <training work folder>`.

## 8. Troubleshooting

| Problem | Cause | Fix |
|---|---|---|
| Container exits: `no API key is configured` | `API_KEYS` not set | Set it, or `ALLOW_ANONYMOUS=1` for a local test |
| Container exits: `... is a directory, not a file` | A mounted certificate path does not exist on the host | Mount the folder that holds `server.crt` and `server.key` |
| `415 type_mismatch` or `bad_type` | The file is not a real JPEG or PNG | Convert it (HEIC and PDF are not accepted) |
| `413 too_large` | Image over 10 MB | Reduce it or raise `MAX_UPLOAD_BYTES` |
| `503 busy` | Too many jobs waiting | With PostgreSQL: add workers or raise `MAX_PENDING_JOBS`. One container: raise `JOB_THREADS` if cores are free, otherwise wait |
| Service does not start: `contains no usable key` | `API_KEYS` holds only commas or spaces | Set a real key. A blank value no longer turns authentication off |
| Correction refused with `409` | The training data for that job was purged | Nothing to do: purged data is not written back |
| `429 rate_limited` | More than `RATE_LIMIT_PER_MIN` requests from one client | Raise the limit or set it to 0 behind the portal |
| `mrz_not_found` | No MRZ visible, or blurred, cut off or sideways | Retake the photo; for cards without an MRZ choose "Iqama / GCC ID" |
| `/readyz` answers `not_ready ... is missing` | `OCR_ENGINE=paddle-ocr` without model files | Fetch the models, or use `ensemble` |
| `/readyz` shows a `warnings` list | `auto` is running on the fallback readers | Fine for the lite image; fetch models for PaddleOCR |
| Reads take several seconds | No hard limit set, or too few cores | Set `OCR_DEADLINE_SECONDS=1.7`; give each worker 2 cores |
| Many documents go to review | Names have no check digit and the two readers disagreed | Send the applicant's form data (`expected_*`) so it can confirm them |
| Changes not visible after an update | The service does not reload code | Restart the container or process |
| zsh: errors when pasting commands | Comments (`#`) or `<placeholders>` in the pasted text | Paste one command at a time, without the comment lines |
