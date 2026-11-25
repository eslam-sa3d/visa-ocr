# Security and operations

| Control | How |
|---|---|
| Authentication | `API_KEYS=key1,key2` (or `API_KEY_SHA256=<hex>`), sent as `X-API-Key` or `Authorization: Bearer`. Unset = open (a warning is logged); a value that holds no usable key (only commas or spaces) stops the service instead of turning authentication off; the Docker image refuses to start without a key unless `ALLOW_ANONYMOUS=1`. Wrong keys are limited separately and more tightly (`BAD_KEY_LIMIT_PER_MIN`, default 10 per client address, also when `RATE_LIMIT_PER_MIN=0`); use long random keys (`openssl rand -hex 32`), a startup warning flags keys under 24 characters. `/healthz`, `/ui` stay open. |
| Rate limit | `RATE_LIMIT_PER_MIN` per key (per IP when open); 429 with `Retry-After`. `TRUST_PROXY=true` only behind your own reverse proxy. |
| Job privacy | Jobs belong to the key that created them; another key gets 404. |
| Retention | Jobs expire after `JOB_TTL_MINUTES` (default 60). Images are not kept after processing (exceptions: consented training storage, and short-lived Tesseract temp crops in the temp directory while a job runs); extracted text lines are not returned unless `INCLUDE_TEXT_LINES=true`. |
| Housekeeping | A background sweep every `HOUSEKEEPING_SECONDS` (300) deletes expired jobs even when the service is idle, and removes training-store folders older than `TRAINING_RETENTION_DAYS` (30; 0 = keep until you purge). SQLite files are created owner-only (0600) with secure delete on. For a consent withdrawal run `python tools/purge_training.py --store-dir <dir> --job-id <id> --work-dir <training work folder>`: it removes the job's folder and its line crops and label rows in training datasets, and leaves a marker so a later correction cannot write the data back (the API answers 409); for a cron job use `--days N`. |
| Store | In-memory by default; `JOB_DB_PATH=/data/jobs.db` for a SQLite file that survives restarts. `DATABASE_URL=postgresql://...` switches to the PostgreSQL store with a work queue (see below). |
| Uploads | Magic-byte check, size cap, 40-megapixel cap, minimum size, aspect-ratio cap (`MAX_IMAGE_ASPECT`, default 20), whole-request body cap (`MAX_FILES` x `MAX_UPLOAD_BYTES` + 1 MB, refused before auth); Tesseract calls time out after `OCR_TIMEOUT_SECONDS` (60); clear error codes. `MAX_PENDING_JOBS` gives 503 when busy. |
| PDF uploads | Accepted on every upload route. The PDF as uploaded goes to the malware scanner (`UPLOAD_SCAN_COMMAND`) first. It is then drawn by PDFium in a separate process started with `python -I` from `/`, with an empty environment and limits on address space (`PDF_RENDER_MEMORY_MB`, 768), CPU (`PDF_RENDER_SECONDS`, 15), wall-clock time, open files and file writes (none). A broken, password-protected, too-large or too-heavy PDF is refused with its own code and can at worst end that process. Only the page images go further, through the same checks as any image; the PDF itself is never queued, stored or kept for training. At most `MAX_PDF_PAGES` (2) pages, each counted toward `MAX_FILES`. |
| TLS | The Docker image serves HTTPS on port 443 itself (see below). Outside Docker: Terminate TLS at your reverse proxy (nginx example below) or use `uvicorn --ssl-keyfile --ssl-certfile`. |

```nginx
server {
  listen 443 ssl;
  ssl_certificate     /etc/ssl/visa-ocr.crt;
  ssl_certificate_key /etc/ssl/visa-ocr.key;
  client_max_body_size 12m;
  location / { proxy_pass http://127.0.0.1:8080; proxy_set_header X-Forwarded-For $remote_addr; }
}
```
Restrict the network path to the portal backend only. The personal data in results is subject to your PDPL retention rules.

## PostgreSQL queue (several workers / hosts)
- `DATABASE_URL=postgresql://user:pass@host/db?sslmode=require` on every API and worker process.
- The API stores the job and returns 202; workers (`WORKER_THREADS`, default 1 per API process, or `python -m app.worker` on separate hosts with `WORKER_THREADS=0` on the API) claim jobs with `FOR UPDATE SKIP LOCKED`.
- A job whose worker died is retried after `JOB_STALE_SECONDS` (300), up to `JOB_MAX_ATTEMPTS` (3), then fails.
- Uploaded images sit in the `jobs` table only while the job runs and are erased when it finishes; all rows are deleted after `JOB_TTL_MINUTES`. Encrypt the database volume.
- Tested with PostgreSQL 16 (`tests/test_pgstore.py`, set `TEST_DATABASE_URL`). Not tested: connection pooling under heavy load, failover.

## HTTPS on port 443
The container listens on 443 with TLS. Provide your certificate:
```
docker run --rm -p 443:443 -v /secure/tls:/data/tls:ro -e API_KEYS=... visa-ocr:0.3     # /data/tls/server.crt + server.key
```
(`TLS_CERT` / `TLS_KEY` change the paths.) With no certificate it generates a self-signed one: browsers warn and clients
need `curl -k`, so use it for testing only. Without Docker: `sudo uvicorn app.main:app --port 443 --ssl-certfile server.crt --ssl-keyfile server.key`
(ports below 1024 need root on macOS/Linux, or put nginx in front as above). On a Mac, Docker Desktop publishes 443 without sudo.

## Second audit (11 Nov 2026)

A full source audit at commit `56f51d3` confirmed 14 low-severity findings and no higher ones. All 14 are fixed, each
with a regression test in `tests/test_security_audit_2.py`:

| Area | Change |
|---|---|
| Authentication | A blank key value stops the service. Wrong-key throttling no longer depends on the general rate limit |
| Review decision | MRZ confidence covers every line used; a padded line has no confidence. Name clean-up keeps real names and any removal goes to review. A printed name one or two letters longer or shorter is a mismatch. Fields filled by a second reader are validated. An ID number that was rebuilt or not found next to its label always goes to review |
| Capacity | Pending-jobs cap is reserved atomically. In-process jobs run on their own pool (`JOB_THREADS`). Image decode runs off the event loop. The card text reader has a time budget and fixed caps. Upload form fields have length limits and are clipped in messages |
| Job queue | A running worker renews its claim; every write carries the attempt number, so a run that lost its claim cannot overwrite the job. Hint and applicant data are erased with the images when a job ends |
| Training data | A correction cannot re-create purged data. Training run folders are owner-only, cleaned when a run ends, and expire |

Still open, because source cannot settle them (see the audit's `NEEDS-VALIDATION.md`): request bodies are read before the
key is checked (put a body-size limit and timeouts on the proxy); PostgreSQL connections were opened per call (fixed after the system test: one reused connection per thread, with a statement time limit);
model promotion is scored on validation lines that reviewers supply (keep `--activate` off until an operator-held
validation set exists). The card parser's slow path on very many text lines is now bounded by `MAX_TEXT_LINES` (250).


## Added after the engineering and ML review

- **Models without a checksum are refused.** A model whose manifest entry has no `sha256` no longer loads and `/readyz`
  reports it. `ALLOW_UNVERIFIED_MODELS=1` allows it for development only. A Tesseract model placed in
  `MODELS_DIR/tessdata` is used only when `SHA256SUMS` beside it lists it and it matches.
- **Signed manifest (optional, recommended).** Set `MANIFEST_HMAC_KEY_FILE` (a file with a long random secret, owner-only)
  on the hosts that register models and on the hosts that serve. The manifest then carries an HMAC-SHA256 over its model
  list; a manifest without a valid signature is refused. Someone who can write the models folder but does not hold the key
  can no longer swap a model and its checksum. Sign an existing manifest by re-saving it with the key set, for example
  `register_model.py --set-metrics <id> --metrics <its metrics.json>`.
- **Model paths stay inside the models folder.** An entry with `../` or an absolute path is refused.
- **Storage encryption must be confirmed.** The service writes images (training store, PostgreSQL queue) and results (job
  database) in plain form and cannot see whether the volume is encrypted. `/readyz` carries a warning until you set
  `DATA_AT_REST_ENCRYPTED=1`, which you do only after putting those paths on encrypted volumes. Treat the warning as a
  go-live blocker.
- **Logs hold no document values.** Errors are logged as type and code location (`ValueError at mrz.py:412 in parse_mrz`),
  not as tracebacks with messages, because a parser's message can repeat a date or a number.
- **Corrections are not trusted.** See docs/MODEL_TRAINING.md, "Controls on the learning loop".
- **Pretrained checkpoints are checked before training.** `finetune.sh` refuses to start unless the SHA-256 of
  `PRETRAINED.pdparams` equals `PRETRAINED_SHA256` (training checkpoints are pickle files and can carry code).
- **The service can encrypt what it stores.** Set `DATA_KEY_FILE` to a file holding a key
  (`python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"`), owner-only and on a
  different volume from the data. Then training-store images and corrections, results and errors in the job database, and
  images, form data, results and errors in the PostgreSQL queue are written encrypted (authenticated: a changed file is
  refused). To rotate, put the new key on the first line and keep the old one below until data written with it has expired.
  An unusable key file stops the service at start. With a key set, `/readyz` no longer shows the storage warning. This
  protects a copied disk, dump or backup; it does not protect against someone who can read the key file.
