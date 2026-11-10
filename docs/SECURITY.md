# Security and operations

| Control | How |
|---|---|
| Authentication | `API_KEYS=key1,key2` (or `API_KEY_SHA256=<hex>`), sent as `X-API-Key` or `Authorization: Bearer`. Unset = open (a warning is logged); the Docker image refuses to start without a key unless `ALLOW_ANONYMOUS=1`. Wrong keys are limited separately and more tightly (`BAD_KEY_LIMIT_PER_MIN`, default 10 per client address); use long random keys (`openssl rand -hex 32`), a startup warning flags keys under 24 characters. `/healthz`, `/ui` stay open. |
| Rate limit | `RATE_LIMIT_PER_MIN` per key (per IP when open); 429 with `Retry-After`. `TRUST_PROXY=true` only behind your own reverse proxy. |
| Job privacy | Jobs belong to the key that created them; another key gets 404. |
| Retention | Jobs expire after `JOB_TTL_MINUTES` (default 60). Images are not kept after processing (exceptions: consented training storage, and short-lived Tesseract temp crops in the temp directory while a job runs); extracted text lines are not returned unless `INCLUDE_TEXT_LINES=true`. |
| Housekeeping | A background sweep every `HOUSEKEEPING_SECONDS` (300) deletes expired jobs even when the service is idle, and removes training-store folders older than `TRAINING_RETENTION_DAYS` (30; 0 = keep until you purge). SQLite files are created owner-only (0600) with secure delete on. For a consent withdrawal run `python tools/purge_training.py --store-dir <dir> --job-id <id>`; for a cron job use `--days N`. |
| Store | In-memory by default; `JOB_DB_PATH=/data/jobs.db` for a SQLite file that survives restarts. `DATABASE_URL=postgresql://...` switches to the PostgreSQL store with a work queue (see below). |
| Uploads | Magic-byte check, size cap, 40-megapixel cap, minimum size, aspect-ratio cap (`MAX_IMAGE_ASPECT`, default 20), whole-request body cap (`MAX_FILES` x `MAX_UPLOAD_BYTES` + 1 MB, refused before auth); Tesseract calls time out after `OCR_TIMEOUT_SECONDS` (60); clear error codes. `MAX_PENDING_JOBS` gives 503 when busy. |
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
