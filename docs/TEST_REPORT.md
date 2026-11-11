# System test report

Date: 11 Nov 2026. Code: the commit that adds this file. Tester: black-box over HTTP against a running service, plus
load and security tools. Rerun with `tools/system_test.py` and `tools/perf_test.py`.

Limits of this test: generated documents only (ICAO test data, synthetic fonts), one 2-core 8 GB machine, API, worker and
PostgreSQL on that same machine, Tesseract + RapidOCR readers (no PaddleOCR), no Docker, no real proxy. It does not
replace the real-specimen accuracy test or a load test on the production servers.

## 1. Result

| Area | Checks | Result |
|---|---|---|
| Functional cycles | 20 | 20 pass, in both deployment shapes |
| Security (live) | 13 | 13 pass after 3 fixes |
| Unit and integration suite | 241 | 241 pass (233 + 8 PostgreSQL tests) |
| Dependency audit (`pip-audit`) | 2 requirement files | No known vulnerabilities |
| Static scan (`bandit`) | app, tools, training | No high. 3 medium, all checked: 2 false positives (fixed column names), 1 temp path in a test tool |
| Performance | See section 4 | Meets 2 s at the median; p95 near 2 s; 2 of 20 ID cards unread at the limit |

Defects found by this test and fixed in the same commit:

| # | Found by | Defect | Fix |
|---|---|---|---|
| 1 | Security | `/docs`, `/redoc`, `/openapi.json` open without a key | Off unless `API_DOCS=1` |
| 2 | Security | A job id containing a NUL byte gave HTTP 500 on PostgreSQL | Ids that are not 32 hex characters answer 404 before the store |
| 3 | Security | A streamed upload over the size limit answered 400, not 413 | Answers 413 `body_too_large` |
| 4 | Functional | A cut-off image was logged as an internal error with a stack trace | Reported as `invalid_image`, one log line |
| 5 | Performance | Unreadable page took up to 3.6 s with a 1.7 s limit (MRZ search, then a full text read) | The text read gets only the time left: 2.2 s worst case |
| 6 | Performance | PostgreSQL: a new connection and a DELETE on every status poll. 80 polling clients cut throughput to about 12 a minute and the burst did not finish | One connection per thread, no DELETE on reads: 46 a minute, burst finished |
| 7 | Performance | PostgreSQL: idle workers checked the queue every 0.5 s, adding up to 0.5 s per read | `WORKER_POLL_SECONDS`, default 0.2: median 1.48 s to 1.07 s |

## 2. Functional cycles

Run against one container (SQLite) and against API + separate worker + PostgreSQL with API keys. All pass in both.

| Cycle | Checked |
|---|---|
| Service pages | `/healthz`, `/readyz`, `/`, `/ui`, `/v1/models` |
| Passport, visa MRV-A, visa MRV-B, ID card TD1, ID card TD2 | Upload, poll, every field equal to the source data, every check digit passes |
| Tilted, blurred, noisy passport | Read; never accepted with a wrong field |
| Form data | Matching data confirms names and nationality; differing data raises one flag per field and forces review |
| Expired document | Flagged, sent to review |
| Iqama (no MRZ) | ID number, dates and nationality read; always sent to review |
| Two images in one upload | One job, correct result |
| Blank page | `mrz_not_found` with a usable message |
| `/v1/mrz/parse` | Valid lines, a changed digit, nonsense |
| 10 upload errors | No file, 5 files, text, GIF, PDF, broken JPEG, too small, too narrow, over 10 MB, long form field: right status and code |
| Job lookup | Unknown id 404; result can be read again; correlation id kept |
| Training cycle | Consent, correction saved owner-only, no-consent refused, control characters refused, purge removes data, later correction refused |
| Worker killed mid-job (PostgreSQL) | Job picked up by another worker after 33 s, correct result, attempt 2 |
| Overload | With the cap at 32, 80 uploads at once: 32 accepted and all correct, 48 answered 503; `/healthz` stayed under 0.25 s |

## 3. Security

| Check | Result |
|---|---|
| No key, wrong key on all five `/v1` routes | 401. Valid key works as header and Bearer |
| Job of key A read or corrected with key B | 404, identical to a missing job |
| Job ids | 128-bit random |
| Headers | `nosniff`, `DENY`, `no-referrer` everywhere; `no-store` on `/v1`; content policy on `/ui`; no CORS header |
| Oversized body | 413 for a declared 60 MB and for a streamed 50 MB upload |
| Hostile files | 3.6-gigapixel PNG header refused before decoding; SVG and HTML refused; PNG with appended script and zip bytes read as a plain image |
| Corrupt image | Clean failed job, no stack trace |
| Injection | Path traversal, SQL text, NUL byte, 5,000-character id, script in correlation id and form fields: no effect, store intact |
| Malformed JSON and multipart | 4xx with no internals |
| Methods | DELETE, PUT, PATCH, TRACE refused |
| API docs pages | Closed |
| Wrong-key guessing | 429 within 15 attempts; the valid key keeps working |
| Rate limit | 429 with `Retry-After` |
| TLS (container entrypoint, self-signed) | TLS 1.0 and 1.1 refused; 1.2 and 1.3 accepted; plain HTTP refused; key file owner-only |
| Files at rest | SQLite database 0600; training files 0600, folders 0700 |
| Logs | No names or document numbers in any server log of the run |

Not testable here: the reverse proxy, network rules, database TLS and disk encryption. A request body is still read before
the key is checked (up to the 41 MB cap): set a body limit and timeouts on the proxy.

## 4. Performance

2 cores, `OCR_DEADLINE_SECONDS=1.7`, time as the client sees it (upload until the result is ready).

One request at a time:

| Document | Median | p95 | Slowest | Correct |
|---|---|---|---|---|
| Passport (20) | 0.90 s | 1.93 s | 2.57 s | 20 of 20 |
| Visa MRV-A (10) | 1.18 s | 1.49 s | 1.49 s | 10 of 10 |
| Visa MRV-B (20) | 1.17 s | 1.19 s | 1.28 s | 19 of 20, 1 sent to review |
| ID card TD2 (20) | 1.28 s | 1.39 s | 1.40 s | 20 of 20 |
| ID card TD1 (20) | 1.19 s | 2.05 s | 2.15 s | 17 of 20, 1 sent to review, 2 not read |
| Iqama (10) | 0.75 s | 0.96 s | 0.96 s | 10 of 10 |
| Phone photo, 12 megapixels (10) | 1.10 s | 1.69 s | 1.69 s | 10 of 10 |

The two unread TD1 cards are read correctly with no time limit, in about 6 seconds. That is the cost of the 2-second rule.
No result was accepted with a wrong value anywhere in the test.

Under load (40 mixed documents per level):

| Clients at once | One container: per minute | Median wait | PostgreSQL + 1 worker: per minute | Median wait |
|---|---|---|---|---|
| 1 | 57 | 1.0 s | 52 | 1.1 s |
| 2 | 59 | 2.0 s | | |
| 4 | 61 | 3.8 s | | |
| 8 | 59 | 8.1 s | 60 | 7.7 s |
| 16 | 54 | 17.4 s | | |

Throughput is flat at about 59 documents a minute for 2 cores, as `SERVER_SIZING.md` assumes (2 core-seconds per read).
More clients only lengthen the queue: the 2-second figure holds while uploads per second stay under what the workers can
read. `/healthz` answered in under 60 ms at every level.

Soak, 200 seconds at 2 clients: 199 documents, 0 errors, median 1.94 s in the first half and 2.05 s in the second. Memory of
the service with its Tesseract processes: 1.31 GB average in the first third, 1.35 GB in the last, peak 1.46 GB. This is a
short run: repeat for several hours before ruling out slow growth. The peak is above the 0.65 to 0.8 GB in
`SERVER_SIZING.md` (measured without a backlog) and within its 1.5 GB planning figure.

## 5. Open points

1. Real documents. Every number above is from generated images.
2. Load test on the production servers with the real proxy and a separate database.
3. A soak of several hours.
4. PaddleOCR path and GPU were not exercised.
5. Proxy body limit and timeouts (section 3).
