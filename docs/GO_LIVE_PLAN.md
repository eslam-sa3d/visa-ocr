# Go-live plan

Status on 11 Nov 2026: supervised pilot only. This page lists what is done, what is measured, and what is still open.
All accuracy figures are on generated documents, not real ones.

## 1. Speed (target: no read over 2 seconds)

Measured on a 2-core machine, CPU engines (Tesseract + RapidOCR), 60 generated documents, one at a time:

| Setting | Median | Slowest | Fully correct | Accepted without review | Wrongly accepted |
|---|---|---|---|---|---|
| Before the speed work | 2.4 to 6.9 s by type | 12.6 s | - | - | 0 |
| Default (no hard limit) | 1.1 s | 6.6 s | 54/60 | 21 | 0 |
| `OCR_DEADLINE_SECONDS=2` | 1.1 s | 1.9 s | 53/60 | 19 | 0 |

Iqama and GCC cards: median 0.8 s.

What changed: the two readers run side by side; the page tilt is estimated first instead of trying every angle; a stable
reading stops the search; the second reader no longer enlarges pages that are already large; the slow MRZ search is
skipped for cards that have no MRZ.

`OCR_DEADLINE_SECONDS=2` is a hard limit: each reader stops starting new attempts in time and returns the best reading
it has. The cost is small (one fewer document fully correct, two fewer accepted without review in the run above); a
reading that is not perfect goes to review as always. Sideways scans are not searched under a hard limit.
The limit covers OCR time only, not upload, queue wait or network.

## 2. Capacity (target: 2,000 concurrent users)

Not load-tested. Sizing rule: cores = uploads per second x CPU-seconds per read (about 1.5 to 2 with both readers).
Example: 2,000 users each uploading once every 2 minutes = 17 uploads/s = about 35 cores busy, so plan about 50 to 60
with headroom. Needs `DATABASE_URL` (PostgreSQL queue), several `python -m app.worker` processes per machine, and a
raised `MAX_PENDING_JOBS`. Pin cores with `docker run --cpuset-cpus`. GPU count for PaddleOCR is not measured.

## 3. Still open

| Item | Owner | Needed |
|---|---|---|
| Real specimens, about 1,200, all five document groups (C1, C2, D6, D7) | Program team | Consent and collection |
| Accuracy gate on real specimens, 99% target | Engineering | The specimens above |
| Load test at real peak on target servers | Engineering + infra | Servers, PostgreSQL |
| PaddleOCR timing on a GPU server | Infra | One GPU server |
| Fine-tuning (B3), only if the real-specimen result is short | Engineering | About 5,000 corrected lines, GPU |
| Automatic retraining (`training/auto_train.py`): built and tested with the training step replaced; first real run on a GPU server still to do | Engineering + infra | GPU server, pretrained weights, 500+ corrected lines |
| US driving licences, old non-standard EU cards | Business owner | Decide if in scope |
| Sign-off of thresholds (E2) | Business owner | `docs/ACCEPTANCE_SIGNOFF.md` |
