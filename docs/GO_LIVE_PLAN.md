# Go-live plan

Status on 11 Nov 2026: supervised pilot only. This page lists what is done, what is measured, and what is still open.
All accuracy figures are on generated documents, not real ones.

## 1. Speed (target: no read over 2 seconds)

Measured on a 2-core machine, CPU readers (Tesseract + RapidOCR), one document at a time:

| Test | Setting | Median | Slowest | Fully correct | Wrongly accepted |
|---|---|---|---|---|---|
| 60 documents, 1100 px | Before the speed work | 2.4 to 6.9 s by type | 12.6 s | - | 0 |
| 60 documents, 1100 px | No hard limit | 1.1 s | 6.5 s | 54/60 | 0 |
| 60 documents, 1100 px | `OCR_DEADLINE_SECONDS=1.7` | 1.1 s | 1.6 s | 53/60 | 0 |
| 27 documents, phone-sized 4000 px | Before image reduction | 6.9 s | 12.9 s | 27/27 | - |
| 27 documents, phone-sized 4000 px | `OCR_DEADLINE_SECONDS=1.7` | 1.5 s | 2.1 s | 24/27 | - |

Iqama and GCC cards: median 0.8 s.

What changed: the two readers run side by side; the page tilt is estimated first; a stable reading stops the search;
photos are reduced to 2000 pixels before reading (`WORK_MAX_SIDE`); the second reader starts at 1200 pixels wide; the
slow MRZ search is skipped for cards without an MRZ.

`OCR_DEADLINE_SECONDS` is a hard limit: each reader stops starting new attempts in time and returns the best reading it
has. A reading that is not perfect goes to review as always. Sideways scans are not searched under a hard limit. The
limit covers reading time only, not upload, queue wait or network. On phone-sized photos the slowest read was 2.1 s,
slightly over the target on this 2-core machine. The 4000 px test photos are enlarged generated documents, not real
phone photos.

## 2. Capacity (target: 2,000 concurrent users)

Not load-tested. The sizing rule, three load cases and the recommended servers are in `SERVER_SIZING.md`. In short, for
one upload per user every 2 minutes: 5 worker nodes of 16 cores, 2 API nodes, PostgreSQL with a standby, and one GPU
server for retraining.

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

## 4. Docker image size

`docker build --build-arg VARIANT=lite -t visa-ocr:lite .` builds without PaddleOCR (Tesseract + RapidOCR only, the
pair all the figures on this page were measured with). Estimated about 0.5 to 0.7 GB instead of roughly 2 GB or more
for the full image; neither size has been measured, because no Docker daemon was available here. The lite image
cannot use fine-tuned PaddleOCR models, so automatic retraining has no effect on it.
