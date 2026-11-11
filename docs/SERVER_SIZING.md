# Server sizing

What hardware the target needs: reads within 2 seconds, 2,000 concurrent users, offline.

These are planning figures from measurements on a 2-core test machine with generated documents. Nothing here has been
load-tested. Run `tools/load_test.py` on the real servers before buying to the last core.

## 1. Measured inputs

| Figure | Value | Source |
|---|---|---|
| Read time, MRZ documents, phone-sized photo (4000 px) | median 1.5 s, slowest 2.1 s | 27 documents, `OCR_DEADLINE_SECONDS=1.7` |
| Read time, MRZ documents, 1100 px image | median 1.1 s, slowest 1.6 s | 60 documents, same setting |
| Read time, Iqama and GCC cards | median 0.8 s | 40 cards |
| CPU per read | about 2 core-seconds | Both readers run in parallel on 2 cores |
| Worker memory | 0.65 to 0.8 GB peak | Tesseract + RapidOCR, phone-sized photos |
| Stored image after reduction | 0.5 to 1 MB | 4000 px JPEG reduced to 2000 px |
| API work per image (validate, reduce) | about 0.1 core-seconds | |

Not measured: PaddleOCR on CPU or GPU on the target servers, and anything at production load.

## 2. The sizing rule

Hardware follows uploads per second at peak, not the number of users signed in.

```
busy cores        = uploads per second x 2
cores to provide  = busy cores / 0.6        (60% load keeps queue wait short)
worker processes  = cores to provide / 2
memory            = worker processes x 1.5 GB (planning figure, with headroom)
```

## 3. Three load cases for 2,000 concurrent users

| Case | Assumption | Uploads/s | Busy cores | Cores to provide | Worker processes |
|---|---|---|---|---|---|
| A. Typical | 2 documents per user in a 10-minute session | 7 | 14 | 24 | 12 |
| B. Busy (plan for this) | 1 document per user every 2 minutes | 17 | 34 | 56 | 28 |
| C. Heavy | 1 document per user every minute | 33 | 66 | 112 | 56 |

A burst is absorbed by the queue: in case B, 2,000 uploads arriving in the same moment clear in about 2 minutes.
During a burst the wait is queue time, not reading time. If every read must return within 2 seconds even in a burst,
size for the burst rate instead.

## 4. Recommended servers for case B

| Role | Count | Each | Notes |
|---|---|---|---|
| Worker nodes | 4 + 1 spare | 16 cores, 32 GB RAM, 50 GB SSD | 7 worker processes per node. The spare covers one node down |
| API nodes | 2 | 8 cores, 16 GB RAM, 30 GB SSD | `WORKER_THREADS=0`. Either one can carry the load alone |
| PostgreSQL | 1 primary + 1 standby | 8 cores, 32 GB RAM, 200 GB SSD | Queue and results |
| Training server | 1 | 1 GPU with 16 to 24 GB memory, 16 cores, 64 GB RAM, 500 GB SSD | Only for retraining. Can be offline |
| Shared models folder | 1 | 5 GB | Read-only mount on every worker |
| Load balancer | Existing | | TLS, health check on `/readyz` |

Totals: about 130 cores, 320 GB RAM and one GPU.

For case A, 2 + 1 worker nodes are enough. For case C, 7 + 1.

No GPU is needed for reading with the Tesseract and RapidOCR pair. PaddleOCR on a GPU may need fewer worker cores,
but it has not been timed.

## 5. Storage

| What | Size | Basis |
|---|---|---|
| Queue (images waiting) | 2 to 4 GB peak | 2,000 jobs waiting, 1 to 2 images of 1 MB each. Erased when a job finishes |
| Results | Under 1 GB | About 5 KB per job, kept 60 minutes |
| PostgreSQL write-ahead log at peak | About 60 GB per hour of peak | 17 MB/s of image writes. Size the log volume and archiving for it |
| Training store | consented documents per day x 1 MB x 30 days | Example: 2,500 a day gives 75 GB |
| Docker image per node | 2 to 3 GB full, about 0.5 to 0.7 GB lite | Estimates, not built here |
| Each training run | 5 to 20 GB | Dataset, checkpoints, exported model |

Without the image reduction (`WORK_MAX_SIDE=0`) queue and log sizes are several times larger.

## 6. Network

Case B with 2 to 3 MB phone photos is about 50 MB/s inbound at peak (400 Mbit/s). Use 1 Gbit/s or better between the
load balancer, API nodes and PostgreSQL. If the portal reduces photos to 2000 pixels before upload, this drops to
about a third.

## 7. Settings that must change for this load

| Setting | Default | Set to | Why |
|---|---|---|---|
| `DATABASE_URL` | empty | the PostgreSQL address | Shared queue across nodes |
| `MAX_PENDING_JOBS` | 32 | 4000 | The default rejects uploads once 32 are waiting |
| `RATE_LIMIT_PER_MIN` | 60 | 0, or a high value | The portal backend is one client; 60 a minute would block it |
| `WORKER_THREADS` | 1 | 0 on API nodes | Reading happens on worker nodes |
| `OCR_DEADLINE_SECONDS` | 0 | 1.7 | Keeps each read within about 2 seconds |
| `OCR_ENGINE` | auto | `ensemble`, or `auto` with models installed | Match what was measured |
| `TRUST_PROXY` | off | on | Client addresses come from the load balancer |

Pin cores per container with `docker run --cpuset-cpus`, or `--cpus 2` per worker container.

## 8. What changes these numbers

- Real photos: lower quality means more second attempts, so more CPU per read.
- Cards and multi-image uploads: each image is read separately.
- PaddleOCR as the main reader: different CPU and memory per read; measure before sizing.
- A real load test: replace every figure in section 1 with the measured one.
