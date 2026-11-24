# Server sizing

What hardware a given number of users needs, for build 1.0.0 with the production readers (PaddleOCR first, then the
Tesseract and RapidOCR pair).

These are planning figures from load tests on a 2-core test machine with generated documents (24 Nov 2026). Nothing here
has been measured on the target servers, with real photos, or with a graphics card. Run `tools/load_test.py` on the real
servers before buying to the last core.

## 1. How the service uses a server

- **It sizes itself.** At start-up the service reads the processor and memory limits of its container and starts one
  reader for each core, as long as each has about 2 GB of memory. `GET /readyz` shows the number. `READER_PROCESSES`
  forces a number.
- **A reader reads one document at a time.** Each reader is a separate process with its own copy of the models.
- **Cores decide capacity, not memory.** A reader uses about 1 GB at rest and up to 1.6 GB while reading.
- **Uploads beyond capacity wait.** 32 uploads may wait for each reader (`MAX_PENDING_JOBS`); beyond that the service
  answers "busy, try again shortly" (HTTP 503).
- **An unusable photo never enters the queue.** It is refused on the upload call itself.

## 2. Measured

One container on 2 cores, mixed generated documents (passports, phone photos of passports, iqama cards, ID cards):

| Set-up | Clients at once | Documents per minute | Median wait | 95 in 100 within |
|---|---|---|---|---|
| 2 readers, build as shipped | 1 | 16.6 | 3.5 s | 6.4 s |
| | 4 | 34.2 | 7.1 s | 9.2 s |
| | 16 | 33.1 | 27.5 s | 30.6 s |
| 1 reader limited to 1 core and 2 GB (before the one-language reading) | 1 | 8.2 | 6.2 s | 17.3 s |
| | 16 | 7.7 | 116.2 s | 130.8 s |

- **About 17 documents a minute for each core**, flat under load: more clients only lengthen the queue.
- **Memory:** 0.6 GB for the service itself and about 1.05 GB for each reader after the test; a single reader held to 2 GB
  peaked at 1.6 GB.
- **Shared queue:** two service instances on one PostgreSQL database read 32 of 32 uploads sent to the first and asked for
  at the second.
- **Real photos are slower.** On a MacBook Pro, 432 real photos took a median of 8.3 seconds each before these changes,
  against about 3.5 seconds for generated documents here. Until real photos are timed on the target servers, plan with half
  the figures above: **about 8 documents a minute for each core**.

## 3. The sizing rule

Hardware follows uploads per second at peak, not the number of users signed in.

```
documents a minute needed = users x documents per user per minute
cores to provide          = documents a minute needed / (documents a minute per core x 0.6)
memory                    = cores x 2 GB, plus 1 GB for each container
```

The 0.6 keeps the readers about 60% busy, so an upload rarely waits for more than the read in front of it.

## 4. Cores for a number of concurrent users

| Use | Documents per user per minute | 500 users | 1,000 users | 2,000 users |
|---|---|---|---|---|
| At 17 a minute per core (generated documents, as measured) | | | | |
| Typical: 2 documents in a 10-minute session | 0.2 | 10 | 20 | 40 |
| Busy: 1 document every 2 minutes | 0.5 | 25 | 50 | 99 |
| Heavy: 1 document every minute | 1.0 | 50 | 99 | 197 |
| At 8 a minute per core (planning figure for real photos) | | | | |
| Typical | 0.2 | 21 | 42 | 84 |
| Busy | 0.5 | 53 | 105 | 209 |
| Heavy | 1.0 | 105 | 209 | 417 |

A burst is absorbed by the queue. If every user uploads in the same moment, the last one waits for the queue to clear:
users divided by the documents a minute of all cores together.

## 5. What a given server carries

Concurrent users served at about the single-read time, for 24 cores (for example 4 containers of 6 cores and 18 GB):

| Use | At 17 a minute per core | At 8 a minute per core |
|---|---|---|
| Typical | about 1,220 | about 575 |
| Busy | about 490 | about 230 |
| Heavy | about 245 | about 115 |

## 6. Layout for more than one server

| Role | Notes |
|---|---|
| Service containers | As many as the cores require. Each sizes itself to its own limits |
| PostgreSQL | One primary and one standby. Set `DATABASE_URL` on every container so they share one queue |
| Load balancer | TLS, health check on `/readyz`. Any container answers for any job |
| Training machine | Only for retraining; can be offline. See `docs/MODEL_TRAINING.md` |

## 7. What shortens a read

| Change | Effect | State |
|---|---|---|
| Portal sends `doc_type_hint` | Skips the search for an MRZ on Gulf cards and iqamas | Portal change |
| One-language reading (`PADDLE_ONE_LANGUAGE`) | Reading step about 58% less processor time on 22 real photos, no reading changed | On by default |
| Time limit on the second card reading (`OCR_SECOND_CARD_SECONDS`) | Up to about 3 seconds less on cards with a missing field | On by default |
| Readers loaded at start-up (`WARM_UP`) | The first upload does not wait for the models | On by default |
| Graphics card for the main reader | Not measured | Open |

## 8. Still to measure on the target servers

1. Seconds per read for a passport, a Gulf card and a residence permit, with real photos.
2. Documents a minute for one container at its real size, at rising client counts.
3. The same with all containers and the PostgreSQL queue.
4. Peak memory of a reader with large real photos.
