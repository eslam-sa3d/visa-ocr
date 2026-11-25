# Go-live record for release 1.0.0

This is `docs/RELEASE_CHECKLIST.md` filled in for the go-live candidate. "Done" means it was run and passed on the date
shown, with where to find the proof. "Open" names who has to do it. The image is fit for go-live only when every Open line
is closed or has a written exception from the system owner.

Image tag: `visa-ocr:1.0.0-<commit>`. The image starts in production mode by default.
The build does not finish unless both proofs pass, so a release folder that exists has passed them.

## Done by the build (2026-11-24)

| Check | Result | Proof |
|---|---|---|
| Automated tests | 415 pass, 9 skipped (the 9 need PostgreSQL) | `pytest` |
| Static check | clean | `python -m pyflakes app training tools tests deploy` |
| System test, production reader, training on | 34 of 34 | `tools/system_test.py --training` |
| System test, production set-up (`PRODUCTION=1`, key file, AES-256-GCM, TLS, audit log) | 33 of 33, 1 skipped (training is off) | `tools/system_test.py --insecure` against the entrypoint |
| Known vulnerabilities in the 90 locked packages | none | `tools/dependency_audit.py --check`, `docs/sbom.json` |
| Packages installed by hash | yes, `LOCKED=1` is the build default | `Dockerfile` |
| Offline start and read, test mode | PASS at build | `BUILD_INFO.txt` (`tools/offline_smoke.sh`) |
| Production set-up in the real image, offline: keys with roles from a file, a request with no key refused, the portal key barred from listing models, a blank photo turned back with a retake message, stored result encrypted, audit log written, read-only image | PASS at build | `BUILD_INFO.txt` (`tools/production_smoke.sh`) |
| Base image pinned by digest | recorded at build | `BUILD_INFO.txt` |
| Real images, production reader | 432 images on a laptop, 25 Nov, build a5bb10d (after the security fixes): 258 of 286 passports and visas pass every check digit, 75 accepted without review (79 on 24 Nov before the fixes: 5 readings whose names only the readers' agreement had confirmed now go to review, 1 now accepted; no status or check-digit result changed; with the form names sent, as the portal will, names can still be confirmed); 64 of 75 residence-card fronts and 67 of 71 Gulf cards read from printed text (all sent to review); 25 not read and turned back with a retake message (16 no MRZ found or cut off, 8 unclear, 1 unreadable file). Median 6.3 s a document, 95% within 19.3 s. The values read have not yet been confirmed by a person (item 3 below) | `tools/read_folder.py`, `tools/accuracy_real.py` |
| Readers sized to the container | One reader process per core the container is given, each needing 2 GB; no fixed limit in the image. Tried at 1 core with 2 GB and at 2 cores | `tests/test_readers.py`, `docs/SERVER_SIZING.md` |
| Load, generated documents, 2-core test machine | About 17 documents a minute for each core; uploads beyond capacity wait, then are turned back with "busy" (HTTP 429), none lost | `tools/capacity_test.py`, `docs/SERVER_SIZING.md` |
| Two instances on one PostgreSQL queue | 32 of 32 uploads read once each | `docs/SERVER_SIZING.md` |
| Unclear photo refused at upload | HTTP 422 `image_unclear` with the retake message, in under 2 seconds | `tests/test_image_clarity.py`, `tools/production_smoke.sh` |
| Passport-only endpoint, card-only endpoint and personal-photo check | `POST /v1/passports/extract`, `POST /v1/cards/extract`, `POST /v1/photos/check` | `tests/test_photo_and_passport.py`, `docs/API_EXAMPLES.md` |

Document types read: passports, visas and ID cards from the MRZ side; Saudi, UAE, Kuwait, Qatar, Bahrain and Oman cards
and the front of European, UK and US residence cards from printed text (always reviewed; built against about 85 sample
images, values not yet checked by a person). The personal-photo check has been tried on very few photos and its limits
(`PHOTO_*`) are not yet set to a MOFA photo standard.

Found and fixed while preparing this record:
- The container refused to start when keys came only from `API_KEYS_FILE` (the production set-up).
- An upload of two images was turned back when one of them was blank.
- PaddleOCR did not report where the MRZ lines are, which the correction panel needs.

## Open before go-live

| # | Check | Who | Why it cannot be done by the build |
|---|---|---|---|
| 1 | Tests with PostgreSQL attached (`TEST_DATABASE_URL`) | Server team | Only if PostgreSQL will be used; one container needs none |
| 2 | Time limit and capacity: `tools/suggest_deadline.py` and `tools/capacity_test.py` on the production servers with real photos (`DEPLOYMENT.txt`, step 6); set `OCR_DEADLINE_SECONDS`; size the servers from the measured documents a minute | Server team | Every timing so far is from a laptop or a 2-core test machine; real photos read at about half the speed of generated ones |
| 3 | Accuracy against answers a person confirmed (`review.html`, then `tools/accuracy_real.py --with-form`); fill `docs/ACCEPTANCE_SIGNOFF.md` | Business owner | The answers so far are the service's own readings |
| 4 | Portal shows `retake.message` to the applicant: on the upload call (HTTP 422, `image_unclear`), for failed jobs (`image_unclear`, `mrz_not_found`, `multiple_documents`, `not_a_passport`, `not_a_card`, `card_not_found`), and `result.retake` on a reading (failing check digit, or passport page cut off) | Portal team | Portal change |
| 5 | Organisation certificate mounted; keys made with `deploy/production/make_keys.py` and stored in the key management system; `data.key` on a different volume from the data | Server team | Site secrets |
| 6 | Audit log arriving in the SIEM; a malware scanner chosen (`UPLOAD_SCAN_COMMAND`) or a written exception | Cybersecurity | Site systems |
| 7 | Second-person code review against `docs/SECURE_CODING.md`: date ______ by ______ | Someone other than the author | Independence |
| 8 | Penetration test of the portal including this service: date ______ by ______; findings closed | Outside party | Independence |
| 9 | Data owner approves keeping consented document images for training (30 days, encrypted), and the portal asks the applicant for that consent | Data owner, portal team | Policy decision |
| 10 | Pilot on real traffic with every result reviewed, then the acceptance numbers signed | Business owner | `docs/GO_LIVE_PLAN.md` |

Training-data collection is ON in the production settings (`TRAINING_STORE_DIR=/data/training`): consented uploads and
reviewers' corrections are stored encrypted for 30 days. No model changes by itself: retraining runs on the training
machine, a correction needs a second person's approval, and a candidate replaces the model in use only when it beats it
on a golden set that a person has confirmed. Until that golden set exists, candidates are built but never activated.
The data owner's approval for keeping consented images is needed before go-live (item below).

Approved by (system owner): ____________   Cybersecurity: ____________   Date: ____________
