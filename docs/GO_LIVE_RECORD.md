# Go-live record for release 1.0.0

This is `docs/RELEASE_CHECKLIST.md` filled in for the go-live candidate. "Done" means it was run and passed on the date
shown, with where to find the proof. "Open" names who has to do it. The image is fit for go-live only when every Open line
is closed or has a written exception from the system owner.

Image tag: `visa-ocr:1.0.0-<commit>`. The image starts in production mode by default.
The build does not finish unless both proofs pass, so a release folder that exists has passed them.

## Done by the build (2026-11-22)

| Check | Result | Proof |
|---|---|---|
| Automated tests | 355 pass, 9 skipped (the 9 need PostgreSQL) | `pytest` |
| Static check | clean | `python -m pyflakes app training tools tests deploy` |
| System test, production reader, training on | 34 of 34 | `tools/system_test.py --training` |
| System test, production set-up (`PRODUCTION=1`, key file, AES-256-GCM, TLS, audit log) | 33 of 33, 1 skipped (training is off) | `tools/system_test.py --insecure` against the entrypoint |
| Known vulnerabilities in the 90 locked packages | none | `tools/dependency_audit.py --check`, `docs/sbom.json` |
| Packages installed by hash | yes, `LOCKED=1` is the build default | `Dockerfile` |
| Offline start and read, test mode | PASS at build | `BUILD_INFO.txt` (`tools/offline_smoke.sh`) |
| Production set-up in the real image, offline: keys with roles from a file, a request with no key refused, the portal key barred from listing models, a blank photo turned back with a retake message, stored result encrypted, audit log written, read-only image | PASS at build | `BUILD_INFO.txt` (`tools/production_smoke.sh`) |
| Base image pinned by digest | recorded at build | `BUILD_INFO.txt` |
| Real images, production reader | 432 images; 257 to 259 of 286 passports and visas pass every check digit; none accepted wrongly against the provisional answers; 51 of 75 permit fronts and 7 of 10 Omani cards read from printed text (all sent to review) | `tools/read_folder.py`, `tools/accuracy_real.py` |

Document types read: passports, visas and ID cards from the MRZ side; Saudi, UAE, Kuwait, Qatar, Bahrain and Oman cards
and the front of European, UK and US residence cards from printed text (always reviewed; built against about 85 sample
images, values not yet checked by a person).

Found and fixed while preparing this record:
- The container refused to start when keys came only from `API_KEYS_FILE` (the production set-up).
- An upload of two images was turned back when one of them was blank.
- PaddleOCR did not report where the MRZ lines are, which the correction panel needs.

## Open before go-live

| # | Check | Who | Why it cannot be done by the build |
|---|---|---|---|
| 1 | Tests with PostgreSQL attached (`TEST_DATABASE_URL`) | Server team | Only if PostgreSQL will be used; one container needs none |
| 2 | Time limit and capacity: `tools/suggest_deadline.py` and `tools/load_test.py` on the production server; set `OCR_DEADLINE_SECONDS` and `JOB_THREADS` | Server team | Every timing so far is from a laptop |
| 3 | Accuracy against answers a person confirmed (`review.html`, then `tools/accuracy_real.py --with-form`); fill `docs/ACCEPTANCE_SIGNOFF.md` | Business owner | The answers so far are the service's own readings |
| 4 | Portal shows `retake.message` to the applicant: on the upload call (HTTP 422, `image_unclear`), and for failed jobs (`image_unclear`, `mrz_not_found`, `multiple_documents`) | Portal team | Portal change |
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
