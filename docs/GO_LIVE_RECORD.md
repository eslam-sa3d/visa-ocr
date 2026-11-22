# Go-live record for release 0.3.0

This is `docs/RELEASE_CHECKLIST.md` filled in for the go-live candidate. "Done" means it was run and passed on the date
shown, with where to find the proof. "Open" names who has to do it. The image is fit for go-live only when every Open line
is closed or has a written exception from the system owner.

Image tag: `visa-ocr:0.3.0-<commit>` (the commit is in the image tag and in `release/README.txt`)

## Done by the build (2026-11-22)

| Check | Result | Proof |
|---|---|---|
| Automated tests | 345 pass, 9 skipped (the 9 need PostgreSQL) | `pytest` |
| Static check | clean | `python -m pyflakes app training tools tests deploy` |
| System test, production reader, training on | 34 of 34 | `tools/system_test.py --training` |
| System test, production set-up (`PRODUCTION=1`, key file, AES-256-GCM, TLS, audit log) | 33 of 33, 1 skipped (training is off) | `tools/system_test.py --insecure` against the entrypoint |
| Known vulnerabilities in the 90 locked packages | none | `tools/dependency_audit.py --check`, `docs/sbom.json` |
| Packages installed by hash | yes, `LOCKED=1` is the build default | `Dockerfile` |
| Offline start and read | PASS | `tools/offline_smoke.sh` on the build host |
| Real images, production reader | 432 images; 257 to 259 of 286 passports and visas pass every check digit; none accepted wrongly against the provisional answers | `tools/read_folder.py`, `tools/accuracy_real.py` |

Found and fixed while preparing this record:
- The container refused to start when keys came only from `API_KEYS_FILE` (the production set-up).
- An upload of two images was turned back when one of them was blank.
- PaddleOCR did not report where the MRZ lines are, which the correction panel needs.

## Open before go-live

| # | Check | Who | Why it cannot be done by the build |
|---|---|---|---|
| 1 | `tools/production_smoke.sh <image> <sample>` prints PASS | Build host | Needs Docker; proves the production set-up in the real image |
| 2 | Tests with PostgreSQL attached (`TEST_DATABASE_URL`) | Server team | Only if PostgreSQL will be used; one container needs none |
| 3 | Time limit and capacity: `tools/suggest_deadline.py` and `tools/load_test.py` on the production server; set `OCR_DEADLINE_SECONDS` and `JOB_THREADS` | Server team | Every timing so far is from a laptop |
| 4 | Accuracy against answers a person confirmed (`review.html`, then `tools/accuracy_real.py --with-form`); fill `docs/ACCEPTANCE_SIGNOFF.md` | Business owner | The answers so far are the service's own readings |
| 5 | Portal shows `retake.message` to the applicant and handles `image_unclear` | Portal team | Portal change |
| 6 | Organisation certificate mounted; keys made with `deploy/production/make_keys.py` and stored in the key management system; `data.key` on a different volume from the data | Server team | Site secrets |
| 7 | Audit log arriving in the SIEM; a malware scanner chosen (`UPLOAD_SCAN_COMMAND`) or a written exception | Cybersecurity | Site systems |
| 8 | Base image pinned by digest at build; digest recorded here: ____________ | Build host | Recorded at build time |
| 9 | Second-person code review against `docs/SECURE_CODING.md`: date ______ by ______ | Someone other than the author | Independence |
| 10 | Penetration test of the portal including this service: date ______ by ______; findings closed | Outside party | Independence |
| 11 | Pilot on real traffic with every result reviewed, then the acceptance numbers signed | Business owner | `docs/GO_LIVE_PLAN.md` |

The training loop stays off at go-live (`TRAINING_STORE_DIR` unset). It is turned on only after item 4 has produced a
golden set.

Approved by (system owner): ____________   Cybersecurity: ____________   Date: ____________
