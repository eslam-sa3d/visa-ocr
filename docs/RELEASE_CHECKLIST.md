# Release checklist (NCA ECC 1-6-2, 1-6-3)

Copy this file into the release record, fill it in and have it signed. A release with an unticked box needs a written
exception from the system owner.

Release: ____________  Commit: ____________  Date: ____________  Prepared by: ____________

## Code and tests
- [ ] `pytest` passes with PostgreSQL attached (`TEST_DATABASE_URL` set); no test was removed or skipped to make it pass
- [ ] `python -m pyflakes app training tools tests` is clean
- [ ] `tools/system_test.py` passes against the release candidate (functional and security)
- [ ] `REAL_SAMPLES_DIR=... pytest -k real_specimens` passes: no wrongly accepted field
- [ ] Changes were reviewed by a second person against `docs/SECURE_CODING.md`

## Components (ECC 1-6-3-2, 2-10-3)
- [ ] `python tools/dependency_audit.py --check` reports no known vulnerability, or each one has a recorded decision
- [ ] `docs/sbom.json` regenerated; no new licence that legal has not accepted
- [ ] Lock files regenerated on the target Python version if `requirements*.txt` changed
- [ ] Base image pinned by digest in the build command; the digest is recorded here: ____________

## Build and offline proof
- [ ] Image built with `LOCKED=1`
- [ ] `tools/offline_smoke.sh <image> <sample>` prints PASS with `status: done`
- [ ] `python tools/make_checksums.py` run on everything carried to the offline site; checked on arrival

## Configuration (ECC 2-8, 2-15)
- [ ] `PRODUCTION=1` and the service starts
- [ ] API keys come from `API_KEYS_FILE` with roles and expiry dates; no shared "full" key
- [ ] `DATA_KEY_FILE` set, on a different volume from the data; `DATA_CIPHER` as the classification requires
- [ ] `MANIFEST_HMAC_KEY_FILE` set; `GET /v1/models` shows every active model `verified`
- [ ] The organisation's TLS certificate is mounted; `TLS_CIPHERS` matches the cryptographic standard
- [ ] `AUDIT_LOG_FILE` set and arriving in the SIEM
- [ ] `UPLOAD_SCAN_COMMAND` set and a test file is rejected

## Independent checks (ECC 2-11, 1-6-2)
- [ ] Source code security review by someone other than the author: date ______ by ______
- [ ] Penetration test of the portal including this service: date ______ by ______; findings closed

## Models
- [ ] No model was activated without passing the golden-set gate; `history.jsonl` attached
- [ ] Rollback tried on the staging server

Approved by (system owner): ____________   Cybersecurity: ____________
