# NCA controls: what the service does, and what must be supplied around it

An engineering map of the service against the NCA Essential Cybersecurity Controls (ECC-2:2024). It is not a statement of
compliance; only the organisation's cybersecurity function or an approved assessor can make one. If the portal is classed as
a critical system (CSCC-1:2019) or the data's classification brings in the Data Cybersecurity Controls (DCC-1:2022), check
those documents as well: they were not available when this was written.

## Built into the service

| ECC control | What the service does | How to turn it on or check it |
| --- | --- | --- |
| 1-6-3-1 Secure coding | Rules in `docs/SECURE_CODING.md`; each has a test | `pytest` |
| 1-6-3-2 Trusted, licensed components | Every package pinned by hash; bill of materials with licences in `docs/sbom.json` | `python tools/dependency_audit.py` |
| 1-6-2, 1-6-3-5 Review before release | `docs/RELEASE_CHECKLIST.md`, signed per release | by hand |
| 2-2-3-3 Least privilege, segregation of duties | API keys carry a role (uploader, reviewer, admin) and an expiry date | `API_KEYS_FILE` (see `app/security.py`) |
| 2-3-3-1 Anti-malware | Every upload is passed to a scanner before it is decoded; no answer from the scanner refuses the upload | `UPLOAD_SCAN_COMMAND="clamdscan --no-summary"` |
| 2-7 Data protection | Results expire (60 min), training data expires (30 days), deletion on request, raw text off by default, no document values in logs | defaults |
| 2-8-3-1 Approved cryptography in transit | TLS 1.2 and 1.3 only, forward-secret authenticated ciphers; no self-signed certificate in production | start script; `TLS_CIPHERS` to match your standard |
| 2-8-3-3 Encryption at rest | Training store, job database and queue encrypted; AES-256-GCM available | `DATA_KEY_FILE`, `DATA_CIPHER=aes256gcm` |
| 2-8-3-2 Key lifecycle | Key rotation for data keys and API keys (several valid at once, expiry dates) | key files |
| 2-10-3 Vulnerability management | Check of every locked package against the advisory database | `python tools/dependency_audit.py --check` on the build machine, weekly and per release |
| 2-12-3 Event logs | One JSON line per security event, no personal data: logins refused or throttled, expired keys, forbidden actions, jobs, corrections and their approval, model activation and rollback, data removal, malware, start-up | `AUDIT_LOG_FILE`, or collect the `visa-ocr.audit` logger |
| 2-15-3 Web application security | HTTPS only; security headers; test page and API pages off; `PRODUCTION=1` refuses to start with authentication off, the test page on, unchecked models or unencrypted storage | `PRODUCTION=1` |
| 4-2 Cloud | None used: the service runs with no network at all | `tools/offline_selftest.sh`, `tools/offline_smoke.sh` |

## Must be supplied by the organisation

| ECC control | Needed |
| --- | --- |
| 1-6-2, 2-11 | Independent source code review and penetration test before go-live, then periodically |
| 2-2-3-2, 2-2-3-4 | Named accounts, multi-factor login and privileged access management for people who log in to the servers (to approve corrections, activate models, read logs) |
| 2-3-3-3, 2-3-3-4 | Operating system patching; clock synchronisation |
| 2-5-3, 2-15-3-1, 2-15-3-2 | Web application firewall, network segmentation, production separated from test, intrusion prevention, denial-of-service protection. The service should not face the internet: only the portal calls it |
| 2-7-1, 2-7-2 | The classification of passport and ID images, recorded, and the handling rules that follow from it |
| 2-8-3-1 | Confirmation of the algorithms and key lengths the National Cryptographic Standards require for that classification |
| 2-8-3-2 | A key management system or hardware module for the data key, manifest key, TLS key and API keys |
| 2-9-3 | Backups of `models/` and the manifest; tested recovery |
| 2-12-3-3 to 2-12-3-5 | Collection of the event log in the SIEM, monitoring, 12 months retention (longer for critical systems) |
| 4-1 | Policy on where source code may be hosted and where test documents may be processed |

## Recommended production settings

```
PRODUCTION=1
API_KEYS_FILE=/etc/visa-ocr/keys.json          # roles and expiry dates; no API_KEYS
DATA_KEY_FILE=/etc/visa-ocr/data.key           # on a different volume from the data
DATA_CIPHER=aes256gcm
MANIFEST_HMAC_KEY_FILE=/etc/visa-ocr/manifest.key
AUDIT_LOG_FILE=/var/log/visa-ocr/audit.log     # shipped to the SIEM
UPLOAD_SCAN_COMMAND=clamdscan --no-summary
TLS_CERT=/etc/visa-ocr/tls/server.crt
TLS_KEY=/etc/visa-ocr/tls/server.key
```
