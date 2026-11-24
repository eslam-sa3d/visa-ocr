# Third security audit (24 Nov 2026, release 248aabe): what was fixed

Every confirmed finding is fixed in the code, with a regression test in `tests/test_security_audit_3.py` that fails on
248aabe and passes now. The leads that need a fact from the servers have their code side hardened; the check on the
servers is still to be done by their owners (listed at the end).

## Review decision (a wrong value must never be accepted silently)

| Finding | Fix |
| --- | --- |
| Expiry year 99 read as 2099 | `expiry_century_check` (error) when the expiry is more than 20 years ahead (`app/validation.py`) |
| Impossible type code or digits in a name | `type_code_invalid` and `mrz_name_invalid_characters` (errors): the first type letter must be one the format allows, the second a letter or digit (`app/validation.py`) |
| Characters removed from a number that passes its check digit | A tidy of a check-digit field always counts as a repair and goes to review (`app/mrz.py`) |
| Short given name erased by the name clean-up | Any letters removed from the name line are an error, and never count as reader agreement (`app/validation.py`, `app/engines/ensemble.py`) |
| Document type "fixed by the format" for any letter | Only for a letter the format allows (`app/review.py`) |
| Form name prefix matched with fillers left | A prefix counts only when the MRZ name field is full (`app/review.py`) |
| Surname alone confirmed the given names | The form confirms the names only when it covers them, or the document has none (`app/review.py`) |
| Printed-page prefix after a cleaned name line | A cleaned name line is not a full field (`app/review.py`) |
| Two readers agreeing on letters, not on the name split | Agreement only on equal parsed values; the reading with a real `<<` separator is kept (`app/engines/ensemble.py`) |
| Card number taken from another label under auto-accept | The number must sit in or beside its own label with no other label between; otherwise `id_number_unlabelled` (error) (`app/visual_zone.py`) |

## Stored data and models

| Finding | Fix |
| --- | --- |
| Sealed result not tied to its job or owner | Each value is sealed with its job id, column and owner (AES-GCM associated data, or sealed inside the Fernet token). Values written by 1.0.0 are read only with `DATA_ALLOW_UNBOUND_READ=1` (`app/crypto.py`, `app/jobs.py`, `app/pgstore.py`) |
| Plain stored data accepted with a key | Refused; `DATA_ALLOW_PLAINTEXT_READ=1` for a migration, refused in production. The time-out errors the service writes itself are now sealed too |
| Tesseract model checked against an unsigned checksum | With `MANIFEST_HMAC_KEY` set, `SHA256SUMS.sig` must hold the checksum file's HMAC (`tools/make_checksums.py` writes it) |
| Older signed manifest accepted again | The signed data carries a serial that every save raises; a server refuses a lower serial (memory and `MANIFEST_STATE_FILE`) |

## Capacity

| Finding | Fix |
| --- | --- |
| Check routes decode full-size images with no limit (medium) | Both check routes share the two-decode limit of the upload route, and decode at working size (JPEG draft, reduce before laying transparency on white) (`app/main.py`, `app/quality.py`, `app/photo.py`, `app/uploads.py`) |
| Pending cap stopped limiting after `JOB_MAX_SECONDS` | Admission also counts the work the process still holds; a job failed as timed out is never read or turned to done (`app/main.py`, `app/worker.py`) |

## Audit trail

| Finding | Fix |
| --- | --- |
| Unbounded lines from callers without a key; silent loss on write failure | Throttled callers are written once a minute per client; an unwritable file makes `/readyz` fail and the event goes to the log and stderr (`app/audit.py`, `app/security.py`) |
| Reads of another key's job not recorded | `job_read` for every cross-owner read, `job_read_refused` for every refused one (`app/main.py`) |
| Tool and model events missing or unattributed | Every event names the OS account; without a file events go to stderr; approvals name approver, file and digest; purges, metrics changes and model switches in a running server are recorded; the retraining unit has its own `AUDIT_LOG_FILE` |

## Training data and approval

| Finding | Fix |
| --- | --- |
| Approval given to a job id | `--approve` must name the correction's digest from `--show`; a later correction stays pending |
| Approver shown only box numbers | `--show` writes each line's crop and its digest; only crops whose pixels were approved become training data |
| Stored text could change what the approver sees | Control, formatting and direction characters are refused at the API and shown as escapes by the tool |
| Calibration counted unapproved corrections | Only approved corrections count; checked fields are only fields the result holds |
| No bound on corrections | 10 per job, 30 a minute per client (always on), and no training writes below 2 GB free (`MAX_CORRECTIONS_PER_JOB`, `CORRECTIONS_PER_MIN`, `TRAINING_MIN_FREE_BYTES`); box and image values are bounded |

## Leads: code hardened, server check still open

| Lead | Code side | Still to check on the servers |
| --- | --- | --- |
| Production flag read three ways | One rule in the service and the entrypoint; a value that is neither on nor off stops the service; no inline comments left in `visa-ocr.env` | `docker inspect` shows `PRODUCTION=1` on each host |
| Bodies parsed before the key is checked | A request with a body and no valid key is refused before any of the body is read | Firewall: only the portal reaches port 443 |
| Release checksum travels with the release | The build records the checksum file's SHA-256 and the image id outside the folder; `DEPLOYMENT.txt` step 1 compares them with values sent by another route | Who can write the transfer medium |
| API keys on the command line of the test tools | The tools read `VISA_OCR_API_KEY`; `DEPLOYMENT.txt` uses header files and `read -s` | Accounts and shell history on the jump host |
| Offline self-test as root, log at a fixed `/tmp` name | Root only makes the namespace; the service and test run as `SELFTEST_USER` (nobody); private log via `mktemp`; root-run checks use `python -I` | Who can write the project folder |
| Model ids used as folder names | Ids limited to letters, digits, `.`, `_`, `-` | Who can write `MODELS_DIR/manifest.json` on the training host |
| MRZ text without `<` counted as the page's copy | A line that looks like the MRZ, or is part of it, is never page evidence | Whether the recognisers ever drop the fillers |
| Test-tool output with default permissions | Output folder 0700, files 0600; launcher text corrected | Existing results folders on machines that ran the tool |
| Typed values kept in the browser | Kept at most 14 days, removed when the sheet is saved, and a button removes them | The browser profile used for the checking |
| Text from a document as a spreadsheet formula | Cells starting with `= + - @` are written as text | The program used to open `sheet.csv` |
| A reader that dies fails other documents | Not changed | Peak memory of one reader for the largest accepted upload on staging |
