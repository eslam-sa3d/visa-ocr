# Secure coding rules for this project (NCA ECC 1-6-3-1)

Short, checkable, and each one already has a test. A change that breaks a rule does not merge.

1. **Input is hostile until checked.** Uploads are identified by their bytes, not their label; size, pixel count, aspect and
   format are checked before any decoder runs (`app/uploads.py`). Request fields have length limits. Ids match a fixed pattern
   before they reach a store.
2. **No shell.** External programs (Tesseract, a malware scanner) are started with an argument list, never a command string
   built from input, and always with a timeout.
3. **No SQL built from input.** Values are always parameters; column names come from a fixed list.
4. **No file path built from input** without confining it: job ids are alphanumeric, model paths must stay inside the models
   folder.
5. **Nothing is loaded that was not verified.** Models need a matching checksum; the manifest can be signed; packages are
   installed by hash.
6. **Fail closed.** A check that cannot be made (scanner down, key file unreadable, manifest unreadable, model unscored)
   refuses; it never lets through.
7. **Secrets are never in code, logs, URLs or error messages.** API keys are compared as hashes in constant time.
8. **No personal data in logs or error messages.** Errors are logged as type and code location. The security log holds ids
   and codes only.
9. **Least privilege.** The service runs as an unprivileged user; files it writes are owner-only; each API key has one role.
10. **Every personal data store has an expiry and a delete.** Nothing is kept by default.
11. **Dependencies are pinned by hash and checked for known vulnerabilities before each release.**
12. **Every fix comes with a test that fails without it.**
