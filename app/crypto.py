"""Encryption of personal data the service writes to disk or to the database, when a key is configured.

Set DATA_KEY_FILE to a file (owner-only, outside the data folders) holding one key per line; make a key with
    python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
The first key encrypts; every listed key can decrypt, so a key is rotated by putting the new one on top and keeping the old
one below until the data written with it has expired (jobs: JOB_TTL_MINUTES; training store: TRAINING_RETENTION_DAYS).

What is encrypted: images and corrections in the training store, results and errors in the job database, and the images,
applicant form data, results and errors in the PostgreSQL queue. Authenticated encryption (Fernet: AES-128-CBC with
HMAC-SHA256), so a changed file is refused rather than read. Without a key everything is written as before, and data
written before the key was set is still read.

This protects a copied disk, database dump or backup. It does not protect against someone who can read the key file: keep it
on a different volume or in the platform's secret store.
"""
from __future__ import annotations

import os
from pathlib import Path

PREFIX = b"VOENC1:"
_cache: tuple[str, float, object] | None = None


class DataKeyError(RuntimeError):
    """Encrypted data cannot be read: no key is configured, or none of the keys fits."""


def _fernet():
    """The cipher for the configured keys, or None when no key is set. Re-read when the key file changes."""
    global _cache
    path = os.environ.get("DATA_KEY_FILE", "").strip()
    if not path:
        return None
    try:
        mtime = os.stat(path).st_mtime
    except OSError as exc:
        raise DataKeyError(f"DATA_KEY_FILE cannot be read: {type(exc).__name__}") from exc
    if _cache and _cache[0] == path and _cache[1] == mtime:
        return _cache[2]
    try:
        from cryptography.fernet import Fernet, MultiFernet  # noqa: WPS433
    except ImportError as exc:
        raise DataKeyError("DATA_KEY_FILE is set but the 'cryptography' package is not installed") from exc
    keys = [ln.strip() for ln in Path(path).read_text(encoding="utf-8").splitlines() if ln.strip() and not ln.startswith("#")]
    if not keys:
        raise DataKeyError("DATA_KEY_FILE holds no key")
    try:
        cipher = MultiFernet([Fernet(k.encode()) for k in keys])
    except ValueError as exc:
        raise DataKeyError("DATA_KEY_FILE holds a value that is not a key") from exc
    _cache = (path, mtime, cipher)
    return cipher


def enabled() -> bool:
    return _fernet() is not None


def seal(data: bytes) -> bytes:
    """Encrypted form of `data` when a key is configured, otherwise `data` unchanged."""
    cipher = _fernet()
    return data if cipher is None else PREFIX + cipher.encrypt(data)


def unseal(data: bytes) -> bytes:
    """The plain bytes. Data that was never encrypted is returned as it is."""
    data = bytes(data)
    if not data.startswith(PREFIX):
        return data
    cipher = _fernet()
    if cipher is None:
        raise DataKeyError("this data is encrypted and no DATA_KEY_FILE is configured")
    try:
        return cipher.decrypt(data[len(PREFIX):])
    except Exception as exc:  # noqa: BLE001 - wrong key or changed data; the library's message adds nothing
        raise DataKeyError("encrypted data could not be opened with the configured keys") from exc


def seal_text(text: str) -> str:
    return seal(text.encode("utf-8")).decode("ascii") if enabled() else text


def unseal_text(text: str) -> str:
    return unseal(text.encode("utf-8")).decode("utf-8") if text.startswith(PREFIX.decode()) else text


def seal_json(value):
    """For a JSON column: the value itself, or {"enc": "<token>"} when a key is configured."""
    import json
    if value is None or not enabled():
        return value
    return {"enc": seal(json.dumps(value, ensure_ascii=False).encode("utf-8")).decode("ascii")}


def unseal_json(value):
    import json
    if isinstance(value, dict) and set(value) == {"enc"} and isinstance(value["enc"], str):
        return json.loads(unseal(value["enc"].encode("ascii")).decode("utf-8"))
    return value
