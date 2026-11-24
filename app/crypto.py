"""Encryption of personal data the service writes to disk or to the database, when a key is configured.

Set DATA_KEY_FILE to a file (owner-only, outside the data folders) holding one key per line; make a key with
    python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
The first key encrypts; every listed key can decrypt, so a key is rotated by putting the new one on top and keeping the old
one below until the data written with it has expired (jobs: JOB_TTL_MINUTES; training store: TRAINING_RETENTION_DAYS).

What is encrypted: images and corrections in the training store, results and errors in the job database, and the images,
applicant form data, results and errors in the PostgreSQL queue. Authenticated encryption (Fernet: AES-128-CBC with
HMAC-SHA256), so a changed file is refused rather than read. Without a key everything is written as before, and data
written before the key was set is still read.

DATA_CIPHER=aes256gcm switches new writes to AES-256-GCM with the same key file (each key is 32 random bytes; Fernet splits
them into two 128-bit keys, AES-256-GCM uses all 256 bits as one). Choose it when the data's classification calls for 256-bit
encryption under the National Cryptographic Standards. Data written with either cipher is read whatever the setting is.

This protects a copied disk, database dump or backup. It does not protect against someone who can read the key file: keep it
on a different volume or in the platform's secret store.
"""
from __future__ import annotations

import os
from pathlib import Path

PREFIX = b"VOENC1:"          # Fernet
PREFIX_GCM = b"VOENC2:"      # AES-256-GCM: base64(12-byte nonce + ciphertext + tag)
PREFIX_BOUND = b"VOENC4:"    # Fernet, with the context it belongs to sealed inside: 2-byte length + context + data
PREFIX_GCM_BOUND = b"VOENC3:"  # AES-256-GCM with the context as associated data
_ALL = (PREFIX, PREFIX_GCM, PREFIX_BOUND, PREFIX_GCM_BOUND)
_cache: tuple[str, float, object, list] | None = None


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
    import base64  # noqa: WPS433
    try:
        from cryptography.fernet import Fernet, MultiFernet  # noqa: WPS433
    except ImportError as exc:
        raise DataKeyError("DATA_KEY_FILE is set but the 'cryptography' package is not installed") from exc
    keys = [ln.strip() for ln in Path(path).read_text(encoding="utf-8").splitlines() if ln.strip() and not ln.startswith("#")]
    if not keys:
        raise DataKeyError("DATA_KEY_FILE holds no key")
    try:
        cipher = MultiFernet([Fernet(k.encode()) for k in keys])
        raw = [base64.urlsafe_b64decode(k.encode()) for k in keys]
    except ValueError as exc:
        raise DataKeyError("DATA_KEY_FILE holds a value that is not a key") from exc
    _cache = (path, mtime, cipher, raw)
    return cipher


def _use_gcm() -> bool:
    return os.environ.get("DATA_CIPHER", "").strip().lower() in ("aes256gcm", "aes-256-gcm")


def enabled() -> bool:
    return _fernet() is not None


def _flag(name: str) -> bool:
    return os.environ.get(name, "").strip() == "1"


def plaintext_read_allowed() -> bool:
    """With a key configured, data that is not encrypted is refused: anyone who can write to the database or the training
    store could otherwise put any result there. DATA_ALLOW_PLAINTEXT_READ=1 lets data written before the key was set be read
    during a migration; the production start-up gate refuses it."""
    return _fernet() is None or _flag("DATA_ALLOW_PLAINTEXT_READ")


def seal(data: bytes, context: bytes | None = None) -> bytes:
    """Encrypted form of `data` when a key is configured, otherwise `data` unchanged. `context` names where the value belongs
    (job id, column, owner): it is sealed with the value, so the value opens only in that place. A value copied to another row,
    or a row whose owner was changed, is refused."""
    cipher = _fernet()
    if cipher is None:
        return data
    if _use_gcm():
        import base64  # noqa: WPS433
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM  # noqa: WPS433
        nonce = os.urandom(12)
        prefix = PREFIX_GCM if context is None else PREFIX_GCM_BOUND
        return prefix + base64.urlsafe_b64encode(nonce + AESGCM(_cache[3][0]).encrypt(nonce, data, context))
    if context is None:
        return PREFIX + cipher.encrypt(data)
    return PREFIX_BOUND + cipher.encrypt(len(context).to_bytes(2, "big") + context + data)


def unseal(data: bytes, context: bytes | None = None) -> bytes:
    """The plain bytes. Data that was never encrypted is returned as it is only when no key is configured (or during a
    migration, see plaintext_read_allowed). With `context`, the value must have been sealed for that same context."""
    import hmac  # noqa: WPS433
    data = bytes(data)
    if not data.startswith(_ALL):
        if not plaintext_read_allowed():
            raise DataKeyError("stored data is not encrypted although a data key is configured")
        return data
    cipher = _fernet()
    if cipher is None:
        raise DataKeyError("this data is encrypted and no DATA_KEY_FILE is configured")
    bound = data.startswith((PREFIX_BOUND, PREFIX_GCM_BOUND))
    if context is not None and not bound and not _flag("DATA_ALLOW_UNBOUND_READ"):
        # written before values were bound to their row; readable only while the operator says so (one JOB_TTL_MINUTES)
        raise DataKeyError("stored data is not bound to its job; set DATA_ALLOW_UNBOUND_READ=1 while older jobs expire")
    if bound and context is None:
        raise DataKeyError("this data is bound to a context that was not given")
    if data.startswith((PREFIX_GCM, PREFIX_GCM_BOUND)):
        import base64  # noqa: WPS433
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM  # noqa: WPS433
        try:
            blob = base64.urlsafe_b64decode(data[len(PREFIX_GCM):])
        except ValueError as exc:
            raise DataKeyError("encrypted data is damaged") from exc
        aad = context if bound else None
        for key in _cache[3]:                      # newest key first; an older one still opens what it wrote
            try:
                return AESGCM(key).decrypt(blob[:12], blob[12:], aad)
            except Exception:  # noqa: BLE001 - wrong key, changed data or another context
                continue
        raise DataKeyError("encrypted data could not be opened with the configured keys")
    try:
        plain = cipher.decrypt(data[len(PREFIX):])
    except Exception as exc:  # noqa: BLE001 - wrong key or changed data; the library's message adds nothing
        raise DataKeyError("encrypted data could not be opened with the configured keys") from exc
    if not bound:
        return plain
    n = int.from_bytes(plain[:2], "big")
    if not hmac.compare_digest(plain[2:2 + n], context or b""):
        raise DataKeyError("encrypted data belongs to another job or owner")
    return plain[2 + n:]


def seal_text(text: str, context: bytes | None = None) -> str:
    return seal(text.encode("utf-8"), context).decode("ascii") if enabled() else text


def unseal_text(text: str, context: bytes | None = None) -> str:
    return unseal(text.encode("utf-8"), context).decode("utf-8")


def seal_json(value, context: bytes | None = None):
    """For a JSON column: the value itself, or {"enc": "<token>"} when a key is configured."""
    import json
    if value is None or not enabled():
        return value
    return {"enc": seal(json.dumps(value, ensure_ascii=False).encode("utf-8"), context).decode("ascii")}


def unseal_json(value, context: bytes | None = None):
    import json
    if isinstance(value, dict) and set(value) == {"enc"} and isinstance(value["enc"], str):
        return json.loads(unseal(value["enc"].encode("ascii"), context).decode("utf-8"))
    if value is not None and not plaintext_read_allowed():
        raise DataKeyError("stored data is not encrypted although a data key is configured")
    return value


def job_context(job_id: str, column: str, owner: str) -> bytes:
    """Where a job value belongs: it opens only in this job's row, in this column, for this owner."""
    return f"{job_id}|{column}|{owner}".encode("utf-8")
