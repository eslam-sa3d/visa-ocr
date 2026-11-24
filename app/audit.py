"""Security event log: one JSON object per line, for collection by a SIEM (NCA ECC 2-12).

Events say WHO did WHAT and WHEN, never what a document contained: a job id, the short hash that stands for an API key, a
client address, a result code. No name, number, date or image from a document is ever written here.

Events go to the "visa-ocr.audit" logger (so they reach wherever the service's logs go) and, when AUDIT_LOG_FILE is set, are
also appended to that file (created owner-only). With no file and no handler on that logger they are written to stderr, and
when the file cannot be written the event goes to the log and stderr and /readyz reports the problem: an event is never lost
silently. Every event names the operating-system account that wrote it (os_user). Keep them for the period your policy
sets; the service does not delete them. Rotate the file on the host (logrotate with copytruncate, or the SIEM agent).
"""
from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
from datetime import datetime, timezone

log = logging.getLogger("visa-ocr.audit")
_lock = threading.Lock()
_ALLOWED = (str, int, float, bool, type(None))
write_failed = False            # set when AUDIT_LOG_FILE cannot be written; /readyz reports it
_seen: dict[str, float] = {}


def _os_user() -> str:
    try:
        import getpass  # noqa: WPS433
        return getpass.getuser()
    except Exception:  # noqa: BLE001 - no user name in this environment
        return str(os.getuid()) if hasattr(os, "getuid") else ""


def first_in_window(what: str, seconds: float = 60.0) -> bool:
    """True once per `what` per window: a repeated event (a caller throttled again and again) is written once per window,
    not once per request, so a caller cannot fill the log at will."""
    now = time.monotonic()
    with _lock:
        if len(_seen) > 10_000:
            for k in [k for k, t in _seen.items() if t <= now - seconds]:
                del _seen[k]
        if _seen.get(what, -seconds - 1) > now - seconds:
            return False
        _seen[what] = now
        return True


def event(name: str, **fields) -> dict:
    """Record one security event. Values must be short plain values (ids, codes, counts); anything else is dropped."""
    global write_failed
    record = {"ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"), "event": name, "os_user": _os_user()}
    for key, value in fields.items():
        if isinstance(value, _ALLOWED):
            record[key] = value[:200] if isinstance(value, str) else value
        elif isinstance(value, (list, tuple)) and all(isinstance(v, str) for v in value):
            record[key] = [v[:100] for v in value[:20]]
    line = json.dumps(record, ensure_ascii=False, sort_keys=True)
    log.info(line)
    handled = log.hasHandlers() and log.isEnabledFor(logging.INFO)
    path = os.environ.get("AUDIT_LOG_FILE", "").strip()
    if path:
        try:
            with _lock:
                fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
                with os.fdopen(fd, "a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
            return record
        except OSError as exc:
            write_failed = True
            # the event itself is never lost: it goes to the service log and to stderr with the reason
            log.warning("audit log file cannot be written (%s); event follows: %s", type(exc).__name__, line)
            sys.stderr.write("visa-ocr audit (file not writable): " + line + "\n")
            return record
    if not handled:
        sys.stderr.write(line + "\n")          # no file and no log handler: stderr reaches the journal or the operator
    return record
