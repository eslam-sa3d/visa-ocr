"""Security event log: one JSON object per line, for collection by a SIEM (NCA ECC 2-12).

Events say WHO did WHAT and WHEN, never what a document contained: a job id, the short hash that stands for an API key, a
client address, a result code. No name, number, date or image from a document is ever written here.

Events go to the "visa-ocr.audit" logger (so they reach wherever the service's logs go) and, when AUDIT_LOG_FILE is set, are
also appended to that file (created owner-only). Keep them for the period your policy sets; the service does not delete them.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime, timezone

log = logging.getLogger("visa-ocr.audit")
_lock = threading.Lock()
_ALLOWED = (str, int, float, bool, type(None))


def event(name: str, **fields) -> dict:
    """Record one security event. Values must be short plain values (ids, codes, counts); anything else is dropped."""
    record = {"ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"), "event": name}
    for key, value in fields.items():
        if isinstance(value, _ALLOWED):
            record[key] = value[:200] if isinstance(value, str) else value
        elif isinstance(value, (list, tuple)) and all(isinstance(v, str) for v in value):
            record[key] = [v[:100] for v in value[:20]]
    line = json.dumps(record, ensure_ascii=False, sort_keys=True)
    log.info(line)
    path = os.environ.get("AUDIT_LOG_FILE", "").strip()
    if path:
        try:
            with _lock:
                fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
                with os.fdopen(fd, "a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
        except OSError as exc:
            log.error("audit log file cannot be written: %s", type(exc).__name__)
    return record
