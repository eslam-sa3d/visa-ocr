"""Runtime settings, read from environment variables at call time (so tests and operators can change them)."""
from __future__ import annotations

import os


def _int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


def _flag(name: str, default: bool = False) -> bool:
    return os.environ.get(name, str(default)).strip().lower() in ("1", "true", "yes", "on")


def max_files() -> int: return _int("MAX_FILES", 4)
def max_bytes() -> int: return _int("MAX_UPLOAD_BYTES", 10 * 1024 * 1024)
def max_pixels() -> int: return _int("MAX_IMAGE_PIXELS", 40_000_000)      # decoded size cap per image
def min_side() -> int: return _int("MIN_IMAGE_SIDE", 64)
def max_aspect() -> int: return max(1, _int("MAX_IMAGE_ASPECT", 20))        # longest side / shortest side
def bad_key_limit_per_minute() -> int: return _int("BAD_KEY_LIMIT_PER_MIN", 10)     # wrong-key attempts per client address
def housekeeping_seconds() -> int: return max(30, _int("HOUSEKEEPING_SECONDS", 300))   # expired-data sweep interval
def training_retention_days() -> int: return _int("TRAINING_RETENTION_DAYS", 30)       # 0 keeps training data until you purge it
def ocr_timeout() -> int: return max(1, _int("OCR_TIMEOUT_SECONDS", 60))     # per Tesseract call
def max_body_bytes() -> int: return max_files() * max_bytes() + 1024 * 1024 # whole multipart request
def job_ttl_seconds() -> int: return _int("JOB_TTL_MINUTES", 60) * 60       # personal data is deleted after this
def job_db_path() -> str: return os.environ.get("JOB_DB_PATH", "")          # empty: keep jobs in memory
def max_pending_jobs() -> int: return _int("MAX_PENDING_JOBS", 32)
def rate_limit_per_minute() -> int: return _int("RATE_LIMIT_PER_MIN", 60)   # 0 disables
def trust_proxy() -> bool: return _flag("TRUST_PROXY")
def ocr_concurrency() -> int: return max(1, _int("OCR_CONCURRENCY", 1))     # PaddleOCR predictors are not thread-safe
def include_text_lines() -> bool: return _flag("INCLUDE_TEXT_LINES")        # raw OCR text is personal data: off by default

# review policy
def min_confidence_unprotected() -> float: return _float("MIN_CONFIDENCE_UNPROTECTED", 0.97)  # names, nationality, sex
def min_confidence_visual() -> float: return _float("MIN_CONFIDENCE_VISUAL", 0.85)
def visual_zone_auto_accept() -> bool: return _flag("VISUAL_ZONE_AUTO_ACCEPT")                # default: always human review
def database_url() -> str: return os.environ.get("DATABASE_URL", "")        # postgresql://... enables the PostgreSQL queue
def worker_threads() -> int: return max(0, _int("WORKER_THREADS", 1))       # in-process workers when using PostgreSQL (0 = API only)
def job_stale_seconds() -> int: return _int("JOB_STALE_SECONDS", 300)       # a claimed job older than this is retried
def job_max_attempts() -> int: return _int("JOB_MAX_ATTEMPTS", 3)
