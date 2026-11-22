"""Runtime settings, read from environment variables at call time (so tests and operators can change them)."""
from __future__ import annotations

import contextlib
import contextvars
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
def ocr_concurrency() -> int: return 1    # kept for callers; PaddleOCR reads one document at a time (predictors are not thread-safe)
def include_text_lines() -> bool: return _flag("INCLUDE_TEXT_LINES")        # raw OCR text is personal data: off by default

# review policy
def min_confidence_unprotected() -> float: return _float("MIN_CONFIDENCE_UNPROTECTED", 0.97)  # names, nationality, sex
def min_confidence_visual() -> float: return _float("MIN_CONFIDENCE_VISUAL", 0.85)
def visual_zone_auto_accept() -> bool: return _flag("VISUAL_ZONE_AUTO_ACCEPT")                # default: always human review
def database_url() -> str: return os.environ.get("DATABASE_URL", "")        # postgresql://... enables the PostgreSQL queue
def worker_threads() -> int: return max(0, _int("WORKER_THREADS", 1))       # in-process workers when using PostgreSQL (0 = API only)
def job_stale_seconds() -> int: return max(30, _int("JOB_STALE_SECONDS", 300))   # a claim not renewed for this long is retried
def job_max_attempts() -> int: return max(1, _int("JOB_MAX_ATTEMPTS", 3))
def db_statement_timeout_ms() -> int: return max(1000, _int("DB_STATEMENT_TIMEOUT_MS", 15000))
def worker_poll_seconds() -> float: return min(5.0, max(0.05, _float("WORKER_POLL_SECONDS", 0.2)))   # idle wait between queue checks
def optional_data_review() -> bool: return _flag("OPTIONAL_DATA_REVIEW")   # force review when only optional data is unconfirmed
def job_threads() -> int:                                                   # in-process OCR jobs at a time (no PostgreSQL queue)
    return max(1, _int("JOB_THREADS", max(1, (os.cpu_count() or 2) // 2)))
def text_read_seconds() -> float: return max(1.0, _float("OCR_TEXT_SECONDS", 20.0))   # ID-card text reader budget with no deadline
def max_text_lines() -> int: return max(20, _int("MAX_TEXT_LINES", 250))    # OCR lines parsed per card job
def job_max_seconds() -> int: return max(30, _int("JOB_MAX_SECONDS", 600))  # a job still running after this is marked failed
def rapidocr_sessions() -> int:                                             # RapidOCR readers that can work at the same time
    return max(1, _int("RAPIDOCR_SESSIONS", min(2, job_threads())))
def search_seconds() -> float: return max(0.3, _float("OCR_SEARCH_SECONDS", 8.0))             # Tesseract MRZ search, per image
def second_reader_seconds() -> float: return max(0.3, _float("OCR_SECOND_READER_SECONDS", 1.5))   # RapidOCR, once a reading is in hand
def agreement_confidence() -> float: return _float("AGREEMENT_CONFIDENCE", 0.99)             # two readers agree, all checks pass
def api_docs() -> bool: return _flag("API_DOCS")
def training_store_dir() -> str: return os.environ.get("TRAINING_STORE_DIR", "")
def storage_encrypted() -> bool: return _flag("DATA_AT_REST_ENCRYPTED")     # operator confirms the data volumes are encrypted
def production() -> bool: return _flag("PRODUCTION")                         # refuse to start in any unsafe set-up (see main.py)
def ui_enabled() -> bool: return _flag("UI_ENABLED", not production())      # the built-in test page; off in production
def upload_scan_command() -> str: return os.environ.get("UPLOAD_SCAN_COMMAND", "").strip()   # e.g. "clamdscan --no-summary"
def upload_scan_seconds() -> float: return max(1.0, _float("UPLOAD_SCAN_SECONDS", 20.0))
def paddle_shared_detection() -> bool: return _flag("PADDLE_SHARED_DETECTION", True)   # find text once, read it per language; 0 = one pipeline per language
def ocr_engine() -> str: return os.environ.get("OCR_ENGINE", "auto")
def ocr_langs() -> list[str]: return [x.strip() for x in os.environ.get("OCR_LANGS", "en,ar").split(",") if x.strip()]
def tesseract_text_langs() -> str: return os.environ.get("TESSERACT_TEXT_LANGS", "ara+eng")
def id_labels_file() -> str: return os.environ.get("ID_LABELS_FILE", "")
def tesseract_lang() -> str: return os.environ.get("TESSERACT_LANG", "").strip()
def tesseract_data_dir() -> str: return os.environ.get("TESSERACT_DATA_DIR", "").strip()


_read_limit: contextvars.ContextVar[float] = contextvars.ContextVar("read_limit", default=0.0)


def ocr_deadline() -> float:
    """Hard limit in seconds for one read (0 = off). When set, each reader stops starting new attempts in time to answer by the
    deadline and returns the best reading it has; a reading that is not perfect then goes to review as usual.
    Inside limited_read() the tighter of the two limits applies."""
    try:
        base = max(0.0, float(os.environ.get("OCR_DEADLINE_SECONDS", "0")))
    except ValueError:
        base = 0.0
    inner = _read_limit.get()
    return min(base, inner) if base and inner else base or inner


@contextlib.contextmanager
def limited_read(seconds: float):
    """Give the readers called inside this block a hard limit (0 = none), for this thread of work only."""
    token = _read_limit.set(max(0.0, seconds))
    try:
        yield
    finally:
        _read_limit.reset(token)


_strip_only: contextvars.ContextVar[bool] = contextvars.ContextVar("strip_only", default=False)


@contextlib.contextmanager
def quick_look(seconds: float):
    """A limited read that only looks where an MRZ strip shows by its shape, and reads no printed text. A clear passport is
    answered from that strip; a page with no such strip (a card front) is given up in a fraction of a second."""
    token = _strip_only.set(True)
    try:
        with limited_read(seconds):
            yield
    finally:
        _strip_only.reset(token)


def strip_only() -> bool:
    return _strip_only.get()


def fast_first_seconds() -> float:
    """In "auto", the Tesseract + RapidOCR pair gets this long to read the MRZ BEFORE PaddleOCR is tried (0 = off: PaddleOCR
    always reads first). The pair's answer is taken only when both readers give the same lines and every check digit
    passes, which is the strongest evidence there is; anything less goes on to PaddleOCR as before. A clear passport is
    answered in a couple of seconds this way instead of waiting for a whole-page read in two languages."""
    return max(0.0, _float("OCR_FAST_FIRST_SECONDS", 3.0))


def handover_seconds() -> float:
    """Limit for the Tesseract + RapidOCR pair when "auto" calls it after PaddleOCR has already read the page (0 = none).
    PaddleOCR has the page text and has turned a sideways photo upright by then, so the pair only has to look for an MRZ:
    without a limit it also searches the page turned each way and reads all the printed text again."""
    return max(0.0, _float("OCR_HANDOVER_SECONDS", 12.0))


def work_max_side() -> int:
    """Longest side, in pixels, that images are reduced to before reading (0 = never reduce). Lower is faster and uses less
    memory; too low loses small print. 2000 keeps a full passport page comfortably readable."""
    return max(0, _int("WORK_MAX_SIDE", 2000))


# Image clarity (app/quality.py). "reject" values stop an image before it is read; "doubt" values give the reason when a
# reading then fails. Defaults were set from 432 real photos: no image that was read correctly falls under a reject value.
_QUALITY_DEFAULTS = {"min_side_reject": 400, "min_side_doubt": 900, "sharpness_reject": 15.0, "sharpness_doubt": 60.0,
                     "dark_reject": 40.0, "dark_doubt": 70.0, "bright_reject": 245.0,
                     "contrast_reject": 12.0, "contrast_doubt": 25.0}


def quality_check() -> bool: return os.environ.get("IMAGE_QUALITY_CHECK", "1") != "0"


def quality_thresholds() -> dict:
    """QUALITY_<NAME>=value overrides one threshold, e.g. QUALITY_SHARPNESS_REJECT=20."""
    return {k: _float("QUALITY_" + k.upper(), v) for k, v in _QUALITY_DEFAULTS.items()}
