"""Offline document OCR API.

Endpoints
  GET  /healthz, /readyz
  POST /v1/mrz/parse           parse and validate MRZ lines you already have
  POST /v1/documents/extract   upload document images, returns a job
  GET  /v1/jobs/{id}           job status, fields, validation flags
  POST /v1/jobs/{id}/corrections   reviewer corrections (future training data, opt-in)
  GET  /v1/models              OCR models on this host and their status

Images are not kept or logged. Exceptions: TRAINING_STORE_DIR is set and the applicant consented
(consent_training=true), and short-lived crops that Tesseract writes to the temp directory while it runs.
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import re
import threading
import time
import unicodedata
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from starlette.concurrency import run_in_threadpool
from pydantic import BaseModel, Field, field_validator

from . import audit, config, crypto, pdf, photo, pipeline, readers, security, training_store, worker, quality
from .engines import get_engine
from .jobs import make_store
from .model_registry import ManifestError, Registry
from .mrz import MrzError
from .review import Applicant, compare_applicant
from .training_store import TrainingStore
from .uploads import UploadError, scan_for_malware, shrink_for_ocr, validate_image

log = logging.getLogger("visa-ocr")

store = make_store()
@contextlib.asynccontextmanager
async def lifespan(_app):
    stop = threading.Event()
    threads = []
    if getattr(store, "queue", False):
        for _ in range(config.worker_threads()):
            t = threading.Thread(target=worker.worker_loop, args=(store, stop, lambda: get_engine()), daemon=True)
            t.start()
            threads.append(t)
    def housekeeping() -> None:
        while not stop.wait(config.housekeeping_seconds()):
            try:
                store.sweep()
                days = config.training_retention_days()
                if training.enabled and days > 0:
                    training.purge_older_than(days)
            except Exception as exc:  # noqa: BLE001 - a failed sweep must not stop the next one
                log.warning("housekeeping sweep failed: %s at %s", type(exc).__name__, pipeline.where(exc))

    hk = threading.Thread(target=housekeeping, daemon=True)
    hk.start()
    readers.start()               # the readers load now, not on the first applicant's upload
    yield
    stop.set()
    readers.stop()
    # Jobs still waiting or running in this process cannot finish after it stops. Say so in the store (it may outlive the
    # process), so a caller polling for the result gets an answer instead of "queued" until the job expires.
    with _inflight_lock:
        left = list(_inflight)
    for jid in left:
        try:
            store.update(jid, status="failed", error={"code": "service_restarted",
                                                      "message": "The service was restarted. Upload the document again."})
        except Exception as exc:  # noqa: BLE001
            log.warning("could not mark job %s as stopped: %s", jid, type(exc).__name__)
    shutdown = getattr(_ocr_pool, "shutdown", None)
    if shutdown:
        shutdown(wait=False, cancel_futures=True)


# The interactive API pages are off unless API_DOCS=1: they are served without a key, describe every endpoint, and load their
# scripts from a public CDN, which an offline server cannot reach anyway.
_docs = config.api_docs()
def _version() -> str:
    try:
        return (Path(__file__).resolve().parent.parent / "VERSION").read_text(encoding="utf-8").strip() or "0"
    except OSError:
        return "0"


app = FastAPI(title="Visa Portal Document OCR", version=_version(), lifespan=lifespan,
              docs_url="/docs" if _docs else None, redoc_url="/redoc" if _docs else None,
              openapi_url="/openapi.json" if _docs else None)
app.add_middleware(security.BodyLimit)
training = TrainingStore()  # disabled unless TRAINING_STORE_DIR is set
# In-process OCR jobs run on their own small pool, never on the framework's thread pool, so waiting or slow jobs cannot
# take the threads that answer /healthz and job lookups.
_ocr_pool = ThreadPoolExecutor(max_workers=config.job_threads(), thread_name_prefix="ocr")
_shrink_slots: asyncio.Semaphore | None = None      # created on first use, inside the running event loop
_inflight: set[str] = set()                         # in-process jobs accepted and not yet finished
_inflight_lock = threading.Lock()

crypto.enabled()      # a key file that is set but unusable stops the service here, not at the first upload
if security.weak_plain_keys():
    log.warning("Some API_KEYS values are shorter than 24 characters. Use long random keys (openssl rand -hex 32).")
if not security.auth_enabled():       # raises when a key variable is set but holds no usable key: refuse to start
    log.warning("API_KEYS is not set: authentication is OFF. Do not expose this service to a network.")


def production_problems() -> list[str]:
    """Settings that must not reach production. With PRODUCTION=1 the service refuses to start while any of them holds."""
    out = []
    if not security.auth_enabled():
        out.append("no API key is configured (API_KEYS, API_KEY_SHA256 or API_KEYS_FILE)")
    if security.weak_plain_keys():
        out.append("an API key is shorter than 24 characters")
    if config.ui_enabled():
        out.append("the test page is on (UI_ENABLED)")
    if config.api_docs():
        out.append("the interactive API pages are on (API_DOCS)")
    if os.environ.get("ALLOW_UNVERIFIED_MODELS", "").strip().lower() in ("1", "true", "yes", "on"):
        out.append("models without a checksum are allowed (ALLOW_UNVERIFIED_MODELS)")
    if (training.enabled or config.job_db_path() or config.database_url()) and not (crypto.enabled() or config.storage_encrypted()):
        out.append("stored personal data is not encrypted (set DATA_KEY_FILE, or DATA_AT_REST_ENCRYPTED=1 on encrypted volumes)")
    if os.environ.get("ALLOW_ANONYMOUS", "").strip() == "1":
        out.append("ALLOW_ANONYMOUS is set")
    if os.environ.get("DATA_ALLOW_PLAINTEXT_READ", "").strip() == "1":
        out.append("stored data that is not encrypted is accepted (DATA_ALLOW_PLAINTEXT_READ)")
    return out


if config.production():
    _problems = production_problems()
    if _problems:
        raise RuntimeError("PRODUCTION=1 but the set-up is not safe for production: " + "; ".join(_problems))
audit.event("service_started", version=app.version, production=config.production(), auth=security.auth_enabled(),
            encrypted=bool(crypto.enabled() or config.storage_encrypted()), engine=config.ocr_engine())


@app.middleware("http")
async def correlation_id(request: Request, call_next):
    cid = security.safe_correlation_id(request.headers.get("X-Correlation-ID"), uuid.uuid4().hex)
    request.state.correlation_id = cid
    response = await call_next(request)
    response.headers["X-Correlation-ID"] = cid
    security.add_security_headers(response, request.url.path)
    return response


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok"}


_model_cache: dict = {"at": 0.0, "problems": []}
_model_lock = threading.Lock()


def _model_problems() -> list[str]:
    """Model check re-hashes every model file, so it is cached for 30 s to keep /readyz cheap for anonymous callers."""
    with _model_lock:
        if time.monotonic() - _model_cache["at"] < 30 and _model_cache["at"]:
            return list(_model_cache["problems"])
        registry = Registry()
        out: list[str] = []
        try:
            for m in registry.models():
                if not m.active:
                    continue
                ok, st = registry.usable(m, fresh=False)     # re-hashed only when a file's size or time changed
                if not ok:
                    out.append(f"model {m.id} is {st}")
        except ManifestError as exc:
            out.append(f"model manifest refused: {exc}")
        _model_cache.update(at=time.monotonic(), problems=out)
        return list(out)


@app.get("/readyz")
def readyz() -> dict:
    """Ready when the job store works and every active OCR model is present and unchanged."""
    problems: list[str] = []
    try:
        getattr(store, "ping", store.count)()      # a cheap check: this route is open to anonymous callers
    except Exception:  # noqa: BLE001
        problems.append("job store unavailable")
    # Which model problems stop the service depends on the engine in use. The Tesseract + RapidOCR pair needs no model files.
    # "auto" falls back to that pair when PaddleOCR models are absent, so a missing model is a warning there; a model whose
    # files changed (checksum mismatch) is always a failure.
    engine = config.ocr_engine()
    warnings: list[str] = []
    if engine in ("auto", "paddle-ocr"):
        for p in _model_problems():
            (warnings if engine == "auto" and p.endswith("is missing") else problems).append(p)
    if audit.write_failed:
        problems.append("the security event log (AUDIT_LOG_FILE) cannot be written")
    if problems:
        raise HTTPException(503, {"code": "not_ready", "message": "; ".join(problems)})
    # The service cannot see whether its volumes are encrypted. The operator confirms it; until then every readiness answer
    # carries the reminder (identity documents and results are written to these places in plain form).
    # With DATA_KEY_FILE the service encrypts what it stores itself (app/crypto.py) and no reminder is needed.
    if (training.enabled or config.job_db_path() or config.database_url()) and not (crypto.enabled() or config.storage_encrypted()):
        warnings.append("stored personal data is not encrypted: set DATA_KEY_FILE so the service encrypts it, or put the job "
                        "database and the training store on encrypted volumes and set DATA_AT_REST_ENCRYPTED=1")
    out = {"status": "ready", "authEnabled": security.auth_enabled(), "readers": readers.count()}
    if warnings:
        out["warnings"] = warnings
    return out


_UI_FILE = Path(__file__).parent / "static" / "index.html"


@app.get("/ui", include_in_schema=False, response_class=HTMLResponse)
def ui() -> HTMLResponse:
    """Plain test page: upload an image and see the result. No external assets (works offline). Off in production."""
    if not config.ui_enabled():
        raise HTTPException(404, {"code": "not_found", "message": "Not found"})
    return HTMLResponse(_UI_FILE.read_text(encoding="utf-8"))


@app.get("/", include_in_schema=False)
def root() -> dict:
    return {
        "service": "visa-ocr",
        "version": app.version,
        "health": "/healthz",
        **({"ui": "/ui"} if config.ui_enabled() else {}),
        **({"docs": "/docs"} if _docs else {}),
        "endpoints": ["POST /v1/documents/extract", "POST /v1/passports/extract", "POST /v1/documents/check",
                      "POST /v1/photos/check", "GET /v1/jobs/{id}", "POST /v1/mrz/parse",
                      "POST /v1/jobs/{id}/corrections", "GET /v1/models"],
    }


class MrzParseRequest(BaseModel):
    lines: list[str] = Field(min_length=2, max_length=3)
    expected_nationality: str | None = Field(None, description="Declared nationality, 3-letter ISO code (optional)")
    expected_surname: str | None = Field(None, max_length=100)
    expected_given_names: str | None = Field(None, max_length=100)
    expected_document_number: str | None = Field(None, max_length=30)
    expected_birth_date: str | None = Field(None, max_length=10, description="YYYY-MM-DD")
    expected_expiry_date: str | None = Field(None, max_length=10, description="YYYY-MM-DD")


def _applicant(**kw) -> Applicant:
    clean = {k: (v.strip() if isinstance(v, str) and v.strip() else None) for k, v in kw.items()}
    return Applicant(**clean)


@app.post("/v1/mrz/parse")
def mrz_parse(body: MrzParseRequest, request: Request, owner: str = Depends(security.authenticate)) -> dict:
    security.limiter.check(security.client_id(request, owner))
    security.allow(request, "parse", owner)
    if any(len(line) > 200 for line in body.lines):
        raise HTTPException(422, {"code": "mrz_unsupported", "message": "An MRZ line is far too long"})
    try:
        result = pipeline.analyse_mrz(body.lines)
    except MrzError as exc:
        raise HTTPException(status_code=422, detail={"code": "mrz_unsupported", "message": str(exc)})
    applicant = _applicant(nationality=body.expected_nationality, surname=body.expected_surname,
                           given_names=body.expected_given_names, document_number=body.expected_document_number,
                           birth_date=body.expected_birth_date, expiry_date=body.expected_expiry_date)
    a_flags, _ = compare_applicant(result["document"], applicant)
    result["flags"] += [f.to_dict() for f in a_flags]
    result["needsReview"] = any(f["severity"] == "error" for f in result["flags"])
    return result


def _decode_slots() -> asyncio.Semaphore:
    """At most two full-size decodes in memory at a time, on every route that decodes an upload (extract and both checks).
    Waiting requests queue here on the event loop and hold no pool thread."""
    global _shrink_slots
    if _shrink_slots is None:
        _shrink_slots = asyncio.Semaphore(2)
    return _shrink_slots


async def _upload_images(data: bytes, claimed_type: str | None, first_page_only: bool = False) -> list[bytes]:
    """The image(s) in one uploaded file. A JPEG or PNG is itself. A PDF is checked by the malware scanner as it came, then
    drawn page by page in a separate, limited process (app/pdf.py); the PDF itself goes no further. Every page image then
    passes validate_image like any upload. Raises UploadError."""
    if not pdf.is_pdf(data):
        validate_image(data, claimed_type)
        return [data]
    if len(data) > config.max_bytes():
        raise UploadError(413, "too_large", f"File exceeds {config.max_bytes() // (1024 * 1024)} MB")
    await run_in_threadpool(scan_for_malware, data)             # the PDF as uploaded, before anything opens it
    try:
        async with _decode_slots():
            pages = await run_in_threadpool(pdf.render, data, first_page_only)
    except pdf.PdfError as exc:
        raise UploadError(422 if exc.code in ("pdf_too_many_pages", "pdf_empty") else 415, exc.code, exc.message) from None
    for page in pages:
        validate_image(page, "image/jpeg")
    return pages


def _process(job_id: str, images: list[bytes], hint: str | None, applicant: Applicant) -> None:
    try:
        worker.process_job(store, job_id, images, hint, applicant, lambda: get_engine())   # looked up per call (tests patch it)
    finally:
        with _inflight_lock:
            _inflight.discard(job_id)


@app.post("/v1/documents/check")
async def check_image(request: Request, file: UploadFile = File(...), owner: str = Depends(security.authenticate)) -> dict:
    """Is this photo clear enough to send? Answers at once from the pixels alone (nothing is read, stored or queued), so the
    portal can ask for another photo while the applicant still has the document in hand. A photo that passes here can still
    come back from the reading with a retake request: glare over a few characters only shows when they are read."""
    security.limiter.check(security.client_id(request, owner))
    security.allow(request, "upload", owner)
    data = await file.read(config.max_bytes() + 1)
    data = (await _upload_images(data, file.content_type, first_page_only=True))[0]
    async with _decode_slots():
        q = await run_in_threadpool(quality.measure, data)
    reasons = q.issues or q.doubts
    body = {"clear": q.clear, "acceptable": not q.issues, "quality": q.to_dict()}
    if reasons:
        body["retake"] = {**quality.retake(reasons), "recommended": bool(q.issues)}
    return body


@app.post("/v1/documents/extract", status_code=202)
async def extract(
    request: Request,
    files: list[UploadFile] = File(...),
    doc_type_hint: str | None = Form(None, max_length=32),
    consent_training: bool = Form(False),
    expected_nationality: str | None = Form(None, max_length=3),
    expected_surname: str | None = Form(None, max_length=100),
    expected_given_names: str | None = Form(None, max_length=100),
    expected_document_number: str | None = Form(None, max_length=30),
    expected_birth_date: str | None = Form(None, max_length=10),
    expected_expiry_date: str | None = Form(None, max_length=10),
    owner: str = Depends(security.authenticate),
) -> dict:
    security.limiter.check(security.client_id(request, owner))
    security.allow(request, "upload", owner)
    if not 1 <= len(files) <= config.max_files():
        raise HTTPException(422, {"code": "bad_file_count", "message": f"Send 1 to {config.max_files()} images"})
    # Count and reserve in one step: a separate "is there room" check lets concurrent uploads overrun the cap.
    job = store.create_if_room(request.state.correlation_id, owner, config.max_pending_jobs())
    if job is None:
        raise HTTPException(503, {"code": "busy", "message": "The service is busy; try again shortly"},
                            headers={"Retry-After": "10"})
    try:
        images: list[bytes] = []
        for f in files:
            data = await f.read(config.max_bytes() + 1)
            was_pdf = pdf.is_pdf(data)
            pages = await _upload_images(data, f.content_type)    # a PDF becomes one image per page
            if len(images) + len(pages) > config.max_files():
                raise HTTPException(422, {"code": "bad_file_count",
                                          "message": f"Send at most {config.max_files()} images in total (each PDF page counts as one)"})
            for page in pages:
                if not was_pdf:
                    await run_in_threadpool(scan_for_malware, page)    # only when UPLOAD_SCAN_COMMAND is set
                async with _decode_slots():        # decode and resize off the event loop
                    images.append(await run_in_threadpool(shrink_for_ocr, page))   # queue, readers and training store get this image
        if config.refuse_unclear_at_upload():
            # Measured from the pixels in a fraction of a second: an unusable photo is answered now, not when its turn comes.
            clarity = await run_in_threadpool(quality.clearest, images)
            if clarity.issues:
                audit.event("upload_refused", key=owner, reason="image_unclear", issues=",".join(clarity.issues))
                raise HTTPException(422, {"code": "image_unclear", "message": quality.retake_message(clarity.issues),
                                          "retake": quality.retake(clarity.issues), "quality": clarity.to_dict()})
        if consent_training and training.enabled:
            try:
                training.save_images(job.id, images)
                store.update(job.id, consent_training=True)
            except training_store.StoreFull:
                # the reading goes on; only the copy for training is not kept
                audit.event("training_store_full", job=job.id)
        applicant = _applicant(nationality=expected_nationality, surname=expected_surname, given_names=expected_given_names,
                               document_number=expected_document_number, birth_date=expected_birth_date,
                               expiry_date=expected_expiry_date)
        if getattr(store, "queue", False):
            store.submit(job.id, images, doc_type_hint, worker.applicant_dict(applicant))
        else:
            with _inflight_lock:
                # The store's count frees a slot when the sweep fails a job that waited too long, but its work is still in
                # this process. Admission also counts what the process still holds, so the cap keeps limiting real work.
                if len(_inflight) >= config.max_pending_jobs():
                    raise HTTPException(503, {"code": "busy", "message": "The service is busy; try again shortly"},
                                        headers={"Retry-After": "10"})
                _inflight.add(job.id)
            _ocr_pool.submit(_process, job.id, images, doc_type_hint, applicant)
    except BaseException:
        with _inflight_lock:
            _inflight.discard(job.id)
        store.discard(job.id)                      # give the slot back; nothing was queued
        if training.enabled:
            training.remove(job.id)
        raise
    audit.event("job_created", job=job.id, key=owner, images=len(images), consent_training=bool(consent_training and training.enabled))
    return {"jobId": job.id, "status": job.status, "correlationId": job.correlation_id}


@app.post("/v1/passports/extract", status_code=202)
async def extract_passport(
    request: Request,
    files: list[UploadFile] = File(...),
    consent_training: bool = Form(False),
    expected_nationality: str | None = Form(None, max_length=3),
    expected_surname: str | None = Form(None, max_length=100),
    expected_given_names: str | None = Form(None, max_length=100),
    expected_document_number: str | None = Form(None, max_length=30),
    expected_birth_date: str | None = Form(None, max_length=10),
    expected_expiry_date: str | None = Form(None, max_length=10),
    owner: str = Depends(security.authenticate),
) -> dict:
    """Passports only, MRZ only. The upload is taken in and read like any other (same checks, same job, same result
    shape), but nothing except the machine-readable zone of a passport is accepted: a card, a visa, a residence permit or
    a page with no MRZ comes back failed, with a message for the applicant. POST /v1/documents/extract stays the endpoint
    for every kind of document."""
    return await extract(request, files, pipeline.PASSPORT_MRZ, consent_training, expected_nationality, expected_surname,
                         expected_given_names, expected_document_number, expected_birth_date, expected_expiry_date, owner)


@app.post("/v1/cards/extract", status_code=202)
async def extract_card(
    request: Request,
    files: list[UploadFile] = File(...),
    consent_training: bool = Form(False),
    expected_nationality: str | None = Form(None, max_length=3),
    expected_surname: str | None = Form(None, max_length=100),
    expected_given_names: str | None = Form(None, max_length=100),
    expected_document_number: str | None = Form(None, max_length=30),
    expected_birth_date: str | None = Form(None, max_length=10),
    expected_expiry_date: str | None = Form(None, max_length=10),
    owner: str = Depends(security.authenticate),
) -> dict:
    """Cards with no MRZ only: the printed front of a Gulf ID or of a European, UK or US residence card. The upload is
    taken in and read like any other (same checks, same job, same result shape), and no MRZ is searched for, which makes it
    the quickest way to read a card. A passport, a visa or the MRZ side of a card comes back failed, with a message for the
    applicant. Nothing printed carries a check digit, so every card read here is for a person to confirm.
    POST /v1/documents/extract stays the endpoint for every kind of document."""
    return await extract(request, files, pipeline.CARD_TEXT, consent_training, expected_nationality, expected_surname,
                         expected_given_names, expected_document_number, expected_birth_date, expected_expiry_date, owner)


@app.post("/v1/photos/check")
async def check_photo(request: Request, file: UploadFile = File(...), owner: str = Depends(security.authenticate)) -> dict:
    """Is this personal photo usable for the application? Answers at once from the picture alone: one face, large enough,
    centred and whole, sharp, well lit, on a light plain background. Nothing is stored or queued, and the face is not
    compared with anything."""
    security.limiter.check(security.client_id(request, owner))
    security.allow(request, "upload", owner)
    data = await file.read(config.max_bytes() + 1)
    data = (await _upload_images(data, file.content_type, first_page_only=True))[0]
    try:
        async with _decode_slots():
            verdict = await run_in_threadpool(photo.check, data)
    except photo.Unavailable as exc:
        raise HTTPException(503, {"code": "photo_check_unavailable", "message": str(exc)}) from None
    audit.event("photo_checked", key=owner, acceptable=verdict["acceptable"], issues=",".join(verdict["issues"]))
    return verdict


def _own_job(job_id: str, owner: str, any_job: bool = False):
    # Job ids are 32 hex characters. Anything else is answered here and never reaches the store (a NUL byte, for one, makes
    # PostgreSQL raise).
    job = store.get(job_id) if re.fullmatch(r"[0-9a-f]{32}", job_id) else None
    # A reviewer or administrator key may open any job; every other key only its own.
    if job is not None and owner != "anonymous" and job.owner != owner and not any_job:
        audit.event("job_read_refused", job=job.id, key=owner)     # another key's job: recorded, answered as not found
        job = None
    if job is None:                                # same answer for "not yours" and "not found"
        raise HTTPException(404, {"code": "job_not_found", "message": "Unknown or expired job id"})
    return job


@app.get("/v1/jobs/{job_id}")
def get_job(job_id: str, request: Request, owner: str = Depends(security.authenticate)) -> dict:
    role = security.allow(request, "read", owner)
    job = _own_job(job_id, owner, security.may_read_any(request))
    if job.owner != owner:
        # A reviewer or administrator opening another key's job reads an applicant's data: always recorded. The owner's own
        # polling is not, since the portal asks every second or two.
        audit.event("job_read", job=job.id, key=owner, role=role, cross_owner=True)
    return job.to_dict()


def _plain(v: str, what: str) -> str:
    """Stored text is shown to a second person in a terminal: no control, formatting or direction-changing characters.
    Arabic needs none of them; the two joiners (U+200C, U+200D) are allowed."""
    if any((not c.isprintable() or unicodedata.category(c) == "Cf") and c not in "\u200c\u200d" for c in v):
        raise ValueError(f"{what}: control and formatting characters are not allowed")
    return v


class LineLabel(BaseModel):
    """One text line a reviewer marked on an uploaded image (used to cut training crops)."""
    image: int = Field(ge=0, le=63, description="Index of the uploaded image")
    box: list[Annotated[int, Field(ge=0, le=100_000)]] = Field(min_length=4, max_length=4, description="x1, y1, x2, y2 in pixels")
    text: str = Field(min_length=1, max_length=200)
    lang: Literal["en", "ar"]

    @field_validator("text")
    @classmethod
    def _no_control_chars(cls, v: str) -> str:
        return _plain(v, "text")


class CorrectionRequest(BaseModel):
    reviewer: str = Field(min_length=1, max_length=100)
    document: dict[str, str] = Field(default_factory=dict, description="Corrected field values")
    lines: list[LineLabel] = Field(default_factory=list, max_length=200)

    @field_validator("reviewer")
    @classmethod
    def _plain_reviewer(cls, v: str) -> str:
        return _plain(v, "reviewer")

    @field_validator("document")
    @classmethod
    def _bounded_document(cls, v: dict[str, str]) -> dict[str, str]:
        if len(v) > 50 or any(len(k) > 64 or len(x) > 500 for k, x in v.items()):
            raise ValueError("document has too many fields or a field that is too long")
        for k, x in v.items():
            _plain(k, "document field name")
            _plain(x, "document field value")
        return v


@app.post("/v1/jobs/{job_id}/corrections", status_code=201)
def add_correction(job_id: str, body: CorrectionRequest, request: Request, owner: str = Depends(security.authenticate)) -> dict:
    # Its own limit, always on (also when RATE_LIMIT_PER_MIN=0): each correction is a file on the volume that holds the jobs
    security.limiter.check("correct:" + security.client_id(request, owner), limit=config.corrections_per_minute())
    security.allow(request, "correct", owner)
    job = _own_job(job_id, owner, security.may_read_any(request))
    if not training.enabled:
        raise HTTPException(409, {"code": "training_store_disabled",
                                  "message": "Training data collection is not enabled on this server"})
    if not job.consent_training:
        raise HTTPException(409, {"code": "no_training_consent",
                                  "message": "The applicant did not consent to training use for this job"})
    # Nothing here is trusted yet. The correction is stored as "pending" with the key that sent it; it becomes training data
    # only after a second person approves it on the server (tools/approve_corrections.py). What the reader produced is kept
    # beside it: which fields the reviewer changed, and how confident the reading was, are what the confidence thresholds
    # are later checked against (tools/calibration_report.py).
    before = (job.result or {}).get("document") or {}
    # Only fields the result holds count as checked or changed: a key the reader never returned says nothing about it.
    checked = sorted(k for k in body.document if k in before)
    changed = [k for k in checked if str(before.get(k, "") or "") != body.document[k]]
    ocr = {"engine": (job.result or {}).get("engine"), "needsReview": (job.result or {}).get("needsReview"),
           "confidence": ((job.result or {}).get("confidence") or {}).get("mrz"), "fieldsChanged": changed,
           "fieldsChecked": checked}
    try:
        training.save_correction(job_id, {**body.model_dump(), "submittedBy": owner, "ocr": ocr})
    except training_store.StoreFull as exc:
        raise HTTPException(429, {"code": "too_many_corrections", "message": str(exc)}) from None
    except FileNotFoundError:                      # purged since the upload (retention, or consent withdrawn)
        raise HTTPException(409, {"code": "no_training_consent",
                                  "message": "Training data for this job was removed"}) from None
    audit.event("correction_submitted", job=job_id, key=owner, lines=len(body.lines), fields_changed=len(changed))
    return {"stored": True, "status": "pending", "lines": len(body.lines)}


@app.get("/v1/models")
def list_models(request: Request, owner: str = Depends(security.authenticate)) -> dict:
    security.allow(request, "models", owner)
    registry = Registry()
    try:
        # fresh=False: a model folder is hashed again only when one of its files changed size or time, so asking often is cheap
        return {"models": [
            {"id": m.id, "task": m.task, "lang": m.lang, "version": m.version, "active": m.active,
             "status": registry.status(m, fresh=False), "metrics": m.metrics, "replaced": m.replaced}
            for m in registry.models()
        ]}
    except ManifestError as exc:
        raise HTTPException(503, {"code": "manifest_refused", "message": str(exc)}) from None
