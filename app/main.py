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
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from starlette.concurrency import run_in_threadpool
from pydantic import BaseModel, Field, field_validator

from . import config, pipeline, security, worker
from .engines import get_engine
from .jobs import make_store
from .model_registry import Registry
from .mrz import MrzError
from .review import Applicant, compare_applicant
from .training_store import TrainingStore
from .uploads import validate_image, shrink_for_ocr

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
            except Exception:  # noqa: BLE001 - a failed sweep must not stop the next one
                log.warning("housekeeping sweep failed")

    hk = threading.Thread(target=housekeeping, daemon=True)
    hk.start()
    yield
    stop.set()


app = FastAPI(title="Visa Portal Document OCR", version="0.3.0", lifespan=lifespan)
app.add_middleware(security.BodyLimit)
training = TrainingStore()  # disabled unless TRAINING_STORE_DIR is set
# In-process OCR jobs run on their own small pool, never on the framework's thread pool, so waiting or slow jobs cannot
# take the threads that answer /healthz and job lookups.
_ocr_pool = ThreadPoolExecutor(max_workers=config.job_threads(), thread_name_prefix="ocr")
_shrink_slots: asyncio.Semaphore | None = None      # created on first use, inside the running event loop

if security.weak_plain_keys():
    log.warning("Some API_KEYS values are shorter than 24 characters. Use long random keys (openssl rand -hex 32).")
if not security.auth_enabled():       # raises when a key variable is set but holds no usable key: refuse to start
    log.warning("API_KEYS is not set: authentication is OFF. Do not expose this service to a network.")


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
        for m in registry.models():
            st = registry.status(m) if m.active else "ok"
            if st in ("missing", "mismatch"):
                out.append(f"model {m.id} is {st}")
        _model_cache.update(at=time.monotonic(), problems=out)
        return list(out)


@app.get("/readyz")
def readyz() -> dict:
    """Ready when the job store works and every active OCR model is present and unchanged."""
    problems: list[str] = []
    try:
        store.count()
    except Exception:  # noqa: BLE001
        problems.append("job store unavailable")
    # Which model problems stop the service depends on the engine in use. The Tesseract + RapidOCR pair needs no model files.
    # "auto" falls back to that pair when PaddleOCR models are absent, so a missing model is a warning there; a model whose
    # files changed (checksum mismatch) is always a failure.
    engine = os.environ.get("OCR_ENGINE", "auto")
    warnings: list[str] = []
    if engine in ("auto", "paddle-ocr"):
        for p in _model_problems():
            (warnings if engine == "auto" and p.endswith("is missing") else problems).append(p)
    if problems:
        raise HTTPException(503, {"code": "not_ready", "message": "; ".join(problems)})
    out = {"status": "ready", "authEnabled": security.auth_enabled()}
    if warnings:
        out["warnings"] = warnings
    return out


_UI_FILE = Path(__file__).parent / "static" / "index.html"


@app.get("/ui", include_in_schema=False, response_class=HTMLResponse)
def ui() -> HTMLResponse:
    """Plain test page: upload an image and see the result. No external assets (works offline)."""
    return HTMLResponse(_UI_FILE.read_text(encoding="utf-8"))


@app.get("/", include_in_schema=False)
def root() -> dict:
    return {
        "service": "visa-ocr",
        "version": app.version,
        "health": "/healthz",
        "ui": "/ui",
        "docs": "/docs",
        "endpoints": ["POST /v1/documents/extract", "GET /v1/jobs/{id}", "POST /v1/mrz/parse",
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


def _process(job_id: str, images: list[bytes], hint: str | None, applicant: Applicant) -> None:
    worker.process_job(store, job_id, images, hint, applicant, lambda: get_engine())   # looked up per call (tests patch it)


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
    if not 1 <= len(files) <= config.max_files():
        raise HTTPException(422, {"code": "bad_file_count", "message": f"Send 1 to {config.max_files()} images"})
    # Count and reserve in one step: a separate "is there room" check lets concurrent uploads overrun the cap.
    job = store.create_if_room(request.state.correlation_id, owner, config.max_pending_jobs())
    if job is None:
        raise HTTPException(503, {"code": "busy", "message": "The service is busy; try again shortly"},
                            headers={"Retry-After": "10"})
    global _shrink_slots
    if _shrink_slots is None:
        _shrink_slots = asyncio.Semaphore(2)       # at most two full-size decodes in memory at a time
    try:
        images: list[bytes] = []
        for f in files:
            data = await f.read(config.max_bytes() + 1)
            validate_image(data, f.content_type)
            async with _shrink_slots:              # decode and resize off the event loop
                images.append(await run_in_threadpool(shrink_for_ocr, data))   # queue, readers and training store get this image
        if consent_training and training.enabled:
            training.save_images(job.id, images)
            store.update(job.id, consent_training=True)
        applicant = _applicant(nationality=expected_nationality, surname=expected_surname, given_names=expected_given_names,
                               document_number=expected_document_number, birth_date=expected_birth_date,
                               expiry_date=expected_expiry_date)
        if getattr(store, "queue", False):
            store.submit(job.id, images, doc_type_hint, worker.applicant_dict(applicant))
        else:
            _ocr_pool.submit(_process, job.id, images, doc_type_hint, applicant)
    except BaseException:
        store.discard(job.id)                      # give the slot back; nothing was queued
        if training.enabled:
            training.remove(job.id)
        raise
    return {"jobId": job.id, "status": job.status, "correlationId": job.correlation_id}


def _own_job(job_id: str, owner: str):
    job = store.get(job_id)
    if job is None or (owner != "anonymous" and job.owner != owner):   # same answer for "not yours" and "not found"
        raise HTTPException(404, {"code": "job_not_found", "message": "Unknown or expired job id"})
    return job


@app.get("/v1/jobs/{job_id}")
def get_job(job_id: str, owner: str = Depends(security.authenticate)) -> dict:
    return _own_job(job_id, owner).to_dict()


class LineLabel(BaseModel):
    """One text line a reviewer marked on an uploaded image (used to cut training crops)."""
    image: int = Field(ge=0, description="Index of the uploaded image")
    box: list[int] = Field(min_length=4, max_length=4, description="x1, y1, x2, y2 in pixels")
    text: str = Field(min_length=1, max_length=200)
    lang: Literal["en", "ar"]

    @field_validator("text")
    @classmethod
    def _no_control_chars(cls, v: str) -> str:
        if any(ord(c) < 32 or ord(c) == 127 or c in "\x85\u2028\u2029" for c in v):
            raise ValueError("control characters (tab, line break) are not allowed")
        return v


class CorrectionRequest(BaseModel):
    reviewer: str = Field(min_length=1, max_length=100)
    document: dict[str, str] = Field(default_factory=dict, description="Corrected field values")
    lines: list[LineLabel] = Field(default_factory=list, max_length=200)

    @field_validator("document")
    @classmethod
    def _bounded_document(cls, v: dict[str, str]) -> dict[str, str]:
        if len(v) > 50 or any(len(k) > 64 or len(x) > 500 for k, x in v.items()):
            raise ValueError("document has too many fields or a field that is too long")
        return v


@app.post("/v1/jobs/{job_id}/corrections", status_code=201)
def add_correction(job_id: str, body: CorrectionRequest, owner: str = Depends(security.authenticate)) -> dict:
    job = _own_job(job_id, owner)
    if not training.enabled:
        raise HTTPException(409, {"code": "training_store_disabled",
                                  "message": "Training data collection is not enabled on this server"})
    if not job.consent_training:
        raise HTTPException(409, {"code": "no_training_consent",
                                  "message": "The applicant did not consent to training use for this job"})
    try:
        training.save_correction(job_id, body.model_dump())
    except FileNotFoundError:                      # purged since the upload (retention, or consent withdrawn)
        raise HTTPException(409, {"code": "no_training_consent",
                                  "message": "Training data for this job was removed"}) from None
    return {"stored": True, "lines": len(body.lines)}


@app.get("/v1/models")
def list_models(owner: str = Depends(security.authenticate)) -> dict:
    registry = Registry()
    return {"models": [
        {"id": m.id, "task": m.task, "lang": m.lang, "version": m.version, "active": m.active,
         "status": registry.status(m), "metrics": m.metrics}
        for m in registry.models()
    ]}
