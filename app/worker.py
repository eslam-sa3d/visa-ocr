"""Runs OCR jobs. Used in-process by the API (threads) and on its own: `python -m app.worker`."""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import asdict
from typing import Callable

from . import pipeline
from .engines import EngineUnavailable, get_engine
from .review import Applicant

log = logging.getLogger("visa-ocr")


def process_job(store, job_id: str, images: list[bytes], hint: str | None, applicant: Applicant,
                engine_getter: Callable = get_engine) -> None:
    store.update(job_id, status="processing")
    try:
        status, payload = pipeline.run(images, hint, applicant, engine_getter())
    except EngineUnavailable as exc:
        status, payload = "failed", {"code": "engine_unavailable", "message": str(exc)}
    except Exception:  # noqa: BLE001
        log.exception("job %s failed", job_id)
        status, payload = "failed", {"code": "internal_error", "message": "Processing failed"}
    if status == "done":
        store.update(job_id, status="done", result=payload)
    else:
        store.update(job_id, status="failed", error=payload)


def work_once(store, engine_getter: Callable = get_engine) -> bool:
    """Claim and run one queued job. Returns False when the queue is empty."""
    claimed = store.claim()
    if claimed is None:
        return False
    job_id, images, hint, applicant = claimed
    known = Applicant.__dataclass_fields__
    process_job(store, job_id, images, hint, Applicant(**{k: v for k, v in applicant.items() if k in known}), engine_getter)
    return True


def worker_loop(store, stop: threading.Event, engine_getter: Callable = get_engine, idle: float = 0.5) -> None:
    last_sweep = 0.0
    while not stop.is_set():
        try:
            if time.time() - last_sweep > 30:
                store.requeue_stale()
                last_sweep = time.time()
            if not work_once(store, engine_getter):
                stop.wait(idle)
        except Exception:  # noqa: BLE001  - a database blip must not kill the worker
            log.exception("worker loop error")
            stop.wait(2)


def applicant_dict(a: Applicant) -> dict:
    return asdict(a)


if __name__ == "__main__":  # pragma: no cover
    import signal
    from .jobs import make_store
    logging.basicConfig(level=logging.INFO)
    st = make_store()
    if not getattr(st, "queue", False):
        raise SystemExit("Set DATABASE_URL: a stand-alone worker needs the PostgreSQL queue")
    ev = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: ev.set())
    signal.signal(signal.SIGINT, lambda *_: ev.set())
    worker_loop(st, ev)
