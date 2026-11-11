"""Runs OCR jobs. Used in-process by the API (threads) and on its own: `python -m app.worker`."""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import asdict
from typing import Callable

from . import config, pipeline
from .engines import EngineUnavailable, get_engine
from .review import Applicant

log = logging.getLogger("visa-ocr")


def _run(job_id: str, images: list[bytes], hint: str | None, applicant: Applicant, engine_getter: Callable) -> tuple[str, dict]:
    try:
        return pipeline.run(images, hint, applicant, engine_getter())
    except EngineUnavailable as exc:
        return "failed", {"code": "engine_unavailable", "message": str(exc)}
    except Exception:  # noqa: BLE001
        log.exception("job %s failed", job_id)
        return "failed", {"code": "internal_error", "message": "Processing failed"}


def process_job(store, job_id: str, images: list[bytes], hint: str | None, applicant: Applicant,
                engine_getter: Callable = get_engine, token: int | None = None) -> None:
    """Run one job and store the outcome. With a claim token (PostgreSQL queue) the claim is renewed while the job runs and
    the outcome is written only if this run still holds the claim."""
    if token is None:
        store.update(job_id, status="processing")
        status, payload = _run(job_id, images, hint, applicant, engine_getter)
        store.update(job_id, status=status, **({"result": payload} if status == "done" else {"error": payload}))
        return
    done = threading.Event()

    def beat() -> None:
        while not done.wait(max(1.0, config.job_stale_seconds() / 3)):
            try:
                if not store.heartbeat(job_id, token):
                    return                              # claim lost: stop renewing; finish() will drop the result
            except Exception:  # noqa: BLE001 - a database blip; try again on the next beat
                log.warning("heartbeat failed for job %s", job_id)

    threading.Thread(target=beat, daemon=True).start()
    try:
        status, payload = _run(job_id, images, hint, applicant, engine_getter)
    finally:
        done.set()
    kept = store.finish(job_id, token, status, **({"result": payload} if status == "done" else {"error": payload}))
    if not kept:
        log.warning("job %s: this run lost its claim; its result was dropped", job_id)


def work_once(store, engine_getter: Callable = get_engine) -> bool:
    """Claim and run one queued job. Returns False when the queue is empty."""
    claimed = store.claim()
    if claimed is None:
        return False
    job_id, images, hint, applicant = claimed
    known = Applicant.__dataclass_fields__
    process_job(store, job_id, images, hint, Applicant(**{k: v for k, v in applicant.items() if k in known}), engine_getter,
                token=getattr(claimed, "token", None))
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
