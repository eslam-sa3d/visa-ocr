"""How many documents this service reads at the same time, worked out from what the container was given.

A reader holds its own copy of the reading models and reads one document at a time. One reader per processor core, as long
as each has about 2 GB of memory, uses the whole container whatever its size: the same image runs on 1 core and 2 GB, or on
6 cores and 18 GB, without a setting. READER_PROCESSES forces a number.

With one reader the documents are read in the service's own process, as before. With more, each reader is a separate
process (the main reader cannot read two documents at once inside one process) and uploads are shared between them.
"""
from __future__ import annotations

import logging
import math
import multiprocessing
import os
import threading
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool

log = logging.getLogger("visa-ocr")
_NO_LIMIT = 1 << 60            # cgroup v1 reports "no limit" as a number near the top of the range


def _first_line(path: str) -> str:
    try:
        with open(path, encoding="ascii") as fh:
            return fh.readline().strip()
    except (OSError, ValueError):
        return ""


def container_cores() -> float:
    """Processor cores this container may use: its CPU limit when one is set, else the cores it can see. The host's core
    count is the wrong number inside a limited container."""
    try:
        seen = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        seen = os.cpu_count() or 1
    quota = None
    parts = _first_line("/sys/fs/cgroup/cpu.max").split()                    # cgroup v2: "<quota|max> <period>"
    if len(parts) == 2 and parts[0] != "max":
        try:
            quota = int(parts[0]) / int(parts[1])
        except (ValueError, ZeroDivisionError):
            quota = None
    if quota is None:                                                        # cgroup v1
        try:
            q = int(_first_line("/sys/fs/cgroup/cpu/cpu.cfs_quota_us") or -1)
            p = int(_first_line("/sys/fs/cgroup/cpu/cpu.cfs_period_us") or 0)
            quota = q / p if q > 0 and p > 0 else None
        except ValueError:
            quota = None
    return max(0.1, min(seen, quota)) if quota else float(seen)


def container_memory_bytes() -> int:
    """Memory this container may use: its limit when one is set, else the machine's memory. 0 when neither can be read."""
    for path in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        text = _first_line(path)
        if text.isdigit() and 0 < int(text) < _NO_LIMIT:
            return int(text)
    try:
        with open("/proc/meminfo", encoding="ascii") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, OSError, ValueError):
        return 0


def count() -> int:
    """Readers to run: READER_PROCESSES when set to a number, else one per core, each with READER_MEMORY_GB (2) of memory
    after half a gigabyte is left for the service itself."""
    forced = os.environ.get("READER_PROCESSES", "").strip().lower()
    if forced.isdigit() and int(forced) > 0:
        return int(forced)
    try:
        each = max(0.5, float(os.environ.get("READER_MEMORY_GB", "2")))
    except ValueError:
        each = 2.0
    by_cores = max(1, math.floor(container_cores() + 0.01))
    memory = container_memory_bytes()
    by_memory = max(1, math.floor((memory / (1 << 30) - 0.5) / each)) if memory else by_cores
    return max(1, min(by_cores, by_memory))


def describe() -> dict:
    """What was found and decided, for /readyz and the start-up log."""
    return {"readers": count(), "cores": round(container_cores(), 2),
            "memoryGb": round(container_memory_bytes() / (1 << 30), 1)}


# ---------------------------------------------------------------------------------------------------------------------
# reader processes

_pool: ProcessPoolExecutor | None = None
_pool_lock = threading.Lock()


def _child_start(threads: int) -> None:
    """Runs once in each reader process: one document at a time here, on this reader's share of the cores, models loaded
    before the first upload arrives."""
    os.environ["READER_PROCESSES"] = "1"
    os.environ["JOB_THREADS"] = "1"
    os.environ.setdefault("PADDLE_CPU_THREADS", str(threads))
    os.environ.setdefault("OMP_NUM_THREADS", str(threads))
    logging.basicConfig(level=logging.INFO)
    parent = os.getppid()

    def leave_with_the_service() -> None:      # a reader never outlives the service that started it
        import time
        while os.getppid() == parent:
            time.sleep(2)
        os._exit(0)
    threading.Thread(target=leave_with_the_service, name="parent-watch", daemon=True).start()
    warm_here()


def warm_here() -> None:
    """Load the readers now by reading a blank page, so the first applicant does not wait for it."""
    try:
        import io
        from PIL import Image
        from .engines import get_engine
        blank = io.BytesIO()
        Image.new("RGB", (640, 420), "white").save(blank, format="PNG")
        get_engine().extract([blank.getvalue()], None)
    except Exception as exc:  # noqa: BLE001 - a failed warm-up is not a failed start: the first read will report the cause
        log.warning("readers not loaded at start-up: %s", type(exc).__name__)


def _child_ping(seconds: float) -> int:
    import time
    time.sleep(seconds)
    return os.getpid()


def _child_read(job_id: str, images: list[bytes], hint: str | None, applicant: dict) -> tuple[str, dict]:
    from . import worker
    from .engines import get_engine
    from .review import Applicant
    known = Applicant.__dataclass_fields__
    return worker.run_here(job_id, images, hint, Applicant(**{k: v for k, v in applicant.items() if k in known}), get_engine)


def active() -> bool:
    """True when documents are read in separate reader processes (more than one reader)."""
    return count() > 1


def _get_pool() -> ProcessPoolExecutor:
    global _pool
    with _pool_lock:
        if _pool is None:
            n = count()
            threads = max(1, math.floor(container_cores() / n + 0.01))
            os.environ.setdefault("MALLOC_ARENA_MAX", "2")        # read by each reader process as it starts
            _pool = ProcessPoolExecutor(max_workers=n, mp_context=multiprocessing.get_context("spawn"),
                                        initializer=_child_start, initargs=(threads,))
            log.info("reading with %d reader processes, %d thread(s) each: %s", n, threads, describe())
        return _pool


def _end(pool: ProcessPoolExecutor) -> None:
    """Stop a pool and its reader processes now (a reader in the middle of a document is not waited for)."""
    processes = list((getattr(pool, "_processes", None) or {}).values())
    pool.shutdown(wait=False, cancel_futures=True)
    for process in processes:
        try:
            process.terminate()
        except Exception:  # noqa: BLE001 - already gone
            pass


def _drop_pool(broken: ProcessPoolExecutor) -> None:
    global _pool
    with _pool_lock:
        if _pool is broken:
            _pool = None
    _end(broken)


def read(job_id: str, images: list[bytes], hint: str | None, applicant: dict, timeout: float) -> tuple[str, dict]:
    """Read one document in a reader process. A reader that died (stopped for memory, say) takes the pool down with it: a
    fresh pool is started and the document is read once more."""
    for attempt in (1, 2):
        pool = _get_pool()
        try:
            return pool.submit(_child_read, job_id, images, hint, applicant).result(timeout=timeout)
        except BrokenProcessPool:
            log.error("a reader process stopped while reading job %s (attempt %d); starting fresh readers", job_id, attempt)
            _drop_pool(pool)
    return "failed", {"code": "internal_error", "message": "Processing failed"}


def start() -> None:
    """Bring the readers up at start-up. More than one reader: start every reader process (each loads its models as it
    starts). One reader: load the models in this process, in the background."""
    if os.environ.get("WARM_UP", "1").strip() == "0":
        return
    if active():
        pool = _get_pool()
        for _ in range(count()):
            pool.submit(_child_ping, 0.5)          # each waits a moment, so the pool starts one process per task
    else:
        threading.Thread(target=warm_here, name="warm-up", daemon=True).start()


def stop() -> None:
    global _pool
    with _pool_lock:
        pool, _pool = _pool, None
    if pool is not None:
        _end(pool)
