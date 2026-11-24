"""The service sizes itself to the container it was given."""
import pytest

from app import config, readers


def _container(monkeypatch, files: dict, seen: int = 64, meminfo_gb: float = 256):
    monkeypatch.delenv("READER_PROCESSES", raising=False)
    monkeypatch.setattr(readers, "_first_line", lambda path: files.get(path, ""))
    monkeypatch.setattr(readers.os, "sched_getaffinity", lambda _pid: set(range(seen)), raising=False)
    if not any("memory" in p for p in files):
        monkeypatch.setattr(readers, "container_memory_bytes", lambda: int(meminfo_gb * (1 << 30)))


@pytest.mark.parametrize("files,expect", [
    ({"/sys/fs/cgroup/cpu.max": "600000 100000", "/sys/fs/cgroup/memory.max": str(18 << 30)}, 6),      # 6 cores, 18 GB
    ({"/sys/fs/cgroup/cpu.max": "100000 100000", "/sys/fs/cgroup/memory.max": str(2 << 30)}, 1),       # 1 core, 2 GB
    ({"/sys/fs/cgroup/cpu.max": "800000 100000", "/sys/fs/cgroup/memory.max": str(4 << 30)}, 1),       # memory decides
    ({"/sys/fs/cgroup/cpu.max": "800000 100000", "/sys/fs/cgroup/memory.max": str(9 << 30)}, 4),
    ({"/sys/fs/cgroup/cpu/cpu.cfs_quota_us": "400000", "/sys/fs/cgroup/cpu/cpu.cfs_period_us": "100000",
      "/sys/fs/cgroup/memory/memory.limit_in_bytes": str(16 << 30)}, 4),                                # cgroup v1
    ({"/sys/fs/cgroup/cpu.max": "50000 100000", "/sys/fs/cgroup/memory.max": str(8 << 30)}, 1),        # half a core
])
def test_readers_follow_the_container_limits_not_the_host(monkeypatch, files, expect):
    """The host here shows 64 cores and 256 GB; the container's own limits decide."""
    _container(monkeypatch, files)
    assert readers.count() == expect


def test_no_limit_uses_what_the_machine_has(monkeypatch):
    _container(monkeypatch, {"/sys/fs/cgroup/cpu.max": "max 100000",
                             "/sys/fs/cgroup/memory/memory.limit_in_bytes": str(9223372036854771712)}, seen=4, meminfo_gb=16)
    monkeypatch.setattr(readers, "container_memory_bytes", lambda: 16 << 30)
    assert readers.container_cores() == 4 and readers.count() == 4


def test_a_number_can_be_forced_and_the_rest_follows(monkeypatch):
    _container(monkeypatch, {"/sys/fs/cgroup/cpu.max": "600000 100000", "/sys/fs/cgroup/memory.max": str(18 << 30)})
    for name in ("MAX_PENDING_JOBS", "JOB_THREADS", "WORKER_THREADS"):
        monkeypatch.delenv(name, raising=False)
    assert (readers.count(), config.job_threads(), config.worker_threads(), config.max_pending_jobs()) == (6, 6, 6, 192)
    monkeypatch.setenv("READER_PROCESSES", "2")
    assert (readers.count(), config.job_threads(), config.max_pending_jobs()) == (2, 2, 64) and readers.active()
    monkeypatch.setenv("MAX_PENDING_JOBS", "500")
    assert config.max_pending_jobs() == 500
    monkeypatch.setenv("READER_PROCESSES", "1")
    assert not readers.active()


def test_one_reader_reads_in_the_service_process(monkeypatch):
    """With one reader nothing changes: the stand-in reader of a test is the one that reads."""
    from app import worker
    from app.engines.base import EngineOutput
    from app.review import Applicant
    lines = ["P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<", "L898902C36UTO7408122F1204159ZE184226B<<<<<10"]

    class Reader:
        name = "stand-in"
        def extract(self, images, hint):
            return EngineOutput(engine=self.name, mrz_lines=lines, mrz_confidence=0.99)
    import io
    from PIL import Image
    b = io.BytesIO(); Image.new("RGB", (400, 300), "white").save(b, "PNG")
    status, res = worker._run("j", [b.getvalue()], None, Applicant(), lambda: Reader())
    assert status == "done" and res["engine"] == "stand-in"


def test_reader_processes_read_documents_and_stop_with_the_service(monkeypatch):
    """Two real reader processes: each reads a document, the readings come from other processes than this one, and no
    reader is left behind when the service stops."""
    import os
    import sys
    import time
    from concurrent.futures import ThreadPoolExecutor
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
    import perf_test
    from app import worker
    from app.review import Applicant
    monkeypatch.setenv("READER_PROCESSES", "2")
    monkeypatch.setenv("OCR_ENGINE", "ensemble")
    monkeypatch.setenv("IMAGE_QUALITY_CHECK", "1")
    readers.stop()
    try:
        pool = readers._get_pool()
        for f in [pool.submit(readers._child_ping, 0.2) for _ in range(2)]:
            f.result(timeout=120)
        pids = set(pool._processes)              # a process is started for each task sent while none is idle
        assert len(pids) == 2 and os.getpid() not in pids
        image = perf_test.make("TD3", 3)[0]
        with ThreadPoolExecutor(2) as ex:
            results = list(ex.map(lambda i: worker._run(f"j{i}", [image], None, Applicant(), None), range(2)))
        assert [status for status, _ in results] == ["done", "done"]
        assert all(res["document"]["format"] == "TD3" and "timings" in res for _, res in results)
    finally:
        readers.stop()
    deadline = time.time() + 15
    while time.time() < deadline and any(_alive(p) for p in pids):
        time.sleep(0.5)
    assert not any(_alive(p) for p in pids)


def _alive(pid: int) -> bool:
    try:
        with open(f"/proc/{pid}/stat", encoding="ascii") as fh:
            return fh.read().split(") ")[-1].split()[0] != "Z"
    except OSError:
        return False
