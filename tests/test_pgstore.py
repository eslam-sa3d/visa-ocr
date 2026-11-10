"""PostgreSQL store and queue. Skipped unless TEST_DATABASE_URL points at a disposable database."""
import io
import os
import threading
import time

import pytest
from fastapi.testclient import TestClient
from PIL import Image

URL = os.environ.get("TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(not URL, reason="set TEST_DATABASE_URL to run PostgreSQL tests")

TD3 = ["P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<", "L898902C36UTO7408122F1204159ZE184226B<<<<<10"]


@pytest.fixture
def store():
    from app.pgstore import PgJobStore
    s = PgJobStore(URL, ttl_seconds=600)
    with s._conn() as c:
        c.execute("DELETE FROM jobs")
    return s


def test_roundtrip_and_images_erased(store):
    job = store.create("cid", "me")
    assert store.get(job.id).status == "queued"
    store.submit(job.id, [b"img1", b"img2"], "passport", {"surname": "X"})
    jid, images, hint, applicant = store.claim()
    assert (jid, images, hint, applicant) == (job.id, [b"img1", b"img2"], "passport", {"surname": "X"})
    assert store.get(job.id).status == "processing"
    store.update(job.id, status="done", result={"document": {"surname": "Y"}})
    got = store.get(job.id)
    assert got.status == "done" and got.result["document"]["surname"] == "Y" and got.owner == "me"
    with store._conn() as c:
        assert c.execute("SELECT images FROM jobs WHERE id=%s", (job.id,)).fetchone()[0] is None


def test_each_job_is_claimed_once_by_concurrent_workers(store):
    ids = []
    for _ in range(20):
        j = store.create("c", "o")
        store.submit(j.id, [b"x"], None, {})
        ids.append(j.id)
    taken, lock = [], threading.Lock()

    def worker():
        while (c := store.claim()) is not None:
            with lock:
                taken.append(c[0])
    ts = [threading.Thread(target=worker) for _ in range(6)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert sorted(taken) == sorted(ids)                    # no job lost, none taken twice


def test_unsubmitted_job_is_not_claimed(store):
    store.create("c", "o")
    assert store.claim() is None


def test_dead_worker_job_is_retried_then_failed(store, monkeypatch):
    monkeypatch.setenv("JOB_STALE_SECONDS", "0")
    monkeypatch.setenv("JOB_MAX_ATTEMPTS", "2")
    j = store.create("c", "o")
    store.submit(j.id, [b"x"], None, {})
    assert store.claim()[0] == j.id                        # worker 1 dies without finishing
    time.sleep(0.05)
    assert store.requeue_stale() == 1 and store.get(j.id).status == "queued"
    assert store.claim()[0] == j.id                        # worker 2 also dies
    time.sleep(0.05)
    store.requeue_stale()
    got = store.get(j.id)
    assert got.status == "failed" and got.error["code"] == "internal_error"


def test_jobs_expire(store):
    from app.pgstore import PgJobStore
    short = PgJobStore(URL, ttl_seconds=0.2)
    j = short.create("c", "o")
    time.sleep(0.4)
    assert short.get(j.id) is None


def test_api_end_to_end_through_queue(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", URL)
    monkeypatch.setenv("WORKER_THREADS", "2")
    import importlib
    import app.main as main
    main = importlib.reload(main)
    from app.engines.base import EngineOutput

    class Fixed:
        name = "fixed"

        def extract(self, images, hint):
            return EngineOutput(engine="fixed", mrz_lines=TD3, mrz_confidence=0.99)

    monkeypatch.setattr(main, "get_engine", lambda: Fixed())
    b = io.BytesIO()
    Image.new("RGB", (300, 200), "white").save(b, "PNG")
    try:
        with TestClient(main.app) as c:                      # the context manager starts the worker threads
            ids = [c.post("/v1/documents/extract", files=[("files", ("a.png", b.getvalue(), "image/png"))],
                          data={"expected_surname": "ERIKSSON", "expected_given_names": "ANNA MARIA"}).json()["jobId"]
                   for _ in range(4)]
            for jid in ids:
                for _ in range(50):
                    j = c.get(f"/v1/jobs/{jid}").json()
                    if j["status"] in ("done", "failed"):
                        break
                    time.sleep(0.1)
                assert j["status"] == "done" and j["result"]["document"]["surname"] == "ERIKSSON"
                assert j["result"]["fieldStatus"]["names"] == "confirmed by form"   # applicant data survived the queue
    finally:
        monkeypatch.delenv("DATABASE_URL")
        importlib.reload(main)
