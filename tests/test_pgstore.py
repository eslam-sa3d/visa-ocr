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
    monkeypatch.setenv("JOB_MAX_ATTEMPTS", "2")

    def age():                                             # the claim was not renewed for longer than JOB_STALE_SECONDS
        with store._conn() as c:
            c.execute("UPDATE jobs SET locked_ts = locked_ts - 10000 WHERE id=%s", (j.id,))

    j = store.create("c", "o")
    store.submit(j.id, [b"x"], None, {})
    assert store.claim()[0] == j.id                        # worker 1 dies without finishing
    assert store.requeue_stale() == 0                      # a fresh claim is not stale
    age()
    assert store.requeue_stale() == 1 and store.get(j.id).status == "queued"
    assert store.claim()[0] == j.id                        # worker 2 also dies
    age()
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


# ---------------------------------------------- third audit: values sealed for their job, column and owner
@pytest.fixture(params=["", "aes256gcm"])
def keyed(request, tmp_path, monkeypatch):
    from cryptography.fernet import Fernet
    k = tmp_path / "data.key"
    k.write_text(Fernet.generate_key().decode() + "\n")
    monkeypatch.setenv("DATA_KEY_FILE", str(k))
    if request.param:
        monkeypatch.setenv("DATA_CIPHER", request.param)
    else:
        monkeypatch.delenv("DATA_CIPHER", raising=False)
    return request.param


def test_queue_values_are_sealed_and_bound_to_their_row(store, keyed):
    import json as _json
    from app import crypto
    a, b = store.create("c1", "ownerA"), store.create("c2", "ownerB")
    store.submit(a.id, [b"imageA"], "passport", {"surname": "ALPHA"})
    store.submit(b.id, [b"imageB"], None, {"surname": "BETA"})
    with store._conn() as c:
        raw = c.execute("SELECT images, applicant FROM jobs WHERE id=%s", (a.id,)).fetchone()
    assert bytes(raw[0][0]).startswith((crypto.PREFIX_BOUND, crypto.PREFIX_GCM_BOUND)) and b"ALPHA" not in _json.dumps(raw[1]).encode()
    claimed = {store.claim()[0], store.claim()[0]}
    assert claimed == {a.id, b.id}
    store.update(a.id, status="done", result={"surname": "ALPHA"})
    store.update(b.id, status="done", result={"surname": "BETA"})
    assert store.get(a.id).result == {"surname": "ALPHA"}
    with store._conn() as c:
        c.execute("UPDATE jobs SET owner='ownerB' WHERE id=%s", (a.id,))       # owner changed in the database
    with pytest.raises(crypto.DataKeyError):
        store.get(a.id)
    with store._conn() as c:
        c.execute("UPDATE jobs SET owner='ownerA' WHERE id=%s", (a.id,))
        c.execute("UPDATE jobs SET result=(SELECT result FROM jobs WHERE id=%s) WHERE id=%s", (a.id, b.id))   # value copied
    with pytest.raises(crypto.DataKeyError):
        store.get(b.id)


def test_plain_value_written_to_the_queue_is_refused(store, keyed):
    from psycopg.types.json import Jsonb
    from app import crypto
    j = store.create("c", "o")
    store.submit(j.id, [b"x"], None, {})
    store.claim()
    store.update(j.id, status="done", result={"needsReview": True})
    with store._conn() as c:
        c.execute("UPDATE jobs SET result=%s WHERE id=%s", (Jsonb({"needsReview": False, "document": {"surname": "FORGED"}}), j.id))
    with pytest.raises(crypto.DataKeyError):
        store.get(j.id)


def test_finish_and_dead_worker_error_are_sealed_and_readable(store, keyed, monkeypatch):
    from app import crypto
    monkeypatch.setenv("JOB_MAX_ATTEMPTS", "1")
    j = store.create("c", "o")
    store.submit(j.id, [b"x"], None, {})
    claim = store.claim()
    with store._conn() as c:
        c.execute("UPDATE jobs SET locked_ts = locked_ts - 10000 WHERE id=%s", (j.id,))
    store.requeue_stale()
    got = store.get(j.id)
    assert got.status == "failed" and got.error["code"] == "internal_error"
    with store._conn() as c:
        err = c.execute("SELECT error FROM jobs WHERE id=%s", (j.id,)).fetchone()[0]
    assert set(err) == {"enc"}
    k = store.create("c", "o2")
    store.submit(k.id, [b"y"], None, {})
    claim = store.claim()
    assert store.finish(k.id, claim.token, "done", result={"surname": "DONE"})
    assert store.get(k.id).result == {"surname": "DONE"}


def test_values_from_before_the_upgrade_need_the_migration_flag(store, keyed, monkeypatch):
    from psycopg.types.json import Jsonb
    from app import crypto
    j = store.create("c", "o")
    old = crypto.seal_json({"surname": "OLD"})                 # 1.0.0 sealed without a context
    with store._conn() as c:
        c.execute("UPDATE jobs SET status='done', result=%s WHERE id=%s", (Jsonb(old), j.id))
    with pytest.raises(crypto.DataKeyError):
        store.get(j.id)
    monkeypatch.setenv("DATA_ALLOW_UNBOUND_READ", "1")
    assert store.get(j.id).result == {"surname": "OLD"}
