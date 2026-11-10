"""Regression tests for the security-audit findings (label injection, auto-accept expiry, image shape, body size, limiter)."""
import io
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

import app.main as main
from app import security
from app.review import names_match
from app.uploads import UploadError, upscaled_size, validate_image
from app.visual_zone import parse_gcc_id

L = lambda text, y, score=0.99: {"text": text, "score": score, "box": [[0, y], [200, y], [200, y + 20], [0, y + 20]]}
GOOD_ID = "2000000006"


def png(size):
    b = io.BytesIO()
    Image.new("L", size, 255).save(b, format="PNG")
    return b.getvalue()


# 1. label-file record injection
@pytest.mark.parametrize("bad", ["x\nimages/a.png\tPOISON", "a\tb", "a\rb", "a b"])
def test_correction_text_rejects_control_characters(bad):
    r = TestClient(main.app).post("/v1/jobs/abc/corrections", json={
        "reviewer": "r", "lines": [{"image": 0, "box": [0, 0, 1, 1], "text": bad, "lang": "en"}]})
    assert r.status_code == 422


def test_dataset_writer_skips_forged_records(tmp_path):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from training.prepare_dataset import write
    out = tmp_path / "x.txt"
    write(out, [{"path": "a.png", "text": "HONEST"}, {"path": "b.png", "text": "x\nc.png\tPOISON"}])
    assert out.read_text() == "a.png\tHONEST\n"


# 2. auto-accept must not wave through an unverifiable expiry
def test_auto_accept_reviews_hijri_and_missing_expiry():
    hijri = parse_gcc_id([L("Iqama No: " + GOOD_ID, 10), L("Expiry Date: 1430/01/01", 40)], auto_accept=True)
    none = parse_gcc_id([L("Iqama No: " + GOOD_ID, 10)], auto_accept=True)
    assert hijri["needsReview"] and none["needsReview"]


# 3. image shape and derived size
def test_narrow_tall_image_is_rejected():
    with pytest.raises(UploadError) as e:
        validate_image(png((64, 4000)), "image/png")
    assert e.value.detail["code"] == "image_bad_aspect"


def test_upscale_never_exceeds_pixel_cap(monkeypatch):
    monkeypatch.setenv("MAX_IMAGE_PIXELS", "2000000")
    assert upscaled_size(64, 625_000, 1600) is None
    assert upscaled_size(800, 400, 1600) == (1600, 800)


# 4. oversized bodies are refused before parsing and before auth
def test_oversized_body_is_refused_before_auth(monkeypatch):
    monkeypatch.setenv("API_KEYS", "secret")
    monkeypatch.setenv("MAX_FILES", "1")
    monkeypatch.setenv("MAX_UPLOAD_BYTES", "1000")
    big = b"x" * (3 * 1024 * 1024)
    r = TestClient(main.app).post("/v1/documents/extract", files=[("files", ("a.png", big, "image/png"))])
    assert r.status_code == 413


# 5. wrong keys are throttled
def test_wrong_keys_are_rate_limited(monkeypatch):
    monkeypatch.setenv("API_KEYS", "secret")
    monkeypatch.setenv("BAD_KEY_LIMIT_PER_MIN", "3")
    security.limiter._hits.clear()
    c = TestClient(main.app)
    codes = [c.get("/v1/models", headers={"X-API-Key": "wrong"}).status_code for _ in range(6)]
    assert codes[:3] == [401] * 3 and 429 in codes


def test_forwarded_for_uses_last_hop(monkeypatch):
    monkeypatch.setenv("TRUST_PROXY", "true")
    class R:  # minimal request stand-in
        headers = {"X-Forwarded-For": "6.6.6.6, 10.0.0.9"}
        client = None
    assert security.client_id(R(), "anonymous") == "10.0.0.9"


# 6. name comparison
def test_names_match_edge_cases():
    assert not names_match([], [])
    assert not names_match(["ABDULLAH"], ["AB"])
    assert names_match(["ABDULLAH"], ["ABD"])


# 7. entrypoint refuses to start open
def test_entrypoint_refuses_without_keys():
    script = Path(__file__).resolve().parents[1] / "docker-entrypoint.sh"
    r = subprocess.run(["sh", str(script)], env={"PATH": "/usr/bin:/bin"}, capture_output=True, text=True)
    assert r.returncode == 1 and "API key" in r.stderr


# ---- housekeeping and permissions
def test_sqlite_store_is_owner_only_and_sweeps(tmp_path):
    import os, stat, time
    from app.jobs import SqliteJobStore
    db = tmp_path / "jobs.db"
    s = SqliteJobStore(str(db), ttl_seconds=1)
    s.create("c1", "owner")
    assert stat.S_IMODE(os.stat(db).st_mode) == 0o600
    time.sleep(1.2)
    s.sweep()                                   # purges without any create/get
    import sqlite3
    assert sqlite3.connect(db).execute("select count(*) from jobs").fetchone()[0] == 0


def test_memory_store_sweep_drops_expired(monkeypatch):
    import time
    from app.jobs import JobStore
    s = JobStore(ttl_seconds=1)
    s.create("c", "o")
    time.sleep(1.2)
    s.sweep()
    assert not s._jobs


def test_training_store_dirs_are_private_and_purge_works(tmp_path):
    import os, stat
    from app.training_store import TrainingStore
    t = TrainingStore(tmp_path / "ts")
    t.save_images("job1", [png((100, 100))])
    assert stat.S_IMODE(os.stat(tmp_path / "ts").st_mode) == 0o700
    assert stat.S_IMODE(os.stat(tmp_path / "ts" / "jobs").st_mode) == 0o700
    old = (tmp_path / "ts" / "jobs" / "job1")
    os.utime(old, (1, 1))
    assert t.purge_older_than(1) == 1


def test_wrong_key_limit_is_stricter_than_normal(monkeypatch):
    monkeypatch.setenv("API_KEYS", "secret")
    monkeypatch.setenv("RATE_LIMIT_PER_MIN", "60")
    monkeypatch.setenv("BAD_KEY_LIMIT_PER_MIN", "2")
    security.limiter._hits.clear()
    c = TestClient(main.app)
    codes = [c.get("/v1/models", headers={"X-API-Key": "nope"}).status_code for _ in range(4)]
    assert codes[:2] == [401, 401] and codes[2] == 429


def test_weak_key_detection(monkeypatch):
    monkeypatch.setenv("API_KEYS", "short,abcdefghijklmnopqrstuvwxyz0123")
    assert security.weak_plain_keys() == 1
