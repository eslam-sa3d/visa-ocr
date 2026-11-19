"""Tests added with the check against the NCA Essential Cybersecurity Controls (ECC-2:2024). Each names its control."""
import hashlib
import json
import ssl
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
from app import audit, config, crypto, main, security  # noqa: E402
from app.training_store import TrainingStore  # noqa: E402
from tests.test_model_and_training import MRZ, FixedEngine as _Fixed, png_bytes  # noqa: E402

KEYS = {"portal": "p" * 40, "reviewer": "r" * 40, "admin": "a" * 40, "old": "o" * 40}


@pytest.fixture
def roles(tmp_path, monkeypatch):
    """Four keys with separate duties, one of them expired yesterday; the engine is a stand-in."""
    entries = [{"name": n, "sha256": hashlib.sha256(k.encode()).hexdigest(), "role": r, **e} for n, k, r, e in (
        ("portal", KEYS["portal"], "uploader", {"expires": (date.today() + timedelta(days=30)).isoformat()}),
        ("review-ui", KEYS["reviewer"], "reviewer", {}), ("ops", KEYS["admin"], "admin", {}),
        ("retired", KEYS["old"], "uploader", {"expires": (datetime.now(timezone.utc).date() - timedelta(days=1)).isoformat()}))]
    f = tmp_path / "keys.json"
    f.write_text(json.dumps(entries))
    monkeypatch.setenv("API_KEYS_FILE", str(f))
    monkeypatch.delenv("API_KEYS", raising=False)
    monkeypatch.delenv("API_KEY_SHA256", raising=False)
    monkeypatch.setenv("AUDIT_LOG_FILE", str(tmp_path / "audit.log"))
    monkeypatch.setattr(main, "get_engine", lambda: _Fixed())
    monkeypatch.setattr(main, "training", TrainingStore(tmp_path / "store"))
    monkeypatch.setattr(security, "limiter", security.RateLimiter())
    return TestClient(main.app), tmp_path / "audit.log"


def h(name):
    return {"X-API-Key": KEYS[name]}


def events(log: Path) -> list[dict]:
    return [json.loads(ln) for ln in log.read_text().splitlines()] if log.exists() else []


def upload(client, name="portal", consent=True):
    return client.post("/v1/documents/extract", files=[("files", ("p.png", png_bytes(), "image/png"))],
                       data={"consent_training": "true" if consent else "false"}, headers=h(name))


def test_each_role_can_do_only_its_own_work(roles):
    """ECC 2-2-3-3: least privilege and segregation of duties between the portal, the reviewers and operations."""
    client, _ = roles
    r = upload(client)
    assert r.status_code == 202
    job = r.json()["jobId"]
    body = {"reviewer": "amal", "document": {}, "lines": []}
    assert client.get(f"/v1/jobs/{job}", headers=h("portal")).status_code == 200
    assert client.post(f"/v1/jobs/{job}/corrections", json=body, headers=h("portal")).status_code == 403      # uploader cannot correct
    assert client.get("/v1/models", headers=h("portal")).status_code == 403
    assert upload(client, "reviewer").status_code == 403                                                   # reviewer cannot upload
    assert client.get(f"/v1/jobs/{job}", headers=h("reviewer")).status_code == 200                         # but reads any job
    assert client.post(f"/v1/jobs/{job}/corrections", json=body, headers=h("reviewer")).status_code == 201
    assert client.get("/v1/models", headers=h("reviewer")).status_code == 403
    assert client.get("/v1/models", headers=h("admin")).status_code == 200
    assert upload(client, "admin").status_code == 403
    assert client.post(f"/v1/jobs/{job}/corrections", json=body, headers=h("admin")).status_code == 403


def test_expired_and_unknown_keys_are_refused_and_recorded(roles):
    """ECC 2-2-3-5 and 2-12-3: a key past its date stops working; failures are in the security log."""
    client, log = roles
    r = upload(client, "old")
    assert r.status_code == 401 and r.json()["detail"]["code"] == "key_expired"
    assert client.get("/v1/models", headers={"X-API-Key": "wrong" * 8}).status_code == 401
    names = [e["event"] for e in events(log)]
    assert "auth_key_expired" in names and "auth_failed" in names
    for _ in range(config.bad_key_limit_per_minute() + 1):
        client.get("/v1/models", headers={"X-API-Key": "wrong" * 8})
    assert "auth_throttled" in [e["event"] for e in events(log)]


def test_security_log_records_actions_without_document_values(roles, tmp_path):
    """ECC 2-12-3: who did what and when, ready for a SIEM; never a name, a number or a line of the document."""
    client, log = roles
    job = upload(client).json()["jobId"]
    body = {"reviewer": "Amal SECRETREVIEWER", "document": {"surname": "SECRETSURNAME"},
            "lines": [{"image": 0, "box": [1, 2, 300, 40], "text": MRZ[0], "lang": "en"}]}
    assert client.post(f"/v1/jobs/{job}/corrections", json=body, headers=h("reviewer")).status_code == 201
    assert client.post(f"/v1/jobs/{job}/corrections", json=body, headers=h("portal")).status_code == 403
    main.training.decide(job, "omar")
    main.training.remove(job, tombstone=True)
    got = events(log)
    by = {e["event"]: e for e in got}
    assert {"job_created", "correction_submitted", "forbidden", "correction_decided", "training_data_removed"} <= set(by)
    assert by["job_created"]["job"] == job and len(by["job_created"]["key"]) == 16 and by["correction_decided"]["decision"] == "approved"
    assert by["forbidden"]["role"] == "uploader" and by["forbidden"]["permission"] == "correct"
    assert all(e["ts"].endswith("+00:00") for e in got)
    text = log.read_text()
    assert "SECRET" not in text and MRZ[0] not in text and KEYS["portal"] not in text and "ERIKSSON" not in text
    assert oct(log.stat().st_mode & 0o777) == "0o600"
    assert audit.event("x", nested={"a": 1}, ok="y", n=3, tags=["a", "b"]) .keys() == {"ts", "event", "ok", "n", "tags"}


def test_model_changes_are_recorded(tmp_path, monkeypatch):
    """ECC 2-12-3: activating a model and going back are privileged actions."""
    sys.path.insert(0, str(ROOT / "training"))
    import register_model
    from app.model_registry import ModelInfo, Registry
    monkeypatch.setenv("AUDIT_LOG_FILE", str(tmp_path / "audit.log"))
    reg = Registry(tmp_path / "models")
    for name in ("v1", "v2"):
        (reg.base / name).mkdir(parents=True)
        (reg.base / name / "w").write_text(name)
        register_model.register(reg, ModelInfo(id=name, task="text_recognition", lang="ar", version=name, path=name,
                                               metrics={"cer": 0.05 if name == "v1" else 0.03}), activate=True, force=False)
    register_model.rollback(reg, "text_recognition", "ar")
    names = [e["event"] for e in events(tmp_path / "audit.log")]
    assert names == ["model_registered", "model_registered", "model_activated", "model_rolled_back"]


def test_production_mode_refuses_every_unsafe_setting(tmp_path, monkeypatch):
    """ECC 2-15-3, 2-8-3: the service cannot reach production with authentication off, the test page on, unchecked models or
    unencrypted storage."""
    for name in ("API_KEYS", "API_KEY_SHA256", "API_KEYS_FILE", "DATA_KEY_FILE", "DATA_AT_REST_ENCRYPTED", "UI_ENABLED"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("API_DOCS", "1")
    monkeypatch.setenv("ALLOW_UNVERIFIED_MODELS", "1")
    monkeypatch.setenv("ALLOW_ANONYMOUS", "1")
    monkeypatch.setenv("JOB_DB_PATH", str(tmp_path / "jobs.db"))
    problems = " | ".join(main.production_problems())
    for word in ("no API key", "test page", "API pages", "without a checksum", "not encrypted", "ALLOW_ANONYMOUS"):
        assert word in problems, word
    monkeypatch.setenv("PRODUCTION", "1")
    for name in ("API_DOCS", "ALLOW_UNVERIFIED_MODELS", "ALLOW_ANONYMOUS"):
        monkeypatch.delenv(name)
    monkeypatch.setenv("API_KEYS", "k" * 40)
    monkeypatch.setenv("DATA_AT_REST_ENCRYPTED", "1")
    assert main.production_problems() == []                    # the test page is off by itself in production
    client = TestClient(main.app)
    assert client.get("/ui").status_code == 404 and "ui" not in client.get("/").json()
    monkeypatch.setenv("API_KEYS", "short")
    assert any("shorter than 24" in p for p in main.production_problems())


def test_start_script_refuses_a_self_signed_certificate_in_production(tmp_path):
    import subprocess
    env = {"PATH": __import__("os").environ["PATH"], "API_KEYS": "k" * 40, "PRODUCTION": "1", "HOME": str(tmp_path),
           "TLS_CERT": str(tmp_path / "none.crt"), "TLS_KEY": str(tmp_path / "none.key")}
    r = subprocess.run(["sh", str(ROOT / "docker-entrypoint.sh")], env=env, capture_output=True, text=True, timeout=30)
    assert r.returncode == 1 and "self-signed one is not generated in production" in r.stderr
    assert not (tmp_path / "none.key").exists()


def test_tls_cipher_list_allows_only_tls_1_2_and_1_3_with_forward_secrecy():
    """ECC 2-8-3-1: no TLS 1.0 or 1.1, no cipher without forward secrecy or authenticated encryption."""
    import re
    script = (ROOT / "docker-entrypoint.sh").read_text()
    ciphers = re.search(r'TLS_CIPHERS="\$\{TLS_CIPHERS:-([^}]+)\}"', script).group(1)
    assert "--ssl-ciphers" in script
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.set_ciphers(ciphers)
    chosen = ctx.get_ciphers()
    assert chosen and {c["protocol"] for c in chosen} <= {"TLSv1.2", "TLSv1.3"}
    assert all(c["aead"] for c in chosen)
    assert all(c["protocol"] == "TLSv1.3" or "ECDHE" in c["name"] for c in chosen)
    assert ctx.minimum_version >= ssl.TLSVersion.TLSv1_2


def _scanner(tmp_path, body):
    script = tmp_path / "scan.sh"
    script.write_text("#!/bin/sh\n" + body + "\n")
    script.chmod(0o755)
    return str(script)


def test_uploads_are_scanned_when_a_scanner_is_configured(tmp_path, monkeypatch):
    """ECC 2-3-3-1: clean passes, infected is refused, and a scanner that cannot answer refuses too (never lets a file by)."""
    monkeypatch.setattr(main, "get_engine", lambda: _Fixed())
    monkeypatch.setenv("AUDIT_LOG_FILE", str(tmp_path / "audit.log"))
    client = TestClient(main.app)
    send = lambda: client.post("/v1/documents/extract", files=[("files", ("p.png", png_bytes(), "image/png"))])  # noqa: E731
    seen = tmp_path / "seen"
    monkeypatch.setenv("UPLOAD_SCAN_COMMAND", _scanner(tmp_path, f'head -c 4 "$1" > {seen}; exit 0'))
    assert send().status_code == 202 and seen.read_bytes() == b"\x89PNG"            # the scanner really got the file
    monkeypatch.setenv("UPLOAD_SCAN_COMMAND", _scanner(tmp_path, "exit 1"))
    r = send()
    assert r.status_code == 422 and r.json()["detail"]["code"] == "malware_detected"
    for broken in (_scanner(tmp_path, "exit 2"), str(tmp_path / "missing-scanner"), _scanner(tmp_path, "sleep 5")):
        monkeypatch.setenv("UPLOAD_SCAN_COMMAND", broken)
        monkeypatch.setenv("UPLOAD_SCAN_SECONDS", "1")
        r = send()
        assert r.status_code == 503 and r.json()["detail"]["code"] == "scan_unavailable"
    names = [e["event"] for e in events(tmp_path / "audit.log")]
    assert names.count("upload_malware_found") == 1 and names.count("upload_scan_unavailable") == 3
    assert not list(Path(__import__("tempfile").gettempdir()).glob("visa-ocr-scan-*"))    # no upload left behind
    monkeypatch.delenv("UPLOAD_SCAN_COMMAND")
    assert send().status_code == 202


def test_aes_256_gcm_option_for_stored_data(tmp_path, monkeypatch):
    """ECC 2-8-3-3: 256-bit authenticated encryption when the classification calls for it; both ciphers stay readable."""
    from cryptography.fernet import Fernet
    key_file = tmp_path / "k"
    key_file.write_text(Fernet.generate_key().decode() + "\n")
    monkeypatch.setenv("DATA_KEY_FILE", str(key_file))
    monkeypatch.delenv("DATA_CIPHER", raising=False)
    old = crypto.seal(b"passport image")
    monkeypatch.setenv("DATA_CIPHER", "aes256gcm")
    new = crypto.seal(b"passport image")
    assert old.startswith(b"VOENC1:") and new.startswith(b"VOENC2:") and b"passport" not in new
    assert crypto.unseal(old) == crypto.unseal(new) == b"passport image"
    assert crypto.seal(b"x") != crypto.seal(b"x")                                    # a fresh nonce every time
    assert crypto.unseal_json(crypto.seal_json({"a": "ب"})) == {"a": "ب"} and crypto.unseal_text(crypto.seal_text("نص")) == "نص"
    tampered = new[:-6] + (b"AAAAAA" if not new.endswith(b"AAAAAA") else b"BBBBBB")
    with pytest.raises(crypto.DataKeyError):
        crypto.unseal(tampered)
    import os
    import time
    key_file.write_text(Fernet.generate_key().decode() + "\n" + key_file.read_text())   # rotate: new key on top
    os.utime(key_file, (time.time() + 5, time.time() + 5))
    assert crypto.unseal(new) == b"passport image" and crypto.unseal(crypto.seal(b"y")) == b"y"


def test_bill_of_materials_lists_every_locked_package_with_hashes():
    """ECC 1-6-3-2: every component is known, pinned by hash, with its licence."""
    sys.path.insert(0, str(ROOT / "tools"))
    import dependency_audit
    for lock in ("requirements.lock", "requirements-lite.lock"):
        packages = dependency_audit.locked(ROOT / lock)
        assert len(packages) > 30 and all(p["hashes"] >= 1 for p in packages), lock
    sbom = json.loads((ROOT / "docs" / "sbom.json").read_text())
    assert sbom["unpinned"] == [] and {p["name"] for p in sbom["packages"]} == {p["name"] for p in dependency_audit.locked(ROOT / "requirements.lock")}
