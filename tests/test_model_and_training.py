import importlib.util
import io
import json
import subprocess
import sys
import types
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import app.main as main
from app.engines.base import EngineUnavailable
from app.engines.paddle_ocr import PaddleOcrEngine
from app.model_registry import ModelInfo, Registry, dir_sha256
from app.textutil import normalize_arabic, script_of, to_ascii_digits
from app.training_store import TrainingStore

ROOT = Path(__file__).resolve().parent.parent
MRZ = ["P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<", "L898902C36UTO7408122F1204159ZE184226B<<<<<10"]
FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/Library/Fonts/Arial Unicode.ttf",
]


def load_script(name):
    spec = importlib.util.spec_from_file_location(f"training_{name}", ROOT / "training" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def png_bytes(size=(400, 120)):
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", size, "white").save(buf, format="PNG")
    return buf.getvalue()


# --------------------------------------------------------------------------- text helpers
def test_arabic_indic_digits_become_ascii():
    assert to_ascii_digits("١٢/٠٣/١٩٩٠") == "12/03/1990"


def test_script_detection():
    assert script_of("محمد") == "arabic"
    assert script_of("MOHAMMED") == "latin"
    assert script_of("الاسم: Ali") == "mixed"
    assert script_of("12/03/1990") == "other"


def test_arabic_normalisation_for_comparison():
    assert normalize_arabic("أحمد") == normalize_arabic("احمد")
    assert normalize_arabic("مُحَمَّد") == "محمد"


# --------------------------------------------------------------------------- evaluation
def test_edit_distance_and_cer_by_script():
    ev = load_script("evaluate")
    assert ev.edit_distance("kitten", "sitting") == 3
    labels = {"a.png": "محمد", "b.png": "ABC"}
    preds = {"a.png": "محمد", "b.png": "ABD"}
    m = ev.evaluate(labels, preds)
    assert m["samples"] == 2 and m["cer"] == round(1 / 7, 4)
    assert m["byScript"]["arabic"]["cer"] == 0.0
    assert m["byScript"]["latin"]["cer"] == round(1 / 3, 4)


def test_evaluate_options_and_missing_predictions():
    ev = load_script("evaluate")
    labels = {"a.png": "أحمد", "b.png": "x"}
    assert ev.evaluate(labels, {"a.png": "احمد"})["cer"] > 0
    m = ev.evaluate(labels, {"a.png": "احمد"}, arabic=True)
    assert m["cer"] == 0.0 and m["missingPredictions"] == 1


# --------------------------------------------------------------------------- registry
def make_models_dir(tmp_path):
    base = tmp_path / "models"
    (base / "ar_v1").mkdir(parents=True)
    (base / "ar_v1" / "weights.bin").write_bytes(b"one")
    (base / "ar_v2").mkdir()
    (base / "ar_v2" / "weights.bin").write_bytes(b"two")
    return Registry(base)


def test_registry_status_and_tamper_detection(tmp_path):
    reg = make_models_dir(tmp_path)
    m = ModelInfo(id="ar_v1", task="text_recognition", lang="ar", version="v1", path="ar_v1", active=True)
    assert reg.status(m) == "unverified"
    m.sha256 = dir_sha256(reg.full_path(m))
    assert reg.status(m) == "verified"
    (reg.full_path(m) / "weights.bin").write_bytes(b"tampered")
    assert reg.status(m) == "mismatch"
    m.path = "does_not_exist"
    assert reg.status(m) == "missing"


def test_register_model_requires_improvement_to_activate(tmp_path):
    reg_mod = load_script("register_model")
    reg = make_models_dir(tmp_path)
    old = ModelInfo(id="ar_v1", task="text_recognition", lang="ar", version="v1", path="ar_v1",
                    metrics={"cer": 0.10})
    reg_mod.register(reg, old, activate=True, force=False)
    worse = ModelInfo(id="ar_v2", task="text_recognition", lang="ar", version="v2", path="ar_v2",
                      metrics={"cer": 0.12})
    with pytest.raises(SystemExit):
        reg_mod.register(reg, worse, activate=True, force=False)
    better = ModelInfo(id="ar_v2", task="text_recognition", lang="ar", version="v2", path="ar_v2",
                       metrics={"cer": 0.07})
    reg_mod.register(reg, better, activate=True, force=False)
    active = reg.active("text_recognition", "ar")
    assert active.id == "ar_v2" and reg.status(active) == "verified"
    assert [m.active for m in reg.models() if m.id == "ar_v1"] == [False]


def test_models_endpoint_lists_default_manifest():
    r = TestClient(main.app).get("/v1/models")
    assert r.status_code == 200
    ids = {m["id"] for m in r.json()["models"]}
    assert {"en_PP-OCRv5_mobile_rec", "arabic_PP-OCRv5_mobile_rec"} <= ids


# --------------------------------------------------------------------------- engine (PaddleOCR faked)
class FakePaddleOCR:
    def __init__(self, **kwargs):
        FakePaddleOCR.kwargs = kwargs

    def predict(self, array):
        return [{"rec_texts": [MRZ[0], MRZ[1], "١٢٣"], "rec_scores": [0.99, 0.98, 0.9],
                 "rec_polys": [[[0, 0], [90, 0], [90, 9], [0, 9]], [[0, 20], [90, 20], [90, 29], [0, 29]],
                              [[0, 40], [90, 40], [90, 49], [0, 49]]]}]


def registry_with_models(tmp_path, status_ok=True):
    reg = Registry(tmp_path / "models")
    infos = []
    for id_, task, lang in [("det", "text_detection", "any"), ("rec_en", "text_recognition", "en"),
                            ("rec_ar", "text_recognition", "ar")]:
        if status_ok:
            (reg.base / id_).mkdir(parents=True, exist_ok=True)
            (reg.base / id_ / "w.bin").write_bytes(id_.encode())
        infos.append(ModelInfo(id=id_, task=task, lang=lang, version="t", path=id_, active=True))
    reg.save(infos)
    return reg


def test_paddle_engine_returns_lines_and_mrz(tmp_path, monkeypatch):
    fake = types.ModuleType("paddleocr")
    fake.PaddleOCR = FakePaddleOCR
    monkeypatch.setitem(sys.modules, "paddleocr", fake)
    engine = PaddleOcrEngine(registry=registry_with_models(tmp_path), langs=["en", "ar"])
    out = engine.extract([png_bytes()], "passport")
    assert out.mrz_lines == MRZ
    assert len(out.fields["textLines"]) == 3      # en and ar both read each box; only one reading is kept
    digits = [l for l in out.fields["textLines"] if l["text"] == "١٢٣"][0]
    assert digits["normalizedText"] == "123" and digits["script"] == "arabic"
    assert out.fields["models"]["ar"]["id"] == "rec_ar"
    assert FakePaddleOCR.kwargs["text_recognition_model_dir"].endswith("rec_ar")


def test_paddle_engine_refuses_missing_or_unregistered_models(tmp_path, monkeypatch):
    fake = types.ModuleType("paddleocr")
    fake.PaddleOCR = FakePaddleOCR
    monkeypatch.setitem(sys.modules, "paddleocr", fake)
    with pytest.raises(EngineUnavailable, match="missing"):
        PaddleOcrEngine(registry=registry_with_models(tmp_path, status_ok=False), langs=["en"]).extract([png_bytes()], None)
    with pytest.raises(EngineUnavailable, match="No active"):
        PaddleOcrEngine(registry=Registry(tmp_path / "empty"), langs=["en"]).extract([png_bytes()], None)


def test_paddle_engine_reports_when_paddleocr_not_installed(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "paddleocr", None)  # makes "import paddleocr" fail
    with pytest.raises(EngineUnavailable, match="not installed"):
        PaddleOcrEngine(registry=registry_with_models(tmp_path), langs=["en"]).extract([png_bytes()], None)


# --------------------------------------------------------------------------- corrections / training store
class FixedEngine:
    name = "fixed"

    def extract(self, images, hint):
        from app.engines.base import EngineOutput
        return EngineOutput(engine=self.name, mrz_lines=MRZ)


def upload(client, consent):
    r = client.post("/v1/documents/extract", files=[("files", ("p.png", png_bytes(), "image/png"))],
                    data={"consent_training": str(consent).lower()})
    assert r.status_code == 202
    return r.json()["jobId"]


CORRECTION = {"reviewer": "reviewer-1", "document": {"surname": "ERIKSSON"},
              "lines": [{"image": 0, "box": [10, 10, 200, 60], "text": "محمد", "lang": "ar"}]}


def test_corrections_need_store_and_consent(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "get_engine", lambda: FixedEngine())
    client = TestClient(main.app)

    monkeypatch.setattr(main, "training", TrainingStore(None))
    job = upload(client, True)
    r = client.post(f"/v1/jobs/{job}/corrections", json=CORRECTION)
    assert r.status_code == 409 and r.json()["detail"]["code"] == "training_store_disabled"

    monkeypatch.setattr(main, "training", TrainingStore(tmp_path / "store"))
    no_consent = upload(client, False)
    r = client.post(f"/v1/jobs/{no_consent}/corrections", json=CORRECTION)
    assert r.status_code == 409 and r.json()["detail"]["code"] == "no_training_consent"
    assert not (tmp_path / "store" / "jobs" / no_consent).exists()

    consented = upload(client, True)
    r = client.post(f"/v1/jobs/{consented}/corrections", json=CORRECTION)
    assert r.status_code == 201 and r.json() == {"stored": True, "lines": 1}
    saved = json.loads((tmp_path / "store" / "jobs" / consented / "correction.json").read_text(encoding="utf-8"))
    assert saved["lines"][0]["text"] == "محمد"
    assert (tmp_path / "store" / "jobs" / consented / "images" / "0.png").exists()

    assert client.post("/v1/jobs/unknown/corrections", json=CORRECTION).status_code == 404
    bad = dict(CORRECTION, lines=[{"image": 0, "box": [1, 2], "text": "x", "lang": "fr"}])
    assert client.post(f"/v1/jobs/{consented}/corrections", json=bad).status_code == 422


def test_training_store_purge(tmp_path):
    import os
    import time
    store = TrainingStore(tmp_path)
    store.save_correction("abc123", {"reviewer": "r"})
    old = tmp_path / "jobs" / "abc123"
    os.utime(old, (time.time() - 40 * 86400,) * 2)
    store.save_correction("def456", {"reviewer": "r"})
    assert store.purge_older_than(30) == 1
    assert not old.exists() and (tmp_path / "jobs" / "def456").exists()


# --------------------------------------------------------------------------- dataset pipeline
def test_synth_prepare_and_split_pipeline(tmp_path):
    pytest.importorskip("arabic_reshaper")
    pytest.importorskip("bidi")
    font = next((f for f in FONT_CANDIDATES if Path(f).exists()), None)
    if font is None:
        pytest.skip("no font with Arabic glyphs available")

    def run(script, *args):
        return subprocess.run([sys.executable, str(ROOT / "training" / script), *map(str, args)],
                              check=True, capture_output=True, text=True)

    run("synth_lines.py", "--out", tmp_path / "synth", "--count", 40, "--fonts", font, "--seed", 3)
    assert (tmp_path / "synth" / "train.txt").read_text(encoding="utf-8").strip()

    from PIL import Image
    store = tmp_path / "store"
    for n in range(3):
        job = store / "jobs" / f"job{n}"
        (job / "images").mkdir(parents=True)
        Image.new("RGB", (300, 100), "white").save(job / "images" / "0.png")
        (job / "correction.json").write_text(json.dumps({"lines": [
            {"image": 0, "box": [5, 5, 250, 60], "text": f"محمد {n}", "lang": "ar"},
            {"image": 0, "box": [5, 5, 3, 6], "text": "too small", "lang": "en"}]}), encoding="utf-8")

    dictionary = tmp_path / "dict.txt"
    dictionary.write_text("\n".join("محد0123456789"), encoding="utf-8")
    out = run("prepare_dataset.py", "--store-dir", store, "--synth-dir", tmp_path / "synth",
              "--dict", dictionary, "--out", tmp_path / "rec", "--val-ratio", 0.34)
    stats = json.loads((tmp_path / "rec" / "stats.json").read_text(encoding="utf-8"))
    assert stats["realTrain"] + stats["realVal"] == 3          # tiny crop skipped
    assert stats["realVal"] >= 1                                # real validation exists
    assert stats["synthTrain"] <= stats["realTrain"] * 3        # synthetic is capped
    train = (tmp_path / "rec" / "train.txt").read_text(encoding="utf-8").splitlines()
    val_real = (tmp_path / "rec" / "val_real.txt").read_text(encoding="utf-8").splitlines()
    assert val_real and not set(train) & set(val_real)
    assert all((tmp_path / "rec" / l.split("\t")[0]).exists() for l in train + val_real)
    assert "outOfDictionary" in stats
    assert "WARNING" in out.stdout or stats["outOfDictionary"] is not None


def test_activation_refused_when_current_model_has_no_metrics(tmp_path):
    import json
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent / "training"))
    from register_model import register
    from app.model_registry import ModelInfo, Registry
    (tmp_path / "old").mkdir(); (tmp_path / "old" / "w.bin").write_bytes(b"1")
    (tmp_path / "new").mkdir(); (tmp_path / "new" / "w.bin").write_bytes(b"2")
    (tmp_path / "manifest.json").write_text(json.dumps({"models": [
        {"id": "old", "task": "text_recognition", "lang": "ar", "version": "1", "path": "old", "arch": "a", "sha256": None, "active": True, "metrics": {}}]}))
    reg = Registry(tmp_path)
    new = ModelInfo(id="new", task="text_recognition", lang="ar", version="2", path="new", arch="a", metrics={"cer": 0.1})
    with pytest.raises(SystemExit, match="no recorded CER"):
        register(reg, new, activate=True, force=False)
    assert next(m for m in Registry(tmp_path).models() if m.id == "old").active           # still the live model
