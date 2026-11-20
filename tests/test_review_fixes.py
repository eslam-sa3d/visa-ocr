"""Tests added with the engineering and ML review: offline behaviour, the promotion gate, label approval, the model switch and
service resilience. Each test names the finding it covers."""
import json
import socket
import subprocess
import sys
import threading
import time
import types
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "training"))
sys.path.insert(0, str(ROOT / "tools"))
import auto_train  # noqa: E402
import prepare_dataset  # noqa: E402
import register_model  # noqa: E402
from app import main  # noqa: E402
from app.engines.base import EngineOutput, EngineUnavailable  # noqa: E402
from app.engines.paddle_ocr import PaddleOcrEngine  # noqa: E402
from app.model_registry import ManifestError, ModelInfo, Registry, dir_sha256  # noqa: E402
from app.training_store import TrainingStore, approved_correction  # noqa: E402
from tests.test_auto_train import _args, _current_model, _runner, _store  # noqa: E402
from tests.test_model_and_training import MRZ, FakePaddleOCR, png_bytes, registry_with_models  # noqa: E402


# ---------------------------------------------------------------------------------------------------- offline (O-01, O-03)
@pytest.fixture
def no_network(monkeypatch):
    """Any attempt to open a connection or look a name up fails the test."""
    def refuse(*a, **k):
        raise AssertionError(f"network use attempted: {a[:2]}")
    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket.socket, "connect_ex", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)


def test_a_document_is_read_with_the_network_blocked(no_network, monkeypatch):
    """O-03: loading the engines and reading a passport opens no socket."""
    import random
    import io
    import accuracy_matrix as A
    import make_scenarios as M
    from app import pipeline
    from app.engines import get_engine_by_name
    from app.review import Applicant
    try:
        engine = get_engine_by_name("ensemble")
        engine.rapid._load()
    except EngineUnavailable:
        pytest.skip("RapidOCR not installed")
    lines, truth = A.build("TD3", **A.rnd_case(random.Random(7), "TD3"))
    buf = io.BytesIO()
    M.draw_document({"lines": lines, "label": "x"}).convert("RGB").save(buf, "PNG")
    status, result = pipeline.run([buf.getvalue()], None, Applicant(), engine)
    assert status == "done" and result["document"]["documentNumber"] == truth["documentNumber"]


def test_training_tools_open_no_socket(no_network, tmp_path):
    """O-03: the retraining cycle (with the heavy steps replaced) and model registration stay offline."""
    _current_model(tmp_path)
    out = auto_train.cycle(_args(tmp_path, _store(tmp_path, 50)), runner=_runner(0.02, 0.05))
    assert out["outcome"] == "candidate_better_waiting_for_approval"


def test_test_page_and_training_scripts_reference_nothing_online():
    """O-03: no page asset or script fetches from the internet (documentation links in comments are not fetches)."""
    import re
    html = (ROOT / "app" / "static" / "index.html").read_text(encoding="utf-8")
    assert not re.search(r"""(src|href)\s*=\s*["'](https?:)?//""", html)
    for name in ("predict_lines.py", "finetune.sh"):
        assert "PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK" in (ROOT / "training" / name).read_text(encoding="utf-8")
    assert TestClient(main.app).get("/docs").status_code == 404           # the CDN-backed API pages are off by default


def test_model_without_a_checksum_is_refused(tmp_path, monkeypatch):
    """O-01: 'unverified' no longer loads, and readiness says so; a development flag allows it."""
    fake = types.ModuleType("paddleocr")
    fake.PaddleOCR = FakePaddleOCR
    monkeypatch.setitem(sys.modules, "paddleocr", fake)
    reg = registry_with_models(tmp_path)
    models = reg.models()
    for m in models:
        m.sha256 = None
    reg.save(models)
    with pytest.raises(EngineUnavailable, match="unverified"):
        PaddleOcrEngine(registry=reg, langs=["en"]).extract([png_bytes()], None)
    monkeypatch.setenv("MODELS_DIR", str(reg.base))
    monkeypatch.setenv("OCR_ENGINE", "paddle-ocr")
    main._model_cache["at"] = 0.0
    r = TestClient(main.app).get("/readyz")
    assert r.status_code == 503 and "unverified" in r.json()["detail"]["message"]
    monkeypatch.setenv("ALLOW_UNVERIFIED_MODELS", "1")
    assert PaddleOcrEngine(registry=reg, langs=["en"]).extract([png_bytes()], None).mrz_lines == MRZ
    main._model_cache["at"] = 0.0


def test_checksum_file_catches_a_changed_or_extra_wheel(tmp_path):
    """O-02: SHA256SUMS for the wheelhouse (or any carried folder) detects a changed, missing or added file."""
    import make_checksums
    (tmp_path / "a.whl").write_bytes(b"one")
    (tmp_path / "b.whl").write_bytes(b"two")
    assert make_checksums.write(tmp_path) == 2 and make_checksums.check(tmp_path) == []
    (tmp_path / "a.whl").write_bytes(b"ONE")
    (tmp_path / "c.whl").write_bytes(b"new")
    (tmp_path / "b.whl").unlink()
    assert sorted(make_checksums.check(tmp_path)) == ["changed: a.whl", "missing: b.whl", "not listed: c.whl"]


def test_local_tesseract_model_needs_a_listed_checksum(tmp_path, monkeypatch):
    """O-04: an OCR-B model in MODELS_DIR/tessdata is used only when SHA256SUMS lists it and it matches."""
    import hashlib
    from app.engines.tesseract_mrz import TesseractMrzEngine
    data = tmp_path / "tessdata"
    data.mkdir()
    (data / "ocrb.traineddata").write_bytes(b"model")
    monkeypatch.setenv("MODELS_DIR", str(tmp_path))
    monkeypatch.delenv("TESSERACT_LANG", raising=False)
    assert TesseractMrzEngine().lang == "eng"                             # not listed: ignored
    (data / "SHA256SUMS").write_text(hashlib.sha256(b"model").hexdigest() + "  ocrb.traineddata\n")
    assert TesseractMrzEngine().lang == "ocrb"
    (data / "ocrb.traineddata").write_bytes(b"tampered")
    assert TesseractMrzEngine().lang == "eng"


# ------------------------------------------------------------------------------------------- promotion gate (T-02..T-04)
OLD = {"cer": 0.05, "exactMatch": 0.80, "chars": 20000, "samples": 500,
       "byScript": {"arabic": {"samples": 300, "cer": 0.06}, "latin": {"samples": 200, "cer": 0.03}},
       "byGroup": {"iqama": {"samples": 250, "cer": 0.05}, "passport": {"samples": 250, "cer": 0.05}}}


def better(**changes):
    new = json.loads(json.dumps(OLD))
    new.update(cer=0.03, exactMatch=0.85)
    new["byScript"]["arabic"]["cer"], new["byScript"]["latin"]["cer"] = 0.04, 0.02
    new["byGroup"]["iqama"]["cer"], new["byGroup"]["passport"]["cer"] = 0.03, 0.03
    for path, value in changes.items():
        node = new
        *parents, leaf = path.split(".")
        for key in parents:
            node = node[key]
        node[leaf] = value
    return new


def test_gate_passes_a_model_that_is_better_everywhere():
    assert auto_train.gate(better(), OLD, 0.002) == []


@pytest.mark.parametrize("old", [None, {"cer": None}])
def test_gate_refuses_when_the_model_in_use_cannot_be_scored(old):
    """T-04: 'no score to compare with' used to count as better."""
    assert auto_train.gate(better(), old, 0.002) == ["the model in use could not be scored on the golden set"]


def test_gate_refuses_a_regression_on_one_script_or_document_type():
    assert any("arabic" in r for r in auto_train.gate(better(**{"byScript.arabic.cer": 0.07}), OLD, 0.002))
    assert any("passport" in r for r in auto_train.gate(better(**{"byGroup.passport.cer": 0.06}), OLD, 0.002))
    assert any("exactly right" in r for r in auto_train.gate(better(exactMatch=0.79), OLD, 0.002))


def test_gate_needs_a_gain_above_min_gain_and_above_the_noise():
    """T-03: on few characters a small gain is noise, however small --min-gain is."""
    assert auto_train.gate(better(cer=0.0481), OLD, 0.002) != []                      # under the noise (about 0.0031)
    assert auto_train.gate(better(cer=0.045), OLD, 0.002) == []
    assert auto_train.gate(better(cer=0.045), OLD, 0.01) != []                        # under --min-gain
    few = dict(OLD, chars=400)
    assert auto_train.gate(better(cer=0.035, chars=400), few, 0.002) != []            # 400 characters: noise is about 0.022


def test_small_buckets_are_reported_but_do_not_block():
    old = json.loads(json.dumps(OLD))
    old["byScript"]["mixed"] = {"samples": 3, "cer": 0.0}
    assert auto_train.gate(better(**{"byScript.mixed": {"samples": 3, "cer": 0.3}}), old, 0.002) == []


def test_cycle_judges_on_the_golden_set_and_fails_closed_without_one(tmp_path):
    """T-02, T-03: the gate reads the golden labels, never the file that picked the checkpoint."""
    reg = _current_model(tmp_path)
    seen = []
    args = _args(tmp_path, _store(tmp_path, 50), activate=True)
    out = auto_train.cycle(args, runner=_runner(0.02, 0.05, seen=seen))
    assert out["outcome"] == "activated" and out["gate"]["passed"] and out["undo"]
    assert seen and all(Path(s) == args.golden_dir / "labels.txt" for s in seen)
    assert reg.active("text_recognition", "ar").replaced == "cur"

    other = tmp_path / "second"
    other.mkdir()
    reg2 = _current_model(other)
    args2 = _args(other, _store(other, 50), activate=True, golden_dir=None)
    out2 = auto_train.cycle(args2, runner=_runner(0.0, 0.5))
    assert out2["outcome"] == "candidate_built_no_golden_set" and reg2.active("text_recognition", "ar").id == "cur"
    assert any(m.id == out2["candidate"]["id"] and not m.active for m in reg2.models())


def test_cycle_refuses_when_one_script_got_worse(tmp_path):
    reg = _current_model(tmp_path)
    worse_arabic = {"byScript": {"arabic": {"samples": 300, "cer": 0.09}}}
    was = {"byScript": {"arabic": {"samples": 300, "cer": 0.06}}}
    out = auto_train.cycle(_args(tmp_path, _store(tmp_path, 50), activate=True),
                           runner=_runner(0.02, 0.05, new_extra=worse_arabic, old_extra=was))
    assert out["outcome"] == "candidate_not_better_kept_current" and reg.active("text_recognition", "ar").id == "cur"
    assert any("arabic" in r for r in out["gate"]["reasons"])


def test_missing_predictions_count_as_errors():
    """T-04: a model that returns nothing for half the lines used to look perfect on the other half."""
    import evaluate
    labels = {f"passport__{i}.png": "ABCDEFGHIJ" for i in range(10)}
    half = {k: v for i, (k, v) in enumerate(labels.items()) if i < 5}
    m = evaluate.evaluate(labels, half)
    assert m["missingPredictions"] == 5 and m["cer"] == 0.5 and m["exactMatch"] == 0.5
    assert m["byGroup"]["passport"]["samples"] == 10 and m["byScript"]["latin"]["cer"] == 0.5


def test_split_never_puts_one_document_on_both_sides():
    import random
    samples = [{"doc": f"d{i % 17}", "path": f"p{i}", "text": "x"} for i in range(200)]
    for seed in range(50):
        train, val = prepare_dataset.split_by_document(samples, 0.2, random.Random(seed))
        assert val and not {s["doc"] for s in train} & {s["doc"] for s in val}


# ------------------------------------------------------------------------------------- label approval (T-01, T-10, T-13)
def _job(store: Path, name: str, record: dict | str, size=(600, 200)) -> Path:
    job = store / "jobs" / name
    (job / "images").mkdir(parents=True)
    Image.new("RGB", size, "white").save(job / "images" / "0.png")
    (job / "corrections").mkdir()
    text = record if isinstance(record, str) else json.dumps(record)
    (job / "corrections" / f"{time.time_ns()}.json").write_text(text, encoding="utf-8")
    return job


def _line(text, lang="en", box=(5, 5, 400, 60)):
    return {"image": 0, "box": list(box), "text": text, "lang": lang}


def _build(store: Path, tmp_path: Path, **kw):
    out = tmp_path / f"ds{time.time_ns()}"
    (out / "images").mkdir(parents=True)
    from collections import Counter
    skipped = Counter()
    return prepare_dataset.from_store(store, out, skipped=skipped, **kw), skipped


def test_only_corrections_approved_by_a_second_person_become_training_data(tmp_path):
    store = tmp_path / "store"
    _job(store, "pending1", {"status": "pending", "reviewer": "amal", "lines": [_line("ONE")]})
    _job(store, "self1", {"status": "approved", "approvedBy": "Amal", "reviewer": "amal", "lines": [_line("TWO")]})
    _job(store, "claimed1", {"status": "approved", "approvedBy": None, "reviewer": "amal", "lines": [_line("THREE")]})
    _job(store, "good1", {"status": "approved", "approvedBy": "omar", "reviewer": "amal", "lines": [_line("FOUR")]})
    samples, skipped = _build(store, tmp_path)
    assert [s["text"] for s in samples] == ["FOUR"] and skipped["notApproved"] == 3


def test_api_cannot_approve_its_own_correction_and_approval_needs_another_person(tmp_path):
    """T-01: 'status' sent by a client is ignored; the store refuses the submitter as approver."""
    ts = TrainingStore(tmp_path / "store")
    ts.save_images("job1", [png_bytes()])
    ts.save_correction("job1", {"reviewer": "amal", "submittedBy": "key123", "lines": [_line("X")]})
    job = tmp_path / "store" / "jobs" / "job1"
    assert approved_correction(job) is None and len(ts.pending()) == 1
    for same in ("amal", "AMAL ", "key123"):
        with pytest.raises(PermissionError):
            ts.decide("job1", same)
    ts.decide("job1", "omar")
    assert approved_correction(job)["approvedBy"] == "omar" and ts.pending() == []


def test_a_second_correction_does_not_erase_the_first(tmp_path):
    ts = TrainingStore(tmp_path / "store")
    ts.save_images("job1", [png_bytes()])
    ts.save_correction("job1", {"reviewer": "a", "lines": [_line("FIRST")]})
    ts.save_correction("job1", {"reviewer": "b", "lines": [_line("SECOND")]})
    files = sorted((tmp_path / "store" / "jobs" / "job1" / "corrections").glob("*.json"))
    assert [json.loads(f.read_text())["lines"][0]["text"] for f in files] == ["FIRST", "SECOND"]


def test_mrz_label_that_fails_its_check_digits_is_never_used(tmp_path):
    store = tmp_path / "store"
    bad = [MRZ[0], MRZ[1][:9] + ("0" if MRZ[1][9] != "0" else "1") + MRZ[1][10:]]
    ok = {"status": "approved", "approvedBy": "omar", "reviewer": "amal"}
    _job(store, "bad1", {**ok, "lines": [_line(bad[0]), _line(bad[1], box=(5, 70, 400, 120))]})
    _job(store, "good1", {**ok, "lines": [_line(MRZ[0]), _line(MRZ[1], box=(5, 70, 400, 120))]})
    samples, skipped = _build(store, tmp_path)
    assert {s["doc"] for s in samples} == {"good1"} and skipped["mrzLabelFailsCheckDigits"] == 1


def test_malformed_corrections_are_skipped_not_fatal(tmp_path):
    """T-10: one bad file used to stop every later retraining run."""
    store = tmp_path / "store"
    ok = {"status": "approved", "approvedBy": "omar", "reviewer": "amal"}
    _job(store, "broken1", "{not json")
    _job(store, "odd1", {**ok, "lines": [{"image": "0", "box": [1, 2, 3, 4], "text": "x"}, {"text": "no box"}, "junk",
                                         {"image": 0, "box": [5, 5, 400], "text": "short box"}, _line("KEPT")]})
    _job(store, "nolines1", {**ok, "lines": None})
    samples, skipped = _build(store, tmp_path)
    assert [s["text"] for s in samples] == ["KEPT"]
    assert skipped["unreadable"] == 1 and skipped["malformedLines"] == 4


def test_language_filter_duplicates_and_golden_crops(tmp_path):
    store = tmp_path / "store"
    ok = {"status": "approved", "approvedBy": "omar", "reviewer": "amal"}
    job = _job(store, "j1", {**ok, "lines": [_line("محمد", "ar"), _line("MOHAMMED", "en"), _line("محمد", "ar")]})
    samples, skipped = _build(store, tmp_path, lang="ar")
    assert [s["text"] for s in samples] == ["محمد"] and skipped["otherLanguage"] == 1 and skipped["duplicate"] == 1
    golden = tmp_path / "golden"
    golden.mkdir()
    with Image.open(job / "images" / "0.png") as im:
        im.convert("RGB").crop((5, 5, 400, 60)).save(golden / "passport__1.png")
    samples, skipped = _build(store, tmp_path, lang="ar", exclude=prepare_dataset.golden_hashes(golden))
    assert samples == [] and skipped["inGoldenSet"] >= 1


def test_pending_corrections_do_not_trigger_training(tmp_path):
    d = tmp_path / "store" / "jobs" / "job1"
    d.mkdir(parents=True)
    (d / "correction.json").write_text(json.dumps({"lines": [{"text": "x"}] * 50}))       # written before approvals existed
    out = auto_train.cycle(_args(tmp_path, tmp_path / "store"), runner=_runner(0.01, 0.05))
    assert out["outcome"] == "skipped_not_enough_new_lines" and out["correctedLines"] == 0


def test_correction_records_what_the_reader_got_wrong_and_calibration_reads_it(tmp_path, monkeypatch):
    """T-13: confidence and reviewer outcome are stored together; the report turns them into a reliability table."""
    import calibration_report
    store = tmp_path / "store"
    for n, (conf, changed) in enumerate([(0.99, []), (0.99, ["surname"]), (0.6, ["surname"]), (None, [])]):
        ts = TrainingStore(store)
        ts.save_images(f"job{n}", [png_bytes()])
        ts.save_correction(f"job{n}", {"reviewer": "a", "lines": [], "ocr": {"confidence": conf, "needsReview": conf is None,
                                                                              "fieldsChanged": changed, "fieldsChecked": ["surname"]}})
    out = calibration_report.report(store, 0.97)
    assert out["byConfidence"]["0.95-1.00"] == {"documents": 2, "wrong": 1, "errorRate": 0.5}
    assert out["byConfidence"]["0.50-0.80"]["wrong"] == 1 and out["byConfidence"]["no confidence"]["documents"] == 1
    assert out["acceptedWithoutReview"] == {"documents": 2, "wrong": 1, "errorRate": 0.5}
    assert [r["ocr"]["confidence"] for _, r in TrainingStore(store).pending()][:2] == [0.6, 0.99]     # least confident first


# ------------------------------------------------------------------------------------------ model switch (T-05..T-07, S-01)
def _fake_paddle(monkeypatch, cls=FakePaddleOCR):
    fake = types.ModuleType("paddleocr")
    fake.PaddleOCR = cls
    monkeypatch.setitem(sys.modules, "paddleocr", fake)


def _add_model(reg: Registry, model_id: str, activate=True, content=b"new", sha=True):
    (reg.base / model_id).mkdir()
    (reg.base / model_id / "w.bin").write_bytes(content)
    with reg.locked():
        models = reg.models()
        if activate:
            for m in models:
                if m.task == "text_recognition" and m.lang == "ar":
                    m.active = False
        reg.save(models + [ModelInfo(id=model_id, task="text_recognition", lang="ar", version="t2", path=model_id, active=activate,
                                     sha256=dir_sha256(reg.base / model_id) if sha else "0" * 64)])


def test_half_written_or_broken_manifest_keeps_the_models_in_use(tmp_path, monkeypatch):
    """T-05: a manifest that cannot be read is an error to log, not a reason to stop reading documents."""
    _fake_paddle(monkeypatch)
    reg = registry_with_models(tmp_path)
    engine = PaddleOcrEngine(registry=reg, langs=["en", "ar"])
    assert engine.extract([png_bytes()], "passport").fields["models"]["ar"]["id"] == "rec_ar"
    whole = reg.manifest_path.read_text()
    reg.manifest_path.write_text(whole[:len(whole) // 2])
    with pytest.raises(ManifestError):
        reg.models()
    out = engine.extract([png_bytes()], "passport")
    assert out.mrz_lines == MRZ and out.fields["models"]["ar"]["id"] == "rec_ar"
    assert not list(reg.base.glob("*.tmp"))


def test_a_bad_new_model_is_refused_and_the_old_one_stays(tmp_path, monkeypatch):
    """T-07: wrong checksum on the newly activated model: the previous readers keep serving."""
    _fake_paddle(monkeypatch)
    reg = registry_with_models(tmp_path)
    engine = PaddleOcrEngine(registry=reg, langs=["en", "ar"])
    engine.extract([png_bytes()], "passport")
    _add_model(reg, "rec_bad", sha=False)
    out = engine.extract([png_bytes()], "passport")
    assert out.fields["models"]["ar"]["id"] == "rec_ar"
    _add_model(reg, "rec_good")                      # the next change is picked up normally
    assert engine.extract([png_bytes()], "passport").fields["models"]["ar"]["id"] == "rec_good"


def test_a_model_that_loads_but_cannot_read_is_refused(tmp_path, monkeypatch):
    """T-07: the new readers are tried on a blank image before they take over."""
    class Picky(FakePaddleOCR):
        def predict(self, array):
            if "rec_broken" in str(self.kwargs.get("text_recognition_model_dir")) or getattr(self, "broken", False):
                raise RuntimeError("weights do not match the architecture")
            return super().predict(array)

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.broken = "rec_broken" in str(kwargs.get("text_recognition_model_dir"))
    _fake_paddle(monkeypatch, Picky)
    reg = registry_with_models(tmp_path)
    engine = PaddleOcrEngine(registry=reg, langs=["en", "ar"])
    engine.extract([png_bytes()], "passport")
    _add_model(reg, "rec_broken")
    assert engine.extract([png_bytes()], "passport").fields["models"]["ar"]["id"] == "rec_ar"


def test_reads_during_repeated_switches_never_fail(tmp_path, monkeypatch):
    """T-05, T-07: eight readers while a model is activated again and again: no failed read, and every result names a
    model that existed."""
    _fake_paddle(monkeypatch)
    reg = registry_with_models(tmp_path)
    engine = PaddleOcrEngine(registry=reg, langs=["en", "ar"])
    engine.extract([png_bytes()], "passport")
    stop, errors, seen = threading.Event(), [], set()

    def read():
        while not stop.is_set():
            try:
                seen.add(engine.extract([png_bytes()], "passport").fields["models"]["ar"]["id"])
            except Exception as exc:  # noqa: BLE001
                errors.append(repr(exc))

    threads = [threading.Thread(target=read) for _ in range(8)]
    [t.start() for t in threads]
    ids = [f"rec_s{i}" for i in range(25)]
    for model_id in ids:
        _add_model(reg, model_id)
        time.sleep(0.005)
    time.sleep(0.05)
    stop.set()
    [t.join() for t in threads]
    assert errors == [] and seen <= {"rec_ar", *ids} and len(seen) > 1
    assert engine.extract([png_bytes()], "passport").fields["models"]["ar"]["id"] == ids[-1]


def test_two_writers_do_not_lose_an_entry(tmp_path):
    """T-05: the retraining timer and an operator registering at the same moment."""
    base = tmp_path / "models"
    for i in range(6):
        (base / f"m{i}").mkdir(parents=True)
        (base / f"m{i}" / "w").write_text(str(i))
    procs = [subprocess.Popen([sys.executable, str(ROOT / "training" / "register_model.py"), "--id", f"m{i}", "--task",
                               "text_recognition", "--lang", "ar", "--version", "v", "--path", f"m{i}", "--models-dir", str(base)])
             for i in range(6)]
    assert [pr.wait() for pr in procs] == [0] * 6
    assert sorted(m.id for m in Registry(base).models()) == [f"m{i}" for i in range(6)]


def test_rollback_and_activate_existing(tmp_path):
    """T-06: one command back to the model before; a registered candidate can be made live; the same id is not taken twice."""
    base = tmp_path / "models"
    reg = Registry(base)
    for name in ("v1", "v2", "v3"):
        (base / name).mkdir(parents=True)
        (base / name / "w").write_text(name)
    mk = lambda name, cer: ModelInfo(id=name, task="text_recognition", lang="ar", version=name, path=name, metrics={"cer": cer})  # noqa: E731
    register_model.register(reg, mk("v1", 0.05), activate=True, force=False)
    register_model.register(reg, mk("v2", 0.03), activate=True, force=False)
    register_model.register(reg, mk("v3", 0.04), activate=False, force=False)
    assert reg.active("text_recognition", "ar").id == "v2" and reg.active("text_recognition", "ar").replaced == "v1"
    assert register_model.rollback(reg, "text_recognition", "ar") == ("v1", "v2")
    assert reg.active("text_recognition", "ar").id == "v1"
    assert register_model.activate_existing(reg, "v3") == ("v3", "v1")
    (base / "v2" / "w").write_text("tampered")
    with pytest.raises(SystemExit, match="mismatch"):
        register_model.activate_existing(reg, "v2")
    with pytest.raises(SystemExit, match="already registered"):
        register_model.register(reg, mk("v1", 0.01), activate=False, force=False)
    with pytest.raises(SystemExit, match="in use"):
        register_model.register(reg, mk("v3", 0.01), activate=False, force=False, replace=True)


def test_signed_manifest_and_paths_outside_the_models_folder(tmp_path, monkeypatch):
    """S-01: with a key set, a manifest edited without the key is refused; a model path may not leave the folder."""
    reg = registry_with_models(tmp_path)
    monkeypatch.setenv("MANIFEST_HMAC_KEY", "k" * 40)
    with pytest.raises(ManifestError, match="signature"):
        reg.models()                                           # written before the key existed: not signed
    monkeypatch.delenv("MANIFEST_HMAC_KEY")
    models = reg.models()
    monkeypatch.setenv("MANIFEST_HMAC_KEY", "k" * 40)
    reg.save(models)
    assert len(reg.models()) == 3
    data = json.loads(reg.manifest_path.read_text())
    data["models"][0]["sha256"] = "0" * 64                     # someone with write access but no key
    reg.manifest_path.write_text(json.dumps(data))
    with pytest.raises(ManifestError, match="signature"):
        reg.models()
    monkeypatch.delenv("MANIFEST_HMAC_KEY")
    outside = ModelInfo(id="x", task="text_recognition", lang="ar", version="v", path="../outside")
    with pytest.raises(ManifestError):
        reg.full_path(outside)
    assert reg.status(outside) == "mismatch"


# --------------------------------------------------------------------------------- automation safety (T-08, T-09, T-11, T-14)
def test_training_that_keeps_failing_on_the_same_data_is_not_repeated_forever(tmp_path):
    from datetime import datetime, timedelta, timezone
    _current_model(tmp_path)
    args = _args(tmp_path, _store(tmp_path, 50))
    t0 = datetime(2026, 11, 1, 2, 0, tzinfo=timezone.utc)
    outcomes = [auto_train.cycle(args, runner=_runner(0.02, 0.05, fail_training=True), now=t0 + timedelta(days=7 * i))["outcome"]
                for i in range(4)]
    assert outcomes == ["training_failed", "training_failed", "skipped_training_failed_before_on_this_data",
                        "skipped_training_failed_before_on_this_data"]


def test_a_lock_left_by_a_dead_run_is_cleared(tmp_path):
    lock = tmp_path / "auto_train.lock"
    lock.write_text("999999999")                                 # no such process
    auto_train.take_lock(lock)
    assert lock.read_text() == str(__import__("os").getpid())
    with pytest.raises(SystemExit, match="Another run"):
        auto_train.take_lock(lock)                               # this process is alive and holds it


def test_run_report_is_kept_and_old_candidates_and_checkpoints_are_pruned(tmp_path):
    from datetime import datetime, timedelta, timezone
    reg = _current_model(tmp_path)
    store = _store(tmp_path, 50)
    t0 = datetime(2026, 11, 1, 2, 0, tzinfo=timezone.utc)
    reports = []
    for i in range(4):
        (store / "jobs" / "job1" / "correction.json").write_text(json.dumps(
            {"status": "approved", "approvedBy": "b", "reviewer": "a", "lines": [{"text": "x"}] * (50 * (i + 1))}))
        reports.append(auto_train.cycle(_args(tmp_path, store, keep_candidates=2), runner=_runner(0.06, 0.05),
                                        now=t0 + timedelta(days=7 * i)))
    work = tmp_path / "work"
    history = [json.loads(ln) for ln in (work / "history.jsonl").read_text().splitlines()]
    assert len(history) == 4 and all(h["dataset"]["datasetSha256"] and h["training"]["seed"] == 1 for h in history)
    assert json.loads((work / reports[-1]["run"] / "report.json").read_text())["outcome"] == "candidate_not_better_kept_current"
    auto = sorted(m.id for m in reg.models() if "_rec_auto_" in m.id)
    assert auto == [r["candidate"]["id"] for r in reports[-2:]] and reports[-1]["prunedCandidates"]
    assert sorted(p.name for p in reg.base.iterdir() if p.is_dir()) == sorted(["cur", *auto])
    assert not list(work.rglob("out/train"))


def test_finetune_refuses_to_train_from_scratch_or_from_unverified_weights(tmp_path):
    env = {"PATH": __import__("os").environ["PATH"], "PADDLEOCR_DIR": str(tmp_path), "BASE_CONFIG": "c.yml",
           "DATA_DIR": str(tmp_path), "OUT_DIR": str(tmp_path / "out")}
    sh = lambda **extra: subprocess.run(["bash", str(ROOT / "training" / "finetune.sh")], env={**env, **extra},  # noqa: E731
                                        capture_output=True, text=True)
    r = sh()
    assert r.returncode == 1 and "PRETRAINED is not set" in r.stderr
    (tmp_path / "w.pdparams").write_bytes(b"weights")
    r = sh(PRETRAINED=str(tmp_path / "w"))
    assert r.returncode == 1 and "PRETRAINED_SHA256" in r.stderr
    r = sh(PRETRAINED=str(tmp_path / "w"), PRETRAINED_SHA256="0" * 64)
    assert r.returncode == 1 and "does not match" in r.stderr


# ------------------------------------------------------------------------------------- service resilience (L-01..L-06, P-01)
MRZ_OUT = EngineOutput(engine="rapid-ocr", mrz_lines=list(MRZ), mrz_confidence=0.98, fields={"textLines": []})


def _ensemble(monkeypatch, tesseract, rapid):
    from app.engines import ensemble
    eng = ensemble.EnsembleEngine()
    monkeypatch.setattr(eng.tesseract, "extract", tesseract)
    monkeypatch.setattr(eng.rapid, "extract", rapid)
    monkeypatch.setattr(eng.rapid, "page_text", lambda images, seconds: [])
    return eng


def _raise(exc):
    def f(images, hint):
        raise exc
    return f


def test_ensemble_survives_a_tesseract_failure(monkeypatch):
    """L-01: a Tesseract timeout or a missing binary used to fail the job although RapidOCR had read the MRZ."""
    for failure in (RuntimeError("Tesseract process timeout"), EngineUnavailable("Tesseract binary not found")):
        out = _ensemble(monkeypatch, _raise(failure), lambda i, h: MRZ_OUT).extract([png_bytes()], None)
        assert out.mrz_lines == MRZ and out.engine == "ensemble" and out.fields["readers"]["failed"] == ["tesseract"]
        assert "agreedFields" not in out.fields                 # one reader: nothing is "agreed"


def test_ensemble_survives_a_rapidocr_failure_and_reports_when_both_fail(monkeypatch):
    t_out = EngineOutput(engine="tesseract-mrz", mrz_lines=list(MRZ))
    out = _ensemble(monkeypatch, lambda i, h: t_out, _raise(ValueError("decoder"))).extract([png_bytes()], None)
    assert out.mrz_lines == MRZ
    with pytest.raises(EngineUnavailable):
        _ensemble(monkeypatch, _raise(EngineUnavailable("a")), _raise(EngineUnavailable("b"))).extract([png_bytes()], None)
    with pytest.raises(RuntimeError):
        _ensemble(monkeypatch, _raise(RuntimeError("x")), _raise(EngineUnavailable("b"))).extract([png_bytes()], None)


def test_stuck_job_is_failed_and_gives_its_place_back(tmp_path, monkeypatch):
    """L-02: a job that never finishes no longer holds one of the MAX_PENDING_JOBS places until restart."""
    from app.jobs import JobStore, SqliteJobStore
    monkeypatch.setenv("JOB_MAX_SECONDS", "30")
    for store in (JobStore(), SqliteJobStore(str(tmp_path / "j.db"))):
        job = store.create_if_room("c", "o", 1)
        store.update(job.id, status="processing")
        assert store.create_if_room("c", "o", 1) is None
        if isinstance(store, JobStore):
            store._jobs[job.id] = (time.time() - 31, store._jobs[job.id][1])
        else:
            with store._conn() as c:
                c.execute("UPDATE jobs SET created_ts = created_ts - 31")
        assert store.create_if_room("c", "o", 1) is not None
        got = store.get(job.id)
        assert got.status == "failed" and got.error["code"] == "timed_out"


def test_first_requests_build_the_readers_once(monkeypatch):
    """L-03, P-01: ten first requests at once build RAPIDOCR_SESSIONS readers, not ten, and time the machine once."""
    from app.engines import rapid_ocr
    built, timed = [], []

    class Slow:
        def __init__(self):
            time.sleep(0.05)
            built.append(1)

        def __call__(self, *a, **k):
            timed.append(1)
            return [], None
    fake = types.ModuleType("rapidocr_onnxruntime")
    fake.RapidOCR = Slow
    monkeypatch.setitem(sys.modules, "rapidocr_onnxruntime", fake)
    monkeypatch.setenv("RAPIDOCR_SESSIONS", "2")
    eng = rapid_ocr.RapidOcrEngine()
    threads = [threading.Thread(target=eng._load) for _ in range(10)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert len(built) == 2 and len(timed) == 1 and eng._sessions.qsize() == 2
    with eng._session() as a, eng._session() as b:            # two documents are read at the same time
        assert a is not b and eng._sessions.qsize() == 0
    assert eng._sessions.qsize() == 2


def test_jobs_in_flight_are_marked_when_the_service_stops(monkeypatch):
    """L-06: a caller polling a job that died with the process gets an answer."""
    job = main.store.create_if_room("c", "anonymous", 100)
    with main._inflight_lock:
        main._inflight.add(job.id)
    with TestClient(main.app):
        pass                                                   # start and stop the service
    with main._inflight_lock:
        main._inflight.discard(job.id)
    got = main.store.get(job.id)
    assert got.status == "failed" and got.error["code"] == "service_restarted"


def test_errors_are_logged_without_document_values(caplog):
    """S-04: the log names the error type and the place, never the message (it can repeat a date or a number)."""
    from app import pipeline
    from app.review import Applicant

    class Boom:
        name = "boom"

        def extract(self, images, hint):
            raise ValueError("Invalid isoformat string: '1985-13-45 SECRETNAME'")
    with caplog.at_level("ERROR", logger="visa-ocr"):
        status, payload = pipeline.run([png_bytes()], None, Applicant(), Boom())
    assert status == "failed" and payload == {"code": "internal_error", "message": "Processing failed"}
    text = caplog.text
    assert "ValueError" in text and "test_review_fixes.py" in text and "SECRETNAME" not in text and "1985" not in text


def test_model_status_is_not_rehashed_while_files_are_unchanged(tmp_path, monkeypatch):
    """S-05, P-03: asking for the model list again and again reads the model files once."""
    from app import model_registry
    reg = registry_with_models(tmp_path)
    calls = []
    real = model_registry.dir_sha256
    monkeypatch.setattr(model_registry, "dir_sha256", lambda p: calls.append(p) or real(p))
    m = reg.models()[0]
    model_registry._HASH_CACHE.clear()
    assert [reg.status(m, fresh=False) for _ in range(5)] == ["verified"] * 5 and len(calls) == 1
    time.sleep(0.01)
    (reg.full_path(m) / "w.bin").write_bytes(b"changed!")
    assert reg.status(m, fresh=False) == "mismatch" and len(calls) == 2
    assert reg.status(m) == "mismatch" and len(calls) == 3      # a load always hashes


def test_readiness_reminds_until_storage_encryption_is_confirmed(tmp_path, monkeypatch):
    """S-02: the service cannot see whether its volumes are encrypted, so it keeps saying so until the operator confirms."""
    monkeypatch.setenv("OCR_ENGINE", "ensemble")
    monkeypatch.setenv("JOB_DB_PATH", str(tmp_path / "jobs.db"))
    client = TestClient(main.app)
    assert any("not encrypted" in w for w in client.get("/readyz").json().get("warnings", []))
    monkeypatch.setenv("DATA_AT_REST_ENCRYPTED", "1")
    assert "warnings" not in client.get("/readyz").json()


def test_tessdata_path_with_a_space_is_quoted(monkeypatch):
    """S-06"""
    import shlex
    from app.engines.tesseract_mrz import TesseractMrzEngine
    monkeypatch.setenv("TESSERACT_LANG", "eng")
    monkeypatch.setenv("TESSERACT_DATA_DIR", "/opt/my models/tessdata")
    seen = {}

    class FakeTess:
        def image_to_string(self, img, lang, config, timeout):
            seen["config"] = config
            return ""
    eng = TesseractMrzEngine()
    from PIL import ImageOps
    monkeypatch.setattr(eng, "_deps", lambda: (FakeTess(), Image, ImageOps))
    monkeypatch.setenv("OCR_SEARCH_SECONDS", "0.3")
    eng.extract([png_bytes()], None)
    parts = shlex.split(seen["config"])
    assert parts[parts.index("--tessdata-dir") + 1] == "/opt/my models/tessdata"


# ------------------------------------------------------------------------------------------------- accuracy (A-02, A-05)
def test_printed_page_settles_a_look_alike_in_the_optional_data():
    """A-02: readers differ only at 6/G in the personal number and the page shows it: the page decides, nothing is unconfirmed."""
    from app.engines.ensemble import _agree_except_optional, _printed_choice
    from app.mrz import parse_mrz

    class P:
        def __init__(self, v):
            self.optional_data = v
    page = [{"text": "Personal No. G901317"}, {"text": "MAROC"}]
    assert _printed_choice(P("G901317"), P("6901317"), page) == 0
    assert _printed_choice(P("6901317"), P("G901317"), page) == 1
    assert _printed_choice(P("6901317"), P("G901317"), [{"text": "nothing useful"}]) is None
    assert _printed_choice(P("6901317"), P("G901317"), page + [{"text": "6901317"}]) is None      # both shown: no decision
    assert _printed_choice(P("G9"), P("69"), [{"text": "G9"}]) is None                             # too short to trust
    assert parse_mrz and callable(_agree_except_optional)


def test_hijri_expiry_is_converted_with_a_margin():
    """A-05: a Hijri expiry is no longer always 'not checked': clearly expired and clearly valid are decided."""
    from datetime import date
    from app.textutil import hijri_to_gregorian
    from app.visual_zone import hijri_expiry_flag
    assert abs((hijri_to_gregorian(1446, 1, 1) - date(2024, 7, 7)).days) <= 2          # 1 Muharram 1446
    assert abs((hijri_to_gregorian(1447, 9, 1) - date(2026, 2, 18)).days) <= 2         # 1 Ramadan 1447
    assert hijri_to_gregorian(1447, 13, 1) is None
    today = date(2026, 11, 18)
    for when, code, severity in (("1430-01-01", "document_expired", "error"), ("1455-01-01", "expiry_hijri_converted", "warning"),
                                 ("1448-06-08", "expiry_hijri_not_checked", "error")):
        doc = {"expiryDate": when}
        flag = hijri_expiry_flag(doc, today, auto_accept=True)
        assert (flag.code, flag.severity) == (code, severity), when
        assert doc["expiryDateGregorianApprox"]


def test_real_specimens_regression():
    """A-01: reads a labelled folder of real documents kept OUTSIDE the repository (set REAL_SAMPLES_DIR). A field the service
    accepted without review and got wrong fails the test; so does accuracy under REAL_SAMPLES_MIN_CORRECT (default 0.9)."""
    import os
    folder = os.environ.get("REAL_SAMPLES_DIR")
    if not folder:
        pytest.skip("REAL_SAMPLES_DIR is not set")
    import accuracy_real
    report = accuracy_real.run(Path(folder))
    assert report["wronglyAccepted"] == [], report["wronglyAccepted"]
    assert report["documentsCorrect"] / max(report["documents"], 1) >= float(os.environ.get("REAL_SAMPLES_MIN_CORRECT", "0.9"))


# --------------------------------------------------------------------------------------------- encryption at rest (S-02)
@pytest.fixture
def data_key(tmp_path, monkeypatch):
    from cryptography.fernet import Fernet
    key_file = tmp_path / "data.key"
    key_file.write_text(Fernet.generate_key().decode() + "\n")
    monkeypatch.setenv("DATA_KEY_FILE", str(key_file))
    return key_file


def test_training_store_and_job_database_are_encrypted_with_a_key(tmp_path, data_key, monkeypatch):
    """S-02: with DATA_KEY_FILE set, no image, correction or result is readable in the files; everything still works."""
    from app import crypto
    from app.jobs import SqliteJobStore
    ts = TrainingStore(tmp_path / "store")
    image = png_bytes()
    ts.save_images("job1", [image])
    ts.save_correction("job1", {"reviewer": "amal", "lines": [_line("SECRETSURNAME")]})
    ts.decide("job1", "omar")
    job = tmp_path / "store" / "jobs" / "job1"
    raw_files = [p.read_bytes() for p in job.rglob("*") if p.is_file()]
    assert all(b.startswith(crypto.PREFIX) for b in raw_files) and not any(b"SECRETSURNAME" in b or b"PNG" in b[:8] for b in raw_files)
    assert approved_correction(job)["lines"][0]["text"] == "SECRETSURNAME"
    samples, skipped = _build(tmp_path / "store", tmp_path)
    assert [s["text"] for s in samples] == ["SECRETSURNAME"] and not skipped["unreadableImage"]

    db = tmp_path / "jobs.db"
    store = SqliteJobStore(str(db))
    j = store.create("c", "o")
    store.update(j.id, status="done", result={"document": {"surname": "SECRETSURNAME"}})
    assert store.get(j.id).result["document"]["surname"] == "SECRETSURNAME"
    with store._conn() as c:
        c.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    assert b"SECRETSURNAME" not in db.read_bytes()

    r = monkeypatch.setenv("JOB_DB_PATH", str(db))
    monkeypatch.setenv("OCR_ENGINE", "ensemble")
    assert "warnings" not in TestClient(main.app).get("/readyz").json() and r is None      # the service encrypts: no reminder


def test_key_rotation_wrong_key_and_data_written_before_the_key(tmp_path, monkeypatch):
    from cryptography.fernet import Fernet
    from app import crypto
    monkeypatch.delenv("DATA_KEY_FILE", raising=False)
    assert crypto.seal(b"plain") == b"plain" and crypto.unseal(b"plain") == b"plain"      # no key: nothing changes
    key_file = tmp_path / "k"
    old, new = Fernet.generate_key().decode(), Fernet.generate_key().decode()
    key_file.write_text(old + "\n")
    monkeypatch.setenv("DATA_KEY_FILE", str(key_file))
    sealed = crypto.seal(b"passport image")
    assert sealed != b"passport image" and crypto.unseal(sealed) == b"passport image"
    assert crypto.unseal(b"written before the key") == b"written before the key"
    assert crypto.unseal_json(crypto.seal_json({"a": "ب"})) == {"a": "ب"} and "enc" in crypto.seal_json({"a": 1})
    time.sleep(0.02)
    key_file.write_text(new + "\n" + old + "\n")                                         # rotate: new on top, old kept
    import os
    os.utime(key_file, (time.time() + 5, time.time() + 5))
    assert crypto.unseal(sealed) == b"passport image" and crypto.unseal(crypto.seal(b"x")) == b"x"
    key_file.write_text(new + "\n")                                                       # old key removed too early
    os.utime(key_file, (time.time() + 10, time.time() + 10))
    with pytest.raises(crypto.DataKeyError):
        crypto.unseal(sealed)
    with pytest.raises(crypto.DataKeyError):
        crypto.unseal(sealed[:-4] + b"AAAA")                                              # changed data is refused, not read
    key_file.write_text("not a key\n")
    os.utime(key_file, (time.time() + 15, time.time() + 15))
    with pytest.raises(crypto.DataKeyError):
        crypto.enabled()


# ------------------------------------------------------------------------------------ shared text detection (P-02)
def _shared_paddle(monkeypatch, calls, det_fail=False, rec_short=False):
    """Stand-ins with the result shapes of PaddleOCR 3.7 (TextDetection: dt_polys; TextRecognition: rec_text, rec_score) and
    of the PaddleX helpers (SortQuadBoxes returns the boxes top to bottom, CropByPolys returns one image per box)."""
    import numpy as np

    class Det:
        def __init__(self, **kw):
            calls.append(("det_load", kw["model_name"]))

        def predict(self, array):
            calls.append(("detect", array.shape))
            if det_fail and array.shape[0] > 64:
                raise RuntimeError("detector broke")
            if array.shape[0] <= 64:
                return [{"dt_polys": []}]
            return [{"dt_polys": [np.array([[0, 20], [90, 20], [90, 29], [0, 29]]), np.array([[0, 0], [90, 0], [90, 9], [0, 9]]),
                                  np.array([[0, 40], [90, 40], [90, 49], [0, 49]])]}]

    class Rec:
        def __init__(self, **kw):
            self.name = kw["model_name"]
            calls.append(("rec_load", self.name))

        def predict(self, crops):
            calls.append(("recognise", self.name, len(crops)))
            texts = [MRZ[0], MRZ[1], "١٢٣" if "ar" in self.name else "123"]
            out = [{"rec_text": t, "rec_score": 0.99 if "en" in self.name else 0.9} for t in texts]
            return out[:-1] if rec_short else out

    fake = types.ModuleType("paddleocr")
    fake.TextDetection, fake.TextRecognition, fake.PaddleOCR = Det, Rec, FakePaddleOCR
    monkeypatch.setitem(sys.modules, "paddleocr", fake)
    comp = types.ModuleType("paddlex.inference.pipelines.components")
    comp.SortQuadBoxes = lambda: (lambda polys: sorted(polys, key=lambda q: q[0][1]))
    comp.CropByPolys = lambda det_box_type="quad": (lambda img, polys: [img[int(q[0][1]):int(q[2][1]) or 1, :] for q in polys])
    for name in ("paddlex", "paddlex.inference", "paddlex.inference.pipelines"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "paddlex.inference.pipelines.components", comp)
    monkeypatch.setenv("PADDLE_SHARED_DETECTION", "1")


def test_shared_detection_finds_text_once_and_reads_it_per_language(tmp_path, monkeypatch):
    """P-02: with two languages the page is searched for text once, not twice, and the result is the same."""
    calls = []
    _shared_paddle(monkeypatch, calls)
    reg = registry_with_models(tmp_path)
    out = PaddleOcrEngine(registry=reg, langs=["en", "ar"]).extract([png_bytes()], "passport")
    assert out.mrz_lines == MRZ and out.mrz_confidence is not None
    assert out.fields["models"] == {"en": {"id": "rec_en", "version": "t", "detector": "det"},
                                    "ar": {"id": "rec_ar", "version": "t", "detector": "det"}}
    page_detects = [c for c in calls if c[0] == "detect" and c[1][0] > 64]
    assert len(page_detects) == 1 and [c[1] for c in calls if c[0] == "recognise" and c[2] == 3] == ["rec_en", "rec_ar"]
    assert len(out.fields["textLines"]) == 3                               # one reading kept per box
    assert {ln["text"] for ln in out.fields["textLines"]} >= set(MRZ)

    monkeypatch.setenv("PADDLE_SHARED_DETECTION", "0")                     # switched off: the usual pipeline, same answer
    usual = PaddleOcrEngine(registry=reg, langs=["en", "ar"]).extract([png_bytes()], "passport")
    assert usual.mrz_lines == out.mrz_lines and usual.fields["models"] == out.fields["models"]


@pytest.mark.parametrize("fault", ["det_fail", "rec_short"])
def test_shared_detection_falls_back_to_the_usual_path_on_any_failure(tmp_path, monkeypatch, fault, caplog):
    calls = []
    _shared_paddle(monkeypatch, calls, **{fault: True})
    engine = PaddleOcrEngine(registry=registry_with_models(tmp_path), langs=["en", "ar"])
    with caplog.at_level("ERROR", logger="visa-ocr"):
        out = engine.extract([png_bytes()], "passport")
    assert out.mrz_lines == MRZ and engine._shared_off and "shared text detection failed" in caplog.text
    n = len(calls)
    assert engine.extract([png_bytes()], "passport").mrz_lines == MRZ and len(calls) == n      # not tried again


def test_shared_detection_refuses_unverified_models_and_follows_a_model_switch(tmp_path, monkeypatch):
    calls = []
    _shared_paddle(monkeypatch, calls)
    reg = registry_with_models(tmp_path)
    engine = PaddleOcrEngine(registry=reg, langs=["en", "ar"])
    assert engine.extract([png_bytes()], "passport").fields["models"]["ar"]["id"] == "rec_ar"
    _add_model(reg, "rec_bad", sha=False)                                  # wrong checksum: the reader in use stays
    assert engine.extract([png_bytes()], "passport").fields["models"]["ar"]["id"] == "rec_ar" and not engine._shared_off
    _add_model(reg, "rec_new")
    assert engine.extract([png_bytes()], "passport").fields["models"]["ar"]["id"] == "rec_new"


def test_a_file_manager_looking_at_a_model_folder_does_not_break_its_checksum(tmp_path):
    reg = registry_with_models(tmp_path)
    m = reg.models()[0]
    (reg.full_path(m) / ".DS_Store").write_bytes(b"finder was here")
    assert reg.status(m) == "verified"
    (reg.full_path(m) / "extra.bin").write_bytes(b"a real change")
    assert reg.status(m) == "mismatch"


def test_auto_engine_reads_with_the_other_pair_when_paddleocr_breaks(monkeypatch, caplog):
    """Found by the first container run: PaddleOCR was installed with its models but could not start, and the job failed
    although Tesseract and RapidOCR were ready."""
    from app import engines

    class Broken:
        name = "paddle-ocr"

        def extract(self, images, hint):
            raise RuntimeError("PDX has already been initialized")

    class Pair:
        name = "ensemble"

        def extract(self, images, hint):
            return EngineOutput(engine="ensemble", mrz_lines=list(MRZ))
    monkeypatch.setitem(engines._INSTANCES, "paddle-ocr", Broken())
    monkeypatch.setitem(engines._INSTANCES, "ensemble", Pair())
    with caplog.at_level("ERROR", logger="visa-ocr"):
        out = engines.AutoEngine().extract([png_bytes()], None)
    assert out.mrz_lines == MRZ and out.engine == "ensemble" and "PaddleOCR failed (RuntimeError)" in caplog.text


def test_paddlex_cache_goes_to_a_temporary_folder_when_there_is_no_home(monkeypatch, tmp_path):
    from app.engines import paddle_ocr
    monkeypatch.delenv("PADDLE_PDX_CACHE_HOME", raising=False)
    blocked = tmp_path / "file-not-folder"
    blocked.write_text("x")
    monkeypatch.setenv("HOME", str(blocked / "home"))             # cannot be created
    paddle_ocr._writable_cache_home()
    import os
    assert os.path.isdir(os.environ["PADDLE_PDX_CACHE_HOME"]) and os.access(os.environ["PADDLE_PDX_CACHE_HOME"], os.W_OK)
    monkeypatch.setenv("PADDLE_PDX_CACHE_HOME", "/chosen/by/operator")
    paddle_ocr._writable_cache_home()
    assert os.environ["PADDLE_PDX_CACHE_HOME"] == "/chosen/by/operator"


# --- card type is decided by what the card says, not by the length of a number ---
def _vz_lines(*texts):
    return [{"text": t, "score": 0.95, "box": [[10, 10 + 30 * i], [300, 10 + 30 * i], [300, 30 + 30 * i], [10, 30 + 30 * i]]}
            for i, t in enumerate(texts)]


def test_foreign_permit_number_is_not_taken_as_a_gulf_id():
    from app.visual_zone import parse_gcc_id
    doc = parse_gcc_id(_vz_lines("RESIDENCE PERMIT", "TITRE DE SEJOUR", "X", "810512345", "Name", "DOE JANE"))["document"]
    assert "idNumber" not in doc and doc["kind"] == "gcc_id"


def test_unlabelled_ten_digit_number_needs_a_saudi_card():
    from app.visual_zone import parse_gcc_id
    assert "idNumber" not in parse_gcc_id(_vz_lines("PASSPORT", "DOE JANE", "1000000008"))["document"]
    doc = parse_gcc_id(_vz_lines("KINGDOM OF SAUDI ARABIA", "DOE JANE", "1000000008"))["document"]
    assert doc["idNumber"] == "1000000008"


def test_number_must_belong_to_the_state_the_card_names():
    from app.visual_zone import parse_gcc_id
    doc = parse_gcc_id(_vz_lines("STATE OF KUWAIT", "CIVIL ID CARD", "Civil ID No", "810512345"))["document"]
    assert doc.get("kind") != "bahrain_cpr" and "idNumber" not in doc
    doc = parse_gcc_id(_vz_lines("KINGDOM OF BAHRAIN", "IDENTITY CARD", "810512345"))["document"]
    assert doc["kind"] == "bahrain_cpr" and doc["issuingCountry"] == "BHR"


def test_self_checking_numbers_need_no_country_name():
    from app.visual_zone import parse_gcc_id
    from app.gcc_ids import luhn_ok
    n = next(f"78419801234567{d}" for d in "0123456789" if luhn_ok(f"78419801234567{d}"))
    assert parse_gcc_id(_vz_lines("DOE JANE", n))["document"]["kind"] == "uae_emirates_id"


def test_read_folder_summary_and_labels(tmp_path):
    import csv
    import importlib.util
    spec = importlib.util.spec_from_file_location("read_folder", Path(__file__).resolve().parent.parent / "tools" / "read_folder.py")
    rf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rf)
    text = rf.summarise({"p/a.jpg": {"format": "TD3", "checksPass": True, "needsReview": False, "status": "done", "engine": "x"},
                         "p/b.jpg": {"format": None, "status": "failed"}})
    total = text.splitlines()[2].split()
    assert total[:6] == ["TOTAL", "2", "1", "1", "0", "1"] and total[-1] == "1"
    with open(tmp_path / "sheet.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=rf.SHEET_HEADER)
        w.writeheader()
        w.writerow({"file": "p/a.jpg", "type": "passport", "documentNumber": "X1", "confirmed": "yes"})
        w.writerow({"file": "p/c.jpg", "type": "passport", "documentNumber": "X2", "confirmed": ""})
    rf.make_labels(tmp_path)
    import json
    assert json.loads((tmp_path / "labels.json").read_text()) == {"p/a.jpg": {"documentNumber": "X1", "type": "passport"}}


def test_text_shaped_like_an_mrz_is_not_taken_as_one():
    from app.mrz import plausible_reading
    good = ["P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<", "L898902C36UTO7408122F1204159ZE184226B<<<<<10"]
    assert plausible_reading(good)
    assert plausible_reading([good[0], good[1][:9] + "0" + good[1][10:]])          # one wrong check digit is still an MRZ
    assert not plausible_reading(["SOMEPRINTEDWORDS<<ONTHEFRONTOFACARDXX", "MORE<PRINTED<TEXTWITHNODIGITSATALLXX<"])
    assert not plausible_reading(["E" + "A" * 35, "A" * 30 + "<<<<<<"])
    assert not plausible_reading(None) and not plausible_reading(["TOO SHORT"])


def test_pipeline_reports_not_found_for_false_mrz():
    from app import pipeline
    from app.engines.base import EngineOutput
    from app.review import Applicant

    class Fake:
        name = "fake"
        def extract(self, images, hint):
            return EngineOutput(engine="ensemble", mrz_lines=["SOMEPRINTEDWORDS<<ONTHEFRONTOFACARDXX", "MORE<PRINTED<TEXTWITHNODIGITSATALLXX<"],
                                mrz_confidence=0.9, fields={"textLines": []}, confidence=0.9)
    status, res = pipeline.run([b"x"], None, Applicant(), Fake())
    assert status == "failed" and res["code"] == "mrz_not_found"


def test_auto_lets_the_second_pair_try_before_paddles_slow_search(monkeypatch):
    """PaddleOCR found no MRZ on the page: the second pair reads next, and PaddleOCR's last-chance search runs only if they
    find nothing either."""
    from app import engines
    from app.engines.base import EngineOutput
    calls = []
    good = ["P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<", "L898902C36UTO7408122F1204159ZE184226B<<<<<10"]

    class Paddle:
        def extract(self, images, hint, deep=True):
            calls.append(("paddle", deep))
            return EngineOutput(engine="paddle-ocr", mrz_lines=None, mrz_confidence=None, fields={"textLines": [{"text": "x"}]}, confidence=0.9)
        def mrz_retries(self, images):
            calls.append(("retries",))
            return good, 0.8

    class Pair:
        found = True
        def extract(self, images, hint):
            calls.append(("pair",))
            return EngineOutput(engine="ensemble", mrz_lines=good if self.found else None, mrz_confidence=0.99, fields={}, confidence=0.9)

    pair = Pair()
    monkeypatch.setattr(engines, "get_engine_by_name", lambda name: Paddle() if name == "paddle-ocr" else pair)
    out = engines.AutoEngine().extract([b"x"], None)
    assert out.engine == "ensemble" and calls == [("paddle", False), ("pair",)]
    calls.clear()
    pair.found = False
    out = engines.AutoEngine().extract([b"x"], None)
    assert out.engine == "paddle-ocr" and out.mrz_lines == good and calls[:3] == [("paddle", False), ("pair",), ("retries",)]


def test_sideways_page_is_turned_before_it_is_read():
    """A photo taken with the passport turned a quarter: the text boxes are tall, and the way up is the one that reads well."""
    import numpy as np
    from app.engines.paddle_ocr import SharedReader

    class Rec:
        def __init__(self, good_when_flipped):
            self.good_when_flipped = good_when_flipped
        def predict(self, crops):
            # a crop is "the right way up" when its marker pixel sits top-left
            return [{"rec_text": "X", "rec_score": 0.95 if (c[0, 0, 0] == 255) != self.good_when_flipped else 0.2} for c in crops]

    def crops_for(n):
        out = []
        for _ in range(n):
            c = np.zeros((10, 100, 3), dtype=np.uint8)
            c[0, 0, 0] = 255
            out.append(c)
        return out
    tall = [np.array([[10 + 30 * i, 10], [30 + 30 * i, 10], [30 + 30 * i, 400], [10 + 30 * i, 400]]) for i in range(5)]
    wide = [np.array([[10, 10 + 30 * i], [400, 10 + 30 * i], [400, 30 + 30 * i], [10, 30 + 30 * i]]) for i in range(5)]
    page = np.zeros((500, 500, 3), dtype=np.uint8)
    assert SharedReader(None, {"en": Rec(False)}, None, None).quarter_turns(page, tall, crops_for(5)) == 1
    assert SharedReader(None, {"en": Rec(True)}, None, None).quarter_turns(page, tall, crops_for(5)) == 3
    assert SharedReader(None, {"en": Rec(False)}, None, None).quarter_turns(page, wide, crops_for(5)) == 0      # upright page
    assert SharedReader(None, {"en": Rec(False)}, None, None).quarter_turns(page, wide + tall[:1], crops_for(6)) == 0
