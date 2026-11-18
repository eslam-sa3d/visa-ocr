"""Automatic retraining cycle: the heavy steps (dataset build, fine-tune, predict) are replaced; the decisions are real."""
import json
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "training"))
import auto_train  # noqa: E402
from app.model_registry import ModelInfo, Registry  # noqa: E402


def _store(tmp_path, lines):
    d = tmp_path / "store" / "jobs" / "job1"
    d.mkdir(parents=True)
    (d / "correction.json").write_text(json.dumps({"status": "approved", "approvedBy": "second person", "reviewer": "first person", "lines": [{"text": "x"}] * lines}))
    return tmp_path / "store"


def _args(tmp_path, store, **kw):
    golden = tmp_path / "golden"
    golden.mkdir(exist_ok=True)
    (golden / "labels.txt").write_text("".join(f"g{i}.png\tX\n" for i in range(20)))
    base = dict(store_dir=store, work_dir=tmp_path / "work", lang="ar", arch="arch", dict=None, synth_dir=[],
                models_dir=tmp_path / "models", min_new=10, min_val=5, min_gain=0.002, activate=False,
                golden_dir=golden, min_golden=10)
    return Namespace(**{**base, **kw})


def _runner(cer_new, cer_old, real_val=50, new_extra=None, old_extra=None, seen=None, fail_training=False):
    """Stands in for the subprocess steps. register_model.py runs for real so the registry rules are exercised."""
    def run(cmd, env=None):
        cmd = [str(c) for c in cmd]
        name = Path(cmd[1]).name
        if name == "prepare_dataset.py":
            out = Path(cmd[cmd.index("--out") + 1]); out.mkdir(parents=True)
            (out / "stats.json").write_text(json.dumps({"realTrain": 400, "realVal": real_val, "synthTrain": 0,
                                                        "datasetSha256": "d" * 64}))
            (out / "val_real.txt").write_text("a.png\tX\n")
        elif name == "finetune.sh":
            if fail_training:
                raise subprocess.CalledProcessError(1, cmd)
            inf = Path(env["OUT_DIR"]) / "inference"; inf.mkdir(parents=True)
            (inf / "inference.pdiparams").write_text("weights")
        elif name == "predict_lines.py":
            if seen is not None:
                seen.append(cmd[cmd.index("--labels") + 1])
            Path(cmd[cmd.index("--out") + 1]).write_text("a.png\tX\n")
        elif name == "evaluate.py":
            out = Path(cmd[cmd.index("--out") + 1])
            cand = "candidate" in out.name
            out.write_text(json.dumps({"cer": cer_new if cand else cer_old, "exactMatch": 0.9, "samples": 500, "chars": 20000,
                                       **((new_extra if cand else old_extra) or {})}))
        else:
            subprocess.run(cmd, check=True, env=env)
    return run


def _current_model(tmp_path):
    reg = Registry(tmp_path / "models")
    (reg.base / "cur").mkdir(parents=True)
    (reg.base / "cur" / "w").write_text("w")
    reg.save([ModelInfo(id="cur", task="text_recognition", lang="ar", version="pretrained", path="cur", arch="arch",
                        active=True, metrics={"cer": 0.05})])
    return reg


def test_skips_when_too_few_new_corrections(tmp_path):
    out = auto_train.cycle(_args(tmp_path, _store(tmp_path, 3)), runner=_runner(0.01, 0.05))
    assert out["outcome"] == "skipped_not_enough_new_lines"


def test_skips_when_too_few_real_validation_lines(tmp_path):
    out = auto_train.cycle(_args(tmp_path, _store(tmp_path, 50)), runner=_runner(0.01, 0.05, real_val=2))
    assert out["outcome"] == "skipped_too_few_real_validation_lines"


def test_better_model_waits_for_approval_by_default(tmp_path):
    reg = _current_model(tmp_path)
    out = auto_train.cycle(_args(tmp_path, _store(tmp_path, 50)), runner=_runner(0.02, 0.05))
    assert out["outcome"] == "candidate_better_waiting_for_approval"
    assert reg.active("text_recognition", "ar").id == "cur"                 # nothing switched
    assert any(m.id == out["candidate"]["id"] and not m.active for m in reg.models())


def test_better_model_is_activated_only_with_the_flag(tmp_path):
    reg = _current_model(tmp_path)
    out = auto_train.cycle(_args(tmp_path, _store(tmp_path, 50), activate=True), runner=_runner(0.02, 0.05))
    assert out["outcome"] == "activated"
    assert reg.active("text_recognition", "ar").id == out["candidate"]["id"]


def test_worse_model_never_replaces_the_current_one(tmp_path):
    reg = _current_model(tmp_path)
    out = auto_train.cycle(_args(tmp_path, _store(tmp_path, 50), activate=True), runner=_runner(0.06, 0.05))
    assert out["outcome"] == "candidate_not_better_kept_current"
    assert reg.active("text_recognition", "ar").id == "cur"
    # the same corrections do not trigger another run
    again = auto_train.cycle(_args(tmp_path, tmp_path / "store", activate=True), runner=_runner(0.06, 0.05))
    assert again["outcome"] == "skipped_not_enough_new_lines"
