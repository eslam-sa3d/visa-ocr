#!/usr/bin/env python3
"""Automatic retraining: run this from cron or a systemd timer on the TRAINING server (GPU, offline is fine).

One run does the whole cycle, and stops early when there is nothing to do:

  1. count the corrections a second person APPROVED since the last run; stop if fewer than --min-new
  2. build the dataset (prepare_dataset.py); stop if there are too few real validation lines to train with
  3. fine-tune (finetune.sh). Its validation file picks the best checkpoint, so it cannot also judge the result
  4. score the new model AND the model in use on the GOLDEN SET (--golden-dir): lines kept apart by the operator, never
     trained on, the same from run to run (predict_lines.py + evaluate.py)
  5. register the new model as a candidate; switch to it only if it passes the gate AND --activate was given

The gate (see gate()) fails closed. A new model replaces the one in use only when, on the golden set, its character error
rate is lower by at least --min-gain and by more than the measurement noise, no script and no document type got worse,
and no fewer lines are read exactly right. A model that could not be scored never passes. Without a golden set a
candidate is built and registered, and a person decides (register_model.py --activate-existing ID).
Without --activate the run stops at "candidate". Every activation can be undone: register_model.py --rollback.

Every run writes report.json in the work folder and in its own run folder and adds a line to history.jsonl: the dataset's
checksum, settings, package versions, seed, scores and the gate's reasons. A lock file stops two runs overlapping; a lock
left by a dead run is cleared. Training that fails twice on the same data is not tried a third time.

    python training/auto_train.py --store-dir "$TRAINING_STORE_DIR" --work-dir /data/autotrain \\
        --lang ar --arch arabic_PP-OCRv5_mobile_rec --dict /models/arabic_dict.txt \\
        --golden-dir /data/golden/ar [--activate] [--continue-from-current]

Environment for finetune.sh is passed through: PADDLEOCR_DIR, BASE_CONFIG, PRETRAINED, PRETRAINED_SHA256, EPOCHS, BATCH, USE_GPU.
This orchestration is covered by tests with the training step replaced; the real fine-tuning step has not been run
with production weights (see docs/MODEL_TRAINING.md).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from app.model_registry import Registry  # noqa: E402
from app.training_store import approved_correction  # noqa: E402


def count_corrections(store_dir: Path) -> int:
    """Corrected lines that a second person approved (see tools/approve_corrections.py). Pending ones do not count."""
    total = 0
    for job in (store_dir / "jobs").glob("*"):
        rec = approved_correction(job)
        if rec and isinstance(rec.get("lines"), list):
            total += len(rec["lines"])
    return total


def run_step(cmd: list[str], env: dict | None = None) -> None:
    print("+", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True, env=env)


def score(labels: Path, image_dir: Path, arch: str, model_dir: Path, out_dir: Path, tag: str, runner) -> dict:
    preds, metrics = out_dir / f"preds_{tag}.txt", out_dir / f"metrics_{tag}.json"
    runner([sys.executable, HERE / "predict_lines.py", "--labels", labels, "--image-dir", image_dir,
            "--model-name", arch, "--model-dir", model_dir, "--out", preds])
    runner([sys.executable, HERE / "evaluate.py", "--labels", labels, "--predictions", preds, "--out", metrics])
    return json.loads(metrics.read_text(encoding="utf-8"))


MIN_BUCKET_SAMPLES = 20       # a script or document type with fewer golden lines than this is reported but does not block


def gate(new: dict, old: dict | None, min_gain: float) -> list[str]:
    """Why the candidate must NOT replace the model in use. An empty list means it may.

    It fails closed: a model that could not be scored never passes. The gain must be at least --min-gain AND larger than the
    noise of the measurement (two standard errors of the current error rate on this many characters). No script and no
    document type with enough golden lines may get worse, and no fewer lines may be read exactly right."""
    if old is None or old.get("cer") is None:
        return ["the model in use could not be scored on the golden set"]
    if new.get("cer") is None:
        return ["the candidate could not be scored on the golden set"]
    why: list[str] = []
    chars = max(int(new.get("chars") or old.get("chars") or 0), 1)
    noise = 2 * math.sqrt(max(old["cer"] * (1 - min(old["cer"], 1.0)), 0.0) / chars)
    need = max(min_gain, noise)
    if old["cer"] - new["cer"] < need:
        why.append(f"CER {new['cer']} is not lower than {old['cer']} by at least {round(need, 4)}")
    if (new.get("exactMatch") or 0) < (old.get("exactMatch") or 0):
        why.append(f"fewer lines are read exactly right ({new.get('exactMatch')} against {old.get('exactMatch')})")
    for key, label in (("byScript", "text"), ("byGroup", "documents")):
        for name, o in (old.get(key) or {}).items():
            n = (new.get(key) or {}).get(name) or {}
            if (o.get("samples") or 0) < MIN_BUCKET_SAMPLES or o.get("cer") is None:
                continue
            if n.get("cer") is None or n["cer"] > o["cer"]:
                why.append(f"worse on {name} {label} (CER {n.get('cer')} against {o['cer']})")
    return why


def _versions() -> dict:
    out = {"python": sys.version.split()[0]}
    for name in ("paddlepaddle", "paddlepaddle-gpu", "paddleocr", "paddlex", "pillow"):
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    try:
        out["commit"] = subprocess.run(["git", "-C", str(HERE.parent), "rev-parse", "HEAD"], capture_output=True, text=True,
                                       timeout=10).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        out["commit"] = None
    return out


def _prune_candidates(reg: Registry, lang: str, keep: int, work: Path) -> list[str]:
    """Remove automatic candidates that were never put in use, keeping the newest `keep`. The model in use and the one it
    replaced (the rollback target) are never touched."""
    if keep < 0:
        return []
    removed: list[str] = []
    with reg.locked():
        models = reg.models()
        protected = {m.id for m in models if m.active} | {m.replaced for m in models if m.active and m.replaced}
        prefix = f"{lang}_rec_auto_"
        spare = sorted(m.id for m in models if m.id.startswith(prefix) and m.id not in protected)
        for model_id in spare[:max(0, len(spare) - keep)]:
            m = next(x for x in models if x.id == model_id)
            try:
                shutil.rmtree(reg.full_path(m), ignore_errors=True)
            except ValueError:
                continue
            shutil.rmtree(work / "checkpoints" / model_id, ignore_errors=True)
            models = [x for x in models if x.id != model_id]
            removed.append(model_id)
        if removed:
            reg.save(models)
    return removed


def cycle(args, runner=run_step, now: datetime | None = None) -> dict:
    """One retraining cycle. Line crops, label files and predictions are personal data taken from the training store, so they
    are removed when the cycle ends, however it ends, and run folders older than --keep-days are deleted. Training
    checkpoints are removed too (the best one is kept outside the run folder when --continue-from-current is used)."""
    now = now or datetime.now(timezone.utc)
    work: Path = args.work_dir
    work.mkdir(parents=True, exist_ok=True, mode=0o700)
    keep = float(getattr(args, "keep_days", 30) or 0)
    if keep > 0:
        for old in work.iterdir():
            if old.is_dir() and old.name != "checkpoints" and now.timestamp() - old.stat().st_mtime > keep * 86400:
                shutil.rmtree(old, ignore_errors=True)
    run_dir = work / now.strftime("%Y%m%d-%H%M")
    try:
        return _cycle(args, runner, now, work)
    finally:
        shutil.rmtree(run_dir / "data", ignore_errors=True)
        shutil.rmtree(run_dir / "out" / "train", ignore_errors=True)      # checkpoints: large, and no longer needed
        if run_dir.is_dir():
            for preds in run_dir.glob("preds_*.txt"):
                preds.unlink(missing_ok=True)


def _cycle(args, runner, now: datetime, work: Path) -> dict:
    state_file = work / "state.json"
    state = json.loads(state_file.read_text(encoding="utf-8")) if state_file.exists() else {"trainedLines": 0}
    stamp = now.strftime("%Y%m%d-%H%M")
    run_dir = work / stamp
    report: dict = {"startedAt": now.isoformat(), "lang": args.lang, "run": stamp}

    def save_state() -> None:
        state_file.write_text(json.dumps(state) + "\n", encoding="utf-8")

    def finish(outcome: str, **extra) -> dict:
        report.update(outcome=outcome, **extra)
        text = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
        (work / "report.json").write_text(text, encoding="utf-8")                # the latest run, where it always was
        if run_dir.is_dir():
            (run_dir / "report.json").write_text(text, encoding="utf-8")         # kept with the run
        with open(work / "history.jsonl", "a", encoding="utf-8") as fh:          # every run, one line each
            fh.write(json.dumps(report, ensure_ascii=False) + "\n")
        print(json.dumps(report, ensure_ascii=False))
        return report

    lines = count_corrections(args.store_dir)
    if lines < state["trainedLines"]:           # the retention purge removed lines: count new ones from the present total
        state["trainedLines"] = lines
        save_state()
    report["correctedLines"], report["newLines"] = lines, lines - state["trainedLines"]
    if report["newLines"] < args.min_new:
        return finish("skipped_not_enough_new_lines", needed=args.min_new)

    golden_dir = getattr(args, "golden_dir", None)
    seed = int(getattr(args, "seed", 1) or 1)
    data_dir, out_dir = run_dir / "data", run_dir / "out"
    cmd = [sys.executable, HERE / "prepare_dataset.py", "--store-dir", args.store_dir, "--out", data_dir, "--lang", args.lang,
           "--seed", seed]
    if golden_dir:
        cmd += ["--golden-dir", golden_dir]
    if args.synth_dir:
        cmd += ["--synth-dir", *args.synth_dir]
    if args.dict:
        cmd += ["--dict", args.dict]
    runner(cmd)
    stats = json.loads((data_dir / "stats.json").read_text(encoding="utf-8"))
    report["dataset"] = {k: stats.get(k) for k in ("realTrain", "realVal", "synthTrain", "skipped", "datasetSha256")}
    if (stats.get("realVal") or 0) < args.min_val:
        return finish("skipped_too_few_real_validation_lines", needed=args.min_val)

    # The same data that already failed is not trained again and again by the timer: a person looks at it first.
    failed = state.get("failed") or {}
    data_id = stats.get("datasetSha256") or f"lines:{lines}"
    if failed.get("dataset") == data_id and failed.get("attempts", 0) >= MAX_ATTEMPTS_PER_DATASET:
        return finish("skipped_training_failed_before_on_this_data", attempts=failed["attempts"], lastError=failed.get("error"))

    reg = Registry(args.models_dir)
    current = next((m for m in reg.models() if m.active and m.task == "text_recognition" and m.lang == args.lang), None)
    env = {**os.environ, "DATA_DIR": str(data_dir), "OUT_DIR": str(out_dir), "SEED": str(seed),
           "PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK": "True", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    start_from = "pretrained"
    if getattr(args, "continue_from_current", False) and current is not None:
        ckpt = work / "checkpoints" / current.id / "best_accuracy"
        if ckpt.with_suffix(".pdparams").is_file():
            # Learning carries over from the model in use, so it is not limited to the corrections still inside the
            # retention period. The golden set is what guards against forgetting older document types.
            digest = hashlib.sha256(ckpt.with_suffix(".pdparams").read_bytes()).hexdigest()
            env.update(PRETRAINED=str(ckpt), PRETRAINED_SHA256=digest)
            start_from = current.id
    report["training"] = {"startFrom": start_from, "seed": seed, "versions": _versions(),
                          "settings": {k: env.get(k) for k in ("BASE_CONFIG", "EPOCHS", "BATCH", "EVAL_STEP", "USE_GPU")},
                          "pretrained": Path(env["PRETRAINED"]).name if env.get("PRETRAINED") else None}
    try:
        runner(["bash", HERE / "finetune.sh"], env)
    except Exception as exc:  # noqa: BLE001 - whatever stopped the training step, record it and stop this cycle
        state["failed"] = {"dataset": data_id, "attempts": (failed.get("attempts", 0) + 1) if failed.get("dataset") == data_id else 1,
                           "error": type(exc).__name__, "at": now.isoformat()}
        save_state()
        return finish("training_failed", error=type(exc).__name__, attempts=state["failed"]["attempts"])
    state.pop("failed", None)

    model_id = f"{args.lang}_rec_auto_{stamp}"
    target = reg.base / model_id
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(out_dir / "inference", target)
    best = out_dir / "train" / "best_accuracy.pdparams"
    if best.is_file():                               # kept so a later run can continue from this model
        keep_dir = work / "checkpoints" / model_id
        keep_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        for part in (out_dir / "train").glob("best_accuracy.*"):
            shutil.copy2(part, keep_dir / part.name)

    def register(metrics_file: Path | None, activate: bool) -> None:
        cmd = [sys.executable, HERE / "register_model.py", "--id", model_id, "--task", "text_recognition", "--lang", args.lang,
               "--version", f"auto-{stamp}", "--path", model_id, "--arch", args.arch, "--replace",
               "--note", f"auto_train {stamp}: {lines} approved lines, dataset {str(data_id)[:12]}, from {start_from}"]
        if metrics_file:
            cmd += ["--metrics", metrics_file]
        if args.models_dir:
            cmd += ["--models-dir", args.models_dir]
        if activate:
            cmd.append("--activate")
        runner(cmd)

    def done(outcome: str, **extra) -> dict:
        state["trainedLines"] = lines
        save_state()
        removed = _prune_candidates(reg, args.lang, int(getattr(args, "keep_candidates", 3)), work)
        return finish(outcome, **({"prunedCandidates": removed} if removed else {}), **extra)

    report["candidate"] = {"id": model_id}
    how_to = f"python training/register_model.py --activate-existing {model_id}"
    # val_real.txt chose the checkpoint during training, so it cannot also judge the result. Only the golden set can: lines
    # the operator keeps apart, that no model was ever trained on.
    labels = (golden_dir / "labels.txt") if golden_dir else None
    golden_lines = sum(1 for ln in labels.read_text(encoding="utf-8").splitlines() if "\t" in ln) if labels and labels.is_file() else 0
    min_golden = int(getattr(args, "min_golden", 200))
    if golden_lines < min_golden:
        register(None, False)
        return done("candidate_built_no_golden_set" if not golden_lines else "candidate_built_golden_set_too_small",
                    goldenLines=golden_lines, needed=min_golden, approve=how_to)

    new = score(labels, golden_dir, args.arch, target, run_dir, "candidate", runner)
    old = score(labels, golden_dir, current.arch or args.arch, reg.full_path(current), run_dir, "current", runner) if current else None
    report["candidate"].update(cer=new.get("cer"), exactMatch=new.get("exactMatch"))
    report["current"] = {"id": current.id, "cer": old.get("cer"), "exactMatch": old.get("exactMatch")} if current else None
    reasons = gate(new, old, args.min_gain)
    better = not reasons
    report["gate"] = {"passed": better, "reasons": reasons, "goldenLines": golden_lines,
                      "byScript": {"candidate": new.get("byScript"), "current": (old or {}).get("byScript")},
                      "byGroup": {"candidate": new.get("byGroup"), "current": (old or {}).get("byGroup")}}
    if current and old and old.get("cer") is not None:      # record the current model's score on the golden set
        runner([sys.executable, HERE / "register_model.py", "--set-metrics", current.id, "--metrics", run_dir / "metrics_current.json",
                *(["--models-dir", args.models_dir] if args.models_dir else [])])
    activate = bool(better and args.activate)
    register(run_dir / "metrics_candidate.json", activate)
    if not better:
        return done("candidate_not_better_kept_current")
    return done("activated" if activate else "candidate_better_waiting_for_approval",
                approve=None if activate else how_to,
                undo=f"python training/register_model.py --rollback --task text_recognition --lang {args.lang}" if activate else None)


MAX_ATTEMPTS_PER_DATASET = 2


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def take_lock(lock: Path) -> None:
    """Create the lock file, or stop. A lock left by a run that is no longer alive (crash, power loss) is cleared: without
    that, one bad night would stop retraining for good with nobody told."""
    for _ in range(2):
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                pid = int(lock.read_text().strip() or "0")
            except (OSError, ValueError):
                pid = 0
            if pid and _pid_alive(pid):
                raise SystemExit(f"Another run (pid {pid}) holds {lock}.")
            print(f"removing a stale lock left by pid {pid or 'unknown'}", file=sys.stderr)
            lock.unlink(missing_ok=True)
            continue
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        return
    raise SystemExit(f"Could not take {lock}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store-dir", type=Path, required=True, help="TRAINING_STORE_DIR with reviewer corrections")
    ap.add_argument("--work-dir", type=Path, required=True)
    ap.add_argument("--lang", required=True, help="en or ar")
    ap.add_argument("--arch", required=True, help="PaddleOCR model name of the weights")
    ap.add_argument("--dict", type=Path)
    ap.add_argument("--synth-dir", type=Path, nargs="*", default=[])
    ap.add_argument("--models-dir", type=Path)
    ap.add_argument("--golden-dir", type=Path,
                    help="the golden set: labels.txt and its line images, kept apart from the training store and never trained on. "
                         "Without it a candidate is built and registered, but never compared or activated")
    ap.add_argument("--min-golden", type=int, default=200, help="golden lines needed to judge a model")
    ap.add_argument("--min-new", type=int, default=500, help="new approved lines needed before retraining")
    ap.add_argument("--min-val", type=int, default=200, help="real validation lines needed for training")
    ap.add_argument("--min-gain", type=float, default=0.002, help="CER must drop by at least this much (and by more than the noise)")
    ap.add_argument("--activate", action="store_true", help="switch to the new model automatically when it passes the gate")
    ap.add_argument("--continue-from-current", action="store_true",
                    help="start from the checkpoint of the model in use (when this tool trained it) instead of the pretrained weights")
    ap.add_argument("--keep-candidates", type=int, default=3, help="unused automatic candidates to keep (-1 keeps all)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--keep-days", type=float, default=30,
                    help="delete run folders older than this (match TRAINING_RETENTION_DAYS; 0 keeps them)")
    args = ap.parse_args()

    os.umask(0o077)      # run folders, reports and everything the child steps write are owner-only, like the training store
    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.work_dir.chmod(0o700)
    lock = args.work_dir / "auto_train.lock"
    take_lock(lock)
    try:
        report = cycle(args)
    finally:
        lock.unlink(missing_ok=True)
    if report.get("outcome") == "training_failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
