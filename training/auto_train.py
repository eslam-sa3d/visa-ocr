#!/usr/bin/env python3
"""Automatic retraining: run this from cron or a systemd timer on the TRAINING server (GPU, offline is fine).

One run does the whole cycle, and stops early when there is nothing to do:

  1. count the reviewer corrections collected since the last run; stop if fewer than --min-new
  2. build the dataset (prepare_dataset.py); stop if there are too few REAL validation lines to judge a model
  3. fine-tune (finetune.sh)
  4. score the new model AND the model in use on the same real validation lines (predict_lines.py + evaluate.py)
  5. register the new model as a candidate; switch to it only if it is clearly better AND --activate was given

A new model never replaces the one in use unless its character error rate on real validation lines is lower by at
least --min-gain. Without --activate the run stops at "candidate": a person reads the report and activates it with
register_model.py. Every run writes report.json; a lock file stops two runs overlapping.

    python training/auto_train.py --store-dir "$TRAINING_STORE_DIR" --work-dir /data/autotrain \\
        --lang ar --arch arabic_PP-OCRv5_mobile_rec --dict /models/arabic_dict.txt [--activate]

Environment for finetune.sh is passed through: PADDLEOCR_DIR, BASE_CONFIG, PRETRAINED, EPOCHS, BATCH, USE_GPU.
This orchestration is covered by tests with the training step replaced; the real fine-tuning step has not been run
with production weights (see docs/MODEL_TRAINING.md).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from app.model_registry import Registry  # noqa: E402


def count_corrections(store_dir: Path) -> int:
    """Corrected lines saved by reviewers (jobs/<id>/correction.json, each with a list of lines)."""
    total = 0
    for path in (store_dir / "jobs").glob("*/correction.json"):
        try:
            total += len(json.loads(path.read_text(encoding="utf-8")).get("lines") or [])
        except (OSError, ValueError):
            continue
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


def cycle(args, runner=run_step, now: datetime | None = None) -> dict:
    now = now or datetime.now(timezone.utc)
    work: Path = args.work_dir
    work.mkdir(parents=True, exist_ok=True)
    state_file = work / "state.json"
    state = json.loads(state_file.read_text(encoding="utf-8")) if state_file.exists() else {"trainedLines": 0}
    report: dict = {"startedAt": now.isoformat(), "lang": args.lang}

    def finish(outcome: str, **extra) -> dict:
        report.update(outcome=outcome, **extra)
        (work / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False))
        return report

    lines = count_corrections(args.store_dir)
    report["correctedLines"], report["newLines"] = lines, lines - state["trainedLines"]
    if report["newLines"] < args.min_new:
        return finish("skipped_not_enough_new_lines", needed=args.min_new)

    stamp = now.strftime("%Y%m%d-%H%M")
    run_dir = work / stamp
    data_dir, out_dir = run_dir / "data", run_dir / "out"
    cmd = [sys.executable, HERE / "prepare_dataset.py", "--store-dir", args.store_dir, "--out", data_dir]
    if args.synth_dir:
        cmd += ["--synth-dir", *args.synth_dir]
    if args.dict:
        cmd += ["--dict", args.dict]
    runner(cmd)
    stats = json.loads((data_dir / "stats.json").read_text(encoding="utf-8"))
    report["dataset"] = {k: stats.get(k) for k in ("realTrain", "realVal", "synthTrain")}
    if (stats.get("realVal") or 0) < args.min_val:
        return finish("skipped_too_few_real_validation_lines", needed=args.min_val)

    env = {**os.environ, "DATA_DIR": str(data_dir), "OUT_DIR": str(out_dir)}
    runner(["bash", HERE / "finetune.sh"], env)

    reg = Registry(args.models_dir)
    model_id = f"{args.lang}_rec_auto_{stamp}"
    target = reg.base / model_id
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(out_dir / "inference", target)

    labels = data_dir / "val_real.txt"
    new = score(labels, data_dir, args.arch, target, run_dir, "candidate", runner)
    current = next((m for m in reg.models() if m.active and m.task == "text_recognition" and m.lang == args.lang), None)
    old = score(labels, data_dir, current.arch or args.arch, reg.full_path(current), run_dir, "current", runner) if current else None
    report["candidate"] = {"id": model_id, "cer": new.get("cer"), "exactMatch": new.get("exactMatch")}
    report["current"] = {"id": current.id, "cer": old.get("cer"), "exactMatch": old.get("exactMatch")} if current else None

    better = new.get("cer") is not None and (old is None or old.get("cer") is None or new["cer"] <= old["cer"] - args.min_gain)
    if current and old and old.get("cer") is not None:      # record the current model's score on today's validation lines
        runner([sys.executable, HERE / "register_model.py", "--set-metrics", current.id, "--metrics", run_dir / "metrics_current.json",
                *(["--models-dir", args.models_dir] if args.models_dir else [])])
    cmd = [sys.executable, HERE / "register_model.py", "--id", model_id, "--task", "text_recognition", "--lang", args.lang,
           "--version", f"auto-{stamp}", "--path", model_id, "--arch", args.arch, "--metrics", run_dir / "metrics_candidate.json",
           "--note", f"auto_train {stamp}: {lines} corrected lines"]
    if args.models_dir:
        cmd += ["--models-dir", args.models_dir]
    activate = bool(better and args.activate)
    if activate:
        cmd.append("--activate")
    runner(cmd)

    state["trainedLines"] = lines
    state_file.write_text(json.dumps(state) + "\n", encoding="utf-8")
    if not better:
        return finish("candidate_not_better_kept_current")
    return finish("activated" if activate else "candidate_better_waiting_for_approval",
                  approve=None if activate else f"python training/register_model.py --id {model_id} ... --activate")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store-dir", type=Path, required=True, help="TRAINING_STORE_DIR with reviewer corrections")
    ap.add_argument("--work-dir", type=Path, required=True)
    ap.add_argument("--lang", required=True, help="en or ar")
    ap.add_argument("--arch", required=True, help="PaddleOCR model name of the weights")
    ap.add_argument("--dict", type=Path)
    ap.add_argument("--synth-dir", type=Path, nargs="*", default=[])
    ap.add_argument("--models-dir", type=Path)
    ap.add_argument("--min-new", type=int, default=500, help="new corrected lines needed before retraining")
    ap.add_argument("--min-val", type=int, default=200, help="real validation lines needed to judge a model")
    ap.add_argument("--min-gain", type=float, default=0.002, help="CER must drop by at least this much")
    ap.add_argument("--activate", action="store_true", help="switch to the new model automatically when it is better")
    args = ap.parse_args()

    args.work_dir.mkdir(parents=True, exist_ok=True)
    lock = args.work_dir / "auto_train.lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise SystemExit(f"Another run holds {lock}. Remove it only if that run is dead.")
    try:
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        cycle(args)
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
