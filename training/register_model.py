#!/usr/bin/env python3
"""Register a model folder in models/manifest.json (checksum + metrics), activate it, or go back to the one before.

Copy the exported inference model into MODELS_DIR first, then:

    python training/register_model.py --id ar_rec_ft_2026-12-01 --task text_recognition --lang ar \
        --version ft-2026-12-01 --path ar_rec_ft_2026-12-01 --arch arabic_PP-OCRv5_mobile_rec \
        --metrics metrics.json --activate

    python training/register_model.py --activate-existing ar_rec_auto_20261201-0200     # a registered candidate goes live
    python training/register_model.py --rollback --task text_recognition --lang ar      # back to the model before this one

Activating a NEW model refuses to replace the current one for the same task and language unless the new model's CER is
lower (use --force to override, and say why in --note). --activate-existing and --rollback are a person's decision and are
not compared, but the model's files must still match its checksum. Each activation records which model it replaced, which is
what --rollback returns to. Running servers notice the change and load the model on their next read; if it does not load,
they keep the one they have and log why.

Every change is made under a lock and written whole, so the retraining timer and an operator cannot damage the manifest
between them.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app import audit  # noqa: E402
from app.model_registry import ModelInfo, Registry, dir_sha256  # noqa: E402


def register(reg: Registry, info: ModelInfo, activate: bool, force: bool, replace: bool = False) -> str:
    with reg.locked():
        existing = reg.models()
        old = next((m for m in existing if m.id == info.id), None)
        if old is not None and not replace:
            raise SystemExit(f"A model with id {info.id} is already registered. Choose another id, or pass --replace to "
                             "register it again (not allowed for the model in use).")
        if old is not None and old.active:
            raise SystemExit(f"{info.id} is the model in use; it cannot be registered again. Register under a new id.")
        models = [m for m in existing if m.id != info.id]
        folder = reg.full_path(info)
        if not folder.is_dir():
            raise SystemExit(f"Model folder not found: {folder}")
        info.sha256 = dir_sha256(folder)

        if activate:
            current = next((m for m in models if m.active and m.task == info.task and m.lang == info.lang), None)
            old_cer = (current.metrics or {}).get("cer") if current else None
            new_cer = (info.metrics or {}).get("cer")
            if current and not force:
                if old_cer is None:
                    raise SystemExit(
                        f"Refusing to activate: the current model {current.id} has no recorded CER, so the new one cannot be "
                        f"compared. Evaluate {current.id} on the same golden set, then run: "
                        f"register_model.py --set-metrics {current.id} --metrics its_metrics.json. Use --force to override.")
                if new_cer is None or new_cer >= old_cer:
                    raise SystemExit(
                        f"Refusing to activate: new CER {new_cer} is not better than current {old_cer} "
                        f"({current.id}). Use --force to override.")
            for m in models:
                if m.task == info.task and m.lang == info.lang:
                    if m.active:
                        info.replaced = m.id
                    m.active = False
            info.active = True
        reg.save(models + [info])
        audit.event("model_registered", model=info.id, task=info.task, lang=info.lang, active=info.active, replaced=info.replaced,
                    forced=bool(force), sha256=info.sha256[:16])
        return info.sha256


def activate_existing(reg: Registry, model_id: str) -> tuple[str, str]:
    """Make a registered model the one in use. Returns (id now in use, id it replaced or '')."""
    with reg.locked():
        models = reg.models()
        target = next((m for m in models if m.id == model_id), None)
        if target is None:
            raise SystemExit(f"{model_id} is not registered")
        status = reg.status(target)
        if status != "verified":
            raise SystemExit(f"Refusing to activate {model_id}: its files are {status}")
        if target.active:
            return target.id, target.replaced
        previous = ""
        for m in models:
            if m.active and m.task == target.task and m.lang == target.lang:
                previous, m.active = m.id, False
        target.active, target.replaced = True, previous
        reg.save(models)
        audit.event("model_activated", model=target.id, task=target.task, lang=target.lang, replaced=previous)
        return target.id, previous


def rollback(reg: Registry, task: str, lang: str) -> tuple[str, str]:
    """Return to the model that the current one replaced. Returns (id now in use, id switched away from)."""
    current = reg.active(task, lang)
    if current is None:
        raise SystemExit(f"No active {task} model for language {lang}")
    if not current.replaced:
        raise SystemExit(f"{current.id} did not replace another model: nothing to roll back to. "
                         "Use --activate-existing ID to choose one.")
    now, _ = activate_existing(reg, current.replaced)
    audit.event("model_rolled_back", model=now, from_model=current.id, task=task, lang=lang)
    return now, current.id


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set-metrics", metavar="ID", help="only record metrics.json for an already registered model")
    ap.add_argument("--activate-existing", metavar="ID", help="make an already registered model the one in use")
    ap.add_argument("--rollback", action="store_true", help="return to the model the current one replaced (needs --task, --lang)")
    ap.add_argument("--id")
    ap.add_argument("--task", choices=["text_detection", "text_recognition"])
    ap.add_argument("--lang", help="en, ar or any")
    ap.add_argument("--version")
    ap.add_argument("--path", help="folder name under the models dir")
    ap.add_argument("--arch", default="", help="PaddleOCR model name the weights belong to")
    ap.add_argument("--metrics", type=Path, help="metrics.json from evaluate.py")
    ap.add_argument("--note", default="")
    ap.add_argument("--models-dir", type=Path)
    ap.add_argument("--activate", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--replace", action="store_true", help="register again under an id that exists (never the model in use)")
    args = ap.parse_args()

    reg = Registry(args.models_dir)
    if args.activate_existing:
        now, before = activate_existing(reg, args.activate_existing)
        print(f"in use now: {now}" + (f" (replaced {before}; undo with --rollback)" if before else ""))
        return
    if args.rollback:
        if not (args.task and args.lang):
            ap.error("--rollback needs --task and --lang")
        now, before = rollback(reg, args.task, args.lang)
        print(f"rolled back: {before} -> {now}")
        return
    metrics = json.loads(args.metrics.read_text(encoding="utf-8")) if args.metrics else {}
    if args.set_metrics:
        with reg.locked():
            models = reg.models()
            target = next((m for m in models if m.id == args.set_metrics), None)
            if target is None or not metrics:
                raise SystemExit("--set-metrics needs an existing model id and --metrics file")
            target.metrics = metrics
            reg.save(models)
        print(f"recorded metrics for {target.id}")
        return
    for need in ("id", "task", "lang", "version", "path"):
        if not getattr(args, need):
            ap.error(f"--{need} is required")
    info = ModelInfo(id=args.id, task=args.task, lang=args.lang, version=args.version, path=args.path,
                     arch=args.arch, source=args.note or "fine-tuned", metrics=metrics)
    digest = register(reg, info, args.activate, args.force, args.replace)
    print(f"registered {args.id} sha256={digest[:16]}... active={info.active}")


if __name__ == "__main__":
    main()
