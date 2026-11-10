#!/usr/bin/env python3
"""Register a model folder in models/manifest.json (checksum + metrics) and optionally activate it.

Copy the exported inference model into MODELS_DIR first, then:

    python training/register_model.py --id ar_rec_ft_2026-12-01 --task text_recognition --lang ar \
        --version ft-2026-12-01 --path ar_rec_ft_2026-12-01 --arch arabic_PP-OCRv5_mobile_rec \
        --metrics metrics.json --activate

Activating refuses to replace the current model for the same task and language unless the new
model's CER on real validation lines is lower (use --force to override, and say why in --note).
Rolling back is just re-running with --activate on the older id, or editing the manifest.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.model_registry import ModelInfo, Registry, dir_sha256  # noqa: E402


def register(reg: Registry, info: ModelInfo, activate: bool, force: bool) -> str:
    models = [m for m in reg.models() if m.id != info.id]
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
                    f"compared. Evaluate {current.id} on the same real validation file, then run: "
                    f"register_model.py --set-metrics {current.id} --metrics its_metrics.json. Use --force to override.")
            if new_cer is None or new_cer >= old_cer:
                raise SystemExit(
                    f"Refusing to activate: new CER {new_cer} is not better than current {old_cer} "
                    f"({current.id}). Use --force to override.")
        for m in models:
            if m.task == info.task and m.lang == info.lang:
                m.active = False
        info.active = True
    reg.save(models + [info])
    return info.sha256


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set-metrics", metavar="ID", help="only record metrics.json for an already registered model")
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
    args = ap.parse_args()

    metrics = json.loads(args.metrics.read_text(encoding="utf-8")) if args.metrics else {}
    if args.set_metrics:
        reg = Registry(args.models_dir)
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
    digest = register(Registry(args.models_dir), info, args.activate, args.force)
    print(f"registered {args.id} sha256={digest[:16]}... active={info.active}")


if __name__ == "__main__":
    main()
