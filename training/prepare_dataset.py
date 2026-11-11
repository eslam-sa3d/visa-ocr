#!/usr/bin/env python3
"""Build a PaddleOCR recognition dataset from reviewer corrections, PPOCRLabel exports and synthetic lines.

Output (in --out):
  images/            cropped line images
  train.txt          real train lines + a capped amount of synthetic lines
  val_real.txt       validation lines from REAL documents (use this to decide whether a model is better)
  val_synth.txt      synthetic validation lines (sanity check only)
  stats.json         counts per script, characters missing from the model dictionary

Label files use PaddleOCR's format: "<relative image path>\\t<text>".
Lines from the same document always go to the same split, so validation is not contaminated.

    python training/prepare_dataset.py --store-dir "$TRAINING_STORE_DIR" \
        --synth-dir data/synth --dict path/to/arabic_dict.txt --out data/rec
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from collections import Counter
from pathlib import Path

from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.textutil import script_of  # noqa: E402

MIN_SIDE = 8


def _read_labels(path: Path) -> list[tuple[str, str]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if "\t" in line:
            rel, text = line.split("\t", 1)
            if text.strip():
                rows.append((rel, text.strip()))
    return rows


def from_store(store_dir: Path, out: Path) -> list[dict]:
    """Cut line crops from reviewer corrections saved by POST /v1/jobs/{id}/corrections."""
    samples = []
    for job in sorted((store_dir / "jobs").glob("*")):
        corr = job / "correction.json"
        if not corr.exists():
            continue
        data = json.loads(corr.read_text(encoding="utf-8"))
        images = {int(p.stem): p for p in (job / "images").glob("*") if p.stem.isdigit()}
        for n, line in enumerate(data.get("lines", [])):
            src = images.get(line["image"])
            if src is None:
                continue
            with Image.open(src) as raw:
                im = ImageOps.exif_transpose(raw)          # boxes are in the upright image, as the readers saw it
                x1, y1, x2, y2 = line["box"]
                x1, y1 = max(0, min(x1, x2)), max(0, min(y1, y2))
                x2, y2 = min(im.width, max(x1, x2)), min(im.height, max(y1, y2))
                if x2 - x1 < MIN_SIDE or y2 - y1 < MIN_SIDE:
                    continue
                rel = f"images/real_{job.name}_{n}.png"
                im.convert("RGB").crop((x1, y1, x2, y2)).save(out / rel)
            samples.append({"doc": job.name, "path": rel, "text": line["text"]})
    return samples


def from_ppocrlabel(folder: Path, out: Path) -> list[dict]:
    """Ingest PPOCRLabel's rec export (rec_gt.txt + crop_img/). Crops of one page share a document id."""
    samples = []
    gt = folder / "rec_gt.txt"
    if not gt.exists():
        raise SystemExit(f"{gt} not found. In PPOCRLabel use File > Export Recognition Result first.")
    for rel_src, text in _read_labels(gt):
        src = folder / rel_src
        if not src.exists():
            continue
        rel = f"images/ppl_{folder.name}_{Path(rel_src).name}"
        shutil.copyfile(src, out / rel)
        samples.append({"doc": f"{folder.name}:{Path(rel_src).stem.rsplit('_crop_', 1)[0]}", "path": rel, "text": text})
    return samples


def from_synth(folder: Path, out: Path) -> tuple[list[dict], list[dict]]:
    def load(name: str, tag: str) -> list[dict]:
        rows = []
        for i, (rel_src, text) in enumerate(_read_labels(folder / name)):
            rel = f"images/synth_{folder.name}_{Path(rel_src).name}"
            shutil.copyfile(folder / rel_src, out / rel)
            rows.append({"doc": f"{tag}{i}", "path": rel, "text": text})
        return rows
    return load("train.txt", "s"), load("val.txt", "v")


def split_by_document(samples: list[dict], val_ratio: float, rng: random.Random):
    docs = sorted({s["doc"] for s in samples})
    rng.shuffle(docs)
    n_val = max(1, round(len(docs) * val_ratio)) if len(docs) >= 2 else 0
    val_docs = set(docs[:n_val])
    return [s for s in samples if s["doc"] not in val_docs], [s for s in samples if s["doc"] in val_docs]


_BAD_LABEL_CHARS = set("\t\r\n\x0b\x0c\x85\u2028\u2029")


def write(path: Path, samples: list[dict]) -> None:
    """One record per line: a label with a tab or line break would forge extra records, so such samples are skipped."""
    rows = [s for s in samples if not (_BAD_LABEL_CHARS & set(s["text"]))]
    if len(rows) != len(samples):
        print(f"WARNING: skipped {len(samples) - len(rows)} label(s) with tab or line-break characters", file=sys.stderr)
    path.write_text("".join(f"{s['path']}\t{s['text']}\n" for s in rows), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store-dir", type=Path, help="TRAINING_STORE_DIR with reviewer corrections")
    ap.add_argument("--ppocrlabel-dir", type=Path, nargs="*", default=[])
    ap.add_argument("--synth-dir", type=Path, nargs="*", default=[])
    ap.add_argument("--dict", type=Path, help="model character dictionary (one character per line)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--max-synth-ratio", type=float, default=3.0,
                    help="synthetic training lines are capped at this multiple of the real training lines")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    (args.out / "images").mkdir(parents=True, exist_ok=True)

    real: list[dict] = []
    if args.store_dir:
        real += from_store(args.store_dir, args.out)
    for folder in args.ppocrlabel_dir:
        real += from_ppocrlabel(folder, args.out)
    synth_train: list[dict] = []
    synth_val: list[dict] = []
    for folder in args.synth_dir:
        t, v = from_synth(folder, args.out)
        synth_train += t
        synth_val += v

    real_train, real_val = split_by_document(real, args.val_ratio, rng)
    rng.shuffle(synth_train)
    cap = int(len(real_train) * args.max_synth_ratio) if real_train else len(synth_train)
    synth_train = synth_train[:cap]

    train = real_train + synth_train
    rng.shuffle(train)
    write(args.out / "train.txt", train)
    write(args.out / "val_real.txt", real_val)
    write(args.out / "val_synth.txt", synth_val)

    stats = {
        "realTrain": len(real_train), "realVal": len(real_val),
        "synthTrain": len(synth_train), "synthVal": len(synth_val),
        "scripts": dict(Counter(script_of(s["text"]) for s in train)),
    }
    if args.dict:
        known = set(args.dict.read_text(encoding="utf-8").splitlines()) | {" "}
        missing = Counter(c for s in train + real_val for c in s["text"] if c not in known)
        stats["outOfDictionary"] = dict(missing.most_common(30))
        if missing:
            print("WARNING: characters not in the model dictionary:", "".join(missing),
                  "(changing the dictionary resets the output layer; prefer mapping these characters first)")
    if not real_val:
        print("WARNING: no real validation lines. Do not trust a model judged on synthetic lines only.")
    (args.out / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()
