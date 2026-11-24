#!/usr/bin/env python3
"""Build a PaddleOCR recognition dataset from reviewer corrections, PPOCRLabel exports and synthetic lines.

Output (in --out):
  images/            cropped line images
  train.txt          real train lines + a capped amount of synthetic lines
  val_real.txt       validation lines from REAL documents. Used DURING training to pick the best checkpoint, so it cannot
                     also judge the finished model: that is the golden set's job (see auto_train.py --golden-dir)
  val_synth.txt      synthetic validation lines (sanity check only)
  stats.json         counts per script, what was skipped and why, characters missing from the model dictionary, and a
                     SHA-256 of the label files (the dataset's identity, recorded with every model trained on it)

Label files use PaddleOCR's format: "<relative image path>\\t<text>".
Lines from the same document always go to the same split, so validation is not contaminated.
Only corrections that a second person approved (tools/approve_corrections.py) are used. A correction whose MRZ lines fail
their own check digits, a malformed record, a line in another language, a duplicate crop and any crop that is also in
the golden set are skipped and counted.

    python training/prepare_dataset.py --store-dir "$TRAINING_STORE_DIR" \
        --synth-dir data/synth --dict path/to/arabic_dict.txt --out data/rec
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import random
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app import crypto  # noqa: E402
from app.mrz import check_score  # noqa: E402
from app.textutil import script_of  # noqa: E402
from app.training_store import approved_correction, load_corrections  # noqa: E402

MIN_SIDE = 8


def _read_labels(path: Path) -> list[tuple[str, str]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if "\t" in line:
            rel, text = line.split("\t", 1)
            if text.strip():
                rows.append((rel, text.strip()))
    return rows


_MRZ_SHAPED = re.compile(r"[A-Z0-9<]{28,44}")


def _well_formed(line) -> bool:
    return (isinstance(line, dict) and isinstance(line.get("image"), int) and not isinstance(line.get("image"), bool)
            and isinstance(line.get("text"), str) and bool(line["text"].strip())
            and isinstance(line.get("box"), list) and len(line["box"]) == 4
            and all(isinstance(v, int) and not isinstance(v, bool) for v in line["box"]))


def golden_hashes(folder: Path | None) -> set[str]:
    """SHA-256 of the pixels of every image in the golden set, so none of them can slip into training data."""
    out: set[str] = set()
    if folder and folder.is_dir():
        for path in folder.rglob("*"):
            if path.suffix.lower() in (".png", ".jpg", ".jpeg"):
                try:
                    with Image.open(path) as im:
                        out.add(_pixel_hash(im))
                except OSError:
                    continue
    return out


def _pixel_hash(im) -> str:
    rgb = im.convert("RGB")
    return hashlib.sha256(repr(rgb.size).encode() + rgb.tobytes()).hexdigest()


def job_images(job_dir: Path) -> dict[int, Path]:
    return {int(p.stem): p for p in (job_dir / "images").glob("*") if p.stem.isdigit()}


def cut_line(job_dir: Path, line: dict, images: dict[int, Path] | None = None):
    """The crop a correction line points at, as (image, None), or (None, reason). The one place crops are cut, so the approver
    (tools/approve_corrections.py --show) sees exactly the pixels that become training data."""
    images = images if images is not None else job_images(job_dir)
    src = images.get(line.get("image"))
    if src is None:
        return None, "noImage"
    try:
        with Image.open(io.BytesIO(crypto.unseal(src.read_bytes()))) as raw:      # stored encrypted when a key is set
            im = ImageOps.exif_transpose(raw)          # boxes are in the upright image, as the readers saw it
            x1, y1, x2, y2 = line["box"]
            x1, y1 = max(0, min(x1, x2)), max(0, min(y1, y2))
            x2, y2 = min(im.width, max(x1, x2)), min(im.height, max(y1, y2))
            if x2 - x1 < MIN_SIDE or y2 - y1 < MIN_SIDE:
                return None, "boxTooSmall"
            return im.convert("RGB").crop((x1, y1, x2, y2)), None
    except (OSError, ValueError, crypto.DataKeyError):
        return None, "unreadableImage"


def crop_digest(crop) -> str:
    return _pixel_hash(crop)


def from_store(store_dir: Path, out: Path, lang: str | None = None, skipped: Counter | None = None,
               exclude: set[str] | None = None) -> list[dict]:
    """Cut line crops from reviewer corrections saved by POST /v1/jobs/{id}/corrections. Only approved corrections count."""
    samples = []
    skipped = skipped if skipped is not None else Counter()
    exclude = exclude or set()
    seen: set[str] = set()
    for job in sorted((store_dir / "jobs").glob("*")):
        found = load_corrections(job)
        if not found:
            if (job / "correction.json").exists() or (job / "corrections").is_dir():
                skipped["unreadable"] += 1
            continue
        data = approved_correction(job)
        if data is None:
            skipped["notApproved"] += 1
            continue
        raw_lines = data.get("lines") if isinstance(data.get("lines"), list) else []
        lines = [ln for ln in raw_lines if _well_formed(ln)]
        skipped["malformedLines"] += len(raw_lines) - len(lines)
        # A corrected MRZ must agree with its own check digits. One that does not is a typing slip or a planted label.
        mrz = [ln["text"] for ln in lines if _MRZ_SHAPED.fullmatch(ln["text"])]
        if len(mrz) in (2, 3) and not check_score(mrz)[1]:
            skipped["mrzLabelFailsCheckDigits"] += 1
            continue
        images = job_images(job)
        # The approver saw each crop and approved its pixel digest. A line whose crop was not shown, or whose pixels differ
        # from what was approved, is not used: the text alone says nothing about what the box covers.
        approved = data.get("approvedCrops") if isinstance(data.get("approvedCrops"), list) else []
        for n, line in enumerate(raw_lines):
            if line not in lines:
                continue
            if lang and line.get("lang") != lang:
                skipped["otherLanguage"] += 1
                continue
            crop, why = cut_line(job, line, images)
            if crop is None:
                skipped[why] += 1
                continue
            digest = _pixel_hash(crop)
            if n >= len(approved) or approved[n] != digest:
                skipped["cropNotApproved"] += 1
                continue
            if digest in exclude:
                skipped["inGoldenSet"] += 1
                continue
            if (digest, line["text"]) in seen:
                skipped["duplicate"] += 1
                continue
            seen.add((digest, line["text"]))
            rel = f"images/real_{job.name}_{n}.png"
            buf = io.BytesIO()
            crop.save(buf, "PNG")
            (out / rel).write_bytes(buf.getvalue())
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
    os.umask(0o077)      # line crops and labels are personal data: owner-only, like the training store
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
    ap.add_argument("--lang", help="use only corrected lines marked with this language (en or ar)")
    ap.add_argument("--golden-dir", type=Path, help="the golden set; any crop that is also in it is left out of training")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    (args.out / "images").mkdir(parents=True, exist_ok=True)

    real: list[dict] = []
    skipped: Counter = Counter()
    if args.store_dir:
        real += from_store(args.store_dir, args.out, args.lang, skipped, golden_hashes(args.golden_dir))
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
        "skipped": dict(skipped), "seed": args.seed, "lang": args.lang,
        "datasetSha256": hashlib.sha256(b"".join((args.out / n).read_bytes() for n in ("train.txt", "val_real.txt"))).hexdigest(),
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
