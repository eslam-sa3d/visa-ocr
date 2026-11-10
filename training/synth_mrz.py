#!/usr/bin/env python3
"""Generate synthetic MRZ line images for fine-tuning the English recognition model (checklist B3).

Lines are made by the same builder the accuracy tests use, so they carry correct check digits, many countries, long
document numbers and every format (TD1, TD2, TD3, MRV-A, MRV-B). Each image is ONE line; the label is its text.

    python training/synth_mrz.py --out data/synth_mrz --count 20000 \
        --fonts /path/OCRB.ttf /usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf

Use a real OCR-B font whenever you have one (the machine-readable zone of every passport is printed in OCR-B); without
it the model learns the shapes of a different font. Synthetic lines do not replace real ones: mix them with
`prepare_dataset.py --synth-dir` (capped at 3x the real lines) and always judge the model on REAL validation lines.
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import accuracy_matrix as A  # noqa: E402

DEFAULT_FONTS = ["/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", "/usr/share/fonts/truetype/freefont/FreeMono.ttf",
                 "/usr/share/texmf/fonts/opentype/public/lm/lmmono10-regular.otf"]


def render(text: str, fonts: list[str], rng: random.Random) -> Image.Image:
    font = ImageFont.truetype(rng.choice(fonts), rng.randint(30, 46))
    box = ImageDraw.Draw(Image.new("L", (1, 1))).textbbox((0, 0), text, font=font)
    w, h = box[2] - box[0], box[3] - box[1]
    pad = rng.randint(6, 16)
    bg = rng.randint(185, 255)
    img = Image.new("L", (w + 2 * pad, h + 2 * pad), bg)
    ImageDraw.Draw(img).text((pad - box[0], pad - box[1]), text, fill=rng.randint(0, 80), font=font)
    img = img.rotate(rng.uniform(-2.5, 2.5), expand=True, fillcolor=bg, resample=Image.BICUBIC)
    if rng.random() < 0.3:                                                 # horizontal shear, like an angled phone photo
        k = rng.uniform(-0.12, 0.12)
        img = img.transform(img.size, Image.AFFINE, (1, k, -k * img.height / 2, 0, 1, 0), resample=Image.BICUBIC, fillcolor=bg)
    if rng.random() < 0.6:
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.4)))
    if rng.random() < 0.5:
        noise = Image.effect_noise(img.size, rng.uniform(4, 18))
        img = Image.blend(img, noise, rng.uniform(0.05, 0.2))
    if rng.random() < 0.4:                                                 # low contrast / glare
        img = Image.blend(img, Image.new("L", img.size, rng.randint(120, 200)), rng.uniform(0.15, 0.5))
    if rng.random() < 0.3:                                                 # low resolution
        f = rng.uniform(0.45, 0.8)
        img = img.resize((max(8, int(img.width * f)), max(8, int(img.height * f))), Image.BILINEAR)
    return img


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--count", type=int, default=5000)
    ap.add_argument("--fonts", nargs="*", default=[f for f in DEFAULT_FONTS if Path(f).exists()])
    ap.add_argument("--val-ratio", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    if not a.fonts:
        sys.exit("no usable font: pass --fonts (ideally an OCR-B font)")
    rng = random.Random(a.seed)
    (a.out / "images").mkdir(parents=True, exist_ok=True)
    train, val, n = [], [], 0
    formats = ["TD3", "TD3", "TD2", "TD1", "MRVA", "MRVB"]
    while n < a.count:
        fmt = rng.choice(formats)
        lines, _ = A.build(fmt, **A.rnd_case(rng, fmt))
        for line in lines:
            if n >= a.count:
                break
            name = f"images/mrz_{n:06d}.png"
            render(line, a.fonts, rng).save(a.out / name)
            (val if rng.random() < a.val_ratio else train).append(f"{name}\t{line}")
            n += 1
    (a.out / "train.txt").write_text("\n".join(train) + "\n", encoding="utf-8")
    (a.out / "val.txt").write_text("\n".join(val) + "\n", encoding="utf-8")
    print(f"{n} MRZ line images -> {a.out} (train {len(train)}, val {len(val)}); fonts: {[Path(f).name for f in a.fonts]}")


if __name__ == "__main__":
    main()
