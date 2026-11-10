#!/usr/bin/env python3
"""Generate synthetic single-line training images (English and Arabic) for OCR recognition.

Use this to bootstrap training before you have enough real, labelled documents. Synthetic data
does not replace real data: mix it with real lines and always judge a model on REAL validation
lines only (prepare_dataset.py keeps them separate).

Arabic is shaped (letters joined) and put in visual order for rendering, while the label stays in
logical (typed) order, which is what recognition training expects.

    python training/synth_lines.py --out data/synth --count 5000 \
        --fonts /path/NotoNaskhArabic-Regular.ttf /path/DejaVuSans.ttf

Fonts must support the script you render. Supply your own fonts; none are downloaded.
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.textutil import has_arabic  # noqa: E402

AR_GIVEN = ["محمد", "أحمد", "عبدالله", "فاطمة", "نورة", "خالد", "سارة", "عمر", "ليلى", "يوسف",
            "مريم", "إبراهيم", "حسن", "علي", "هدى", "سلمان", "رنا", "طارق", "منى", "ياسر"]
AR_FAMILY = ["القحطاني", "الغامدي", "العتيبي", "الحربي", "المصري", "الشمري", "الزهراني", "السيد",
             "إبراهيم", "حسين", "سعد", "الدوسري", "العنزي", "الخطيب"]
EN_GIVEN = ["MOHAMMED", "AHMED", "ABDULLAH", "FATIMA", "NORA", "KHALED", "SARA", "OMAR", "LAYLA",
            "YOUSEF", "MARIAM", "IBRAHIM", "HASSAN", "ALI", "HUDA", "SALMAN", "RANA", "TAREK"]
EN_FAMILY = ["ALQAHTANI", "ALGHAMDI", "ALOTAIBI", "ALHARBI", "ELMASRY", "ALSHAMMARI", "ALZAHRANI",
             "ELSAYED", "HUSSEIN", "SAAD", "ALDOSARI", "ALANAZI", "ALKHATIB"]
AR_LABELS = ["الاسم", "الجنس", "تاريخ الميلاد", "تاريخ الانتهاء", "رقم الجواز", "الجنسية", "مكان الميلاد"]
EN_LABELS = ["Name", "Sex", "Date of birth", "Date of expiry", "Passport No.", "Nationality", "Place of birth"]
_INDIC = str.maketrans("0123456789", "٠١٢٣٤٥٦٧٨٩")


def make_text(rng: random.Random, indic_ratio: float) -> str:
    kind = rng.choice(["ar_name", "en_name", "date", "number", "ar_field", "en_field"])
    digits_fn = (lambda s: s.translate(_INDIC)) if rng.random() < indic_ratio else (lambda s: s)
    date = f"{rng.randint(1, 28):02d}/{rng.randint(1, 12):02d}/{rng.randint(1950, 2040)}"
    number = "".join(rng.choice("0123456789") for _ in range(rng.choice([8, 9, 10])))
    if kind == "ar_name":
        return f"{rng.choice(AR_GIVEN)} {rng.choice(AR_GIVEN)} {rng.choice(AR_FAMILY)}"
    if kind == "en_name":
        return f"{rng.choice(EN_GIVEN)} {rng.choice(EN_GIVEN)} {rng.choice(EN_FAMILY)}"
    if kind == "date":
        return digits_fn(date) if rng.random() < 0.7 else date.replace("/", "-")
    if kind == "number":
        return digits_fn(number)
    if kind == "ar_field":
        return f"{rng.choice(AR_LABELS)}: {rng.choice([digits_fn(date), rng.choice(AR_GIVEN), digits_fn(number)])}"
    return f"{rng.choice(EN_LABELS)}: {rng.choice([date, rng.choice(EN_GIVEN), number])}"


def shape(text: str) -> str:
    """Logical order text to what a renderer draws (joined letters, right-to-left order)."""
    if not has_arabic(text):
        return text
    import arabic_reshaper
    from bidi.algorithm import get_display
    return get_display(arabic_reshaper.reshape(text))


def render(text: str, fonts: list[str], rng: random.Random) -> Image.Image:
    font = ImageFont.truetype(rng.choice(fonts), rng.randint(28, 42))
    shown = shape(text)
    box = ImageDraw.Draw(Image.new("L", (1, 1))).textbbox((0, 0), shown, font=font)
    w, h = box[2] - box[0], box[3] - box[1]
    pad = rng.randint(6, 14)
    bg = rng.randint(190, 255)
    img = Image.new("L", (w + 2 * pad, h + 2 * pad), bg)
    ImageDraw.Draw(img).text((pad - box[0], pad - box[1]), shown, fill=rng.randint(0, 70), font=font)
    img = img.rotate(rng.uniform(-1.5, 1.5), expand=True, fillcolor=bg, resample=Image.BICUBIC)
    if rng.random() < 0.6:
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.1)))
    noise = Image.effect_noise(img.size, rng.uniform(4, 14))
    return Image.blend(img, noise, 0.08)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--count", type=int, default=2000)
    ap.add_argument("--fonts", nargs="+", required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--val-ratio", type=float, default=0.05)
    ap.add_argument("--indic-digit-ratio", type=float, default=0.2,
                    help="share of numbers/dates written with Arabic-Indic digits")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    (args.out / "images").mkdir(parents=True, exist_ok=True)
    train, val = [], []
    for i in range(args.count):
        text = make_text(rng, args.indic_digit_ratio)
        rel = f"images/{i:06d}.png"
        render(text, args.fonts, rng).save(args.out / rel)
        (val if rng.random() < args.val_ratio else train).append(f"{rel}\t{text}")
    (args.out / "train.txt").write_text("\n".join(train) + "\n", encoding="utf-8")
    (args.out / "val.txt").write_text("\n".join(val) + "\n", encoding="utf-8")
    print(f"wrote {len(train)} train and {len(val)} val lines to {args.out}")


if __name__ == "__main__":
    main()
