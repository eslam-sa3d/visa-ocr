#!/usr/bin/env python3
"""Run a recognition model over the images in a label file and write predictions for evaluate.py.

Needs PaddleOCR on the training machine. The model is read from a local folder only.

    python training/predict_lines.py --labels data/rec/val_real.txt --image-dir data/rec \
        --model-name arabic_PP-OCRv5_mobile_rec --model-dir models/ar_rec_ft_2026-12-01 \
        --out preds.txt

Check the TextRecognition API against the PaddleOCR version you installed; names can change between releases.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path


def main() -> None:
    os.umask(0o077)      # predictions repeat text read from identity documents
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--image-dir", type=Path, required=True)
    ap.add_argument("--model-name", required=True, help="PaddleOCR architecture name")
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    from paddleocr import TextRecognition  # imported late so --help works without PaddleOCR

    model = TextRecognition(model_name=args.model_name, model_dir=args.model_dir)
    rows = []
    for line in args.labels.read_text(encoding="utf-8").splitlines():
        if "\t" not in line:
            continue
        rel = line.split("\t", 1)[0]
        text = ""
        for res in model.predict(input=str(args.image_dir / rel), batch_size=1):
            data = res if isinstance(res, dict) else getattr(res, "json", {}).get("res", {})
            text = str(data.get("rec_text", ""))
        rows.append(f"{rel}\t{text}")
    args.out.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"wrote {len(rows)} predictions to {args.out}")


if __name__ == "__main__":
    main()
