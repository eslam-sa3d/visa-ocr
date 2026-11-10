#!/usr/bin/env python3
"""ONE-TIME step on a machine WITH internet: download the pretrained PaddleOCR models, copy them into
models/ and record their SHA-256 in models/manifest.json. After this the service runs fully offline;
copy the whole models/ folder to the air-gapped server.

    python tools/fetch_models.py
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app.model_registry import dir_sha256  # noqa: E402

NAMES = {"PP-OCRv5_mobile_det": "TextDetection",
         "en_PP-OCRv5_mobile_rec": "TextRecognition",
         "arabic_PP-OCRv5_mobile_rec": "TextRecognition"}


def main() -> None:
    import paddleocr
    manifest_path = ROOT / "models" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cache = Path.home() / ".paddlex" / "official_models"
    for name, cls in NAMES.items():
        print(f"downloading {name} ...")
        getattr(paddleocr, cls)(model_name=name)  # downloads into the PaddleX cache
        src = cache / name
        if not src.is_dir():
            sys.exit(f"expected {src} after download; check PaddleOCR's model cache location")
        dst = ROOT / "models" / name
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        for m in manifest["models"]:
            if m["id"] == name:
                m["sha256"] = dir_sha256(dst)
        print(f"  copied to {dst}")
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("done. Start the service with: OCR_ENGINE=paddle-ocr uvicorn app.main:app --port 8080")


if __name__ == "__main__":
    main()
