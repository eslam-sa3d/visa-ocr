"""Registry of OCR models available on this host.

Everything is local: models live in MODELS_DIR (default ./models) and are described by
models/manifest.json. Fine-tuned models are added with training/register_model.py, which
records a checksum and the evaluation metrics. The service refuses to load a model whose
files no longer match the recorded checksum.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path


def default_models_dir() -> Path:
    return Path(os.environ.get("MODELS_DIR", Path(__file__).resolve().parent.parent / "models"))


def dir_sha256(path: Path) -> str:
    """Hash of a model directory: file names and contents, in sorted order."""
    h = hashlib.sha256()
    for f in sorted(p for p in Path(path).rglob("*") if p.is_file()):
        h.update(str(f.relative_to(path)).encode())
        with open(f, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
    return h.hexdigest()


@dataclass
class ModelInfo:
    id: str                      # unique id, for example ar_rec_ft_2026-12-01
    task: str                    # text_detection | text_recognition
    lang: str                    # en | ar | any
    version: str                 # pretrained | fine-tune version label
    path: str                    # folder under the models dir
    arch: str = ""               # PaddleOCR model name the weights belong to
    sha256: str | None = None
    active: bool = False
    source: str = ""
    metrics: dict = field(default_factory=dict)

    def architecture(self) -> str:
        return self.arch or self.id


class Registry:
    def __init__(self, base: Path | str | None = None) -> None:
        self.base = Path(base) if base else default_models_dir()
        self.manifest_path = self.base / "manifest.json"

    def models(self) -> list[ModelInfo]:
        if not self.manifest_path.exists():
            return []
        data = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        return [ModelInfo(**m) for m in data.get("models", [])]

    def save(self, models: list[ModelInfo]) -> None:
        self.base.mkdir(parents=True, exist_ok=True)
        payload = {"models": [asdict(m) for m in models]}
        self.manifest_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    def active(self, task: str, lang: str) -> ModelInfo | None:
        return next((m for m in self.models() if m.active and m.task == task and m.lang == lang), None)

    def full_path(self, m: ModelInfo) -> Path:
        return self.base / m.path

    def status(self, m: ModelInfo) -> str:
        """missing | unverified (no checksum recorded) | verified | mismatch."""
        p = self.full_path(m)
        if not p.is_dir():
            return "missing"
        if not m.sha256:
            return "unverified"
        return "verified" if dir_sha256(p) == m.sha256 else "mismatch"
