"""Second open-source reader (RapidOCR, PP-OCR models packaged inside the pip wheel, so it works offline).

It gives a recognition score per line, which Tesseract does not. It is used by the ensemble engine, which compares its
reading with Tesseract's. Optional: without the package the ensemble simply uses Tesseract alone."""
from __future__ import annotations

import io
import re
import threading

from ..mrz import MrzError, find_mrz_lines, parse_mrz
from ..uploads import upscaled_size
from .base import EngineOutput, EngineUnavailable

_FILLER_LIKE = str.maketrans({">": "<", "«": "<", "‹": "<", "＜": "<", "《": "<", " ": "", "\t": ""})


def normalise(text: str) -> str:
    return str(text).upper().translate(_FILLER_LIKE)


class RapidOcrEngine:
    name = "rapid-ocr"

    def __init__(self) -> None:
        self._reader = None
        self._lock = threading.Lock()                  # the ONNX session is shared; read one image at a time

    def _load(self):
        if self._reader is None:
            try:
                from rapidocr_onnxruntime import RapidOCR  # noqa: WPS433
            except ImportError as exc:
                raise EngineUnavailable("rapidocr-onnxruntime is not installed") from exc
            self._reader = RapidOCR()
        return self._reader

    @staticmethod
    def _score(lines: list[str]) -> tuple[int, bool]:
        try:
            checks = parse_mrz(lines).checks
        except MrzError:
            return -1, False
        return sum(checks.values()), all(checks.values())

    def _read(self, reader, image, np) -> tuple[list[str], list[float]]:
        result, _ = reader(np.array(image))
        texts = [normalise(r[1]) for r in (result or [])]
        scores = [float(r[2]) for r in (result or [])]
        return texts, scores

    def extract(self, images: list[bytes], doc_type_hint: str | None) -> EngineOutput:
        try:
            import numpy as np  # noqa: WPS433
            from PIL import Image, ImageOps  # noqa: WPS433
        except ImportError as exc:
            raise EngineUnavailable(f"Missing Python dependency: {exc.name}") from exc
        reader = self._load()
        best, best_conf, best_score = None, None, -2
        with self._lock:
            for raw in images:
                img = ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert("RGB")
                for angle in (0, 3, -3, 5, -5):
                    view = img if angle == 0 else img.rotate(angle, Image.BICUBIC, expand=True, fillcolor=(255, 255, 255))
                    for frac in (1.0, 0.45):
                        part = view.crop((0, int(view.height * (1 - frac)), view.width, view.height))
                        size = upscaled_size(part.width, part.height, 1800)
                        if size and size[0] > part.width:
                            part = part.resize(size, Image.LANCZOS)
                        texts, scores = self._read(reader, part, np)
                        lines = find_mrz_lines(texts)
                        if not lines:
                            continue
                        used = [s for t, s in zip(texts, scores) if len(t) >= 25 and re.fullmatch(r"[A-Z0-9<]+", t)]
                        score, perfect = self._score(lines)
                        if score > best_score:
                            best, best_score = lines, score
                            best_conf = round(min(used), 4) if used else None
                        if perfect:
                            return EngineOutput(engine=self.name, mrz_lines=lines, mrz_confidence=best_conf)
        return EngineOutput(engine=self.name, mrz_lines=best, mrz_confidence=best_conf)
