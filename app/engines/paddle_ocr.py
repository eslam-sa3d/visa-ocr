"""Trainable text-reading engine (PaddleOCR) for English and Arabic.

Models are loaded only from the local models directory (see app/model_registry.py), never
downloaded, so it works with no internet. Fine-tuned models replace the pretrained ones by
being registered and marked active; no code change is needed.

What it returns
  - every text line found, with language, script, confidence and box
  - the MRZ lines if one is among them (so the MRZ parser and validation still apply)
Mapping lines to document fields (name, place of birth, ...) is a later step that needs
document templates or a layout model.
"""
from __future__ import annotations

import io
import os
import re
import threading

# Offline by construction: never let PaddleX check or fetch models from the internet.
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from .. import config
from ..mrz import find_mrz_lines, lines_confidence
from ..uploads import upscaled_size
from ..model_registry import Registry
from ..textutil import dedupe_overlapping, script_of, to_ascii_digits
from .base import EngineOutput, EngineUnavailable


def _result_dict(res) -> dict:
    """PaddleOCR 3.x results behave like dicts; some versions wrap them under .json['res']."""
    if isinstance(res, dict):
        return res.get("res", res)
    payload = getattr(res, "json", None)
    if isinstance(payload, dict):
        return payload.get("res", payload)
    return {}


def _mrz_score(found, texts, scores) -> float | None:
    """Lowest recognition score over the detected texts the returned MRZ lines came from; None when a line was padded or
    adjusted (see mrz.lines_confidence). A short, low-score line 1 must not ride on the score of a clean line 2."""
    return lines_confidence(found, [str(t) for t in texts], list(scores))


class PaddleOcrEngine:
    name = "paddle-ocr"

    def __init__(self, registry: Registry | None = None, langs: list[str] | None = None) -> None:
        self.registry = registry or Registry()
        self.langs = langs or [l.strip() for l in os.environ.get("OCR_LANGS", "en,ar").split(",") if l.strip()]
        self._readers: dict[str, object] = {}
        self._stamp: tuple = self._manifest_stamp()
        self._used_models: dict[str, dict] = {}
        self._init_lock = threading.Lock()                     # one model load at a time
        self._infer = threading.Semaphore(config.ocr_concurrency())  # Paddle predictors are not re-entrant

    # -- loading -----------------------------------------------------------------------
    def _paddle_class(self):
        try:
            from paddleocr import PaddleOCR  # noqa: WPS433
        except ImportError as exc:
            raise EngineUnavailable("paddleocr is not installed on this host") from exc
        return PaddleOCR

    def _checked(self, task: str, lang: str):
        m = self.registry.active(task, lang)
        if m is None:
            raise EngineUnavailable(f"No active {task} model registered for language '{lang}'")
        status = self.registry.status(m)
        if status in ("missing", "mismatch"):
            raise EngineUnavailable(f"Model {m.id} is {status}")
        return m

    def _manifest_stamp(self) -> tuple:
        try:
            st = self.registry.manifest_path.stat()
            return (st.st_mtime_ns, st.st_size)
        except OSError:
            return ()

    def _reader(self, lang: str):
        # A model switch (register_model.py --activate, or automatic retraining) rewrites the manifest. Notice that and load
        # the newly active models on the next read, so the workers do not need a restart.
        stamp = self._manifest_stamp()
        if stamp != self._stamp:
            with self._init_lock:
                if stamp != self._stamp:
                    self._readers, self._used_models, self._stamp = {}, {}, stamp
        if lang in self._readers:
            return self._readers[lang]
        with self._init_lock:
            if lang not in self._readers:
                self._readers[lang] = self._build_reader(lang)
        return self._readers[lang]

    def _build_reader(self, lang: str):
        det = self._checked("text_detection", "any")
        rec = self._checked("text_recognition", lang)
        PaddleOCR = self._paddle_class()
        reader = PaddleOCR(
            text_detection_model_name=det.architecture(),
            text_detection_model_dir=str(self.registry.full_path(det)),
            text_recognition_model_name=rec.architecture(),
            text_recognition_model_dir=str(self.registry.full_path(rec)),
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            enable_mkldnn=False,  # oneDNN breaks some x86 CPU builds; speed is fine without it
        )
        self._used_models[lang] = {"id": rec.id, "version": rec.version, "detector": det.id}
        return reader

    def _mrz_from_bottom_band(self, images, reader, np, Image, ImageOps):
        """Second chance: the MRZ is always the bottom band, so crop it, enlarge it and read only that."""
        for raw in images:
            img = ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert("RGB")
            w, h = img.size
            for frac in (0.30, 0.45):
                band = img.crop((0, int(h * (1 - frac)), w, h))
                size = upscaled_size(band.width, band.height, 1800)
                if size:
                    band = band.resize(size, Image.LANCZOS)
                texts, scores = [], []
                for res in reader.predict(np.array(band)):
                    d = _result_dict(res)
                    texts += [str(t) for t in d.get("rec_texts", [])]
                    scores += [float(x) for x in d.get("rec_scores", [])]
                found = find_mrz_lines(texts)
                if found:
                    return found, _mrz_score(found, texts, scores)
        return None, None

    def _mrz_from_variants(self, images, reader, np, Image, ImageOps):
        """Last chance for angled photos: straighten the page, then read the whole page and its bottom band."""
        from .. import preprocess
        for raw in images:
            img = ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert("RGB")
            for arr in preprocess.variants(np.array(img)):
                fixed = Image.fromarray(arr)
                for frac in (1.0, 0.4):
                    h = fixed.height
                    part = fixed.crop((0, int(h * (1 - frac)), fixed.width, h))
                    size = upscaled_size(part.width, part.height, 1800)
                    if size:
                        part = part.resize(size, Image.LANCZOS)
                    texts, scores = [], []
                    for res in reader.predict(np.array(part)):
                        d = _result_dict(res)
                        texts += [str(t) for t in d.get("rec_texts", [])]
                        scores += [float(x) for x in d.get("rec_scores", [])]
                    found = find_mrz_lines(texts)
                    if found:
                        return found, _mrz_score(found, texts, scores)
        return None, None

    # -- extraction --------------------------------------------------------------------
    def extract(self, images: list[bytes], doc_type_hint: str | None) -> EngineOutput:
        try:
            import numpy as np  # noqa: WPS433
            from PIL import Image, ImageOps  # noqa: WPS433
        except ImportError as exc:
            raise EngineUnavailable(f"Missing Python dependency: {exc.name}") from exc

        with self._infer:
            return self._extract(images, np, Image, ImageOps)

    def _extract(self, images, np, Image, ImageOps) -> EngineOutput:
        readers = {lang: self._reader(lang) for lang in self.langs}
        lines: list[dict] = []
        for index, raw in enumerate(images):
            img = ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert("RGB")
            array = np.array(img)
            for lang, reader in readers.items():
                for res in reader.predict(array):
                    data = _result_dict(res)
                    texts = data.get("rec_texts", [])
                    scores = data.get("rec_scores", [])
                    boxes = data.get("rec_polys", data.get("dt_polys", []))
                    for i, text in enumerate(texts):
                        if not str(text).strip():
                            continue
                        box = boxes[i] if i < len(boxes) else []
                        lines.append({
                            "image": index,
                            "lang": lang,
                            "text": str(text),
                            "normalizedText": to_ascii_digits(str(text)),
                            "script": script_of(str(text)),
                            "score": round(float(scores[i]), 4) if i < len(scores) else None,
                            "box": [[int(x), int(y)] for x, y in (box.tolist() if hasattr(box, "tolist") else box)],
                        })

        mrz = find_mrz_lines([l["text"] for l in lines if l["lang"] == "en"]) or \
            find_mrz_lines([l["text"] for l in lines])
        mrz_conf = None
        if mrz:
            en = [l for l in lines if l["lang"] == "en"] or lines
            mrz_conf = _mrz_score(mrz, [l["text"] for l in lines], [l["score"] or 0 for l in lines])
        elif "en" in readers:
            mrz, mrz_conf = self._mrz_from_bottom_band(images, readers["en"], np, Image, ImageOps)
            if not mrz:
                mrz, mrz_conf = self._mrz_from_variants(images, readers["en"], np, Image, ImageOps)
        # every language reader reads every box; keep the best reading per box (per image)
        lines = [l for i in range(len(images)) for l in dedupe_overlapping([x for x in lines if x["image"] == i])]
        scores = [l["score"] for l in lines if l["score"] is not None]
        return EngineOutput(
            engine=self.name,
            mrz_lines=mrz,
            mrz_confidence=mrz_conf,
            fields={"textLines": lines, "models": self._used_models},
            confidence=round(sum(scores) / len(scores), 4) if scores else None,
        )
