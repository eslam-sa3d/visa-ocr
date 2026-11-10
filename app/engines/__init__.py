"""Engine registry. Add the winning POC engine here (for example a Regula adapter)."""
from __future__ import annotations

from typing import Callable
import os

from .base import Engine, EngineOutput, EngineUnavailable
from .ensemble import EnsembleEngine
from .paddle_ocr import PaddleOcrEngine
from .rapid_ocr import RapidOcrEngine
from .tesseract_mrz import TesseractMrzEngine

_REGISTRY: dict[str, Callable[[], Engine]] = {
    TesseractMrzEngine.name: TesseractMrzEngine,
    PaddleOcrEngine.name: PaddleOcrEngine,  # trainable English + Arabic text reader
    RapidOcrEngine.name: RapidOcrEngine,    # second open-source reader with scores (models ship inside the pip wheel)
    EnsembleEngine.name: EnsembleEngine,    # Tesseract + RapidOCR; agreement gives a confidence
    "auto": lambda: AutoEngine(),
    # "regula": RegulaEngine,        # on-prem Docker Web API adapter (after POC)
}
_INSTANCES: dict[str, Engine] = {}


class AutoEngine:
    """Default: PaddleOCR (English + Arabic) when it and its verified models are available, otherwise the Tesseract +
    RapidOCR ensemble (Tesseract alone when RapidOCR is not installed)."""
    name = "auto"

    def extract(self, images, doc_type_hint):
        try:
            out = get_engine_by_name(PaddleOcrEngine.name).extract(images, doc_type_hint)
        except EngineUnavailable:
            return get_engine_by_name(EnsembleEngine.name).extract(images, doc_type_hint)
        # PaddleOCR reads whole pages. When it finds no MRZ on a document that should have one (a passport small inside a
        # full-page scan, a tilted or glary photo), the ensemble's strip search and rotation search often still succeeds.
        has_text = bool(out.fields.get("textLines"))
        wants_mrz = doc_type_hint not in ("iqama", "gcc_id")
        if wants_mrz and not out.mrz_lines or (not out.mrz_lines and not has_text):
            try:
                second = get_engine_by_name(EnsembleEngine.name).extract(images, doc_type_hint)
            except EngineUnavailable:
                return out
            if second.mrz_lines or not has_text:
                return second
        if out.mrz_lines:
            out = self._second_opinion(out, images, doc_type_hint)
        return out

    @staticmethod
    def _second_opinion(out, images, doc_type_hint):
        """PaddleOCR often reads an MRZ correctly but scores it low (0.6 on a phone photo), which sends the whole document to
        review. When its confidence is below the review threshold, the Tesseract + RapidOCR pair reads it too: if they return the
        same lines and every check digit passes, two independent readers agree, and that agreement is the confidence. If the
        pair reads it better (more check digits pass), their reading is used."""
        from .. import config
        from ..mrz import MrzError, parse_mrz
        threshold = config.min_confidence_unprotected()
        if out.mrz_confidence is not None and out.mrz_confidence >= threshold:
            return out

        def score(lines):
            try:
                checks = parse_mrz(lines).checks
            except MrzError:
                return -1, False
            return sum(checks.values()), all(checks.values())

        try:
            second = get_engine_by_name(EnsembleEngine.name).extract(images, doc_type_hint)
        except EngineUnavailable:
            return out
        if not second.mrz_lines:
            return out
        s_out, perfect_out = score(out.mrz_lines)
        s_sec, perfect_sec = score(second.mrz_lines)
        if second.mrz_lines == out.mrz_lines and perfect_out:
            from .ensemble import agreement_confidence
            out.mrz_confidence = max(out.mrz_confidence or 0.0, agreement_confidence())      # two independent readers agree
            return out
        if s_sec > s_out:
            return second
        return out


def get_engine() -> Engine:
    """The engine chosen by OCR_ENGINE (default: auto = PaddleOCR if available, else Tesseract)."""
    return get_engine_by_name(os.environ.get("OCR_ENGINE", "auto"))


def get_engine_by_name(name: str) -> Engine:
    """One engine instance per name, so models are loaded once, not on every request."""
    if name not in _REGISTRY:
        raise EngineUnavailable(f"Unknown OCR_ENGINE: {name}")
    if name not in _INSTANCES:
        _INSTANCES[name] = _REGISTRY[name]()
    return _INSTANCES[name]


__all__ = ["Engine", "EngineOutput", "EngineUnavailable", "get_engine", "get_engine_by_name"]
