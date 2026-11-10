"""Tesseract and RapidOCR read the same page; agreement between two independent readers is the confidence signal.

* both give identical MRZ lines            -> mrz_confidence = AGREEMENT_CONFIDENCE (default 0.99), if they also pass check digits
* they differ                              -> the reading with more passing check digits wins, no confidence (a person checks)
* only one reader finds an MRZ             -> that reading, no confidence
* RapidOCR not installed                   -> Tesseract alone, no confidence (same as before)
Agreement is only trusted when every check digit passes: two engines can share a mistake that the check digits cannot see."""
from __future__ import annotations

import os

from ..mrz import MrzError, parse_mrz
from .base import EngineOutput, EngineUnavailable
from .rapid_ocr import RapidOcrEngine
from .tesseract_mrz import TesseractMrzEngine
from .tesseract_text import read_text_lines


def agreement_confidence() -> float:
    try:
        return float(os.environ.get("AGREEMENT_CONFIDENCE", "0.99"))
    except ValueError:
        return 0.99


def _score(lines):
    try:
        checks = parse_mrz(lines).checks
    except MrzError:
        return -1, False
    return sum(checks.values()), all(checks.values())


class EnsembleEngine:
    @staticmethod
    def _text_fields(images) -> dict:
        """No MRZ on the page (Iqama, GCC IDs): return Arabic + English text lines for the ID-card reader."""
        try:
            lines = read_text_lines(images)
        except EngineUnavailable:
            return {}
        return {"textLines": lines} if lines else {}

    name = "ensemble"

    def __init__(self) -> None:
        self.tesseract = TesseractMrzEngine()
        self.rapid = RapidOcrEngine()

    def extract(self, images, doc_type_hint):
        if doc_type_hint in ("iqama", "gcc_id"):                 # no MRZ on these cards: skip the slow MRZ search
            fields = self._text_fields(images)
            if fields:
                return EngineOutput(engine=self.name, mrz_lines=None, fields=fields)
        t = self.tesseract.extract(images, doc_type_hint)
        try:
            r = self.rapid.extract(images, doc_type_hint)
        except EngineUnavailable:
            if not t.mrz_lines:
                t.fields = self._text_fields(images)
            return t
        if t.mrz_lines and r.mrz_lines and t.mrz_lines == r.mrz_lines:
            score, perfect = _score(t.mrz_lines)
            conf = agreement_confidence() if perfect else None
            return EngineOutput(engine=self.name, mrz_lines=t.mrz_lines, mrz_confidence=conf)
        candidates = [x for x in (t, r) if x.mrz_lines]
        if not candidates:
            return EngineOutput(engine=self.name, mrz_lines=None, fields=self._text_fields(images))
        best = max(candidates, key=lambda x: _score(x.mrz_lines)[0])      # ties keep Tesseract (listed first)
        return EngineOutput(engine=self.name, mrz_lines=best.mrz_lines, mrz_confidence=None)
