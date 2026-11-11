"""Tesseract and RapidOCR read the same page; agreement between two independent readers is the confidence signal.

* both give identical MRZ lines            -> mrz_confidence = AGREEMENT_CONFIDENCE (default 0.99), if they also pass check digits
* they differ                              -> the reading with more passing check digits wins, no confidence (a person checks)
* only one reader finds an MRZ             -> that reading, no confidence
* RapidOCR not installed                   -> Tesseract alone, no confidence (same as before)
Agreement is only trusted when every check digit passes: two engines can share a mistake that the check digits cannot see."""
from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor

from .. import config as app_config

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


def _differences(a: list[str] | None, b: list[str] | None) -> list[dict]:
    """Where two MRZ readings differ: line, position, and the character each reader saw (first 12 only)."""
    out: list[dict] = []
    if not a or not b:
        return out
    if len(a) != len(b):
        return [{"lines": [len(a), len(b)]}]
    for n, (la, lb) in enumerate(zip(a, b), 1):
        if len(la) != len(lb):
            out.append({"line": n, "length": [len(la), len(lb)]})
        for i, (x, y) in enumerate(zip(la, lb), 1):
            if x != y and len(out) < 12:
                out.append({"line": n, "position": i, "tesseract": x, "rapidocr": y})
    return out


class EnsembleEngine:
    @staticmethod
    def _text_fields(images, started: float | None = None) -> dict:
        """No MRZ on the page (Iqama, GCC IDs): return Arabic + English text lines for the ID-card reader. `started` is when
        this job's reading began: with a hard limit set, the text read only gets what the MRZ search left of it."""
        left = None
        if started is not None and app_config.ocr_deadline():
            left = max(0.0, app_config.ocr_deadline() - (time.monotonic() - started))
        try:
            lines = read_text_lines(images, budget_seconds=left)
        except EngineUnavailable:
            return {}
        return {"textLines": lines} if lines else {}

    name = "ensemble"

    def __init__(self) -> None:
        self.tesseract = TesseractMrzEngine()
        self.rapid = RapidOcrEngine()

    def extract(self, images, doc_type_hint):
        started = time.monotonic()
        if doc_type_hint in ("iqama", "gcc_id"):                 # no MRZ on these cards: skip the slow MRZ search
            fields = self._text_fields(images)
            if fields:
                return EngineOutput(engine=self.name, mrz_lines=None, fields=fields)
        # The two readers run side by side: Tesseract is a separate process and RapidOCR releases the interpreter lock, so on two
        # or more cores the wait is the slower of the two, not the sum.
        took: dict[str, float] = {}

        def timed(name, engine):
            t0 = time.monotonic()
            try:
                return engine.extract(images, doc_type_hint)
            finally:
                took[name] = round(time.monotonic() - t0, 2)

        with ThreadPoolExecutor(max_workers=2) as pool:
            ft = pool.submit(timed, "tesseract", self.tesseract)
            fr = pool.submit(timed, "rapidocr", self.rapid)
            t = ft.result()
            try:
                r = fr.result()
            except EngineUnavailable:
                r = None
        if r is None:
            if not t.mrz_lines:
                t.fields = self._text_fields(images, started)
            return t
        # Why a result was or was not confirmed, for the reviewer and for support: which reader found the MRZ, how long each
        # took, and where the two readings differ.
        readers = {"tesseract": bool(t.mrz_lines), "rapidocr": bool(r.mrz_lines), "seconds": took,
                   "agree": bool(t.mrz_lines) and t.mrz_lines == r.mrz_lines, "differences": _differences(t.mrz_lines, r.mrz_lines)}
        if t.mrz_lines and r.mrz_lines and t.mrz_lines == r.mrz_lines:
            score, perfect = _score(t.mrz_lines)
            conf = agreement_confidence() if perfect else None
            return EngineOutput(engine=self.name, mrz_lines=t.mrz_lines, mrz_confidence=conf, fields={**r.fields, "readers": readers})
        candidates = [x for x in (t, r) if x.mrz_lines]
        if not candidates:
            return EngineOutput(engine=self.name, mrz_lines=None, fields=self._text_fields(images, started))
        best = max(candidates, key=lambda x: _score(x.mrz_lines)[0])      # ties keep Tesseract (listed first)
        return EngineOutput(engine=self.name, mrz_lines=best.mrz_lines, mrz_confidence=None,
                            fields={**(r.fields if r.mrz_lines else {}), "readers": readers})   # line positions, for corrections
