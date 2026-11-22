"""Engine registry. Add the winning POC engine here (for example a Regula adapter)."""
from __future__ import annotations

from typing import Callable
import logging
import threading

from .. import timing
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
log = logging.getLogger("visa-ocr")
_INSTANCES: dict[str, Engine] = {}
_INSTANCES_LOCK = threading.Lock()


class AutoEngine:
    """Default: PaddleOCR (English + Arabic) when it and its verified models are available, otherwise the Tesseract +
    RapidOCR ensemble (Tesseract alone when RapidOCR is not installed)."""
    name = "auto"

    def extract(self, images, doc_type_hint):
        wants_mrz = doc_type_hint not in ("iqama", "gcc_id")
        with timing.step("quickLook"):
            seen: list = []
            quick = self._fast_first(images, doc_type_hint, seen) if wants_mrz else None
        if quick is not None:
            return quick
        try:
            paddle = get_engine_by_name(PaddleOcrEngine.name)
            # When an MRZ is expected, PaddleOCR's slow last-chance searches wait until the second pair has had its turn.
            staged = wants_mrz and hasattr(paddle, "mrz_retries")
            with timing.step("paddle"):
                out = paddle.extract(images, doc_type_hint, deep=False) if staged else paddle.extract(images, doc_type_hint)
        except EngineUnavailable:
            return get_engine_by_name(EnsembleEngine.name).extract(images, doc_type_hint)
        except Exception as exc:  # noqa: BLE001
            # PaddleOCR is installed and its models are in place, yet it failed (it could not start in this environment, or
            # broke on this image). "auto" promises a reading whenever any reader can give one, so the other pair reads it,
            # and the failure is logged so that it does not go unnoticed.
            log.error("PaddleOCR failed (%s); this document is read by Tesseract + RapidOCR", type(exc).__name__)
            return get_engine_by_name(EnsembleEngine.name).extract(images, doc_type_hint)
        # PaddleOCR reads whole pages. When it finds no MRZ on a document that should have one (a passport small inside a
        # full-page scan, a tilted or glary photo), the ensemble's strip search and rotation search often still succeeds.
        images = getattr(out, "upright_images", None) or images      # a photo taken sideways: the other readers get it upright
        has_text = bool(out.fields.get("textLines"))
        if wants_mrz and not out.mrz_lines or (not out.mrz_lines and not has_text):
            limited = False
            if has_text and _card_front(out) and not _mrz_trace(out):
                # A card front (a Gulf card, a residence-card front) with nothing on it that looks like an MRZ: the other
                # readers and the slow searches only look for an MRZ, and here they cost half a minute to find none.
                out.mrz_searched = True
                return out
            try:
                # The pair gets a time limit only when the page is already known to be a card front with no MRZ (a Gulf
                # card or a residence-card front): a full search there finds nothing and costs twenty seconds. On any other
                # page the pair searches in full: with a limit it reads less thoroughly, and that lost passports it could read.
                from .. import config
                limited = _card_front(out)
                with timing.step("secondPair"), config.limited_read(config.handover_seconds() if limited else 0):
                    second = get_engine_by_name(EnsembleEngine.name).extract(images, doc_type_hint)
            except EngineUnavailable:
                second = None
            if second is not None and second.mrz_lines:
                from ..mrz import second_mrz
                page = [str(l.get("text", "")) for l in out.fields.get("textLines") or []]
                if second_mrz(page, second.mrz_lines):       # PaddleOCR saw another MRZ on the page besides this one
                    second.fields = {**(second.fields or {}), "secondMrz": True}
                return second
            out.mrz_searched = second is not None and not limited      # the pair searched in full: no one need look again
            if staged:
                try:
                    # the slowest search (straightening the page) only when something on the page looks like an MRZ
                    trace = _mrz_trace(out) or (second is not None and _mrz_trace(second))
                    with timing.step("paddleMrzRetries"):
                        out.mrz_lines, out.mrz_confidence = paddle.mrz_retries(images, straighten=bool(trace))
                except Exception as exc:  # noqa: BLE001 - a failed last look must not lose the text already read
                    log.error("PaddleOCR's last MRZ search failed (%s)", type(exc).__name__)
            if second is not None and not out.mrz_lines and not has_text:
                return second
            if second is None:
                return out
        if out.mrz_lines:
            with timing.step("secondOpinion"):
                out = self._second_opinion(out, images, doc_type_hint, seen[0] if seen else None)
        return out

    @staticmethod
    def _fast_first(images, doc_type_hint, seen: list | None = None):
        """The quick answer for a clear document: the pair's reading, when both readers agree and every check digit passes.
        None in every other case (off, pair not installed, no MRZ, any doubt), and PaddleOCR then reads as usual."""
        from .. import config
        from ..mrz import check_score
        limit = config.fast_first_seconds()
        if limit <= 0:
            return None
        try:
            get_engine_by_name(PaddleOcrEngine.name)          # no PaddleOCR: the pair is the whole reader anyway, unlimited
            with config.quick_look(limit):
                first = get_engine_by_name(EnsembleEngine.name).extract(images, doc_type_hint)
        except Exception:  # noqa: BLE001 - the quick look is optional; any trouble means the normal path
            return None
        if seen is not None:
            seen.append(first)            # kept even when it is not the answer: it may confirm PaddleOCR's reading later
        # Full agreement only: the readers agreeing on line 2 alone leaves the name read once, and that needs the page.
        from .ensemble import _ALL_FIELDS
        agree = set(_ALL_FIELDS) <= set((first.fields or {}).get("agreedFields") or [])
        if first.mrz_lines and agree and check_score(first.mrz_lines)[1]:
            try:
                # a second document in the photo is for the whole-page read to see and report
                if get_engine_by_name(TesseractMrzEngine.name).second_zone(images, first.mrz_lines):
                    return None
            except Exception:  # noqa: BLE001 - the extra look is optional
                pass
            return first
        return None

    @staticmethod
    def _second_opinion(out, images, doc_type_hint, seen=None):
        """PaddleOCR often reads an MRZ correctly but scores it low (0.6 on a phone photo), which sends the whole document to
        review. When its confidence is below the review threshold, the Tesseract + RapidOCR pair reads it too: if they return the
        same lines and every check digit passes, two independent readers agree, and that agreement is the confidence. If the
        pair reads it better (more check digits pass), their reading is used."""
        from .. import config
        from ..mrz import check_score as score
        threshold = config.min_confidence_unprotected()
        if out.mrz_confidence is not None and out.mrz_confidence >= threshold:
            return out
        # The quick look at the start already read the strip with the other readers. The same lines from them, with every
        # check digit passing, is the agreement this step exists to find: no need to read the page again for it.
        if seen is not None and seen.mrz_lines and seen.mrz_lines == out.mrz_lines and score(out.mrz_lines)[1]:
            out.mrz_confidence = max(out.mrz_confidence or 0.0, config.agreement_confidence())
            return out

        try:
            with config.limited_read(config.handover_seconds()):
                second = get_engine_by_name(EnsembleEngine.name).extract(images, doc_type_hint)
        except EngineUnavailable:
            return out
        if not second.mrz_lines:
            return out
        from ..mrz import plausible_reading, same_zone
        if not same_zone(second.mrz_lines, out.mrz_lines) and plausible_reading(second.mrz_lines) and plausible_reading(out.mrz_lines):
            out.fields = {**(out.fields or {}), "secondMrz": True}      # the two readers each read a different document
            return out
        s_out, perfect_out = score(out.mrz_lines)
        s_sec, perfect_sec = score(second.mrz_lines)
        if second.mrz_lines == out.mrz_lines and perfect_out:
            out.mrz_confidence = max(out.mrz_confidence or 0.0, config.agreement_confidence())      # two independent readers agree
            return out
        if s_sec > s_out:
            return second
        return out


def _card_front(out) -> bool:
    """True when the text PaddleOCR read shows a card side that carries no MRZ: it names a Gulf state or carries a Gulf ID
    label, or it has the title of a residence card."""
    lines = (out.fields or {}).get("textLines") or []
    if not lines:
        return False
    from ..permits import kind_of
    from ..visual_zone import gulf_evidence
    named, labelled = gulf_evidence(lines)
    return bool(named or labelled or kind_of(lines))


def _mrz_trace(out) -> bool:
    """True when some line of the text read from the page looks like part of an MRZ (filler characters in a run of capitals
    and digits). A page with no such line is very unlikely to carry an MRZ the slow searches could still find."""
    import re
    for line in (out.fields or {}).get("textLines") or []:
        text = re.sub(r"\s+", "", str(line.get("text", ""))).upper()
        if len(text) >= 12 and text.count("<") >= 2 and re.fullmatch(r"[A-Z0-9<«‹]+", text):
            return True
    return False


def get_engine() -> Engine:
    """The engine chosen by OCR_ENGINE (default: auto = PaddleOCR if available, else Tesseract)."""
    from .. import config
    return get_engine_by_name(config.ocr_engine())


def get_engine_by_name(name: str) -> Engine:
    """One engine instance per name, so models are loaded once, not on every request."""
    if name not in _REGISTRY:
        raise EngineUnavailable(f"Unknown OCR_ENGINE: {name}")
    if name not in _INSTANCES:
        with _INSTANCES_LOCK:                      # two first requests must not build two sets of models
            if name not in _INSTANCES:
                _INSTANCES[name] = _REGISTRY[name]()
    return _INSTANCES[name]


__all__ = ["Engine", "EngineOutput", "EngineUnavailable", "get_engine", "get_engine_by_name"]
