"""Baseline open-source engine: finds the MRZ with Tesseract.

This is the POC baseline, not the final engine. For better accuracy install the
OCR-B traineddata (ocrb.traineddata) into the Tesseract data directory and set
TESSERACT_LANG=ocrb. Everything here works offline.
"""
from __future__ import annotations

import io
import os
import time

from .. import config as app_config
from ..mrz import MrzError, find_mrz_lines, parse_mrz
from ..uploads import upscaled_size
from .base import EngineOutput, EngineUnavailable

_WHITELIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<"


class TesseractMrzEngine:
    name = "tesseract-mrz"

    def __init__(self) -> None:
        self.lang = os.environ.get("TESSERACT_LANG", "eng")

    def _deps(self):
        try:
            import pytesseract  # noqa: WPS433
            from PIL import Image, ImageOps  # noqa: WPS433
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise EngineUnavailable(f"Missing Python dependency: {exc.name}") from exc
        try:
            pytesseract.get_tesseract_version()
        except Exception as exc:  # noqa: BLE001
            raise EngineUnavailable("Tesseract binary not found on this host") from exc
        return pytesseract, Image, ImageOps

    @staticmethod
    def _score(lines: list[str]) -> tuple[int, bool]:
        """(check digits passing, all passing) for these MRZ lines. More passing digits means a more trustworthy reading."""
        try:
            checks = parse_mrz(lines).checks
        except MrzError:
            return -1, False
        return sum(checks.values()), all(checks.values())

    def extract(self, images: list[bytes], doc_type_hint: str | None) -> EngineOutput:
        pytesseract, Image, ImageOps = self._deps()
        config = f"--psm 6 -c tessedit_char_whitelist={_WHITELIST}"
        budget = float(os.environ.get("OCR_SEARCH_SECONDS", "8"))

        def read(img, top: float, width: int) -> list[str] | None:
            crop = ImageOps.autocontrast(img.crop((0, int(img.height * top), img.width, img.height)))
            size = upscaled_size(crop.width, crop.height, width)   # enlarge small captures, never beyond the pixel cap
            if size:
                crop = crop.resize(size)
            text = pytesseract.image_to_string(crop, lang=self.lang, config=config, timeout=app_config.ocr_timeout())
            return find_mrz_lines(text.splitlines())

        best: list[str] | None = None
        for raw in images:
            img = ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert("L")
            started = time.monotonic()
            best_score = -2
            # Plain image first, then small rotations (hand-held photos are rarely level), then a larger scale.
            for angle in (0, 2, -2, 4, -4, 6, -6):
                view = img if angle == 0 else img.rotate(angle, Image.BICUBIC, expand=True, fillcolor=255)
                for width in (1600, 2400):
                    if angle and width != 1600:
                        continue
                    for top in (0.65, 0.5, 0.0):
                        if time.monotonic() - started > budget:
                            break
                        lines = read(view, top, width)
                        if not lines:
                            continue
                        score, perfect = self._score(lines)
                        if score > best_score:
                            best, best_score = lines, score
                        if perfect:
                            return EngineOutput(engine=self.name, mrz_lines=lines)
            if best:
                return EngineOutput(engine=self.name, mrz_lines=best)
            # Nothing found upright: the page may be sideways or upside down (a scan fed in the wrong way round).
            # EXIF orientation does not cover that, so try the three other quarter-turns with a short search.
            started = time.monotonic()
            for quarter in (90, 270, 180):
                view = img.rotate(quarter, Image.BICUBIC, expand=True, fillcolor=255)
                for top in (0.65, 0.5, 0.0):
                    if time.monotonic() - started > budget:
                        break
                    lines = read(view, top, 1600)
                    if not lines:
                        continue
                    score, perfect = self._score(lines)
                    if score > best_score:
                        best, best_score = lines, score
                    if perfect:
                        return EngineOutput(engine=self.name, mrz_lines=lines)
                if best:
                    break
            if best:
                return EngineOutput(engine=self.name, mrz_lines=best)
        return EngineOutput(engine=self.name, mrz_lines=best)
