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
from .mrz_band import enlarge, find_bands
from .base import EngineOutput, EngineUnavailable

_WHITELIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<"


def _estimate_tilt(img, Image) -> int:
    """Whole degrees (-6..6) to rotate the page so its text rows are level. Level text gives the sharpest row profile: rows of ink
    and rows of blank paper differ most. Runs on a small copy, so it costs a few milliseconds."""
    try:
        import numpy as np  # noqa: WPS433
    except ImportError:
        return 0
    w = 480
    small = img.resize((w, max(1, round(img.height * w / img.width))), Image.BILINEAR)
    best, best_score = 0, -1.0
    for angle in range(-6, 7):
        view = small if angle == 0 else small.rotate(angle, Image.BILINEAR, fillcolor=255)
        rows = (255 - np.asarray(view, dtype=np.float32)).sum(axis=1)
        score = float(((rows[1:] - rows[:-1]) ** 2).sum())
        if score > best_score * (1.03 if angle != 0 else 1.0):      # prefer level unless another angle is clearly sharper
            if angle == 0 or score > best_score * 1.03:
                best, best_score = angle, score
    return best


class TesseractMrzEngine:
    name = "tesseract-mrz"

    def __init__(self) -> None:
        # An OCR-B model reads MRZ print far better than the general English one. It is used when TESSERACT_LANG names it, or
        # when MODELS_DIR/tessdata holds ocrb_int.traineddata or ocrb.traineddata (see docs/MODEL_TRAINING.md).
        self.lang = os.environ.get("TESSERACT_LANG", "").strip()
        self.data_dir = os.environ.get("TESSERACT_DATA_DIR", "").strip()
        if not self.lang:
            self.lang = "eng"
            from ..model_registry import default_models_dir as models_dir
            local = models_dir() / "tessdata"
            for name in ("ocrb_int", "ocrb"):
                if (local / f"{name}.traineddata").is_file():
                    self.lang, self.data_dir = name, str(local)
                    break

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
        if self.data_dir:
            config += f" --tessdata-dir {self.data_dir}"
        budget = float(os.environ.get("OCR_SEARCH_SECONDS", "8"))
        deadline = app_config.ocr_deadline()
        if deadline:
            budget = min(budget, max(0.3, deadline - 0.5))            # one attempt takes about 0.2 to 0.4 s

        def read(img, top: float, width: int) -> list[str] | None:
            crop = ImageOps.autocontrast(img.crop((0, int(img.height * top), img.width, img.height)))
            size = upscaled_size(crop.width, crop.height, width)   # enlarge small captures, never beyond the pixel cap
            if size:
                crop = crop.resize(size)
            text = pytesseract.image_to_string(crop, lang=self.lang, config=config, timeout=app_config.ocr_timeout())
            return find_mrz_lines(text.splitlines())

        def read_strip(strip) -> list[str] | None:
            text = pytesseract.image_to_string(ImageOps.autocontrast(strip), lang=self.lang, config=config,
                                               timeout=app_config.ocr_timeout())
            return find_mrz_lines(text.splitlines())

        best: list[str] | None = None
        for raw in images:
            img = ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert("L")
            started = time.monotonic()
            best_score = -2
            # First the MRZ strip alone, located by its shape: small, level and enlarged, in any page orientation.
            for band in find_bands(img):
                for flip in (0, 180):
                    if time.monotonic() - started > budget:
                        break
                    turned = band.image if not flip else band.image.rotate(180)
                    got = False
                    for width in (1800, 1600):              # a character misread at one size is often right at the other
                        lines = read_strip(enlarge(turned, width))
                        if not lines:
                            continue
                        got = True
                        score, perfect = self._score(lines)
                        if score > best_score:
                            best, best_score = lines, score
                        if perfect:
                            return EngineOutput(engine=self.name, mrz_lines=lines)
                    if got:
                        break                               # read, not perfect: the other way up will not be better
            # Plain image first, then small rotations (hand-held photos are rarely level), then a larger scale.
            # The tilt is estimated first so the likeliest angle is tried first instead of walking through all of them.
            tilt = _estimate_tilt(img, Image)
            angles = [0, 2, -2, 4, -4, 6, -6]
            if abs(tilt) >= 2:
                angles = [tilt] + sorted((a for a in angles if a != tilt), key=lambda a: abs(a - tilt))
            seen: dict[tuple, int] = {}
            for angle in angles:
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
                        # The same lines from two different crops or scales is a stable reading: the page is level and a
                        # failing check digit is a look-alike character (O/0), which rotating will not fix. Stop searching.
                        seen[tuple(lines)] = seen.get(tuple(lines), 0) + 1
                        if seen[tuple(lines)] >= 2 and score >= 0:
                            return EngineOutput(engine=self.name, mrz_lines=best)
            if best:
                return EngineOutput(engine=self.name, mrz_lines=best)
            # Nothing found upright: the page may be sideways or upside down (a scan fed in the wrong way round).
            # EXIF orientation does not cover that, so try the three other quarter-turns with a short search.
            if deadline:
                continue                                              # no time left for the sideways search under a hard limit
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
