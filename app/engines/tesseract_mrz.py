"""Baseline open-source engine: finds the MRZ with Tesseract.

MRZ print is OCR B, which the general English model reads poorly. The project ships its own model for it,
models/tessdata/ocrb_int.traineddata (how it was made: training/mrz_tesseract/README.md). It is used automatically when its
checksum in models/tessdata/SHA256SUMS matches; otherwise the engine falls back to the installed English model.
Everything here works offline.
"""
from __future__ import annotations

import hashlib
import io
import logging
import shlex
import time

from .. import config as app_config
from ..mrz import check_score, find_mrz_lines
from ..uploads import upscaled_size
from .mrz_band import enlarge, find_bands
from .base import EngineOutput, EngineUnavailable

log = logging.getLogger("visa-ocr")

_WHITELIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<"
STRIP_WIDTHS = (1800, 1600)          # the located MRZ strip is read at both: a character misread at one size is often right at the other
PAGE_WIDTHS = (1600, 2400)           # full-page attempts; the larger one only for the level page
PAGE_TOPS = (0.65, 0.5, 0.0)         # share of the page skipped from the top: bottom third, bottom half, whole page
TILT_ANGLES = (0, 2, -2, 4, -4, 6, -6)    # small rotations tried for hand-held photos, in degrees
QUARTER_TURNS = (90, 270, 180)       # a scan fed in sideways or upside down
MIN_BUDGET_SECONDS = 0.3
DEADLINE_MARGIN_SECONDS = 0.5        # left for the rest of the job when a hard limit is set


def tessdata_verified(folder, name: str) -> bool:
    """A Tesseract model in MODELS_DIR/tessdata is used only when its SHA-256 is listed in SHA256SUMS beside it
    ('<hex>  <file name>' per line, as sha256sum writes), unless ALLOW_UNVERIFIED_MODELS=1."""
    from ..model_registry import allow_unverified
    file = folder / f"{name}.traineddata"
    try:
        listed = {parts[-1].lstrip("*"): parts[0].lower() for parts in
                  (ln.split() for ln in (folder / "SHA256SUMS").read_text(encoding="utf-8").splitlines()) if len(parts) >= 2}
    except OSError:
        listed = {}
    want = listed.get(file.name)
    if want is None:
        if not allow_unverified():
            log.warning("%s is not listed in %s: not used", file.name, folder / "SHA256SUMS")
        return allow_unverified()
    h = hashlib.sha256()
    with open(file, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    if h.hexdigest() != want:
        log.error("%s does not match its checksum: not used", file.name)
        return False
    return True


def estimate_tilt(img, Image) -> int:
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


class _Found:
    """The best reading so far: the one with the most check digits passing."""
    def __init__(self) -> None:
        self.lines: list[str] | None = None
        self.score = -2

    def offer(self, lines: list[str]) -> bool:
        """Keep `lines` if they beat what is held. True when every check digit passes (nothing can beat that)."""
        score, perfect = check_score(lines)
        if score > self.score or perfect:
            self.lines, self.score = lines, score
        return perfect


class TesseractMrzEngine:
    name = "tesseract-mrz"

    def __init__(self) -> None:
        # An OCR-B model reads MRZ print far better than the general English one. It is used when TESSERACT_LANG names it, or
        # when MODELS_DIR/tessdata holds ocrb_int.traineddata or ocrb.traineddata (see docs/MODEL_TRAINING.md).
        self.lang = app_config.tesseract_lang()
        self.data_dir = app_config.tesseract_data_dir()
        if not self.lang:
            self.lang = "eng"
            from ..model_registry import default_models_dir as models_dir
            local = models_dir() / "tessdata"
            for name in ("ocrb_int", "ocrb"):
                if (local / f"{name}.traineddata").is_file() and tessdata_verified(local, name):
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

    _score = staticmethod(check_score)      # more passing check digits means a more trustworthy reading

    def _reader(self, pytesseract, ImageOps):
        """(read a crop of the page, read a located strip): each returns MRZ lines or None."""
        config = f"--psm 6 -c tessedit_char_whitelist={_WHITELIST}"
        if self.data_dir:
            config += f" --tessdata-dir {shlex.quote(self.data_dir)}"     # the option string is split like a shell line

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

        return read, read_strip

    def _strip_search(self, img, read_strip, best: _Found, expired) -> bool:
        """The MRZ strip alone, located by its shape: small, level and enlarged, in any page orientation. True when a
        reading with every check digit passing is in `best`."""
        for band in find_bands(img):
            for flip in (0, 180):
                if expired():
                    break
                turned = band.image if not flip else band.image.rotate(180)
                got = False
                for width in STRIP_WIDTHS:
                    lines = read_strip(enlarge(turned, width))
                    if not lines:
                        continue
                    got = True
                    if best.offer(lines):
                        return True
                if got:
                    break                               # read, not perfect: the other way up will not be better
        return False

    def _level_search(self, img, Image, read, best: _Found, expired) -> bool:
        """Plain image first, then small rotations (hand-held photos are rarely level), then a larger scale. The tilt is
        estimated first so the likeliest angle is tried first. True when the search is over: a perfect reading, or the same
        lines from two different crops or scales (a stable reading: the page is level and a failing check digit is a
        look-alike character, which rotating will not fix)."""
        tilt = estimate_tilt(img, Image)
        angles = list(TILT_ANGLES)
        if abs(tilt) >= 2:
            angles = [tilt] + sorted((a for a in angles if a != tilt), key=lambda a: abs(a - tilt))
        seen: dict[tuple, int] = {}
        for angle in angles:
            view = img if angle == 0 else img.rotate(angle, Image.BICUBIC, expand=True, fillcolor=255)
            for width in PAGE_WIDTHS:
                if angle and width != PAGE_WIDTHS[0]:
                    continue
                for top in PAGE_TOPS:
                    if expired():
                        break
                    lines = read(view, top, width)
                    if not lines:
                        continue
                    if best.offer(lines):
                        return True
                    seen[tuple(lines)] = seen.get(tuple(lines), 0) + 1
                    if seen[tuple(lines)] >= 2 and check_score(lines)[0] >= 0:
                        return True
        return False

    def _sideways_search(self, img, Image, read, best: _Found, expired) -> bool:
        """The page may be sideways or upside down (a scan fed in the wrong way round); EXIF orientation does not cover
        that, so the three other quarter-turns get a short search. True when a perfect reading is in `best`."""
        for quarter in QUARTER_TURNS:
            view = img.rotate(quarter, Image.BICUBIC, expand=True, fillcolor=255)
            for top in PAGE_TOPS:
                if expired():
                    break
                lines = read(view, top, PAGE_WIDTHS[0])
                if lines and best.offer(lines):
                    return True
            if best.lines:
                break
        return False

    def extract(self, images: list[bytes], doc_type_hint: str | None) -> EngineOutput:
        pytesseract, Image, ImageOps = self._deps()
        read, read_strip = self._reader(pytesseract, ImageOps)
        budget = app_config.search_seconds()
        deadline = app_config.ocr_deadline()
        if deadline:
            budget = min(budget, max(MIN_BUDGET_SECONDS, deadline - DEADLINE_MARGIN_SECONDS))   # one attempt takes about 0.2 to 0.4 s
        best = _Found()
        for raw in images:
            img = ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert("L")
            started = time.monotonic()
            best.score = -2                             # each image is ranked afresh; the lines of an earlier one are kept
            expired = lambda: time.monotonic() - started > budget   # noqa: E731
            if self._strip_search(img, read_strip, best, expired) or self._level_search(img, Image, read, best, expired) or best.lines:
                return EngineOutput(engine=self.name, mrz_lines=best.lines)
            if deadline:
                continue                                # no time left for the sideways search under a hard limit
            started = time.monotonic()
            if self._sideways_search(img, Image, read, best, expired) or best.lines:
                return EngineOutput(engine=self.name, mrz_lines=best.lines)
        return EngineOutput(engine=self.name, mrz_lines=best.lines)
