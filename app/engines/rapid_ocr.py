"""Second open-source reader (RapidOCR, PP-OCR models packaged inside the pip wheel, so it works offline).

It gives a recognition score per line, which Tesseract does not. It is used by the ensemble engine, which compares its
reading with Tesseract's. Optional: without the package the ensemble simply uses Tesseract alone."""
from __future__ import annotations

import contextlib
import difflib
import io
import logging
import queue
import re
import threading
import time
from dataclasses import dataclass, field

from .. import config as app_config
from ..mrz import check_score, find_mrz_lines, lines_confidence, lone_passport_line2
from ..uploads import upscaled_size
from .base import EngineOutput, EngineUnavailable
from .mrz_band import enlarge as enlarge_strip, find_bands, text_rows
from .tesseract_mrz import estimate_tilt

_FILLER_LIKE = str.maketrans({">": "<", "«": "<", "‹": "<", "＜": "<", "《": "<", "〈": "<", "人": "<", "く": "<", " ": "", "\t": ""})   # 人 and く: how this model reads a filler on small print


log = logging.getLogger("visa-ocr")

STRIP_WIDTH = 1600                 # the located MRZ strip is enlarged to this width before its rows are read
STRIP_MAX_DETECT_HEIGHT = 420      # a strip taller than this holds more than the MRZ: do not run the text detector on it
STRIP_SHARE_OF_DEADLINE = 0.6      # under a hard limit, the strip search stops after this share of it ...
STRIP_DETECT_SHARE = 0.5           # ... and the slower detector pass is not started after this share
PAGE_TILTS = (0, 3, -3, 5, -5)     # full-page attempts, in degrees
PAGE_FIRST_WIDTH = (1000, 1200)    # first full-page attempt is read between these widths; reading time grows with size
PAGE_LARGE_WIDTH = 1800            # later attempts, for small print
BOTTOM_SHARE = 0.45                # the "bottom strip" attempt reads this share of the page
PAGE_TEXT_WIDTH = 1100             # page_text() reads the printed page at about this width
MAX_PAGE_TEXT_LINES = 120
BOX_MATCH_RATIO = 0.75             # how alike a detected text and an MRZ line must be to take the box as that line's
DEFAULT_PAGE_COST = 1.2            # seconds for one full-page read until this machine has been timed
GIVE_UP_FACTOR = 3                 # with no hard limit and nothing found, stop after this many times the search budget


@dataclass
class _Best:
    """The best reading so far for one document, and what else was seen on the way."""
    lines: list | None = None
    conf: float | None = None
    score: int = -2
    fields: dict = field(default_factory=dict)
    lone: str | None = None                 # a passport line 2 read on its own, when no complete MRZ was found
    page_text: list = field(default_factory=list)


def normalise(text: str) -> str:
    return str(text).upper().translate(_FILLER_LIKE)


def restore_fillers(text: str) -> str:
    """This reader merges repeated fillers ('<<' comes out as '<'), which shortens a line and shifts everything after it.
    For a passport line 2 the check digits say which reading is right: each filler run is lengthened in turn until the line
    is 44 characters and every check passes. Any other text is returned unchanged."""
    if not (36 <= len(text) <= 43) or not re.fullmatch(r"[A-Z0-9<]+", text) or "<" not in text or not re.search(r"\d{6}", text):
        return text
    missing = 44 - len(text)
    for m in re.finditer(r"<+", text):
        cand = text[:m.end()] + "<" * missing + text[m.end():]
        if lone_passport_line2([cand]):
            return cand
    return text


class RapidOcrEngine:
    name = "rapid-ocr"

    def __init__(self) -> None:
        self._page_cost = DEFAULT_PAGE_COST     # seconds one full-page reading takes on this machine; measured as it runs
        self._sessions: queue.LifoQueue | None = None       # RAPIDOCR_SESSIONS readers; one document uses one at a time
        self._load_lock = threading.Lock()
        self._tl = threading.local()                        # boxes of the last read, per thread

    def _load(self):
        """The pool of readers, built once however many requests arrive together."""
        if self._sessions is None:
            with self._load_lock:
                if self._sessions is None:
                    try:
                        from rapidocr_onnxruntime import RapidOCR  # noqa: WPS433
                    except ImportError as exc:
                        raise EngineUnavailable("rapidocr-onnxruntime is not installed") from exc
                    readers = [RapidOCR() for _ in range(app_config.rapidocr_sessions())]
                    self._calibrate(readers[0])
                    pool: queue.LifoQueue = queue.LifoQueue()
                    for r in readers:
                        pool.put(r)
                    self._sessions = pool                   # published only when ready to use
        return self._sessions

    @contextlib.contextmanager
    def _session(self):
        """One reader for the length of one document; others wait only when every reader is busy."""
        pool = self._load()
        reader = pool.get()
        try:
            yield reader
        finally:
            pool.put(reader)

    def _calibrate(self, reader) -> None:
        """Time one full-page reading on this machine, once, so a hard time limit can be respected from the first request."""
        try:
            import numpy as np  # noqa: WPS433
            from PIL import Image, ImageDraw  # noqa: WPS433
            page = Image.new("RGB", (1100, 780), "white")
            d = ImageDraw.Draw(page)
            for i in range(14):
                d.text((60, 40 + i * 50), "SPECIMEN PAGE TEXT 0123456789 UTOPIA " * 2, fill="black")
            t0 = time.monotonic()
            reader(np.array(page))
            self._page_cost = max(0.2, (time.monotonic() - t0) * 1.3)
        except Exception as exc:  # noqa: BLE001 - keep the default estimate
            log.warning("RapidOCR timing run failed (%s); using the default estimate", type(exc).__name__)

    _score = staticmethod(check_score)

    def _read(self, reader, image, np) -> tuple[list[str], list[float]]:
        result, _ = reader(np.array(image))
        texts = [normalise(r[1]) for r in (result or [])]
        scores = [float(r[2]) for r in (result or [])]
        self._tl.polys = [r[0] for r in (result or [])]
        return texts, scores

    def _read_rows(self, reader, strip, np) -> tuple[list[str], list[float]]:
        """Recognise each text row of a level strip directly (no text detection). Empty when the strip has no clear rows."""
        rows = text_rows(strip)
        if not 2 <= len(rows) <= 14:
            return [], []
        texts, scores = [], []
        for top, bottom in rows[-6:]:                      # the MRZ is the last rows of a strip that holds more
            pad = int((bottom - top) * 0.35) + 2
            crop = strip.crop((0, max(0, top - pad), strip.width, min(strip.height, bottom + pad)))
            result, _ = reader(np.array(crop), use_det=False, use_cls=False, use_rec=True)
            if result:
                texts.append(restore_fillers(normalise(result[0][1])))
                scores.append(float(result[0][2]))
        return texts, scores

    def page_text(self, images: list[bytes], seconds: float) -> list[dict]:
        """The printed part of the page (name, country code, sex ...) for the review rules to compare with the MRZ. One reading
        per image at about 1100 pixels wide, tried also a quarter turn round when the first finds little; stops at `seconds`."""
        try:
            import numpy as np  # noqa: WPS433
            from PIL import Image, ImageOps  # noqa: WPS433
        except ImportError:
            return []
        out: list[dict] = []
        seen: set[str] = set()
        started = time.monotonic()                          # waiting for a free reader counts against the time given
        with self._session() as reader:
            for raw in images:
                img = ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert("RGB")
                for turn in (0, 90, 270):
                    if time.monotonic() - started + self._page_cost > seconds:
                        return out                         # a reading would not finish in the time given
                    view = img if not turn else img.rotate(turn, expand=True)
                    target = PAGE_TEXT_WIDTH
                    if view.width != target:
                        f = min(3.0, target / float(view.width))
                        view = view.resize((max(1, int(view.width * f)), max(1, int(view.height * f))), Image.LANCZOS)
                    t0 = time.monotonic()
                    texts, scores = self._read(reader, view, np)
                    self._page_cost = 0.7 * self._page_cost + 0.3 * (time.monotonic() - t0)
                    words = [t for t in texts if re.fullmatch(r"[A-Z][A-Z,./' -]{3,}", t)]
                    for tx, sc in zip(texts, scores):
                        if "<" not in tx and tx not in seen and len(out) < 150:
                            seen.add(tx)
                            out.append({"text": tx, "score": round(float(sc), 3)})
                    if len(words) >= 4:
                        break                              # this way up reads as text: no need to turn the page
        return out

    def _line_boxes(self, lines: list[str], texts: list[str], index: int, scale: float, y_off: int) -> list[dict] | None:
        """Where each MRZ line sits on the uploaded image, in its own pixels (x1, y1, x2, y2). A reviewer's correction is saved
        with this box so the line can be cut out for training. None when a line cannot be matched to one detected box."""
        boxes, used = [], set()
        for line in lines:
            ranked = sorted(((difflib.SequenceMatcher(None, line, t).ratio(), i) for i, t in enumerate(texts) if i not in used),
                            reverse=True)
            if not ranked or ranked[0][0] < BOX_MATCH_RATIO:
                return None
            i = ranked[0][1]
            used.add(i)
            xs = [float(p[0]) / scale for p in self._tl.polys[i]]
            ys = [float(p[1]) / scale + y_off for p in self._tl.polys[i]]
            boxes.append({"image": index, "box": [int(min(xs)), int(min(ys)), int(max(xs)) + 1, int(max(ys)) + 1]})
        return boxes

    def _strip_stage(self, reader, img, index: int, best: _Best, deadline: float, t0: float, np) -> tuple[bool, bool]:
        """Read the MRZ strip alone, located by its shape: small, level and enlarged, in any page orientation. Each row of the
        strip is recognised on its own; letting the text detector cut the rows out first is slower and often trims the first
        characters of a line. Returns (a strip was found, a perfect reading is now in `best`)."""
        tried = False
        for band in find_bands(img.convert("L")):
            tried = True
            for flip in (0, 180):
                if deadline and time.monotonic() - t0 > deadline * STRIP_SHARE_OF_DEADLINE:
                    break
                strip = enlarge_strip(band.image if not flip else band.image.rotate(180), STRIP_WIDTH).convert("RGB")
                texts, scores = self._read_rows(reader, strip, np)
                lines = find_mrz_lines(texts) if texts else None
                in_time = not (deadline and time.monotonic() - t0 > deadline * STRIP_DETECT_SHARE)
                if not lines and strip.height <= STRIP_MAX_DETECT_HEIGHT and in_time:
                    texts, scores = self._read(reader, strip, np)          # rows not clear: let the detector find them
                    lines = find_mrz_lines(texts)
                if not lines:
                    best.lone = best.lone or lone_passport_line2(texts)
                    continue
                score, perfect = self._score(lines)
                if score > best.score:
                    best.lines, best.score = lines, score
                    best.conf = lines_confidence(lines, texts, scores)
                    ok_rows = band.rows and not flip and len(band.rows) == len(lines)
                    best.fields = {"mrzLineBoxes": [{"image": index, "box": r} for r in band.rows]} if ok_rows else {}
                if perfect:
                    best.lines, best.fields["textLines"] = lines, best.page_text
                    return tried, True
                break                              # read, not perfect: the other way up will not be better
        return tried, False

    def _page_stage(self, reader, img, index: int, best: _Best, deadline: float, budget: float, t0: float, tried_strip: bool,
                    np, Image) -> bool:
        """Read the whole page, then its enlarged bottom part, at the estimated tilt first. Also collects the printed text of
        the page for the review rules. Returns True when a perfect reading is now in `best`."""
        hard = budget * GIVE_UP_FACTOR
        name_lines: list[str] = []      # passport name lines seen without a matching line 2
        seen_text = {t["text"] for t in best.page_text}
        # Under a hard limit, one full-page reading follows at most.
        page_reads_left = 1 if deadline else 99
        tilt = estimate_tilt(img.convert("L"), Image)
        angles = list(PAGE_TILTS)
        if abs(tilt) >= 2:
            angles = [tilt] + sorted((a for a in angles if a != tilt), key=lambda a: abs(a - tilt))
        started = time.monotonic()
        last_cost, last_area, out_of_time = 0.0, 0, False
        for angle in angles:
            if out_of_time or (best.lines and time.monotonic() - started > budget):
                break                              # a reading is in hand: do not spend longer looking for a better one
            view = img if angle == 0 else img.rotate(angle, Image.BICUBIC, expand=True, fillcolor=(255, 255, 255))
            # Under a hard limit the enlarged bottom strip comes before the enlarged full page: the MRZ is at the bottom, and
            # the strip costs less than half as much to read.
            variants = (((1.0, False), (BOTTOM_SHARE, True), (1.0, True)) if deadline
                        else ((1.0, False), (1.0, True), (BOTTOM_SHARE, True)))
            ran = False
            for frac, enlarge in variants:
                y_off = int(view.height * (1 - frac))
                part = view.crop((0, y_off, view.width, view.height))
                native_width = part.width
                # First attempt at about 1200 pixels wide: reading time grows with image size, and a full page reads
                # well at that width. The later attempts use 1800 for small print.
                target = PAGE_LARGE_WIDTH if enlarge else min(PAGE_FIRST_WIDTH[1], max(PAGE_FIRST_WIDTH[0], native_width))
                if native_width > target:
                    part = part.resize((target, max(1, round(part.height * target / native_width))), Image.LANCZOS)
                else:
                    size = upscaled_size(part.width, part.height, target)
                    if size and size[0] > part.width:
                        part = part.resize(size, Image.LANCZOS)
                    elif enlarge and frac == 1.0 and native_width <= PAGE_FIRST_WIDTH[1]:
                        continue                   # could not be enlarged and the first reading covered this size
                elapsed = time.monotonic() - started
                if page_reads_left <= 0:
                    continue
                if deadline and tried_strip and (time.monotonic() - t0) + self._page_cost > deadline:
                    continue                       # a page reading does not fit in the time left on this machine
                if deadline:
                    # Under a hard limit, start an attempt only if it should finish in time. Its cost is estimated
                    # from the last attempt, scaled by image area.
                    area = part.width * part.height
                    expected = max(0.4, last_cost * area / last_area) if last_area else 0.8
                    if elapsed + expected > deadline - 0.1:
                        continue                   # this attempt does not fit; a cheaper one after it may
                elif elapsed > hard and not best.lines:
                    break                          # nothing found and well over time: give up on this image
                ran = True
                page_reads_left -= 1
                t_read = time.monotonic()
                texts, scores = self._read(reader, part, np)
                last_cost, last_area = time.monotonic() - t_read, part.width * part.height
                self._page_cost = 0.7 * self._page_cost + 0.3 * last_cost
                # Page text from every attempt on this image: the MRZ is often read from the enlarged bottom strip, which
                # does not contain the printed name.
                for tx, sc in zip(texts, scores):
                    if "<" not in tx and tx not in seen_text and len(best.page_text) < MAX_PAGE_TEXT_LINES:
                        seen_text.add(tx)
                        best.page_text.append({"text": tx, "score": round(float(sc), 3)})
                lines = find_mrz_lines(texts)
                if not lines:
                    best.lone = best.lone or lone_passport_line2(texts)
                    # A name line seen in one attempt and line 2 in another still make a complete MRZ.
                    name_lines += [tx for tx in texts if re.fullmatch(r"P[A-Z<][A-Z<]{3}[A-Z<]{7,39}", tx) and "<<" in tx]
                    if best.lone and name_lines:
                        for cand in sorted(set(name_lines), key=len, reverse=True):
                            lines = find_mrz_lines([cand, best.lone])
                            if lines and self._score(lines)[1]:
                                break
                            lines = None
                if not lines:
                    continue
                score, perfect = self._score(lines)
                if score > best.score:
                    best.lines, best.score = lines, score
                    best.conf = lines_confidence(lines, texts, scores)   # every line used, not only the long ones
                    # boxes are only exact on the unrotated page
                    boxes = self._line_boxes(lines, texts, index, part.width / native_width, y_off) if angle == 0 else None
                    best.fields = {"mrzLineBoxes": boxes} if boxes else {}
                    # The rest of the page as this reader saw it (name, country code, sex as printed): the review rules
                    # compare it with the MRZ. Personal data: it leaves the service only when INCLUDE_TEXT_LINES is set.
                    best.fields["textLines"] = best.page_text
                if perfect:
                    best.lines = lines
                    return True
            if deadline and not ran:
                out_of_time = True                 # nothing fits any more: stop trying other angles
        return False

    def extract(self, images: list[bytes], doc_type_hint: str | None) -> EngineOutput:
        try:
            import numpy as np  # noqa: WPS433
            from PIL import Image, ImageOps  # noqa: WPS433
        except ImportError as exc:
            raise EngineUnavailable(f"Missing Python dependency: {exc.name}") from exc
        budget = app_config.second_reader_seconds()
        deadline = app_config.ocr_deadline()
        best = _Best()
        t_begin = time.monotonic()                 # a hard limit runs from here: waiting for a free reader counts
        with self._session() as reader:
            for index, raw in enumerate(images):
                img = ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert("RGB")
                t0 = t_begin if index == 0 else time.monotonic()
                tried_strip, perfect = self._strip_stage(reader, img, index, best, deadline, t0, np)
                if not perfect and not app_config.strip_only():
                    perfect = self._page_stage(reader, img, index, best, deadline, budget, t0, tried_strip, np, Image)
                if perfect:
                    return EngineOutput(engine=self.name, mrz_lines=best.lines, mrz_confidence=best.conf, fields=best.fields)
        if best.lines is None and best.lone:
            best.fields = {"mrzLine2": best.lone, "textLines": best.page_text}
        return EngineOutput(engine=self.name, mrz_lines=best.lines, mrz_confidence=best.conf, fields=best.fields)
