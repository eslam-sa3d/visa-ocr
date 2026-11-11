"""Second open-source reader (RapidOCR, PP-OCR models packaged inside the pip wheel, so it works offline).

It gives a recognition score per line, which Tesseract does not. It is used by the ensemble engine, which compares its
reading with Tesseract's. Optional: without the package the ensemble simply uses Tesseract alone."""
from __future__ import annotations

import difflib
import io
import os
import re
import threading
import time

from .. import config as app_config
from ..mrz import MrzError, find_mrz_lines, lines_confidence, lone_passport_line2, parse_mrz
from ..uploads import upscaled_size
from .base import EngineOutput, EngineUnavailable
from .mrz_band import enlarge as enlarge_strip, find_bands, text_rows
from .tesseract_mrz import _estimate_tilt

_FILLER_LIKE = str.maketrans({">": "<", "«": "<", "‹": "<", "＜": "<", "《": "<", "〈": "<", "人": "<", "く": "<", " ": "", "\t": ""})   # 人 and く: how this model reads a filler on small print


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
        self._page_cost = 1.2      # seconds one full-page reading takes on this machine; measured as it runs
        self._reader = None
        self._lock = threading.Lock()                  # the ONNX session is shared; read one image at a time

    def _load(self):
        if self._reader is None:
            try:
                from rapidocr_onnxruntime import RapidOCR  # noqa: WPS433
            except ImportError as exc:
                raise EngineUnavailable("rapidocr-onnxruntime is not installed") from exc
            self._reader = RapidOCR()
            self._calibrate()
        return self._reader

    def _calibrate(self) -> None:
        """Time one full-page reading on this machine, once, so a hard time limit can be respected from the first request."""
        try:
            import numpy as np  # noqa: WPS433
            from PIL import Image, ImageDraw  # noqa: WPS433
            page = Image.new("RGB", (1100, 780), "white")
            d = ImageDraw.Draw(page)
            for i in range(14):
                d.text((60, 40 + i * 50), "SPECIMEN PAGE TEXT 0123456789 UTOPIA " * 2, fill="black")
            t0 = time.monotonic()
            self._reader(np.array(page))
            self._page_cost = max(0.2, (time.monotonic() - t0) * 1.3)
        except Exception:  # noqa: BLE001 - keep the default estimate
            pass

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
        self._last_polys = [r[0] for r in (result or [])]
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
        reader = self._load()
        out: list[dict] = []
        seen: set[str] = set()
        started = time.monotonic()
        with self._lock:
            for raw in images:
                img = ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert("RGB")
                for turn in (0, 90, 270):
                    if time.monotonic() - started + self._page_cost > seconds:
                        return out                         # a reading would not finish in the time given
                    view = img if not turn else img.rotate(turn, expand=True)
                    target = 1100
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
            if not ranked or ranked[0][0] < 0.75:
                return None
            i = ranked[0][1]
            used.add(i)
            xs = [float(p[0]) / scale for p in self._last_polys[i]]
            ys = [float(p[1]) / scale + y_off for p in self._last_polys[i]]
            boxes.append({"image": index, "box": [int(min(xs)), int(min(ys)), int(max(xs)) + 1, int(max(ys)) + 1]})
        return boxes

    def extract(self, images: list[bytes], doc_type_hint: str | None) -> EngineOutput:
        try:
            import numpy as np  # noqa: WPS433
            from PIL import Image, ImageOps  # noqa: WPS433
        except ImportError as exc:
            raise EngineUnavailable(f"Missing Python dependency: {exc.name}") from exc
        reader = self._load()
        budget = float(os.environ.get("OCR_SECOND_READER_SECONDS", "1.5"))
        deadline = app_config.ocr_deadline()
        hard = budget * 3
        best, best_conf, best_score, best_fields = None, None, -2, {}
        lone = None        # a passport line 2 read on its own, when no complete MRZ was found
        name_lines: list[str] = []      # passport name lines seen without a matching line 2
        page_text: list[dict] = []
        seen_text: set[str] = set()
        with self._lock:
            for index, raw in enumerate(images):
                img = ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert("RGB")
                # The estimated tilt is tried first. Each view is read at its own size first (fast), and enlarged only if
                # that reading is not perfect: enlarging a page that is already big enough only makes detection slower.
                # First the MRZ strip alone, located by its shape: small, level and enlarged, in any page orientation.
                t_strip = time.monotonic()
                tried_strip = False
                for band in find_bands(img.convert("L")):
                    tried_strip = True
                    for flip in (0, 180):
                        if deadline and time.monotonic() - t_strip > deadline * 0.6:
                            break
                        strip = enlarge_strip(band.image if not flip else band.image.rotate(180), 1600).convert("RGB")
                        # Each row of the strip is recognised on its own. Letting the text detector cut the rows out first
                        # is slower and often trims the first characters of a line.
                        texts, scores = self._read_rows(reader, strip, np)
                        lines = find_mrz_lines(texts) if texts else None
                        if not lines and strip.height <= 420 and not (deadline and time.monotonic() - t_strip > deadline * 0.5):
                            texts, scores = self._read(reader, strip, np)          # rows not clear: let the detector find them
                            lines = find_mrz_lines(texts)
                        if not lines:
                            lone = lone or lone_passport_line2(texts)
                            continue
                        score, perfect = self._score(lines)
                        if score > best_score:
                            best, best_score = lines, score
                            best_conf = lines_confidence(lines, texts, scores)
                            ok_rows = band.rows and not flip and len(band.rows) == len(lines)
                            best_fields = {"mrzLineBoxes": [{"image": index, "box": r} for r in band.rows]} if ok_rows else {}
                        if perfect:
                            best_fields["textLines"] = page_text
                            return EngineOutput(engine=self.name, mrz_lines=lines, mrz_confidence=best_conf, fields=best_fields)
                        break                              # read, not perfect: the other way up will not be better
                # Under a hard limit, one full-page reading follows at most: it also collects the printed text of the page.
                page_reads_left = 1 if deadline else 99
                tilt = _estimate_tilt(img.convert("L"), Image)
                angles = [0, 3, -3, 5, -5]
                if abs(tilt) >= 2:
                    angles = [tilt] + sorted((a for a in angles if a != tilt), key=lambda a: abs(a - tilt))
                started = time.monotonic()
                last_cost, last_area, out_of_time = 0.0, 0, False
                for angle in angles:
                    if out_of_time or (best and time.monotonic() - started > budget):
                        break                              # a reading is in hand: do not spend longer looking for a better one
                    view = img if angle == 0 else img.rotate(angle, Image.BICUBIC, expand=True, fillcolor=(255, 255, 255))
                    # Under a hard limit the enlarged bottom strip comes before the enlarged full page: the MRZ is at the bottom, and
                    # the strip costs less than half as much to read.
                    variants = ((1.0, False), (0.45, True), (1.0, True)) if deadline else ((1.0, False), (1.0, True), (0.45, True))
                    ran = False
                    for frac, enlarge in variants:
                        y_off = int(view.height * (1 - frac))
                        part = view.crop((0, y_off, view.width, view.height))
                        native_width = part.width
                        # First attempt at about 1200 pixels wide: reading time grows with image size, and a full page reads
                        # well at that width. The later attempts use 1800 for small print.
                        target = 1800 if enlarge else min(1200, max(1000, native_width))
                        if native_width > target:
                            part = part.resize((target, max(1, round(part.height * target / native_width))), Image.LANCZOS)
                        else:
                            size = upscaled_size(part.width, part.height, target)
                            if size and size[0] > part.width:
                                part = part.resize(size, Image.LANCZOS)
                            elif enlarge and frac == 1.0 and native_width <= 1200:
                                continue                   # could not be enlarged and the first reading covered this size
                        elapsed = time.monotonic() - started
                        if page_reads_left <= 0:
                            continue
                        if deadline and tried_strip and (time.monotonic() - t_strip) + self._page_cost > deadline:
                            continue                       # a page reading does not fit in the time left on this machine
                        if deadline:
                            # Under a hard limit, start an attempt only if it should finish in time. Its cost is estimated
                            # from the last attempt, scaled by image area.
                            area = part.width * part.height
                            expected = max(0.4, last_cost * area / last_area) if last_area else 0.8
                            if elapsed + expected > deadline - 0.1:
                                continue                   # this attempt does not fit; a cheaper one after it may
                        elif elapsed > hard and not best:
                            break                          # nothing found and well over time: give up on this image
                        ran = True
                        page_reads_left -= 1
                        t0 = time.monotonic()
                        texts, scores = self._read(reader, part, np)
                        last_cost, last_area = time.monotonic() - t0, part.width * part.height
                        self._page_cost = 0.7 * self._page_cost + 0.3 * last_cost
                        # Page text from every attempt on this image: the MRZ is often read from the enlarged bottom strip, which
                        # does not contain the printed name.
                        for tx, sc in zip(texts, scores):
                            if "<" not in tx and tx not in seen_text and len(page_text) < 120:
                                seen_text.add(tx)
                                page_text.append({"text": tx, "score": round(float(sc), 3)})
                        lines = find_mrz_lines(texts)
                        if not lines:
                            lone = lone or lone_passport_line2(texts)
                            # A name line seen in one attempt and line 2 in another still make a complete MRZ.
                            name_lines += [tx for tx in texts if re.fullmatch(r"P[A-Z<][A-Z<]{3}[A-Z<]{7,39}", tx) and "<<" in tx]
                            if lone and name_lines:
                                for cand in sorted(set(name_lines), key=len, reverse=True):
                                    lines = find_mrz_lines([cand, lone])
                                    if lines and self._score(lines)[1]:
                                        break
                                    lines = None
                        if not lines:
                            continue
                        score, perfect = self._score(lines)
                        if score > best_score:
                            best, best_score = lines, score
                            best_conf = lines_confidence(lines, texts, scores)   # every line used, not only the long ones
                            # boxes are only exact on the unrotated page
                            boxes = self._line_boxes(lines, texts, index, part.width / native_width, y_off) if angle == 0 else None
                            best_fields = {"mrzLineBoxes": boxes} if boxes else {}
                            # The rest of the page as this reader saw it (name, country code, sex as printed): the review rules
                            # compare it with the MRZ. Personal data: it leaves the service only when INCLUDE_TEXT_LINES is set.
                            best_fields["textLines"] = page_text
                        if perfect:
                            return EngineOutput(engine=self.name, mrz_lines=lines, mrz_confidence=best_conf, fields=best_fields)
                    if deadline and not ran:
                        out_of_time = True                 # nothing fits any more: stop trying other angles
        if best is None and lone:
            best_fields = {"mrzLine2": lone, "textLines": page_text}
        return EngineOutput(engine=self.name, mrz_lines=best, mrz_confidence=best_conf, fields=best_fields)
