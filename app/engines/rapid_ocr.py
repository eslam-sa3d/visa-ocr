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
from .tesseract_mrz import _estimate_tilt

_FILLER_LIKE = str.maketrans({">": "<", "«": "<", "‹": "<", "＜": "<", "《": "<", "〈": "<", "人": "<", "く": "<", " ": "", "\t": ""})   # 人 and く: how this model reads a filler on small print


def normalise(text: str) -> str:
    return str(text).upper().translate(_FILLER_LIKE)


class RapidOcrEngine:
    name = "rapid-ocr"

    def __init__(self) -> None:
        self._reader = None
        self._lock = threading.Lock()                  # the ONNX session is shared; read one image at a time

    def _load(self):
        if self._reader is None:
            try:
                from rapidocr_onnxruntime import RapidOCR  # noqa: WPS433
            except ImportError as exc:
                raise EngineUnavailable("rapidocr-onnxruntime is not installed") from exc
            self._reader = RapidOCR()
        return self._reader

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
                        t0 = time.monotonic()
                        texts, scores = self._read(reader, part, np)
                        last_cost, last_area = time.monotonic() - t0, part.width * part.height
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
