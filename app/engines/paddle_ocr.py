"""Trainable text-reading engine (PaddleOCR) for English and Arabic.

Models are loaded only from the local models directory (see app/model_registry.py), never
downloaded, so it works with no internet. Fine-tuned models replace the pretrained ones by
being registered and marked active; no code change is needed.

What it returns
  - every text line found, with language, script, confidence and box
  - the MRZ lines if one is among them (so the MRZ parser and validation still apply)
Mapping lines to document fields (name, place of birth, ...) is a later step that needs
document templates or a layout model.
"""
from __future__ import annotations

import io
import logging
import os
import threading

# Offline by construction: never let PaddleX check or fetch models from the internet.
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def _writable_cache_home() -> None:
    """PaddleX keeps a small cache under ~/.paddlex and fails to start when it cannot create it. A service account often has no
    home folder (a container user made with --no-create-home), so in that case the cache goes to a private temporary folder.
    Must run before PaddleX is imported. Nothing is downloaded into it: models come from MODELS_DIR only."""
    if os.environ.get("PADDLE_PDX_CACHE_HOME"):
        return
    target = os.path.join(os.path.expanduser("~"), ".paddlex")
    try:
        os.makedirs(target, exist_ok=True)
        if os.access(target, os.W_OK):
            return
    except OSError:
        pass
    import tempfile  # noqa: WPS433
    os.environ["PADDLE_PDX_CACHE_HOME"] = tempfile.mkdtemp(prefix="visa-ocr-paddlex-")


_writable_cache_home()

from .. import config
from ..mrz import check_score, clean_line, find_mrz_lines, lines_confidence
from ..uploads import upscaled_size
from ..model_registry import ManifestError, Registry
from ..textutil import dedupe_overlapping, script_of, to_ascii_digits
from .base import EngineOutput, EngineUnavailable


log = logging.getLogger("visa-ocr")


def _result_dict(res) -> dict:
    """PaddleOCR 3.x results behave like dicts; some versions wrap them under .json['res']."""
    if isinstance(res, dict):
        return res.get("res", res)
    payload = getattr(res, "json", None)
    if isinstance(payload, dict):
        return payload.get("res", payload)
    return {}


def _mrz_score(found, texts, scores) -> float | None:
    """Lowest recognition score over the detected texts the returned MRZ lines came from; None when a line was padded or
    adjusted (see mrz.lines_confidence). A short, low-score line 1 must not ride on the score of a clean line 2."""
    return lines_confidence(found, [str(t) for t in texts], list(scores))


# The text-detection settings of the PaddleX OCR pipeline (configs/pipelines/OCR.yaml in PaddleX 3.7). A stand-alone detector has
# other defaults; with these the shared path finds the same boxes as the usual one.
PIPELINE_DETECTION = {"limit_side_len": 64, "limit_type": "min", "thresh": 0.3, "box_thresh": 0.6,
                      "unclip_ratio": 1.5}


class SharedReader:
    """One text detector and one recogniser per language: the text boxes are found once and each recogniser reads the same
    crops. The usual pipeline (one PaddleOCR object per language) finds the boxes again for every language, which is most
    of the reading time on a page with little text. The steps are the ones the PaddleX OCR pipeline takes: detect, sort
    the boxes top to bottom, cut each one out level, recognise."""

    def __init__(self, detector, recognisers: dict, crop, sort) -> None:
        self.detector, self.recognisers, self.crop, self.sort = detector, recognisers, crop, sort

    def _detect(self, array) -> tuple[list, list]:
        polys = []
        for res in self.detector.predict(array):
            polys += list(_result_dict(res).get("dt_polys", []))
        polys = list(self.sort(polys)) if polys else []
        kept, crops = [], []
        for poly, sub in zip(polys, self.crop(array, polys) if polys else []):
            if getattr(sub, "size", 0) > 0 and sub.shape[0] > 0 and sub.shape[1] > 0:
                kept.append(poly)
                crops.append(sub)
        return kept, crops

    @staticmethod
    def _score(result) -> float:
        score = _result_dict(result).get("rec_score")
        score = score[0] if isinstance(score, (list, tuple)) and score else score
        return float(score or 0)

    def quarter_turns(self, array, kept: list, crops: list) -> int:
        """0 when the page is upright; otherwise how many quarter turns (anticlockwise, as numpy.rot90 counts) set it upright.
        A photo taken with the document turned has its text running up or down the page. Which of the two is settled by
        reading the longest few lines both ways up: the right way scores far higher."""
        tall = wide = 0.0
        shaped = []
        for poly, sub in zip(kept, crops):
            xs, ys = [float(p[0]) for p in poly], [float(p[1]) for p in poly]
            w, h = max(xs) - min(xs), max(ys) - min(ys)
            if max(w, h) >= 3 * max(1.0, min(w, h)):          # only boxes shaped like a line of text say anything
                if h > w:
                    tall += h
                    shaped.append((h, sub))
                else:
                    wide += w
        if not shaped or tall <= 1.5 * wide:
            return 0
        rec = self.recognisers.get("en") or next(iter(self.recognisers.values()))
        sample = [sub for _, sub in sorted(shaped, key=lambda x: -x[0])[:4]]
        as_cut = sum(self._score(r) for r in rec.predict(sample))
        flipped = sum(self._score(r) for r in rec.predict([sub[::-1, ::-1].copy() for sub in sample]))
        # the cutter lays a tall box flat by turning it a quarter anticlockwise: if that reads well, the page needs the same
        return 1 if as_cut >= flipped else 3

    def read_upright(self, array, index: int, np) -> tuple[list[dict], object]:
        """Read a page, turning it upright first when it was photographed sideways. Returns the lines and the page as read."""
        kept, crops = self._detect(array)
        turns = self.quarter_turns(array, kept, crops) if crops else 0
        if turns:
            array = np.ascontiguousarray(np.rot90(array, turns))
            kept, crops = self._detect(array)
        return self._recognise(kept, crops, index), array

    def read(self, array, index: int) -> list[dict]:
        kept, crops = self._detect(array)
        return self._recognise(kept, crops, index)

    def _recognise(self, kept: list, crops: list, index: int) -> list[dict]:
        lines: list[dict] = []
        if not crops:
            return lines
        for lang, rec in self.recognisers.items():
            results = list(rec.predict(crops))
            if len(results) != len(crops):
                raise RuntimeError("the recogniser did not answer for every text box")
            for poly, res in zip(kept, results):
                d = _result_dict(res)
                text = d.get("rec_text", "")
                text = str(text[0] if isinstance(text, (list, tuple)) and text else text)
                if not text.strip():
                    continue
                score = d.get("rec_score")
                score = score[0] if isinstance(score, (list, tuple)) and score else score
                lines.append({"image": index, "lang": lang, "text": text, "normalizedText": to_ascii_digits(text),
                              "script": script_of(text), "score": round(float(score), 4) if score is not None else None,
                              "box": [[int(x), int(y)] for x, y in (poly.tolist() if hasattr(poly, "tolist") else poly)]})
        return lines


class PaddleOcrEngine:
    name = "paddle-ocr"

    def __init__(self, registry: Registry | None = None, langs: list[str] | None = None) -> None:
        self.registry = registry or Registry()
        self.langs = langs or config.ocr_langs()
        self._readers: dict[str, object] = {}
        self._stamp: tuple = self._manifest_stamp()
        self._used_models: dict[str, dict] = {}
        self._shared: SharedReader | None = None               # PADDLE_SHARED_DETECTION=1: one detector for all languages
        self._shared_used: dict[str, dict] = {}
        self._shared_stamp: tuple = ()
        self._shared_off = False                               # set when the shared path failed; the usual path is used
        self._init_lock = threading.Lock()                     # one model load at a time
        # Paddle predictors are not re-entrant and each language has one predictor, so reads go one at a time whatever
        # OCR_CONCURRENCY says. More throughput on this engine means more worker processes, not more threads.
        self._infer = threading.Lock()

    # -- loading -----------------------------------------------------------------------
    def _paddle_class(self):
        try:
            from paddleocr import PaddleOCR  # noqa: WPS433
        except ImportError as exc:
            raise EngineUnavailable("paddleocr is not installed on this host") from exc
        return PaddleOCR

    def _checked(self, task: str, lang: str):
        try:
            m = self.registry.active(task, lang)
            if m is None:
                raise EngineUnavailable(f"No active {task} model registered for language '{lang}'")
            ok, status = self.registry.usable(m)            # every file is hashed again before a model is loaded
        except ManifestError as exc:
            raise EngineUnavailable(f"Model manifest refused: {exc}") from exc
        if not ok:
            hint = " (no checksum recorded; run training/register_model.py or tools/fetch_models.py)" if status == "unverified" else ""
            raise EngineUnavailable(f"Model {m.id} is {status}{hint}")
        return m

    def _manifest_stamp(self) -> tuple:
        try:
            st = self.registry.manifest_path.stat()
            return (st.st_mtime_ns, st.st_size)
        except OSError:
            return ()

    def _reader(self, lang: str):
        # A model switch (register_model.py --activate, or automatic retraining) rewrites the manifest. Notice that and load
        # the newly active models on the next read, so the workers do not need a restart.
        stamp = self._manifest_stamp()
        if stamp != self._stamp:
            with self._init_lock:
                if stamp != self._stamp:
                    self._switch(stamp)
        readers = self._readers                              # one consistent set, even if a switch happens right now
        if lang in readers:
            return readers[lang]
        with self._init_lock:
            if lang not in self._readers:
                reader = self._build_reader(lang, self._used_models)
                self._readers = {**self._readers, lang: reader}
        return self._readers[lang]

    def _switch(self, stamp: tuple) -> None:
        """The manifest changed. Load and try the newly active models first; only when every one of them works do they
        replace the models in use. If anything fails (half-copied folder, wrong checksum, unreadable manifest) the models in
        use stay, and the failure is logged. Readers that were never loaded are simply loaded from the new manifest."""
        if not self._readers:
            self._stamp = stamp
            return
        used: dict[str, dict] = {}
        try:
            fresh = {lang: self._build_reader(lang, used) for lang in self._readers}
            for reader in fresh.values():
                self._smoke(reader)
        except Exception as exc:  # noqa: BLE001 - whatever went wrong, keep serving with the models already loaded
            log.error("model switch refused; keeping the models in use: %s: %s", type(exc).__name__, exc)
            self._stamp = stamp                               # do not retry on every request; the next manifest change will
            return
        self._readers, self._used_models, self._stamp = fresh, used, stamp
        log.info("model switch done: %s", ", ".join(f"{k}={v['id']}" for k, v in sorted(used.items())))

    @staticmethod
    def _smoke(reader) -> None:
        """One read of a blank image: proves the weights load and run before they take over."""
        import numpy as np  # noqa: WPS433
        list(reader.predict(np.full((64, 256, 3), 255, dtype="uint8")))

    def _build_shared(self) -> tuple[SharedReader, dict]:
        try:
            from paddleocr import TextDetection, TextRecognition  # noqa: WPS433
            from paddlex.inference.pipelines.components import CropByPolys, SortQuadBoxes  # noqa: WPS433
        except ImportError as exc:
            raise EngineUnavailable("paddleocr is not installed on this host") from exc
        det = self._checked("text_detection", "any")
        used: dict[str, dict] = {}
        recs = {}
        # enable_mkldnn=False as in the usual path: oneDNN fails on some x86 CPU builds (seen with these very models)
        detector = TextDetection(model_name=det.architecture(), model_dir=str(self.registry.full_path(det)), enable_mkldnn=False,
                                 **PIPELINE_DETECTION)
        for lang in self.langs:
            rec = self._checked("text_recognition", lang)
            recs[lang] = TextRecognition(model_name=rec.architecture(), model_dir=str(self.registry.full_path(rec)),
                                         enable_mkldnn=False)
            used[lang] = {"id": rec.id, "version": rec.version, "detector": det.id}
        return SharedReader(detector, recs, CropByPolys(det_box_type="quad"), SortQuadBoxes()), used

    def _shared_reader(self) -> tuple[SharedReader, dict] | None:
        """The shared reader for the manifest as it is now, or None when it is off or could not be built. A manifest change
        builds and tries the new one first and keeps the one in use if that fails, like the usual path."""
        if not config.paddle_shared_detection() or self._shared_off:
            return None
        stamp = self._manifest_stamp()
        if self._shared is None or stamp != self._shared_stamp:
            with self._init_lock:
                if self._shared is None or stamp != self._shared_stamp:
                    try:
                        import numpy as np  # noqa: WPS433
                        fresh, used = self._build_shared()
                        fresh.read(np.full((64, 256, 3), 255, dtype="uint8"), 0)       # proves the weights load and run
                    except Exception as exc:  # noqa: BLE001
                        if self._shared is None:
                            log.error("shared text detection could not start (%s: %s); using one pipeline per language",
                                      type(exc).__name__, exc)
                            self._shared_off = True
                            return None
                        log.error("model switch refused; keeping the models in use: %s: %s", type(exc).__name__, exc)
                        self._shared_stamp = stamp
                    else:
                        self._shared, self._shared_used, self._shared_stamp = fresh, used, stamp
        return self._shared, dict(self._shared_used)

    def _build_reader(self, lang: str, used: dict):
        det = self._checked("text_detection", "any")
        rec = self._checked("text_recognition", lang)
        PaddleOCR = self._paddle_class()
        reader = PaddleOCR(
            text_detection_model_name=det.architecture(),
            text_detection_model_dir=str(self.registry.full_path(det)),
            text_recognition_model_name=rec.architecture(),
            text_recognition_model_dir=str(self.registry.full_path(rec)),
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            enable_mkldnn=False,  # oneDNN breaks some x86 CPU builds; speed is fine without it
        )
        used[lang] = {"id": rec.id, "version": rec.version, "detector": det.id}
        return reader

    def _mrz_from_bottom_band(self, images, reader, np, Image, ImageOps):
        """Second chance: the MRZ is always the bottom band, so crop it, enlarge it and read only that."""
        for raw in images:
            img = ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert("RGB")
            w, h = img.size
            for frac in (0.30, 0.45):
                band = img.crop((0, int(h * (1 - frac)), w, h))
                size = upscaled_size(band.width, band.height, 1800)
                if size:
                    band = band.resize(size, Image.LANCZOS)
                texts, scores = [], []
                for res in reader.predict(np.array(band)):
                    d = _result_dict(res)
                    texts += [str(t) for t in d.get("rec_texts", [])]
                    scores += [float(x) for x in d.get("rec_scores", [])]
                found = find_mrz_lines(texts)
                if found:
                    return found, _mrz_score(found, texts, scores)
        return None, None

    def _mrz_around_anchor(self, images, lines: list[dict], reader, np, Image, ImageOps):
        """In a phone photo the passport is small in the frame: one MRZ line is often read whole while the other comes out
        broken or with wrong digits. A line that is clearly MRZ shows where the zone is, so cut that strip out (a few line
        heights above and below), enlarge it and read it again: at that size both lines are read properly."""
        import re
        best, best_conf, best_score = None, None, -2
        for index, raw in enumerate(images):
            anchors = []
            for l in lines:
                text = clean_line(l["text"])
                if (l["image"] == index and len(l.get("box") or []) == 4 and len(text) >= 20
                        and re.fullmatch(r"[A-Z0-9<]+", text) and ("<<" in text or len(text) in (30, 36, 44))):
                    anchors.append(l["box"])
            if not anchors:
                continue
            img = ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert("RGB")
            for box in anchors[:2]:
                xs, ys = [p[0] for p in box], [p[1] for p in box]
                height = max(8, max(ys) - min(ys))
                if height * 4 > max(xs) - min(xs):
                    continue        # a line running up the page (turned photo): the straightened-page search handles those
                margin = (max(xs) - min(xs)) * 0.12
                strip = img.crop((max(0, int(min(xs) - margin)), max(0, int(min(ys) - 3.2 * height)),
                                  min(img.width, int(max(xs) + margin)), min(img.height, int(max(ys) + 3.2 * height))))
                size = upscaled_size(strip.width, strip.height, 1800)
                if size:
                    strip = strip.resize(size, Image.LANCZOS)
                texts, scores = [], []
                for res in reader.predict(np.array(strip)):
                    d = _result_dict(res)
                    texts += [str(t) for t in d.get("rec_texts", [])]
                    scores += [float(x) for x in d.get("rec_scores", [])]
                found = find_mrz_lines(texts)
                score, perfect = check_score(found)
                if found and score > best_score:
                    best, best_conf, best_score = found, _mrz_score(found, texts, scores), score
                if perfect:
                    return best, best_conf
        return best, best_conf

    def _mrz_from_variants(self, images, reader, np, Image, ImageOps):
        """Last chance for angled photos: straighten the page, then read the whole page and its bottom band."""
        from .. import preprocess
        for raw in images:
            img = ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert("RGB")
            for arr in preprocess.variants(np.array(img)):
                fixed = Image.fromarray(arr)
                for frac in (1.0, 0.4):
                    h = fixed.height
                    part = fixed.crop((0, int(h * (1 - frac)), fixed.width, h))
                    size = upscaled_size(part.width, part.height, 1800)
                    if size:
                        part = part.resize(size, Image.LANCZOS)
                    texts, scores = [], []
                    for res in reader.predict(np.array(part)):
                        d = _result_dict(res)
                        texts += [str(t) for t in d.get("rec_texts", [])]
                        scores += [float(x) for x in d.get("rec_scores", [])]
                    found = find_mrz_lines(texts)
                    if found:
                        return found, _mrz_score(found, texts, scores)
        return None, None

    # -- extraction --------------------------------------------------------------------
    def extract(self, images: list[bytes], doc_type_hint: str | None, deep: bool = True) -> EngineOutput:
        """deep=False leaves out the slow last-chance searches (bottom band, straightened page); mrz_retries runs them later.
        "auto" uses this to let the second pair of readers try first: they are quicker at finding an MRZ this reader missed."""
        try:
            import numpy as np  # noqa: WPS433
            from PIL import Image, ImageOps  # noqa: WPS433
        except ImportError as exc:
            raise EngineUnavailable(f"Missing Python dependency: {exc.name}") from exc

        with self._infer:
            return self._extract(images, np, Image, ImageOps, deep)

    def mrz_retries(self, images: list[bytes], straighten: bool = True) -> tuple[list[str] | None, float | None]:
        """The last-chance MRZ searches on their own, for a page already read with deep=False. straighten=False leaves out
        the slowest one (the page straightened and read again several ways), for a page that shows no trace of an MRZ."""
        import numpy as np  # noqa: WPS433
        from PIL import Image, ImageOps  # noqa: WPS433
        if "en" not in self.langs:
            return None, None
        with self._infer:
            reader = self._reader("en")
            mrz, conf = self._mrz_from_bottom_band(images, reader, np, Image, ImageOps)
            if not mrz and straighten:
                mrz, conf = self._mrz_from_variants(images, reader, np, Image, ImageOps)
            return mrz, conf

    def _extract(self, images, np, Image, ImageOps, deep: bool = True) -> EngineOutput:
        given = images
        lines: list[dict] | None = None
        readers: dict = {}
        shared = self._shared_reader()
        if shared is not None:
            try:
                lines = []
                images = list(images)
                for index, raw in enumerate(images):
                    array = self._array(raw, np, Image, ImageOps)
                    page_lines, read_as = shared[0].read_upright(array, index, np)
                    lines += page_lines
                    if read_as is not array:                  # photographed sideways: later steps work on the upright page
                        buffer = io.BytesIO()
                        Image.fromarray(read_as).save(buffer, format="PNG", compress_level=1)
                        images[index] = buffer.getvalue()
                used_models = shared[1]                       # the per-language pipelines load only if the MRZ search needs one
            except Exception as exc:  # noqa: BLE001 - the faster path must never cost a reading
                log.error("shared text detection failed (%s); using one pipeline per language from now on", type(exc).__name__)
                self._shared_off, lines, shared = True, None, None
        if lines is None:
            readers = {lang: self._reader(lang) for lang in self.langs}
            used_models = dict(self._used_models)             # the models that read THIS document
            lines = []
            for index, raw in enumerate(images):
                lines += self._read_per_language(self._array(raw, np, Image, ImageOps), index, readers)

        if shared is None:      # the shared reader turns sideways pages itself, cheaply; here each way is read and compared
            images, lines = self._turn_upright(list(images), lines, lambda array, index: self._read_per_language(
                array, index, readers), np, Image, ImageOps)
        out = self._finish(images, lines, readers, used_models, np, Image, ImageOps, deep)
        if any(a is not b for a, b in zip(images, given)):
            out.upright_images = list(images)
        return out

    @staticmethod
    def _array(raw: bytes, np, Image, ImageOps):
        return np.array(ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert("RGB"))

    @staticmethod
    def _sideways(lines: list[dict]) -> bool:
        """True when most of the text on a page runs up or down it: the photo was taken with the document turned a quarter."""
        tall = wide = 0.0
        for l in lines:
            box = l.get("box") or []
            if len(box) != 4:
                continue
            w = max(p[0] for p in box) - min(p[0] for p in box)
            h = max(p[1] for p in box) - min(p[1] for p in box)
            if max(w, h) >= 3 * max(1, min(w, h)):            # only boxes shaped like a line of text say anything
                if h > w:
                    tall += h
                else:
                    wide += w
        return tall > 1.5 * wide and tall > 0

    @staticmethod
    def _page_quality(lines: list[dict]) -> tuple:
        """How well a page was read: an MRZ with every check digit passing first, then check digits, then confident text."""
        mrz = find_mrz_lines([l["text"] for l in lines])
        passing, perfect = check_score(mrz)
        good = sum(len(l["text"]) for l in lines if (l["score"] or 0) >= 0.9)
        return (perfect, passing, good)

    def _turn_upright(self, images: list[bytes], lines: list[dict], read, np, Image, ImageOps) -> tuple[list[bytes], list[dict]]:
        """Phone photos are often taken with the passport turned a quarter and carry no orientation tag. Text read sideways
        comes out one line right and the next wrong. When a page's text runs up the page, read it turned each way and keep
        the better reading; everything after this (MRZ searches, printed-text fields) then works on the upright page."""
        for index, raw in enumerate(images):
            mine = [l for l in lines if l["image"] == index]
            if not self._sideways(mine):
                continue
            img = ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert("RGB")
            best, best_quality, best_img = mine, self._page_quality(mine), None
            for angle in (90, 270):
                turned = img.rotate(angle, expand=True)
                try:
                    candidate = read(np.array(turned), index)
                except Exception as exc:  # noqa: BLE001 - a failed extra look must not lose the first reading
                    log.error("reading the turned page failed (%s)", type(exc).__name__)
                    continue
                quality = self._page_quality(candidate)
                if quality > best_quality:
                    best, best_quality, best_img = candidate, quality, turned
                if quality[0]:
                    break
            if best_img is not None:
                buffer = io.BytesIO()
                best_img.save(buffer, format="PNG")
                images[index] = buffer.getvalue()
                lines = [l for l in lines if l["image"] != index] + best
        return images, lines

    def _read_per_language(self, array, index: int, readers: dict) -> list[dict]:
        """The usual path: one PaddleOCR pipeline per language, each finding and reading the text of the page."""
        lines: list[dict] = []
        for lang, reader in readers.items():
            for res in reader.predict(array):
                data = _result_dict(res)
                texts = data.get("rec_texts", [])
                scores = data.get("rec_scores", [])
                boxes = data.get("rec_polys", data.get("dt_polys", []))
                for i, text in enumerate(texts):
                    if not str(text).strip():
                        continue
                    box = boxes[i] if i < len(boxes) else []
                    lines.append({
                        "image": index,
                        "lang": lang,
                        "text": str(text),
                        "normalizedText": to_ascii_digits(str(text)),
                        "script": script_of(str(text)),
                        "score": round(float(scores[i]), 4) if i < len(scores) else None,
                        "box": [[int(x), int(y)] for x, y in (box.tolist() if hasattr(box, "tolist") else box)],
                    })
        return lines

    def _finish(self, images, lines: list[dict], readers: dict, used_models: dict, np, Image, ImageOps,
                deep: bool = True) -> EngineOutput:
        """Find the MRZ among the lines (with a second and third look at the page when it is not there), drop duplicate
        readings of one box, and build the result."""
        mrz = find_mrz_lines([l["text"] for l in lines if l["lang"] == "en"]) or \
            find_mrz_lines([l["text"] for l in lines])
        mrz_conf = None
        if mrz:
            mrz_conf = _mrz_score(mrz, [l["text"] for l in lines], [l["score"] or 0 for l in lines])
        on_page = bool(mrz)
        if "en" in self.langs and not check_score(mrz)[1]:
            # nothing found, or a reading with a failing check digit: read the strip around the MRZ again, enlarged
            en_reader = readers.get("en") or self._reader("en")
            again, again_conf = self._mrz_around_anchor(images, lines, en_reader, np, Image, ImageOps)
            if again and check_score(again)[1]:        # only a reading whose every check digit passes replaces what was read
                mrz, mrz_conf = again, again_conf
        if deep and "en" in self.langs and not on_page and not check_score(mrz)[1]:
            en_reader = readers.get("en") or self._reader("en")
            band, band_conf = self._mrz_from_bottom_band(images, en_reader, np, Image, ImageOps)
            if band and check_score(band)[0] > check_score(mrz)[0]:
                mrz, mrz_conf = band, band_conf
            if not mrz:
                mrz, mrz_conf = self._mrz_from_variants(images, en_reader, np, Image, ImageOps)
        # every language reader reads every box; keep the best reading per box (per image)
        lines = [l for i in range(len(images)) for l in dedupe_overlapping([x for x in lines if x["image"] == i])]
        scores = [l["score"] for l in lines if l["score"] is not None]
        return EngineOutput(
            engine=self.name,
            mrz_lines=mrz,
            mrz_confidence=mrz_conf,
            fields={"textLines": lines, "models": used_models},
            confidence=round(sum(scores) / len(scores), 4) if scores else None,
        )
