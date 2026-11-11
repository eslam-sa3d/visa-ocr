"""Arabic + English text lines from Tesseract, for ID cards that have no MRZ (Iqama, GCC IDs).

Needs the `ara` and `eng` traineddata files. Gives a line score (Tesseract word confidence / 100) and a box per line,
the same shape the PaddleOCR engine returns, so app/visual_zone.py reads both the same way."""
from __future__ import annotations

import io
import os
from collections import defaultdict

from .. import config as app_config
from ..textutil import script_of, to_ascii_digits
from ..uploads import upscaled_size
from .base import EngineUnavailable


def available_langs() -> set[str]:
    try:
        import pytesseract
        return set(pytesseract.get_languages(config=""))
    except Exception:  # noqa: BLE001
        return set()


def read_text_lines(images: list[bytes], langs: str | None = None) -> list[dict]:
    try:
        import pytesseract
        from PIL import Image, ImageOps
    except ImportError as exc:  # pragma: no cover
        raise EngineUnavailable(f"Missing Python dependency: {exc.name}") from exc
    langs = langs or os.environ.get("TESSERACT_TEXT_LANGS", "ara+eng")
    have = available_langs()
    wanted = [l for l in langs.split("+") if l in have]
    if not wanted:
        raise EngineUnavailable("Tesseract has none of the requested languages: " + langs)
    lines: list[dict] = []
    for index, raw in enumerate(images):
        base = ImageOps.autocontrast(ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert("L"))
        page = _read_page(base, 2000, index, pytesseract, "+".join(wanted), Image)
        # A second reading at another scale: some lines read cleanly at one scale and not at the other. Per row, the more
        # confident reading wins, and a row only one scale found is kept.
        alt = _read_page(base, 1300, index, pytesseract, "+".join(wanted), Image)
        pw = page[0]["_page_w"] if page else base.width
        merged = _merge_scales(page, alt)
        merged += _name_line_above_latin(merged, base.resize((pw, max(1, round(base.height * pw / base.width))), Image.LANCZOS),
                                         index, pytesseract, "+".join(wanted), to_ascii_digits)
        merged = _dedupe(merged)
        scaled = base.resize((pw, max(1, round(base.height * pw / base.width))), Image.LANCZOS)
        merged += _fill_row_gaps(merged, scaled, index, pytesseract, "+".join(wanted), to_ascii_digits)
        _reread_id_digits(merged, scaled, pytesseract, to_ascii_digits)
        for ln in merged:
            ln["scale"] = round(pw / base.width, 4)      # box pixels per pixel of the uploaded image
        lines.extend(merged)
    return lines


def _dedupe(lines: list[dict]) -> list[dict]:
    """Drop a line that repeats another line's text on the same row (a blurred row is sometimes reported twice)."""
    kept: list[dict] = []
    for ln in sorted(lines, key=lambda l: -l["score"]):
        if any(o["text"] == ln["text"] and abs(o["box"][0][1] - ln["box"][0][1]) < max(8, o["box"][2][1] - o["box"][0][1]) for o in kept):
            continue
        kept.append(ln)
    return sorted(kept, key=lambda l: (l["box"][0][1], l["box"][0][0]))


def _fill_row_gaps(lines: list[dict], img, index: int, pytesseract, langs: str, to_digits) -> list[dict]:
    """Cards print rows at an even pitch. When two neighbouring lines are about two pitches apart, a row between them was dropped
    (a blurred row often is): read that band on its own."""
    rows = sorted((l for l in lines if l["box"][2][1] - l["box"][0][1] > 8), key=lambda l: l["box"][0][1])
    if len(rows) < 4:
        return []
    tops = [r["box"][0][1] for r in rows]
    pitches = sorted(b - a for a, b in zip(tops, tops[1:]) if b - a > 20)
    if not pitches:
        return []
    pitch = pitches[len(pitches) // 2]
    x0 = min(r["box"][0][0] for r in rows)
    x1 = max(r["box"][1][0] for r in rows)
    found: list[dict] = []
    for a, b in zip(rows, rows[1:]):
        gap = b["box"][0][1] - a["box"][0][1]
        if not 1.7 * pitch <= gap <= 3.2 * pitch:
            continue
        top, bottom = a["box"][2][1] + 2, b["box"][0][1] - 2
        if bottom - top < 0.5 * pitch:
            continue
        band = img.crop((max(0, x0 - 20), top, min(img.width, x1 + 20), bottom))
        best = None
        for scale in (0.5, 1.0):
            c = band.resize((max(1, int(band.width * scale)), max(1, int(band.height * scale))))
            for psm in (6, 7):
                try:
                    d = pytesseract.image_to_data(c, lang=langs, config=f"--psm {psm}", output_type=pytesseract.Output.DICT,
                                                  timeout=app_config.ocr_timeout())
                except Exception:  # noqa: BLE001
                    continue
                idx = [i for i, w in enumerate(d["text"]) if str(w).strip() and float(d["conf"][i]) >= 0]
                if not idx:
                    continue
                text = " ".join(d["text"][i] for i in idx).replace("\u200f", "").replace("\u200e", "").strip()
                score = sum(float(d["conf"][i]) for i in idx) / len(idx) / 100
                if len(text) >= 4 and (best is None or score > best[1]):
                    best = (text, score)
        if best and best[1] >= 0.5:
            found.append({"image": index, "lang": "ara+eng", "text": best[0], "normalizedText": to_digits(best[0]),
                          "script": script_of(best[0]), "score": round(best[1], 4),
                          "box": [[x0, top], [x1, top], [x1, bottom], [x0, bottom]]})
    return found


def _name_line_above_latin(lines: list[dict], img, index: int, pytesseract, langs: str, to_digits) -> list[dict]:
    """Cards print the Arabic name directly above the Latin one. When there is a Latin name line (capital words) and no Arabic line
    in the strip above it, that line was dropped by the page reading: read the strip on its own."""
    import re
    for ln in lines:
        t = ln["text"].strip()
        if not re.fullmatch(r"[A-Z][A-Z' .-]{4,}", t) or len(t.split()) < 2:
            continue
        (x0, y0), (x1, _), (_, y1) = ln["box"][0], ln["box"][1], ln["box"][2]
        h = max(1, y1 - y0)
        top, bottom = max(0, y0 - int(h * 2.4)), max(0, y0 - 2)
        if bottom - top < h * 0.8:
            continue
        if any(o is not ln and o["script"] == "arabic" and top - h <= o["box"][0][1] and o["box"][2][1] <= y0 + h * 0.3
               and min(o["box"][1][0], x1) - max(o["box"][0][0], x0) > 0.3 * (x1 - x0) for o in lines):
            continue
        strip = img.crop((max(0, x0 - int(h * 2)), top, min(img.width, x1 + int(h * 0.6)), bottom))
        best = None
        arabic_only = "ara" if "ara" in langs.split("+") else langs
        for scale in (0.5, 1.0):
            c = strip.resize((max(1, int(strip.width * scale)), max(1, int(strip.height * scale))))
            for psm in (6, 13):
                try:
                    d = pytesseract.image_to_data(c, lang=arabic_only, config=f"--psm {psm}", output_type=pytesseract.Output.DICT,
                                                  timeout=app_config.ocr_timeout())
                except Exception:  # noqa: BLE001
                    continue
                idx = [i for i, w in enumerate(d["text"]) if str(w).strip() and float(d["conf"][i]) >= 0]
                words = [d["text"][i] for i in idx if re.search("[؀-ۿ]", d["text"][i]) and len(d["text"][i]) >= 2]
                if len(words) < 2:
                    continue
                score = sum(float(d["conf"][i]) for i in idx if d["text"][i] in words) / len(words) / 100
                if best is None or score > best[1]:
                    best = (" ".join(words), score, scale)
        if best and best[1] >= 0.6:
            return [{"image": index, "lang": "ara+eng", "text": best[0], "normalizedText": to_digits(best[0]), "script": "arabic",
                     "score": round(best[1], 4), "box": [[x0, top], [x1, top], [x1, bottom], [x0, bottom]]}]
    return []


def _read_page(base, width: int, index: int, pytesseract, langs: str, Image) -> list[dict]:
    img = base
    size = upscaled_size(img.width, img.height, width)
    if size:
        img = img.resize(size, Image.LANCZOS)
    data = pytesseract.image_to_data(img, lang=langs, config="--psm 6", output_type=pytesseract.Output.DICT,
                                     timeout=app_config.ocr_timeout())
    groups = defaultdict(list)
    for i, text in enumerate(data["text"]):
        if str(text).strip() and float(data["conf"][i]) >= 0:
            groups[(data["block_num"][i], data["par_num"][i], data["line_num"][i])].append(i)
    out: list[dict] = []
    for idx in groups.values():
        # A noise mark from another row is sometimes grouped into a line and stretches its box across the card: drop such outliers
        if len(idx) > 2:
            cys = sorted(data["top"][i] + data["height"][i] / 2 for i in idx)
            med, hs = cys[len(cys) // 2], sorted(data["height"][i] for i in idx)[len(idx) // 2]
            kept = [i for i in idx if abs(data["top"][i] + data["height"][i] / 2 - med) <= max(hs, 1) * 0.9]
            idx = kept or idx
        # Tesseract already lists the words of a line in reading order (right to left for Arabic): keep that order
        text = " ".join(data["text"][i] for i in idx).replace("\u200f", "").replace("\u200e", "").strip()
        x0 = min(data["left"][i] for i in idx); y0 = min(data["top"][i] for i in idx)
        x1 = max(data["left"][i] + data["width"][i] for i in idx); y1 = max(data["top"][i] + data["height"][i] for i in idx)
        score = sum(float(data["conf"][i]) for i in idx) / len(idx) / 100
        out.append({"image": index, "lang": "ara+eng", "text": text, "normalizedText": to_ascii_digits(text),
                    "script": script_of(text), "score": round(score, 4), "box": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
                    "_page_w": img.width})
    _reread_weak(out, img, pytesseract, langs, Image)
    return out


def _merge_scales(a: list[dict], b: list[dict]) -> list[dict]:
    """Merge two readings of the same page. Boxes of b are rescaled to a's pixel size first."""
    if not a:
        result = b
    else:
        k = (a[0]["_page_w"] / b[0]["_page_w"]) if b else 1.0
        result = list(a)
        for lb in b:
            bx = [[p[0] * k, p[1] * k] for p in lb["box"]]
            cy = (bx[0][1] + bx[2][1]) / 2
            hit = None
            for i, la in enumerate(result):
                ya, yb = la["box"][0][1], la["box"][2][1]
                xa0, xa1 = la["box"][0][0], la["box"][1][0]
                if ya <= cy <= yb and min(xa1, bx[1][0]) - max(xa0, bx[0][0]) > 0.3 * min(xa1 - xa0, bx[1][0] - bx[0][0]):
                    hit = i
                    break
            lb = {**lb, "box": [[round(p[0]), round(p[1])] for p in bx]}
            if hit is None:
                result.append(lb)
            elif lb["score"] > result[hit]["score"] + 0.05:
                result[hit] = lb
    for ln in result:
        ln.pop("_page_w", None)
    return result


def _reread_weak(page_lines: list[dict], img, pytesseract, langs: str, Image) -> None:
    """A line read with low confidence is read again on its own, at two scales, and replaced only when the new reading is more
    confident. Whole-page reading garbles some single lines (a short Arabic country name, say) that read cleanly alone."""
    for line in page_lines:
        if line["score"] >= 0.6:
            continue
        (x0, y0), (x1, _), (_, y1) = line["box"][0], line["box"][1], line["box"][2]
        pad = max(10, int((y1 - y0) * 0.35))
        crop = img.crop((max(0, x0 - pad), max(0, y0 - pad), min(img.width, x1 + pad), min(img.height, y1 + pad)))
        best = None
        for scale in (0.5, 1.0):
            c = crop.resize((max(1, int(crop.width * scale)), max(1, int(crop.height * scale))), Image.LANCZOS)
            try:
                d = pytesseract.image_to_data(c, lang=langs, config="--psm 7", output_type=pytesseract.Output.DICT,
                                              timeout=app_config.ocr_timeout())
            except Exception:  # noqa: BLE001
                continue
            words = [(str(t).strip(), float(cf)) for t, cf in zip(d["text"], d["conf"]) if str(t).strip() and float(cf) >= 0]
            if not words:
                continue
            text = " ".join(w for w, _ in words).replace("\u200f", "").replace("\u200e", "").strip()
            score = sum(cf for _, cf in words) / len(words) / 100
            if best is None or score > best[1]:
                best = (text, score)
        if best and best[1] > line["score"] + 0.1:
            line["text"], line["normalizedText"] = best[0], to_ascii_digits(best[0])
            line["script"], line["score"] = script_of(best[0]), round(best[1], 4)


def _reread_id_digits(lines: list[dict], img, pytesseract, to_digits) -> None:
    """The ID number row is read again with digits only. Blur makes the whole-page reading drop or swap digits; a digits-only read
    of the one row often gets them all. The new digits replace the old ones only when they form a number that passes an ID check
    and the old ones do not."""
    import re
    from .. import gcc_ids
    from ..visual_zone import saudi_id_check

    def passes(num: str) -> bool:
        info = gcc_ids.classify(num)
        return bool(info and info["valid"]) or (len(num) == 10 and num[0] in "12" and saudi_id_check(num))

    for ln in lines:
        text = to_digits(ln["text"])
        runs = re.findall(r"\d[\d\s-]{6,}\d", text)
        if not runs or re.search(r"\d{4}\s*[/.]\s*\d{1,2}", text) or re.search(r"\d{1,2}\s*[/.]\s*\d{1,2}\s*[/.]\s*\d{4}", text):
            continue
        old = re.sub(r"[\s-]", "", runs[0])
        if passes(old):
            continue
        (x0, y0), (x1, _), (_, y1) = ln["box"][0], ln["box"][1], ln["box"][2]
        pad = max(6, int((y1 - y0) * 0.2))
        row = img.crop((max(0, x0 - pad), max(0, y0 - pad), min(img.width, x1 + pad), min(img.height, y1 + pad)))
        tries = [(scale, psm, share) for share in (0.7, 1.0) for scale in (0.5, 1.0) for psm in (6, 7)]
        for scale, psm, share in tries:
            part = row.crop((0, 0, max(1, int(row.width * share)), row.height))
            c = part.resize((max(1, int(part.width * scale)), max(1, int(part.height * scale))))
            try:
                got = pytesseract.image_to_string(c, lang="eng", config=f"--psm {psm} -c tessedit_char_whitelist=0123456789-",
                                                  timeout=app_config.ocr_timeout())
            except Exception:  # noqa: BLE001
                continue
            new = re.sub(r"\D", "", got)
            if new != old and 9 <= len(new) <= 15 and passes(new):
                ln["text"] = ln["text"].replace(runs[0], new) if runs[0] in ln["text"] else ln["text"] + " " + new
                ln["normalizedText"] = to_digits(ln["text"])
                break
