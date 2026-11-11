"""Find the MRZ on a page and cut it out, level, before any reading.

A scan of an open passport on an A4 sheet, a photo with a hand in it, a page fed sideways: the MRZ is a small part of the image
and its letters are a few pixels high. Reading the whole page at one size wastes most of the time and loses the small print.
This module finds the two or three long rows of evenly spaced characters by their shape (no OCR), in any of the four page
orientations, and returns that strip alone, level and enlarged. The readers then read a small, sharp image.

Needs OpenCV and NumPy. Returns an empty list when they are missing or nothing looks like an MRZ; callers fall back to their
full-page search.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Band:
    image: object          # PIL image, grey, level, MRZ fills it; may still be upside down
    quarter: int           # quarter turns (0 or 1) applied to the page before the strip was cut
    score: float           # bigger is more MRZ-like
    box: tuple[int, int, int, int]   # x1, y1, x2, y2 of the strip on the (turned) page, in page pixels
    rows: list | None = None         # [x1, y1, x2, y2] of each text row on the page as uploaded; None when not known exactly


def _rhythm(profile, np) -> float:
    """How strongly a row's ink repeats at the pitch of 30 to 50 evenly spaced characters across its width (0 to 1).
    MRZ print is monospaced from edge to edge; ordinary text, with its words and gaps, is not."""
    n = len(profile)
    if n < 120:
        return 0.0
    x = profile.astype("float32") - float(profile.mean())
    denom = float((x * x).sum())
    if denom <= 1e-6:
        return 0.0
    best = 0.0
    for lag in range(max(3, n // 52), max(4, n // 28) + 1):
        v = float((x[:-lag] * x[lag:]).sum()) / denom
        if v > best:
            best = v
    return best


def _candidates(gray, cv2, np, work_w: int = 1000):
    """MRZ-shaped blocks of a grey page whose text rows are horizontal. Returns (score, rotated_rect) in page pixels.

    Single rows are found first (long, thin, level, with the regular rhythm of monospaced print); rows that sit under one
    another with the same extent are then joined into one block. A row with no partner is widened to take in its neighbours."""
    h, w = gray.shape[:2]
    scale = work_w / float(w)
    small = cv2.resize(gray, (work_w, max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    blur = cv2.GaussianBlur(small, (3, 3), 0)
    blackhat = cv2.morphologyEx(blur, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (17, 9)))
    grad = np.absolute(cv2.Sobel(blackhat, cv2.CV_32F, 1, 0, ksize=3))
    hi = float(grad.max())
    if hi < 1e-6:
        return []
    grad8 = (grad / hi * 255).astype("uint8")
    joined = cv2.morphologyEx(grad8, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (21, 1)))
    _, mask = cv2.threshold(joined, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3)))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rows = []
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        if bw < 0.2 * work_w or bh < 4 or bw / float(bh) < 9:
            continue
        (cx, cy), (rw, rh), ang = cv2.minAreaRect(c)
        if rw < rh:
            rw, rh, ang = rh, rw, ang + 90
        while ang > 45:
            ang -= 90
        while ang < -45:
            ang += 90
        if abs(ang) > 12:
            continue
        rhythm = _rhythm(blackhat[y:y + bh, x:x + bw].sum(axis=0), np)
        if rhythm < 0.12:
            continue
        rows.append({"x": x, "y": y, "w": bw, "h": bh, "cx": cx, "cy": cy, "rw": rw, "rh": rh, "ang": ang, "rhythm": rhythm})
    rows.sort(key=lambda r: r["y"])
    blocks, used = [], set()
    for i, a in enumerate(rows):
        if i in used:
            continue
        group = [a]
        for j in range(i + 1, len(rows)):
            b, last = rows[j], group[-1]
            gap = b["cy"] - last["cy"]
            if j in used or not (0.55 * last["h"] <= gap <= 4.5 * max(last["h"], b["h"])):
                continue
            overlap = min(b["x"] + b["w"], last["x"] + last["w"]) - max(b["x"], last["x"])
            if overlap < 0.75 * min(b["w"], last["w"]) or abs(b["w"] - last["w"]) > 0.35 * max(b["w"], last["w"]):
                continue
            group.append(b)
            used.add(j)
            if len(group) == 3:
                break
        x1 = min(r["x"] for r in group)
        x2 = max(r["x"] + r["w"] for r in group)
        y1 = min(r["y"] for r in group)
        y2 = max(r["y"] + r["h"] for r in group)
        row_h = sum(r["h"] for r in group) / len(group)
        if len(group) == 1:                      # the partner row was not found: take a generous slice around this one
            if (x2 - x1) / max(row_h, 1.0) < 14:   # too tall for one row: the two rows ran together into one shape
                y1, y2, row_h = y1 - 0.35 * row_h, y2 + 0.35 * row_h, row_h / 2.0
            else:
                y1, y2 = y1 - 2.6 * row_h, y2 + 2.6 * row_h
        ang = sum(r["ang"] for r in group) / len(group)
        rhythm = sum(r["rhythm"] for r in group) / len(group)
        score = (x2 - x1) * rhythm * (1.0 + 0.8 * (len(group) - 1))
        exact = None
        if len(group) >= 2 and abs(ang) < 1.0:   # level rows found one by one: their own boxes, with a little margin
            m = row_h * 0.25
            exact = [[int((r["x"] - m) / scale), int((r["y"] - m) / scale), int((r["x"] + r["w"] + m) / scale),
                      int((r["y"] + r["h"] + m) / scale)] for r in group]
        blocks.append((score, (((x1 + x2) / 2 / scale, (y1 + y2) / 2 / scale), ((x2 - x1) / scale, (y2 - y1) / scale), ang),
                       row_h / scale, exact))
    return blocks


def find_bands(pil_gray, limit: int = 3) -> list[Band]:
    """Up to `limit` MRZ-like strips, best first, from the page as given and turned a quarter turn."""
    try:
        import cv2  # noqa: WPS433
        import numpy as np  # noqa: WPS433
        from PIL import Image  # noqa: WPS433
    except ImportError:
        return []
    page = np.asarray(pil_gray.convert("L"))
    found: list[tuple[float, int, tuple]] = []
    for quarter in (0, 1):
        view = page if quarter == 0 else cv2.rotate(page, cv2.ROTATE_90_CLOCKWISE)
        for score, rect, row_h, exact in _candidates(view, cv2, np):
            found.append((score, quarter, rect, row_h, exact if quarter == 0 else None))
    found.sort(key=lambda f: -f[0])
    bands: list[Band] = []
    for score, quarter, ((cx, cy), (rw, rh), ang), row_h, exact in found[:limit]:
        view = page if quarter == 0 else cv2.rotate(page, cv2.ROTATE_90_CLOCKWISE)
        # room around the strip: the mask is tighter than the print, and a third row (ID cards) may sit just outside it
        pad_w, pad_h = rw * 0.05 + 10, row_h * 0.9 + 6
        w, h = int(rw + 2 * pad_w), int(rh + 2 * pad_h)
        m = cv2.getRotationMatrix2D((cx, cy), ang, 1.0)
        m[0, 2] += w / 2.0 - cx
        m[1, 2] += h / 2.0 - cy
        strip = cv2.warpAffine(view, m, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        bands.append(Band(image=Image.fromarray(strip), quarter=quarter, score=float(score),
                          box=(int(cx - w / 2), int(cy - h / 2), int(cx + w / 2), int(cy + h / 2)), rows=exact))
    return bands


def enlarge(strip, target_w: int = 1800):
    """The strip at a size where its letters are comfortably readable (about 30 pixels high), and never absurdly large."""
    from PIL import Image  # noqa: WPS433
    if strip.width >= target_w:
        return strip
    f = min(4.0, target_w / float(strip.width))
    return strip.resize((int(strip.width * f), int(strip.height * f)), Image.LANCZOS)


def text_rows(strip) -> list[tuple[int, int]]:
    """(top, bottom) of each text row in a level strip, from its horizontal ink profile, top to bottom."""
    import numpy as np  # noqa: WPS433
    from PIL import ImageOps  # noqa: WPS433
    ink = 255.0 - np.asarray(ImageOps.autocontrast(strip.convert("L")), dtype=np.float32)
    profile = ink.mean(axis=1)
    lo, hi = float(profile.min()), float(profile.max())
    if hi - lo < 8:
        return []
    on = profile > lo + 0.35 * (hi - lo)
    rows, start = [], None
    for i, v in enumerate(on):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if i - start >= 8:
                rows.append((start, i))
            start = None
    if start is not None and len(on) - start >= 8:
        rows.append((start, len(on)))
    return rows
