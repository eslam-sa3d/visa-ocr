"""Is the photo clear enough to read? Measured from the pixels alone, in a few milliseconds, before and apart from any OCR.

An unclear photo cannot be fixed by reading harder: the applicant has to take another one. This module names what is wrong
(blurred, too dark, washed out by light, glare on the page, too small) so the message can say what to do differently.

Each measure is taken on a copy of the image scaled to a fixed size, so a threshold means the same thing for a 12-megapixel
phone photo and a small scan.
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field

from . import config

WORK_SIDE = 1000          # long side of the copy the measures are taken on

MESSAGES = {
    "blurred": "The photo is blurred. Hold the phone steady, wait for it to focus on the document, and take the photo again.",
    "too_dark": "The photo is too dark. Move to a brighter place or turn on a light, and take the photo again.",
    "too_bright": "The photo is washed out by light. Move away from direct light or turn off the flash, and take the photo again.",
    "glare": "Light is reflecting off the document and hides part of it. Tilt the document or move away from the lamp until "
             "the reflection is gone, and take the photo again.",
    "mrz_unclear": "The two long lines of letters and '<' at the bottom of the document could not be read clearly. Take the "
                   "photo again with those lines fully inside the frame, flat, in focus, and with no reflection or shadow on them.",
    "card_unclear": "Some of the print on the card could not be read clearly. Take the photo again with the whole card flat, "
                    "in focus and in good light, filling the frame, with no reflection on it.",
    "not_found": "The document could not be read from this photo. Take the photo again: show the page with the two long lines "
                 "of letters and '<' (for a card, the side that has them), flat, in focus and in good light, filling the frame.",
    "multiple_documents": "The photo shows more than one document. Take a photo of one document only: the page with the two "
                          "long lines of letters and '<', with no other passport, visa or card in the frame.",
    "page_cut_off": "Part of the page is outside the photo. Take the photo again with the whole page inside the frame: the "
                    "photo, the printed details and the two long lines of letters and '<' at the bottom.",
    "not_a_passport": "This is not the photo page of a passport. Take a photo of the passport page that shows your photo and "
                      "has two long lines of letters and '<' at the bottom.",
    "too_small": "The image is too small to read. Move closer so the document fills the frame, and take the photo again.",
    "low_contrast": "The print is too faint to read in this photo. Use better light and a plain, darker background, and take "
                    "the photo again.",
}
RETAKE = "The image is not clear enough to read. "


@dataclass
class Quality:
    width: int = 0
    height: int = 0
    sharpness: float = 0.0        # how much fine detail there is (variance of the Laplacian); low = blurred
    brightness: float = 0.0       # mean grey level, 0 to 255
    contrast: float = 0.0         # spread of grey levels (standard deviation)
    glare: float = 0.0            # share of the image covered by large blown-out patches, 0 to 1
    issues: list[str] = field(default_factory=list)          # clearly unusable: reject without reading
    doubts: list[str] = field(default_factory=list)          # poor: the reason to give if the reading then fails

    @property
    def clear(self) -> bool:
        return not self.issues and not self.doubts

    def to_dict(self) -> dict:
        return {"clear": self.clear, "width": self.width, "height": self.height, "sharpness": round(self.sharpness, 1),
                "brightness": round(self.brightness, 1), "contrast": round(self.contrast, 1), "glare": round(self.glare, 3),
                "issues": list(self.issues), "doubts": list(self.doubts)}


def measure(raw: bytes) -> Quality:
    """The measures and what they say. Never raises for an image that opens; an image that does not open has no issues here
    (the upload checks report that)."""
    try:
        import cv2
        import numpy as np
        from PIL import Image, ImageOps
        img = ImageOps.exif_transpose(Image.open(io.BytesIO(raw)))
        if img.mode in ("RGBA", "LA", "PA") or "transparency" in img.info:       # transparent parts are white paper, not black
            img = Image.alpha_composite(Image.new("RGBA", img.size, (255, 255, 255, 255)), img.convert("RGBA"))
        img = img.convert("L")
    except Exception:  # noqa: BLE001 - not this module's failure to report
        return Quality()
    q = Quality(width=img.width, height=img.height)
    scale = WORK_SIDE / max(img.size)
    if scale < 1:
        img = img.resize((max(1, round(img.width * scale)), max(1, round(img.height * scale))), Image.LANCZOS)
    grey = np.asarray(img, dtype=np.uint8)
    q.sharpness = float(cv2.Laplacian(grey, cv2.CV_64F).var())
    q.brightness = float(grey.mean())
    q.contrast = float(grey.std())
    q.glare = _glare(grey, cv2, np)

    t = config.quality_thresholds()
    if max(q.width, q.height) < t["min_side_reject"]:
        q.issues.append("too_small")
    elif max(q.width, q.height) < t["min_side_doubt"]:
        q.doubts.append("too_small")
    if q.sharpness < t["sharpness_reject"]:
        q.issues.append("blurred")
    elif q.sharpness < t["sharpness_doubt"]:
        q.doubts.append("blurred")
    if q.brightness < t["dark_reject"]:
        q.issues.append("too_dark")
    elif q.brightness < t["dark_doubt"]:
        q.doubts.append("too_dark")
    if q.brightness > t["bright_reject"]:
        q.issues.append("too_bright")
    # Glare is measured for the record but decides nothing: a white table or scanner lid looks the same as a reflection from
    # the whole image, and the glare that matters covers a few characters of the MRZ. The reading itself shows that.
    if q.contrast < t["contrast_reject"]:
        q.issues.append("low_contrast")
    elif q.contrast < t["contrast_doubt"]:
        q.doubts.append("low_contrast")
    return q


def _glare(grey, cv2, np) -> float:
    """Share of the image inside large blown-out patches. White paper is bright but keeps its print; a reflection is a
    solid patch at the very top of the range with nothing in it. Small specks (a hologram glint) do not count."""
    blown = (grey >= 250).astype(np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(blown, connectivity=8)
    floor = grey.size * 0.004
    area = sum(int(stats[i, cv2.CC_STAT_AREA]) for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] >= floor)
    return area / grey.size


def clearest(images: list[bytes]) -> Quality:
    """The measures of the clearest image of an upload. An upload may carry several images (both sides of a card, or a second
    try): one good image is enough to read from, so the upload is only turned back when every image is unusable."""
    best: Quality | None = None
    for raw in images:
        q = measure(raw)
        if best is None or (len(q.issues), len(q.doubts)) < (len(best.issues), len(best.doubts)):
            best = q
    return best or Quality()


def retake_message(reasons: list[str]) -> str:
    """What to tell the applicant: the first reason decides the advice (they are listed most serious first)."""
    for reason in reasons:
        if reason in MESSAGES:
            return MESSAGES[reason]
    return RETAKE + "Take the photo again in good light, with the whole document sharp and flat in the frame."


def retake(reasons: list[str]) -> dict:
    """The block a result carries when the applicant should take the photo again."""
    return {"recommended": True, "reasons": list(reasons), "message": retake_message(reasons)}
