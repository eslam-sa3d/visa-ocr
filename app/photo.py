"""Is this a usable personal photo for a visa application? Checked from the picture alone, in a fraction of a second.

What is checked: the size and shape of the picture, that there is one face, that the face is large enough, centred and
whole, that the face is sharp and neither dark nor washed out, and that the background is light and plain. Open eyes and a
level head are looked for and reported as doubts, not as reasons to refuse: glasses and hair make that look unreliable.

Nothing is stored, nothing is compared: this says whether the photo meets the rules, not whose face it shows.
The face is found with the detector that ships inside OpenCV (no model file to fetch, nothing leaves the machine).
"""
from __future__ import annotations

import math
import os
import threading

from . import config

MESSAGES = {
    "unreadable": "The photo could not be opened. Take the photo again and upload it as a JPEG or PNG.",
    "too_small": "The photo is too small. Take it again closer to the camera or at a higher quality.",
    "wrong_shape": "The photo has the wrong shape. Use a portrait photo: taller than it is wide, showing head and shoulders.",
    "no_face": "No face could be found in the photo. Take a photo of your face from the front, in good light, looking at "
               "the camera.",
    "several_faces": "More than one face is in the photo. Take the photo again with only you in it.",
    "face_cut_off": "Part of your face is outside the photo. Take the photo again with your whole head and the top of "
                    "your shoulders inside the frame.",
    "face_too_small": "Your face is too small in the photo. Move closer so that your head and shoulders fill the frame.",
    "face_too_close": "Your face is too close to the camera. Move back so that your whole head and the top of your "
                      "shoulders are in the frame.",
    "not_centred": "Your face is not in the middle of the photo. Take the photo again with your face in the centre.",
    "blurred": "The photo is blurred. Hold the camera steady, wait for it to focus on your face, and take the photo again.",
    "too_dark": "Your face is too dark in the photo. Face a window or a light, and take the photo again.",
    "too_bright": "Your face is washed out by light. Move away from direct light or turn off the flash, and take the photo again.",
    "background_not_light": "The background is too dark. Stand in front of a plain white or light wall and take the photo again.",
    "background_not_plain": "The background is not plain. Stand in front of a plain white or light wall, with nothing "
                            "behind you, and take the photo again.",
    "eyes_not_clear": "Your eyes could not be seen clearly. Look straight at the camera with your eyes open, and remove "
                      "dark glasses or anything covering your eyes.",
    "head_tilted": "Your head is tilted. Hold your head straight and level, facing the camera.",
    "not_colour": "The photo is black and white. Use a colour photo.",
    "check_unavailable": "The photo could not be checked on this server.",
}
_ORDER = list(MESSAGES)                     # most serious first
_detectors: dict = {}
_lock = threading.Lock()                    # an OpenCV detector is not safe to use from two threads at once


class Unavailable(RuntimeError):
    """The face detector is not installed on this host."""


def _detector(name: str):
    if name not in _detectors:
        import cv2
        folders = [os.environ.get("HAARCASCADES_DIR", ""), getattr(getattr(cv2, "data", None), "haarcascades", ""),
                   os.path.join(os.environ.get("MODELS_DIR", ""), "haarcascades") if os.environ.get("MODELS_DIR") else ""]
        for folder in folders:
            path = os.path.join(folder, name) if folder else ""
            if path and os.path.exists(path):
                found = cv2.CascadeClassifier(path)
                if not found.empty():
                    _detectors[name] = found
                    break
        else:
            raise Unavailable(name + " was not found (it ships with opencv 4; set HAARCASCADES_DIR to its folder)")
    return _detectors[name]


def find_faces(grey) -> list[tuple[int, int, int, int]]:
    """Faces seen from the front, as (x, y, width, height), largest first. Small look-alikes in the background are dropped."""
    side = min(grey.shape[:2])
    least = max(24, int(side * 0.12))
    with _lock:
        found = _detector("haarcascade_frontalface_default.xml").detectMultiScale(grey, scaleFactor=1.1, minNeighbors=6,
                                                                                  minSize=(least, least))
    faces = sorted((tuple(int(v) for v in f) for f in found), key=lambda f: f[2] * f[3], reverse=True)
    return [f for f in faces if f[2] * f[3] >= 0.35 * faces[0][2] * faces[0][3]] if faces else []


def find_eyes(face_grey) -> list[tuple[int, int, int, int]]:
    least = max(8, int(face_grey.shape[1] * 0.12))
    with _lock:
        found = _detector("haarcascade_eye.xml").detectMultiScale(face_grey, scaleFactor=1.1, minNeighbors=6, minSize=(least, least))
    return [tuple(int(v) for v in e) for e in found]


def check(raw: bytes) -> dict:
    """The verdict on one photo: acceptable or not, why, and what to tell the applicant."""
    try:
        import cv2
        import numpy as np
        from .quality import open_reduced
        img, (width, height) = open_reduced(raw, "RGB", 900)      # decoded at the working size, not full size
    except Exception:  # noqa: BLE001 - whatever stops the picture from opening, the applicant's answer is the same
        return _verdict(["unreadable"], [], {})
    t = config.photo_thresholds()
    measures: dict = {"width": width, "height": height}
    issues: list[str] = []
    doubts: list[str] = []
    if min(width, height) < t["min_side"]:
        issues.append("too_small")
    if not t["shape_min"] <= width / height <= t["shape_max"]:
        issues.append("wrong_shape")
    rgb = np.asarray(img, dtype=np.uint8)
    grey = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    h, w = grey.shape
    spread = rgb.astype(np.int16)
    if float(np.abs(spread[..., 0] - spread[..., 1]).mean() + np.abs(spread[..., 1] - spread[..., 2]).mean()) < 2.0:
        doubts.append("not_colour")

    faces = find_faces(grey)
    measures["faces"] = len(faces)
    if not faces:
        overall = float(grey.mean())
        measures["brightness"] = round(overall, 1)
        # a face lost in the dark or burnt out by light is not found either: say what to change, not just "no face"
        burnt = float((grey >= 250).mean())                   # a light wall is bright; a burnt-out photo is pure white
        issues.append("too_dark" if overall < t["dark"] * 0.8 else "too_bright" if burnt > 0.35 else "no_face")
        return _verdict(issues, doubts, measures)
    if len(faces) > 1:
        issues.append("several_faces")
    x, y, fw, fh = faces[0]
    measures.update(faceHeight=round(fh / h, 2), faceCentre=round((x + fw / 2) / w, 2))
    edge = 0.01 * min(w, h)
    if x <= edge or y <= edge or x + fw >= w - edge or y + fh >= h - edge:
        issues.append("face_cut_off")
    if fh / h < t["face_min"]:
        issues.append("face_too_small")
    elif fh / h > t["face_max"]:
        issues.append("face_too_close")
    if not 0.5 - t["centre"] <= (x + fw / 2) / w <= 0.5 + t["centre"]:
        issues.append("not_centred")

    face = cv2.resize(grey[y:y + fh, x:x + fw], (200, 200), interpolation=cv2.INTER_AREA)
    sharp, light = float(cv2.Laplacian(face, cv2.CV_64F).var()), float(face.mean())
    measures.update(faceSharpness=round(sharp, 1), faceBrightness=round(light, 1))
    if sharp < t["sharpness"]:
        issues.append("blurred")
    if light < t["dark"]:
        issues.append("too_dark")
    elif light > t["bright"]:
        issues.append("too_bright")

    # The background: beside the head and above it, away from hair and shoulders.
    parts = []
    upper = max(1, int(y + fh * 0.55))
    left, right = int(x - fw * 0.45), int(x + fw * 1.45)
    if left > 4:
        parts.append(rgb[:upper, :left])
    if right < w - 4:
        parts.append(rgb[:upper, right:])
    above = int(y - fh * 0.45)
    if above > 4:
        parts.append(rgb[:above, max(0, left):min(w, right)])
    if parts:
        pixels = np.concatenate([p.reshape(-1, 3) for p in parts]).astype(np.float32)
        tone = pixels.mean(axis=1)
        measures.update(backgroundBrightness=round(float(tone.mean()), 1), backgroundSpread=round(float(tone.std()), 1))
        if tone.mean() < t["background_light"]:
            issues.append("background_not_light")
        if tone.std() > t["background_spread"]:
            issues.append("background_not_plain")

    eyes = [e for e in find_eyes(grey[y:y + int(fh * 0.62), x:x + fw])]
    measures["eyes"] = len(eyes)
    if len(eyes) < 2:
        doubts.append("eyes_not_clear")
    else:
        a, b = sorted(sorted(eyes, key=lambda e: e[2] * e[3], reverse=True)[:2])
        angle = math.degrees(math.atan2((b[1] + b[3] / 2) - (a[1] + a[3] / 2), (b[0] + b[2] / 2) - (a[0] + a[2] / 2)))
        measures["tiltDegrees"] = round(angle, 1)
        if abs(angle) > t["tilt"]:
            doubts.append("head_tilted")
    return _verdict(issues, doubts, measures)


def _verdict(issues: list[str], doubts: list[str], measures: dict) -> dict:
    issues = sorted(dict.fromkeys(issues), key=_ORDER.index)
    doubts = sorted(dict.fromkeys(doubts), key=_ORDER.index)
    out = {"acceptable": not issues, "issues": issues, "doubts": doubts, "measures": measures,
           "messages": [MESSAGES[c] for c in issues + doubts]}
    if issues or doubts:
        reasons = issues or doubts
        out["retake"] = {"recommended": bool(issues), "reasons": reasons, "message": MESSAGES[reasons[0]]}
    return out
