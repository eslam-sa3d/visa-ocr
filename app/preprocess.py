"""Image clean-up for photographed documents: find the card/page outline, straighten it, fix small rotation.

Used only as extra attempts when the plain image gives no MRZ, so good scans are never altered.
Needs OpenCV; without it every function returns an empty list and the engine just uses the original.
"""
from __future__ import annotations


try:
    import cv2
    import numpy as np
except ImportError:  # pragma: no cover
    cv2 = None  # type: ignore[assignment]
    np = None  # type: ignore[assignment]


def _order(pts):
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    return np.array([pts[s.argmin()], pts[d.argmin()], pts[s.argmax()], pts[d.argmax()]], dtype="float32")


def find_document_quad(img):
    """Largest 4-corner shape covering at least 25% of the picture, or None."""
    h, w = img.shape[:2]
    scale = 800 / max(h, w)
    small = cv2.resize(img, None, fx=scale, fy=scale) if scale < 1 else img
    gray = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_RGB2GRAY), (5, 5), 0)
    edges = cv2.dilate(cv2.Canny(gray, 40, 120), np.ones((3, 3), np.uint8), iterations=2)
    cnts, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    area_min = 0.25 * small.shape[0] * small.shape[1]
    best = None
    for c in sorted(cnts, key=cv2.contourArea, reverse=True)[:8]:
        if cv2.contourArea(c) < area_min:
            break
        approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            best = approx.reshape(4, 2).astype("float32") / (scale if scale < 1 else 1)
            break
    return best


def warp(img, quad):
    q = _order(quad)
    w = int(max(np.linalg.norm(q[0] - q[1]), np.linalg.norm(q[3] - q[2])))
    h = int(max(np.linalg.norm(q[0] - q[3]), np.linalg.norm(q[1] - q[2])))
    if w < 200 or h < 100:
        return None
    if h > w:                                   # photographed upright: rotate to landscape
        q = np.roll(q, -1, axis=0)
        w, h = h, w
    m = cv2.getPerspectiveTransform(q, np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype="float32"))
    return cv2.warpPerspective(img, m, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def deskew(img):
    """Rotate by the median angle of long text-like lines (handles up to about 20 degrees)."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    bw = cv2.dilate(bw, cv2.getStructuringElement(cv2.MORPH_RECT, (25, 3)))
    lines = cv2.HoughLinesP(bw, 1, np.pi / 360, threshold=100, minLineLength=img.shape[1] // 4, maxLineGap=20)
    if lines is None:
        return None
    angles = [np.degrees(np.arctan2(y2 - y1, x2 - x1)) for x1, y1, x2, y2 in lines[:, 0]]
    angles = [a for a in angles if abs(a) <= 20]
    if len(angles) < 3:
        return None
    angle = float(np.median(angles))
    if abs(angle) < 0.7:
        return None
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def variants(array):
    """Extra versions of an RGB image to try when the original gives no MRZ. Empty when OpenCV is missing."""
    if cv2 is None:
        return []
    out = []
    try:
        quad = find_document_quad(array)
        flat = warp(array, quad) if quad is not None else None
        if flat is not None:
            out.append(flat)
        base = flat if flat is not None else array
        d = deskew(base)
        if d is not None:
            out.append(d)
    except Exception:  # noqa: BLE001  - preprocessing must never break extraction
        return out
    return out
