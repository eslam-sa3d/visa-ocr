"""Checks on uploaded images before any decoding or OCR happens."""
from __future__ import annotations

import io

from fastapi import HTTPException

from . import config

_MAGIC = {"image/jpeg": b"\xff\xd8\xff", "image/png": b"\x89PNG\r\n\x1a\n"}


class UploadError(HTTPException):
    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(status_code=status, detail={"code": code, "message": message})


def validate_image(data: bytes, claimed_type: str | None) -> tuple[int, int]:
    """Return (width, height) or raise UploadError with a specific code."""
    from PIL import Image  # imported late: keeps module import cheap

    if len(data) > config.max_bytes():
        raise UploadError(413, "too_large", f"Image exceeds {config.max_bytes() // (1024 * 1024)} MB")
    # The label sent by a browser or chat app is often wrong (a JPEG named .png, "image/jpg", no label at all), so the bytes
    # decide. Only real JPEG or PNG content gets through, and Pillow checks the format again below.
    sniffed = next((t for t, magic in _MAGIC.items() if data.startswith(magic)), None)
    if sniffed is None:
        if claimed_type in _MAGIC:
            raise UploadError(415, "type_mismatch", "The file content is not the image type it claims to be")
        raise UploadError(415, "bad_type", "Only JPEG and PNG images are accepted")
    try:
        with Image.open(io.BytesIO(data)) as img:     # reads the header only, does not decode pixels
            width, height = img.size
            fmt = img.format
    except Image.DecompressionBombError:
        raise UploadError(413, "image_too_many_pixels", "The image is too large to process") from None
    except Exception:  # noqa: BLE001 - corrupt, truncated or not an image
        raise UploadError(422, "invalid_image", "The file could not be opened as an image") from None
    if fmt not in ("JPEG", "PNG"):
        raise UploadError(415, "type_mismatch", "The file content is not a JPEG or PNG image")
    if width * height > config.max_pixels():
        raise UploadError(413, "image_too_many_pixels",
                          f"The image has {width * height // 1_000_000} megapixels; the limit is {config.max_pixels() // 1_000_000}")
    if max(width, height) > config.max_aspect() * min(width, height):
        raise UploadError(422, "image_bad_aspect", "The image is too long and narrow; upload a photo of the whole page")
    if min(width, height) < config.min_side():
        raise UploadError(422, "image_too_small", f"The image is too small ({width}x{height}); use a photo of at least "
                                                  f"{config.min_side()} pixels on each side")
    return width, height


def upscaled_size(width: int, height: int, target_width: int) -> tuple[int, int] | None:
    """Size for enlarging an image to target_width, or None when the result would exceed the pixel cap."""
    if width >= target_width or width <= 0:
        return None
    new_h = max(1, int(height * target_width / width))
    return (target_width, new_h) if target_width * new_h <= config.max_pixels() else None


def shrink_for_ocr(data: bytes) -> bytes:
    """Bring a large photo down to the working size (WORK_MAX_SIDE, longest side in pixels). A 12-megapixel phone photo takes
    several times longer to read and several times the memory, for no gain: the readers work at about 1600 to 2400 pixels.
    The page is turned upright first (EXIF), so everything downstream sees one consistent image. Small images are returned
    unchanged, byte for byte. A decoding problem also returns the original, so the normal error handling deals with it."""
    limit = config.work_max_side()
    if limit <= 0:
        return data
    try:
        from PIL import Image, ImageOps  # noqa: WPS433
        with Image.open(io.BytesIO(data)) as img:
            if max(img.size) <= limit:
                return data
            if img.format == "JPEG":
                img.draft("RGB", (limit, limit))            # decode at a reduced scale: several times faster
            img = ImageOps.exif_transpose(img).convert("RGB")
            ratio = limit / max(img.size)
            if ratio < 1:
                img = img.resize((max(1, round(img.width * ratio)), max(1, round(img.height * ratio))), Image.LANCZOS)
            out = io.BytesIO()
            img.save(out, "JPEG", quality=92)
            return out.getvalue()
    except Exception:  # noqa: BLE001
        return data
