"""PDF uploads: each page is drawn as an image, and the images are read like any JPEG or PNG.

A PDF is a program-like format, so the PDF itself is never kept, stored, queued or read by the OCR readers. It is drawn by
PDFium (the PDF engine of the Chrome browser, via pypdfium2) in a separate, short-lived process with its own limits on
memory, CPU time, wall-clock time and file writes, with an empty environment and no access to the project's code. A PDF that
is broken, hostile or too heavy can at worst end that process; the service answers with a clear error and goes on.

What comes back is one image per page (PNG, or JPEG when the PNG would be larger than MAX_UPLOAD_BYTES), which then goes through the same checks as any upload (validate_image: size, pixels,
shape) and the same readers. Only those page images are queued, stored encrypted and, with consent, kept for training.
"""
from __future__ import annotations

import os
import struct
import subprocess
import sys

from . import config

MAGIC = b"%PDF-"


class PdfError(Exception):
    """A PDF that cannot be used: code is one of bad_pdf, pdf_encrypted, pdf_too_many_pages, pdf_empty, pdf_too_complex."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def is_pdf(data: bytes) -> bool:
    """The PDF header may follow a few bytes of junk; the standard allows it within the first 1024 bytes."""
    return MAGIC in data[:1024]


# Runs in the child process with `python -I -c`: nothing from the project or the current folder is imported. Limits are
# set before the PDF is opened. Protocol: the PDF on stdin; on stdout, for each page, a 4-byte big-endian length and the
# JPEG bytes; on failure, exit code 3 and one line "<code>|<message>" on stderr.
_CHILD = r'''
import io, resource, struct, sys
mem, cpu, pages_max, dpi, max_side, max_pixels, first_only, max_bytes = (int(x) for x in sys.argv[1:9])
resource.setrlimit(resource.RLIMIT_AS, (mem, mem))
resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
try:
    resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
except (ValueError, OSError):
    pass
def fail(code, msg):
    sys.stderr.write(code + "|" + msg + "\n"); sys.stderr.flush(); sys.exit(3)
data = sys.stdin.buffer.read()
import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_c
try:
    doc = pdfium.PdfDocument(data)
except pdfium.PdfiumError as exc:
    text = str(exc).lower()
    fail("pdf_encrypted" if "password" in text or "security" in text else "bad_pdf", str(exc)[:200])
n = len(doc)
if n == 0:
    fail("pdf_empty", "the PDF has no pages")
if n > pages_max and not first_only:
    fail("pdf_too_many_pages", str(n))
out = sys.stdout.buffer
for i in range(1 if first_only else n):
    page = doc[i]
    w, h = page.get_size()                         # points, 1/72 inch
    if w <= 0 or h <= 0:
        fail("bad_pdf", "a page has no size")
    scale = dpi / 72.0
    # A scanned PDF is one image filling the page: draw the page at that image's own resolution, so its pixels come out
    # 1:1 and small print (an MRZ) is not resampled. A page with text or several images is drawn at `dpi`.
    try:
        objs = list(page.get_objects(filter=(pdfium_c.FPDF_PAGEOBJ_IMAGE,), max_depth=2))
        if len(objs) == 1:
            left, bottom, right, top = objs[0].get_bounds()
            px_w, px_h = objs[0].get_px_size()
            if right > left and top > bottom and (right - left) * (top - bottom) >= 0.8 * w * h:
                scale = max(1.0, px_w / (right - left) - 1e-6)    # just under: PDFium rounds the size up
    except Exception:
        pass
    cap = min(max_side / max(w, h), (max_pixels / (w * h)) ** 0.5)
    if scale > cap:
        scale = cap * 0.999                        # PDFium rounds the size up
    img = page.render(scale=scale, rotation=0).to_pil().convert("RGB")
    buf = io.BytesIO()
    img.save(buf, "PNG", compress_level=1)         # lossless, like the scan inside the PDF
    if buf.tell() > max_bytes:
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=95)
    b = buf.getvalue()
    out.write(struct.pack(">I", len(b))); out.write(b)
    page.close()
doc.close()
out.flush()
'''

_MESSAGES = {
    "bad_pdf": "The PDF could not be opened. Send the document as a photo or scan (JPEG or PNG), or as a different PDF.",
    "pdf_encrypted": "The PDF is protected with a password. Send a PDF without a password, or a photo of the document.",
    "pdf_empty": "The PDF has no pages.",
    "pdf_too_complex": "The PDF took too long or needed too much memory to open. Send a simpler scan, or a photo (JPEG or PNG).",
}


def render(data: bytes, first_page_only: bool = False) -> list[bytes]:
    """The pages of a PDF as images (only the first with first_page_only). Raises PdfError."""
    pages_max = config.max_pdf_pages()
    args = [str(config.pdf_render_memory_mb() * 1024 * 1024), str(config.pdf_render_seconds()), str(pages_max),
            str(config.pdf_dpi()), str(config.pdf_max_side()), str(config.max_pixels()), "1" if first_page_only else "0",
            str(config.max_bytes())]
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "OMP_NUM_THREADS": "1", "LANG": "C.UTF-8"}
    try:
        proc = subprocess.run([sys.executable, "-I", "-c", _CHILD, *args], input=data, capture_output=True,
                              timeout=config.pdf_render_seconds() + 5, env=env, cwd="/", check=False)
    except subprocess.TimeoutExpired:
        raise PdfError("pdf_too_complex", _MESSAGES["pdf_too_complex"]) from None
    if proc.returncode == 3:
        code, _, detail = proc.stderr.decode("utf-8", "replace").strip().splitlines()[-1].partition("|")
        if code == "pdf_too_many_pages":
            raise PdfError(code, f"The PDF has {detail} pages; send at most {pages_max}, one page per side of the document.")
        raise PdfError(code if code in _MESSAGES else "bad_pdf", _MESSAGES.get(code, _MESSAGES["bad_pdf"]))
    if proc.returncode != 0:                       # killed by a limit (memory, CPU) or crashed inside PDFium
        raise PdfError("pdf_too_complex", _MESSAGES["pdf_too_complex"])
    out, pos, pages = proc.stdout, 0, []
    while pos + 4 <= len(out):
        (size,) = struct.unpack(">I", out[pos:pos + 4])
        pages.append(out[pos + 4:pos + 4 + size])
        pos += 4 + size
    if not pages or pos != len(out):
        raise PdfError("bad_pdf", _MESSAGES["bad_pdf"])
    return pages
