"""Line images of machine-readable-zone text in OCR B, degraded the way scans and phone photos are, with their text.

    OCRB_FONT=/path/OCRB.ttf python make_lines.py <out folder> <how many> <seed>

Writes lNNNNN.png, lNNNNN.gt.txt and lNNNNN.box (the form Tesseract's lstm.train step reads) for each line."""
import io, os, random, sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "tools")); sys.path.insert(0, str(ROOT))
import accuracy_matrix as A
out, n, seed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
os.makedirs(out, exist_ok=True)
rng = random.Random(seed); nprng = np.random.default_rng(seed)
FONTS = [os.environ.get("OCRB_FONT", "font/OCRB.ttf")]     # the upright regular cut only: the other files are outline and slanted variants
ALNUM = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
def lines():
    fmt = rng.choice(["TD3"] * 5 + ["MRVA", "MRVB", "TD1", "TD2"])
    kw = A.rnd_case(rng, fmt)
    if fmt in ("TD1", "TD2"): kw["doc"] = A.rnd_doc(rng, fmt)
    ls, _ = A.build(fmt, **kw)
    if rng.random() < 0.25:                                # arbitrary characters, so look-alikes are learnt without context
        L = len(ls[0]); body = "".join(rng.choice(ALNUM) for _ in range(rng.randint(8, L - 4)))
        ls = [(body + "<" * L)[:L]]
    return ls
def render(text):
    size = rng.randint(34, 52)
    font = ImageFont.truetype(rng.choice(FONTS), size)
    track = rng.uniform(-0.02, 0.10) * size
    widths = [font.getlength(c) + track for c in text]
    W, H = int(sum(widths) + size * 1.6), int(size * rng.uniform(1.7, 2.3))
    bg = rng.randint(185, 250)
    img = Image.new("L", (W, H), bg)
    d = ImageDraw.Draw(img); x = size * rng.uniform(0.4, 0.9); y = (H - size) / 2 + rng.uniform(-3, 3)
    ink = rng.randint(10, 95)
    for c, w in zip(text, widths):
        d.text((x, y), c, font=font, fill=ink); x += w
    w = rng.random()                                        # stroke weight: printers lay the ink down thinner or heavier
    if w < 0.2: img = img.filter(ImageFilter.MaxFilter(3))
    elif w > 0.75: img = img.filter(ImageFilter.MinFilter(3))
    a = np.asarray(img, dtype=np.float32)
    if rng.random() < 0.6:                                  # security-print background: soft waves and a gradient
        yy, xx = np.mgrid[0:H, 0:W]
        a += rng.uniform(3, 14) * np.sin(xx / rng.uniform(6, 30) + yy / rng.uniform(5, 25)) + rng.uniform(-18, 18) * (xx / W)
    img = Image.fromarray(np.clip(a, 0, 255).astype("uint8"))
    if rng.random() < 0.7: img = img.rotate(rng.uniform(-1.2, 1.2), Image.BICUBIC, fillcolor=bg)
    if rng.random() < 0.6:                                  # low resolution: letters 9 to 22 pixels high, then enlarged again
        h = rng.randint(14, 34); small = img.resize((max(8, int(W * h / H)), h), Image.BILINEAR)
        img = small.resize((W, H), rng.choice([Image.BICUBIC, Image.LANCZOS, Image.BILINEAR]))
    if rng.random() < 0.6: img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.6)))
    a = np.asarray(img, dtype=np.float32)
    if rng.random() < 0.7: a += nprng.normal(0, rng.uniform(2, 11), a.shape)
    if rng.random() < 0.3: a = (a - 128) * rng.uniform(0.55, 0.9) + 128 + rng.uniform(-25, 15)     # washed out or dark
    img = Image.fromarray(np.clip(a, 0, 255).astype("uint8"))
    if rng.random() < 0.6:
        b = io.BytesIO(); img.save(b, "JPEG", quality=rng.randint(35, 90)); img = Image.open(io.BytesIO(b.getvalue())).convert("L")
    return img
i = 0
while i < n:
    for text in lines():
        if i >= n: break
        render(text).save(f"{out}/l{i:05d}.png")
        open(f"{out}/l{i:05d}.gt.txt", "w").write(text + "\n")
        i += 1
print("wrote", i, "to", out)
for k in range(i):
    w, h = Image.open(f"{out}/l{k:05d}.png").size
    text = open(f"{out}/l{k:05d}.gt.txt").read().strip()
    with open(f"{out}/l{k:05d}.box", "w") as fh:
        fh.write("".join(f"{c} 0 0 {w} {h} 0\n" for c in text) + f"\t 0 0 {w} {h} 0\n")
