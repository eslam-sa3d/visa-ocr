#!/usr/bin/env python3
"""Generate the end-to-end scenario set: synthetic documents of every supported type, each in several
'bad photo' variants, plus negative cases. Writes scenarios/images/* and scenarios/manifest.json.

    python tools/make_scenarios.py        (needs Pillow, numpy, arabic-reshaper, python-bidi, and a font with Arabic glyphs)

All people are fictional. The images are rendered with a monospaced font, not the real OCR-B face, so results on
real documents will be lower: treat this suite as a regression and capability check, and add real specimens with
tools/run_scenarios.py --real-dir.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app.mrz import check_digit, parse_mrz  # noqa: E402
from app.visual_zone import saudi_id_check  # noqa: E402

OUT = ROOT / "scenarios"
FONT_DIRS = ["/usr/share/fonts/truetype/dejavu", "/System/Library/Fonts/Supplemental", "/Library/Fonts"]
rng = random.Random(7)


def font(name: str, size: int):
    for d in FONT_DIRS:
        p = Path(d) / name
        if p.exists():
            return ImageFont.truetype(str(p), size)
    raise SystemExit(f"font {name} not found; install DejaVu fonts")


MONO, SANS = "DejaVuSansMono.ttf", "DejaVuSans.ttf"


def cd(s: str) -> str:
    return str(check_digit(s))


# ------------------------------------------------------------------ MRZ builders (valid check digits)
def td2(surname, given, nat, doc, dob, sex, exp, opt=""):
    l1 = ("I<" + nat + surname + "<<" + given.replace(" ", "<")).ljust(36, "<")
    opt = opt.ljust(7, "<")
    body = doc.ljust(9, "<") + cd(doc.ljust(9, "<")) + nat + dob + cd(dob) + sex + exp + cd(exp) + opt
    l2 = body + cd(body[:10] + body[13:20] + body[21:43])
    return [l1, l2]


def mrvb(surname, given, issuing, nat, doc, dob, sex, exp, opt=""):
    l1 = ("V<" + issuing + surname + "<<" + given.replace(" ", "<")).ljust(36, "<")
    l2 = doc.ljust(9, "<") + cd(doc.ljust(9, "<")) + nat + dob + cd(dob) + sex + exp + cd(exp) + opt.ljust(8, "<")
    return [l1, l2]


def load_mrz_samples():
    items = []
    for s in json.loads((ROOT / "samples" / "mrz_samples.json").read_text(encoding="utf-8")):
        e = s["expected"]
        items.append({"id": s["id"], "type": s["kind"] + ":" + e["format"], "lines": s["lines"],
                      "expected": {k: e[k] for k in ("documentNumber", "surname", "givenNames", "nationality",
                                                     "birthDate", "expiryDate")}})
    extra = [
        ("idcard_td2_egy", "id_card:TD2", td2("SPECIMEN", "AHMED TEST", "EGY", "A1234567", "850702", "M", "321128", "OPT1234"),
         dict(documentNumber="A1234567", surname="SPECIMEN", givenNames="AHMED TEST", nationality="EGY", birthDate="1985-07-02", expiryDate="2032-11-28")),
        ("visa_mrvb_deu", "visa:MRVB", mrvb("SPECIMEN", "MARIA SAMPLE", "D<<", "IND", "L8989023", "920314", "F", "280630"),
         dict(documentNumber="L8989023", surname="SPECIMEN", givenNames="MARIA SAMPLE", nationality="IND", birthDate="1992-03-14", expiryDate="2028-06-30")),
    ]
    for id_, typ, lines, exp in extra:
        r = parse_mrz(lines)
        assert r.valid, (id_, r.checks)
        items.append({"id": id_, "type": typ, "lines": lines, "expected": exp})
    return items


# ------------------------------------------------------------------ rendering
def draw_document(item) -> Image.Image:
    lines = item["lines"]
    w = 1100 if len(lines[0]) >= 36 else 900
    h = 760
    img = Image.new("RGB", (w, h), (236, 233, 224))
    d = ImageDraw.Draw(img)
    d.rectangle([40, 40, w - 40, h - 40], outline=(150, 150, 140), width=3)
    d.rectangle([70, 110, 330, 420], fill=(190, 195, 200))               # photo placeholder
    d.text((70, 60), item["label"], fill=(40, 40, 60), font=font(SANS, 26))
    for i, t in enumerate(["Surname / Nom", "Given names / Prenoms", "Date of birth", "Nationality"]):
        d.text((370, 130 + i * 70), t, fill=(60, 60, 60), font=font(SANS, 22))
    size = 36 if len(lines[0]) <= 36 else 31 if len(lines[0]) <= 44 else 30
    size = min(size, int((w - 100) / (0.602 * max(len(l) for l in lines))))
    f = font(MONO, size)
    y = h - 90 - len(lines) * int(size * 1.5)
    for i, l in enumerate(lines):
        d.text((60, y + i * int(size * 1.5)), l, fill=(20, 20, 20), font=f)
    return img


def v_clean(i): return i
def v_rot_p4(i): return i.rotate(-4, Image.BICUBIC, expand=True, fillcolor=(245, 245, 245))
def v_rot_m3(i): return i.rotate(3, Image.BICUBIC, expand=True, fillcolor=(245, 245, 245))
def v_blur(i): return i.filter(ImageFilter.GaussianBlur(1.3))


def v_noise(i):
    a = np.array(i).astype(np.float32) + np.random.default_rng(3).normal(0, 16, (i.height, i.width, 3))
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def v_lowres(i): return i.resize((int(i.width * 0.55), int(i.height * 0.55)), Image.LANCZOS)
def v_lowcontrast(i): return Image.blend(i, Image.new("RGB", i.size, (150, 150, 150)), 0.55)


def v_glare(i):
    ov = Image.new("RGBA", i.size, (0, 0, 0, 0))
    ImageDraw.Draw(ov).ellipse([i.width * 0.55, i.height * 0.55, i.width * 1.05, i.height * 1.0], fill=(255, 255, 255, 120))
    return Image.alpha_composite(i.convert("RGBA"), ov.filter(ImageFilter.GaussianBlur(25))).convert("RGB")


def v_shear(i):
    return i.transform(i.size, Image.AFFINE, (1, 0.06, -20, 0.02, 1, 0), Image.BICUBIC, fillcolor=(245, 245, 245))


VARIANTS = {"clean": v_clean, "rot+4": v_rot_p4, "rot-3": v_rot_m3, "blur": v_blur, "noise": v_noise,
            "lowres": v_lowres, "lowcontrast": v_lowcontrast, "glare": v_glare, "shear": v_shear}


# ------------------------------------------------------------------ Iqama / GCC ID (no MRZ)
def make_id(prefix9: str) -> str:
    for c in range(10):
        if saudi_id_check(prefix9 + str(c)):
            return prefix9 + str(c)
    raise AssertionError


def ar(text: str) -> str:
    """Text ready for ImageDraw. With libraqm Pillow shapes and orders Arabic itself; without it, do it by hand."""
    from PIL import features
    if features.check("raqm"):
        return text
    import arabic_reshaper
    from bidi.algorithm import get_display
    return get_display(arabic_reshaper.reshape(text))


IQAMA_PEOPLE = [
    dict(id=make_id("261242825"), en="ALI HASSAN KHAN", ar="علي حسن خان", nat_ar="باكستان", nat="PAK", job="مهندس",
         birth="1990/03/12", expiry="2027/05/14", religion="الاسلام"),
    dict(id=make_id("245678901"), en="MARIAM NOOR SAEED", ar="مريم نور سعيد", nat_ar="مصر", nat="EGY", job="محاسب",
         birth="1988/11/02", expiry="2028/01/30", religion="الاسلام"),
]


def draw_iqama(p) -> Image.Image:
    w, h = 1000, 640
    img = Image.new("RGB", (w, h), (246, 243, 232))
    d = ImageDraw.Draw(img)
    fb, fl = font(SANS, 34), font(SANS, 26)

    def right(text, x_right, y, f):
        t = ar(text)
        d.text((x_right - d.textlength(t, font=f), y), t, fill=(20, 20, 20), font=f)

    def row(label, value, y, value_font=fb):
        right(label, 960, y, fl)
        t = ar(value) if any("؀" <= c <= "ۿ" for c in value) else value
        d.text((960 - d.textlength(ar(label), font=fl) - 30 - d.textlength(t, font=value_font), y - 4), t, fill=(0, 0, 0), font=value_font)

    right("هوية مقيم", 450, 30, font(SANS, 44))
    right(p["ar"], 960, 110, font(SANS, 40))
    d.text((960 - d.textlength(p["en"], font=font(SANS, 34)), 160), p["en"], fill=(0, 0, 0), font=font(SANS, 34))
    row("رقم الهوية", p["id"], 230)
    row("تاريخ الميلاد", p["birth"], 290)
    row("الجنسية", p["nat_ar"], 350)
    row("المهنة", p["job"], 410)
    row("تاريخ الانتهاء", p["expiry"], 470)
    row("الديانة", p["religion"], 530)
    d.rectangle([40, 120, 260, 400], fill=(190, 195, 200))
    return img


# ------------------------------------------------------------------ negatives
def negatives() -> list[tuple[str, Image.Image]]:
    blank = Image.new("RGB", (900, 600), "white")
    noise = Image.fromarray(np.random.default_rng(1).integers(0, 255, (600, 900, 3), dtype=np.uint8))
    text = Image.new("RGB", (900, 600), "white")
    ImageDraw.Draw(text).multiline_text((40, 60), "Dear applicant,\nPlease bring your documents\nto the visa centre on Monday.\n" * 3,
                                        fill="black", font=font(SANS, 30))
    tiny = Image.new("RGB", (60, 40), "white")
    return [("neg_blank", blank), ("neg_noise", noise), ("neg_letter_text", text), ("neg_tiny", tiny)]


def main() -> None:
    img_dir = OUT / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    for f in img_dir.glob("*"):
        f.unlink()
    scenarios = []
    for item in load_mrz_samples():
        item["label"] = item["id"].replace("_", " ").upper()
        base = draw_document(item)
        for vname, fn in VARIANTS.items():
            im = fn(base)
            fname = f"{item['id']}__{vname}." + ("png" if vname == "clean" else "jpg")
            if fname.endswith("png"):
                im.save(img_dir / fname)
            else:
                im.save(img_dir / fname, quality=88)
            scenarios.append({"id": f"{item['id']}__{vname}", "group": "mrz", "docType": item["type"], "variant": vname,
                              "image": f"images/{fname}", "expectStatus": "done", "expected": item["expected"]})
    for n, p in enumerate(IQAMA_PEOPLE):
        base = draw_iqama(p)
        for vname in ("clean", "rot+4", "blur", "lowres"):
            im = VARIANTS[vname](base)
            fname = f"iqama_{n + 1}__{vname}." + ("png" if vname == "clean" else "jpg")
            im.save(img_dir / fname) if fname.endswith("png") else im.save(img_dir / fname, quality=88)
            scenarios.append({"id": f"iqama_{n + 1}__{vname}", "group": "visual", "docType": "iqama:visual", "variant": vname, "hint": "iqama",
                              "image": f"images/{fname}", "expectStatus": "done",
                              "expected": {"idNumber": p["id"], "nameEn": p["en"], "nameAr": p["ar"], "nationalityCode": p["nat"],
                                           "birthDate": p["birth"].replace("/", "-"), "expiryDate": p["expiry"].replace("/", "-")}})
    for id_, im in negatives():
        im.save(img_dir / f"{id_}.png")
        scenarios.append({"id": id_, "group": "negative", "docType": "none", "variant": "negative",
                          "image": f"images/{id_}.png", "expectStatus": "failed", "expected": {}})
    (OUT / "manifest.json").write_text(json.dumps({"scenarios": scenarios}, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"{len(scenarios)} scenarios -> {OUT}")


if __name__ == "__main__":
    main()
