"""Photograph-like test: the whole document is visible but tilted in 3D on a dark table. Run with the engine installed."""
import io
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
import make_scenarios as ms
from fastapi.testclient import TestClient

import app.main as m

c = TestClient(m.app)
IDS = ("passport_sau", "passport_gbr", "passport_ind", "visa_fra", "greencard_usa", "idcard_td2_egy", "visa_mrvb_deu")
QUADS = {  # destination corners as fractions of a 1.3x canvas
    "tilt_left": [(.10, .14), (.93, .06), (.90, .94), (.07, .84)],
    "tilt_right": [(.07, .06), (.90, .15), (.93, .85), (.10, .94)],
    "strong": [(.16, .12), (.88, .04), (.95, .96), (.05, .80)],
}


def shot(doc, quad):
    w, h = doc.size
    W, H = int(w * 1.3), int(h * 1.3)
    dst = np.float32([(x * W, y * H) for x, y in quad])
    mat = cv2.getPerspectiveTransform(np.float32([[0, 0], [w, 0], [w, h], [0, h]]), dst)
    table = np.full((H, W, 3), (55, 60, 70), np.uint8)
    out = cv2.warpPerspective(np.array(doc), mat, (W, H), dst=table, borderMode=cv2.BORDER_TRANSPARENT)
    return Image.fromarray(out)


bad = silent = 0
total = 0
for it in (i for i in ms.load_mrz_samples() if i["id"] in IDS):
    it["label"] = it["id"].upper()
    doc = ms.draw_document(it)
    for name, quad in QUADS.items():
        b = io.BytesIO()
        shot(doc, quad).save(b, "JPEG", quality=90)
        jid = c.post("/v1/documents/extract", files=[("files", ("x.jpg", b.getvalue(), "image/jpeg"))]).json()["jobId"]
        for _ in range(300):
            j = c.get(f"/v1/jobs/{jid}").json()
            if j["status"] in ("done", "failed"):
                break
            time.sleep(.3)
        total += 1
        if j["status"] != "done":
            bad += 1
            print(it["id"], name, "NOT_READ")
            continue
        d = j["result"]["document"]
        wrong = [k for k, v in it["expected"].items() if d.get(k) != v]
        if wrong and not j["result"]["needsReview"]:
            silent += 1
        print(it["id"], name, "OK" if not wrong else ("WRONG_FLAGGED" if j["result"]["needsReview"] else "WRONG_SILENT"), wrong)
print(f"total {total}  not read {bad}  silent wrong {silent}")
