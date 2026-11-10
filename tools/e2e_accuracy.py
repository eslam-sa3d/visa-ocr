"""End-to-end accuracy: render synthetic documents (many countries/formats/variants), run the real pipeline, score field by field.
Synthetic fonts only -- measures the pipeline, not real-world camera quality."""
import sys, json, random, time, os
from pathlib import Path
from multiprocessing import Pool
sys.path.insert(0, str(Path(__file__).parent)); sys.path.insert(0, str(Path(__file__).parent.parent))
import io
os.environ.setdefault("OCR_ENGINE", "ensemble")
import accuracy_matrix as A
import make_scenarios as M
from app import pipeline
from app.review import Applicant
from app.engines import get_engine

FIELDS = ["documentNumber", "surname", "givenNames", "nationality", "birthDate", "expiryDate", "sex", "issuingState"]
UNPROT = {"surname", "givenNames", "nationality", "sex", "issuingState"}

def job(arg):
    fmt, k, vname, seed = arg
    rng = random.Random(seed)
    kw = A.rnd_case(rng, fmt)
    if fmt in ("TD1", "TD2"):
        kw["doc"] = A.rnd_doc(rng, fmt)
    lines, truth = A.build(fmt, **kw)
    img = M.VARIANTS[vname](M.draw_document({"lines": lines, "label": f"{fmt} TEST"}))
    b = io.BytesIO(); img.convert("RGB").save(b, "JPEG", quality=90)
    t = time.time()
    status, res = pipeline.run([b.getvalue()], None, Applicant(), get_engine())
    out = {"fmt": fmt, "variant": vname, "status": status, "sec": round(time.time() - t, 1), "truth": truth, "lines": lines}
    if status == "done":
        doc = res.get("document", res)
        out["doc"] = {f: doc.get(f) for f in FIELDS}
        out["needsReview"] = res.get("needsReview")
        out["conf"] = (res.get("confidence") or {}).get("mrz")
        out["engine"] = res.get("engine")
        out["flags"] = [(f["code"], f["severity"]) for f in res.get("flags", [])]
        out["wrong"] = [f for f in FIELDS if f in truth and (doc.get(f) or "") != (truth[f] or "")]
    else:
        out["res"] = res
    return out

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    jobs = [(fmt, k, v, 1000 + i * 7 + k) for i, fmt in enumerate(["TD3", "TD2", "TD1", "MRVA", "MRVB"])
            for k in range(n) for v in ("clean", "rot+4", "blur", "noise")]
    with Pool(2) as p:
        res = p.map(job, jobs, chunksize=1)
    Path(sys.argv[2] if len(sys.argv) > 2 else "/tmp/claude-0/e2e/res.json").write_text(json.dumps(res, default=str))
    print(len(res), "done")
