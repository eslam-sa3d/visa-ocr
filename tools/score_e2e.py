"""Summarise tools/e2e_accuracy.py output: python tools/score_e2e.py res.json"""
import json, sys, collections
R = json.load(open(sys.argv[1]))
def cls(r):
    if r["status"] != "done": return "not_read"
    if not r["wrong"]: return "correct_auto" if not r["needsReview"] else "correct_review"
    return "wrong_flagged" if r["needsReview"] else "WRONG_SILENT"
def table(key):
    t = collections.defaultdict(collections.Counter)
    for r in R: t[r[key]][cls(r)] += 1; t["ALL"][cls(r)] += 1
    for k, v in t.items():
        n = sum(v.values()); print(f"{k:6} n={n:3} " + "  ".join(f"{c}={v[c]} ({100*v[c]/n:.0f}%)" for c in ("correct_auto","correct_review","wrong_flagged","WRONG_SILENT","not_read")))
table("fmt"); print(); table("variant")
fa, fn = collections.Counter(), collections.Counter()
for r in R:
    if r["status"] == "done":
        for f, v in r["doc"].items():
            if f in r["truth"]: fn[f] += 1; fa[f] += (v or "") == (r["truth"][f] or "")
print({f: f"{fa[f]}/{fn[f]} ({100*fa[f]/fn[f]:.0f}%)" for f in fn})
print("max sec", max(r["sec"] for r in R), "mean", round(sum(r["sec"] for r in R)/len(R),1))
