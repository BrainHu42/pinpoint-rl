import csv
import json
import os
import sys

sys.path.insert(0, "/home/brianhu/pinpoint-rl/.claude/worktrees/density-probe/src")
os.chdir("/home/brianhu/pinpoint-rl/.claude/worktrees/density-probe")
import numpy as np
from geo_search_env.experiment.clue_verify import _answers, _auc, _scores, _haversine, RIGHT_KM

tier = sys.argv[1] if len(sys.argv) > 1 else "landmark"
kp = {r["image_id"]: r for r in json.load(open(f"artifacts/clue_verify/commons26_dev_kp_{tier}.json"))}
rows = [r for r in _scores("commons26", "dev") if r["image_id"] in kp]
km, _, _ = _answers("commons26")
groups = {r["IMG_ID"]: r["group"] for r in csv.DictReader(open("data/pinpoint/commons26/release/benchmark.csv"))}
for r in rows:  # attach best inliers per place (same order as the shortlist)
    for p, q in zip(r["places"], kp[r["image_id"]]["places"]):
        v = [x for x in q["kp"] if x >= 0]
        p["kp"] = max(v) if v else -1
        p["kp_text"] = max([x for x, i in zip(q["kp"], q["kp_ids"]) if i in p["text_ids"] and x >= 0], default=-1)
text = [p for r in rows for p in r["places"] if p["kind"] == "text"]
pos, neg = [p for p in text if p["right"]], [p for p in text if not p["right"]]
print(f"tier {tier}: {len(rows)} photos, text places right {len(pos)} / wrong {len(neg)}")
for key in ("kp", "kp_text", "sim", "weight"):
    print(f"  AUC {key:8s} {_auc([p[key] for p in pos], [p[key] for p in neg]):.3f}")
print("  best inliers, right vs wrong places:")
for t in (15, 25, 40, 60, 100):
    print(f"    >= {t:3d}: right {100 * np.mean([p['kp'] >= t for p in pos]):5.1f}%  wrong {100 * np.mean([p['kp'] >= t for p in neg]):5.1f}%")
ans = [(p, km[r["photo"]] < RIGHT_KM) for r in rows for p in r["places"] if p["kind"] == "answer"]
print("  pipeline answer place, best inliers median when right / wrong: %.0f / %.0f" % (
    np.median([p["kp"] for p, ok in ans if ok]), np.median([p["kp"] for p, ok in ans if not ok])))
hits = {k: [] for k in ("kp", "sim", "weight")}
for r in rows:
    t = [p for p in r["places"] if p["kind"] == "text"]
    if any(p["right"] for p in t) and not all(p["right"] for p in t):
        for k in hits:
            hits[k].append(max(t, key=lambda p: (p[k], p["weight"]))["right"])
print("  within-photo, right text place on top: " + ", ".join(f"{k} {100 * np.mean(v):.1f}%" for k, v in hits.items()) + f" (n={len(hits['kp'])})")

base = np.asarray([km[r["photo"]] < RIGHT_KM for r in rows])


def apply(r, t, ratio):
    a = next((p for p in r["places"] if p["kind"] == "answer"), None)
    tp = [p for p in r["places"] if p["kind"] == "text" and p["kp"] >= 0]
    if tp:
        b = max(tp, key=lambda p: p["kp"])
        a_kp = a["kp"] if a else 0
        if b["kp"] >= t and b["kp"] >= ratio * max(a_kp, 1) and (a is None or _haversine(b["lat"], b["lon"], a["lat"], a["lon"]) >= RIGHT_KM):
            return b["right"], True
    return km[r["photo"]] < RIGHT_KM, False


grid = [(t, ra) for t in (10, 15, 20, 25, 30, 40, 50, 60, 80, 100, 150) for ra in (1.0, 1.5, 2.0, 3.0)]
g = np.asarray([groups[r["image_id"]] for r in rows]); ug = np.unique(g)
fold = {x: k % 5 for k, x in enumerate(np.random.default_rng(0).permutation(ug))}
f = np.asarray([fold[x] for x in g])
res, sw = np.zeros(len(rows), bool), np.zeros(len(rows), bool)
for k in range(5):
    tr = np.flatnonzero(f != k)
    best = max(grid, key=lambda tm: sum(apply(rows[i], *tm)[0] for i in tr))
    for i in np.flatnonzero(f == k):
        res[i], sw[i] = apply(rows[i], *best)
print(f"switch (grouped CV): {100 * base.mean():.1f} -> {100 * res.mean():.1f}% (switched {sw.sum()}, fixes {int((res & ~base).sum())}, breaks {int((~res & base).sum())}); "
      f"ceiling {100 * np.mean([b or any(p['right'] for p in r['places'] if p['kind'] == 'text') for r, b in zip(rows, base)]):.1f}%")
for t, ra in ((25, 1.5), (40, 1.5), (60, 2.0), (100, 2.0)):
    o = [apply(r, t, ra) for r in rows]; rr = np.asarray([x[0] for x in o]); ss = np.asarray([x[1] for x in o])
    print(f"  fixed t={t}, ratio={ra}: {100 * rr.mean():.1f}% (switched {ss.sum()}, fixes {int((rr & ~base).sum())}, breaks {int((~rr & base).sum())})")

print("\nfixable photos (pipeline wrong, a right text place exists) vs photos where the pipeline is right:")
for name, sel in (("fixable", lambda r, b: (not b) and any(p["right"] for p in r["places"] if p["kind"] == "text")),
                  ("pipeline right", lambda r, b: b)):
    rs = [r for r, b in zip(rows, base) if sel(r, b)]
    rp = [p for r in rs for p in r["places"] if p["kind"] == "text" and p["right"]]
    print(f"  {name:15s} photos {len(rs):3d}; right text places {len(rp):3d}: kp >= 40 {100 * np.mean([p['kp'] >= 40 for p in rp]):4.1f}%, "
          f"MegaLoc >= 0.2 {100 * np.mean([p['sim'] >= .2 for p in rp]):4.1f}%, has targeted photos {100 * np.mean([p['text_n'] > 0 for p in rp]):4.1f}%")
