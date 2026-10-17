import csv
import json
import os
import sys

sys.path.insert(0, "/home/brianhu/pinpoint-rl/.claude/worktrees/density-probe/src")
os.chdir("/home/brianhu/pinpoint-rl/.claude/worktrees/density-probe")
import numpy as np
from geo_search_env.experiment.clue_verify import _answers, _auc, _haversine, RIGHT_KM

rows = json.load(open("artifacts/clue_verify/commons26_dev_osv.json"))
km, _, _ = _answers("commons26")
meta = {r["IMG_ID"]: r for r in csv.DictReader(open("data/pinpoint/commons26/release/benchmark.csv"))}
base = np.asarray([km[r["photo"]] < RIGHT_KM for r in rows])
print(f"commons26 dev {len(rows)} photos; pipeline < 25 km {100 * base.mean():.1f}%")
for kind in ("text", "area"):
    pl = [p for r in rows for p in r["places"] if p["kind"].startswith(kind)]
    print(f"{kind} places {len(pl)}: OSV image within 1 km {100 * np.mean([p['n_1'] > 0 for p in pl]):.1f}%, within 5 km {100 * np.mean([p['n_5'] > 0 for p in pl]):.1f}%")
print("\nAUC right vs wrong (places with an OSV image within the radius):")
for kind in ("text", "area"):
    for key in ("ml",):
        pl = [p for r in rows for p in r["places"] if p["kind"].startswith(kind) and p[key] > -1]
        print(f"  {kind:5s} {key}: {_auc([p[key] for p in pl if p['right']], [p[key] for p in pl if not p['right']]):.3f}  "
              f"(right {sum(p['right'] for p in pl)}, wrong {sum(not p['right'] for p in pl)})")
    if kind == "text":
        pl = [p for r in rows for p in r["places"] if p["kind"] == "text"]
        print(f"  text  weight: {_auc([p['weight'] for p in pl if p['right']], [p['weight'] for p in pl if not p['right']]):.3f}")
print("\nwithin-photo top pick right (photos with a right and a wrong candidate of that kind, OSV within 5 km for all):")
for kind, extra in (("text", "weight"), ("area0", "rank")):
    hits = {"ml": [], extra: []}
    for r in rows:
        pl = [p for p in r["places"] if p["kind"].startswith(kind)]
        for p in pl:
            p["rank"] = -int(p["kind"].split("_")[1]) if p["kind"].startswith("area") else 0
        if pl and any(p["right"] for p in pl) and not all(p["right"] for p in pl):
            for k in hits:
                hits[k].append(max(pl, key=lambda p: p[k])["right"])
    print(f"  {kind:5s} " + ", ".join(f"{k} {100 * np.mean(v):.1f}%" for k, v in hits.items()) + f" (n={len(hits[extra])})")
print("\nby tier, area places: AUC ml right vs wrong")
for tier in ("landmark", "city", "region", "none"):
    pl = [p for r in rows if meta[r["image_id"]]["tier"] == tier for p in r["places"] if p["kind"].startswith("area") and p["ml"] > -1]
    print(f"  {tier:8s} {_auc([p['ml'] for p in pl if p['right']], [p['ml'] for p in pl if not p['right']]):.3f} (right {sum(p['right'] for p in pl)}, wrong {sum(not p['right'] for p in pl)})")

groups = np.asarray([meta[r["image_id"]]["group"] for r in rows]); ug = np.unique(groups)
fold = {x: k % 5 for k, x in enumerate(np.random.default_rng(0).permutation(ug))}
f = np.asarray([fold[x] for x in groups])


def apply(r, kinds, key, t, m):
    a = next((p for p in r["places"] if p["kind"] == "answer"), None)
    pl = [p for p in r["places"] if any(p["kind"].startswith(k) for k in kinds) and p[key] > -1]
    if pl and a is not None:
        b = max(pl, key=lambda p: p[key])
        if b[key] >= t and b[key] - a[key] >= m and _haversine(b["lat"], b["lon"], a["lat"], a["lon"]) >= RIGHT_KM:
            return b["right"], True
    return km[r["photo"]] < RIGHT_KM, False


grid = [(t, m) for t in np.arange(0.05, 0.8, 0.0125) for m in (0.0, 0.025, 0.05, 0.1, 0.15)]
print("\nswitch from the answer to the best-scoring candidate (score >= t, beats the answer by >= m, > 25 km away); grouped 5-fold CV:")
for kinds, label in ((("text",), "text places"), (("area",), "Pinpoint / clue areas"), (("text", "area"), "both")):
    for key in ("ml",):
        res, sw = np.zeros(len(rows), bool), np.zeros(len(rows), bool)
        for k in range(5):
            tr = np.flatnonzero(f != k)
            best = max(grid, key=lambda tm: sum(apply(rows[i], kinds, key, *tm)[0] for i in tr))
            for i in np.flatnonzero(f == k):
                res[i], sw[i] = apply(rows[i], kinds, key, *best)
        ceil = np.mean([b or any(p["right"] for p in r["places"] if any(p["kind"].startswith(k) for k in kinds)) for r, b in zip(rows, base)])
        print(f"  {label:22s} {key}: {100 * base.mean():.1f} -> {100 * res.mean():.1f}% ({100 * (res.mean() - base.mean()):+.1f}; switched {sw.sum()}, "
              f"fixes {int((res & ~base).sum())}, breaks {int((~res & base).sum())}); ceiling {100 * ceil:.1f}%")
