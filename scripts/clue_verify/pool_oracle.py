import json, os, sys
sys.path.insert(0, "/home/brianhu/pinpoint-rl/.claude/worktrees/density-probe/src")
os.chdir("/home/brianhu/pinpoint-rl/.claude/worktrees/density-probe")
import numpy as np
from geo_search_env.experiment.clue_verify import _haversine
d = np.load("artifacts/osv_propose/search_contrastive.npz")
for st, tag in (("dev3k", "dev3k_setft_own"), ("commons26", "commons26_setft_own")):
    photos = json.load(open(f"artifacts/photo_sets/{tag}/photos.json"))
    m = np.flatnonzero(d["set"] == st)
    res = {}
    for k in (0, 4, 8, 16, 32):
        ok1, ok25 = [], []
        for j in m:
            p = photos[int(d["photo"][j])]; t = p["truth"]
            pts = np.asarray(p["pool"], float).reshape(-1, 2)
            if k:
                pts = np.concatenate([pts, d["osv_ll"][j][:k]])
            dist = _haversine(t[0], t[1], pts[:, 0], pts[:, 1])
            ok1.append((dist < 1).any()); ok25.append((dist < 25).any())
        res[k] = (np.mean(ok1), np.mean(ok25))
    pool_n = np.mean([len(photos[int(d['photo'][j])]['pool']) for j in m])
    print(f"{st:9s} pool ({pool_n:.1f} candidates) oracle < 1 / 25 km: " + " | ".join(f"+{k} OSV {100*a:.1f} / {100*b:.1f}" for k, (a, b) in res.items()))
