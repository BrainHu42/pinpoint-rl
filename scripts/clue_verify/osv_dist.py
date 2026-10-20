import json, os, sys
sys.path.insert(0, "/home/brianhu/pinpoint-rl/.claude/worktrees/density-probe/src")
os.chdir("/home/brianhu/pinpoint-rl/.claude/worktrees/density-probe")
import numpy as np
from geo_search_env.experiment.clue_verify import _haversine, OSV_EMBED
rows = json.load(open("artifacts/clue_verify/commons26_dev_osv.json"))
photos = json.load(open("artifacts/photo_sets/commons26_setft_own/photos.json"))
n = 4894684
ll = np.fromfile(OSV_EMBED / "latlon_deg.f32.bin", dtype=np.float32).reshape(n, 2)
size = 0.05
cell = (np.floor(ll[:, 0] / size).astype(np.int64) + 10000) * 100000 + (np.floor(ll[:, 1] / size).astype(np.int64) + 10000)
o = np.argsort(cell, kind="stable"); cs = cell[o]
d_near, cnt = [], {0.05: [], 0.1: [], 0.25: [], 1.0: []}
for r in rows:
    la, lo = photos[r["photo"]]["truth"]
    ci, cj = int(np.floor(la / size)), int(np.floor(lo / size))
    idx = np.concatenate([o[np.searchsorted(cs, (ci+di+10000)*100000+cj+dj+10000):np.searchsorted(cs, (ci+di+10000)*100000+cj+dj+10000, "right")] for di in (-1,0,1) for dj in (-1,0,1)])
    d = _haversine(la, lo, ll[idx, 0], ll[idx, 1]) if len(idx) else np.array([99.0])
    d_near.append(d.min() if len(d) else 99.0)
    for k in cnt: cnt[k].append((d < k).sum())
d_near = np.asarray(d_near)
print("commons26 dev: nearest OSV-5M image to the true location")
for k in (0.05, 0.1, 0.25, 1.0, 5.0):
    print(f"  within {k*1000:5.0f} m: {100*np.mean(d_near < k):5.1f}% of photos")
for k, v in cnt.items():
    print(f"  median OSV images within {k*1000:.0f} m: {np.median(v):.0f}")
ml = [p["ml"] for r in rows for p in r["places"] if p["ml"] > -1]
print("MegaLoc best-of-5 per place: median %.3f p90 %.3f p99 %.3f" % (np.median(ml), np.quantile(ml, .9), np.quantile(ml, .99)))
