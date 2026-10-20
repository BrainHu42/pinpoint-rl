import os, sys
sys.path.insert(0, "/home/brianhu/pinpoint-rl/.claude/worktrees/density-probe/src")
os.chdir("/home/brianhu/pinpoint-rl/.claude/worktrees/density-probe")
import numpy as np
from geo_search_env.experiment.clue_verify import _haversine
from geo_search_env.experiment.osv_propose import _clusters
d = np.load("artifacts/osv_propose/search.npz")
ans = np.load("artifacts/agent/action_agent_apply_km_setft_own_1km0.5_commons26.npz")
print("OSV areas (agreement clusters of the OSV top 50): a right one that no pipeline area covers, % of set")
for st, key in (("dev3k", "dev3k|RL policy, lambda 0"), ("commons26", "commons26:setft_own|RL policy, lambda 0")):
    m = np.flatnonzero(d["set"] == st); truth = d["truth"][m]; pipe = ans[key][d["photo"][m]]; wrong = pipe >= 25
    zm = np.load(f"artifacts/pinpoint_set/zoom_{st}_setft_own.npz")
    areas = {}
    for p, c in zip(zm["photo"], zm["center"]):
        areas.setdefault(int(p), []).append(c)
    res = {1: [], 2: [], 3: [], 5: []}; n_new = {k: [] for k in res}
    for j in range(len(m)):
        cs = _clusters(d["osv_ll"][m][j], d["osv_sim"][m][j])
        pa = areas.get(int(d["photo"][m][j]), [])
        area_ok = any(_haversine(truth[j, 0], truth[j, 1], c[0], c[1]) < 25 for c in pa)
        new = [c for c in cs if all(_haversine(c[0], c[1], a[0], a[1]) >= 25 for a in pa)]  # OSV areas no pipeline area covers
        for k in res:
            top = new[:k]
            n_new[k].append(len(top))
            res[k].append(wrong[j] and not area_ok and any(_haversine(truth[j, 0], truth[j, 1], c[0], c[1]) < 25 for c in top))
    print(f"  {st:9s} " + " | ".join(f"top {k} new OSV areas: right {100 * np.mean(v):4.1f}% (avg {np.mean(n_new[k]):.1f} added)" for k, v in res.items()))
