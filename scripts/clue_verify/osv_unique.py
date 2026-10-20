import os, sys
sys.path.insert(0, "/home/brianhu/pinpoint-rl/.claude/worktrees/density-probe/src")
os.chdir("/home/brianhu/pinpoint-rl/.claude/worktrees/density-probe")
import numpy as np
from geo_search_env.experiment.clue_verify import _haversine

d = np.load("artifacts/osv_propose/search.npz")
ans = np.load("artifacts/agent/action_agent_apply_km_setft_own_1km0.5_commons26.npz")
print("unique OSV contribution: pipeline wrong, OSV top k has a right place, MP16 top 50 has none and no pipeline area is right")
for st, key in (("dev3k", "dev3k|RL policy, lambda 0"), ("commons26", "commons26:setft_own|RL policy, lambda 0")):
    m = np.flatnonzero(d["set"] == st); truth = d["truth"][m]; pipe = ans[key][d["photo"][m]]; wrong = pipe >= 25
    zm = np.load(f"artifacts/pinpoint_set/zoom_{st}_setft_own.npz")
    area_ok = np.zeros(len(m), bool)
    pos = {int(p): j for j, p in enumerate(d["photo"][m])}
    for p, c in zip(zm["photo"], zm["center"]):
        j = pos.get(int(p))
        if j is not None and _haversine(truth[j, 0], truth[j, 1], c[0], c[1]) < 25:
            area_ok[j] = True
    mp = (_haversine(truth[:, None, 0], truth[:, None, 1], d["mp16_ll"][m][..., 0], d["mp16_ll"][m][..., 1]) < 25).any(1)
    od = _haversine(truth[:, None, 0], truth[:, None, 1], d["osv_ll"][m][..., 0], d["osv_ll"][m][..., 1])
    print(f"  {st:9s} pipeline wrong {100 * wrong.mean():.1f}%; no pipeline area right {100 * np.mean(wrong & ~area_ok):.1f}%; "
          f"and MP16 top 50 none either {100 * np.mean(wrong & ~area_ok & ~mp):.1f}%")
    for k in (5, 20, 50):
        o = (od[:, :k] < 25).any(1)
        print(f"     OSV top {k:2d}: right where nothing else is {100 * np.mean(wrong & ~area_ok & ~mp & o):4.1f}% of set; "
              f"right where no area is (MP16 top 50 may be) {100 * np.mean(wrong & ~area_ok & o):4.1f}%")
