import os, sys
sys.path.insert(0, "/home/brianhu/pinpoint-rl/.claude/worktrees/density-probe/src")
os.chdir("/home/brianhu/pinpoint-rl/.claude/worktrees/density-probe")
import numpy as np
from geo_search_env.experiment.clue_verify import _haversine
d = np.load("artifacts/osv_propose/search.npz")
ans = np.load("artifacts/agent/action_agent_apply_km_setft_own_1km0.5_commons26.npz")
rng = np.random.default_rng(0)
for st, key in (("dev3k", "dev3k|RL policy, lambda 0"), ("commons26", "commons26:setft_own|RL policy, lambda 0")):
    m = np.flatnonzero(d["set"] == st); truth = d["truth"][m]; pipe = ans[key][d["photo"][m]]; wrong = pipe >= 25
    for src in ("osv", "mp16"):
        ll = d[f"{src}_ll"][m]
        real = _haversine(truth[:, None, 0], truth[:, None, 1], ll[..., 0], ll[..., 1])
        out = []
        for k in (5, 20, 50):
            r = (real[:, :k] < 25).any(1)
            ch = []
            for _ in range(5):
                perm = rng.permutation(len(m))
                fake = _haversine(truth[:, None, 0], truth[:, None, 1], ll[perm, :k, 0], ll[perm, :k, 1])
                ch.append(np.mean(wrong & (fake < 25).any(1)))
            out.append(f"top {k}: real {100*np.mean(wrong & r):4.1f} vs chance {100*np.mean(ch):4.1f}")
        print(f"{st:9s} {src:4s} pipeline-wrong photos with a right candidate (% of set) | " + " | ".join(out))
