"""Street-search variant vs baseline agent (setft_own, --one-km 0.5): how often the RL answer is an OSV area (its centre equals an OSV
hypothesis centre) and how those answers fare against the baseline's on the same photos.

Usage: PYTHONPATH=src .venv/bin/python scripts/clue_verify/osv_street_picks.py
"""
import numpy as np

base = np.load("artifacts/agent/action_agent_apply_km_setft_own_1km0.5_commons26dev.npz")
var = np.load("artifacts/agent/action_agent_apply_km_setft_own_moreosv_1km0.5_commons26dev.npz")
for tag, key in (("dev3k", "dev3k"), ("commons26dev", "commons26dev:setft_own")):
    h = np.load(f"artifacts/agent/hypotheses_{tag}_setft_own_moreosv.npz")
    for lam in ("0", "0.005"):
        k = f"{key}|RL policy, lambda {lam}"
        km_v, km_b, centre = var[k], base[k], var[k + "|center"]
        n = len(km_v)
        osv_pick = np.zeros(n, bool)
        for p, c in zip(h["photo"], h["center"]):
            osv_pick[p] |= np.allclose(centre[p], c, atol=1e-4)
        o = osv_pick
        print(f"{tag} lambda {lam}: < 25 km variant {np.mean(km_v < 25):.1%} vs baseline {np.mean(km_b < 25):.1%}; answers from an OSV area "
              f"{o.mean():.1%} ({o.sum()}): right {np.mean(km_v[o] < 25):.1%} vs baseline on those photos {np.mean(km_b[o] < 25):.1%}; "
              f"other photos: variant {np.mean(km_v[~o] < 25):.1%} vs baseline {np.mean(km_b[~o] < 25):.1%}")
