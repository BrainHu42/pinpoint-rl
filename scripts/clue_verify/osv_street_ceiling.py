"""Ceiling of the street-search areas: how often an OSV area (source 2) is right (< 25 km: its restricted top-1 or agreement pick) where the
baseline agent (setft_own, --one-km 0.5, RL lambda 0) is wrong, on dev3k and commons26 dev.

Usage: PYTHONPATH=src .venv/bin/python scripts/clue_verify/osv_street_ceiling.py
"""
import numpy as np

base = np.load("artifacts/agent/action_agent_apply_km_setft_own_1km0.5_commons26dev.npz")
for tag, key in (("dev3k", "dev3k"), ("commons26dev", "commons26dev:setft_own")):
    f = np.load(f"artifacts/agent/hypothesis_features_{tag}_setft_own_e4_moreosv_set.npz")
    pipe = base[f"{key}|RL policy, lambda 0"]
    n = len(pipe)
    osv = f["source"] == 2
    ok = np.zeros(n, bool)
    ok_any = np.zeros(n, bool)
    np.logical_or.at(ok, f["photo"][osv], f["km"][osv] < 25)
    np.logical_or.at(ok_any, f["photo"][osv], (f["km"][osv] < 25) | (f["km_agree"][osv] < 25))
    old = ~osv
    old_ok = np.zeros(n, bool)
    np.logical_or.at(old_ok, f["photo"][old], (f["km"][old] < 25) | (f["km_agree"][old] < 25))
    wrong = pipe >= 25
    print(f"{tag}: baseline RL < 25 km {np.mean(~wrong):.1%}; an OSV area right (top-1 / top-1 or agreement) {ok.mean():.1%} / {ok_any.mean():.1%}; "
          f"right where the baseline is wrong {np.mean(ok_any & wrong):.1%}, and no existing area is right {np.mean(ok_any & wrong & ~old_ok):.1%}")
