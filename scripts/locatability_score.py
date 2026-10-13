"""Locatability ceiling (2026-10-09): score the blind by-eye pass (artifacts/agent/locatability_dev3k.json: 120 random dev3k photos, each with a
tier P / C / R / N / K / X and a guess, made without seeing the truth or Pinpoint) against the truth, Pinpoint, its pool oracle and Pinpoint
restricted to the true region (pinpoint_constrained). Sheets for a new pass: scripts/locatability_sheets.py <dir> <n>.
Usage: PYTHONPATH=src .venv/bin/python scripts/locatability_score.py"""
import collections
import json

import numpy as np

from geo_search_env.experiment.strategy_search import _haversine_km

study = json.load(open("artifacts/agent/locatability_dev3k.json"))["photos"]
sample = [p["image_id"] for p in study]
blind = {str(i): (p["tier"], p["guess"]) for i, p in enumerate(study)}
photos = {e["image_id"]: e for e in json.load(open("artifacts/photo_sets/dev3k/photos.json"))}
z = np.load("artifacts/agent/constrained_dev3k.npz", allow_pickle=True)
order = [e["image_id"] for e in json.load(open("artifacts/photo_sets/dev3k/photos.json"))]
arms = list(z["arms"])
true_region = {k: z["latlon"][arms.index("true region")][i] for i, k in enumerate(order)}
km = lambda a, b: _haversine_km(*a, np.asarray([b]))[0] if b is not None and np.isfinite(b).all() else np.inf
rows = []
for i, k in enumerate(sample):
    e = photos[k]
    tier, guess = blind[str(i)]
    rows.append({"tier": tier, "me": km(e["truth"], guess), "pin": km(e["truth"], e["pool"][0]),
                 "oracle": min(km(e["truth"], p) for p in e["pool"]), "region": km(e["truth"], true_region[k])})
n = len(rows)
print(f"{n} random dev3k photos")
print(f"  {'tier':6s} {'n':>3s} {'share':>6s} | < 25 km: {'blind':>5s} {'Pinpoint':>8s} {'pool oracle':>11s} {'true-region Pinpoint':>20s}")
for t in ("P", "C", "R", "N", "K", "X", "all"):
    sel = rows if t == "all" else [r for r in rows if r["tier"] == t]
    if not sel:
        continue
    f = lambda key: sum(r[key] < 25 for r in sel)
    print(f"  {t:6s} {len(sel):3d} {len(sel) / n:6.0%} | {f('me'):14d} {f('pin'):8d} {f('oracle'):11d} {f('region'):20d}")
both = sum(r["me"] < 25 or r["pin"] < 25 for r in rows)
print(f"  blind or Pinpoint right: {both} ({both / n:.0%}); blind right where Pinpoint wrong: {sum(r['me'] < 25 <= r['pin'] for r in rows)}; "
      f"Pinpoint right where blind wrong: {sum(r['pin'] < 25 <= r['me'] for r in rows)}")
