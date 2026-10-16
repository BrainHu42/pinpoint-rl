"""Probe: offline Wikipedia (web_search) on commons26 dev: for each photo's 4B place names + queries, does any of the top-3 hits with a
coordinate land < 25 km? Split by whether the current read chain already has a right result."""
import collections
import csv
import json
import sys
import time

import numpy as np

from geo_search_env.experiment.refine_headroom import _hav
from geo_search_env.experiment.web_search import WebSearch

rows = {r["IMG_ID"]: r for r in csv.DictReader(open("data/pinpoint/commons26/release/benchmark.csv"))}
dev = [i for i, r in rows.items() if r["split"] == "dev"]
obs = json.load(open("artifacts/agent/observe_commons26_4b.json"))
qs = dict(json.load(open("artifacts/agent/queries_commons26_4b.json")))
ev = collections.defaultdict(list)
for r in json.load(open("artifacts/agent/region_evidence_commons26_4b.json")):
    ev[r["image_id"]].append(r)
photos = json.load(open("artifacts/photo_sets/commons26_setft_own/photos.json"))
z = np.load("artifacts/agent/action_agent_apply_km_setft_own_1km0.5_commons26.npz")
final = dict(zip([e["image_id"] for e in photos], z["commons26:setft_own|RL policy, lambda 0"]))
ws = WebSearch()
print("dbs", len(ws.dbs), flush=True)
res = {}
t0 = time.time()
for n, i in enumerate(dev):
    p = obs[i].get("parsed", {})
    queries = list(dict.fromkeys((p.get("place_names") or [])[:3] + (qs.get(i) or [])[:3]))
    lat, lon = float(rows[i]["LAT"]), float(rows[i]["LON"])
    best_top1, best_any = 1e9, 1e9
    for q in queries:
        hits, _ = ws.query(q, 3)
        for k, h in enumerate(hits):
            if h.lat is None:
                continue
            d = float(_hav(lat, lon, h.lat, h.lon))
            best_any = min(best_any, d)
            if k == 0:
                best_top1 = min(best_top1, d)
    res[i] = (best_top1, best_any, len(queries))
    if n % 100 == 0:
        print(n, f"{time.time() - t0:.0f}s", flush=True)
found = {i: any(e.get("km", 1e9) < 25 for e in ev[i]) for i in dev}
for label, sel in (("all dev", dev), ("read chain has a right result", [i for i in dev if found[i]]), ("read chain has none", [i for i in dev if not found[i]]),
                   ("pipeline misses", [i for i in dev if final[i] >= 25])):
    t1 = np.mean([res[i][0] < 25 for i in sel])
    ta = np.mean([res[i][1] < 25 for i in sel])
    print(f"{label:32s} n={len(sel):4d}  wiki top-1 hit < 25 km {t1:5.1%}  any of top-3 {ta:5.1%}")
for t in ("landmark", "city", "region", "none"):
    sel = [i for i in dev if rows[i]["tier"] == t and not found[i]]
    print(f"  no right result, tier {t:8s} n={len(sel):4d} wiki top-1 < 25 km {np.mean([res[i][0] < 25 for i in sel]):5.1%}")
json.dump(res, open("artifacts/commons26/wiki_probe_dev.json", "w"))
