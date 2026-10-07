# Lesson 53: is the knowledge model's region miss on the photos retrieval fails on a coverage problem? Region accuracy by how many SFT training
# photos share the true region, on region-fixable vs reranker-right photos (full eval halves). CPU, seconds.
# Usage: PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.knowledge_coverage

"""Is the knowledge model's region miss on the fixable photos a coverage problem (true region rare in the 250k SFT photos) or a photo problem?"""

import collections
import json

import numpy as np

from geo_search_env.experiment.knowledge_check import LEVELS, OUT, _norm, _right
from geo_search_env.experiment.name_score import _labels
from geo_search_env.experiment.photo_sets import load_photos
from geo_search_env.experiment.strategy_search import _haversine_km

train = collections.Counter()
for line in open("artifacts/knowledge/knowledge.jsonl"):
    r = json.loads(line)
    if r["split"] == "train":
        parts = r["target"].split(" > ")
        if len(parts) >= 2:
            train[(_norm(parts[0]), _norm(parts[1]))] += 1
print(f"{sum(train.values())} train photos, {len(train)} (country, region) pairs")

photos = load_photos("full")
labels = {p["image_id"]: p for p in _labels("full")}
answers = {m: json.loads((OUT / f"check_{m}.json").read_text()) for m in ("knowledge-a", "base")}
rows = []
for e in photos:
    lab = labels[e["image_id"]]
    t = lab["truth"]
    if not (t.get("country") and t.get("region")):
        continue
    km = _haversine_km(*e["truth"], np.asarray(e["pool"]))
    hit = next((j for j, c in enumerate(lab["pool"]) if _right(c.get("country") or "", t["country"]) and _right(c.get("region") or "", t["region"])), None)
    fixable = km[0] >= 25 and hit is not None and km[hit] < 25
    count = train[(_norm(t["country"]), _norm(t["region"]))]
    right = {}
    for m, ans in answers.items():
        parts = [x.strip() for x in (ans.get(e["image_id"]) or "").split(">")]
        right[m] = len(parts) >= 2 and _right(parts[0], t["country"]) and _right(parts[1], t["region"])
    rows.append((fixable, km[0] < 25, count, right["knowledge-a"], right["base"]))

rows = np.array(rows, dtype=float)
fix, rr, cnt, ka, ba = rows.T
bins = [(0, 1), (1, 50), (50, 300), (300, 1500), (1500, 10**9)]
print("\nregion right (knowledge-a / base 4B) by training photos in the photo's true region:")
print(f"{'train photos':>14s} | {'fixable photos':>26s} | {'reranker-right photos':>26s}")
for lo, hi in bins:
    f, r = fix.astype(bool) & (cnt >= lo) & (cnt < hi), rr.astype(bool) & (cnt >= lo) & (cnt < hi)
    fmt = lambda m: f"n={int(m.sum()):4d}  {ka[m].mean():5.1%} / {ba[m].mean():5.1%}" if m.sum() else "-"
    print(f"{f'{lo}-{hi if hi < 10**9 else chr(8734)}':>14s} | {fmt(f):>26s} | {fmt(r):>26s}")
print(f"\nmedian training photos in true region: fixable {np.median(cnt[fix == 1]):.0f}, reranker-right {np.median(cnt[rr == 1]):.0f}")
