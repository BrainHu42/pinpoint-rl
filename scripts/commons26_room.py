# Where is the room on commons26 dev, especially the geolocatable photos (tier landmark / city / region)? (retrieval track, lesson R6)
# Usage: PYTHONPATH=src .venv/bin/python scripts/commons26_room.py
#   needs retrieval_recall ranks --photo-set commons26dev_setft_own --tag released, the cached action_agent apply outputs and lesson 119's
#   text-search candidates (artifacts/agent/cands_commons26_dev.jsonl)
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")
from geo_search_env.experiment.area_model import _paths  # noqa: E402
from geo_search_env.experiment.hypothesis_agent import SOURCES  # noqa: E402
from geo_search_env.experiment.pinpoint_reference import load_pools  # noqa: E402

TAG, KEY = "commons26dev_setft_own", "commons26dev:setft_own"
A = np.load("artifacts/agent/action_agent_apply_km_setft_own_1km0.5_commons26dev.npz")
R = np.load(f"artifacts/retrieval/recall_{TAG}_released.npz")
photos = json.loads(Path(f"artifacts/photo_sets/{TAG}/photos.json").read_text())
n = len(photos)
truth = np.asarray([e["truth"] for e in photos])


def hav(a, b, c, d):
    a, b, c, d = map(np.radians, (a, b, c, d))
    h = np.sin((c - a) / 2) ** 2 + np.cos(a) * np.cos(c) * np.sin((d - b) / 2) ** 2
    return 2 * 6371 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


km, top1 = A[f"{KEY}|RL policy, lambda 0"], A[f"{KEY}|retrained"]
pools = load_pools("commons26_setft_own")
pool_ok = np.asarray([(hav(truth[i, 0], truth[i, 1], *pools["latlon"][e["index"]][pools["valid"][e["index"]]].T) < 25).any() for i, e in enumerate(photos)])
f = np.load(_paths("commons26dev", "setft_own")[0])
keep = np.isin(f["source"], SOURCES)
area_ok = np.zeros(n, bool)
np.logical_or.at(area_ok, f["photo"][keep], f["km_agree"][keep] < 25)
area_ok |= km < 25
# lesson 119's text-search clusters: any within 25 km of the truth
text_ok = np.zeros(n, bool)
pos = {e["image_id"]: i for i, e in enumerate(photos)}
for line in Path("artifacts/agent/cands_commons26_dev.jsonl").read_text().splitlines():
    rec = json.loads(line)
    i = pos.get(rec["image_id"])
    if i is None:
        continue
    c = [(cl[0], cl[1]) for q in rec["queries"] for s, r in q["src"].items() for cl in r["clusters"]]
    if c:
        c = np.asarray(c)
        text_ok[i] = (hav(truth[i, 0], truth[i, 1], c[:, 0], c[:, 1]) < 25).any()
bm = {r["IMG_ID"]: r for r in csv.DictReader(open("data/pinpoint/commons26/release/benchmark.csv"))}
B = [bm[e["image_id"]] for e in photos]
tier = np.asarray([b["tier"] for b in B])
ptext = np.asarray([b["place_text"] == "1" for b in B])
cont = np.asarray([b["continent"] for b in B])
mp16cos = np.asarray([float(b["mp16_cos"] or 0) for b in B])
rank = R["rank"]
v = {k: i for i, k in enumerate(R["variants"])}
r25, r1 = rank[:, :, v["r25"]], rank[:, :, v["r1"]]
dens = R["count_near"][:, 1]
deep = lambda r, k: (r < k).any(1)
locatable = tier != "none"


def row(label, m):
    if m.sum() < 20:
        return
    w = ~(km < 25) & m
    print(f"{label:28s} {m.sum():4d} | pipe {100 * np.mean(km[m] < 25):5.1f} ({100 * np.mean(km[m] < 1):4.1f}) top1 {100 * np.mean(top1[m] < 25):5.1f} | "
          f"area {100 * area_ok[m].mean():5.1f} pool {100 * pool_ok[m].mean():5.1f} | 256x3 {100 * deep(r25[m], 256).mean():5.1f} 1000x3 "
          f"{100 * deep(r25[m], 1000).mean():5.1f} any {100 * np.mean(dens[m] > 0):5.1f} | text {100 * text_ok[m].mean():5.1f} | "
          f"wrong: pts of set {100 * w.sum() / n:4.1f}; of which text-right {100 * (w & text_ok).sum() / n:4.1f}, deep-right "
          f"{100 * (w & ~pool_ok & deep(r25, 1000)).sum() / n:4.1f}, sparse<100 {100 * (w & (dens < 100)).sum() / n:4.1f}")


print(f"commons26 dev ({n}); % < 25 km (1 km): pipeline, retrained top-1 | area / pool oracle | any right photo in the top 256 / 1000 per search, "
      f"any within 25 km in the index | some text-search cluster < 25 km | wrong photos as pts of the whole set")
row("all", np.ones(n, bool))
row("geolocatable (tier != none)", locatable)
for t in ("landmark", "city", "region", "none"):
    row(f"tier {t}", tier == t)
row("geolocatable, place text", locatable & ptext)
row("geolocatable, no place text", locatable & ~ptext)
for lo, hi in ((0, 100), (100, 1000), (1000, 10 ** 9)):
    row(f"geolocatable, dens {lo}-{hi}", locatable & (dens >= lo) & (dens < hi))
for c in sorted(set(cont)):
    row(f"geolocatable, {c}", locatable & (cont == c))
L = locatable
print(f"\ngeolocatable photos ({L.sum()}): where the misses are (% of them)")
w = L & (km >= 25)
print(f"  wrong {100 * w.sum() / L.sum():.1f}: right area offered {100 * (w & area_ok).sum() / L.sum():.1f}, right in pool but no area "
      f"{100 * (w & ~area_ok & pool_ok).sum() / L.sum():.1f}, retrieval miss {100 * (w & ~pool_ok).sum() / L.sum():.1f}")
miss = w & ~pool_ok
print(f"  retrieval misses: right photo deeper (top 1000 per search) {100 * (miss & deep(r25, 1000)).sum() / L.sum():.1f}, none in top 1000 "
      f"{100 * (miss & ~deep(r25, 1000)).sum() / L.sum():.1f} (no index photo within 25 km {100 * (miss & (dens == 0)).sum() / L.sum():.1f}); "
      f"text-right among misses {100 * (miss & text_ok).sum() / L.sum():.1f}")
print(f"  wrong and text-right {100 * (w & text_ok).sum() / L.sum():.1f}; wrong, no text-right, no right photo in the top 1000 "
      f"{100 * (w & ~text_ok & ~deep(r25, 1000) & ~area_ok).sum() / L.sum():.1f}")
q = np.quantile(mp16cos[L], [0.25, 0.5, 0.75])
for lo, hi, lab in ((-1, q[0], "q1"), (q[0], q[1], "q2"), (q[1], q[2], "q3"), (q[2], 9, "q4")):
    row(f"geolocatable, mp16_cos {lab}", L & (mp16cos >= lo) & (mp16cos < hi))
print("\ngeolocatable and wrong: which independent source holds the right place (% of geolocatable photos)")
dr = deep(r25, 1000) | pool_ok
for t_ok, t_lab in ((text_ok, "text right"), (~text_ok, "text wrong")):
    for d_ok, d_lab in ((dr, "visual top-1000 right"), (~dr, "visual top-1000 wrong")):
        m = w & t_ok & d_ok
        print(f"  {t_lab:11s} & {d_lab:22s} {100 * m.sum() / L.sum():5.1f}  (tier landmark / city / region "
              f"{(m & (tier == 'landmark')).sum()} / {(m & (tier == 'city')).sum()} / {(m & (tier == 'region')).sum()})")
