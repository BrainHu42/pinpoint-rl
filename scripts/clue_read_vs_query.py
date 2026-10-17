# Reading or querying? commons26 dev photos split by the 27B's location-blind flags (benchmark.csv: tier, place_text = visible text names the
# place), on all photos and on those no Pinpoint area / clue area gets right ("uncovered"): did the 4B read text, did any phrase search to a
# place, did any phrase's top result land < 25 km (read text, place names, the 4B's own queries).
import collections
import csv
import json

import numpy as np

A = "artifacts/agent/"
TAG = "commons26dev_setft_own"


def hav(a, b, c, d):
    a, b, c, d = map(np.radians, (a, b, c, d))
    h = np.sin((c - a) / 2) ** 2 + np.cos(a) * np.cos(c) * np.sin((d - b) / 2) ** 2
    return 2 * 6371 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


meta = {r["IMG_ID"]: r for r in csv.DictReader(open("data/pinpoint/commons26/release/benchmark.csv"))}
photos = json.load(open(f"artifacts/photo_sets/{TAG}/photos.json"))
truth = np.asarray([e["truth"] for e in photos])
h = np.load(A + f"hypotheses_{TAG}.npz")
ch = np.load(A + f"hypotheses_{TAG}_cluehyps.npz")
cov = np.zeros(len(photos), bool)
for j in np.flatnonzero(h["source"] == 0):
    p = int(h["photo"][j])
    sc = np.where(h["rows"][j] >= 0, h["scores"][j], -np.inf)
    if np.isfinite(sc).any():
        cov[p] |= hav(*truth[p], *h["latlon"][j][int(np.argmax(sc))]) < 25
clue_ok = np.zeros(len(photos), bool)
for j in range(len(ch["photo"])):
    clue_ok[int(ch["photo"][j])] |= hav(*truth[int(ch["photo"][j])], *ch["center"][j]) < 25
by = collections.defaultdict(list)
for r in json.load(open(A + f"region_evidence_{TAG}_4b.json")):
    by[r["image_id"]].append(r)
obs = json.load(open(A + f"observe_{TAG}_4b.json"))

READ = ("text 4b", "place name 4b")


def stats(idx):
    ids = [photos[i]["image_id"] for i in idx]
    n = max(len(ids), 1)
    read = sum(any(r["source"] in READ for r in by[i]) for i in ids)
    found = sum(any(r["source"] in READ and "region" in r for r in by[i]) for i in ids)
    right_read = sum(any(r["source"] in READ and "region" in r and r["km"] < 25 for r in by[i]) for i in ids)
    right_q = sum(any(r["source"].startswith("query") and "region" in r and r["km"] < 25 for r in by[i]) for i in ids)
    right_any = sum(any("region" in r and r["km"] < 25 for r in by[i]) for i in ids)
    return (f"n {len(ids):4d} | read text/names {100 * read / n:5.1f}% | a read phrase searches to a place {100 * found / n:5.1f}% | "
            f"a read phrase's top < 25 km {100 * right_read / n:5.1f}% | a 4B query's top < 25 km {100 * right_q / n:5.1f}% | any {100 * right_any / n:5.1f}%")


allp = np.arange(len(photos))
unc = np.flatnonzero(~cov & ~clue_ok)
pt = np.asarray([meta[e["image_id"]]["place_text"] == "1" for e in photos])
tier = np.asarray([meta[e["image_id"]]["tier"] for e in photos])
print(f"{TAG}: {len(photos)} photos; uncovered (no Pinpoint area or clue area right) {len(unc)} ({100 * len(unc) / len(photos):.1f}%)")
for name, sel in (("all photos", allp), ("uncovered", unc)):
    print(f"\n== {name}")
    for lab, m in (("place_text", pt), ("no place_text", ~pt)):
        idx = np.intersect1d(sel, np.flatnonzero(m))
        print(f"  {lab:16s} {100 * len(idx) / len(sel):5.1f}% of them: {stats(idx)}")
    for t in ("landmark", "city", "region", "none"):
        idx = np.intersect1d(sel, np.flatnonzero(tier == t))
        print(f"  tier {t:11s} {100 * len(idx) / len(sel):5.1f}% of them: {stats(idx)}")
# uncovered place_text photos: what did the 4B read? (sample for inspection)
idx = np.intersect1d(unc, np.flatnonzero(pt))
rng = np.random.default_rng(0)
print("\nsample of uncovered place_text photos: 4B read phrases -> top result km")
for i in rng.choice(idx, size=min(12, len(idx)), replace=False):
    iid = photos[i]["image_id"]
    rs = [r for r in by[iid] if r["source"] in READ]
    print(f"  {iid} ({meta[iid]['country']}, tier {meta[iid]['tier']}): " +
          "; ".join(f"'{r['query'][:40]}' -> {r['km']:.0f} km" if "region" in r else f"'{r['query'][:40]}' -> none" for r in rs[:5]))
