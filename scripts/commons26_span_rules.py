"""Span search on commons26 dev: recall of phrases' top clusters vs spans' clusters, and agreement-based picks (vectorised)."""
import collections
import csv
import json

import numpy as np

bench = {r["IMG_ID"]: r for r in csv.DictReader(open("data/pinpoint/commons26/release/benchmark.csv", encoding="utf-8"))}
dev = [i for i, r in bench.items() if r["split"] == "dev"]
truth = {i: np.radians([float(bench[i]["LAT"]), float(bench[i]["LON"])]) for i in dev}
phr = collections.defaultdict(list)
for r in json.load(open("artifacts/agent/clusters_commons26_4b.json")):
    if r["image_id"] in truth:
        phr[r["image_id"]].append(r)
spn = collections.defaultdict(list)
for r in json.load(open("artifacts/agent/spans_commons26_4b_dev.json")):
    spn[r["image_id"]].append(r)


def hav(a, b):
    """a [n, 2], b [m, 2] radians -> km [n, m]"""
    d = np.sin((b[None, :, 0] - a[:, None, 0]) / 2) ** 2 + np.cos(a[:, None, 0]) * np.cos(b[None, :, 0]) * np.sin((b[None, :, 1] - a[:, None, 1]) / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(d, 0, 1)))


def ok(i, cs):
    if not cs:
        return False
    return bool((hav(truth[i][None], np.radians([[c["lat"], c["lon"]] for c in cs]))[0] < 25).any())


groups = {"all dev": dev, "place_text=1": [i for i in dev if bench[i]["place_text"] == "1"], "landmark": [i for i in dev if bench[i]["tier"] == "landmark"]}
for g, ids in groups.items():
    rec = lambda f: np.mean([f(i) for i in ids])
    print(f"\n{g} ({len(ids)})", flush=True)
    print(f"  phrase top cluster < 25 km   {rec(lambda i: ok(i, [r['clusters'][0] for r in phr[i] if r['clusters']])):.1%}")
    print(f"  phrase any cluster           {rec(lambda i: ok(i, [c for r in phr[i] for c in r['clusters']])):.1%}")
    print(f"  span top cluster             {rec(lambda i: ok(i, [r['clusters'][0] for r in spn[i] if r['clusters']])):.1%}")
    print(f"  span any cluster             {rec(lambda i: ok(i, [c for r in spn[i] for c in r['clusters']])):.1%}")
    print(f"  span top, <= 50 matches      {rec(lambda i: ok(i, [r['clusters'][0] for r in spn[i] if r['clusters'] and r['matches'] <= 50])):.1%}", flush=True)


def agree(i):
    """Candidate points: top-3 clusters of every query (spans + phrases). Score = sum over queries of the best share within 25 km, weighted by
    specificity 1 / log2(2 + matches) (phrases count as 1 match). Returns (best point radians, score, supporting queries)."""
    qs = [(r["clusters"][:8], r.get("matches", 1)) for r in spn[i]] + [(r["clusters"][:8], 1) for r in phr[i]]
    qs = [(cs, m) for cs, m in qs if cs]
    if not qs:
        return None, 0.0, 0
    allc = np.radians([[c["lat"], c["lon"]] for cs, _ in qs for c in cs])
    share = np.asarray([c["share"] for cs, _ in qs for c in cs])
    qid = np.asarray([k for k, (cs, _) in enumerate(qs) for _ in cs])
    rank = np.asarray([j for cs, _ in qs for j in range(len(cs))])
    wq = 1 / np.log2(2 + np.asarray([m for _, m in qs], float))
    cand = rank < 3
    d = hav(allc[cand], allc) < 25  # [n_cand, n_all]
    best = np.zeros((cand.sum(), len(qs)))
    np.maximum.at(best.T, qid, (d * share[None]).T) if False else None
    for k in range(len(qs)):
        m = qid == k
        best[:, k] = (d[:, m] * share[m][None]).max(1)
    score = (best * wq[None]).sum(1)
    j = int(np.argmax(score))
    return allc[cand][j], float(score[j]), int((best[j] > 0).sum())


for g, ids in groups.items():
    res = {i: agree(i) for i in ids}
    hit = {i: res[i][0] is not None and bool(hav(truth[i][None], res[i][0][None])[0, 0] < 25) for i in ids}
    print(f"{g}: agreement pick < 25 km {np.mean(list(hit.values())):.1%}", flush=True)
    for t in (2, 3, 5):
        sel = [i for i in ids if res[i][2] >= t]
        print(f"   supported by >= {t} queries: {len(sel) / len(ids):.1%} of photos, right {np.mean([hit[i] for i in sel]) if sel else 0:.1%}")
