# Uncovered photos (no Pinpoint area restricted top-1 and no proposed clue area < 25 km) that have a right text-search result: why isn't it proposed?
import collections
import json

import numpy as np

A = "artifacts/agent/"


def hav(a, b, c, d):
    a, b, c, d = map(np.radians, (a, b, c, d))
    h = np.sin((c - a) / 2) ** 2 + np.cos(a) * np.cos(c) * np.sin((d - b) / 2) ** 2
    return 2 * 6371 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


ORDER = ["proposable (judge >= 0.2)", "within 25 km of a Pinpoint area (its restricted top-1 wrong)", "judge < 0.2", "never judged", "match share < 0.3"]
for tag in ("commons26dev_setft_own", "dev3k_setft_own"):
    photos = json.load(open(f"artifacts/photo_sets/{tag}/photos.json"))
    pos = {e["image_id"]: i for i, e in enumerate(photos)}
    truth = np.asarray([e["truth"] for e in photos])
    h = np.load(A + f"hypotheses_{tag}.npz")
    ch = np.load(A + f"hypotheses_{tag}_cluehyps.npz")
    cov = np.zeros(len(photos), bool)
    cen = collections.defaultdict(list)
    for j in np.flatnonzero(h["source"] == 0):
        p = int(h["photo"][j])
        cen[p].append(h["center"][j])
        sc = np.where(h["rows"][j] >= 0, h["scores"][j], -np.inf)
        if np.isfinite(sc).any():
            ll = h["latlon"][j][int(np.argmax(sc))]
            cov[p] |= hav(*truth[p], *ll) < 25
    for j in range(len(ch["photo"])):
        p = int(ch["photo"][j])
        cov[p] |= hav(*truth[p], *ch["center"][j]) < 25
    J = {(r["image_id"], r["query"]): r["p_yes"] for r in json.load(open(A + f"clue_judge_{tag}_4b_4b.json"))}
    by = collections.defaultdict(list)
    for r in json.load(open(A + f"region_evidence_{tag}_4b.json")):
        if "region" in r and r["km"] < 25:
            by[pos[r["image_id"]]].append(r)
    reasons = collections.Counter()
    for p, rs in by.items():
        if cov[p]:
            continue
        best = None
        for r in rs:
            cs = np.asarray(cen[p]).reshape(-1, 2)
            d = hav(r["lat"], r["lon"], cs[:, 0], cs[:, 1]).min() if len(cs) else 1e9
            pj = J.get((r["image_id"], r["query"]))
            why = (ORDER[1] if d < 25 else ORDER[4] if r["share"] < 0.3 else ORDER[3] if pj is None else ORDER[2] if pj < 0.2 else ORDER[0])
            if best is None or ORDER.index(why) < ORDER.index(best):
                best = why
        reasons[best] += 1
    n = len(photos)
    tot = sum(reasons.values())
    print(f"{tag}: uncovered photos with a right search result {tot} ({100 * tot / n:.1f}%); best reason each:")
    for k, v in reasons.most_common():
        print(f"   {k:62s} {v:4d} ({100 * v / n:.1f}%)")
