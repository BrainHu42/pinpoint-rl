"""Diagnose lesson 119 on commons26 dev: recall vs chance, cap loss, where the right candidate ranks on pipeline-wrong photos, why."""
import collections
import json
import math

import numpy as np

from geo_search_env.experiment import candidate_ranker as cr

ctx = cr._context("commons26", "dev")
recs = [json.loads(l) for l in open("artifacts/agent/cands_commons26_dev.jsonl")]
ids = [r["image_id"] for r in recs]
n = len(ids)
truth = np.asarray([ctx["photos"][ctx["pidx"][i]]["truth"] for i in ids], float)
fin = np.asarray([ctx["final"][ctx["pidx"][i]] < 25 for i in ids])
tier = np.asarray([ctx["bench"][i]["tier"] for i in ids])
ptext = np.asarray([ctx["bench"][i]["place_text"] == "1" for i in ids])
rng = np.random.default_rng(0)
perm = rng.permutation(n)
while (perm == np.arange(n)).any():
    perm = rng.permutation(n)
F = {f: k for k, f in enumerate(cr.FEATURES)}

cands = {}
for cap in (64, 10**6):
    cr.MAX_CANDIDATES = cap
    cands[cap] = [cr._candidates(r, ctx) for r in recs]

print("A. recall: some candidate < 25 km of the truth vs of a random other photo's truth (chance)")
for cap, cs in cands.items():
    right = np.asarray([len(C) > 0 and (cr._hav(t[0], t[1], C[:, 0], C[:, 1]) < 25).any() for (C, _), t in zip(cs, truth)])
    chance = np.asarray([len(C) > 0 and (cr._hav(t[0], t[1], C[:, 0], C[:, 1]) < 25).any() for (C, _), t in zip(cs, truth[perm])])
    ncand = np.median([len(C) for C, _ in cs])
    print(f"  cap {cap if cap < 1e6 else 'none'} (median {ncand:.0f} cands): right {right.mean():.1%}, chance {chance.mean():.1%}; right & pipeline wrong {np.mean(right & ~fin):.1%}")
    for g in ("landmark", "city", "region", "none"):
        s = tier == g
        print(f"     {g:8s} right {right[s].mean():5.1%} chance {chance[s].mean():5.1%}")
    if cap == 64:
        right64 = right

# ranker (linear, capped), out of fold
cs = cands[64]
Ys = [cr._hav(t[0], t[1], C[:, 0], C[:, 1]) < 25 if len(C) else np.zeros(0, bool) for (C, _), t in zip(cs, truth)]
X, M, Y = cr._pad([x for _, x in cs], Ys)
fold = cr._folds([ctx["bench"][i]["group"] or i for i in ids])
P = np.zeros(M.shape, np.float32)
for k in range(5):
    tr, te = fold != k, fold == k
    P[te] = cr._fit_predict(X[tr], M[tr], Y[tr], X[te], M[te], 0)
P = np.where(M, P, -1)
order = np.argsort(-P, 1)
top = order[:, 0]
top_right = Y[np.arange(n), top]
rank_right = np.asarray([int(np.flatnonzero(Y[i][order[i]])[0]) + 1 if Y[i].any() else 0 for i in range(n)])

fx = right64 & ~fin  # fixable under the cap
print(f"\nB. fixable photos (right candidate, pipeline wrong): {fx.sum()} ({fx.mean():.1%})")
print(f"   ranker rank of the best right candidate: 1: {np.mean(rank_right[fx] == 1):.0%}, 2-3: {np.mean((rank_right[fx] >= 2) & (rank_right[fx] <= 3)):.0%}, "
      f"4-10: {np.mean((rank_right[fx] >= 4) & (rank_right[fx] <= 10)):.0%}, >10: {np.mean(rank_right[fx] > 10):.0%}")
print(f"   same on photos the pipeline gets right: rank 1 {np.mean(rank_right[right64 & fin] == 1):.0%}")
print(f"   place_text {ptext[fx].mean():.0%} of fixable (dev {ptext.mean():.0%}); tiers {dict(collections.Counter(tier[fx]))}")

def feat_of(i, j):
    return X[i, j]

cols = ["queries", "phrase queries", "sources", "specificity", "sharpness", "top of a query", "exact name", "near an area", "near area 1", "Pinpoint mass",
        "country = 4B", "population", "support rank"]
best_right = [int(order[i][rank_right[i] - 1]) for i in range(n)]
print("\nC. features, fixable photos: best right candidate vs ranker's top (wrong) candidate | right candidates on pipeline-right photos")
w = fx & ~top_right
ok = right64 & fin
for c in cols:
    k = F[c]
    a = np.mean([X[i, best_right[i], k] for i in np.flatnonzero(w)])
    b = np.mean([X[i, top[i], k] for i in np.flatnonzero(w)])
    d = np.mean([X[i, best_right[i], k] for i in np.flatnonzero(ok)])
    print(f"   {c:15s} right {a:7.2f}   top wrong {b:7.2f}   | right, pipeline-right photos {d:7.2f}")
print(f"   P(best right) median {np.median([P[i, best_right[i]] for i in np.flatnonzero(w)]):.3f}; P(top) median {np.median([P[i, top[i]] for i in np.flatnonzero(w)]):.3f}")

# how specific is the evidence for the right candidate on fixable photos: min matches of supporting (query, source)
def support(i, C):
    r = recs[i]
    out = []
    for q in r["queries"]:
        for s, v in q["src"].items():
            for c in v["clusters"]:
                if cr._hav(C[0], C[1], c[0], c[1]) < 25:
                    out.append((q["q"], q["from"], s, v["n"], c[4], round(c[3], 2)))
    return out

spec = []
for i in np.flatnonzero(fx):
    C = cs[i][0][best_right[i]]
    sup = support(i, C)
    spec.append(min(x[3] for x in sup) if sup else 1e9)
spec = np.asarray(spec)
print(f"\nD. fixable photos: rarest supporting query's matches for the right candidate: <= 5: {np.mean(spec <= 5):.0%}, 6-100: {np.mean((spec > 5) & (spec <= 100)):.0%}, "
      f"101-2000: {np.mean((spec > 100) & (spec <= 2000)):.0%}, > 2000: {np.mean(spec > 2000):.0%}")
onlyspan = []
for i in np.flatnonzero(fx):
    sup = support(i, cs[i][0][best_right[i]])
    onlyspan.append(all(x[1] == "span" for x in sup))
print(f"   right candidate supported only by spans (not whole phrases): {np.mean(onlyspan):.0%}")

E_SKIP = True
print("\nE. examples (fixable, ranker top wrong):")
for i in rng.choice(np.flatnonzero(w), 12, replace=False):
    C = cs[i][0]
    r_sup = sorted(support(i, C[best_right[i]]), key=lambda x: x[3])[:4]
    t_sup = sorted(support(i, C[top[i]]), key=lambda x: x[3])[:4]
    print(f"- {ids[i]} tier {tier[i]} place_text {int(ptext[i])} rank of right {rank_right[i]}; P right {P[i, best_right[i]]:.2f} top {P[i, top[i]]:.2f}")
    print(f"    right: {[(q, s, nm) for q, _, s, nm, _, _ in r_sup]}")
    print(f"    top:   {[(q, s, nm) for q, _, s, nm, _, _ in t_sup]}")

print("\nF. fixable photos with a wrong ranker top: distance of that top candidate to the truth")
dt = np.asarray([float(cr._hav(truth[i][0], truth[i][1], cs[i][0][top[i]][0], cs[i][0][top[i]][1])) for i in np.flatnonzero(w)])
print("   " + ", ".join(f"{a}-{b} km {np.mean((dt >= a) & (dt < b)):.0%}" for a, b in ((25, 50), (50, 100), (100, 300), (300, 1000), (1000, 1e5))))
dtp = np.asarray([float(cr._hav(truth[i][0], truth[i][1], cs[i][0][top[i]][0], cs[i][0][top[i]][1])) for i in np.flatnonzero(~top_right & M.any(1) & fin)])
print("   (pipeline-right photos, wrong ranker top) " + ", ".join(f"{a}-{b} km {np.mean((dtp >= a) & (dtp < b)):.0%}" for a, b in ((25, 50), (50, 100), (100, 300), (300, 1000), (1000, 1e5))))
src1 = []
for i in np.flatnonzero(w):
    sup = support(i, cs[i][0][top[i]])
    r = min(sup, key=lambda x: x[3]) if sup else None
    src1.append(r is not None and r[3] <= 2 and r[2] == "Flickr")
print(f"   top wrong candidate whose rarest support is a Flickr query with <= 2 matching photos: {np.mean(src1):.0%}")
