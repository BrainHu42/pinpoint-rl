"""Does OSV street-level similarity add to text-search weight? Grouped-CV logistic per candidate place, then within-photo top pick and a switch."""
import csv
import json
import os
import sys

sys.path.insert(0, "/home/brianhu/pinpoint-rl/.claude/worktrees/density-probe/src")
os.chdir("/home/brianhu/pinpoint-rl/.claude/worktrees/density-probe")
import numpy as np
from geo_search_env.experiment.clue_verify import _answers, _auc, _haversine, RIGHT_KM

rows = json.load(open("artifacts/clue_verify/commons26_dev_osv.json"))
km, _, _ = _answers("commons26")
meta = {r["IMG_ID"]: r for r in csv.DictReader(open("data/pinpoint/commons26/release/benchmark.csv"))}
base = np.asarray([km[r["photo"]] < RIGHT_KM for r in rows])
groups = np.asarray([meta[r["image_id"]]["group"] for r in rows]); ug = np.unique(groups)
fold = {x: k % 5 for k, x in enumerate(np.random.default_rng(0).permutation(ug))}
f = np.asarray([fold[x] for x in groups])


def feats(p, r, use_osv):
    w = np.log1p(p["weight"]) if p["kind"] == "text" else 0.0
    is_text = float(p["kind"] == "text")
    rank = int(p["kind"].split("_")[1]) if p["kind"].startswith("area") else -1
    x = [is_text, w, float(rank == 0), float(rank == 1), float(rank >= 2), float(p["kind"] == "answer")]
    if use_osv:
        for k in ("ml",):
            x += [max(p[k], 0.0), float(p[k] > -1)]
        x += [np.log1p(p["ml_n"])]
    return x


def fit(X, y, l2=1e-2, it=3000, lr=0.5):
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Z = (X - mu) / sd
    w = np.zeros(Z.shape[1]); b = 0.0
    for _ in range(it):
        pr = 1 / (1 + np.exp(-(Z @ w + b)))
        g = pr - y
        w -= lr * (Z.T @ g / len(y) + l2 * w); b -= lr * g.mean()
    return lambda X2: 1 / (1 + np.exp(-(((X2 - mu) / sd) @ w + b)))


for use_osv in (False, True):
    probs = [None] * len(rows)
    for k in range(5):
        tr = [i for i in range(len(rows)) if f[i] != k]
        X = np.asarray([feats(p, rows[i], use_osv) for i in tr for p in rows[i]["places"]])
        y = np.asarray([p["right"] for i in tr for p in rows[i]["places"]], float)
        model = fit(X, y)
        for i in range(len(rows)):
            if f[i] == k:
                probs[i] = model(np.asarray([feats(p, rows[i], use_osv) for p in rows[i]["places"]]))
    allp = np.concatenate(probs); ally = np.asarray([p["right"] for r in rows for p in r["places"]])
    # within-photo top pick among all candidates (text places, areas, answer)
    top = np.asarray([r["places"][int(np.argmax(pr))]["right"] if len(pr) else False for r, pr in zip(rows, probs)])
    # switch: top candidate replaces the answer when it is > 25 km from it and its prob beats the answer's by margin m (chosen by CV)
    def apply(i, m):
        r, pr = rows[i], probs[i]
        a = next((j for j, p in enumerate(r["places"]) if p["kind"] == "answer"), None)
        if a is None or not len(pr):
            return base[i]
        j = int(np.argmax(pr)); b, ap = r["places"][j], r["places"][a]
        if pr[j] - pr[a] >= m and _haversine(b["lat"], b["lon"], ap["lat"], ap["lon"]) >= RIGHT_KM:
            return b["right"]
        return base[i]
    res = np.zeros(len(rows), bool)
    for k in range(5):
        tr = np.flatnonzero(f != k)
        mbest = max(np.arange(0.0, 0.8, 0.02), key=lambda m: sum(apply(i, m) for i in tr))
        for i in np.flatnonzero(f == k):
            res[i] = apply(i, mbest)
    print(f"{'with OSV' if use_osv else 'no OSV  '}: candidate AUC {_auc(allp[ally], allp[~ally]):.3f}; top candidate right {100 * top.mean():.1f}% "
          f"(answer {100 * base.mean():.1f}%); switch {100 * base.mean():.1f} -> {100 * res.mean():.1f}% "
          f"(fixes {int((res & ~base).sum())}, breaks {int((~res & base).sum())})")
