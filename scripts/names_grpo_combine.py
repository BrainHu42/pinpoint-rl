"""Head margin + judge logit, logistic regression fit on held, switch threshold chosen on held, applied to dev3k (names_grpo results).

Usage: .venv/bin/python scripts/names_grpo_combine.py
"""
import json

import numpy as np

pairs = {(r["split"], r["photo"]): r for r in map(json.loads, open("artifacts/agent/names_grpo_pairs.jsonl"))}
print("pair keys:", sorted(next(iter(pairs.values())).keys()))


def data(name, split):
    X, y = [], []
    for r in map(json.loads, open(f"artifacts/agent/names_grpo_{name}.jsonl")):
        if r["split"] != split or "p_top" not in r:
            continue
        pr = pairs[(split, r["photo"])]
        q = np.clip(np.mean(r["p_top"]), 1e-4, 1 - 1e-4)
        X.append([np.log(pr["p"][0] / pr["p"][1]), np.log(q / (1 - q))])
        y.append(pr["km"][1] < 25)  # head wrong: switching fixes
    return np.array(X), np.array(y)


def fit(X, y, steps=5000, lr=0.1):
    Xb = np.c_[X, np.ones(len(X))]
    w = np.zeros(Xb.shape[1])
    for _ in range(steps):
        p = 1 / (1 + np.exp(-Xb @ w))
        w -= lr * Xb.T @ (p - y) / len(y)
    return w


def auc(s, y):
    r = np.argsort(np.argsort(s)) + 1
    return (r[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum())


for name in ("names-exact-a", "base2"):
    Xh, yh = data(name, "held")
    Xd, yd = data(name, "dev")
    for cols, label in (([0], "head only"), ([0, 1], "head + judge")):
        w = fit(Xh[:, cols], yh)
        sh = np.c_[Xh[:, cols], np.ones(len(Xh))] @ w
        sd = np.c_[Xd[:, cols], np.ones(len(Xd))] @ w
        nets = [(t, (yh & (sh > t)).sum() - (~yh & (sh > t)).sum()) for t in np.quantile(sh, np.linspace(0.5, 0.999, 200))]
        t, nh = max(nets, key=lambda x: x[1])
        fx, br = (yd & (sd > t)).sum(), (~yd & (sd > t)).sum()
        print(f"{name:14s} {label:13s} w={np.round(w, 2)}  AUC held {auc(sh, yh):.3f} dev {auc(sd, yd):.3f}  held best net {nh:+d}; "
              f"dev switches {fx + br}, fixes {fx}, breaks {br}, net {100 * (fx - br) / 3000:+.2f} pts")
