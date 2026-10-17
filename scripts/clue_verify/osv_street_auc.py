"""Can anything tell a right OSV area from a wrong one? AUC (right = restricted top-1 < 25 km) of single area features on train, OSV areas
(source 2) vs Pinpoint's areas (source 0), and the base rate.

Usage: PYTHONPATH=src .venv/bin/python scripts/clue_verify/osv_street_auc.py
"""
import numpy as np


def auc(x, y):
    x = np.nan_to_num(np.where(np.isfinite(x), x, -1e9), nan=-1e9)
    r = np.argsort(np.argsort(x)) + 1
    n1 = y.sum()
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1))


for tag in ("train", "dev3k", "commons26dev"):
    f = np.load(f"artifacts/agent/hypothesis_features_{tag}_setft_own_e4_moreosv_set.npz")
    ml = np.load(f"artifacts/agent/megaloc_{tag}_setft_own_moreosv.npz")["x"]
    feats = {"support": f["support"], "restricted top-1 score": f["top1"], "score mass (T 0.2)": np.log(np.maximum(f["mass_0.2"], 1e-12)),
             "photographers": f["photographers"], "gallery density": f["density"], "set-model mass": f["set_mass"],
             "MegaLoc best": ml[:, 0], "MegaLoc top-3": ml[:, 1]}
    print(f"== {tag}")
    for src, name in ((0, "Pinpoint areas"), (2, "OSV areas")):
        m = f["source"] == src
        y = f["km"][m] < 25
        print(f"  {name:14s} n={m.sum():6d}, right {y.mean():5.1%}: " + ", ".join(f"{k} {auc(v[m], y):.2f}" for k, v in feats.items()))
