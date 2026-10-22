# Does Mapillary add to the head? On scripts/mapillary_dev3k.py's 1,334 pairs: a logistic model of P(the head's top area is the right one) from
# the head's margin alone vs margin + Mapillary (best and z of both areas, coverage flags), 5-fold CV; switch to the second area when the model
# says P < 0.5. Usage: PYTHONPATH=src .venv/bin/python scripts/mapillary_dev3k_combine.py
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

OUT = Path("artifacts/tool_use/dev3k")


def _fit(X, y, l2=1e-2, steps=50):
    """L2-regularised logistic regression by Newton steps on standardised features (with intercept)."""
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = np.c_[np.ones(len(X)), (X - mu) / sd]
    w = np.zeros(Z.shape[1])
    for _ in range(steps):
        p = 1 / (1 + np.exp(-Z @ w))
        g = Z.T @ (p - y) + l2 * np.r_[0, w[1:]]
        H = (Z * (p * (1 - p))[:, None]).T @ Z + l2 * np.diag(np.r_[0, np.ones(len(w) - 1)])
        w -= np.linalg.solve(H, g)
    return lambda Xn: 1 / (1 + np.exp(-np.c_[np.ones(len(Xn)), (Xn - mu) / sd] @ w))


def _auc(s, y):
    r = rankdata(s)
    return (r[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum())


def main() -> None:
    sc = json.loads((OUT / "scores.json").read_text())
    tg = {t["photo"]: t for t in json.loads((OUT / "targets.json").read_text())}
    place = {(r["photo"], r["role"]): r for r in sc}
    photos = sorted(tg)

    def stat(p, role):
        r = place.get((p, role))
        v = sorted((f["megaloc"] for f in r["files"] if f["megaloc"] is not None), reverse=True) if r else []
        if not v:
            return 0.0, 0.0, 0.0
        return v[0], (v[0] - r["null_mu"]) / max(r["null_sd"], 1e-3), 1.0

    y = np.asarray([tg[p]["head_right"] for p in photos])
    margin = np.asarray([np.log(tg[p]["p_top"] / tg[p]["p_second"]) for p in photos])
    t = np.asarray([stat(p, "top") for p in photos])
    s = np.asarray([stat(p, "second") for p in photos])
    feats = {"head margin": margin[:, None],
             "margin + Mapillary": np.c_[margin, t, s, t[:, 0] - s[:, 0], t[:, 1] - s[:, 1], margin * t[:, 2] * s[:, 2]]}
    fold = np.random.default_rng(0).permutation(len(y)) % 5
    lines = [f"Mapillary + head on {len(y)} dev3k pairs (head right {y.mean():.1%}; switching to the second area when P(head right) < 0.5, 5-fold CV)"]
    for name, X in feats.items():
        P = np.zeros(len(y))
        for k in range(5):
            P[fold == k] = _fit(X[fold != k], y[fold != k].astype(float))(X[fold == k])
        sw = P < 0.5
        fx, br = int((sw & ~y).sum()), int((sw & y).sum())
        lines.append(f"  {name:20s} AUC {_auc(P, y):.3f}; {sw.sum()} switches, {fx} fixes / {br} breaks ({100 * (fx - br) / 3000:+.2f} pts of dev3k)")
        for thr in (0.4, 0.3):
            sw = P < thr
            fx, br = int((sw & ~y).sum()), int((sw & y).sum())
            lines.append(f"    P < {thr}: {sw.sum()} switches, {fx} fixes / {br} breaks ({100 * (fx - br) / 3000:+.2f} pts)")
    text = "\n".join(lines)
    print(text)
    (OUT / "combine.txt").write_text(text + "\n")


if __name__ == "__main__":
    main()
