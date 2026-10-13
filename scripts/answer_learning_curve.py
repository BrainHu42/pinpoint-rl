"""Learning curve of the merged pipeline's answer head (hypothesis_agent, retrained reranker setft_a, gated zoom rule): train on 1/8 .. all of
25k train photos, evaluate on a fixed 10k held-out train slice and on dev3k (all areas zoomed and no zoom). Decides whether more labels
(cross-fitting, lesson 75) can help: a curve still rising at 25k says yes.

  PYTHONPATH=src .venv/bin/python scripts/answer_learning_curve.py
"""

import numpy as np

from geo_search_env.experiment.hypothesis_agent import evaluate, fit_answer, load

tr = load("train", "setft_a")
dv = load("dev3k", "setft_a", stats=tr["stats"])
order = np.random.default_rng(0).permutation(tr["n"])
held, pool = order[:10_000], order[10_000:]
print(f"answer head learning curve: held-out {len(held)} train photos, dev3k {dv['n']}; pool {len(pool)}", flush=True)
for n in (len(pool) // 8, len(pool) // 4, len(pool) // 2, len(pool)):
    rows = []
    for seed in range(3):
        fit = np.random.default_rng(100 + seed).choice(pool, n, replace=False)
        net = fit_answer(tr, fit, seed=seed, epochs=min(100, round(25 * len(pool) / n)))
        r_h = evaluate(net, tr, held, [0])
        r_d = evaluate(net, dv, np.arange(dv["n"]), [0])
        rows.append([100 * np.mean(r[k] < t) for r in (r_h, r_d) for k in (("all", "all", "agree"), (0, "top", "agree")) for t in (25, 1)])
    m, s = np.mean(rows, 0), np.std(rows, 0)
    print(f"  n {n:6d}: held-out all-zoomed {m[0]:.1f}±{s[0]:.1f} / {m[1]:.1f} (25 / 1 km), no zoom {m[2]:.1f}±{s[2]:.1f}; "
          f"dev3k all-zoomed {m[4]:.1f}±{s[4]:.1f} / {m[5]:.1f}, no zoom {m[6]:.1f}±{s[6]:.1f}", flush=True)
base_h, base_d = tr["base_km"][held], dv["base_km"]
print(f"  released Pinpoint: held-out {100 * np.mean(base_h < 25):.1f}, dev3k {100 * np.mean(base_d < 25):.1f}; retrained top-1: held-out "
      f"{100 * np.mean(tr['rt_base_km'][held] < 25):.1f}, dev3k {100 * np.mean(dv['rt_base_km'] < 25):.1f}")
