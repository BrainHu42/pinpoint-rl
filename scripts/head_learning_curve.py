# Is the answer head data-limited? Train it on fractions of the train split and score dev3k (all areas zoomed, clues read, MegaLoc on).
# Usage: PYTHONPATH=src .venv/bin/python scripts/head_learning_curve.py
import sys

import numpy as np

sys.path.insert(0, "src")
from geo_search_env.experiment.hypothesis_agent import evaluate, fit_answer, load  # noqa: E402

tr = load("train", "setft_own", extra=("megaloc",))
dv = load("dev3k", "setft_own", stats=tr["stats"], extra=("megaloc",))
n = tr["n"]
print(f"train {n} photos, dev3k {dv['n']}", flush=True)
for frac in (0.125, 0.25, 0.5, 1.0):
    accs = []
    for seed in range(3):
        idx = np.sort(np.random.default_rng(seed).permutation(n)[: int(frac * n)])  # a different subset per seed
        net = fit_answer(tr, idx, seed=seed)
        km = evaluate(net, dv, np.arange(dv["n"]), budgets=(0,))[("all", "all", "agree")]
        accs.append(100 * np.mean(km < 25))
    print(f"train fraction {frac:5.3f} ({int(frac * n):5d} photos): dev3k < 25 km {np.mean(accs):.2f} (seeds {', '.join(f'{a:.1f}' for a in accs)})", flush=True)
