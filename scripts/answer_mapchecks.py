"""Map checks as area evidence (map_checks.features): the merged pipeline's answer head trained with and without the map-check column groups
(density: OSM objects near each area's restricted top-1; summary: share of the features the 4B saw that are there, and how many it saw;
checks: per feature word, there or not). Same held-out 10k train photos and dev3k as scripts/answer_xfit.py; clues read; gated zoom rule;
all areas zoomed and no zoom. All variants x seeds train at once as one batched model (hypothesis_agent.fit_answers; one data load).

  PYTHONPATH=src .venv/bin/python scripts/answer_mapchecks.py [--seeds 3] [--epochs 25]
"""

import argparse
import time

import numpy as np

from geo_search_env.experiment.hypothesis_agent import evaluate, fit_answers, load

VARIANTS = ("", "density", "summary", "density,summary", "checks", "density,summary,checks")
GROUPS = ("density", "summary", "checks")

parser = argparse.ArgumentParser()
parser.add_argument("--seeds", type=int, default=3)
parser.add_argument("--epochs", type=int, default=25)
args = parser.parse_args()

n_base = load("dev3k", "setft_a")["pre"].shape[-1]
tr = load("train", "setft_a", extra=GROUPS)
dv = load("dev3k", "setft_a", stats=tr["stats"], extra=GROUPS)
groups = np.load("artifacts/agent/mapcheck_train_setft_a.npz")["groups"]
assert n_base + len(groups) == tr["pre"].shape[-1]
order = np.random.default_rng(0).permutation(tr["n"])
held, pool = order[:10_000], order[10_000:]
jobs = [(v, s) for v in VARIANTS for s in range(args.seeds)]
masks = [np.concatenate([np.ones(n_base), np.isin(groups, [g for g in v.split(",") if g]).astype(float)]) for v, _ in jobs]
t = time.time()
nets = fit_answers(tr, pool, [s for _, s in jobs], masks, epochs=args.epochs)
print(f"{len(nets)} answer heads trained together in {time.time() - t:.0f} s", flush=True)
res = {}
for (v, s), net in zip(jobs, nets):
    r_h, r_d = evaluate(net, tr, held, [0]), evaluate(net, dv, np.arange(dv["n"]), [0])
    res.setdefault(v, []).append({f"{split}_{name}": r[k] for split, r in (("held", r_h), ("dev", r_d))
                                  for name, k in (("zoom", ("all", "all", "agree")), ("nozoom", (0, "top", "agree")))})
hit = {v: np.mean([np.concatenate([r["held_zoom"] < 25, r["dev_zoom"] < 25]) for r in rs], 0) for v, rs in res.items()}
nh = len(held)
se = lambda x: 100 * x.std() / np.sqrt(len(x))
for v, rs in res.items():
    acc = lambda key, th: [100 * np.mean(r[key] < th) for r in rs]
    diff = hit[v] - hit[""]
    print(f"  {v.replace(',', '+') or 'no map checks':24s}: held-out all-zoomed {np.mean(acc('held_zoom', 25)):.1f}±{np.std(acc('held_zoom', 25)):.1f}"
          f" / {np.mean(acc('held_zoom', 1)):.1f} (25 / 1 km), no zoom {np.mean(acc('held_nozoom', 25)):.1f}; dev3k all-zoomed "
          f"{np.mean(acc('dev_zoom', 25)):.1f}±{np.std(acc('dev_zoom', 25)):.1f} / {np.mean(acc('dev_zoom', 1)):.1f}, no zoom {np.mean(acc('dev_nozoom', 25)):.1f}; "
          f"paired vs none at 25 km: held-out {100 * diff[:nh].mean():+.2f}±{se(diff[:nh]):.2f}, dev3k {100 * diff[nh:].mean():+.2f}±{se(diff[nh:]):.2f}", flush=True)
