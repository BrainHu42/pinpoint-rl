"""Map / name checks as area evidence: the merged pipeline's answer head trained with and without extra column groups (map_checks:
density = OSM objects near each area's restricted top-1, summary = share of the features the 4B saw that are there, checks = per feature
word; name_checks: names = the photo's read / recalled names found near the area). Same held-out 10k train photos and dev3k as
scripts/answer_xfit.py; clues read; gated zoom rule; all areas zoomed and no zoom. All variants x seeds train at once as one batched model
(hypothesis_agent.fit_answers; one data load).

  PYTHONPATH=src .venv/bin/python scripts/answer_mapchecks.py [--variants "" density summary ...] [--seeds 3] [--epochs 25]
"""

import argparse
from pathlib import Path
import time

import numpy as np

from geo_search_env.experiment.hypothesis_agent import evaluate, fit_answers, load

VARIANTS = ("", "density", "summary", "density,summary", "checks", "density,summary,checks")

parser = argparse.ArgumentParser()
parser.add_argument("--seeds", type=int, default=3)
parser.add_argument("--epochs", type=int, default=25)
parser.add_argument("--variants", nargs="+", default=list(VARIANTS), help='comma-joined groups per variant; "" = none (the reference)')
args = parser.parse_args()
VARIANTS = tuple(args.variants)
GROUPS = tuple(sorted({g for v in VARIANTS for g in v.split(",") if g and g != "noclue"}))

n_base = load("dev3k", "setft_a")["pre"].shape[-1]
tr = load("train", "setft_a", extra=GROUPS)
dv = load("dev3k", "setft_a", stats=tr["stats"], extra=GROUPS)
groups = np.concatenate([g[np.isin(g, GROUPS)] for kind in ("mapcheck", "namecheck")  # load's column order
                         if (path := Path(f"artifacts/agent/{kind}_train_setft_a.npz")).exists() for g in [np.load(path)["groups"]]])
assert n_base + len(groups) == tr["pre"].shape[-1]
order = np.random.default_rng(0).permutation(tr["n"])
held, pool = order[:10_000], order[10_000:]
jobs = [(v, s) for v in VARIANTS for s in range(args.seeds)]
CLUE_COLS = slice(12, 20)  # hypothesis_agent.CLUE_COLS: the text-search clue evidence on each area; variant group 'noclue' masks it


def base_mask(v):
    m = np.ones(n_base)
    if "noclue" in v.split(","):
        m[CLUE_COLS] = 0
    return m


masks = [np.concatenate([base_mask(v), np.isin(groups, [g for g in v.split(",") if g]).astype(float)]) for v, _ in jobs]
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
