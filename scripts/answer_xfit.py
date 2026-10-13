"""Cross-fitting (lesson 75): the merged pipeline's answer head trained on train alone vs train + cross-fitted photo sets (each half's trained-on
photos run through the pipeline with the reranker of the other photographer half; scripts/xfit_queue.sh, scripts/xfit_pipeline.sh). Same held-out
10k train photos and dev3k as scripts/answer_learning_curve.py; gated zoom rule; inputs normalised with train statistics.

  PYTHONPATH=src .venv/bin/python scripts/answer_xfit.py [--extra xfit1:xfit_a xfit0:xfit_b] [--seeds 3]
"""

import argparse

import numpy as np
import torch

from geo_search_env.experiment.hypothesis_agent import evaluate, fit_answer, load

FILL = {"km_zoom": np.inf, "km_zoom_set": np.inf, "km_unzoom": np.inf, "rank": 99.0}


def concat(parts: list[dict]) -> dict:
    """Photo-level concatenation of load() outputs (area axis padded to the longest)."""

    h = max(p["mask"].shape[1] for p in parts)
    out = {"n": sum(p["n"] for p in parts), "stats": parts[0]["stats"]}
    for k in ("base_km", "rt_base_km", "benchmark"):
        out[k] = np.concatenate([p[k] for p in parts])
    for k, v in parts[0].items():
        if not torch.is_tensor(v):
            continue
        padded = []
        for p in parts:
            x = p[k]
            if x.shape[1] < h:
                pad = torch.full((x.shape[0], h - x.shape[1]) + tuple(x.shape[2:]), FILL.get(k, 0), dtype=x.dtype, device=x.device)
                x = torch.cat([x, pad], 1)
            padded.append(x)
        out[k] = torch.cat(padded)
    return out


parser = argparse.ArgumentParser()
parser.add_argument("--extra", nargs="*", default=["xfit1:xfit_a", "xfit0:xfit_b"])
parser.add_argument("--seeds", type=int, default=3)
args = parser.parse_args()

tr = load("train", "setft_a")
dv = load("dev3k", "setft_a", stats=tr["stats"])
extras = [load(t.split(":")[0], t.split(":")[1], stats=tr["stats"]) for t in args.extra]
both = concat([tr] + extras)
n_extra = both["n"] - tr["n"]
order = np.random.default_rng(0).permutation(tr["n"])
held, pool = order[:10_000], order[10_000:]
extra_idx = np.arange(tr["n"], both["n"])
print(f"answer head: held-out {len(held)} train photos, dev3k {dv['n']}; train pool {len(pool)}, cross-fitted photos {n_extra}", flush=True)
for name, fit_rows in (("train only", pool), ("train + cross-fitted", np.concatenate([pool, extra_idx])), ("cross-fitted only", extra_idx)):
    rows = []
    for seed in range(args.seeds):
        net = fit_answer(both, fit_rows, seed=seed, epochs=max(5, min(25, round(25 * len(pool) / len(fit_rows) * 2))))
        r_h = evaluate(net, tr, held, [0])
        r_d = evaluate(net, dv, np.arange(dv["n"]), [0])
        rows.append([100 * np.mean(r[k] < t) for r in (r_h, r_d) for k in (("all", "all", "agree"), (0, "top", "agree")) for t in (25, 1)])
    m, s = np.mean(rows, 0), np.std(rows, 0)
    print(f"  {name:22s} ({len(fit_rows)}): held-out all-zoomed {m[0]:.1f}±{s[0]:.1f} / {m[1]:.1f} (25 / 1 km), no zoom {m[2]:.1f}±{s[2]:.1f}; "
          f"dev3k all-zoomed {m[4]:.1f}±{s[4]:.1f} / {m[5]:.1f}, no zoom {m[6]:.1f}±{s[6]:.1f}", flush=True)
