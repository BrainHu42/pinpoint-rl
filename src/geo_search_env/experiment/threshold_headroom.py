# Is there more choosing headroom, and more signal in our evidence, at region / country / continent scale than at 25 km? (CPU, existing scores)
# Usage: PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.threshold_headroom

"""For each threshold (25 / 200 / 750 / 2500 km: city, region, country, continent as in im2gps): reranker top-1, top-8 and whole-pool oracle on the
1,985 dev + val photos (placeholders dropped), and 5-fold, 3-seed cross-validated listwise combiners over the top-8 candidates trained with the
reward "within this threshold": rank only, + `comparator-a`, + each zero-shot chooser's answer as a vote (lesson 33), + Overture category match
(lesson 34), and all together. A combiner on rank alone shows what retargeting the ranking to the threshold gains by itself."""

from __future__ import annotations

import json

import numpy as np

from .category_evidence import OUT as CATEGORY_OUT, RADII_KM
from .exemplar_judge import _load_topk, _logit
from .query_evidence import ROOT
from .stage1_eval import _bootstrap
from .wiki_backend import _km

THRESHOLDS = (25.0, 200.0, 750.0, 2500.0)
TOPK = 8
MODELS = ("qwen3.5-4b", "qwen3.5-9b", "qwen3.6-27b")


def _cv(hit: np.ndarray, F: np.ndarray, valid: np.ndarray) -> np.ndarray:
    import torch

    reward = np.where(valid, hit, 0.0).astype(np.float32)
    mean, std = F[valid].mean(0), F[valid].std(0) + 1e-6
    X, M, R = torch.as_tensor((F - mean) / std), torch.as_tensor(valid), torch.as_tensor(reward)
    folds = np.array_split(np.random.default_rng(0).permutation(len(F)), 5)
    out = np.zeros(len(F))
    for seed in range(3):
        for f in folds:
            torch.manual_seed(seed)
            net = torch.nn.Sequential(torch.nn.Linear(F.shape[-1], 16), torch.nn.GELU(), torch.nn.Linear(16, 1))
            opt = torch.optim.AdamW(net.parameters(), lr=1e-2, weight_decay=1e-2)
            idx = torch.as_tensor(np.setdiff1d(np.arange(len(F)), f))
            for _ in range(300):
                loss = -(torch.softmax(net(X[idx]).squeeze(-1).masked_fill(~M[idx], float("-inf")), -1) * R[idx]).sum(-1).mean()
                opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                pick = net(X[f]).squeeze(-1).masked_fill(~M[f], float("-inf")).argmax(1).numpy()
            out[f] += hit[f, pick] / 3
    return out


def main() -> None:
    excluded = set(json.loads((ROOT / "val" / "exclude.json").read_text(encoding="utf-8")))
    photos = [dict(e, tag=tag) for tag in ("dev", "val") for e in json.loads((ROOT / tag / "dev.json").read_text(encoding="utf-8")) if e["image_id"] not in excluded]
    judged = _load_topk("comparator-a")
    assert len(judged) == len(photos)
    answers = {m: {**json.loads((ROOT / "dev" / f"scaling_{m}.json").read_text(encoding="utf-8")), **json.loads((ROOT / "val" / f"scaling_{m}.json").read_text(encoding="utf-8"))} for m in MODELS}
    saved = np.load(CATEGORY_OUT / "counts.npz")
    present, offsets = saved["counts"] > 0, saved["offsets"]
    p = np.load(CATEGORY_OUT / "p_visible.npy").astype(np.float64)
    logit = np.log(p) - np.log1p(-p)
    p_vis = np.exp(logit - logit.max(1, keepdims=True))
    p_vis /= p_vis.sum(1, keepdims=True)
    idf = np.log(present.shape[1] / (present.sum(1) + 1.0))

    D, pool_min, base_f, votes, cats = [], [], [], {m: [] for m in MODELS}, []
    tags = np.asarray([e["tag"] for e in photos])
    for i, (e, j) in enumerate(zip(photos, judged)):
        n = len(j["dist"])
        lg, rank = _logit(j["p"]), np.arange(n, dtype=float)
        order = (-lg).argsort().argsort().astype(float)
        f = np.stack([rank, (rank == 0).astype(float), (rank < 3).astype(float), lg, lg * rank, lg - lg.max(), order, (order == 0).astype(float), lg * (rank == 0)], axis=1)
        pad = lambda a: np.pad(a, ((0, TOPK - n), (0, 0)))
        D.append(np.pad(j["dist"], (0, TOPK - n), constant_values=1e5))
        pool_min.append(_km(np.asarray(e["pool"]), *e["truth"]).min())
        base_f.append(pad(f))
        cand = np.asarray(e["pool"][:n])
        for m in MODELS:
            ans = answers[m].get(e["image_id"], {}).get("answer")
            d = _km(cand, *ans) if ans is not None else None
            votes[m].append(pad(np.zeros((n, 3)) if d is None else np.stack([np.log1p(d), (d < 200).astype(float), (d == d.min()).astype(float)], axis=1)))
        rows = slice(offsets[i], offsets[i] + n)
        cats.append(pad(np.stack([(present[k, rows] * p_vis[i] * idf[k]).sum(1) for k in range(len(RADII_KM))], axis=1)))
    D, pool_min = np.asarray(D), np.asarray(pool_min)
    base_f, cats = np.asarray(base_f, dtype=np.float32), np.asarray(cats, dtype=np.float32)
    votes = {m: np.asarray(v, dtype=np.float32) for m, v in votes.items()}
    valid = D < 9e4
    configs = {
        "rank only (retargeted)": base_f[..., :3],
        "rank + comparator": base_f,
        **{f"rank + {m} vote": np.concatenate((base_f[..., :3], votes[m]), -1) for m in MODELS},
        "rank + categories": np.concatenate((base_f[..., :3], cats), -1),
        "rank + comparator + 27B vote + categories": np.concatenate((base_f, votes["qwen3.6-27b"], cats), -1),
    }

    report = {}
    print(f"{len(D)} photos (dev {int((tags == 'dev').sum())}, val {int((tags == 'val').sum())}); combiners over the top {TOPK}, change vs reranker top-1 [95% CI]")
    for t in THRESHOLDS:
        hit = (D < t).astype(float)
        base = hit[:, 0]
        entry = {"reranker top-1": float(base.mean()), "top-8 oracle": float(hit.max(1).mean()), "pool oracle": float((pool_min < t).mean()),
                 "dev reranker / pool oracle": [float(base[tags == "dev"].mean()), float((pool_min[tags == "dev"] < t).mean())],
                 "val reranker / pool oracle": [float(base[tags == "val"].mean()), float((pool_min[tags == "val"] < t).mean())]}
        print(f"\n< {t:g} km: reranker top-1 {base.mean():.1%}, top-8 oracle {hit.max(1).mean():.1%} (gap {hit.max(1).mean() - base.mean():+.1%}), "
              f"pool oracle {(pool_min < t).mean():.1%} (gap {(pool_min < t).mean() - base.mean():+.1%})")
        for name, F in configs.items():
            h = _cv(hit, F, valid)
            ci = _bootstrap(h - base)
            entry[name] = ci
            print(f"  {name:44s} {h.mean():6.1%}  {ci[0]:+.1%} [{ci[1]:+.1%}, {ci[2]:+.1%}]  (dev {h[tags == 'dev'].mean() - base[tags == 'dev'].mean():+.1%}, val {h[tags == 'val'].mean() - base[tags == 'val'].mean():+.1%})")
        report[f"{t:g} km"] = entry
    (ROOT / "threshold_headroom.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
