# Stage-2 go/no-go: is there signal for an RL chooser beyond the reranker + comparator? (CPU, existing scores and zero-shot answers)
# Usage: PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.model_vote_cv

"""Does the VLM's own answer add information beyond reranker rank + comparator? 5-fold CV combiner over dev + val top-8 candidates."""
import json

import numpy as np
import torch

from geo_search_env.experiment.exemplar_judge import TOPK, _load_topk, _logit
from geo_search_env.experiment.query_evidence import ROOT
from geo_search_env.experiment.stage1_eval import _bootstrap
from geo_search_env.experiment.wiki_backend import _km

photos = _load_topk("comparator-a")
excluded = set(json.load(open(ROOT / "val" / "exclude.json")))
entries = [e for tag in ("dev", "val") for e in json.load(open(ROOT / tag / "dev.json")) if e["image_id"] not in excluded]
assert len(entries) == len(photos)
answers = {m: {**json.load(open(ROOT / "dev" / f"scaling_{m}.json")), **json.load(open(ROOT / "val" / f"scaling_{m}.json"))} for m in ("qwen3.5-4b", "qwen3.5-9b", "qwen3.6-27b")}

D, base_f, votes, tags = [], [], {m: [] for m in answers}, np.asarray([p["tag"] for p in photos])
for p, e in zip(photos, entries):
    lg, n = _logit(p["p"]), len(p["dist"])
    rank = np.arange(n, dtype=float)
    order = (-lg).argsort().argsort().astype(float)
    f = np.stack([rank, (rank == 0).astype(float), (rank < 3).astype(float), lg, lg * rank, lg - lg.max(), order, (order == 0).astype(float), lg * (rank == 0)], axis=1)
    D.append(np.pad(p["dist"], (0, TOPK - n), constant_values=1e5)); base_f.append(np.pad(f, ((0, TOPK - n), (0, 0))))
    cand = np.asarray(e["pool"][:n])
    for m, a in answers.items():
        ans = a.get(e["image_id"], {}).get("answer")
        if ans is None:
            v = np.zeros((n, 3))
        else:
            d = _km(cand, *ans)
            v = np.stack([np.log1p(d), (d < 25).astype(float), (d == d.min()).astype(float)], axis=1)
        votes[m].append(np.pad(v, ((0, TOPK - n), (0, 0))))
D, base_f = np.asarray(D), np.asarray(base_f, dtype=np.float32)
valid = D < 9e4
reward = np.where(valid, (D < 25) * 1.0 + (D < 1) * 0.5, 0.0).astype(np.float32)


def cv(F):
    mean, std = F[valid].mean(0), F[valid].std(0) + 1e-6
    X, M, R = torch.as_tensor((F - mean) / std), torch.as_tensor(valid), torch.as_tensor(reward)
    folds = np.array_split(np.random.default_rng(0).permutation(len(D)), 5)
    hits = np.zeros(len(D))
    for seed in range(3):
        for f in folds:
            torch.manual_seed(seed)
            net = torch.nn.Sequential(torch.nn.Linear(F.shape[-1], 16), torch.nn.GELU(), torch.nn.Linear(16, 1))
            opt = torch.optim.AdamW(net.parameters(), lr=1e-2, weight_decay=1e-2)
            idx = torch.as_tensor(np.setdiff1d(np.arange(len(D)), f))
            for _ in range(300):
                loss = -(torch.softmax(net(X[idx]).squeeze(-1).masked_fill(~M[idx], float("-inf")), -1) * R[idx]).sum(-1).mean()
                opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                pick = net(X[f]).squeeze(-1).masked_fill(~M[f], float("-inf")).argmax(1).numpy()
            hits[f] += (D[f, pick] < 25) / 3
    return hits


base = (D[:, 0] < 25).astype(float)
rank_only = base_f[..., :3]
configs = {"rank + comparator": base_f}
for m in answers:
    configs[f"rank + {m} vote"] = np.concatenate((rank_only, np.asarray(votes[m], dtype=np.float32)), -1)
    configs[f"rank + comparator + {m} vote"] = np.concatenate((base_f, np.asarray(votes[m], dtype=np.float32)), -1)
configs["rank + comparator + all three votes"] = np.concatenate([base_f] + [np.asarray(votes[m], dtype=np.float32) for m in answers], -1)
print(f"{len(D)} photos (dev {int((tags == 'dev').sum())}, val {int((tags == 'val').sum())}); reranker top-1 <25 km {base.mean():.1%}; top-8 oracle {(D < 25).any(1).mean():.1%}")
results = {}
for name, F in configs.items():
    h = cv(F)
    results[name] = h
    ci = _bootstrap(h - base)
    print(f"  {name:42s} {h.mean():.1%}  vs reranker {ci[0]:+.1%} [{ci[1]:+.1%}, {ci[2]:+.1%}]  dev {h[tags == 'dev'].mean() - base[tags == 'dev'].mean():+.1%} val {h[tags == 'val'].mean() - base[tags == 'val'].mean():+.1%}")
for m in answers:
    ci = _bootstrap(results[f"rank + comparator + {m} vote"] - results["rank + comparator"])
    print(f"  {m} vote beyond rank + comparator: {ci[0]:+.1%} [{ci[1]:+.1%}, {ci[2]:+.1%}]")
