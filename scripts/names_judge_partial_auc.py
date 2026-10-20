"""Dev AUC of several judges on the dev pairs the first one has answered so far (P(head's top), orders averaged, for the head being right),
with a paired bootstrap CI; for comparing a slow judge mid-run.

Usage: .venv/bin/python scripts/names_judge_partial_auc.py qwen25vl72b base2 gemma27b names-exact-a
"""
import json
import sys

import numpy as np


def auc(s, y):
    r = np.argsort(np.argsort(s)) + 1
    return (r[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum())


names = sys.argv[1:]
pairs = {r["photo"]: r for r in map(json.loads, open("artifacts/agent/names_grpo_pairs.jsonl")) if r["split"] == "dev"}
ans = {n: {r["photo"]: np.mean(r["p_top"]) for r in map(json.loads, open(f"artifacts/agent/names_grpo_{n}.jsonl"))
           if r.get("split") == "dev" and "p_top" in r} for n in names}
photos = [p for p in ans[names[0]] if all(p in ans[n] for n in names)]
y = np.array([pairs[p]["km"][0] < 25 for p in photos])
print(f"{len(photos)} dev pairs answered by all ({(~y).sum()} head-wrong)")
rng = np.random.default_rng(0)
boot = [rng.integers(len(y), size=len(y)) for _ in range(2000)]
scores = {n: np.array([ans[n][p] for p in photos]) for n in names}
for n in names:
    s = scores[n]
    b = [auc(s[i], y[i]) for i in boot]
    d = [auc(s[i], y[i]) - auc(scores[names[0]][i], y[i]) for i in boot] if n != names[0] else None
    print(f"  {n:14s} AUC {auc(s, y):.3f} [{np.quantile(b, .025):.3f}, {np.quantile(b, .975):.3f}]"
          + (f"  minus {names[0]}: {np.mean(d):+.3f} [{np.quantile(d, .025):+.3f}, {np.quantile(d, .975):+.3f}]" if d else "")
          + f"  right on head-wrong {np.mean(s[~y] < 0.5):.1%}")
