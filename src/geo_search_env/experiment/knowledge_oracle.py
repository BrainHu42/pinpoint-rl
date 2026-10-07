# Lesson 52 / archive/OVERNIGHT.md NO-GO diagnosis: what perfect place knowledge at each level would add over the reranker top-1 on the full eval halves,
# and how often the knowledge model names the truth on the photos it would fix. CPU, seconds; needs knowledge_check answer outputs.
# Usage: PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.knowledge_oracle [tag]   (default full; MP16 sets e.g. dev)

"""NO-GO diagnosis (archive/OVERNIGHT.md): can place knowledge at a given level fix reranker failures at all, and does knowledge-a have it there?

Oracle-knowledge rule: take the best-ranked pool candidate whose labels match the TRUTH's labels down to a level (else keep the top-1).
Then, on the photos that rule fixes, how often knowledge-a / base name the truth down to that level."""

import json
import sys

import numpy as np

from geo_search_env.experiment.knowledge_check import LEVELS, OUT, _right
from geo_search_env.experiment.name_score import _labels
from geo_search_env.experiment.photo_sets import load_photos
from geo_search_env.experiment.strategy_search import _bootstrap, _haversine_km

TAG = sys.argv[1] if len(sys.argv) > 1 else "full"  # an MP16 set (dev) for the development-set version of the target photos
photos = load_photos(TAG)
labels = {p["image_id"]: p for p in _labels(TAG)}
answers = {p.stem.removeprefix("check_"): json.loads(p.read_text()) for p in sorted((OUT if TAG == "full" else OUT / TAG).glob("check_*.json"))}  # every model answered by knowledge_check
km = [_haversine_km(*e["truth"], np.asarray(e["pool"])) for e in photos]
base = np.array([d[0] for d in km])
oracle = np.array([d.min() for d in km])
choosing = (base >= 25) & (oracle < 25)
print(f"{TAG} n={len(photos)}; reranker top-1 <25 km {np.mean(base < 25):.1%}; choosing photos (top-1 wrong, a pool candidate right) {choosing.sum()} ({choosing.mean():.1%})")


def same(a, b):
    """Two division names agree (normalised; one contains the other)."""
    return _right(a or "", b)


for depth in (1, 2, 3):
    keys = [k for _, k in LEVELS[:depth]]
    chosen, has_truth_label = np.zeros(len(photos)), np.zeros(len(photos), bool)
    for i, e in enumerate(photos):
        lab = labels[e["image_id"]]
        t = lab["truth"]
        if not all(t.get(k) for k in keys):
            chosen[i] = base[i]
            continue
        has_truth_label[i] = True
        hit = next((j for j, c in enumerate(lab["pool"]) if all(same(c.get(k), t[k]) for k in keys)), None)
        chosen[i] = km[i][hit if hit is not None else 0]
    fixed, broke = (chosen < 25) & (base >= 25), (chosen >= 25) & (base < 25)
    c = _bootstrap((chosen < 25) * 1.0 - (base < 25))
    name = LEVELS[depth - 1][0]
    print(f"\nperfect {name}-level knowledge (truth labelled to this level on {has_truth_label.sum()}): <25 km {100 * c[0]:+.1f} [{100 * c[1]:+.1f}, {100 * c[2]:+.1f}]; "
          f"fixes {fixed.sum()} of {choosing.sum()} choosing photos, breaks {broke.sum()}")
    for m, ans in answers.items():
        said_right = np.zeros(len(photos), bool)
        for i, e in enumerate(photos):
            parts = [x.strip() for x in (ans.get(e["image_id"]) or "").split(">")]
            t = labels[e["image_id"]]["truth"]
            said_right[i] = len(parts) >= depth and all(_right(parts[d], t.get(keys[d])) for d in range(depth))
        print(f"  {m:12s} names the truth to this level: on the photos this fixes {said_right[fixed].mean():.1%} ({said_right[fixed].sum()}), "
              f"on all choosing photos {said_right[choosing].mean():.1%}, on photos the reranker gets right {said_right[base < 25].mean():.1%}")
