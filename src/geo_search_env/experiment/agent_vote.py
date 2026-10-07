# Do independent place guesses that agree buy precision? Sources: knowledge_check answers of several models ("country > region > city"). Rule: if at
# least `need` sources name the same (country[, region]) and the reranker top-1 is not there, switch to the best-ranked of the first K pool candidates
# that is; else keep the top-1. Fixes / breaks vs the reranker top-1 at 25 km on MP16 dev / val (choose here; benchmarks only for a fixed rule).
# Usage: PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.agent_vote --models base4b 9b [--tags dev val]

from __future__ import annotations

import argparse
import itertools
import json
from typing import Sequence

import numpy as np

from .agent_targets import OUT as TARGETS, load_labels
from .knowledge_check import OUT, _norm, _right
from .photo_sets import load_photos
from .strategy_search import _haversine_km


def _guess(answer: str | None, depth: int) -> tuple[str, ...] | None:
    parts = [p.strip() for p in (answer or "").split(">")]
    return tuple(_norm(p) for p in parts[:depth]) if len(parts) >= depth and all(parts[:depth]) else None


def _in(cand: dict, guess: tuple[str, ...]) -> bool:
    keys = ("country", "region")
    return all(_right(guess[d], cand.get(keys[d])) for d in range(len(guess)))


def run(models: Sequence[str], tags: Sequence[str], depth: int, need: int, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    chosen, base, fix, switched = [], [], [], []
    for tag in tags:
        photos = load_photos(tag)
        labels = {p["image_id"]: p["pool"] for p in load_labels(tag)}
        targets = {r["image_id"]: r for r in json.loads((TARGETS / f"targets_{tag}.json").read_text(encoding="utf-8"))}
        answers = [json.loads((OUT / tag / f"check_{m}.json").read_text(encoding="utf-8")) for m in models]
        for e in photos:
            km = _haversine_km(*e["truth"], np.asarray(e["pool"]))
            pool = labels[e["image_id"]]
            guesses = [g for g in (_guess(a.get(e["image_id"]), depth) for a in answers) if g]
            pick = 0
            # the guess with the most agreeing sources (a source agrees if every level matches), if it has enough support
            support = {g: sum(all(_right(g[d], h[d]) for d in range(depth)) for h in guesses) for g in set(guesses)}
            best = max(support, key=support.get, default=None)
            if best is not None and support[best] >= need and not _in(pool[0], best):
                hit = next((j for j, c in enumerate(pool[:k]) if _in(c, best)), None)
                pick = hit if hit is not None else 0
            chosen.append(km[pick]); base.append(km[0]); fix.append(targets[e["image_id"]]["fixable"]); switched.append(pick != 0)
    return np.asarray(chosen), np.asarray(base), np.asarray(fix), np.asarray(switched)


def main(argv: Sequence[str] | None = None) -> int:
    from .strategy_search import _bootstrap

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--tags", nargs="+", default=["dev3k"])
    args = parser.parse_args(argv)
    print(f"{'+'.join(args.tags)}; sources {', '.join(args.models)}")
    for depth, k in itertools.product((1, 2), (3, 10)):
        for need in range(1, len(args.models) + 1):
            chosen, base, fix, sw = run(args.models, args.tags, depth, need, k)
            c = _bootstrap((chosen < 25) * 1.0 - (base < 25))
            fixes, breaks = ((chosen < 25) & (base >= 25)).sum(), ((chosen >= 25) & (base < 25)).sum()
            print(f"  {'country' if depth == 1 else 'region':7s} top {k:2d}, agree >= {need}: switched {sw.mean():5.1%}; < 25 km {100 * c[0]:+.1f} "
                  f"[{100 * c[1]:+.1f}, {100 * c[2]:+.1f}]; fixes {fixes} (fixable {((chosen < 25) & fix).sum()}), breaks {breaks}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
