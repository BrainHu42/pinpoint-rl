# Capacity question (AGENT_DIRECTION.md): per model, how often its "country > region > city" answer (knowledge_check answer --tag) names the truth,
# on the MP16 dev + val fixable photos vs the photos the reranker gets right; paired difference to a reference model with a bootstrap interval.
# Usage: PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.agent_capacity --models base4b 9b [--ref base4b] [--tags dev val]

from __future__ import annotations

import argparse
import json
from typing import Sequence

import numpy as np

from .agent_targets import OUT as TARGETS, load_labels
from .knowledge_check import LEVELS, OUT, _right


def hits(model: str, tags: Sequence[str]) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    out = {lv: [] for lv, _ in LEVELS}
    fix, right = [], []
    for tag in tags:
        answers = json.loads((OUT / tag / f"check_{model}.json").read_text(encoding="utf-8"))
        truth = {p["image_id"]: p["truth"] for p in load_labels(tag)}
        for r in json.loads((TARGETS / f"targets_{tag}.json").read_text(encoding="utf-8")):
            parts = [x.strip() for x in (answers.get(r["image_id"]) or "").split(">")]
            t = truth[r["image_id"]]
            for d, (lv, key) in enumerate(LEVELS):
                ok = len(parts) > d and all(_right(parts[e], t.get(LEVELS[e][1])) for e in range(d + 1)) if t.get(key) else None
                out[lv].append(np.nan if ok is None else float(ok))
            fix.append(r["fixable"]); right.append(r["top1_km"] < 25)
    return {lv: np.asarray(v) for lv, v in out.items()}, np.asarray(fix), np.asarray(right)


def main(argv: Sequence[str] | None = None) -> int:
    from .strategy_search import _bootstrap

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--ref", default=None)
    parser.add_argument("--tags", nargs="+", default=["dev3k"])
    args = parser.parse_args(argv)
    ref = args.ref or args.models[0]
    data = {m: hits(m, args.tags) for m in args.models}
    _, fix, right = data[ref]
    print(f"{'+'.join(args.tags)}: n={len(fix)}, fixable {fix.sum()}, reranker right {right.sum()}. Truth named down to each level (cumulative):")
    for label, mask in (("fixable", fix), ("reranker right", right), ("all", np.ones(len(fix), bool))):
        print(f"  {label}:")
        for lv, _ in LEVELS:
            row = "   ".join(f"{m} {np.nanmean(data[m][0][lv][mask]):.1%}" for m in args.models)
            diffs = []
            for m in args.models:
                if m == ref:
                    continue
                a, b = data[m][0][lv][mask], data[ref][0][lv][mask]
                ok = ~np.isnan(a) & ~np.isnan(b)
                c = _bootstrap(a[ok] - b[ok])
                diffs.append(f"{m} - {ref} {100 * c[0]:+.1f} [{100 * c[1]:+.1f}, {100 * c[2]:+.1f}]")
            print(f"    {lv:8s} {row}   " + "; ".join(diffs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
