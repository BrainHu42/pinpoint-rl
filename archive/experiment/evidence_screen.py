# Screen a kind of evidence before building on it: is it redundant with retrieval, and does it help where retrieval is confused?
# Usage: .venv/bin/python -m geo_search_env.experiment.evidence_screen --tag dev --variant text-9b   (CPU; needs stage1_eval's cached results)

"""For one evidence variant (the queries in `places_<variant>.json`, their results in `results_<variant>.json`), on one photo set:

headroom: the share of photos the evidence applies to (here: photos with at least one query), overall and among the photos the
          pool misses or the reranker gets wrong, and how good retrieval already is on them.
redundancy: how often the evidence lands within 25 km of the truth when the pool already holds the answer vs when it misses it.
discrimination: on "choosing" photos (answer in the pool, reranker top-1 wrong), how often it supports a right candidate vs the
          wrong top-1.
gain: oracle gain over the pool, over all photos and over the photos with evidence.
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Sequence

import numpy as np

from .query_evidence import ROOT
from .stage1_eval import _suffix
from .wiki_backend import _km

R = 25.0
ARMS = ("siglip", "bm25", "dense")  # results pooled over these backends


def screen(tag: str, variant: str) -> dict[str, Any]:
    dev = json.loads((ROOT / tag / "dev.json").read_text(encoding="utf-8"))
    places = json.loads((ROOT / tag / f"places{_suffix(variant)}.json").read_text(encoding="utf-8"))
    results = json.loads((ROOT / tag / f"results{_suffix(variant)}.json").read_text(encoding="utf-8"))
    n = len(dev)
    applies = np.asarray([bool(places[e["image_id"]]["queries"]) for e in dev])
    pool_hit, top1_hit, ev_hit, sup_top1, sup_right = (np.zeros(n, bool) for _ in range(5))
    for m, e in enumerate(dev):
        pool, truth = np.asarray(e["pool"]), e["truth"]
        d_pool = _km(pool, *truth)
        pool_hit[m], top1_hit[m] = d_pool.min() < R, d_pool[0] < R
        pts = np.asarray([p for arm in ARMS for p in results[arm][m]]).reshape(-1, 2)
        if len(pts):
            ev_hit[m] = _km(pts, *truth).min() < R
            sup_top1[m] = _km(pts, *pool[0]).min() < R
            sup_right[m] = any(_km(pts, *pool[c]).min() < R for c in np.flatnonzero(d_pool < R))
    choosing = pool_hit & ~top1_hit
    miss = ~pool_hit

    def rate(mask: np.ndarray, of: np.ndarray) -> float:
        return float(of[mask].mean()) if mask.any() else float("nan")

    report: dict[str, Any] = {
        "n": n, "apply share": float(applies.mean()),
        "apply share | pool miss": rate(miss, applies), "apply share | choosing": rate(choosing, applies), "apply share | top-1 right": rate(top1_hit, applies),
        "reranker top-1 right | applies": rate(applies, top1_hit), "reranker top-1 right | not": rate(~applies, top1_hit),
        "pool hit | applies": rate(applies, pool_hit), "pool hit | not": rate(~applies, pool_hit),
        "evidence hit | applies & pool hit": rate(applies & pool_hit, ev_hit), "evidence hit | applies & pool miss": rate(applies & miss, ev_hit),
        "choosing & applies: n": int((choosing & applies).sum()),
        "choosing & applies: supports right": rate(choosing & applies, sup_right), "choosing & applies: supports wrong top-1": rate(choosing & applies, sup_top1),
        "misses recovered": int((ev_hit & miss).sum()), "misses": int(miss.sum()),
        "oracle gain <25 km (all photos)": float((ev_hit & miss).mean()),
        "oracle gain <25 km (photos with evidence, of those)": rate(applies, ev_hit & miss),
    }
    print(f"\n{tag} / {variant}  (n={n})")
    print(f"  applies to {report['apply share']:.0%} of photos; among pool misses {report['apply share | pool miss']:.0%}, among choosing photos "
          f"{report['apply share | choosing']:.0%}, among top-1-right photos {report['apply share | top-1 right']:.0%}")
    print(f"  reranker top-1 right: {report['reranker top-1 right | applies']:.0%} on photos with evidence vs {report['reranker top-1 right | not']:.0%} without; "
          f"pool holds the answer: {report['pool hit | applies']:.0%} vs {report['pool hit | not']:.0%}")
    print(f"  evidence within 25 km of truth: {report['evidence hit | applies & pool hit']:.0%} when the pool holds the answer, "
          f"{report['evidence hit | applies & pool miss']:.1%} when it misses (recovers {report['misses recovered']} of {report['misses']} misses)")
    print(f"  choosing photos with evidence (n={report['choosing & applies: n']}): supports a right candidate {report['choosing & applies: supports right']:.0%}, "
          f"supports the wrong top-1 {report['choosing & applies: supports wrong top-1']:.0%}")
    print(f"  oracle gain over the pool at 25 km: {report['oracle gain <25 km (all photos)']:+.1%} of all photos")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--tags", nargs="+", default=["dev", "val"])
    args = parser.parse_args(argv)
    out = {tag: screen(tag, args.variant) for tag in args.tags}
    (ROOT / f"screen_{args.variant}.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
