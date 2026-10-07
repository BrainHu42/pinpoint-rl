# AGENT_DIRECTION.md target photos per photo set: where the reranker top-1 is >= 25 km off, is a pool candidate within 25 km ("choosing"),
# and would knowing the true region fix it ("fixable": the best-ranked candidate in the truth's (country, region) is < 25 km)?
# Usage: PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.agent_targets [--tags dev val full]

"""Writes artifacts/agent/targets_{tag}.json: per photo the image id, reranker top-1 km, pool oracle km, choosing / fixable flags and the rank of
the first candidate in the true region (None if no candidate is there or the truth has no region label). Prints the counts and what a perfect
region chooser would add."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from .knowledge_check import _right
from .query_evidence import ROOT
from .strategy_search import _haversine_km

LABELS = Path("artifacts/place_labels")
OUT = Path("artifacts/agent")


def same_region(cand: dict, truth: dict) -> bool:
    return bool(truth.get("country") and truth.get("region")) and _right(cand.get("country") or "", truth["country"]) and _right(cand.get("region") or "", truth["region"])


def targets(tag: str) -> list[dict]:
    photos = json.loads((ROOT / tag / "dev.json").read_text(encoding="utf-8"))
    labels = {p["image_id"]: p for p in json.loads((LABELS / f"candidates_{tag}.json").read_text(encoding="utf-8"))}
    out = []
    for e in photos:
        lab = labels[e["image_id"]]
        km = _haversine_km(*e["truth"], np.asarray(e["pool"]))
        rank = next((j for j, c in enumerate(lab["pool"]) if same_region(c, lab["truth"])), None)
        out.append({"image_id": e["image_id"], "top1_km": float(km[0]), "oracle_km": float(km.min()),
                    "choosing": bool(km[0] >= 25 and km.min() < 25), "fixable": bool(km[0] >= 25 and rank is not None and km[rank] < 25),
                    "region_rank": rank, "region_labelled": bool(lab["truth"].get("region"))})
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tags", nargs="+", default=["dev", "val", "full"])
    args = parser.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    for tag in args.tags:
        rows = targets(tag)
        (OUT / f"targets_{tag}.json").write_text(json.dumps(rows) + "\n", encoding="utf-8")
        n = len(rows)
        top1 = np.mean([r["top1_km"] < 25 for r in rows])
        choosing, fixable = sum(r["choosing"] for r in rows), sum(r["fixable"] for r in rows)
        ranks = np.asarray([r["region_rank"] for r in rows if r["fixable"]])
        print(f"{tag}: n={n}, reranker top-1 < 25 km {top1:.1%}; choosing {choosing} ({choosing / n:.1%}); fixable {fixable} ({fixable / n:.1%}, "
              f"perfect region chooser +{100 * fixable / n:.1f} pts minus breaks); fixable rank of first true-region candidate: "
              f"1 {np.mean(ranks == 1):.0%}, 2-5 {np.mean((ranks >= 2) & (ranks <= 5)):.0%}, 6+ {np.mean(ranks >= 6):.0%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
