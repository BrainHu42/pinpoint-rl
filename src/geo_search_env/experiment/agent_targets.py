# AGENT_DIRECTION.md target photos per photo set: where Pinpoint's top-1 is >= 25 km off, is a pool candidate within 25 km ("choosing"), and would
# knowing the true region fix it ("fixable": the best-ranked candidate in the truth's (country, region) is < 25 km)? Also the dev photo set and the splits.
# Usage: PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.agent_targets --make-dev dev3k   (photo set from the dev ids, Pinpoint's pool)
#        PYTHONPATH=src .venvs/overture/bin/python -m geo_search_env.experiment.place_labels pool-labels --tag dev3k   (Overture labels)
#        PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.agent_targets --tags dev3k                (targets)
#        PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.agent_targets --splits                    (artifacts/agent/splits.json)

"""Writes artifacts/agent/targets_{tag}.json: per photo the image id, reranker top-1 km, pool oracle km, choosing / fixable flags and the rank of
the first candidate in the true region (None if no candidate is there or the truth has no region label). Prints the counts and what a perfect
region chooser would add. Photo sets and labels come from photo_sets.py / place_labels.py (artifacts/photo_sets/<tag>/)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .knowledge_check import _right
from .photo_sets import PHOTO_SETS, load_photos
from .strategy_search import _haversine_km

OUT = Path("artifacts/agent")
DEV_IDS = OUT / "dev3k_ids.json"  # the 3,000 dev photos (a fixed random subset of MP16 bucket 99; CLAUDE.md "Data")


def load_labels(tag: str) -> list[dict[str, Any]]:
    return json.loads((PHOTO_SETS / tag / "labels.json").read_text(encoding="utf-8"))


def same_region(cand: dict, truth: dict) -> bool:
    return bool(truth.get("country") and truth.get("region")) and _right(cand.get("country") or "", truth["country"]) and _right(cand.get("region") or "", truth["region"])


def make_dev(tag: str = "dev3k") -> None:
    """The dev photo set: the photos listed in DEV_IDS, with Pinpoint's pool (photo_sets format)."""

    from .photo_sets import _entries, _save
    from .pinpoint_reference import load_pools

    pools = load_pools("mp16")
    queries = pools["queries"]
    position = {q["image_id"]: i for i, q in enumerate(queries)}
    picked = sorted(position[i] for i in json.loads(DEV_IDS.read_text(encoding="utf-8")))
    _save(tag, _entries(pools, picked, lambda i: {"index": i, "image_id": queries[i]["image_id"], "row": queries[i]["row"]}))
    print(f"{tag}: {len(picked)} MP16 photos with Pinpoint's pool -> {PHOTO_SETS / tag / 'photos.json'}")


def write_splits(dev_tag: str = "dev3k") -> None:
    """The one data split (CLAUDE.md): train = MP16 bucket-99 photos (held out from Pinpoint, burned-in GPS filtered) not in dev; dev = dev3k;
    test = the im2gps3k + yfcc4k eval halves (photo set `full`). Image ids -> artifacts/agent/splits.json."""

    from .sft_data import OVERLAY_THRESHOLD

    queries = json.loads(Path("artifacts/sft/queries.json").read_text(encoding="utf-8"))
    scores = json.loads(Path("artifacts/sft/overlay_scores.json").read_text(encoding="utf-8"))
    dev = [e["image_id"] for e in load_photos(dev_tag)]
    held = {q["image_id"] for q in queries if q["group"] == "held_out" and scores.get(q["image_id"]) is not None and scores[q["image_id"]] < OVERLAY_THRESHOLD}
    train = sorted(held - set(dev))
    test = [e["image_id"] for e in load_photos("full")]
    (OUT / "splits.json").write_text(json.dumps({"train": train, "dev": sorted(dev), "test": test}) + "\n", encoding="utf-8")
    print(f"splits: train {len(train)}, dev {len(dev)}, test {len(test)} -> {OUT / 'splits.json'}")


def targets(tag: str) -> list[dict]:
    photos = load_photos(tag)
    labels = {p["image_id"]: p for p in load_labels(tag)}
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
    parser.add_argument("--tags", nargs="+", default=["dev3k"])
    parser.add_argument("--make-dev", default=None, help="build this photo set from the dev ids and exit")
    parser.add_argument("--splits", action="store_true", help="write artifacts/agent/splits.json and exit")
    args = parser.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    if args.make_dev:
        make_dev(args.make_dev)
        return 0
    if args.splits:
        write_splits()
        return 0
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
