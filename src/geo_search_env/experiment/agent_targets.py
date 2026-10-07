# AGENT_DIRECTION.md target photos per photo set: where the reranker top-1 is >= 25 km off, is a pool candidate within 25 km ("choosing"),
# and would knowing the true region fix it ("fixable": the best-ranked candidate in the truth's (country, region) is < 25 km)?
# Usage: PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.agent_targets [--tags dev val full]
#        PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.agent_targets --select-big dev10k --n 10000   (bigger MP16 development set)
# Note: only `dev` (and sets built here) are MP16; `val` is 1,000 benchmark eval-half photos (query_evidence select --source bench).

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
    if truth.get("source") == "geonames":  # GeoNames codes ("US", "US.CA"): exact match
        return bool(truth.get("region")) and cand.get("region") == truth["region"]
    return bool(truth.get("country") and truth.get("region")) and _right(cand.get("country") or "", truth["country"]) and _right(cand.get("region") or "", truth["region"])


def select_big(tag: str, n: int) -> None:
    """A bigger MP16 development set: every eligible held-out `val`-split photo of the SFT pool (so `dev` is included), then held-out `train`-split
    photos (fixed shuffle) up to `n`; same filters and format as query_evidence.select. The train-split ids used are listed in
    artifacts/agent/{tag}_train_ids.json so later training on bucket-99 train can exclude them."""

    from .query_evidence import CANDIDATES, SFT_ROOT
    from .sft_data import OVERLAY_THRESHOLD
    from .strategy_search import MP16_EMBED
    from .verifiers import _place_names

    queries = json.loads((SFT_ROOT / "queries.json").read_text(encoding="utf-8"))
    scores = json.loads((SFT_ROOT / "overlay_scores.json").read_text(encoding="utf-8"))
    ok = lambda q: q["group"] == "held_out" and scores.get(q["image_id"]) is not None and scores[q["image_id"]] < OVERLAY_THRESHOLD
    val = [i for i, q in enumerate(queries) if ok(q) and q["split"] == "val"]
    train = [i for i, q in enumerate(queries) if ok(q) and q["split"] == "train"]
    picked = sorted(val + [int(i) for i in np.random.default_rng(1).permutation(train)[: max(0, n - len(val))]])
    saved = np.load(SFT_ROOT / "candidates.npz")
    coords, valid, ranking, name_row = saved["coords"], saved["valid"], saved["ranking"], saved["name_row"]
    latlon = np.fromfile(MP16_EMBED / "latlon_deg.f32.bin", dtype=np.float32).reshape(-1, 2).astype(np.float64)
    shown = {i: [int(c) for c in ranking[i, :CANDIDATES] if valid[i, c]] for i in picked}
    names = _place_names({int(name_row[i, c]) for i, cs in shown.items() for c in cs})
    rows = [{"index": i, "image_id": queries[i]["image_id"], "row": queries[i]["row"], "truth": latlon[queries[i]["row"]].tolist(),
             "candidates": [[names[int(name_row[i, c])], float(coords[i, c, 0]), float(coords[i, c, 1])] for c in shown[i]],
             "pool": [[float(coords[i, c, 0]), float(coords[i, c, 1])] for c in ranking[i] if valid[i, c]]} for i in picked]
    (ROOT / tag).mkdir(parents=True, exist_ok=True)
    (ROOT / tag / "dev.json").write_text(json.dumps(rows, ensure_ascii=False) + "\n", encoding="utf-8")
    used_train = sorted(queries[i]["image_id"] for i in picked if queries[i]["split"] == "train")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{tag}_train_ids.json").write_text(json.dumps(used_train) + "\n", encoding="utf-8")
    print(f"{tag}: {len(rows)} MP16 photos ({len(val)} val-split, {len(used_train)} train-split) -> {ROOT / tag / 'dev.json'}")


def geonames_labels(tag: str) -> None:
    """Place labels from the GeoNames index (country code and "cc.admin1" of the nearest populated place) for the truth and every pool candidate,
    in the candidates_{tag}.json format, marked source=geonames (Overture divisions aren't on this machine)."""

    from .geo_tools import Gazetteer

    g = Gazetteer()
    photos = json.loads((ROOT / tag / "dev.json").read_text(encoding="utf-8"))
    lab = lambda lat, lon: dict(zip(("country", "region"), g.region(lat, lon))) | {"source": "geonames"}
    out = [{"image_id": e["image_id"], "truth": lab(*e["truth"]), "pool": [lab(lat, lon) for lat, lon in e["pool"]]} for e in photos]
    (LABELS / f"candidates_{tag}.json").write_text(json.dumps(out) + "\n", encoding="utf-8")
    print(f"{tag}: GeoNames labels for {len(out)} photos -> {LABELS / f'candidates_{tag}.json'}")


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
    parser.add_argument("--select-big", default=None, help="build this MP16 development set (then GeoNames labels and targets)")
    parser.add_argument("--n", type=int, default=10_000)
    args = parser.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    if args.select_big:
        select_big(args.select_big, args.n)
        geonames_labels(args.select_big)
        args.tags = [args.select_big]
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
