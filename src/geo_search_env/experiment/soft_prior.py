# Rerank retrieval with a soft VLM country/region prior instead of a hard filter.
# Usage: PYTHONPATH=src python -m geo_search_env.experiment.soft_prior --hypotheses artifacts/vlm_guesses/gemma4_26b_a4b.jsonl --output artifacts/soft_prior/v1

"""Offline study: retrieval score + lambda * log p_VLM(country/region), tuned on half the random queries."""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
import time
from typing import Any, Sequence

import numpy as np

from ..core.contracts import Coordinate
from ..core.geography import distance_m
from .candidate_locations import RAW_MP16_K, RAW_OSV5M_K, merge_candidate_locations
from .coarse_filter import THRESHOLDS_M, _locations, _masked_topk, load_world


LAMBDAS = (0.0, 0.003, 0.01, 0.03)
PRIOR_FLOOR = 0.02
ANCHORS = 3
BUDGETS = (1, 10, 25, 50)
OBJECTIVE = "le_25km@10"


def _log_prior(label_ids: np.ndarray, probabilities: Sequence[float], size: int):
    import torch

    mass = np.zeros(size + 1, dtype=np.float64)  # last slot = unlabelled gallery rows (label -1)
    for label, probability in zip(label_ids, probabilities):
        if label >= 0:
            mass[label] += probability
    return torch.as_tensor(np.log(mass + PRIOR_FLOOR), dtype=torch.float32)


def run(output: Path, *, hypotheses_path: Path, seed: int = 0) -> dict[str, Any]:
    import torch

    if output.exists():
        raise FileExistsError(f"refusing to overwrite soft-prior output: {output}")
    started = time.time()
    world = load_world(seed)
    hypotheses = {
        row["episode_id"]: row
        for row in map(json.loads, hypotheses_path.read_text(encoding="utf-8").splitlines())
        if row["guesses"]
    }
    n_countries, n_regions = len(world.vocab["country"]), len(world.vocab["region"])
    random_ids = [q["query_id"] for q in world.queries if q["set"] == "random"]
    tune_ids = set(random_ids[: len(random_ids) // 2])

    rows: list[dict[str, Any]] = []
    for number, (query, projection) in enumerate(zip(world.queries, world.query_projection)):
        hyp = hypotheses.get(query["query_id"])
        if hyp is None:
            continue
        index = query["index"]
        truth = Coordinate(float(world.test["latitude"][index]), float(world.test["longitude"][index]))
        allowed_osv = world.osv_base_allowed.clone()
        allowed_osv[torch.as_tensor(world.sequence_rows[world.test["sequences"][index]], device=world.device)] = False

        latitudes = np.asarray([g[0] for g in hyp["guesses"]])
        longitudes = np.asarray([g[1] for g in hyp["guesses"]])
        probabilities = hyp["probabilities"]
        country_prior = _log_prior(world.country_grid.lookup(latitudes, longitudes), probabilities, n_countries).to(world.device)
        region_prior = _log_prior(world.region_grid.lookup(latitudes, longitudes), probabilities, n_regions).to(world.device)
        mp16_prior = (country_prior[world.mp16_country], region_prior[world.mp16_region])
        osv_prior = (country_prior[world.osv_country], region_prior[world.osv_region])
        mp16_scores = (world.mp16_embeddings @ projection).float()
        osv_scores = (world.osv_projection @ projection).float()
        order = np.argsort(probabilities)[::-1][:ANCHORS]
        anchors = [Coordinate(float(latitudes[i]), float(longitudes[i])) for i in order]

        split = "pilot" if query["set"] == "pilot_report" else ("tune" if query["query_id"] in tune_ids else "eval")
        result: dict[str, Any] = {"query_id": query["query_id"], "split": split, "overlay": bool(hyp.get("overlay"))}
        for lambda_country, lambda_region in itertools.product(LAMBDAS, LAMBDAS):
            mp16_idx, mp16_val = _masked_topk(mp16_scores + lambda_country * mp16_prior[0] + lambda_region * mp16_prior[1], None, RAW_MP16_K)
            osv_idx, osv_val = _masked_topk(osv_scores + lambda_country * osv_prior[0] + lambda_region * osv_prior[1], allowed_osv, RAW_OSV5M_K)
            mp16_locations = _locations(mp16_idx, mp16_val, world.mp16_coordinates, "mp16")
            osv_locations = _locations(osv_idx, osv_val, world.dataset._coordinates, "osv5m")
            merged = [p.coordinate for p in merge_candidate_locations(mp16_locations, osv_locations, limit=max(BUDGETS))]
            osv_only = [p.coordinate for p in osv_locations[: max(BUDGETS)]]
            pools = {"merged": merged, "merged+anchors": anchors + merged, "osv": osv_only, "osv+anchors": anchors + osv_only}
            for pool_name, pool in pools.items():
                extra = ANCHORS if pool_name.endswith("+anchors") else 0
                for budget in BUDGETS:
                    error = min((distance_m(truth, c) for c in pool[: budget + extra]), default=math.inf)
                    result[f"{pool_name}|{lambda_country}|{lambda_region}|{budget}"] = error
        rows.append(result)
        if (number + 1) % 20 == 0:
            print(f"queries {number + 1}/{len(world.queries)}", flush=True)

    def metrics(subset: Sequence[dict[str, Any]], key_prefix: str) -> dict[str, float]:
        out: dict[str, float] = {"n": len(subset)}
        for budget in BUDGETS:
            values = [row[f"{key_prefix}|{budget}"] for row in subset]
            for threshold in THRESHOLDS_M:
                out[f"le_{int(threshold / 1000)}km@{budget}"] = sum(v <= threshold for v in values) / len(values)
            out[f"median_km@{budget}"] = float(np.median(values)) / 1000
        return out

    clean = [row for row in rows if not row["overlay"]]
    splits = {name: [row for row in clean if row["split"] == name] for name in ("tune", "eval", "pilot")}
    summary: dict[str, Any] = {
        "experiment": "offline-soft-vlm-prior-v1",
        "hypotheses": str(hypotheses_path),
        "overlay_flagged_excluded": sum(row["overlay"] for row in rows),
        "objective": OBJECTIVE,
        "pools": {},
    }
    for pool_name in ("merged", "merged+anchors", "osv", "osv+anchors"):
        grid = {
            f"{lc}|{lr}": metrics(splits["tune"], f"{pool_name}|{lc}|{lr}")
            for lc, lr in itertools.product(LAMBDAS, LAMBDAS)
        }
        best = max(grid, key=lambda key: grid[key][OBJECTIVE])
        summary["pools"][pool_name] = {
            "best_lambdas_country_region": best,
            "tune_grid": grid,
            "baseline": {split: metrics(subset, f"{pool_name}|0.0|0.0") for split, subset in splits.items() if split != "tune"},
            "tuned": {split: metrics(subset, f"{pool_name}|{best}") for split, subset in splits.items() if split != "tune"},
        }
    summary["elapsed_seconds"] = time.time() - started
    output.mkdir(parents=True)
    with (output / "rows.jsonl").open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hypotheses", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    summary = run(args.output, hypotheses_path=args.hypotheses, seed=args.seed)
    for pool_name, pool in summary["pools"].items():
        print(pool_name, "best", pool["best_lambdas_country_region"])
        for split in ("eval", "pilot"):
            base, tuned = pool["baseline"][split], pool["tuned"][split]
            print(f"  {split} n={base['n']}: {OBJECTIVE} {base[OBJECTIVE]:.1%} -> {tuned[OBJECTIVE]:.1%}; le_1km@50 {base['le_1km@50']:.1%} -> {tuned['le_1km@50']:.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
