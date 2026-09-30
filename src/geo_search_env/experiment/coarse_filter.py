# Measure how much a coarse (country/region) prior improves candidate-location recall.
# Usage: PYTHONPATH=src python -m geo_search_env.experiment.coarse_filter --output artifacts/coarse_filter/v1 [--vlm-guesses PATH]

"""Offline ceiling study: retrieval restricted to an oracle or VLM-predicted country/region."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import time
from types import SimpleNamespace
from typing import Any, Sequence

import numpy as np

from ..core.contracts import Coordinate
from ..core.geography import distance_m
from ..models.pinpoint import PinpointRetrievalBaseline
from .candidate_locations import (
    RAW_MP16_K,
    RAW_OSV5M_K,
    REFERENCE_ARTIFACT,
    RankedCoordinate,
    cluster_candidate_locations,
    merge_candidate_locations,
)
from .feasibility import OSVTestReferenceDataset


OSV_ROOT = Path("/data/hf/datasets/osv5m")
RANDOM_QUERY_COUNT = 500
BUDGETS = (1, 10, 25, 50)
THRESHOLDS_M = (1_000.0, 25_000.0, 200_000.0)
GRID_LEVELS_DEG = (0.1, 0.5, 2.0)
UNKNOWN = -1


class LabelGrid:
    """Majority country/region per lat-lon cell, built from OSV-5M labels, with coarser fallbacks."""

    def __init__(self, latitudes: np.ndarray, longitudes: np.ndarray, labels: np.ndarray) -> None:
        self._levels: list[tuple[float, np.ndarray, np.ndarray]] = []
        known = labels != UNKNOWN
        latitudes, longitudes, labels = latitudes[known], longitudes[known], labels[known]
        for step in GRID_LEVELS_DEG:
            cells = self._cells(latitudes, longitudes, step)
            pairs, counts = np.unique(np.stack((cells, labels.astype(np.int64)), axis=1), axis=0, return_counts=True)
            order = np.lexsort((-counts, pairs[:, 0]))
            pairs = pairs[order]
            first = np.concatenate(([True], pairs[1:, 0] != pairs[:-1, 0]))
            self._levels.append((step, pairs[first, 0], pairs[first, 1]))

    @staticmethod
    def _cells(latitudes: np.ndarray, longitudes: np.ndarray, step: float) -> np.ndarray:
        rows = np.floor((np.clip(latitudes, -90, 89.999) + 90.0) / step).astype(np.int64)
        cols = np.floor((((longitudes + 180.0) % 360.0)) / step).astype(np.int64)
        return rows * int(round(360.0 / step)) + cols

    def lookup(self, latitudes: np.ndarray, longitudes: np.ndarray) -> np.ndarray:
        result = np.full(len(latitudes), UNKNOWN, dtype=np.int64)
        for step, keys, values in self._levels:
            pending = result == UNKNOWN
            if not pending.any():
                break
            cells = self._cells(latitudes[pending], longitudes[pending], step)
            positions = np.clip(np.searchsorted(keys, cells), 0, len(keys) - 1)
            hit = keys[positions] == cells
            found = np.full(len(cells), UNKNOWN, dtype=np.int64)
            found[hit] = values[positions[hit]]
            result[pending] = found
        return result


def _read_osv_labels(path: Path, vocab: dict[str, dict[str, int]]) -> dict[str, np.ndarray | list[str]]:
    ids, sequences, latitudes, longitudes, countries, regions = [], [], [], [], [], []
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        column = {name: index for index, name in enumerate(header)}
        for row in reader:
            ids.append(row[column["id"]])
            sequences.append(row[column["sequence"]])
            latitudes.append(float(row[column["latitude"]]))
            longitudes.append(float(row[column["longitude"]]))
            country = row[column["country"]]
            region = row[column["unique_region"]]
            countries.append(vocab["country"].setdefault(country, len(vocab["country"])) if country else UNKNOWN)
            regions.append(vocab["region"].setdefault(region, len(vocab["region"])) if region else UNKNOWN)
    return {
        "ids": ids,
        "sequences": sequences,
        "latitude": np.asarray(latitudes, dtype=np.float64),
        "longitude": np.asarray(longitudes, dtype=np.float64),
        "country": np.asarray(countries, dtype=np.int64),
        "region": np.asarray(regions, dtype=np.int64),
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _masked_topk(scores, allowed, top_k: int):
    import torch

    masked = scores.masked_fill(~allowed, float("-inf")) if allowed is not None else scores
    values, indices = torch.topk(masked.float(), min(top_k, masked.shape[0]))
    keep = torch.isfinite(values)
    return indices[keep].cpu().numpy(), values[keep].cpu().numpy()


def _locations(indices: np.ndarray, scores: np.ndarray, coordinates: np.ndarray, source: str):
    rows = [
        RankedCoordinate(Coordinate(float(np.clip(coordinates[i, 0], -90, 90)), float(((coordinates[i, 1] + 180) % 360) - 180)), float(s), rank)
        for rank, (i, s) in enumerate(zip(indices, scores), start=1)
    ]
    return cluster_candidate_locations(rows, source=source)


def load_world(seed: int = 0) -> SimpleNamespace:
    """Load labels, query set, both retrieval galleries and their coarse labels onto the GPU."""
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab: dict[str, dict[str, int]] = {"country": {}, "region": {}}

    print("reading OSV-5M labels", flush=True)
    test = _read_osv_labels(OSV_ROOT / "test.csv", vocab)
    train = _read_osv_labels(OSV_ROOT / "train.csv", vocab)
    all_lat = np.concatenate((train["latitude"], test["latitude"]))
    all_lon = np.concatenate((train["longitude"], test["longitude"]))
    country_grid = LabelGrid(all_lat, all_lon, np.concatenate((train["country"], test["country"])))
    region_grid = LabelGrid(all_lat, all_lon, np.concatenate((train["region"], test["region"])))

    dataset = OSVTestReferenceDataset(OSV_ROOT)
    assert list(dataset._ids) == test["ids"], "test.csv must align with the OSV embedding cache"
    id_to_index = {image_id: index for index, image_id in enumerate(dataset._ids)}
    sequence_rows: dict[str, list[int]] = {}
    for index, sequence in enumerate(test["sequences"]):
        sequence_rows.setdefault(sequence, []).append(index)

    reference = REFERENCE_ARTIFACT.resolve()
    pilot = _read_jsonl(reference / "private" / "labels.jsonl")
    excluded = {int(row["asset_id"].removeprefix("osv:")) for row in _read_jsonl(reference / "exclusions.jsonl")}
    queries: list[dict[str, Any]] = [
        {"query_id": row["episode_id"], "index": id_to_index[row["source_image_id"]], "set": "pilot_report"}
        for row in pilot
        if row["cohort"] == "report"
    ]
    rng = np.random.default_rng(seed)
    taken_sequences = {test["sequences"][q["index"]] for q in queries} | {row["sequence"] for row in pilot}
    for index in rng.permutation(dataset.num_samples):
        if len(queries) >= RANDOM_QUERY_COUNT + 40:
            break
        sequence = test["sequences"][int(index)]
        if sequence in taken_sequences or int(index) in excluded:
            continue
        taken_sequences.add(sequence)
        queries.append({"query_id": f"random:{int(index)}", "index": int(index), "set": "random"})


    baseline = PinpointRetrievalBaseline(device="auto")
    query_indices = np.asarray([q["index"] for q in queries], dtype=np.int64)
    raw_queries = np.array(dataset.embeddings_at(query_indices), dtype=np.float32, copy=True)
    query_projection = torch.as_tensor(np.asarray(baseline.project_image_embeddings(raw_queries, source="mp16")), dtype=torch.float16, device=device)

    print("projecting OSV reference", flush=True)
    osv_projection = []
    for start in range(0, dataset.num_samples, 8_192):
        indices = np.arange(start, min(start + 8_192, dataset.num_samples))
        raw = np.array(dataset.embeddings_at(indices), dtype=np.float32, copy=True)
        osv_projection.append(torch.as_tensor(np.asarray(baseline.project_image_embeddings(raw, source="osv5m")), dtype=torch.float16))
    osv_projection = torch.cat(osv_projection).to(device)
    osv_country = torch.as_tensor(test["country"], device=device)
    osv_region = torch.as_tensor(test["region"], device=device)
    osv_base_allowed = torch.ones(dataset.num_samples, dtype=torch.bool, device=device)
    osv_base_allowed[torch.as_tensor(sorted(excluded), device=device)] = False

    index_dir = baseline.index_dir
    manifest = json.loads((index_dir / "manifest.json").read_text(encoding="utf-8"))
    count, dim = manifest["shapes"]["gps_embeddings"]
    mp16_coordinates = np.fromfile(index_dir / manifest["files"]["latlon_deg"], dtype=np.float32).reshape(count, 2)
    mp16_embeddings = torch.from_numpy(np.fromfile(index_dir / manifest["files"]["gps_embeddings"], dtype=np.float16).reshape(count, dim)).to(device)
    print("labelling MP16 gallery", flush=True)
    mp16_country = torch.as_tensor(country_grid.lookup(mp16_coordinates[:, 0].astype(np.float64), mp16_coordinates[:, 1].astype(np.float64)), device=device)
    mp16_region = torch.as_tensor(region_grid.lookup(mp16_coordinates[:, 0].astype(np.float64), mp16_coordinates[:, 1].astype(np.float64)), device=device)
    return SimpleNamespace(
        device=device,
        test=test,
        country_grid=country_grid,
        region_grid=region_grid,
        dataset=dataset,
        sequence_rows=sequence_rows,
        excluded=excluded,
        queries=queries,
        baseline=baseline,
        query_projection=query_projection,
        osv_projection=osv_projection,
        osv_country=osv_country,
        osv_region=osv_region,
        osv_base_allowed=osv_base_allowed,
        mp16_coordinates=mp16_coordinates,
        mp16_embeddings=mp16_embeddings,
        mp16_country=mp16_country,
        mp16_region=mp16_region,
        vocab=vocab,
        train=train,
    )


def run(output: Path, *, vlm_guesses: Path | None, seed: int = 0) -> dict[str, Any]:
    import torch

    if output.exists():
        raise FileExistsError(f"refusing to overwrite coarse-filter output: {output}")
    started = time.time()
    world = load_world(seed)
    (device, test, country_grid, region_grid, dataset, sequence_rows, excluded, queries, baseline, query_projection, osv_projection, osv_country, osv_region, osv_base_allowed, mp16_coordinates, mp16_embeddings, mp16_country, mp16_region,) = (
        world.device,
        world.test,
        world.country_grid,
        world.region_grid,
        world.dataset,
        world.sequence_rows,
        world.excluded,
        world.queries,
        world.baseline,
        world.query_projection,
        world.osv_projection,
        world.osv_country,
        world.osv_region,
        world.osv_base_allowed,
        world.mp16_coordinates,
        world.mp16_embeddings,
        world.mp16_country,
        world.mp16_region,
    )

    guesses: dict[str, list[tuple[float, float]]] = {}
    overlay: set[str] = set()
    if vlm_guesses is not None:
        payload = json.loads(vlm_guesses.read_text(encoding="utf-8"))
        for row in payload["rows"]:
            guesses[row["episode_id"]] = [tuple(g) for g in row["guesses"]]
            if row.get("overlay"):
                overlay.add(row["episode_id"])

    rows: list[dict[str, Any]] = []
    for number, query in enumerate(queries):
        index = query["index"]
        truth = Coordinate(float(test["latitude"][index]), float(test["longitude"][index]))
        allowed_osv = osv_base_allowed.clone()
        allowed_osv[torch.as_tensor(sequence_rows[test["sequences"][index]], device=device)] = False

        arms: dict[str, tuple[Any, Any] | None] = {"none": None}
        truth_country, truth_region = int(test["country"][index]), int(test["region"][index])
        arms["oracle_country"] = (mp16_country == truth_country, osv_country == truth_country)
        if truth_region != UNKNOWN:
            arms["oracle_region"] = (mp16_region == truth_region, osv_region == truth_region)
        else:
            arms["oracle_region"] = arms["oracle_country"]
        vlm = guesses.get(query["query_id"])
        if vlm:
            predicted = country_grid.lookup(np.asarray([g[0] for g in vlm]), np.asarray([g[1] for g in vlm]))
            for name, countries in (("vlm_country_top1", predicted[:1]), ("vlm_country_top3", predicted[:3])):
                allowed = torch.as_tensor(sorted({int(c) for c in countries if c != UNKNOWN}) or [UNKNOWN - 1], device=device)
                arms[name] = (torch.isin(mp16_country, allowed), torch.isin(osv_country, allowed))

        mp16_scores = mp16_embeddings @ query_projection[number]
        osv_scores = osv_projection @ query_projection[number]
        result: dict[str, Any] = {
            **query,
            "truth_country_known": truth_country != UNKNOWN,
            "overlay": query["query_id"] in overlay,
        }
        for arm, masks in arms.items():
            mp16_mask, osv_mask = masks if masks is not None else (None, None)
            mp16_idx, mp16_val = _masked_topk(mp16_scores, mp16_mask, RAW_MP16_K)
            osv_idx, osv_val = _masked_topk(osv_scores, allowed_osv if osv_mask is None else allowed_osv & osv_mask, RAW_OSV5M_K)
            merged = merge_candidate_locations(
                _locations(mp16_idx, mp16_val, mp16_coordinates, "mp16"),
                _locations(osv_idx, osv_val, dataset._coordinates, "osv5m"),
                limit=max(BUDGETS),
            )
            pools = {"": [p.coordinate for p in merged]}
            if vlm:
                pools["+vlm3"] = [Coordinate(*g) for g in vlm[:3]] + [p.coordinate for p in merged]
            for suffix, pool in pools.items():
                for budget in BUDGETS:
                    selected = pool[: budget + (3 if suffix else 0)]
                    error = min((distance_m(truth, c) for c in selected), default=math.inf)
                    result[f"{arm}{suffix}@{budget}"] = error
        rows.append(result)
        if (number + 1) % 20 == 0:
            print(f"queries {number + 1}/{len(queries)}", flush=True)

    def summarize(subset: Sequence[dict[str, Any]]) -> dict[str, Any]:
        keys = sorted({key for row in subset for key in row if "@" in key})
        table: dict[str, Any] = {}
        for key in keys:
            values = [row[key] for row in subset if key in row]
            if len(values) != len(subset):
                continue
            table[key] = {
                **{f"le_{int(t / 1000)}km": sum(v <= t for v in values) / len(values) for t in THRESHOLDS_M},
                "median_km": float(np.median(values)) / 1000,
            }
        return {"n": len(subset), "arms": table}

    pilot_rows = [r for r in rows if r["set"] == "pilot_report"]
    summary = {
        "experiment": "offline-coarse-filter-ceiling-v1",
        "random": summarize([r for r in rows if r["set"] == "random"]),
        "pilot_report": summarize(pilot_rows),
        "pilot_report_no_overlay": summarize([r for r in pilot_rows if not r["overlay"]]),
        "config": {
            "raw_mp16_k": RAW_MP16_K,
            "raw_osv5m_k": RAW_OSV5M_K,
            "random_query_count": RANDOM_QUERY_COUNT,
            "seed": seed,
            "mp16_labels": "majority OSV-5M country/unique_region per grid cell, fallbacks " + ", ".join(f"{s} deg" for s in GRID_LEVELS_DEG),
            "osv_exclusions": "pilot exclusions plus each query's own sequence",
            "vlm_guesses": str(vlm_guesses) if vlm_guesses else None,
        },
        "elapsed_seconds": time.time() - started,
    }
    output.mkdir(parents=True)
    with (output / "rows.jsonl").open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--vlm-guesses", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    summary = run(args.output, vlm_guesses=args.vlm_guesses, seed=args.seed)
    print(json.dumps({k: v for k, v in summary.items() if k != "config"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
