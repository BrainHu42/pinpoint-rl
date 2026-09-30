# Compare location proposals from MP16-Pro and OSV-5M without exposing retrieved images.
# Usage: PYTHONPATH=src python -m geo_search_env.experiment.candidate_locations --output artifacts/candidate_locations/v1

"""Offline candidate-location recall experiment over frozen street-view corpora."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Iterable, Sequence

import numpy as np

from ..core.contracts import Coordinate
from ..core.geography import distance_m
from ..models.pinpoint import DEFAULT_PINPOINT_CHECKPOINT, PinpointRetrievalBaseline
from .feasibility import OSVTestReferenceDataset


RAW_MP16_K = 500
RAW_OSV5M_K = 1_000
MAX_LOCATION_K = 50
CLUSTER_RADIUS_M = 1_000.0
SEARCH_RADIUS_M = 5_000.0
BUDGETS = (10, 25, 50)
REFERENCE_ARTIFACT = Path("artifacts/feasibility/v6-top50-test-reference")


@dataclass(frozen=True, slots=True)
class RankedCoordinate:
    coordinate: Coordinate
    score: float
    raw_rank: int


@dataclass(frozen=True, slots=True)
class LocationProposal:
    coordinate: Coordinate
    source: str
    source_rank: int
    score: float
    support_count: int
    dispersion_m: float


def _centroid(coordinates: Sequence[Coordinate], weights: Sequence[float]) -> Coordinate:
    xyz = np.zeros(3, dtype=np.float64)
    for coordinate, weight in zip(coordinates, weights):
        latitude = math.radians(coordinate.latitude)
        longitude = math.radians(coordinate.longitude)
        xyz += max(float(weight), 1e-6) * np.array(
            [math.cos(latitude) * math.cos(longitude), math.cos(latitude) * math.sin(longitude), math.sin(latitude)]
        )
    norm = float(np.linalg.norm(xyz))
    if norm <= 1e-12:
        return coordinates[0]
    xyz /= norm
    return Coordinate(
        math.degrees(math.atan2(float(xyz[2]), math.hypot(float(xyz[0]), float(xyz[1])))),
        math.degrees(math.atan2(float(xyz[1]), float(xyz[0]))),
    )


def cluster_candidate_locations(
    rows: Sequence[RankedCoordinate],
    *,
    source: str,
    limit: int = MAX_LOCATION_K,
    radius_m: float = CLUSTER_RADIUS_M,
) -> list[LocationProposal]:
    """Collapse ranked image hits into geographically distinct location proposals."""

    clusters: list[list[RankedCoordinate]] = []
    seeds: list[Coordinate] = []
    for row in sorted(rows, key=lambda item: (item.raw_rank, -item.score)):
        cluster_index = next(
            (index for index, seed in enumerate(seeds) if distance_m(seed, row.coordinate) <= radius_m),
            None,
        )
        if cluster_index is None:
            seeds.append(row.coordinate)
            clusters.append([row])
        else:
            clusters[cluster_index].append(row)

    proposals: list[LocationProposal] = []
    for source_rank, cluster in enumerate(clusters[:limit], start=1):
        best_score = max(item.score for item in cluster)
        weights = [math.exp(max(-20.0, min(0.0, (item.score - best_score) / 0.05))) for item in cluster]
        coordinate = _centroid([item.coordinate for item in cluster], weights)
        proposals.append(
            LocationProposal(
                coordinate=coordinate,
                source=source,
                source_rank=source_rank,
                score=best_score,
                support_count=len(cluster),
                dispersion_m=max(distance_m(coordinate, item.coordinate) for item in cluster),
            )
        )
    return proposals


def merge_candidate_locations(
    mp16: Sequence[LocationProposal],
    osv5m: Sequence[LocationProposal],
    *,
    limit: int,
    radius_m: float = CLUSTER_RADIUS_M,
) -> list[LocationProposal]:
    """Round-robin sources without comparing their uncalibrated scores."""

    output: list[LocationProposal] = []
    positions = {"mp16": 0, "osv5m": 0}
    sources = {"mp16": mp16, "osv5m": osv5m}
    while len(output) < limit:
        progressed = False
        for source in ("mp16", "osv5m"):
            rows = sources[source]
            while positions[source] < len(rows):
                candidate = rows[positions[source]]
                positions[source] += 1
                progressed = True
                if all(distance_m(candidate.coordinate, prior.coordinate) > radius_m for prior in output):
                    output.append(candidate)
                    break
            if len(output) >= limit:
                break
        if not progressed:
            break
    return output


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _top_osv5m_hits(
    baseline: PinpointRetrievalBaseline,
    dataset: OSVTestReferenceDataset,
    query_projection: np.ndarray,
    excluded: set[int],
    *,
    top_k: int = RAW_OSV5M_K,
    batch_size: int = 4_096,
) -> tuple[np.ndarray, np.ndarray]:
    query_count = len(query_projection)
    best_scores = np.empty((query_count, 0), dtype=np.float32)
    best_indices = np.empty((query_count, 0), dtype=np.int64)
    for start in range(0, dataset.num_samples, batch_size):
        end = min(start + batch_size, dataset.num_samples)
        indices = np.arange(start, end, dtype=np.int64)
        if excluded:
            indices = indices[~np.isin(indices, np.fromiter(excluded, dtype=np.int64))]
        if not len(indices):
            continue
        raw = np.array(dataset.embeddings_at(indices), dtype=np.float32, copy=True)
        reference_projection = baseline.project_image_embeddings(raw, source="osv5m")
        scores = query_projection @ reference_projection.T
        local_k = min(top_k, scores.shape[1])
        local_positions = np.argpartition(scores, -local_k, axis=1)[:, -local_k:]
        local_scores = np.take_along_axis(scores, local_positions, axis=1)
        local_indices = indices[local_positions]
        combined_scores = np.concatenate((best_scores, local_scores), axis=1)
        combined_indices = np.concatenate((best_indices, local_indices), axis=1)
        keep_k = min(top_k, combined_scores.shape[1])
        keep = np.argpartition(combined_scores, -keep_k, axis=1)[:, -keep_k:]
        best_scores = np.take_along_axis(combined_scores, keep, axis=1)
        best_indices = np.take_along_axis(combined_indices, keep, axis=1)
        print(f"OSV retrieval {end}/{dataset.num_samples}", flush=True)

    for query_index in range(query_count):
        order = np.lexsort((best_indices[query_index], -best_scores[query_index]))
        best_scores[query_index] = best_scores[query_index, order]
        best_indices[query_index] = best_indices[query_index, order]
    return best_indices, best_scores


def _minimum_distance(coordinate: Coordinate, candidates: Iterable[Coordinate]) -> float:
    return min((distance_m(coordinate, candidate) for candidate in candidates), default=float("inf"))


def _location_card(proposal: LocationProposal, rank: int) -> dict[str, Any]:
    return {
        "candidate_id": f"{proposal.source}:location:{proposal.source_rank}",
        "rank": rank,
        "latitude": proposal.coordinate.latitude,
        "longitude": proposal.coordinate.longitude,
        "source": proposal.source,
        "source_rank": proposal.source_rank,
        "source_score": proposal.score,
        "support_count": proposal.support_count,
        "dispersion_m": proposal.dispersion_m,
    }


def run(output: Path, *, reference_artifact: Path = REFERENCE_ARTIFACT) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite candidate-location output: {output}")
    output.mkdir(parents=True)
    started = time.time()
    reference_artifact = reference_artifact.resolve()
    labels = _read_jsonl(reference_artifact / "private" / "labels.jsonl")
    eligibility_rows = _read_jsonl(reference_artifact / "private" / "eligibility.jsonl")
    eligibility = {
        row["episode_id"]: {int(value.removeprefix("osv:")) for value in row["eligible_1000m"]}
        for row in eligibility_rows
    }
    excluded = {
        int(row["asset_id"].removeprefix("osv:"))
        for row in _read_jsonl(reference_artifact / "exclusions.jsonl")
    }

    dataset = OSVTestReferenceDataset(Path("/data/hf/datasets/osv5m"))
    id_to_index = {image_id: index for index, image_id in enumerate(dataset._ids)}
    query_indices = np.asarray([id_to_index[row["source_image_id"]] for row in labels], dtype=np.int64)
    raw_queries = np.array(dataset.embeddings_at(query_indices), dtype=np.float32, copy=True)
    baseline = PinpointRetrievalBaseline(device="auto")
    query_projection = baseline.project_image_embeddings(raw_queries, source="mp16")

    mp16_locations: dict[str, list[LocationProposal]] = {}
    for number, (label, raw_query) in enumerate(zip(labels, raw_queries), start=1):
        hits = baseline.predict_candidates(raw_query, top_k=RAW_MP16_K)
        rows = [RankedCoordinate(hit.coordinate, hit.score, hit.rank) for hit in hits]
        mp16_locations[label["episode_id"]] = cluster_candidate_locations(rows, source="mp16")
        print(f"MP16 retrieval {number}/{len(labels)}", flush=True)

    osv_indices, osv_scores = _top_osv5m_hits(
        baseline, dataset, query_projection, excluded, top_k=RAW_OSV5M_K
    )
    osv_locations: dict[str, list[LocationProposal]] = {}
    for query_number, label in enumerate(labels):
        rows = [
            RankedCoordinate(
                Coordinate(float(dataset._coordinates[index][0]), float(dataset._coordinates[index][1])),
                float(score),
                rank,
            )
            for rank, (index, score) in enumerate(
                zip(osv_indices[query_number], osv_scores[query_number]), start=1
            )
        ]
        osv_locations[label["episode_id"]] = cluster_candidate_locations(rows, source="osv5m")

    public_rows: list[dict[str, Any]] = []
    private_metrics: list[dict[str, Any]] = []
    for label in labels:
        episode_id = label["episode_id"]
        truth = Coordinate(float(label["latitude"]), float(label["longitude"]))
        merged = merge_candidate_locations(
            mp16_locations[episode_id], osv_locations[episode_id], limit=MAX_LOCATION_K
        )
        arms = {
            "mp16": mp16_locations[episode_id],
            "osv5m": osv_locations[episode_id],
            "union": merged,
        }
        public_rows.append(
            {
                "episode_id": episode_id,
                "cohort": label["cohort"],
                "candidate_locations": {
                    source: [_location_card(proposal, rank) for rank, proposal in enumerate(rows, start=1)]
                    for source, rows in arms.items()
                },
            }
        )
        eligible_coordinates = [
            Coordinate(float(dataset._coordinates[index][0]), float(dataset._coordinates[index][1]))
            for index in eligibility[episode_id]
        ]
        metrics: dict[str, Any] = {
            "episode_id": episode_id,
            "cohort": label["cohort"],
            "evidence_available": bool(eligible_coordinates),
        }
        for source, rows in arms.items():
            for budget in BUDGETS:
                selected = rows[:budget]
                truth_error = _minimum_distance(truth, (item.coordinate for item in selected))
                evidence_error = min(
                    (
                        _minimum_distance(item.coordinate, eligible_coordinates)
                        for item in selected
                    ),
                    default=float("inf"),
                )
                prefix = f"{source}_k{budget}"
                metrics[f"{prefix}_truth_error_m"] = truth_error
                metrics[f"{prefix}_truth_1km"] = truth_error <= 1_000
                metrics[f"{prefix}_truth_5km"] = truth_error <= 5_000
                metrics[f"{prefix}_truth_25km"] = truth_error <= 25_000
                metrics[f"{prefix}_evidence_reachable_5km"] = evidence_error <= SEARCH_RADIUS_M
        private_metrics.append(metrics)

    report_metrics = [row for row in private_metrics if row["cohort"] == "report"]
    available = [row for row in report_metrics if row["evidence_available"]]
    results: dict[str, Any] = {}
    for source in ("mp16", "osv5m", "union"):
        results[source] = {}
        for budget in BUDGETS:
            prefix = f"{source}_k{budget}"
            reached = sum(row[f"{prefix}_evidence_reachable_5km"] for row in available)
            results[source][str(budget)] = {
                "truth_recall_1km": sum(row[f"{prefix}_truth_1km"] for row in report_metrics) / len(report_metrics),
                "truth_recall_5km": sum(row[f"{prefix}_truth_5km"] for row in report_metrics) / len(report_metrics),
                "truth_recall_25km": sum(row[f"{prefix}_truth_25km"] for row in report_metrics) / len(report_metrics),
                "evidence_reachable_queries": reached,
                "evidence_recall_conditional": reached / len(available) if available else 0.0,
            }

    union_recall = results["union"]["50"]["evidence_recall_conditional"]
    mp16_recall = results["mp16"]["50"]["evidence_recall_conditional"]
    gate = union_recall >= 0.30 and union_recall - mp16_recall >= 0.10
    summary = {
        "experiment": "offline-multicorpus-candidate-location-recall-v1",
        "report_queries": len(report_metrics),
        "available_report_queries": len(available),
        "cluster_radius_m": CLUSTER_RADIUS_M,
        "search_radius_m": SEARCH_RADIUS_M,
        "results": results,
        "gate": {
            "passed": gate,
            "requirement": "union top-50 conditional evidence recall >= 30% and >= 10 percentage points above MP16",
        },
        "elapsed_seconds": time.time() - started,
    }
    config = {
        "reference_artifact": str(reference_artifact),
        "query_count": len(labels),
        "report_count": len(report_metrics),
        "raw_mp16_retrieval_k": RAW_MP16_K,
        "raw_osv5m_retrieval_k": RAW_OSV5M_K,
        "candidate_location_budgets": list(BUDGETS),
        "cluster_radius_m": CLUSTER_RADIUS_M,
        "search_radius_m": SEARCH_RADIUS_M,
        "fusion": "source-alternating round robin with geographic deduplication; no cross-source score comparison",
        "candidate_contract": "locations only; underlying retrieved image IDs and pixels are not emitted",
    }
    source_manifest = {
        "checkpoint": {
            "path": str(DEFAULT_PINPOINT_CHECKPOINT),
            "sha256": _sha256(Path(DEFAULT_PINPOINT_CHECKPOINT)),
        },
        "osv_embedding_manifest": {
            "path": str(dataset.manifest_path),
            "sha256": _sha256(dataset.manifest_path),
        },
        "reference_summary": {
            "path": str(reference_artifact / "summary.json"),
            "sha256": _sha256(reference_artifact / "summary.json"),
        },
    }
    (output / "public").mkdir()
    (output / "private").mkdir()
    (output / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "source_manifest.json").write_text(json.dumps(source_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (output / "public" / "candidate_locations.jsonl").open("x", encoding="utf-8") as stream:
        for row in public_rows:
            stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    with (output / "private" / "metrics.jsonl").open("x", encoding="utf-8") as stream:
        for row in private_metrics:
            stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# Candidate-Location Recall Report",
        "",
        "This offline diagnostic exposes geographic location proposals only; retrieved corpus images are not experiment outputs.",
        "",
        f"Independent evidence is available for **{len(available)}/{len(report_metrics)}** report queries.",
        "",
        "| Source | Locations | Truth ≤1 km | Truth ≤5 km | Truth ≤25 km | Evidence reachable ≤5 km |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for source in ("mp16", "osv5m", "union"):
        for budget in BUDGETS:
            value = results[source][str(budget)]
            lines.append(
                f"| {source} | {budget} | {value['truth_recall_1km']:.1%} | {value['truth_recall_5km']:.1%} | "
                f"{value['truth_recall_25km']:.1%} | {value['evidence_reachable_queries']}/{len(available)} "
                f"({value['evidence_recall_conditional']:.1%}) |"
            )
    lines.extend(
        [
            "",
            f"**Gate: {'PASS' if gate else 'FAIL'}.** {summary['gate']['requirement']}.",
            "",
            "MP16-Pro and OSV-5M scores are not compared during fusion. Candidate locations are alternated and geographically deduplicated.",
        ]
    )
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-artifact", type=Path, default=REFERENCE_ARTIFACT)
    args = parser.parse_args(argv)
    print(json.dumps(run(args.output.resolve(), reference_artifact=args.reference_artifact), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
