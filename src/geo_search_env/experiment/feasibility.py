# Run the gated OSV-5M feasibility pilot and write auditable public/private artifacts.
# Usage: PYTHONPATH=src /home/brian/workspace/pinpoint-submission/submission/.venv/bin/python -m geo_search_env.experiment.feasibility --output artifacts/feasibility/v1

"""Executable OSV-5M feasibility pilot for the Mapillary-first experiment."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import statistics
import time
from typing import Any, Iterable, Sequence

import numpy as np

from ..core.contracts import Coordinate
from ..core.geography import distance_m
from ..data.osv5m import OSV5MDataset
from ..models.pinpoint import (
    DEFAULT_PINPOINT_CHECKPOINT,
    PinpointImageEmbedder,
    PinpointRetrievalBaseline,
)


SEED = 20260925
EARTH_RADIUS_M = 6_371_008.8
CALIBRATION_COUNT = 10
REPORT_COUNT = 40
COUNTRY_CAP = 5
RETRIEVAL_TOP_K = 50


@dataclass(frozen=True, slots=True)
class QueryRow:
    image_id: str
    latitude: float
    longitude: float
    country: str
    sequence: str
    captured_at: int | None
    creator_id: str | None
    city_present: bool
    region: str
    source_row: int
    cohort: str = ""

    @property
    def coordinate(self) -> Coordinate:
        return Coordinate(self.latitude, self.longitude)

    @property
    def opaque_id(self) -> str:
        return "query:" + hashlib.sha256(self.image_id.encode()).hexdigest()[:16]


@dataclass(frozen=True, slots=True)
class ReferenceMeta:
    cache_index: int
    image_id: str
    sequence: str
    captured_at: int | None
    creator_id: str | None
    country: str


class OSVTestReferenceDataset:
    """Read-only test-split reference cache used by the revised pilot."""

    def __init__(self, raw_root: Path) -> None:
        self.raw_root = raw_root
        self.embedding_root = Path(
            "/data/pinpoint/osv5m-test/image_embeddings/google_siglip2-giant-opt-patch16-384"
        )
        self.manifest_path = self.embedding_root / "manifest.json"
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self.num_samples = int(manifest["num_embeddings"])
        self.embedding_dim = int(manifest["embedding_dim"])
        self.source_split = "test"
        self._embeddings = np.memmap(
            self.embedding_root / manifest["files"]["embeddings"],
            dtype=np.float16,
            mode="r",
            shape=(self.num_samples, self.embedding_dim),
        )
        relative_paths = (self.embedding_root / manifest["files"]["image_ids"]).read_text(
            encoding="utf-8"
        ).splitlines()
        if len(relative_paths) != self.num_samples:
            raise ValueError("OSV test embedding ID count does not match its manifest")
        self._relative_paths = tuple(relative_paths)
        self._ids = tuple(Path(value).stem for value in relative_paths)
        self._relative_path_by_id = dict(zip(self._ids, self._relative_paths))
        coordinates = np.empty((self.num_samples, 2), dtype=np.float32)
        csv_ids: list[str] = []
        with (raw_root / "test.csv").open("r", encoding="utf-8", newline="") as stream:
            for index, row in enumerate(csv.DictReader(stream)):
                coordinates[index] = (float(row["latitude"]), float(row["longitude"]))
                csv_ids.append(str(row["id"]))
        if tuple(csv_ids) != self._ids:
            raise ValueError("OSV test embedding rows do not align with test.csv")
        self._coordinates = coordinates
        self._row_indices = np.arange(self.num_samples, dtype=np.int64)

    def _ensure_memmaps(self) -> None:
        return None

    def image_id_at(self, index: int) -> str:
        return self._ids[index]

    def image_path_for_id(self, image_id: str) -> Path:
        try:
            relative_path = self._relative_path_by_id[image_id]
        except KeyError as error:
            raise FileNotFoundError(f"raw OSV test image is missing for ID {image_id!r}") from error
        path = self.raw_root / "images" / "test" / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"raw OSV test image is missing for ID {image_id!r}")
        return path

    def embeddings_at(self, indices: list[int] | tuple[int, ...] | np.ndarray) -> np.ndarray:
        return np.asarray(self._embeddings[list(map(int, indices))])


def _stable_digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _optional_int(value: str | None) -> int | None:
    if value is None or not value.strip():
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


def _macro_region(latitude: float, longitude: float) -> str:
    if longitude < -30:
        return "americas"
    if -20 <= longitude <= 55 and -36 <= latitude < 37:
        return "africa"
    if -25 <= longitude <= 60 and latitude >= 37:
        return "europe"
    if latitude < 0 and (longitude >= 100 or longitude <= -130):
        return "oceania"
    return "asia"


def _read_query_population(path: Path) -> tuple[list[QueryRow], dict[str, int]]:
    rows: list[QueryRow] = []
    country_counts: dict[str, int] = {}
    with path.open("r", encoding="utf-8", newline="") as stream:
        for source_row, raw in enumerate(csv.DictReader(stream)):
            try:
                image_id = str(raw["id"]).strip()
                latitude = float(raw["latitude"])
                longitude = float(raw["longitude"])
                Coordinate(latitude, longitude)
            except (KeyError, TypeError, ValueError):
                continue
            country = str(raw.get("country") or "unknown").strip() or "unknown"
            sequence = str(raw.get("sequence") or "").strip()
            if not image_id or not sequence:
                continue
            country_counts[country] = country_counts.get(country, 0) + 1
            rows.append(
                QueryRow(
                    image_id=image_id,
                    latitude=latitude,
                    longitude=longitude,
                    country=country,
                    sequence=sequence,
                    captured_at=_optional_int(raw.get("captured_at")),
                    creator_id=(str(raw.get("creator_id") or "").strip() or None),
                    city_present=bool(str(raw.get("city") or "").strip()),
                    region=_macro_region(latitude, longitude),
                    source_row=source_row,
                )
            )
    return rows, country_counts


def _select_cohort(population: Sequence[QueryRow]) -> tuple[QueryRow, ...]:
    ordered = sorted(population, key=lambda row: _stable_digest(f"{SEED}:{row.image_id}"))
    regions = sorted({row.region for row in ordered})
    selected: list[QueryRow] = []
    used_sequences: set[str] = set()
    country_counts: dict[str, int] = {}

    def add(row: QueryRow) -> bool:
        if row.sequence in used_sequences or country_counts.get(row.country, 0) >= COUNTRY_CAP:
            return False
        selected.append(row)
        used_sequences.add(row.sequence)
        country_counts[row.country] = country_counts.get(row.country, 0) + 1
        return True

    for _ in range(2):
        for region in regions:
            for row in ordered:
                if row.region == region and add(row):
                    break
    for row in ordered:
        if len(selected) >= CALIBRATION_COUNT + REPORT_COUNT:
            break
        add(row)
    if len(selected) != CALIBRATION_COUNT + REPORT_COUNT:
        raise RuntimeError("could not construct the requested deterministic cohort")
    return tuple(
        QueryRow(**{**asdict(row), "cohort": "calibration" if index < CALIBRATION_COUNT else "report"})
        for index, row in enumerate(selected)
    )


def _resolve_test_image(root: Path, image_id: str) -> Path | None:
    for shard in sorted((root / "images" / "test").iterdir()):
        for suffix in (".jpg", ".jpeg", ".png", ".webp"):
            candidate = shard / f"{image_id}{suffix}"
            if candidate.is_file():
                return candidate
    return None


def _latlon_to_xyz(values: np.ndarray) -> np.ndarray:
    radians = np.radians(values.astype(np.float64, copy=False))
    latitudes, longitudes = radians[:, 0], radians[:, 1]
    cosine = np.cos(latitudes)
    return np.column_stack((cosine * np.cos(longitudes), cosine * np.sin(longitudes), np.sin(latitudes)))


def _coordinate_xyz(coordinate: Coordinate) -> np.ndarray:
    return _latlon_to_xyz(np.array([[coordinate.latitude, coordinate.longitude]], dtype=np.float64))[0]


def _chord(radius_m: float) -> float:
    return 2.0 * math.sin(radius_m / (2.0 * EARTH_RADIUS_M))


def _haversine_many(coordinate: Coordinate, rows: np.ndarray) -> np.ndarray:
    lat1 = math.radians(coordinate.latitude)
    lon1 = math.radians(coordinate.longitude)
    lat2 = np.radians(rows[:, 0].astype(np.float64, copy=False))
    lon2 = np.radians(rows[:, 1].astype(np.float64, copy=False))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    value = np.sin(dlat / 2) ** 2 + math.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_M * np.arcsin(np.minimum(1.0, np.sqrt(value)))


class SpatialWorld:
    def __init__(self, dataset: OSV5MDataset) -> None:
        from scipy.spatial import cKDTree

        dataset._ensure_memmaps()
        assert dataset._coordinates is not None
        self.dataset = dataset
        self.coordinates = dataset._coordinates
        self.xyz = _latlon_to_xyz(self.coordinates)
        self.tree = cKDTree(self.xyz, compact_nodes=True, balanced_tree=True)

    def within(self, coordinate: Coordinate, radius_m: float) -> list[int]:
        raw = self.tree.query_ball_point(_coordinate_xyz(coordinate), _chord(radius_m))
        if not raw:
            return []
        indices = np.asarray(raw, dtype=np.int64)
        distances = _haversine_many(coordinate, self.coordinates[indices])
        return indices[distances <= radius_m].astype(int).tolist()

    def nearest(self, coordinate: Coordinate, radius_m: float, limit: int) -> list[int]:
        distance, index = self.tree.query(
            _coordinate_xyz(coordinate),
            k=limit,
            distance_upper_bound=_chord(radius_m),
        )
        indices = np.atleast_1d(index)
        valid = indices < len(self.coordinates)
        indices = indices[valid].astype(np.int64)
        if not len(indices):
            return []
        distances = _haversine_many(coordinate, self.coordinates[indices])
        order = np.lexsort((indices, distances))
        return indices[order][distances[order] <= radius_m].astype(int).tolist()


def _source_to_cache(dataset: Any) -> np.ndarray:
    dataset._ensure_memmaps()
    assert dataset._row_indices is not None
    rows = np.asarray(dataset._row_indices)
    if len(rows) != dataset.num_samples or int(rows.min()) < 0 or int(rows.max()) >= dataset.num_samples:
        raise ValueError("OSV source row index is not a permutation of the train CSV")
    inverse = np.empty(dataset.num_samples, dtype=np.int64)
    inverse[rows] = np.arange(dataset.num_samples, dtype=np.int64)
    if len(np.unique(inverse[: min(100_000, len(inverse))])) != min(100_000, len(inverse)):
        raise ValueError("OSV source row index contains duplicates")
    return inverse


def _scan_reference_metadata(
    csv_path: Path,
    dataset: Any,
    inverse_rows: np.ndarray,
    wanted_cache_indices: set[int],
    query_sequences: set[str],
) -> tuple[dict[int, ReferenceMeta], set[int], int]:
    wanted_sources = {
        int(dataset._row_indices[index]): index  # type: ignore[index]
        for index in wanted_cache_indices
    }
    metadata: dict[int, ReferenceMeta] = {}
    excluded_sequences: set[int] = set()
    global_sequence_exclusion_count = 0
    with csv_path.open("r", encoding="utf-8", newline="") as stream:
        for source_row, raw in enumerate(csv.DictReader(stream)):
            sequence = str(raw.get("sequence") or "").strip()
            if sequence in query_sequences:
                cache_index = int(inverse_rows[source_row])
                excluded_sequences.add(cache_index)
                global_sequence_exclusion_count += 1
            cache_index = wanted_sources.get(source_row)
            if cache_index is not None:
                metadata[cache_index] = ReferenceMeta(
                    cache_index=cache_index,
                    image_id=str(raw.get("id") or dataset.image_id_at(cache_index)).strip(),
                    sequence=sequence,
                    captured_at=_optional_int(raw.get("captured_at")),
                    creator_id=(str(raw.get("creator_id") or "").strip() or None),
                    country=(str(raw.get("country") or "unknown").strip() or "unknown"),
                )
    missing = wanted_cache_indices - metadata.keys()
    if missing:
        raise ValueError(f"missing reference metadata for {len(missing)} candidate rows")
    return metadata, excluded_sequences, global_sequence_exclusion_count


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _perceptual_hash(path: Path) -> int:
    from PIL import Image
    from scipy.fft import dctn

    with Image.open(path) as image:
        gray = image.convert("L").resize((32, 32))
        pixels = np.asarray(gray, dtype=np.float32)
    values = dctn(pixels, type=2, norm="ortho")[:8, :8].reshape(-1)
    median = float(np.median(values[1:]))
    bits = values > median
    result = 0
    for bit in bits:
        result = (result << 1) | int(bit)
    return result


def _asset_path(dataset: Any, metadata: ReferenceMeta) -> Path | None:
    try:
        return dataset.image_path_for_id(metadata.image_id)
    except FileNotFoundError:
        return None


def _stable_subset(indices: Iterable[int], count: int, salt: str) -> list[int]:
    return sorted(indices, key=lambda index: _stable_digest(f"{salt}:{index}"))[:count]


def _round_robin(groups: Sequence[Sequence[int]], limit: int) -> list[int]:
    output: list[int] = []
    seen: set[int] = set()
    depth = 0
    while len(output) < limit:
        progressed = False
        for group in groups:
            if depth < len(group):
                progressed = True
                value = int(group[depth])
                if value not in seen:
                    seen.add(value)
                    output.append(value)
                    if len(output) >= limit:
                        break
        if not progressed:
            break
        depth += 1
    return output


def _best_threshold(scores: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    positives = int(labels.sum())
    if positives == 0:
        return {"threshold": 1.0, "precision": 0.0, "recall": 0.0}
    best = (0.0, 0.0, 1.0)
    for threshold in sorted(set(map(float, scores)), reverse=True):
        selected = scores >= threshold
        count = int(selected.sum())
        true_positive = int(labels[selected].sum())
        precision = true_positive / count if count else 0.0
        recall = true_positive / positives
        if precision >= 0.8 and (recall, precision, threshold) > best:
            best = (recall, precision, threshold)
    return {"threshold": best[2], "precision": best[1], "recall": best[0]}


def _average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    positives = int(labels.sum())
    if positives == 0:
        return 0.0
    order = np.argsort(-scores, kind="stable")
    ranked = labels[order].astype(np.int64)
    precision = np.cumsum(ranked) / np.arange(1, len(ranked) + 1)
    return float((precision * ranked).sum() / positives)


def _json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _jsonl_dump(path: Path, values: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        for value in values:
            stream.write(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")


def _percent(value: float) -> str:
    return f"{100 * value:.1f}%"


def run(output: Path, *, reference_source: str = "train") -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite feasibility output: {output}")
    output.mkdir(parents=True)
    started = time.time()
    raw_root = Path("/data/hf/datasets/osv5m")
    if reference_source == "train":
        dataset: Any = OSV5MDataset()
        reference_csv = raw_root / "train.csv"
    elif reference_source == "test":
        dataset = OSVTestReferenceDataset(raw_root)
        reference_csv = raw_root / "test.csv"
    else:
        raise ValueError("reference_source must be train or test")

    population, population_countries = _read_query_population(raw_root / "test.csv")
    cohort = _select_cohort(population)
    query_paths = {row.opaque_id: _resolve_test_image(raw_root, row.image_id) for row in cohort}

    embedder = PinpointImageEmbedder(device="cuda", inference_dtype="float16")
    baseline = PinpointRetrievalBaseline(device="cuda", inference_dtype="float16")
    raw_query_embeddings: dict[str, np.ndarray] = {}
    retrieval: dict[str, list[dict[str, Any]]] = {}
    query_failures: dict[str, str] = {}
    for number, row in enumerate(cohort, start=1):
        path = query_paths[row.opaque_id]
        if path is None:
            query_failures[row.opaque_id] = "missing_query_image"
            continue
        try:
            raw = embedder.embed_path(path).numpy()
            raw_query_embeddings[row.opaque_id] = raw
            retrieval[row.opaque_id] = [
                {
                    "rank": candidate.rank,
                    "latitude": candidate.coordinate.latitude,
                    "longitude": candidate.coordinate.longitude,
                    "score": candidate.score,
                    "retrieval_index": candidate.retrieval_index,
                }
                for candidate in baseline.predict_candidates(raw, top_k=RETRIEVAL_TOP_K)
            ]
        except Exception as error:
            query_failures[row.opaque_id] = f"query_inference_failure:{type(error).__name__}"
        print(f"retrieval {number}/{len(cohort)}", flush=True)

    world = SpatialWorld(dataset)
    inverse_rows = _source_to_cache(dataset)
    truth_neighbors: dict[str, dict[int, list[int]]] = {}
    search_groups: dict[str, list[list[int]]] = {}
    diagnostic_pools: dict[str, list[int]] = {}
    wanted: set[int] = set()
    random_source = random.Random(SEED)
    for row in cohort:
        query_id = row.opaque_id
        if query_id in query_failures:
            truth_neighbors[query_id] = {25: [], 100: [], 1000: []}
            search_groups[query_id] = []
            diagnostic_pools[query_id] = []
            continue
        truth_neighbors[query_id] = {
            radius: world.within(row.coordinate, float(radius))
            for radius in (25, 100, 1000)
        }
        groups: list[list[int]] = []
        for candidate in retrieval[query_id]:
            anchor = Coordinate(candidate["latitude"], candidate["longitude"])
            for radius in (250, 1000, 5000):
                groups.append(world.nearest(anchor, float(radius), 32))
        search_groups[query_id] = groups

        eligible_sample = _stable_subset(truth_neighbors[query_id][1000], 50, query_id + ":eligible")
        hard_raw = world.nearest(row.coordinate, 25_000.0, 600)
        hard = [
            index for index in hard_raw
            if distance_m(
                row.coordinate,
                Coordinate(float(world.coordinates[index][0]), float(world.coordinates[index][1])),
            ) > 1_000
        ]
        hard_sample = _stable_subset(hard, 50, query_id + ":hard")
        global_sample: list[int] = []
        while len(global_sample) < 150:
            index = random_source.randrange(dataset.num_samples)
            coordinate = Coordinate(float(world.coordinates[index][0]), float(world.coordinates[index][1]))
            if distance_m(row.coordinate, coordinate) > 25_000 and index not in global_sample:
                global_sample.append(index)
        pool = list(dict.fromkeys(eligible_sample + hard_sample + global_sample))
        diagnostic_pools[query_id] = pool
        wanted.update(truth_neighbors[query_id][1000])
        wanted.update(pool)
        for group in groups:
            wanted.update(group)

    metadata, sequence_exclusions, sequence_exclusion_count = _scan_reference_metadata(
        reference_csv,
        dataset,
        inverse_rows,
        wanted,
        {row.sequence for row in cohort},
    )

    available: set[int] = set()
    reference_paths: dict[int, Path] = {}
    for number, index in enumerate(sorted(wanted), start=1):
        path = _asset_path(dataset, metadata[index])
        if path is not None:
            available.add(index)
            reference_paths[index] = path
        if number % 1000 == 0:
            print(f"asset audit {number}/{len(wanted)}", flush=True)

    duplicate_exclusions: set[int] = set()
    exact_duplicate_count = near_duplicate_count = same_capture_count = 0
    query_hashes: dict[str, tuple[str, int]] = {}
    for row in cohort:
        path = query_paths[row.opaque_id]
        if path is None:
            continue
        try:
            query_hashes[row.opaque_id] = (_sha256_file(path), _perceptual_hash(path))
        except Exception:
            query_failures[row.opaque_id] = "corrupt_query_image"
            continue
        for index in truth_neighbors[row.opaque_id][100]:
            path_ref = reference_paths.get(index)
            if path_ref is None:
                continue
            meta = metadata[index]
            same_capture = (
                row.creator_id is not None
                and meta.creator_id == row.creator_id
                and row.captured_at is not None
                and meta.captured_at is not None
                and abs(row.captured_at - meta.captured_at) <= 2_000
                and distance_m(
                    row.coordinate,
                    Coordinate(float(world.coordinates[index][0]), float(world.coordinates[index][1])),
                ) <= 10
            )
            if same_capture:
                duplicate_exclusions.add(index)
                same_capture_count += 1
                continue
            try:
                ref_sha = _sha256_file(path_ref)
                if ref_sha == query_hashes[row.opaque_id][0]:
                    duplicate_exclusions.add(index)
                    exact_duplicate_count += 1
                    continue
                ref_phash = _perceptual_hash(path_ref)
                if (ref_phash ^ query_hashes[row.opaque_id][1]).bit_count() <= 6:
                    duplicate_exclusions.add(index)
                    near_duplicate_count += 1
            except Exception:
                available.discard(index)

    excluded = sequence_exclusions | duplicate_exclusions

    eligible_by_query: dict[str, set[int]] = {}
    fixed_opened: dict[str, list[int]] = {}
    relaxed_opened: dict[str, list[int]] = {}
    for row in cohort:
        query_id = row.opaque_id
        eligible_by_query[query_id] = {
            index for index in truth_neighbors[query_id][1000]
            if index in available and index not in excluded
        }
        diagnostic_negatives = [
            index
            for index in diagnostic_pools[query_id]
            if index not in truth_neighbors[query_id][1000] and index not in excluded
        ]
        diagnostic_pools[query_id] = list(
            dict.fromkeys(
                _stable_subset(eligible_by_query[query_id], 50, query_id + ":eligible-audited")
                + diagnostic_negatives
            )
        )
        filtered_groups = [
            [index for index in group if index not in excluded]
            for group in search_groups[query_id]
        ]
        first_page_top_five = [group[:8] for group in filtered_groups[::3][:5]]
        fixed_cards = [index for group in first_page_top_five for index in group]
        fixed_opened[query_id] = fixed_cards[:6]
        relaxed_group_priority = (
            filtered_groups[2::3] + filtered_groups[1::3] + filtered_groups[0::3]
        )
        relaxed_opened[query_id] = _round_robin(
            [group[:24] for group in relaxed_group_priority], 64
        )

    projection_indices = sorted(
        set().union(*diagnostic_pools.values(), *fixed_opened.values(), *relaxed_opened.values())
        & available
    )
    reference_projection: dict[int, np.ndarray] = {}
    for start in range(0, len(projection_indices), 512):
        indices = projection_indices[start : start + 512]
        raw = np.array(dataset.embeddings_at(indices), dtype=np.float32, copy=True)
        projected = baseline.project_image_embeddings(raw, source="osv5m")
        reference_projection.update(zip(indices, projected))
        print(f"reference projection {min(start + 512, len(projection_indices))}/{len(projection_indices)}", flush=True)

    query_projection = {
        query_id: baseline.project_image_embeddings(np.array(raw, copy=True), source="mp16")[0]
        for query_id, raw in raw_query_embeddings.items()
    }

    def score(query_id: str, index: int) -> float | None:
        if query_id not in query_projection or index not in reference_projection:
            return None
        cosine = float(np.dot(query_projection[query_id], reference_projection[index]))
        return (max(-1.0, min(1.0, cosine)) + 1.0) / 2.0

    calibration_scores: list[float] = []
    calibration_labels: list[bool] = []
    for row in cohort[:CALIBRATION_COUNT]:
        for index in diagnostic_pools[row.opaque_id]:
            value = score(row.opaque_id, index)
            if value is not None and index not in excluded:
                calibration_scores.append(value)
                calibration_labels.append(index in eligible_by_query[row.opaque_id])
    threshold_result = _best_threshold(
        np.asarray(calibration_scores, dtype=np.float64),
        np.asarray(calibration_labels, dtype=bool),
    )
    threshold = threshold_result["threshold"]

    report_scores: list[float] = []
    report_labels: list[bool] = []
    report_metrics: list[dict[str, Any]] = []
    for row in cohort[CALIBRATION_COUNT:]:
        query_id = row.opaque_id
        baseline_candidates = retrieval.get(query_id, [])
        baseline_error = (
            distance_m(row.coordinate, Coordinate(baseline_candidates[0]["latitude"], baseline_candidates[0]["longitude"]))
            if baseline_candidates else float("inf")
        )
        pool_pairs = [
            (index, value)
            for index in diagnostic_pools[query_id]
            if index not in excluded and (value := score(query_id, index)) is not None
        ]
        for index, value in pool_pairs:
            report_scores.append(value)
            report_labels.append(index in eligible_by_query[query_id])
        diagnostic_top_eligible = bool(pool_pairs) and max(pool_pairs, key=lambda pair: (pair[1], -pair[0]))[0] in eligible_by_query[query_id]

        def evaluate(opened: list[int]) -> tuple[int | None, float, bool]:
            scored = [
                (value, -index, index)
                for index in opened
                if index not in excluded and (value := score(query_id, index)) is not None
            ]
            if not scored or max(scored)[0] < threshold or not baseline_candidates:
                return None, baseline_error, False
            chosen = max(scored)[2]
            coordinate = Coordinate(float(world.coordinates[chosen][0]), float(world.coordinates[chosen][1]))
            return chosen, distance_m(row.coordinate, coordinate), True

        fixed_choice, fixed_error, fixed_refined = evaluate(fixed_opened[query_id])
        relaxed_choice, relaxed_error, relaxed_refined = evaluate(relaxed_opened[query_id])
        eligible = eligible_by_query[query_id]
        fixed_acquired = bool(eligible.intersection(fixed_opened[query_id]))
        relaxed_acquired = bool(eligible.intersection(relaxed_opened[query_id]))
        acquired_scores = [
            (index, value)
            for index in relaxed_opened[query_id]
            if (value := score(query_id, index)) is not None
        ]
        acquired_top_eligible = bool(acquired_scores) and max(acquired_scores, key=lambda pair: (pair[1], -pair[0]))[0] in eligible
        report_metrics.append(
            {
                "episode_id": query_id,
                "region": row.region,
                "country": row.country,
                "city_present": row.city_present,
                "query_failure": query_failures.get(query_id),
                "available_25m": len(set(truth_neighbors[query_id][25]) & eligible),
                "available_100m": len(set(truth_neighbors[query_id][100]) & eligible),
                "available_1000m": len(eligible),
                "baseline_error_m": baseline_error,
                "baseline_success_1000m": baseline_error <= 1_000,
                "top50_oracle_error_m": min(
                    [distance_m(row.coordinate, Coordinate(item["latitude"], item["longitude"])) for item in baseline_candidates]
                    or [float("inf")]
                ),
                "diagnostic_top_eligible": diagnostic_top_eligible,
                "fixed_opened": len(fixed_opened[query_id]),
                "fixed_acquired": fixed_acquired,
                "fixed_choice": None if fixed_choice is None else f"osv:{fixed_choice}",
                "fixed_refined": fixed_refined,
                "fixed_error_m": fixed_error,
                "fixed_success_1000m": fixed_error <= 1_000,
                "fixed_harmful": baseline_error <= 1_000 < fixed_error,
                "relaxed_opened": len(relaxed_opened[query_id]),
                "relaxed_acquired": relaxed_acquired,
                "relaxed_top_eligible": acquired_top_eligible,
                "relaxed_choice": None if relaxed_choice is None else f"osv:{relaxed_choice}",
                "relaxed_refined": relaxed_refined,
                "relaxed_error_m": relaxed_error,
                "relaxed_success_1000m": relaxed_error <= 1_000,
                "rescue_opportunity": baseline_error > 1_000 and relaxed_acquired and relaxed_error <= 1_000,
            }
        )

    scores_array = np.asarray(report_scores, dtype=np.float64)
    labels_array = np.asarray(report_labels, dtype=bool)
    selected = scores_array >= threshold
    true_positive = int(labels_array[selected].sum()) if len(labels_array) else 0
    matcher_precision = true_positive / int(selected.sum()) if int(selected.sum()) else 0.0
    matcher_recall = true_positive / int(labels_array.sum()) if int(labels_array.sum()) else 0.0
    available_queries = sum(item["available_1000m"] > 0 for item in report_metrics)
    acquired_queries = sum(item["relaxed_acquired"] for item in report_metrics)
    acquired_cases = [item for item in report_metrics if item["relaxed_acquired"]]
    matcher_top_rate = (
        sum(item["relaxed_top_eligible"] for item in acquired_cases) / len(acquired_cases)
        if acquired_cases else 0.0
    )
    baseline_accuracy = sum(item["baseline_success_1000m"] for item in report_metrics) / REPORT_COUNT
    fixed_accuracy = sum(item["fixed_success_1000m"] for item in report_metrics) / REPORT_COUNT
    harmful_rate = sum(item["fixed_harmful"] for item in report_metrics) / REPORT_COUNT
    rescue_count = sum(item["rescue_opportunity"] for item in report_metrics)

    audit_limitations = [
        f"Exact and perceptual duplicate checks cover references within 100 m of pilot queries, not all {dataset.num_samples:,} {reference_source} reference images.",
        "Checkpoint metadata proves OSV training but does not by itself prove that OSV-5M test images were excluded.",
        "The pilot uses an in-memory cKDTree over a disk-backed coordinate cache rather than the planned persistent SQLite catalog.",
        "Country-frequency weighted sensitivity estimates are not implemented in this diagnostic run.",
    ]
    gates = {
        "audit_integrity": False,
        "independent_availability": available_queries >= 12,
        "relaxed_acquisition": available_queries > 0 and acquired_queries / available_queries >= 0.5,
        "matcher_ranking": matcher_top_rate >= 0.5,
        "matcher_safety": matcher_precision >= 0.8 and matcher_recall >= 0.3,
        "consequentiality": rescue_count >= 5,
        "fixed_policy_safety": harmful_rate <= 0.05 and fixed_accuracy - baseline_accuracy >= -0.025,
    }
    scientific_gates = [
        "independent_availability",
        "relaxed_acquisition",
        "matcher_ranking",
        "matcher_safety",
        "consequentiality",
    ]
    if gates["audit_integrity"] and all(gates[name] for name in scientific_gates):
        outcome = "proceed"
    elif available_queries < 12:
        outcome = "stop_or_narrow"
    else:
        outcome = "revise_and_repeat"

    summary = {
        "experiment": f"osv5m-feasibility-{reference_source}-reference-diagnostic",
        "outcome": outcome,
        "report_queries": REPORT_COUNT,
        "query_failures": sum(item["query_failure"] is not None for item in report_metrics),
        "available_queries_1000m": available_queries,
        "availability_rate_1000m": available_queries / REPORT_COUNT,
        "relaxed_acquired_queries": acquired_queries,
        "relaxed_acquisition_rate_conditional": acquired_queries / available_queries if available_queries else 0.0,
        "matcher_calibration": threshold_result,
        "matcher_average_precision": _average_precision(scores_array, labels_array),
        "matcher_precision_at_threshold": matcher_precision,
        "matcher_recall_at_threshold": matcher_recall,
        "matcher_top1_rate_conditional_on_acquisition": matcher_top_rate,
        "baseline_accuracy_1000m": baseline_accuracy,
        "fixed_accuracy_1000m": fixed_accuracy,
        "fixed_accuracy_delta_1000m": fixed_accuracy - baseline_accuracy,
        "fixed_harmful_refinement_rate": harmful_rate,
        "rescue_opportunity_count": rescue_count,
        "sequence_exclusion_count": sequence_exclusion_count,
        "exact_duplicate_count_checked_nearby": exact_duplicate_count,
        "near_duplicate_count_checked_nearby": near_duplicate_count,
        "same_capture_count_checked_nearby": same_capture_count,
        "gates": gates,
        "audit_limitations": audit_limitations,
        "elapsed_seconds": time.time() - started,
    }

    configuration = {
        "seed": SEED,
        "calibration_queries": CALIBRATION_COUNT,
        "report_queries": REPORT_COUNT,
        "country_cap": COUNTRY_CAP,
        "reference_count": dataset.num_samples,
        "reference_source": reference_source,
        "embedding_dimension": dataset.embedding_dim,
        "matcher": {
            "query_source": "mp16",
            "reference_source": "osv5m",
            "output_dimension": 256,
            "score": "(cosine+1)/2",
        },
        "fixed_budget": {"credits": 32, "actions": 12, "anchors": 5, "radius_m": 1000, "opens": 6},
        "relaxed_budget": {"credits": 2048, "actions": 600, "anchors": RETRIEVAL_TOP_K, "radii_m": [250, 1000, 5000], "pages_per_radius": 3, "opens": 64},
    }
    source_files = {
        "reference_csv": reference_csv,
        "test_csv": raw_root / "test.csv",
        "dataset_readme": raw_root / "README.md",
        "embedding_manifest": dataset.manifest_path,
        "checkpoint": Path(DEFAULT_PINPOINT_CHECKPOINT),
        "retrieval_manifest": baseline.index_dir / "manifest.json",
    }
    source_manifest = {
        name: {"path": str(path), "size": path.stat().st_size, "sha256": _sha256_file(path)}
        for name, path in source_files.items()
    }
    source_manifest["runtime"] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pinpoint_version": baseline.version,
    }

    public_episodes = []
    public_candidates = []
    private_labels = []
    private_eligibility = []
    for row in cohort:
        candidates = retrieval.get(row.opaque_id, [])
        public_episodes.append(
            {
                "episode_id": row.opaque_id,
                "query_asset_id": row.opaque_id,
                "cohort": row.cohort,
                "baseline_candidate_id": "retrieval:1" if candidates else None,
                "candidate_count": len(candidates),
            }
        )
        public_candidates.append(
            {
                "episode_id": row.opaque_id,
                "candidates": [
                    {key: item[key] for key in ("rank", "latitude", "longitude", "score")}
                    for item in candidates
                ],
            }
        )
        private_labels.append(
            {
                "episode_id": row.opaque_id,
                "source_image_id": row.image_id,
                "latitude": row.latitude,
                "longitude": row.longitude,
                "sequence": row.sequence,
                "country": row.country,
                "region": row.region,
                "cohort": row.cohort,
            }
        )
        private_eligibility.append(
            {
                "episode_id": row.opaque_id,
                "eligible_1000m": sorted(f"osv:{index}" for index in eligible_by_query[row.opaque_id]),
            }
        )

    _json_dump(output / "config.json", configuration)
    _json_dump(output / "source_manifest.json", source_manifest)
    _json_dump(
        output / "catalog_manifest.json",
        {
            "reference_count": dataset.num_samples,
            "candidate_assets_audited": len(wanted),
            "candidate_assets_available": len(available),
            "spatial_index": "scipy-cKDTree-v1",
            "license": "CC-BY-SA-4.0",
        },
    )
    _json_dump(
        output / "cohort_manifest.json",
        {
            "selection": "sha256(seed:image_id), unique sequence, country cap, macro-region calibration coverage",
            "calibration_count": CALIBRATION_COUNT,
            "report_count": REPORT_COUNT,
            "population_country_counts": population_countries,
            "cohort_region_counts": {
                region: sum(row.region == region for row in cohort)
                for region in sorted({row.region for row in cohort})
            },
        },
    )
    _jsonl_dump(output / "exclusions.jsonl", [
        {"asset_id": f"osv:{index}", "reason": "pilot_query_sequence"}
        for index in sorted(sequence_exclusions)
    ] + [
        {"asset_id": f"osv:{index}", "reason": "duplicate_or_same_capture"}
        for index in sorted(duplicate_exclusions)
    ])
    _jsonl_dump(output / "public" / "episodes.jsonl", public_episodes)
    _jsonl_dump(output / "public" / "retrieval_candidates.jsonl", public_candidates)
    for policy in ("pinpoint_only", "fixed", "relaxed"):
        _jsonl_dump(
            output / "public" / "traces" / f"{policy}.jsonl",
            [
                {
                    "episode_id": item["episode_id"],
                    "policy": policy,
                    "opened_count": 0 if policy == "pinpoint_only" else item[f"{policy}_opened"] if policy != "pinpoint_only" else 0,
                    "used_fallback": policy == "pinpoint_only" or not item.get(f"{policy}_refined", False),
                }
                for item in report_metrics
            ],
        )
    _jsonl_dump(output / "private" / "labels.jsonl", private_labels)
    _jsonl_dump(output / "private" / "eligibility.jsonl", private_eligibility)
    _json_dump(
        output / "private" / "matcher_diagnostics.json",
        {
            "calibration": threshold_result,
            "report_pair_count": len(report_scores),
            "report_positive_pair_count": int(labels_array.sum()),
            "report_average_precision": summary["matcher_average_precision"],
        },
    )
    _jsonl_dump(output / "metrics.jsonl", report_metrics)
    _json_dump(output / "summary.json", summary)

    report = f"""# OSV-5M Feasibility Report\n\n## Decision\n\n**{outcome.replace('_', ' ').title()}**\n\nThis is a diagnostic pilot. Audit limitations prevent treating it as a final held-out result.\n\n## Primary results\n\n- Independent 1 km availability: **{available_queries}/{REPORT_COUNT} ({_percent(available_queries / REPORT_COUNT)})**\n- Relaxed acquisition conditional on availability: **{acquired_queries}/{available_queries} ({_percent(acquired_queries / available_queries if available_queries else 0)})**\n- Matcher top-1 conditional on acquisition: **{_percent(matcher_top_rate)}**\n- Matcher precision / recall at frozen threshold `{threshold:.6f}`: **{_percent(matcher_precision)} / {_percent(matcher_recall)}**\n- Pinpoint baseline 1 km accuracy: **{_percent(baseline_accuracy)}**\n- Fixed-search 1 km accuracy: **{_percent(fixed_accuracy)}** ({100 * (fixed_accuracy - baseline_accuracy):+.1f} pp)\n- Fixed-search harmful refinement: **{_percent(harmful_rate)}**\n- End-to-end rescue opportunities: **{rescue_count}/{REPORT_COUNT}**\n\n## Gates\n\n"""
    report += "\n".join(f"- {'PASS' if passed else 'FAIL'} — `{name}`" for name, passed in gates.items())
    report += "\n\n## Audit limitations\n\n" + "\n".join(f"- {item}" for item in audit_limitations) + "\n"
    (output / "report.md").write_text(report, encoding="utf-8")
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-source", choices=("train", "test"), default="train")
    args = parser.parse_args(argv)
    summary = run(args.output.resolve(), reference_source=args.reference_source)
    print(json.dumps(summary, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
