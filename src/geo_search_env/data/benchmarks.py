"""im2gps3k / yfcc4k benchmark queries and metrics, ported from the Pinpoint submission evaluator."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np


DATA_ROOT = Path("/data/pinpoint")
EMBEDDING_KEY = "google_siglip2-giant-opt-patch16-384"
DISTANCE_THRESHOLDS_KM = (1, 25, 200, 750, 2500)
GEOGUESSR_DECAY_KM = 1492.7


@dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    csv_relpath: str
    image_root_relpath: str
    author_column: str


BENCHMARKS = {
    "im2gps3k": BenchmarkSpec("im2gps3k", "im2gps3k/im2gps3k_places365.csv", "im2gps3k/images", "AUTHOR"),
    "yfcc4k": BenchmarkSpec("yfcc4k", "yfcc4k/yfcc4k.csv", "yfcc4k/images", "OwnerNSID"),
    # Recent Wikimedia Commons photos (all 6,017 rows flagged usable_for_geolocation; test_balanced.csv is a 3,036-row density-balanced subset of it).
    # Photographers are Commons usernames, so no overlap with MP16's Flickr ids is expected. Not part of the default benchmark pair (BENCHMARK_NAMES).
    "wikimedia": BenchmarkSpec("wikimedia", "wikimedia/test.csv", "wikimedia/images", "artist"),
    # Commons photos taken after 2026-07-01 with device GPS, <= 3 per uploader, split by uploader (`split` column: dev / test); built by
    # experiment/commons_bench.py. `group` is the anonymised uploader. Not part of the default benchmark pair (BENCHMARK_NAMES).
    "commons26": BenchmarkSpec("commons26", "commons26/release/benchmark.csv", "commons26/release/images", "group"),
}


@dataclass(frozen=True)
class BenchmarkQueries:
    name: str
    image_ids: tuple[str, ...]
    authors: tuple[str, ...]
    latlon: np.ndarray  # [N, 2] degrees
    embeddings: np.ndarray  # [N, D] raw SigLIP2, float32
    image_paths: tuple[Path, ...]


def load_benchmark(name: str, data_root: Path = DATA_ROOT) -> BenchmarkQueries:
    """Rows present in both the CSV and the cached embeddings, in embedding-cache order (as the submission evaluates)."""

    spec = BENCHMARKS[name]
    rows: dict[str, dict[str, str]] = {}
    with (data_root / spec.csv_relpath).open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            image_id = row.get("IMG_ID", "").strip()
            try:
                lat, lon = float(row["LAT"]), float(row["LON"])
            except (KeyError, ValueError):
                continue
            if image_id and -90 <= lat <= 90 and -180 <= lon <= 180:
                rows[image_id] = row
    cache = data_root / Path(spec.image_root_relpath).parent / "image_embeddings" / EMBEDDING_KEY
    manifest = json.loads((cache / "manifest.json").read_text(encoding="utf-8"))
    ids = (cache / manifest["files"]["image_ids"]).read_text(encoding="utf-8").splitlines()
    embeddings = np.fromfile(cache / manifest["files"]["embeddings"], dtype=np.float16).reshape(len(ids), manifest["embedding_dim"])
    keep = [i for i, image_id in enumerate(ids) if image_id in rows]
    kept_ids = tuple(ids[i] for i in keep)
    return BenchmarkQueries(
        name=name,
        image_ids=kept_ids,
        authors=tuple(rows[i].get(spec.author_column, "").strip() for i in kept_ids),
        latlon=np.asarray([(float(rows[i]["LAT"]), float(rows[i]["LON"])) for i in kept_ids], dtype=np.float64),
        embeddings=embeddings[keep].astype(np.float32),
        image_paths=tuple(data_root / spec.image_root_relpath / i for i in kept_ids),
    )


def geodesic_km(predictions: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """WGS84 geodesic distance (as the submission uses), falling back to haversine without geographiclib."""

    try:
        from geographiclib.geodesic import Geodesic
    except ImportError:
        a, b = np.radians(predictions), np.radians(labels)
        h = np.sin((b[:, 0] - a[:, 0]) / 2) ** 2 + np.cos(a[:, 0]) * np.cos(b[:, 0]) * np.sin((b[:, 1] - a[:, 1]) / 2) ** 2
        return 6371.0 * 2 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))
    return np.asarray([Geodesic.WGS84.Inverse(p[0], p[1], l[0], l[1])["s12"] / 1000.0 for p, l in zip(predictions, labels)])


def compute_metrics(predictions: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    distances = geodesic_km(np.asarray(predictions, dtype=np.float64), np.asarray(labels, dtype=np.float64))
    out = {
        "Geoguessr_score": float(np.mean(np.round(5000 * np.exp(-distances / GEOGUESSR_DECAY_KM)))),
        "Median_km_error": float(np.median(distances)),
    }
    for radius in DISTANCE_THRESHOLDS_KM:
        out[f"Under_{radius}_km"] = float((distances < radius).mean())
    return out
