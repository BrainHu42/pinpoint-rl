# The reference to beat: Pinpoint's attention reranker (its released final model) on the benchmarks and MP16 dev.
# Usage: .venv/bin/python -m geo_search_env.experiment.pinpoint_reference {bench,mp16,report} [--root artifacts/pinpoint_reference]

"""Cache Pinpoint's reranked candidate pool for every benchmark photo and MP16 dev photo, and report its accuracy.

The reranker (`models.pinpoint_reranker`) retrieves <= 12 candidate photos from its own index (MP16 without bucket 99)
and predicts the top-scoring one. Index photos by the query's photographer are excluded (benchmark rule). MP16 dev =
the Pinpoint-held-out validation photos without burned-in GPS (`query_evidence select`'s pool).

Saved per set (`{bench,mp16}.npz` + `{bench,mp16}_queries.json`): candidates sorted by reranker score, best first:
rows / cache_rows (attention-index and MP16 embedding-cache rows, -1 = empty), latlon, scores, features (the 9
retrieval features), and truth. Reported: reranker top-1 and the oracle over its pool, % within 1/25/200/750/2500 km.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from ..data.benchmarks import compute_metrics, geodesic_km
from .strategy_search import BENCHMARK_NAMES, load_world

ROOT = Path("artifacts/pinpoint_reference")
SFT_ROOT = Path("artifacts/sft")
THRESHOLDS = (1, 25, 200, 750, 2500)


def _rank(world, root: Path, name: str) -> None:
    from ..models.pinpoint_reranker import PinpointReranker

    reranker = PinpointReranker(device="cuda")
    index_author = world.mp16["author"][reranker.cache_indices]
    found = reranker.rank(world.query_embeddings, query_author=world.query_author, index_author=index_author)
    root.mkdir(parents=True, exist_ok=True)
    np.savez(
        root / f"{name}.npz", rows=found.rows, cache_rows=found.cache_rows, latlon=found.latlon, scores=found.scores,
        features=found.features, truth=world.query_latlon,
    )
    (root / f"{name}_queries.json").write_text(json.dumps(world.queries) + "\n", encoding="utf-8")
    report(root)


def bench(root: Path) -> None:
    _rank(load_world(), root, "bench")


def mp16(root: Path) -> None:
    from .sft_data import OVERLAY_THRESHOLD

    queries = json.loads((SFT_ROOT / "queries.json").read_text(encoding="utf-8"))
    overlay = json.loads((SFT_ROOT / "overlay_scores.json").read_text(encoding="utf-8"))
    dev = [
        dict(q, index=i) for i, q in enumerate(queries)
        if q["group"] == "held_out" and q["split"] == "val" and overlay.get(q["image_id"]) is not None and overlay[q["image_id"]] < OVERLAY_THRESHOLD
    ]
    _rank(load_world(mp16_queries=dev), root, "mp16")


def _line(label: str, predictions: np.ndarray, truth: np.ndarray) -> str:
    m = compute_metrics(predictions, truth)
    return f"  {label:34s}" + "".join(f"{m[f'Under_{t}_km']:7.1%}" for t in THRESHOLDS) + f"   median {m['Median_km_error']:7.0f} km"


def report(root: Path) -> None:
    for name in ("bench", "mp16"):
        if not (root / f"{name}.npz").exists():
            continue
        saved = np.load(root / f"{name}.npz")
        queries = json.loads((root / f"{name}_queries.json").read_text(encoding="utf-8"))
        latlon, truth, valid = saved["latlon"], saved["truth"], saved["rows"] >= 0
        distance = np.full(valid.shape, np.inf)
        for c in range(valid.shape[1]):
            has = valid[:, c]
            distance[has, c] = geodesic_km(latlon[has, c], truth[has])
        oracle = latlon[np.arange(len(truth)), np.argmin(distance, axis=1)]
        groups = {"MP16 dev": np.arange(len(queries))} if name == "mp16" else {
            f"{b} eval": np.flatnonzero([q["benchmark"] == b and q["split"] == "eval" for q in queries]) for b in BENCHMARK_NAMES
        } | {"both eval halves": np.flatnonzero([q["split"] == "eval" for q in queries]),
             "both tune halves": np.flatnonzero([q["split"] == "tune" for q in queries])}
        print(f"{name}: % within " + " / ".join(f"{t} km" for t in THRESHOLDS) + f"; mean pool {valid.sum(1).mean():.1f} candidates")
        for label, members in groups.items():
            print(f" {label} ({len(members)} photos)")
            print(_line("Pinpoint reranker top-1", latlon[members, 0], truth[members]))
            print(_line("oracle over its pool", oracle[members], truth[members]))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("bench", "mp16", "report"))
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    {"bench": bench, "mp16": mp16, "report": report}[args.command](args.root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
