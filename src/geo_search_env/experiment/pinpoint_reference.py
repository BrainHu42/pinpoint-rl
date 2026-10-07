# The reference to beat: Pinpoint's attention reranker (its released final model), and its candidate pool, which is the pool every
# experiment chooses from. Usage: .venv/bin/python -m geo_search_env.experiment.pinpoint_reference {bench,mp16,report} [--root artifacts/pinpoint_reference]

"""Cache Pinpoint's reranked candidate pool for every benchmark photo and every MP16 query, and report its accuracy.

The reranker (`models.pinpoint_reranker`) retrieves <= 12 candidate photos from its own index (MP16 without bucket 99)
and predicts the top-scoring one. Index photos by the query's photographer are excluded (benchmark rule).

bench: both halves of im2gps3k and yfcc4k (query order of `strategy_search.load_world`).
mp16:  every query of `sft_data pool` (artifacts/sft/queries.json, same order): the Pinpoint-held-out bucket-99 photos and the
       sample of photos its retriever trained on. Reported: MP16 dev (held-out validation photographers, no burned-in GPS),
       MP16 train (held-out, the rest) and, as a leak check, trained-on photos (a big gap to held-out would mean the index
       remembers its own training photos, and training on them would teach over-trust in retrieval).

Saved per set (`{bench,mp16}.npz` + `{bench,mp16}_queries.json`), candidates sorted by reranker score, best first: rows /
cache_rows (attention-index and MP16 embedding-cache rows, -1 = empty), latlon, scores, features (the 9 retrieval
features), truth. `load_pools` reads them. Reported: reranker top-1 and the oracle over its pool, % within 1/25/200/750/2500 km.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..data.benchmarks import compute_metrics, geodesic_km
from .strategy_search import BENCHMARK_NAMES, MP16_EMBED, load_world

ROOT = Path("artifacts/pinpoint_reference")
SFT_ROOT = Path("artifacts/sft")
THRESHOLDS = (1, 25, 200, 750, 2500)


def load_pools(name: str, root: Path = ROOT) -> dict[str, Any]:
    """A saved set's arrays (rows, cache_rows, latlon, scores, features, truth; plus valid, csv_rows) and its queries."""

    with np.load(root / f"{name}.npz") as saved:
        pools: dict[str, Any] = {k: saved[k] for k in saved.files}
    pools["valid"] = pools["rows"] >= 0
    row_index = np.fromfile(MP16_EMBED / "row_index.i64.bin", dtype=np.int64)
    pools["csv_rows"] = np.where(pools["valid"], row_index[np.where(pools["valid"], pools["cache_rows"], 0)], -1)  # MP16-Pro CSV row per candidate
    pools["queries"] = json.loads((root / f"{name}_queries.json").read_text(encoding="utf-8"))
    return pools


def retrieval_order(features: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Candidate positions per query in the order retrieval produced them (best similarity among the searches that found
    each candidate, as the original's union orders its pool), before the reranker; empty slots last."""

    from ..models.pinpoint_reranker import FEATURE_GROUPS

    similarity = np.stack([features[..., FEATURE_GROUPS[m][0]] for m in ("image", "raw_image", "gps")], -1)
    found = np.stack([features[..., FEATURE_GROUPS[m][2]] for m in ("image", "raw_image", "gps")], -1) > 0
    score = np.where(found, similarity, -np.inf).max(-1)
    return np.argsort(-np.where(valid, score, -np.inf), axis=1, kind="stable")


def mp16_dev_mask(queries: Sequence[dict[str, Any]]) -> np.ndarray:
    """MP16 dev: held-out validation photographers, burned-in-GPS photos dropped (as `photo_sets select`)."""

    from .sft_data import OVERLAY_THRESHOLD

    overlay = json.loads((SFT_ROOT / "overlay_scores.json").read_text(encoding="utf-8"))
    return np.asarray([
        q["group"] == "held_out" and q["split"] == "val" and overlay.get(q["image_id"]) is not None and overlay[q["image_id"]] < OVERLAY_THRESHOLD
        for q in queries
    ])


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
    queries = json.loads((SFT_ROOT / "queries.json").read_text(encoding="utf-8"))
    _rank(load_world(mp16_queries=queries), root, "mp16")


def _line(label: str, predictions: np.ndarray, truth: np.ndarray) -> str:
    m = compute_metrics(predictions, truth)
    return f"  {label:34s}" + "".join(f"{m[f'Under_{t}_km']:7.1%}" for t in THRESHOLDS) + f"   median {m['Median_km_error']:7.0f} km"


def report(root: Path) -> None:
    for name in ("bench", "mp16"):
        if not (root / f"{name}.npz").exists():
            continue
        pools = load_pools(name, root)
        latlon, truth, valid, queries = pools["latlon"], pools["truth"], pools["valid"], pools["queries"]
        distance = np.full(valid.shape, np.inf)
        for c in range(valid.shape[1]):
            has = valid[:, c]
            distance[has, c] = geodesic_km(latlon[has, c], truth[has])
        oracle = latlon[np.arange(len(truth)), np.argmin(distance, axis=1)]
        if name == "mp16":
            dev = mp16_dev_mask(queries)
            held_out = np.asarray([q["group"] == "held_out" for q in queries])
            groups = {"MP16 dev": np.flatnonzero(dev), "MP16 held-out, rest": np.flatnonzero(held_out & ~dev),
                      "MP16 trained-on (leak check)": np.flatnonzero(~held_out)}
        else:
            groups = {f"{b} eval": np.flatnonzero([q["benchmark"] == b and q["split"] == "eval" for q in queries]) for b in BENCHMARK_NAMES}
            groups |= {"both eval halves": np.flatnonzero([q["split"] == "eval" for q in queries]),
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
