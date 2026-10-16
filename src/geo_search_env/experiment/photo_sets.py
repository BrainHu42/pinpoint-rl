# Fixed photo sets with Pinpoint's candidate pool: MP16 dev / val sets to develop on, benchmark sets to measure once.
# Usage: .venv/bin/python -m geo_search_env.experiment.photo_sets select --tag dev --n 1000                 (MP16 dev photos)
#        .venv/bin/python -m geo_search_env.experiment.photo_sets select --source bench --tag full --n 100000   (all benchmark eval-half photos)
#        (needs `pinpoint_reference mp16` / `bench`; then `place_labels pool-labels --tag <tag>` for Overture labels)

"""Each set is PHOTO_SETS/<tag>/photos.json, one entry per photo:
  image_id, truth [lat, lon]; MP16 photos: index (position in artifacts/sft/queries.json), row (MP16 embedding-cache row);
  benchmark photos: index (position in the benchmark caches / `load_world` order), path, benchmark;
  pool: every candidate of Pinpoint's pool [lat, lon], reranker order (pool[0] = Pinpoint's top-1, the reference);
  pool_rows: each candidate's MP16 embedding-cache row; candidates: the first CANDIDATES as [place name, lat, lon], the
  name being the candidate photo's own MP16-Pro 'city, state, country'.

MP16 sets: the first n of one fixed shuffle of the Pinpoint-held-out validation photos without burned-in GPS (a pilot is a
prefix of the full set). Benchmark sets: n // 2 random eval-half photos from each of im2gps3k and yfcc4k (fixed seed; a large n
takes them all). The photos chosen do not depend on the pool.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .pinpoint_reference import load_pools, mp16_dev_mask
from .strategy_search import BENCHMARK_NAMES

PHOTO_SETS = Path("artifacts/photo_sets")
CANDIDATES = 10


def load_photos(tag: str) -> list[dict[str, Any]]:
    return json.loads((PHOTO_SETS / tag / "photos.json").read_text(encoding="utf-8"))


def _entries(pools: dict[str, Any], picked: Sequence[int], extra) -> list[dict[str, Any]]:
    from .sft_data import _place_names

    valid, latlon, csv_rows = pools["valid"], pools["latlon"], pools["csv_rows"]
    names = _place_names({int(csv_rows[i, c]) for i in picked for c in np.flatnonzero(valid[i])[:CANDIDATES]})
    out = []
    for i in picked:
        slots = np.flatnonzero(valid[i])  # sorted best first; empty slots are at the end
        out.append(extra(i) | {
            "truth": pools["truth"][i].tolist(),
            "candidates": [[names[int(csv_rows[i, c])], float(latlon[i, c, 0]), float(latlon[i, c, 1])] for c in slots[:CANDIDATES]],
            "pool": [[float(latlon[i, c, 0]), float(latlon[i, c, 1])] for c in slots],
            "pool_rows": [int(pools["cache_rows"][i, c]) for c in slots],
        })
    return out


def select(tag: str, n: int) -> None:
    pools = load_pools("mp16")
    queries = pools["queries"]
    picked = sorted(int(i) for i in np.random.default_rng(0).permutation(np.flatnonzero(mp16_dev_mask(queries)))[:n])
    photos = _entries(pools, picked, lambda i: {"index": i, "image_id": queries[i]["image_id"], "row": queries[i]["row"]})
    _save(tag, photos)
    print(f"{len(photos)} MP16 dev photos of {int(mp16_dev_mask(queries).sum())} eligible -> {PHOTO_SETS / tag}")


def select_bench(tag: str, n: int) -> None:
    pools = load_pools("bench")
    queries = pools["queries"]
    rng = np.random.default_rng(0)
    picked: list[int] = []
    for name in BENCHMARK_NAMES:
        members = [q for q, x in enumerate(queries) if x["benchmark"] == name and x["split"] == "eval"]
        picked += sorted(int(q) for q in rng.permutation(members)[: n // 2])
    keys = ("image_id", "path", "benchmark")
    photos = _entries(pools, picked, lambda q: {"index": q} | {k: queries[q][k] for k in keys})
    _save(tag, photos)
    print(f"{len(photos)} benchmark photos: " + ", ".join(f"{b} {sum(p['benchmark'] == b for p in photos)}" for b in BENCHMARK_NAMES) + f" -> {PHOTO_SETS / tag}")


def select_commons26(tag: str) -> None:
    """All 5,023 commons26 photos (user, 2026-10-16), each with the benchmark's own split (dev / test) in "split"."""

    pools = load_pools("commons26")
    queries = pools["queries"]
    photos = _entries(pools, range(len(queries)), lambda q: {"index": q} | {k: queries[q][k] for k in ("image_id", "path", "benchmark", "split")})
    _save(tag, photos)
    print(f"{len(photos)} commons26 photos -> {PHOTO_SETS / tag}")


def _save(tag: str, photos: list[dict[str, Any]]) -> None:
    (PHOTO_SETS / tag).mkdir(parents=True, exist_ok=True)
    (PHOTO_SETS / tag / "photos.json").write_text(json.dumps(photos, ensure_ascii=False) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("node", choices=("select",))
    parser.add_argument("--tag", default="dev", help="output folder under artifacts/photo_sets")
    parser.add_argument("--n", type=int, default=1000, help="number of photos")
    parser.add_argument("--source", choices=("mp16", "bench", "commons26"), default="mp16", help="MP16 dev photos, benchmark eval-half photos or all of commons26")
    args = parser.parse_args(argv)
    if args.source == "commons26":
        select_commons26(args.tag)
    else:
        (select_bench if args.source == "bench" else select)(args.tag, args.n)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
