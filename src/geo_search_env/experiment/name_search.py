"""Name-restricted Pinpoint search: OSM names as a candidate generator (lesson 79). Names checked against Pinpoint's own areas add nothing
(lesson 78), but on the half of photos where no area is right a fairly specific name often matches near the truth. Here each such name
narrows the world to the places it matches, and Pinpoint, searched only among gallery photos near those places, picks among them by look.

search:  per photo (train / dev3k, {tag}_setft_a) and per name (name_checks: read text / place names and recalled queries) with
         1..K_MAX worldwide OSM matches, the retrained reranker (setft_a, as the pipeline) over the index photos within R_KM of any match
         (query photographer's photos excluded, as all our retrieval) -> artifacts/agent/name_search_{tag}_setft_a.npz: per (photo, name)
         query: photo, kind, worldwide matches, allowed rows, candidate latlon / scores / features (best first).
selftest: rank_subset against the box-restricted search on a few dev3k photos.

  PINPOINT_RERANKER_CKPT=artifacts/pinpoint_train/setft_a/checkpoints/ckpt_final.pt \\
    python -m geo_search_env.experiment.name_search search [--tags dev3k train]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from .name_checks import KINDS, Hits, photo_names

OUT = Path("artifacts/agent")
CKPT_TAG = "setft_a"
K_MAX = 1000
R_KM = 2.0
NVME_INDEX = Path("/opt/dlami/nvme/pinpoint/attention_retrieval_index")


def _setup():
    from scipy.spatial import cKDTree

    from ..models.pinpoint_reranker import PinpointReranker, _unit_xyz
    from .strategy_search import load_world

    index_dir = NVME_INDEX if (NVME_INDEX / "manifest.json").exists() else None
    reranker = PinpointReranker(device="cuda", index_dir=index_dir)
    world = load_world()
    author = np.asarray(world.mp16["author"])
    index_author = author[reranker.cache_indices]
    tree = cKDTree(_unit_xyz(np.asarray(reranker.latlon)), leafsize=64)
    return reranker, author, index_author, tree, _unit_xyz


def _embeddings(rows: np.ndarray) -> np.ndarray:
    from .strategy_search import MP16_EMBED

    manifest = json.loads((MP16_EMBED / "manifest.json").read_text(encoding="utf-8"))
    emb = np.memmap(MP16_EMBED / manifest["files"]["embeddings"], dtype=np.dtype(manifest["dtypes"]["embeddings"]), mode="r",
                    shape=(manifest["num_samples"], manifest["embedding_dim"]))
    order = np.argsort(rows)
    out = np.empty((len(rows), emb.shape[1]), np.float32)
    out[order] = emb[rows[order]]
    return out


def search(tags: list[str], batch: int = 256) -> None:
    reranker, author, index_author, tree, unit = _setup()
    reranker.load_resident()
    hits = Hits()
    chord = 2 * math.sin(R_KM / 6371.0 / 2)
    for tag in tags:
        photos = json.loads((Path("artifacts/photo_sets") / f"{tag}_{CKPT_TAG}" / "photos.json").read_text(encoding="utf-8"))
        prow = np.asarray([p["row"] for p in photos])
        names = photo_names(tag)
        by_name: dict[str, list[tuple[int, int]]] = {}
        for i, ph in enumerate(names):
            for k, kind in enumerate(KINDS):
                for x in ph[kind]:
                    by_name.setdefault(x, []).append((i, k))
        queries = []  # (photo, kind, name count, allowed rows)
        for x, users in by_name.items():
            n, xy = hits.get(x)
            if not (0 < n <= K_MAX) or not len(xy):
                continue
            near = tree.query_ball_point(unit(xy), chord)
            rows = np.unique(np.concatenate([np.asarray(r, np.int64) for r in near])) if len(near) else np.empty(0, np.int64)
            for i, k in users:
                queries.append((i, k, n, rows[index_author[rows] != author[prow[i]]]))
        print(f"{tag}: {len(queries):,} (photo, name) queries on {len({q[0] for q in queries}):,} photos; median {np.median([len(q[3]) for q in queries]):.0f} "
              f"allowed index photos", flush=True)
        emb = _embeddings(prow)
        parts = []
        for b in range(0, len(queries), batch):
            qs = queries[b: b + batch]
            found = reranker.rank_subset(emb[[q[0] for q in qs]], [q[3] for q in qs])
            parts.append(found)
            if (b // batch) % 20 == 0:
                print(f"  {b + len(qs):,} / {len(queries):,}", flush=True)
        cat = lambda key: np.concatenate([getattr(f, key) for f in parts])
        out = OUT / f"name_search_{tag}_{CKPT_TAG}.npz"
        np.savez(out, photo=np.asarray([q[0] for q in queries]), kind=np.asarray([q[1] for q in queries]), count=np.asarray([q[2] for q in queries]),
                 n_allowed=np.asarray([len(q[3]) for q in queries]), latlon=cat("latlon").astype(np.float32), scores=cat("scores").astype(np.float32),
                 features=cat("features").astype(np.float32), truth=np.asarray([p["truth"] for p in photos], np.float32))
        print(f"-> {out}", flush=True)


def selftest(n: int = 64) -> None:
    """rank_subset with the index rows inside a box == rank(query_box=box) on n dev3k photos."""

    reranker, author, index_author, tree, unit = _setup()
    reranker.load_resident()
    photos = json.loads((Path("artifacts/photo_sets") / f"dev3k_{CKPT_TAG}" / "photos.json").read_text(encoding="utf-8"))[:n]
    prow = np.asarray([p["row"] for p in photos])
    truth = np.asarray([p["truth"] for p in photos])
    box = np.stack([truth[:, 0] - 0.3, truth[:, 0] + 0.3, truth[:, 1] - 0.4, truth[:, 1] + 0.4], 1)
    emb = _embeddings(prow)
    a = reranker.rank(emb, query_author=author[prow], index_author=index_author, query_box=box)
    ll = np.asarray(reranker.latlon)
    allowed = [np.flatnonzero((ll[:, 0] >= b[0]) & (ll[:, 0] <= b[1]) & (ll[:, 1] >= b[2]) & (ll[:, 1] <= b[3]) & (index_author != author[r]))
               for b, r in zip(box, prow)]
    s = reranker.rank_subset(emb, allowed)
    same_top = np.mean(a.rows[:, 0] == s.rows[:, 0])
    same_set = np.mean([set(x[x >= 0]) == set(y[y >= 0]) for x, y in zip(a.rows, s.rows)])
    places = lambda ll_, r: {tuple(np.round(v, 4)) for v, ok in zip(ll_, r >= 0) if ok}
    same_places = np.mean([places(x, rx) == places(y, ry) for x, rx, y, ry in zip(a.latlon, a.rows, s.latlon, s.rows)])
    top_km = np.asarray([np.hypot(*(x[0] - y[0])) * 111 for x, y in zip(a.latlon, s.latlon)])
    print(f"  same candidate locations {same_places:.0%}; top-1 location differs by > 1 km on {np.mean(top_km > 1):.0%}")
    print(f"selftest on {n} photos: same top-1 {same_top:.0%}, same candidate set {same_set:.0%}; max score diff "
          f"{np.nanmax(np.abs(np.where(np.isfinite(a.scores[:, 0]), a.scores[:, 0] - s.scores[:, 0], 0))):.4f}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cmd", choices=("search", "selftest"))
    parser.add_argument("--tags", nargs="+", default=["dev3k", "train"])
    args = parser.parse_args(argv)
    if args.cmd == "search":
        search(args.tags)
    else:
        selftest()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
