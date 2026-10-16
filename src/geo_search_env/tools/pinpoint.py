"""pinpoint(photo, constraint): Pinpoint's retrieval + attention reranker over its MP16 index, the agent's visual search tool, optionally restricted
to where other evidence points. One entry point for what lessons 64, 69, 70 and 79 did with separate code:

  none             the whole index (Pinpoint as released / retrained)
  box              a lat / lon box (the pipeline's zoom: an area's 25 km box)
  country, region  index photos labelled with that Overture country / 'country|region' (lesson 64: the true region takes dev3k 33.4 -> 61.3%)
  near             index photos within km of any of a set of points (a name's map matches; lesson 79)
  rows             an explicit set of index rows

Index photos by the query's photographer are excluded whenever its author id is given (our rule for all retrieval). none / box / country /
region stream the index once per batch (PinpointReranker.rank); near / rows search only their rows on a GPU-resident index
(PinpointReranker.rank_subset; the same top-1 location within 1 km as the box search on 98% of photos). The reranker checkpoint comes from
PINPOINT_RERANKER_CKPT (the pipeline's: artifacts/pinpoint_train/setft_a/checkpoints/ckpt_final.pt); pools are the pipeline's wide ones
(32 / 8 / 8 hits per modality) unless targets=None.

  PINPOINT_RERANKER_CKPT=... python -m geo_search_env.tools.pinpoint selftest     (dev3k: unconstrained = the saved pool, rows = box)
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np

LABELS_NPZ = Path("artifacts/place_labels/mp16_country_region.npz")  # MP16 photos' Overture country / region (place_labels; built by duckdb once)
NVME_INDEX = Path("/opt/dlami/nvme/pinpoint/attention_retrieval_index")


@dataclass
class Constraint:
    box: tuple[float, float, float, float] | None = None  # lat min, lat max, lon min, lon max
    country: str | None = None
    region: str | None = None  # 'country|region'
    near: tuple[np.ndarray, float] | None = None  # points [k, 2], km
    rows: np.ndarray | None = None

    def kind(self) -> str:
        for k in ("rows", "near", "region", "country", "box"):
            if getattr(self, k) is not None:
                return k
        return "none"


class Pinpoint:
    def __init__(self, device: str = "cuda", targets: Sequence[int] | None = (32, 8, 8)) -> None:
        """targets: hits per modality (the pipeline's wide pools: 32 8 8; None: Pinpoint's own 8 2 2)."""

        from ..models.pinpoint_reranker import PinpointReranker

        index_dir = NVME_INDEX if (NVME_INDEX / "manifest.json").exists() else None
        self.reranker = PinpointReranker(device=device, index_dir=index_dir)
        if targets is not None:
            self.reranker.widen(targets)
        self._author = self._labels = self._tree = None
        self._resident = False

    # lazily built: MP16 authors (photographer exclusion), area labels (country / region), a KD-tree over index locations (near)
    def index_author(self) -> np.ndarray:
        if self._author is None:
            from ..experiment.strategy_search import load_world

            self._mp16_author = np.asarray(load_world().mp16["author"])
            self._author = self._mp16_author[self.reranker.cache_indices]
        return self._author

    def mp16_author(self, cache_rows: np.ndarray) -> np.ndarray:
        """Author ids of MP16 photos (embedding-cache rows), e.g. train / dev queries."""

        self.index_author()
        return self._mp16_author[cache_rows]

    def labels(self) -> tuple[np.ndarray, np.ndarray]:
        if self._labels is None:
            z = np.load(LABELS_NPZ)
            country = np.full(z["row"].max() + 1, "", dtype=object)
            region = np.full(z["row"].max() + 1, "", dtype=object)
            country[z["row"]] = z["country"]
            region[z["row"]] = [f"{c}|{r}" if c and r else "" for c, r in zip(z["country"], z["region"])]
            self._mp16_labels = country, region
            self._labels = country[self.reranker.cache_indices], region[self.reranker.cache_indices]
        return self._labels

    def mp16_labels(self, cache_rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(country, 'country|region') of MP16 photos (embedding-cache rows), e.g. a dev photo's true area."""

        self.labels()
        rows = np.minimum(cache_rows, len(self._mp16_labels[0]) - 1)
        return self._mp16_labels[0][rows], self._mp16_labels[1][rows]

    def labels_at(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(country, 'country|region') of the index photo nearest each point: an area label for any coordinate."""

        from ..models.pinpoint_reranker import _unit_xyz

        self.rows_near(np.zeros((1, 2)), 0.0)  # builds the tree
        _, nearest = self._tree.query(_unit_xyz(np.asarray(points, np.float64).reshape(-1, 2)))
        country, region = self.labels()
        return country[nearest], region[nearest]

    def rows_near(self, points: np.ndarray, km: float) -> np.ndarray:
        from scipy.spatial import cKDTree

        from ..models.pinpoint_reranker import _unit_xyz

        if self._tree is None:
            self._tree = cKDTree(_unit_xyz(np.asarray(self.reranker.latlon)), leafsize=64)
        near = self._tree.query_ball_point(_unit_xyz(np.asarray(points, np.float64).reshape(-1, 2)), 2 * math.sin(km / 6371.0 / 2))
        return np.unique(np.concatenate([np.asarray(r, np.int64) for r in near])) if len(near) else np.empty(0, np.int64)

    def search(self, embeddings: np.ndarray, constraints: Sequence[Constraint | None] | None = None, authors: np.ndarray | None = None):
        """Raw SigLIP2 embeddings [Q, 1536] -> RerankedCandidates (candidates sorted by reranker score). constraints: one per query (None =
        unconstrained); authors: the queries' MP16 author ids (their own photos excluded) or None."""

        from ..models.pinpoint_reranker import RerankedCandidates

        q = len(embeddings)
        constraints = list(constraints) if constraints is not None else [None] * q
        kinds = np.asarray([(c or Constraint()).kind() for c in constraints])
        parts: list[tuple[np.ndarray, object]] = []
        stream = np.flatnonzero(np.isin(kinds, ("none", "box")))
        if len(stream):
            box = np.asarray([c.box if c is not None and c.box is not None else (np.nan,) * 4 for c in (constraints[i] for i in stream)], np.float32)
            parts.append((stream, self._rank(embeddings[stream], authors, stream, query_box=box)))
        for level in ("country", "region"):
            idx = np.flatnonzero(kinds == level)
            if len(idx):
                names = self.labels()[0 if level == "country" else 1]
                vocab = {v: i for i, v in enumerate(sorted(set(names) - {""}))}
                index_ids = np.asarray([vocab.get(v, -2) for v in names])
                query_ids = np.asarray([vocab.get(getattr(constraints[i], level), len(vocab)) for i in idx])  # unknown label: nothing matches (< 0 = anywhere)
                parts.append((idx, self._rank(embeddings[idx], authors, idx, query_region=query_ids, index_region=index_ids)))
        subset = np.flatnonzero(np.isin(kinds, ("near", "rows")))
        if len(subset):
            if not self._resident:
                self.reranker.load_resident()
                self._resident = True
            allowed = []
            for i in subset:
                c = constraints[i]
                rows = c.rows if c.rows is not None else self.rows_near(*c.near)
                if authors is not None:
                    rows = rows[self.index_author()[rows] != authors[i]]
                allowed.append(np.asarray(rows, np.int64))
            parts.append((subset, self.reranker.rank_subset(embeddings[subset], allowed)))
        c = max(p.rows.shape[1] for _, p in parts)
        out = {k: None for k in ("rows", "cache_rows", "latlon", "scores", "features")}
        fill = {"rows": -1, "cache_rows": -1, "latlon": np.nan, "scores": -np.inf, "features": 0.0}
        for k in out:
            shape = (q, c) + getattr(parts[0][1], k).shape[2:]
            out[k] = np.full(shape, fill[k], dtype=getattr(parts[0][1], k).dtype)
            for idx, p in parts:
                v = getattr(p, k)
                out[k][idx, : v.shape[1]] = v
        return RerankedCandidates(**out)

    def _rank(self, embeddings, authors, idx, **kw):
        if authors is None:
            return self.reranker.rank(embeddings, **kw)
        return self.reranker.rank(embeddings, query_author=authors[idx], index_author=self.index_author(), **kw)


def format_candidates(found, i: int, top: int = 5, label=None) -> str:
    """One query's candidates as the agent reads them: rank, coordinates, reranker score, and a place label if a labeller is given."""

    lines = [f"pinpoint: {int((found.rows[i] >= 0).sum())} candidates"]
    for j in range(min(top, found.rows.shape[1])):
        if found.rows[i, j] < 0:
            break
        lat, lon = found.latlon[i, j]
        where = f" near {label(lat, lon)}" if label else ""
        lines.append(f"  {j + 1}. ({lat:.4f}, {lon:.4f}) score {found.scores[i, j]:.2f}{where}")
    return "\n".join(lines)


def selftest(n: int = 64) -> None:
    """On n dev3k photos: unconstrained search vs the saved retrained-reranker pool, and a near-constraint vs the same box."""

    from ..experiment.strategy_search import load_world

    photos = json.loads(Path("artifacts/photo_sets/dev3k_setft_a/photos.json").read_text(encoding="utf-8"))[:n]
    rows = np.asarray([int(p["row"]) for p in photos])
    e = np.asarray(load_world().mp16["embeddings"][rows], np.float32)
    tool = Pinpoint()
    authors = tool.mp16_author(rows)
    free = tool.search(e, authors=authors)
    saved = np.asarray([json.loads(p["candidates"])[0][1:] if isinstance(p["candidates"], str) else p["candidates"][0][1:] for p in photos], np.float64)
    print(f"unconstrained top-1 = the saved pool's top-1 (< 0.1 km): {np.mean(np.hypot(*(free.latlon[:, 0] - saved).T) * 111 < 0.1):.0%}")
    truth = np.asarray([json.loads(p["truth"]) if isinstance(p["truth"], str) else p["truth"] for p in photos], np.float64)
    boxes = [Constraint(box=(t[0] - 0.3, t[0] + 0.3, t[1] - 0.4, t[1] + 0.4)) for t in truth]
    ll = np.asarray(tool.reranker.latlon)
    inside = [Constraint(rows=np.flatnonzero((ll[:, 0] >= b.box[0]) & (ll[:, 0] <= b.box[1]) & (ll[:, 1] >= b.box[2]) & (ll[:, 1] <= b.box[3])))
              for b in boxes]
    a, s = tool.search(e, boxes, authors), tool.search(e, inside, authors)
    km = np.hypot(*(np.nan_to_num(a.latlon[:, 0]) - np.nan_to_num(s.latlon[:, 0])).T) * 111
    print(f"box (streamed) vs the same rows (subset): top-1 location within 1 km {np.mean(km < 1):.0%}")
    country = tool.search(e, [Constraint(country=tool.labels()[0][int(np.argmin(np.hypot(*(ll[::50] - t).T)) * 50)]) for t in truth], authors)
    print(f"restricted to the truth's country (nearest index photo's label): top-1 < 25 km {np.mean(np.hypot(*(country.latlon[:, 0] - truth).T) * 111 < 25):.0%}, "
          f"unconstrained {np.mean(np.hypot(*(free.latlon[:, 0] - truth).T) * 111 < 25):.0%}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cmd", choices=("selftest",))
    parser.parse_args(argv)
    selftest()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
