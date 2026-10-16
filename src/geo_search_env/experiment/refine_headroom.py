# Refinement to the next threshold bucket (lesson 91, the user's idea): keep the coarse answer, move it into the next tighter bucket
# (25 km -> 1 km, 200 km -> 25 km, 750 km -> 200 km). Headroom on dev3k from the retrained reranker's top-1: per current bucket, how often
# (a) the saved pool and (b) Pinpoint restricted to the bucket's radius around the top-1 (photographer excluded) hold a candidate inside the
# next threshold, and how often the restricted search's own top-1 lands there.
# Usage: PYTHONPATH=src PINPOINT_RERANKER_CKPT=artifacts/pinpoint_train/setft_a/checkpoints/ckpt_final.pt \
#          .venv/bin/python -m geo_search_env.experiment.refine_headroom --tag dev3k

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np

OUT = Path("artifacts/agent")
CKPT_TAG = "setft_a"
EDGES = (1.0, 25.0, 200.0, 750.0, 2500.0)


def _hav(lat1, lon1, lat2, lon2) -> np.ndarray:
    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(v, np.float64)) for v in (lat1, lon1, lat2, lon2))
    h = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371.0088 * np.arcsin(np.sqrt(np.clip(np.nan_to_num(h), 0, 1)))


def _km(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return _hav(a[:, 0], a[:, 1], b[:, 0], b[:, 1])


def run(tag: str) -> None:
    from ..tools.pinpoint import Constraint, Pinpoint

    photos = json.loads((Path("artifacts/photo_sets") / f"{tag}_{CKPT_TAG}" / "photos.json").read_text(encoding="utf-8"))
    truth = np.asarray([e["truth"] for e in photos], np.float64)
    top1 = np.asarray([e["candidates"][0][1:] for e in photos], np.float64)
    d = _km(truth, top1)
    bucket = np.searchsorted(EDGES, d)  # 0: < 1 km, 1: 1-25, 2: 25-200, 3: 200-750, 4: 750-2500, 5: beyond
    n = len(photos)
    print(f"{tag}: {n} photos; retrained top-1 buckets: " + ", ".join(
        f"{lab} {np.mean(bucket == b):.1%}" for b, lab in enumerate(("<1", "1-25", "25-200", "200-750", "750-2500", ">2500"))))
    pool_hit = np.zeros(n, bool)
    for i, e in enumerate(photos):
        if 1 <= bucket[i] <= 4:
            pool = np.asarray(e["pool"], np.float64)
            pool_hit[i] = (_hav(truth[i, 0], truth[i, 1], pool[:, 0], pool[:, 1]) < EDGES[bucket[i] - 1]).any()
    tool = Pinpoint()
    views = np.load(OUT / f"crops_{tag}_{CKPT_TAG}.npy", mmap_mode="r")
    authors = tool.mp16_author(np.asarray([e["row"] for e in photos]))
    rows_out = {}
    for b in (1, 2, 3):
        idx = np.flatnonzero(bucket == b)
        radius, target = EDGES[b], EDGES[b - 1]
        found = tool.search(np.asarray(views[idx, 0], np.float32), [Constraint(near=(top1[i][None], radius)) for i in idx], authors[idx])
        ll = found.latlon
        dk = np.where(np.isnan(ll[..., 0]), np.inf, _hav(truth[idx, 0][:, None], truth[idx, 1][:, None], ll[..., 0], ll[..., 1]))
        oracle, first = (dk < target).any(1), dk[:, 0] < target
        top5 = (dk[:, :5] < target).any(1)
        rows_out[b] = {"idx": idx, "km": dk, "latlon": ll, "scores": found.scores}
        print(f"\n  bucket {EDGES[b - 1]:g}-{EDGES[b]:g} km ({len(idx)} photos, {len(idx) / n:.1%}): next threshold {target:g} km")
        print(f"    saved pool has a candidate < {target:g} km: {pool_hit[idx].mean():.1%}")
        print(f"    restricted to {radius:g} km around the top-1 ({np.mean((ll[:, :, 0] == ll[:, :, 0]).sum(1)):.0f} candidates): any < {target:g} km "
              f"{oracle.mean():.1%}, in its top 5 {top5.mean():.1%}, its top-1 {first.mean():.1%}")
        print(f"    -> moving every photo of the bucket: +{100 * first.mean() * len(idx) / n:.1f} pts at {target:g} km if the restricted top-1 is taken "
              f"(oracle +{100 * oracle.mean() * len(idx) / n:.1f})")
    np.savez(OUT / f"refine_headroom_{tag}_{CKPT_TAG}.npz", bucket=bucket, km=d,
             **{f"b{b}_{k}": v for b, r in rows_out.items() for k, v in r.items()})


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="dev3k")
    args = parser.parse_args(argv)
    run(args.tag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
