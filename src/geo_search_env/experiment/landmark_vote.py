# Landmark / object identification, decided by place instead of by single hit (lessons 58-60 follow-up), from the crop search's top-100 MP16 hits and
# their scores (landmark_id.py search --top-k 100 --no-verify, rescore).
# Usage: PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.landmark_vote vote --tag dev3k        (idea 1: location voting, switch rule)
#        PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.landmark_vote candidates --tag dev3k  (idea 2: per-candidate features + CV combiner)

"""vote:       per photo, every hit's place score = sum of weight(score) over hits within VOTE_KM of it; the best place is the hit with the top place
            score; margin = best place score minus the best place score >= 25 km away. Switch to the best place (snapped to a pool candidate within
            25 km, else its own location) if it is >= 25 km from Pinpoint's top-1 and place score / margin clear thresholds (quantile sweep on dev).
candidates: per Pinpoint pool candidate, features from the crop hits near it (max score and summed score within 1 / 5 km, hit counts) for each
            score (SigLIP2, DINOv2 CLS, DINOv2 GeM); listwise combiner of rank + these features, CV on the dev set (5 folds x 3 seeds), vs Pinpoint's
            top-1: change in top-1 < 1 / 25 km, fixes and breaks. Photos without an object crop get zero crop features."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np

from .photo_sets import load_photos
from .strategy_search import _bootstrap, _haversine_km

OUT = Path("artifacts/agent")
GALLERY = Path("data/pinpoint/mp16-embed/siglip2-giant-opt-patch16-384")
VOTE_KM = 1.0


def _load(tag: str, top_k: int = 100):
    photos = load_photos(tag)
    z = np.load(OUT / f"lmid_{tag}_k{top_k}.npz")
    d = np.load(OUT / f"lmid_dino_{tag}.npz")
    latlon = np.fromfile(GALLERY / "latlon_deg.f32.bin", dtype=np.float32).reshape(-1, 2).astype(np.float64)
    scores = {"siglip": z["sims"], "dino cls": d["cls"], "dino gem": d["gem"]}
    hits_ll = latlon[z["rows"]]  # [searched photos, top_k, 2]
    base = np.asarray([_haversine_km(*e["truth"], np.asarray(e["pool"][:1]))[0] for e in photos])
    return photos, z["photo"], scores, hits_ll, base


def _pairwise_km(ll: np.ndarray) -> np.ndarray:
    la, lo = np.radians(ll[:, 0]), np.radians(ll[:, 1])
    h = np.sin((la[:, None] - la[None]) / 2) ** 2 + np.cos(la[:, None]) * np.cos(la[None]) * np.sin((lo[:, None] - lo[None]) / 2) ** 2
    return 2 * 6371.0088 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


def vote(tag: str) -> None:
    photos, searched, scores, hits_ll, base = _load(tag)
    n = len(photos)
    print(f"{tag}: n={n}, Pinpoint top-1 < 25 km {(base < 25).mean():.1%}; photos with crop hits {len(searched)}")
    for name, sc in scores.items():
        lo, hi = np.quantile(sc, 0.5), sc.max()
        weight = np.clip((sc - lo) / (hi - lo), 0, None)  # hits below the median score don't vote
        best_ll, best_score, margin = [], [], []
        for i in range(len(searched)):
            km = _pairwise_km(hits_ll[i])
            place = (weight[i][None, :] * (km <= VOTE_KM)).sum(1)
            j = int(place.argmax())
            far = km[j] >= 25
            best_ll.append(hits_ll[i, j]); best_score.append(place[j]); margin.append(place[j] - (place[far].max() if far.any() else 0.0))
        best_ll, best_score, margin = np.asarray(best_ll), np.asarray(best_score), np.asarray(margin)
        for qs in (0.5, 0.8, 0.9, 0.95):
            for qm in (0.0, 0.5, 0.8, 0.9):
                Ts, Tm = np.quantile(best_score, qs), np.quantile(margin, qm)
                chosen, switched = base.copy(), 0
                for i, p in enumerate(searched):
                    if best_score[i] < Ts or margin[i] < Tm:
                        continue
                    e = photos[p]
                    if _haversine_km(*best_ll[i], np.asarray(e["pool"][:1]))[0] < 25:
                        continue
                    d_pool = _haversine_km(*best_ll[i], np.asarray(e["pool"]))
                    answer = np.asarray(e["pool"][int(d_pool.argmin())]) if d_pool.min() < 25 else best_ll[i]
                    chosen[p] = _haversine_km(*e["truth"], answer[None])[0]
                    switched += 1
                c = _bootstrap((chosen < 25) * 1.0 - (base < 25))
                fixes, breaks = ((chosen < 25) & (base >= 25)).sum(), ((chosen >= 25) & (base < 25)).sum()
                print(f"  {name:9s} place score >= q{qs:.2f}, margin >= q{qm:.2f}: switched {switched:4d}; < 25 km {100 * c[0]:+.1f} "
                      f"[{100 * c[1]:+.1f}, {100 * c[2]:+.1f}]; fixes {fixes}, breaks {breaks}")


FEATURES = [f"{s} {k} {r}" for s in ("siglip", "dino cls", "dino gem") for k in ("max", "sum") for r in ("1km", "5km")] + ["hits 1km", "hits 5km"]


def candidates(tag: str) -> None:
    from .name_score import _fit

    photos, searched, scores, hits_ll, base = _load(tag)
    n, C = len(photos), max(len(e["pool"]) for e in photos)
    F = np.zeros((n, C, 3 + len(FEATURES)), np.float32)
    D, V = np.full((n, C), 1e5), np.zeros((n, C), bool)
    for p, e in enumerate(photos):
        k = len(e["pool"])
        D[p, :k] = _haversine_km(*e["truth"], np.asarray(e["pool"]))
        V[p, :k] = True
        F[p, :k, 0], F[p, :k, 1], F[p, :k, 2] = np.arange(k), np.log1p(np.arange(k)), np.arange(k) == 0
    position = {int(p): i for i, p in enumerate(searched)}
    for p, e in enumerate(photos):
        i = position.get(p)
        if i is None:
            continue
        for c, (lat, lon) in enumerate(e["pool"]):
            km = _haversine_km(lat, lon, hits_ll[i])
            col = 3
            for name in ("siglip", "dino cls", "dino gem"):
                sc = scores[name][i]
                for agg in ("max", "sum"):
                    for r in (1.0, 5.0):
                        near = km <= r
                        F[p, c, col] = (sc[near].max() if agg == "max" else sc[near].sum()) if near.any() else 0.0
                        col += 1
            F[p, c, col], F[p, c, col + 1] = (km <= 1.0).sum(), (km <= 5.0).sum()
    print(f"{tag}: n={n}, Pinpoint top-1 < 1 / 25 km {(base < 1).mean():.1%} / {(base < 25).mean():.1%}; photos with crop hits {len(searched)}")
    folds = np.array_split(np.random.default_rng(0).permutation(n), 5)
    groups = {"rank only": [0, 1, 2], "rank + crop hits near candidates": list(range(F.shape[-1]))}
    for label, cols in groups.items():
        hits = {km: np.zeros(n) for km in (1, 25)}
        for seed in range(3):
            for f in folds:
                tr = np.setdiff1d(np.arange(n), f)
                pick = np.argmax(_fit(F[tr][..., cols], D[tr], V[tr], seed)(F[f][..., cols], V[f]), 1)
                for km in hits:
                    hits[km][f] += (D[f, pick] < km) / 3
        parts = []
        for km in (1, 25):
            c = _bootstrap(hits[km] - (base < km))
            parts.append(f"< {km} km {100 * c[0]:+.1f} [{100 * c[1]:+.1f}, {100 * c[2]:+.1f}]")
        fixes, breaks = hits[25][base >= 25].sum(), (1 - hits[25][base < 25]).sum()
        print(f"  CV {label:34s}: " + "; ".join(parts) + f"; 25 km fixes {fixes:.0f}, breaks {breaks:.0f}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("node", choices=("vote", "candidates"))
    parser.add_argument("--tag", default="dev3k")
    args = parser.parse_args(argv)
    vote(args.tag) if args.node == "vote" else candidates(args.tag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
