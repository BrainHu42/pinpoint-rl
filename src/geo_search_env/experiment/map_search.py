# Map search: instead of choosing among a fixed candidate list, move over the map. From the reranker's top seeds, score gallery photos (MP16 not by the query's
# photographer + OSV-5M, SigLIP2 similarity to the query) inside a radius, keep the best spots, and zoom in (25 -> 5 -> 1.5 km).
# Usage: PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.map_search probe [--limit 50]   (CPU)

"""probe: a hand-written zoom search with SigLIP2 similarity only, on the MP16 dev photos. Reports (a) the share of photos with a visited point within 1 / 25 km
       of the truth, at matched budgets (the top K visited points by score) against the fixed local-candidate list (near_miss.py, <= 24 points) and the
       reranker pool, and (b) simple pick rules against the reranker top-1. The question: does searching reach the truth more often than the fixed list?"""

from __future__ import annotations

import argparse
import json
import time
from typing import Sequence

import numpy as np

from .near_miss import _km_matrix, _local_set, _top1_photos
from .query_evidence import ROOT
from .strategy_search import MP16_EMBED, OSV_EMBED

SEEDS = 3  # reranker pool candidates the search starts from
RADII = (25.0, 5.0, 1.5)  # zoom schedule (km)
BEAM = 2  # spots kept per seed after each step
CAP = 5000  # gallery photos scored per ball at most (a random sample beyond)
SPOT_KM = 1.0  # spots: photos at least this far apart, most similar first; a spot's score sums the similarity weights of the scored photos within it
TOP_PHOTOS = 400  # most similar scored photos of a seed that spots are made from
TEMPERATURE = 0.03
BUDGETS = (1, 3, 8, 24)


class Gallery:
    def __init__(self, world) -> None:
        from scipy.spatial import cKDTree

        from .strategy_search import _xyz

        self.parts = [(world.mp16["embeddings"], world.mp16["latlon"], world.mp16["author"]), (world.osv["embeddings"], world.osv["latlon"], world.osv["author"])]
        print("building location trees", flush=True)
        self.trees = [cKDTree(_xyz(p[1])) for p in self.parts]
        self._xyz = _xyz

    def ball(self, center: np.ndarray, radius_km: float, q: np.ndarray, author: int, rng: np.random.Generator, seen: set) -> tuple[np.ndarray, np.ndarray, list]:
        """Locations, similarities and keys of not-yet-scored gallery photos within radius_km of center (at most CAP, a random sample)."""

        chord = 2 * np.sin(radius_km / 6371.0088 / 2)
        found = []
        for part, (tree, (emb, latlon, authors)) in enumerate(zip(self.trees, self.parts)):
            rows = np.asarray(tree.query_ball_point(self._xyz(center), chord), dtype=np.int64)
            rows = rows[authors[rows] != author] if len(rows) else rows
            found += [(part, int(r)) for r in rows if (part, int(r)) not in seen]
        if len(found) > CAP:
            found = [found[i] for i in rng.choice(len(found), CAP, replace=False)]
        if not found:
            return np.zeros((0, 2)), np.zeros(0), []
        pts, sims = np.zeros((len(found), 2)), np.zeros(len(found))
        for part in (0, 1):
            idx = [i for i, (p, _) in enumerate(found) if p == part]
            if not idx:
                continue
            rows = np.asarray([found[i][1] for i in idx])
            order = np.argsort(rows)
            emb = np.asarray(self.parts[part][0][rows[order]], dtype=np.float32)
            s = emb @ q / np.maximum(np.linalg.norm(emb, axis=1), 1e-6)
            sims[np.asarray(idx)[order]] = s
            pts[idx] = self.parts[part][1][rows]
        return pts, sims, found


def _spots(pts: np.ndarray, sims: np.ndarray, smax: float, limit: int = 24) -> tuple[np.ndarray, np.ndarray]:
    """Spots (photo locations >= SPOT_KM apart, most similar first) among the TOP_PHOTOS most similar scored photos, and their scores."""

    top = np.argsort(-sims)[:TOP_PHOTOS]
    p, s = pts[top], sims[top]
    d = _km_matrix(p, p)
    chosen: list[int] = []
    for i in range(len(p)):
        if all(d[i, j] >= SPOT_KM for j in chosen):
            chosen.append(i)
            if len(chosen) == limit:
                break
    w = np.exp((s - smax) / TEMPERATURE)
    score = (w[None, :] * (d[chosen] < SPOT_KM)).sum(1)
    return p[chosen], score


def _search(gallery: Gallery, q: np.ndarray, author: int, seeds: np.ndarray, rng: np.random.Generator) -> dict:
    seen: set = set()
    pts_all, sims_all, seed_of = [], [], []
    for k, seed in enumerate(seeds):
        frontier = [seed]
        pts, sims = np.zeros((0, 2)), np.zeros(0)
        for radius in RADII:
            for f in frontier:
                p, s, keys = gallery.ball(f, radius, q, author, rng, seen)
                seen.update(keys)
                pts, sims = np.concatenate((pts, p)), np.concatenate((sims, s))
            if not len(sims):
                break
            spots, score = _spots(pts, sims, sims.max())
            frontier = list(spots[np.argsort(-score)[:BEAM]])
        pts_all.append(pts)
        sims_all.append(sims)
        seed_of.append(np.full(len(sims), k))
    pts, sims, seed_of = np.concatenate(pts_all), np.concatenate(sims_all), np.concatenate(seed_of)
    spots, score, spot_seed = [], [], []
    for k in range(len(seeds)):  # spots per seed region, scored on one scale (the photo's best similarity overall)
        sel = seed_of == k
        if sel.any():
            sp, sc = _spots(pts[sel], sims[sel], sims.max())
            spots.append(sp); score.append(sc); spot_seed.append(np.full(len(sc), k))
    return {"spots": np.concatenate(spots) if spots else seeds[:1], "score": np.concatenate(score) if score else np.zeros(1),
            "seed": np.concatenate(spot_seed) if spot_seed else np.zeros(1, int), "scored": int(len(sims))}


def probe(limit: int | None = None) -> None:
    from .stage1_eval import _bootstrap
    from .strategy_search import load_world

    photos, top, truth = _top1_photos("dev")
    if limit:
        photos, top, truth = photos[:limit], top[:limit], truth[:limit]
    world = load_world(mp16_queries=[{"row": e["row"], "image_id": e["image_id"]} for e in photos])
    gallery = Gallery(world)
    local = _local_set("dev")
    rng = np.random.default_rng(0)
    rows = []
    start = time.time()
    for m, e in enumerate(photos):
        q = world.query_embeddings[m] / np.linalg.norm(world.query_embeddings[m])
        pool = np.asarray(e["pool"], dtype=np.float64)
        out = _search(gallery, q, int(world.query_author[m]), pool[:SEEDS], rng)
        d_spot = _km_matrix(truth[m][None], out["spots"])[0]
        order = np.argsort(-out["score"])
        lv = local["dist"][m] < 9e4
        rows.append({
            "image_id": e["image_id"], "scored": out["scored"], "spots": int(len(order)),
            "spot_km_by_score": d_spot[order].tolist(), "spot_seed_by_score": out["seed"][order].tolist(), "spot_score_by_score": out["score"][order].tolist(),
            "spot_km_seed0": d_spot[out["seed"] == 0][np.argsort(-out["score"][out["seed"] == 0])].tolist(),
            "top1_km": float(_km_matrix(truth[m][None], top[m][None])[0, 0]),
            "local_km": local["dist"][m][lv].tolist(), "pool_km": _km_matrix(truth[m][None], pool)[0].tolist(),
        })
        if (m + 1) % 25 == 0:
            print(f"  {m + 1}/{len(photos)} photos, {(time.time() - start) / (m + 1):.1f} s per photo, {np.mean([r['scored'] for r in rows]):.0f} photos scored each", flush=True)
    (ROOT / "map_search_probe_dev.json").write_text(json.dumps(rows) + "\n", encoding="utf-8")
    n = len(rows)

    def share(get, km: float) -> float:
        return float(np.mean([bool(len(x := get(r))) and min(x) < km for r in rows]))

    base1, base25 = share(lambda r: [r["top1_km"]], 1), share(lambda r: [r["top1_km"]], 25)
    print(f"\ndev (n={n}); reranker top-1: < 1 km {base1:.1%}, < 25 km {base25:.1%}")
    print(f"  fixed lists: local candidates (<= 24 around the top-1) < 1 km {share(lambda r: r['local_km'], 1):.1%}; "
          f"reranker pool (~17) < 1 km {share(lambda r: r['pool_km'], 1):.1%}, < 25 km {share(lambda r: r['pool_km'], 25):.1%}")
    for K in BUDGETS:
        print(f"  search, top {K:2d} spots by score: < 1 km {share(lambda r: r['spot_km_by_score'][:K], 1):.1%}, < 25 km {share(lambda r: r['spot_km_by_score'][:K], 25):.1%}; "
              f"top {K} around the top-1 only: < 1 km {share(lambda r: r['spot_km_seed0'][:K], 1):.1%}")
    print(f"  search, all spots (mean {np.mean([r['spots'] for r in rows]):.0f}): < 1 km {share(lambda r: r['spot_km_by_score'], 1):.1%}; "
          f"photos scored per query {np.mean([r['scored'] for r in rows]):.0f}")
    for label, pick in (("best spot around the top-1", lambda r: r["spot_km_seed0"][:1] or [r["top1_km"]]), ("best spot overall", lambda r: r["spot_km_by_score"][:1])):
        hit1 = np.asarray([pick(r)[0] < 1 for r in rows], float) - np.asarray([r["top1_km"] < 1 for r in rows], float)
        hit25 = np.asarray([pick(r)[0] < 25 for r in rows], float) - np.asarray([r["top1_km"] < 25 for r in rows], float)
        c1, c25 = _bootstrap(hit1), _bootstrap(hit25)
        print(f"  pick {label:28s}: change < 1 km {100 * c1[0]:+.1f} [{100 * c1[1]:+.1f}, {100 * c1[2]:+.1f}], < 25 km {100 * c25[0]:+.1f} [{100 * c25[1]:+.1f}, {100 * c25[2]:+.1f}]")


RANKS = (1, 10, 30, 100, 300, 1000, 3000)


def truth_rank(limit: int | None = None) -> None:
    """Where the gallery photos within 1 km of the truth rank by SigLIP2 similarity among all gallery photos within 25 km of the reranker top-1 (no cap): the
    reach of a verifier that checks the N most similar photos there, against the fixed local-candidate list. Dev photos whose truth is within 25 km of the top-1."""

    from .strategy_search import load_world

    global CAP
    CAP = 10**9
    photos, top, truth = _top1_photos("dev")
    if limit:
        photos, top, truth = photos[:limit], top[:limit], truth[:limit]
    world = load_world(mp16_queries=[{"row": e["row"], "image_id": e["image_id"]} for e in photos])
    gallery = Gallery(world)
    local = _local_set("dev")
    rng = np.random.default_rng(0)
    rows, start = [], time.time()
    for m, e in enumerate(photos):
        top1_km = float(_km_matrix(truth[m][None], top[m][None])[0, 0])
        if top1_km >= 25:
            continue
        q = world.query_embeddings[m] / np.linalg.norm(world.query_embeddings[m])
        pts, sims, _ = gallery.ball(top[m], 25.0, q, int(world.query_author[m]), rng, set())
        order = np.argsort(-sims)
        near = _km_matrix(truth[m][None], pts[order])[0] < 1.0
        rows.append({"top1_km": top1_km, "n": int(len(sims)), "first_rank": int(np.argmax(near)) + 1 if near.any() else None,
                     "local_hit": bool((local["dist"][m][local["dist"][m] < 9e4] < 1).any())})
        if len(rows) % 50 == 0:
            print(f"  {len(rows)} photos ({m + 1}/{len(photos)} seen), {(time.time() - start) / (m + 1):.1f} s per photo", flush=True)
    (ROOT / "map_search_truth_rank_dev.json").write_text(json.dumps(rows) + "\n", encoding="utf-8")
    for label, sel in (("top-1 < 25 km", [True] * len(rows)), ("near-misses (1-25 km)", [r["top1_km"] >= 1 for r in rows])):
        rs = [r for r, s in zip(rows, sel) if s]
        ranks = np.asarray([r["first_rank"] or 10**9 for r in rs])
        print(f"\n{label} (n={len(rs)}; median {np.median([r['n'] for r in rs]):.0f} gallery photos within 25 km of the top-1): "
              f"a photo within 1 km of the truth exists for {np.mean([r['first_rank'] is not None for r in rs]):.1%}; fixed local list reaches {np.mean([r['local_hit'] for r in rs]):.1%}")
        print("  reach when checking the N most similar photos: " + ", ".join(f"N={k}: {(ranks <= k).mean():.1%}" for k in RANKS))


MATCH_TOP = 100


def match_pairs(limit: int | None = None) -> None:
    """For dev photos whose reranker top-1 is within 25 km of the truth: the MATCH_TOP most similar gallery photos within 25 km of the top-1 (image ids, source,
    location, similarity), for geo_match.py to verify by keypoint matching. Writes map_search_match_pairs_dev.json."""

    from .strategy_search import load_world

    global CAP
    CAP = 10**9
    photos, top, truth = _top1_photos("dev")
    if limit:
        photos, top, truth = photos[:limit], top[:limit], truth[:limit]
    world = load_world(mp16_queries=[{"row": e["row"], "image_id": e["image_id"]} for e in photos])
    gallery = Gallery(world)
    mp16_ids = (MP16_EMBED / "image_ids.txt").read_text(encoding="utf-8").splitlines()
    osv_ids = (OSV_EMBED / "image_ids.txt").read_text(encoding="utf-8").splitlines()
    rng = np.random.default_rng(0)
    out = []
    for m, e in enumerate(photos):
        top1_km = float(_km_matrix(truth[m][None], top[m][None])[0, 0])
        if top1_km >= 25:
            continue
        q = world.query_embeddings[m] / np.linalg.norm(world.query_embeddings[m])
        pts, sims, keys = gallery.ball(top[m], 25.0, q, int(world.query_author[m]), rng, set())
        order = np.argsort(-sims)[:MATCH_TOP]
        km = _km_matrix(truth[m][None], pts[order])[0]
        out.append({"image_id": e["image_id"], "top": top[m].tolist(), "truth": truth[m].tolist(), "top1_km": top1_km,
                    "gallery": [{"source": "mp16" if keys[i][0] == 0 else "osv", "id": (mp16_ids if keys[i][0] == 0 else osv_ids)[keys[i][1]],
                                 "latlon": pts[i].tolist(), "sim": float(sims[i]), "km": float(k)} for i, k in zip(order, km)]})
    (ROOT / "map_search_match_pairs_dev.json").write_text(json.dumps(out) + "\n", encoding="utf-8")
    print(f"{len(out)} photos, {sum(len(p['gallery']) for p in out)} pairs")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("node", choices=("probe", "truth-rank", "match-pairs"))
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)
    {"probe": probe, "truth-rank": truth_rank, "match-pairs": match_pairs}[args.node](args.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
