"""VALUE_PLAN step 4 ("hedge or search"): when the value network would hedge, does searching inside the hedged region find the place?

Per photo, Pinpoint (tools.pinpoint, setft_own reranker, original protocol) restricted to a box of +-R km (R in RADII) around each of the 7
hedge answers of lesson V8 (the region probe's top 5 cells, the probe-mass and pool-mass points within 500 km); top TOP candidates saved.
Report, per set, overall and on unsure photos (the step 1 network's P(< 25 km) < 0.3, `value_agent --dump`):
  - new right: some restricted top-1 < 25 km on a photo where no current area option is < 25 km (area oracle miss), against luck (the same
    restricted top-1s scored on another photo's truth, lesson 120), by budget (top cell, top 3 cells, all 7 points; per radius);
  - the same at 1 km, and with the top 5 of each search;
  - whether the searches' own scores find it: AUC of the restricted top-1's reranker score for right vs wrong searches, and the hit rate
    of the best-scoring search's top-1 on area-oracle misses.
Bar (pre-declared, VALUE_PLAN step 4): >= 3 pts of the set's photos with new right candidates net of luck; below it, stop.

  PINPOINT_KEEP_OWN=1 PINPOINT_RERANKER_CKPT=artifacts/pinpoint_train/setft_own/checkpoints/ckpt_final.pt \\
    PYTHONPATH=src .venv/bin/python scripts/hedge_search.py run --tag dev3k            (GPU, 2-4 GB; also commons26dev)
  PYTHONPATH=src .venv/bin/python scripts/hedge_search.py report
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

OUT = Path("artifacts/value")
RADII = (50.0, 150.0)
TOP, HEDGE_R = 16, 500.0
POINTS = ("probe cell 1", "probe cell 2", "probe cell 3", "probe cell 4", "probe cell 5", "probe mass", "pool mass")
SETS = {"dev3k": "dev3k_setft_own", "commons26dev": "commons26dev_setft_own"}


def _xyz(ll):
    la, lo = np.radians(ll[..., 0]), np.radians(ll[..., 1])
    return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], -1)


def _km(a, b):
    return 6371.0 * np.arccos(np.clip((_xyz(a) * _xyz(b)).sum(-1), -1, 1))


def _photos(tag: str) -> list[dict]:
    return json.loads((Path("artifacts/photo_sets") / SETS[tag] / "photos.json").read_text(encoding="utf-8"))


def hedge_points(tag: str) -> np.ndarray:
    """[n, 7, 2] hedge answers as in value_agent._hedges (lesson V8)."""

    photos = _photos(tag)
    pools = np.load(Path("artifacts/pinpoint_reference") / f"{photos[0]['pools']}.npz")
    idx = np.asarray([e["index"] for e in photos])
    pll, sc, pv = pools["latlon"][idx].astype(np.float64), pools["scores"][idx], pools["rows"][idx] >= 0
    w = np.exp(np.where(pv, sc - np.where(pv, sc, -np.inf).max(1, keepdims=True), -np.inf))
    w = np.where(pv, w / w.sum(1, keepdims=True), 0.0)
    lp = np.load(OUT / f"probe_logp_{tag}.npy").astype(np.float32)
    cent = np.load("artifacts/retrieval/probe/cells.npy").astype(np.float64)
    cc = (6371.0 * np.arccos(np.clip(_xyz(cent) @ _xyz(cent).T, -1, 1)) < HEDGE_R).astype(np.float32)
    dpp = 6371.0 * np.arccos(np.clip(np.einsum("nid,njd->nij", _xyz(pll), _xyz(pll)), -1, 1))
    pm = pll[np.arange(len(pll)), ((dpp < HEDGE_R) * w[:, None, :]).sum(-1).argmax(1)]
    return np.concatenate([cent[np.argsort(-lp, 1)[:, :5]], cent[(np.exp(lp) @ cc.T).argmax(1)][:, None], pm[:, None]], 1)


def _box(p, r: float):
    dlat = r / 111.0
    dlon = r / (111.0 * max(np.cos(np.radians(p[0])), 0.05))
    return (float(max(p[0] - dlat, -90)), float(min(p[0] + dlat, 90)), float(max(p[1] - dlon, -180)), float(min(p[1] + dlon, 180)))


def run(tag: str) -> None:
    if os.environ.get("PINPOINT_KEEP_OWN") != "1" or "setft_own" not in os.environ.get("PINPOINT_RERANKER_CKPT", ""):
        raise SystemExit("set PINPOINT_KEEP_OWN=1 and PINPOINT_RERANKER_CKPT to the setft_own checkpoint")
    from geo_search_env.tools.pinpoint import Constraint, Pinpoint

    photos = _photos(tag)
    pts = hedge_points(tag)
    tool = Pinpoint()
    if "row" in photos[0]:  # MP16 queries: the cached embedding; KEEP_OWN drops only the query itself
        from geo_search_env.experiment.strategy_search import load_world

        rows = np.asarray([int(e["row"]) for e in photos])
        emb, authors = np.asarray(load_world().mp16["embeddings"][rows], np.float32), tool.mp16_author(rows)
    else:  # benchmark queries (not in the index): no exclusion
        from geo_search_env.data.benchmarks import load_benchmark

        emb, authors = np.asarray(load_benchmark("commons26").embeddings[[e["index"] for e in photos]], np.float32), None
    items = [(i, k, r) for i in range(len(photos)) for k in range(len(POINTS)) for r in range(len(RADII))]
    out = {k: [] for k in ("latlon", "scores", "rows")}
    t = time.perf_counter()
    for s in range(0, len(items), 2048):
        chunk = items[s:s + 2048]
        p = np.asarray([i for i, _, _ in chunk])
        found = tool.search(emb[p], [Constraint(box=_box(pts[i, k], RADII[r])) for i, k, r in chunk], None if authors is None else authors[p])
        out["latlon"].append(found.latlon[:, :TOP].astype(np.float32))
        out["scores"].append(found.scores[:, :TOP].astype(np.float32))
        out["rows"].append(found.rows[:, :TOP].astype(np.int32))
        print(f"  {s + len(chunk)} / {len(items)} searches, {time.perf_counter() - t:.0f} s", flush=True)
    n, K, R = len(photos), len(POINTS), len(RADII)
    np.savez(OUT / f"hedge_search_{tag}.npz", points=pts, **{k: np.concatenate(v).reshape((n, K, R) + np.concatenate(v).shape[1:]) for k, v in out.items()})
    print(f"-> {OUT / f'hedge_search_{tag}.npz'}", flush=True)


def _area_oracle(tag: str) -> np.ndarray:
    """Per photo, whether some current area option (unzoomed or zoomed point) is < 25 km (action_agent's data, all tokens)."""

    from geo_search_env.experiment import action_agent as aa

    aa.CKPT_TAG, aa.ONE_KM = "setft_own", 0.5
    d = aa._data("dev3k") if tag == "dev3k" else aa._data("commons26dev", ckpt="setft_own")
    return ((d["km_unzoom"] < 25) | (d["km_zoom"] < 25)).any(1).cpu().numpy()


def _auc(s, y):
    from scipy.stats import rankdata

    r = rankdata(s)
    p = y.sum()
    return (r[y].sum() - p * (p + 1) / 2) / max(p * (len(y) - p), 1)


def report(seed: int = 7) -> None:
    for tag in SETS:
        if not (OUT / f"hedge_search_{tag}.npz").exists():
            continue
        z = np.load(OUT / f"hedge_search_{tag}.npz")
        photos = _photos(tag)
        truth = np.asarray([e["truth"] for e in photos], np.float64)
        n = len(truth)
        other = truth[np.random.default_rng(0).permutation(n)]  # luck: another photo's truth
        ll, sc = z["latlon"].astype(np.float64), z["scores"]
        ok = np.isfinite(ll[..., 0])
        d_true = np.where(ok, _km(truth[:, None, None, None, :], ll), np.inf)  # [n, K, R, TOP]
        d_luck = np.where(ok, _km(other[:, None, None, None, :], ll), np.inf)
        area = _area_oracle(tag)
        ans = np.load(OUT / f"answers_{tag}_seed{seed}.npz")
        unsure = ans["p25"] < 0.3
        print(f"\n {tag} ({n} photos; area oracle {area.mean():.1%}; unsure {unsure.mean():.1%}): new right = some restricted candidate < r km on a photo "
              f"no area option reaches (pts of all photos; luck in brackets; net)")
        budgets = {"top cell": [0], "top 3 cells": [0, 1, 2], "5 cells": [0, 1, 2, 3, 4], "probe mass": [5], "pool mass": [6], "all 7": list(range(7))}
        best_net = -np.inf
        for r_i, r in enumerate(RADII):
            for name, ks in budgets.items():
                cells = []
                for thr, topk in ((25, 1), (25, 5), (1, 1)):
                    hit = (d_true[:, ks, r_i, :topk] < thr).any((1, 2)) & ~area
                    luck = (d_luck[:, ks, r_i, :topk] < thr).any((1, 2)) & ~area
                    cells.append(f"{100 * hit.mean():5.2f} [{100 * luck.mean():4.2f}] {100 * (hit.mean() - luck.mean()):+5.2f}")
                    if thr == 25 and topk == 1:
                        best_net = max(best_net, 100 * (hit.mean() - luck.mean()))
                        hu = 100 * (hit & unsure).mean()
                print(f"   box +-{r:3.0f} km, {name:11s}  top-1 < 25 km: {cells[0]}   top-5 < 25 km: {cells[1]}   top-1 < 1 km: {cells[2]}   "
                      f"(on unsure photos {hu:.2f})")
        print(f"   best top-1 < 25 km net of luck: {best_net:+.2f} pts (bar +3.00: {'passed' if best_net >= 3 else 'not met'})")
        # stricter luck: another photo whose truth is 100-750 km from this one's (dense-region luck); photos without one are skipped
        dt = _km(truth[:, None, :], truth[None, :, :])
        rng = np.random.default_rng(1)
        near = np.asarray([rng.choice(c) if len(c := np.flatnonzero((dt[i] > 100) & (dt[i] < 750))) else -1 for i in range(n)])
        has = near >= 0
        d_near = np.where(ok, _km(truth[np.maximum(near, 0)][:, None, None, None, :], ll), np.inf)
        for r_i, r in enumerate(RADII):
            hit = (d_true[:, :, r_i, :1] < 25).any((1, 2)) & ~area & has
            luck = (d_near[:, :, r_i, :1] < 25).any((1, 2)) & ~area & has
            print(f"   box +-{r:3.0f} km, all 7, top-1 < 25 km, photos with a neighbour 100-750 km away ({has.mean():.0%}): {100 * hit.mean():.2f} "
                  f"[nearby-truth luck {100 * luck.mean():.2f}] net {100 * (hit.mean() - luck.mean()):+.2f} pts of all photos")
        right = d_true[..., 0] < 25  # [n, K, R] the restricted top-1
        s1 = sc[..., 0]
        valid = np.isfinite(s1)
        for r_i, r in enumerate(RADII):  # within a photo: does the best score pick the right search more often than a random search would?
            m = ~area & right[:, :, r_i].any(1) & (~right[:, :, r_i] & valid[:, :, r_i]).any(1)
            pick = np.where(valid[:, :, r_i], s1[:, :, r_i], -np.inf).argmax(1)
            chance = (right[:, :, r_i] & valid[:, :, r_i]).sum(1) / valid[:, :, r_i].sum(1).clip(1)
            print(f"   box +-{r:3.0f} km, area-oracle misses with right and wrong searches ({100 * m.mean():.2f} pts): best score picks a right one "
                  f"{np.mean(right[np.arange(n), pick, r_i][m]):.1%} vs random {np.mean(chance[m]):.1%}", flush=True)
        for r_i, r in enumerate(RADII):
            v = valid[:, :, r_i]
            print(f"   box +-{r:3.0f} km: AUC of the restricted top-1's score, right vs wrong searches {_auc(s1[:, :, r_i][v], right[:, :, r_i][v]):.3f}; "
                  f"on area-oracle misses, the best-scoring search's top-1 < 25 km: "
                  f"{np.mean(right[np.arange(n), np.where(v, s1[:, :, r_i], -np.inf).argmax(1), r_i][~area]):.1%} "
                  f"(any search's top-1: {np.mean(right[:, :, r_i].any(1)[~area]):.1%})", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("node", choices=("run", "report"))
    ap.add_argument("--tag", default="dev3k", choices=tuple(SETS))
    args = ap.parse_args()
    run(args.tag) if args.node == "run" else report()


if __name__ == "__main__":
    main()
