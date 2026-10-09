# Verifying recalled place names (TEXT_SEARCH_PLAN.md, after lesson 66): a name the model recalls from the photo's appearance ("Lake Champlain")
# is a hypothesis its own judgement can't test. Two independent checks of the hypothesis, for every recalled-name clue (the 4B's search queries)
# whose region differs from Pinpoint's top-1 region:
#   restricted: Pinpoint's retrieval + reranker restricted to a box around the named place; does its best match there score at least as high as
#               Pinpoint's own top-1? (A real recognition should find strong matches there; a look-alike only weaker ones.)
#   exemplars:  SigLIP2 similarity of the photo to gallery photos within 1 km of the named place, vs to Pinpoint's top-1 candidate photo.
# Measured as separation of right vs wrong recalled names (AUC) and as switch rules vs Pinpoint (thresholds chosen on one half of the photos,
# counted on the other). dev sets only.
# Usage: PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.recall_verify run --tag dev3k     (GPU, one index pass; then the report)
#        PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.recall_verify report --tag dev3k

"""run:    per recalled-name clue disagreeing with Pinpoint: the restricted top-1 (box of BOX_DEG around the clue's place) and its reranker score;
        the best SigLIP2 similarity to index photos within EXEMPLAR_KM of the place (not by the photo's photographer) and to Pinpoint's top-1
        candidate photo -> artifacts/agent/recall_verify_{tag}.npz.
report: AUC of each feature for right (clue's place < 25 km from the truth) vs wrong clues; switch rules (the clue's own coordinate, or the
        restricted top-1) by feature threshold, alone and with the fine-tuned judge's P(Yes)."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np

OUT = Path("artifacts/agent")
PHOTO_SETS = Path("artifacts/photo_sets")
SFT_ROOT = Path("artifacts/sft")
NVME_INDEX = Path("/opt/dlami/nvme/pinpoint/attention_retrieval_index")
BOX_DEG = 0.5  # half-width of the restriction box in latitude degrees (~55 km); longitude scaled by latitude
EXEMPLAR_KM = 1.0
JUDGE = "ftvsall4b"  # clue_judge run whose P(Yes) is combined in the report


def _clues(tag: str) -> list[dict]:
    from .clue_judge import _differing, _items

    return [r for r, _ in _differing(tag, _items(tag, "4b")) if r["source"].startswith("query")]


def run(tag: str) -> None:
    from scipy.spatial import cKDTree

    from ..models.pinpoint_reranker import PinpointReranker, _unit_xyz
    from .pinpoint_reference import load_pools
    from .strategy_search import load_world

    photos = {e["image_id"]: e for e in json.loads((PHOTO_SETS / tag / "photos.json").read_text(encoding="utf-8"))}
    clues = _clues(tag)
    queries = json.loads((SFT_ROOT / "queries.json").read_text(encoding="utf-8"))
    owners = sorted({c["image_id"] for c in clues})
    slot = {k: i for i, k in enumerate(owners)}
    world = load_world(mp16_queries=[queries[photos[k]["index"]] for k in owners])
    reranker = PinpointReranker(device="cuda", index_dir=NVME_INDEX if (NVME_INDEX / "manifest.json").exists() else None)
    q = np.asarray([slot[c["image_id"]] for c in clues])
    lat = np.asarray([c["lat"] for c in clues])
    lon = np.asarray([c["lon"] for c in clues])
    dlon = BOX_DEG / np.maximum(np.cos(np.radians(lat)), 0.05)
    box = np.stack([lat - BOX_DEG, lat + BOX_DEG, lon - dlon, lon + dlon], 1)
    print(f"{len(clues)} recalled-name clues disagreeing with Pinpoint on {len(owners)} photos", flush=True)
    index_author = world.mp16["author"][reranker.cache_indices]
    found = reranker.rank(world.query_embeddings[q], query_author=world.query_author[q], index_author=index_author, query_box=box)
    pools = load_pools("mp16")
    pin_score = np.asarray([pools["scores"][photos[c["image_id"]]["index"]][0] for c in clues])

    emb = world.mp16["embeddings"]
    norm = lambda v: v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-8)
    query_vec = norm(np.asarray(world.query_embeddings, np.float32))
    index_latlon = np.asarray(reranker.latlon, np.float64)
    tree = cKDTree(_unit_xyz(index_latlon))
    chord = 2 * math.sin(EXEMPLAR_KM / 6371.0 / 2)
    near = tree.query_ball_point(_unit_xyz(np.stack([lat, lon], 1)), chord)
    best_near, top1_sim, n_near = np.full(len(clues), np.nan), np.zeros(len(clues)), np.zeros(len(clues), int)
    for i, c in enumerate(clues):
        qv = query_vec[slot[c["image_id"]]]
        top1_sim[i] = float(norm(np.asarray(emb[photos[c["image_id"]]["pool_rows"][0]], np.float32)) @ qv)
        rows = [r for r in near[i] if index_author[r] != world.query_author[slot[c["image_id"]]]][:200]
        n_near[i] = len(rows)
        if rows:
            vecs = norm(np.asarray(emb[np.sort(reranker.cache_indices[rows])], np.float32))
            best_near[i] = float((vecs @ qv).max())
    np.savez(OUT / f"recall_verify_{tag}.npz", image_id=np.asarray([c["image_id"] for c in clues]), query=np.asarray([c["query"] for c in clues]),
             clue_lat=lat, clue_lon=lon, clue_km=np.asarray([c["km"] for c in clues]), restricted_latlon=found.top1, restricted_score=found.scores[:, 0],
             pin_score=pin_score, best_near=best_near, top1_sim=top1_sim, n_near=n_near)
    report(tag)


def _auc(pos: np.ndarray, neg: np.ndarray) -> float:
    pos, neg = pos[np.isfinite(pos)], neg[np.isfinite(neg)]
    if not len(pos) or not len(neg):
        return float("nan")
    ranks = np.argsort(np.argsort(np.concatenate([pos, neg]))) + 1
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def report(tag: str) -> None:
    from .strategy_search import _bootstrap, _haversine_km

    z = dict(np.load(OUT / f"recall_verify_{tag}.npz"))
    photos = json.loads((PHOTO_SETS / tag / "photos.json").read_text(encoding="utf-8"))
    base = {e["image_id"]: _haversine_km(*e["truth"], np.asarray([e["pool"][0]]))[0] for e in photos}
    truth = {e["image_id"]: e["truth"] for e in photos}
    judge_path = OUT / f"clue_judge_{tag}_{JUDGE}_4b.json"
    judge = {(s["image_id"], s["query"]): s["p_yes"] for s in json.loads(judge_path.read_text(encoding="utf-8"))} if judge_path.exists() else {}
    right = z["clue_km"] < 25
    restricted_km = np.asarray([_haversine_km(*truth[k], np.asarray([ll]))[0] if np.isfinite(ll).all() else np.inf
                                for k, ll in zip(z["image_id"], z["restricted_latlon"])])
    features = {
        "restricted score - Pinpoint score": z["restricted_score"] - z["pin_score"],
        "exemplar sim (best within 1 km) - sim to Pinpoint top-1": z["best_near"] - z["top1_sim"],
        "judge P(Yes)": np.asarray([judge.get((k, q), np.nan) for k, q in zip(z["image_id"], z["query"])]),
    }
    print(f"{tag}: {len(right)} recalled-name clues disagreeing with Pinpoint; right (< 25 km) {right.sum()}; restricted top-1 found for "
          f"{np.isfinite(z['restricted_score']).sum()}; gallery photos within {EXEMPLAR_KM} km for {(z['n_near'] > 0).sum()}")
    for name, f in features.items():
        print(f"  AUC right vs wrong, {name:56s} {_auc(f[right], f[~right]):.3f}")
    ids = sorted(set(z["image_id"]))
    fold = {k: i % 2 for i, k in enumerate(np.random.default_rng(0).permutation(ids))}
    combos = {
        "restricted margin": features["restricted score - Pinpoint score"],
        "exemplar margin": features["exemplar sim (best within 1 km) - sim to Pinpoint top-1"],
        "restricted margin, judge >= 0.5": np.where(features["judge P(Yes)"] >= 0.5, features["restricted score - Pinpoint score"], -np.inf),
        "exemplar margin, judge >= 0.5": np.where(features["judge P(Yes)"] >= 0.5, features["exemplar sim (best within 1 km) - sim to Pinpoint top-1"], -np.inf),
    }
    for name, f in combos.items():
        finite = f[np.isfinite(f)]
        if not len(finite):
            continue
        grid = np.quantile(finite, [0.5, 0.7, 0.8, 0.9, 0.95, 0.98])
        for answer, km in (("clue's coordinate", z["clue_km"]), ("restricted top-1", restricted_km)):

            def switched(t: float, keep: set[str]) -> dict[str, float]:
                out: dict[str, float] = {}
                for i, k in enumerate(z["image_id"]):
                    if k in keep and f[i] >= t and (k not in out or f[i] > out[k][0]):
                        out[k] = (f[i], km[i])
                return {k: v[1] for k, v in out.items()}

            def score(t: float, keep: set[str]) -> int:
                s = switched(t, keep)
                return sum(int(d < 25) - int(base[k] < 25) for k, d in s.items())

            chosen: dict[str, float] = {}
            for half in (0, 1):
                fit = {k for k in ids if fold[k] != half}
                held = {k for k in ids if fold[k] == half}
                t = max(grid, key=lambda t: score(t, fit))
                chosen |= switched(t, held)
            gain = np.asarray([int(chosen.get(e["image_id"], base[e["image_id"]]) < 25) - int(base[e["image_id"]] < 25) for e in photos], float)
            c = _bootstrap(gain)
            fixes = sum(d < 25 <= base[k] for k, d in chosen.items())
            breaks = sum(base[k] < 25 <= d for k, d in chosen.items())
            print(f"  switch on {name:34s} -> {answer:18s}: switched {len(chosen):4d}, fixes {fixes:3d}, breaks {breaks:3d}, "
                  f"< 25 km {100 * c[0]:+.1f} [{100 * c[1]:+.1f}, {100 * c[2]:+.1f}] (threshold chosen on the other half)")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cmd", choices=("run", "report"))
    parser.add_argument("--tag", default="dev3k")
    args = parser.parse_args(argv)
    run(args.tag) if args.cmd == "run" else report(args.tag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
