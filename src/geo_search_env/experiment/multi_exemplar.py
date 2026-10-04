# Does the comparator get better when it sees several exemplar photos of each candidate instead of one? (no training; comparator-a as is)
# Usage: PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.multi_exemplar pairs              (CPU, a few min)
#        PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.multi_exemplar judge --name comparator-a   (vLLM on :8765; scripts/multi_exemplar.sh)
#        PYTHONPATH=src .venv/bin/python -m geo_search_env.experiment.multi_exemplar report --name comparator-a

"""pairs:  for each dev / val photo's top TOPK candidates, up to EXEMPLARS database photos within 1 km (not by the query's photographer), at most one
        per photographer, most similar to the query first. Exemplar 0 is the one exemplar_judge.topk_pairs picks, so "first" reproduces lesson 24.
judge:  P(same place) for every (query, exemplar) pair.
report: top-1 <25 km of -rank + w * aggregate(logit P) with w fitted on dev, for aggregates first / max / mean / mean of the best 2, and the
        cross-validated listwise combiner (category_evidence._cv) with one-exemplar vs all-exemplar features, against the reranker top-1."""

from __future__ import annotations

import argparse
import json
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .category_evidence import _cv
from .exemplar_judge import EXEMPLAR_KM, TOPK, WEIGHTS, _jpeg, _logit, _p_same
from .query_evidence import ROOT
from .sft_data import MP16Images
from .stage1_eval import _bootstrap
from .strategy_search import EARTH_KM, MP16_EMBED, _xyz, load_world

EXEMPLARS = 4
PAIRS = ROOT / "multi_exemplar_pairs.json"


def _scores_path(name: str) -> Path:
    return ROOT / f"multi_exemplar_scores_{name}.json"


def pairs() -> None:
    from scipy.spatial import cKDTree

    ids = (MP16_EMBED / "image_ids.txt").read_text(encoding="utf-8").splitlines()
    out: list[dict[str, Any]] = []
    tree = None
    chord = 2 * math.sin(EXEMPLAR_KM / EARTH_KM / 2)
    for tag in ("dev", "val"):
        photos = json.loads((ROOT / tag / "dev.json").read_text(encoding="utf-8"))
        world = load_world(mp16_queries=[{"row": e["row"], "image_id": e["image_id"]} for e in photos]) if tag == "dev" else load_world()
        if tree is None:
            tree = cKDTree(_xyz(world.mp16["latlon"]))
        for m, e in enumerate(photos):
            pos = e["index"] if tag == "val" else m
            q = world.query_embeddings[pos] / np.linalg.norm(world.query_embeddings[pos])
            author = int(world.query_author[pos])
            found: list[list[str]] = []
            for latlon in e["pool"][:TOPK]:
                rows = np.sort([r for r in tree.query_ball_point(_xyz(np.asarray(latlon)), chord) if world.mp16["author"][r] != author])
                chosen: list[str] = []
                if len(rows):
                    emb = np.asarray(world.mp16["embeddings"][rows], dtype=np.float32)
                    seen: set[int] = set()
                    for i in np.argsort(-(emb @ q / np.linalg.norm(emb, axis=1))):
                        a = int(world.mp16["author"][rows[i]])
                        if a not in seen:
                            seen.add(a)
                            chosen.append(ids[rows[i]])
                            if len(chosen) == EXEMPLARS:
                                break
                found.append(chosen)
            out.append({"tag": tag, "image_id": e["image_id"], "path": e.get("path"), "exemplars": found})
        mine = [len(x) for p in out if p["tag"] == tag for x in p["exemplars"]]
        print(f"{tag}: {sum(p['tag'] == tag for p in out)} photos; exemplars per candidate: mean {np.mean(mine):.2f}, "
              f"none {np.mean(np.asarray(mine) == 0):.1%}, all {EXEMPLARS} {np.mean(np.asarray(mine) == EXEMPLARS):.0%}", flush=True)
    PAIRS.write_text(json.dumps(out) + "\n", encoding="utf-8")


def judge(server: str, name: str) -> None:
    photos = json.loads(PAIRS.read_text(encoding="utf-8"))
    images = MP16Images()
    jobs = [(i, r, j) for i, p in enumerate(photos) for r, xs in enumerate(p["exemplars"]) for j in range(len(xs))]

    def run(job: tuple[int, int, int]) -> float | None:
        i, r, j = job
        p = photos[i]
        query = Path(p["path"]).read_bytes() if p["path"] else images.read(p["image_id"])
        return _p_same(server, _jpeg(query), _jpeg(images.read(p["exemplars"][r][j])))

    with ThreadPoolExecutor(32) as pool:
        scores = list(pool.map(run, jobs))
    for p in photos:
        p["p_same"] = [[None] * len(xs) for xs in p["exemplars"]]
    for (i, r, j), score in zip(jobs, scores):
        photos[i]["p_same"][r][j] = score
    _scores_path(name).write_text(json.dumps(photos) + "\n", encoding="utf-8")
    print(f"{len(jobs)} comparisons, {sum(s is None for s in scores)} failed -> {_scores_path(name)}")


def _load(name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Distances [photo, TOPK] (1e5 = no candidate), logits [photo, TOPK, EXEMPLARS] (NaN = no exemplar), tags and benchmarks; placeholders dropped."""

    from .wiki_backend import _km

    scored = json.loads(_scores_path(name).read_text(encoding="utf-8"))
    excluded = set(json.loads((ROOT / "val" / "exclude.json").read_text(encoding="utf-8")))
    D, L, tags, bench = [], [], [], []
    for tag in ("dev", "val"):
        photos = json.loads((ROOT / tag / "dev.json").read_text(encoding="utf-8"))
        mine = [p for p in scored if p["tag"] == tag]
        assert len(mine) == len(photos)
        for e, p in zip(photos, mine):
            if e["image_id"] in excluded:
                continue
            d = _km(np.asarray(e["pool"][:TOPK]), *e["truth"])
            lg = np.full((TOPK, EXEMPLARS), np.nan)
            for r, xs in enumerate(p["p_same"][: len(d)]):
                for j, x in enumerate(xs):
                    if x is not None:
                        lg[r, j] = _logit(np.asarray([x]))[0]
            D.append(np.pad(d, (0, TOPK - len(d)), constant_values=1e5))
            L.append(lg)
            tags.append(tag)
            bench.append(e.get("benchmark", "mp16"))
    return np.asarray(D), np.asarray(L), np.asarray(tags), np.asarray(bench)


def _aggregates(L: np.ndarray) -> dict[str, np.ndarray]:
    """Per candidate, one score from its exemplars' logits (0 = neutral when it has none)."""

    has = ~np.isnan(L)
    filled = np.where(has, L, -np.inf)
    count = has.sum(-1)
    best2 = -np.sort(-filled, axis=-1)[..., :2]
    best2 = np.where(np.isinf(best2), np.nan, best2)
    with np.errstate(invalid="ignore", divide="ignore"):
        return {
            "first": np.where(has[..., 0], L[..., 0], 0.0),
            "max": np.where(count > 0, filled.max(-1), 0.0),
            "mean": np.where(count > 0, np.nansum(L, -1) / np.maximum(count, 1), 0.0),
            "mean of best 2": np.where(count > 0, np.nan_to_num(np.nanmean(best2, -1)), 0.0),
        }


def report(name: str) -> None:
    D, L, tags, bench = _load(name)
    valid = D < 9e4
    rank = np.broadcast_to(np.arange(TOPK, dtype=float), D.shape)
    base = (D[:, 0] < 25).astype(float)
    count = (~np.isnan(L)).sum(-1)
    print(f"{name}: {len(D)} photos (dev {int((tags == 'dev').sum())}, val {int((tags == 'val').sum())}); exemplars per candidate mean "
          f"{count[valid].mean():.2f}; reranker top-1 <25 km dev {base[tags == 'dev'].mean():.1%}, val {base[tags == 'val'].mean():.1%}\n")

    def top1(score: np.ndarray) -> np.ndarray:
        return (D[np.arange(len(D)), np.where(valid, score, -np.inf).argmax(1)] < 25).astype(float)

    out: dict[str, Any] = {"n": len(D), "rule": {}, "cv": {}}
    print("score = -rank + w * aggregate(logit P(same)), w fitted on dev; change in top-1 <25 km vs the reranker [95% CI]")
    for agg, a in _aggregates(L).items():
        w = max(WEIGHTS, key=lambda v: (top1(-rank + v * a)[tags == "dev"].mean(), -v))
        hit = top1(-rank + w * a)
        cells = {}
        for label, sel in (("dev", tags == "dev"), ("val", tags == "val"), ("im2gps3k", bench == "im2gps3k"), ("yfcc4k", bench == "yfcc4k")):
            cells[label] = _bootstrap(hit[sel] - base[sel])
        out["rule"][agg] = {"w": w, **cells}
        print(f"  {agg:15s} w={w:<4g} " + "  ".join(f"{k} {100 * v[0]:+.1f} [{100 * v[1]:+.1f}, {100 * v[2]:+.1f}]" for k, v in cells.items()))

    aggs = _aggregates(L)
    rank_f = [rank, (rank == 0).astype(float), (rank < 3).astype(float)]

    def judge_f(a: np.ndarray) -> list[np.ndarray]:
        order = (-a).argsort(1).argsort(1).astype(float)
        return [a, a * rank, a - a.max(1, keepdims=True), order, (order == 0).astype(float), a * (rank == 0)]

    configs = {
        "rank + first exemplar": rank_f + judge_f(aggs["first"]),
        "rank + all exemplars": rank_f + judge_f(aggs["max"]) + judge_f(aggs["mean"]) + [aggs["first"], count.astype(float)],
    }
    print("\ncross-validated listwise combiner (5 folds, 3 seeds, dev + val), change in top-1 <25 km vs the reranker [95% CI]")
    for label, feats in configs.items():
        hit = _cv(D, np.stack(feats, -1).astype(np.float32), valid)
        cells = {k: _bootstrap(hit[s] - base[s]) for k, s in (("all", np.ones(len(D), bool)), ("dev", tags == "dev"), ("val", tags == "val"))}
        out["cv"][label] = cells
        print(f"  {label:24s} " + "  ".join(f"{k} {100 * v[0]:+.1f} [{100 * v[1]:+.1f}, {100 * v[2]:+.1f}]" for k, v in cells.items()))
    (ROOT / f"multi_exemplar_report_{name}.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("node", choices=("pairs", "judge", "report"))
    parser.add_argument("--name", default="comparator-a")
    parser.add_argument("--server", default="http://127.0.0.1:8765")
    args = parser.parse_args(argv)
    if args.node == "pairs":
        pairs()
    elif args.node == "judge":
        judge(args.server, args.name)
    else:
        report(args.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
